import pytest

from protoguard.errors import ProtoGuardError
from protoguard.labels import Interval, IntervalIndex, label_windows, load_labels, save_labels
from protoguard.messages import REQUEST, RESPONSE, Message
from protoguard.metrics import confusion_metrics, roc_auc
from protoguard.util import mean_std, median, shannon_entropy, threshold_for_fpr
from protoguard.windows import FEATURES, build_windows, vector


def mb(ts, fn=3, direction=REQUEST, client="c1", server="s1", **kw):
    kw.setdefault("address", 0)
    kw.setdefault("quantity", 10)
    kw.setdefault("unit_id", 1)
    return Message(ts=ts, proto="modbus", client=client, server=server, direction=direction,
                   kind=f"fc{fn}", size=kw.pop("size", 12), function=fn, **kw)


def pub(ts, topic="a/b", payload_len=10, direction=REQUEST, client="c1", **kw):
    return Message(ts=ts, proto="mqtt", client=client, server="broker", direction=direction, kind="PUBLISH",
                   size=payload_len + 10, topic=topic, qos=0, payload_len=payload_len, payload_entropy=2.0, **kw)


# ---------------------------------------------------------------- windows
def test_windows_are_aligned_to_the_clock_and_split_by_channel():
    msgs = [mb(4.9), mb(5.1), mb(5.2, client="c2")]
    windows = build_windows(msgs, 5.0)
    assert [(w.start, w.end, w.client) for w in windows] == [(0.0, 5.0, "c1"), (5.0, 10.0, "c1"), (5.0, 10.0, "c2")]


def test_modbus_features():
    msgs = [
        mb(0.0, 3), mb(1.0, 3, quantity=20), mb(2.0, 6, address=200, quantity=1),
        mb(0.01, 3, direction=RESPONSE, quantity=None, address=None),
        mb(1.01, 3, direction=RESPONSE, quantity=None, address=None, exception=6),
        Message(ts=3.0, proto="modbus", client="c1", server="s1", direction=REQUEST, kind="invalid", size=7,
                valid=False, problems=("bad_mbap",)),
    ]
    (w,) = build_windows(msgs, 5.0)
    f = w.features
    assert f["req_rate"] == pytest.approx(4 / 5)
    assert f["resp_ratio"] == pytest.approx(2 / 4)
    assert f["read_ratio"] == pytest.approx(2 / 4) and f["write_ratio"] == pytest.approx(1 / 4)
    assert f["n_functions"] == 2 and f["n_addresses"] == 2
    assert f["exception_ratio"] == pytest.approx(1 / 2)
    assert f["invalid_ratio"] == pytest.approx(1 / 6)
    assert f["max_quantity"] == 20 and f["mean_quantity"] == pytest.approx((10 + 20 + 1) / 3)
    assert set(f) == set(FEATURES["modbus"])
    assert len(vector(w)) == len(FEATURES["modbus"])


def test_mqtt_features():
    conn = Message(ts=0.5, proto="mqtt", client="c1", server="broker", direction=REQUEST, kind="CONNECT", size=20, client_id="x")
    sub = Message(ts=0.6, proto="mqtt", client="c1", server="broker", direction=REQUEST, kind="SUBSCRIBE", size=20,
                  filters=("a/#", "b/c"))
    msgs = [conn, sub, pub(1.0, "t1", 100), pub(2.0, "t2", 300), pub(3.0, "t1", 50, direction=RESPONSE, retain=True)]
    (w,) = build_windows(msgs, 5.0)
    f = w.features
    assert f["pub_rate"] == pytest.approx(2 / 5) and f["deliver_rate"] == pytest.approx(1 / 5)
    assert f["connect_rate"] == pytest.approx(1 / 5) and f["sub_rate"] == pytest.approx(1 / 5)
    assert f["n_topics"] == 2 and f["n_client_ids"] == 1
    assert f["wildcard_subs"] == 1 and f["n_filters"] == 2
    assert f["max_payload_len"] == 300 and f["mean_payload_len"] == pytest.approx(150)
    assert f["retain_ratio"] == pytest.approx(1 / 3)
    assert set(f) == set(FEATURES["mqtt"])


def test_window_validation_and_unknown_protocol():
    with pytest.raises(ValueError):
        build_windows([], 0)
    assert build_windows([], 5.0) == []
    bad = Message(ts=1.0, proto="dnp3", client="a", server="b", direction=REQUEST, kind="x", size=1)
    with pytest.raises(ValueError):
        build_windows([bad], 5.0)


# ----------------------------------------------------------------- labels
def test_label_file_round_trip_and_validation(tmp_path):
    ivs = [Interval(10.0, 20.0, "flood", "1.1.1.1", "2.2.2.2"), Interval(30.0, 40.0)]
    path = tmp_path / "labels.csv"
    save_labels(path, ivs)
    assert load_labels(path) == ivs

    for text in ("start,end\n5,5\n", "start,end\nx,6\n", "a,b\n1,2\n"):
        bad = tmp_path / "bad.csv"
        bad.write_text(text)
        with pytest.raises(ProtoGuardError):
            load_labels(bad)
    with pytest.raises(ProtoGuardError):
        load_labels(tmp_path / "missing.csv")


def test_interval_matching_by_time_source_and_destination():
    windows = build_windows(
        [mb(12.0, client="a", server="x"), mb(12.0, client="b", server="x"), mb(12.0, client="a", server="y")], 5.0
    )
    label_windows(windows, [Interval(10.0, 15.0, "scan", "a", "x")])
    assert {(w.client, w.server): w.label for w in windows} == {("a", "x"): 1, ("b", "x"): 0, ("a", "y"): 0}

    label_windows(windows, [Interval(10.0, 15.0, "scan")])  # no src/dst: everyone in that period
    assert all(w.label == 1 and w.attack == "scan" for w in windows)

    label_windows(windows, [Interval(100.0, 105.0, "later")])
    assert all(w.label == 0 and w.attack is None for w in windows)


def test_long_interval_that_started_earlier_still_matches():
    (w,) = build_windows([mb(500.0)], 5.0)
    index = IntervalIndex([Interval(0.0, 1000.0, "slow", "c1"), Interval(10.0, 20.0, "short", "c1")])
    assert [iv.attack for iv in index.matches(w)] == ["slow"]


def test_window_touching_interval_edge_is_not_a_match():
    (w,) = build_windows([mb(5.5)], 5.0)  # window [5, 10)
    assert IntervalIndex([Interval(0.0, 5.0)]).matches(w) == []
    assert len(IntervalIndex([Interval(0.0, 5.01)]).matches(w)) == 1


# ---------------------------------------------------------------- metrics
def test_confusion_metrics():
    m = confusion_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3) and m["accuracy"] == pytest.approx(4 / 6)
    assert confusion_metrics([], [])["f1"] == 0.0
    with pytest.raises(ValueError):
        confusion_metrics([1], [1, 0])


def test_roc_auc():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # ties count half
    assert roc_auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert roc_auc([0, 0], [1, 2]) is None
    with pytest.raises(ValueError):
        roc_auc([0], [1, 2])


# ------------------------------------------------------------------- util
def test_threshold_for_fpr():
    scores = list(range(100))  # 0..99
    assert threshold_for_fpr(scores, 0.0) == 99
    t = threshold_for_fpr(scores, 0.05)
    assert sum(1 for s in scores if s > t) == 5
    with pytest.raises(ValueError):
        threshold_for_fpr([], 0.1)
    with pytest.raises(ValueError):
        threshold_for_fpr([1.0], 1.0)


def test_numeric_helpers():
    assert median([3, 1, 2]) == 2 and median([1, 2, 3, 4]) == 2.5
    assert mean_std([2, 4]) == (3.0, 1.0) and mean_std([]) == (0.0, 0.0)
    assert shannon_entropy(b"") == 0.0 and shannon_entropy(b"aaaa") == 0.0
    assert shannon_entropy(bytes(range(256))) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        median([])
