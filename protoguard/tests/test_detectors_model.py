import json
import random

import pytest

from protoguard import GuardModel, ModelError, build_windows
from protoguard.detectors import IsolationForestDetector, MarkovDetector, RuleDetector, ZScoreDetector
from protoguard.messages import REQUEST, RESPONSE, Message
from protoguard.windows import Window


def mb(ts, fn=3, address=0, quantity=10, direction=REQUEST, client="c1", server="s1", unit=1, **kw):
    return Message(ts=ts, proto="modbus", client=client, server=server, direction=direction, kind=f"fc{fn}",
                   size=12, function=fn, address=address, quantity=quantity, unit_id=unit, **kw)


def normal_modbus(duration=600.0, seed=0, client="c1"):
    """A poller reading registers 0-9 every second, with small jitter, plus replies."""
    rng = random.Random(seed)
    msgs, t = [], 0.0
    while t < duration:
        msgs.append(mb(t, client=client))
        msgs.append(mb(t + 0.01, direction=RESPONSE, address=None, quantity=None, client=client))
        t += 1.0 + rng.uniform(-0.1, 0.1)
    return build_windows(msgs, 5.0)


@pytest.fixture(scope="module")
def trained():
    return GuardModel(seed=1).fit(normal_modbus())


# ---------------------------------------------------------------- detectors
def test_rules_flag_unseen_function_address_and_spec_violations():
    d = RuleDetector()
    d.fit(normal_modbus())
    (ok,) = build_windows([mb(0.0)], 5.0)
    assert d.score(ok) == 0
    (write,) = build_windows([mb(0.0, fn=6, address=0, quantity=1)], 5.0)
    assert d.score(write) > 0 and any("function" in r for r in d.explain(write))
    (addr,) = build_windows([mb(0.0, address=500)], 5.0)
    assert d.score(addr) > 0
    (invalid,) = build_windows([mb(0.0, valid=False, problems=("bad_quantity",))], 5.0)
    assert d.score(invalid) > 0


def test_rules_client_check_is_opt_in():
    windows = normal_modbus()
    (stranger,) = build_windows([mb(0.0, client="9.9.9.9")], 5.0)
    off, on = RuleDetector(), RuleDetector(check_clients=True)
    off.fit(windows)
    on.fit(windows)
    assert off.score(stranger) == 0 and on.score(stranger) > 0


def test_zscore_scores_a_burst_above_normal():
    d = ZScoreDetector()
    windows = normal_modbus()
    d.fit(windows)
    (burst,) = build_windows([mb(i * 0.01, client="c1") for i in range(400)], 5.0)
    assert d.score(burst) > 3 * max(d.score(w) for w in windows)
    assert d.explain(burst)


def test_markov_scores_unusual_ordering_higher():
    windows = normal_modbus()
    d = MarkovDetector()
    d.fit(windows)
    odd = [mb(i * 0.1, fn=3 if i % 2 else 16, quantity=1) for i in range(30)]
    (w,) = build_windows(odd, 5.0)
    assert d.score(w) > d.score(windows[3])


def test_iforest_is_deterministic_and_roundtrips():
    windows = normal_modbus()
    a, b = IsolationForestDetector(seed=5), IsolationForestDetector(seed=5)
    a.fit(windows)
    b.fit(windows)
    assert [a.score(w) for w in windows[:5]] == [b.score(w) for w in windows[:5]]
    c = IsolationForestDetector.from_dict(json.loads(json.dumps(a.to_dict())))
    assert [c.score(w) for w in windows[:5]] == pytest.approx([a.score(w) for w in windows[:5]])


@pytest.mark.parametrize("cls", [RuleDetector, ZScoreDetector, MarkovDetector, IsolationForestDetector])
def test_detector_state_round_trips_through_json(cls):
    windows = normal_modbus()
    d = cls()
    d.fit(windows)
    again = cls.from_dict(json.loads(json.dumps(d.to_dict())))
    assert [again.score(w) for w in windows[:6]] == pytest.approx([d.score(w) for w in windows[:6]])


@pytest.mark.parametrize("cls", [RuleDetector, ZScoreDetector, MarkovDetector, IsolationForestDetector])
def test_detector_from_dict_rejects_garbage(cls):
    for bad in ({}, {"x": 1}, None, "text", {"features": "no"}):
        with pytest.raises((ModelError, ValueError, TypeError, KeyError)):
            cls.from_dict(bad)


# -------------------------------------------------------------------- model
def test_model_refuses_too_little_data():
    with pytest.raises(ModelError):
        GuardModel().fit(normal_modbus(duration=60.0))
    with pytest.raises(ModelError):
        GuardModel().fit([])


def test_constructor_validation():
    for kwargs in ({"window_s": 0}, {"target_fpr": 0}, {"target_fpr": 1}, {"detectors": ()}, {"detectors": ("nope",)}):
        with pytest.raises(ModelError):
            GuardModel(**kwargs)
    with pytest.raises(ModelError):
        GuardModel().fit(normal_modbus(), calibration_fraction=0.9)


def test_scoring_before_training_fails():
    with pytest.raises(ModelError):
        GuardModel().score_windows(normal_modbus(60))


def test_normal_traffic_is_quiet_and_an_attack_alerts_with_reasons(trained):
    held_out = normal_modbus(seed=99)
    assert sum(1 for s in trained.score_windows(held_out) if s.alert) <= max(2, int(0.05 * len(held_out)))
    attack = [mb(i * 0.02, fn=5, address=0, quantity=1) for i in range(100)]
    alerts = trained.detect(build_windows(attack, 5.0))
    assert alerts and all(a.reasons for a in alerts)
    assert alerts[0].to_dict()["proto"] == "modbus"


def test_windows_of_untrained_protocol_are_skipped(trained):
    mqtt_msg = Message(ts=1.0, proto="mqtt", client="a", server="b", direction=REQUEST, kind="PUBLISH", size=20,
                       topic="t", payload_len=1, payload_entropy=0.0)
    assert trained.score_windows(build_windows([mqtt_msg], 5.0)) == []


def test_save_and_load_give_identical_scores(trained, tmp_path):
    path = tmp_path / "m.json"
    trained.save(path)
    assert not list(tmp_path.glob("*.tmp"))
    loaded = GuardModel.load(path)
    windows = normal_modbus(seed=7)
    assert [s.combined for s in loaded.score_windows(windows)] == pytest.approx(
        [s.combined for s in trained.score_windows(windows)]
    )


def test_load_rejects_bad_files(trained, tmp_path):
    good = trained.to_dict()

    def write(obj, name="bad.json"):
        p = tmp_path / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
        return p

    with pytest.raises(ModelError):
        GuardModel.load(tmp_path / "missing.json")
    with pytest.raises(ModelError):
        GuardModel.load(write("{not json"))
    with pytest.raises(ModelError):
        GuardModel.load(write([1, 2]))
    with pytest.raises(ModelError):
        GuardModel.load(write({**good, "schema_version": 99}))
    broken = json.loads(json.dumps(good))
    del broken["protocols"]["modbus"]["thresholds"]
    with pytest.raises(ModelError):
        GuardModel.load(write(broken))
    unknown = json.loads(json.dumps(good))
    unknown["protocols"]["modbus"]["detectors"]["evil"] = {}
    with pytest.raises(ModelError):
        GuardModel.load(write(unknown))
    mismatch = json.loads(json.dumps(good))
    mismatch["protocols"]["modbus"]["thresholds"].pop("zscore")
    with pytest.raises(ModelError):
        GuardModel.load(write(mismatch))
    nonfinite = write(json.dumps(good).replace('"zscore": ', '"zscore": 1e999, "_x": ', 1))
    with pytest.raises(ModelError):
        GuardModel.load(nonfinite)


def test_custom_factory_models_cannot_be_saved():
    m = GuardModel(detectors=("mine",), factories={"mine": ZScoreDetector}).fit(normal_modbus())
    with pytest.raises(ModelError):
        m.to_dict()


def test_window_dataclass_is_usable_directly():
    w = Window(proto="modbus", client="a", server="b", start=0.0, end=5.0, messages=[], features={})
    assert w.label == 0
