import json

import pytest

from protoguard import GuardModel
from protoguard.cli import main
from protoguard.evaluate import evaluate_model, format_report


def test_simulated_dataset_parses_cleanly(dataset):
    from protoguard import load_messages

    msgs = load_messages(dataset / "test.pcap")
    assert msgs and all(m.valid for m in load_messages(dataset / "train_normal.pcap"))
    assert {m.proto for m in msgs} == {"modbus", "mqtt"}


def test_pipeline_detects_most_attacks_with_few_false_alarms(model, test_windows, intervals):
    """Loose bounds on synthetic data: this checks the pipeline works, not real-world performance."""
    result = evaluate_model(model, test_windows, intervals)
    o = result["overall"]
    assert o["false_positive_rate"] < 0.05
    assert result["intervals"]["detection_rate"] >= 0.7
    assert o["auc"] is not None and o["auc"] > 0.8
    assert set(result["detectors"]) == {"rules", "zscore", "markov", "iforest"}
    assert "Overall" in format_report(result) or "overall" in format_report(result).lower()


def test_model_survives_save_and_load_in_the_pipeline(model, test_windows, tmp_path):
    p = tmp_path / "m.json"
    model.save(p)
    a = [s.alert for s in model.score_windows(test_windows)]
    b = [s.alert for s in GuardModel.load(p).score_windows(test_windows)]
    assert a == b


def test_training_is_deterministic(train_windows, test_windows):
    one = GuardModel(seed=2).fit(train_windows).score_windows(test_windows[:50])
    two = GuardModel(seed=2).fit(train_windows).score_windows(test_windows[:50])
    assert [s.combined for s in one] == [s.combined for s in two]


# ------------------------------------------------------------------- CLI
def test_cli_train_detect_inspect(dataset, tmp_path, capsys):
    model = str(tmp_path / "model.json")
    assert main(["train", str(dataset / "train_normal.pcap"), "--out", model]) == 0
    assert main(["inspect", str(dataset / "test.pcap")]) == 0
    assert "modbus" in capsys.readouterr().out
    assert main(["detect", str(dataset / "test.pcap"), "--model", model]) == 1  # alerts found
    capsys.readouterr()
    assert main(["detect", str(dataset / "test.pcap"), "--model", model, "--json"]) == 1
    alerts = json.loads(capsys.readouterr().out)
    assert alerts and {"proto", "start", "reasons"} <= set(alerts[0])
    assert main(["detect", str(dataset / "train_normal.pcap"), "--model", model]) in (0, 1)


def test_cli_eval_with_ablation_and_save(dataset, tmp_path, capsys):
    out = tmp_path / "r.json"
    code = main(["eval", "--train", str(dataset / "train_normal.pcap"), "--test", str(dataset / "test.pcap"),
                 "--labels", str(dataset / "labels.csv"), "--detectors", "rules,zscore", "--save", str(out)])
    assert code == 0 and "rules + zscore" in capsys.readouterr().out
    assert set(json.loads(out.read_text())["protoguard"]["detectors"]) == {"rules", "zscore"}


def test_cli_errors_return_2(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.pcap")]) == 2
    assert main(["detect", str(tmp_path / "nope.pcap"), "--model", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([])


def test_cli_simulate(tmp_path, capsys):
    assert main(["simulate", "--out", str(tmp_path / "s"), "--train-duration", "300", "--test-duration", "400"]) == 0
    assert (tmp_path / "s" / "labels.csv").exists()
    assert "next:" in capsys.readouterr().out


def test_sklearn_baselines_run_if_installed(dataset, capsys):
    pytest.importorskip("sklearn")
    code = main(["eval", "--train", str(dataset / "train_normal.pcap"), "--test", str(dataset / "test.pcap"),
                 "--labels", str(dataset / "labels.csv"), "--baselines"])
    assert code == 0 and "baseline" in capsys.readouterr().out
