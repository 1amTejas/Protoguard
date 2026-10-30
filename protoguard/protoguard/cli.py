"""Command line interface: ``protoguard simulate | inspect | train | detect | eval``."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

from ._version import __version__
from .capture import load_messages
from .errors import ProtoGuardError
from .windows import build_windows


def _ports(args: argparse.Namespace):
    return args.modbus_port, args.mqtt_port


def _load_windows(paths: List[str], args: argparse.Namespace, window_s: float):
    modbus_ports, mqtt_ports = _ports(args)
    messages = []
    for path in paths:
        stats: Counter = Counter()
        found = load_messages(path, modbus_ports, mqtt_ports, stats)
        print(f"read {path}: {stats['frames']} frames, {len(found)} messages", file=sys.stderr)
        for key in ("gaps", "buffer_overflow", "evicted_streams"):
            if stats[key]:
                print(f"  warning: {stats[key]} {key.replace('_', ' ')}", file=sys.stderr)
        messages.extend(found)
    if not messages:
        raise ProtoGuardError("no Modbus/TCP or MQTT messages found (check the ports and that the traffic is not TLS)")
    return build_windows(messages, window_s)


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .simulate import build_dataset

    info = build_dataset(args.out, args.seed, args.train_duration, args.test_duration)
    print("wrote synthetic data (no traffic was sent on any network):")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print(
        "\nnext: protoguard eval --train {train_pcap} --test {test_pcap} --labels {labels}".format(**info)
    )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    modbus_ports, mqtt_ports = _ports(args)
    stats: Counter = Counter()
    messages = load_messages(args.capture, modbus_ports, mqtt_ports, stats)
    print(f"{args.capture}: {stats['frames']} frames, {len(messages)} messages")
    if not messages:
        return 0
    print(f"time span: {_stamp(min(m.ts for m in messages))} to {_stamp(max(m.ts for m in messages))} UTC")
    for proto in sorted({m.proto for m in messages}):
        part = [m for m in messages if m.proto == proto]
        print(f"\n{proto}: {len(part)} messages, {sum(1 for m in part if not m.valid)} invalid")
        print(f"  clients: {', '.join(sorted({m.client for m in part}))}")
        print(f"  servers: {', '.join(sorted({m.server for m in part}))}")
        kinds = Counter(m.kind for m in part)
        print("  message types: " + ", ".join(f"{k}={v}" for k, v in kinds.most_common(12)))
        problems = Counter(p for m in part for p in m.problems)
        if problems:
            print("  problems: " + ", ".join(f"{k}={v}" for k, v in problems.most_common()))
    for key in ("gaps", "retransmissions", "buffer_overflow", "non_tcp_or_unsupported"):
        if stats[key]:
            print(f"note: {stats[key]} {key.replace('_', ' ')}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .model import GuardModel

    windows = _load_windows(args.capture, args, args.window)
    model = GuardModel(
        window_s=args.window, target_fpr=args.target_fpr, detectors=tuple(args.detectors.split(",")),
        seed=args.seed, check_clients=args.check_clients,
    ).fit(windows)
    model.save(args.out)
    print(f"saved {args.out}")
    for proto, pm in model.protocols.items():
        print(f"  {proto}: {pm.train_windows} training + {pm.calibration_windows} calibration windows")
        print("    thresholds: " + ", ".join(f"{k}={v:.3g}" for k, v in pm.thresholds.items()))
    for note in model.notes:
        print(f"  note: {note}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    from .model import GuardModel

    model = GuardModel.load(args.model)
    windows = _load_windows([args.capture], args, model.window_s)
    alerts = model.detect(windows)
    if args.json:
        print(json.dumps([a.to_dict() for a in alerts], indent=2))
    else:
        for a in alerts[: args.max_alerts]:
            print(f"[{_stamp(a.start)}] {a.proto} {a.client} -> {a.server}  score {a.score:.1f}  ({', '.join(a.detectors)})")
            for reason in a.reasons[:4]:
                print(f"    {reason}")
        if len(alerts) > args.max_alerts:
            print(f"... and {len(alerts) - args.max_alerts} more (raise --max-alerts or use --json)")
        print(f"{len(alerts)} alert(s) in {len(windows)} windows")
    return 1 if alerts else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_model, format_report
    from .labels import load_labels
    from .model import GuardModel

    intervals = load_labels(args.labels)
    train_windows = _load_windows(args.train, args, args.window)
    test_windows = _load_windows([args.test], args, args.window)

    def run(model: GuardModel, title: str) -> dict:
        model.fit(train_windows)
        result = evaluate_model(model, test_windows, intervals)
        print(f"=== {title} ===")
        print(format_report(result))
        print()
        return result

    results = {}
    names = tuple(args.detectors.split(","))
    results["protoguard"] = run(
        GuardModel(window_s=args.window, target_fpr=args.target_fpr, detectors=names, seed=args.seed,
                   check_clients=args.check_clients),
        "ProtoGuard (" + " + ".join(names) + ")",
    )
    if args.baselines:
        from .detectors.sklearn_adapter import baseline_factories

        factories = baseline_factories(args.seed)
        for name in factories:
            results[name] = run(
                GuardModel(window_s=args.window, target_fpr=args.target_fpr, detectors=(name,),
                           seed=args.seed, factories={name: factories[name]}),
                f"baseline: {name}",
            )
        print("summary (same train data, same calibration protocol):")
        print(f"  {'model':14s} {'precision':>9s} {'recall':>7s} {'FPR':>6s} {'AUC':>6s}")
        for name, r in results.items():
            o = r["overall"]
            auc = "n/a" if o["auc"] is None else f"{o['auc']:.3f}"
            print(f"  {name:14s} {o['precision']:9.3f} {o['recall']:7.3f} {o['false_positive_rate']:6.3f} {auc:>6s}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"saved {args.save}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protoguard", description="Anomaly detection for Modbus/TCP and MQTT traffic.",
    )
    parser.add_argument("--version", action="version", version=f"protoguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ports = argparse.ArgumentParser(add_help=False)
    ports.add_argument("--modbus-port", type=int, action="append", default=None, help="Modbus/TCP port (repeatable, default 502)")
    ports.add_argument("--mqtt-port", type=int, action="append", default=None, help="MQTT port (repeatable, default 1883)")

    tuning = argparse.ArgumentParser(add_help=False)
    tuning.add_argument("--window", type=float, default=5.0, help="window length in seconds (default 5)")
    tuning.add_argument("--target-fpr", type=float, default=0.01, help="target false-positive rate on normal traffic")
    tuning.add_argument("--seed", type=int, default=0)
    tuning.add_argument("--check-clients", action="store_true", help="also alert on never-seen client IPs (strict; off by default)")

    sim = sub.add_parser("simulate", help="write synthetic labelled pcaps (nothing is sent on a network)")
    sim.add_argument("--out", default="sim_data")
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--train-duration", type=float, default=1800.0)
    sim.add_argument("--test-duration", type=float, default=1200.0)
    sim.set_defaults(func=_cmd_simulate)

    insp = sub.add_parser("inspect", parents=[ports], help="summarise a capture")
    insp.add_argument("capture")
    insp.set_defaults(func=_cmd_inspect)

    train = sub.add_parser("train", parents=[ports, tuning], help="train on captures of NORMAL traffic")
    train.add_argument("capture", nargs="+")
    train.add_argument("--out", default="model.json")
    train.add_argument("--detectors", default="rules,zscore,markov,iforest")
    train.set_defaults(func=_cmd_train)

    det = sub.add_parser("detect", parents=[ports], help="find anomalies in a capture")
    det.add_argument("capture")
    det.add_argument("--model", default="model.json")
    det.add_argument("--json", action="store_true")
    det.add_argument("--max-alerts", type=int, default=50)
    det.set_defaults(func=_cmd_detect)

    ev = sub.add_parser("eval", parents=[ports, tuning], help="train, then measure on a labelled capture")
    ev.add_argument("--train", nargs="+", required=True, help="capture(s) of normal traffic")
    ev.add_argument("--test", required=True, help="labelled capture")
    ev.add_argument("--labels", required=True, help="CSV of attack intervals")
    ev.add_argument("--detectors", default="rules,zscore,markov,iforest", help="comma-separated subset, for ablations")
    ev.add_argument("--baselines", action="store_true", help="also run scikit-learn baselines (needs scikit-learn)")
    ev.add_argument("--save", help="write all results to this JSON file")
    ev.set_defaults(func=_cmd_eval)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "modbus_port"):
        args.modbus_port = args.modbus_port or [502]
        args.mqtt_port = args.mqtt_port or [1883]
    try:
        return args.func(args)
    except (ProtoGuardError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
