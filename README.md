# ProtoGuard

Anomaly detection for **Modbus/TCP** and **MQTT** network traffic. It reads packet captures, parses both protocols,
learns what *normal* looks like from attack-free traffic, and flags windows of traffic that deviate, with a short
reason for every alert.

Author: [1amTejas](https://github.com/1amTejas) · License: MIT

> **Status: research prototype.** All results below come from a *synthetic* traffic simulator that ships with the
> project. They show that the pipeline works end to end. They are **not** evidence of performance on real networks.
> See [Limitations](#limitations).

## What it does

1. **Parse.** Pure-Python Modbus/TCP and MQTT (3.1.1 and 5) parsers with bounds checks and spec checks (unknown
   function codes, bad quantities, wildcard in PUBLISH topic, ...). TCP streams are reassembled
   (retransmissions, gaps, sequence wrap-around). pcap, pcapng and `.gz` files are supported.
2. **Window.** Traffic is cut into 5 s windows per (protocol, client, server) and turned into numeric features
   (request rate, function mix, distinct addresses/topics, payload size and entropy, invalid ratio, ...).
3. **Detect.** Four detectors trained on normal traffic only:
   - `rules`: allow-list of function codes, address ranges, topics and message types seen in training, plus spec violations
   - `zscore`: robust z-score (median/MAD) of every window feature
   - `markov`: first-order Markov chain over message types; flags rare transitions
   - `iforest`: isolation forest implemented in pure Python
4. **Calibrate.** Thresholds are set on held-out normal windows (the later 30 % of the training capture, never seen
   during fitting) so that the combined false-positive rate targets `--target-fpr` (default 1 %). A window alerts
   if *any* detector is over its threshold.
5. **Evaluate.** Precision/recall/F1/FPR/AUC, recall per attack type, detection delay, per-detector ablation, and
   optional scikit-learn baselines run under the same protocol.

The core has **no third-party dependencies**. scikit-learn is only needed for `--baselines`.
Models are saved as JSON, never pickle.

## Install

```bash
git clone https://github.com/1amTejas/protoguard
cd protoguard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"      # or just: pip install -e .
pytest -q                    # 92 tests; the end-to-end ones take about 30 s
```

Python 3.9 or newer. Nothing in this project opens a socket or sends traffic: it only reads capture files.

## Quick start (synthetic data)

```bash
protoguard simulate --out sim_data
protoguard eval --train sim_data/train_normal.pcap --test sim_data/test.pcap --labels sim_data/labels.csv --baselines
```

On your own data:

```bash
protoguard inspect  capture.pcap                       # what is in it?
protoguard train    normal1.pcap normal2.pcap --out model.json
protoguard detect   new.pcap --model model.json        # exit code 1 if any alert
protoguard detect   new.pcap --model model.json --json
```

Non-standard ports: `--modbus-port 1502 --mqtt-port 8883` (repeatable; plaintext only). Use
`--detectors rules,zscore` to pick a subset, and `--check-clients` to alert on never-seen client IPs (strict, off by default).

Python API:

```python
from protoguard import GuardModel, load_messages, build_windows

model = GuardModel().fit(build_windows(load_messages("normal.pcap")))
for alert in model.detect(build_windows(load_messages("new.pcap"))):
    print(alert.start, alert.client, alert.server, alert.reasons)
```

## Measured results (synthetic data only)

`protoguard simulate` (seed 1: 1800 s normal training traffic, 1200 s test traffic with 11 injected attacks:
Modbus function scan, malformed frames, read flood, write flood, setpoint tampering; MQTT connect flood, malformed
packets, oversized payload, payload injection, publish flood, topic scan). 2188 test windows, 52 of them overlap an attack.

| Model                                  | Precision | Recall |   FPR |   AUC | Attacks detected |
|----------------------------------------|----------:|-------:|------:|------:|-----------------:|
| ProtoGuard, all 4 detectors (default)  |     0.672 |  0.827 | 0.010 | 0.958 |            11/11 |
| ProtoGuard, `rules` + `zscore`         |     0.911 |  0.788 | 0.002 | 0.958 |            10/11 |
| scikit-learn IsolationForest           |     0.586 |  0.654 | 0.011 | 0.937 |            10/11 |
| scikit-learn OneClassSVM               |     0.537 |  0.827 | 0.017 | 0.930 |                - |
| scikit-learn LocalOutlierFactor        |     0.514 |  0.731 | 0.017 | 0.953 |            10/11 |

Reproduce with the `eval` command above (add `--detectors rules,zscore` for the second row).

What to take from this, and what not to:

- **The default ensemble is not the best F1 here.** `rules` + `zscore` reach F1 0.845 versus 0.741 for all four detectors.
  `markov` and `iforest` are the only ones that catch the subtle *setpoint tampering* attack (in the simulator: a few valid writes
  per minute to a known address with out-of-range values; the features do not look at register values), but they also alert on rare normal events, which costs precision. I did not tune the
  defaults to this benchmark. If false alarms matter more than catching subtle attacks, use the smaller set.
- Setpoint tampering is the hard case: recall 0.22 (2 of 9 attack windows) with all four detectors, 0 of 9 with `rules` + `zscore`.
- The simulator was written by the same person as the detectors. The attacks are the kind these features were designed to
  see, so the numbers are optimistic by construction. Real attackers and real plants will be harder.
- Attack windows are few (52), so every percentage here has wide uncertainty. Only one simulator seed was used for this table.
- Detection delay is about one window (5 s) on this data.

## Using real captures

See [docs/real-data.md](docs/real-data.md) for the label format and notes on public datasets. I have not run
ProtoGuard on any real or public capture yet, so there are no real-world results in this repository.

## Limitations

- **Evaluated on synthetic traffic only.** No real-capture evaluation has been done.
- **Plaintext only.** MQTT over TLS and Modbus/TCP Security cannot be inspected.
- **Offline analysis.** No live sniffing. Windows are built in memory, so very large captures need a lot of RAM.
- **TCP over Ethernet / Linux cooked / raw IP only** (IPv4 and IPv6 are read; IP fragments and non-TCP packets are skipped). The pcap *writer* used by the simulator is IPv4-only. Register values are not inspected.
- **Window-level labels.** A window counts as an attack window if it overlaps a labelled interval, so metrics are per
  window, not per packet.
- **The false-positive rate is a target, not a guarantee.** Thresholds come from a held-out slice of the training
  capture; if live traffic drifts (new devices, new schedules), alerts will increase. Retrain periodically and measure FPR on your own held-out normal data.
- **Training data must be attack-free.** Anything in the training capture is learned as normal.
- **MQTT 5 property skipping works only if the CONNECT packet is in the capture** (otherwise packets are read as 3.1.1).
- Needs at least 40 training and 20 calibration windows per protocol (a few minutes of traffic at 5 s windows, more for stable estimates).
- Not a replacement for a real IDS or for network segmentation.

## Layout

```
protoguard/parsers/    Modbus and MQTT parsers + packet builders (used by the simulator and tests)
protoguard/capture.py  pcap/pcapng reader and writer, frame decode, TCP reassembly
protoguard/windows.py  feature extraction
protoguard/detectors/  rules, zscore, markov, iforest, scikit-learn adapter
protoguard/model.py    GuardModel: training, calibration, alerts, JSON persistence
protoguard/simulate.py synthetic labelled traffic
protoguard/evaluate.py metrics, per-attack recall, ablation
tests/                 pytest suite
```
