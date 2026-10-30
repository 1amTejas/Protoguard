# Changelog

## 0.1.0
- Modbus/TCP and MQTT 3.1.1/5 parsers with spec checks, TCP stream reassembly, pcap/pcapng reading.
- Windowed features, four detectors (rules, robust z-score, Markov chain, isolation forest), calibrated thresholds.
- JSON model files (no pickle), CLI (`simulate`, `inspect`, `train`, `detect`, `eval`), synthetic labelled traffic generator.
- Evaluation with per-attack recall, detection delay, detector ablation and optional scikit-learn baselines.
