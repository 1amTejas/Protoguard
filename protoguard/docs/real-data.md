# Using real captures

ProtoGuard only reads captures you give it. It ships no real traffic.

1. **Normal traffic to train on.** Capture your own plant/lab/broker for as long as you can (hours, ideally days),
   with no attacks in it. Training on traffic that contains attacks teaches the model that they are normal.
2. **Labels.** Write a CSV with the header `start,end,attack,src_ip,dst_ip` (times are Unix seconds; `attack`, `src_ip`,
   `dst_ip` are optional). Without IPs an interval labels every channel active in that period.
3. **Public datasets.** Several public IIoT/IoT datasets include Modbus or MQTT attacks (for example Edge-IIoTset,
   ToN-IoT, MQTTset). I have not tested ProtoGuard on any of them: check that the data is available, read its licence,
   and check that it contains packet captures (not only pre-extracted CSV features) before relying on it.
4. **Plaintext only.** MQTT over TLS and Modbus/TCP Security cannot be parsed.
