"""Probe an RTC-serving deploy server without a robot.

Checks the two things that must hold before a real run:
1. LATENCY: predicts after the two cold starts stay fast (no per-call
   torch.compile re-capture from varying leftover shapes / delay ints).
2. CONTINUITY: with RTC on, the first `horizon` rows of a new chunk track the
   leftover of the previous one (that IS the anti-jerk property). Flow matching
   resamples noise every call, so without RTC these rows diverge freely.

Synthetic observations are fine here: continuity is measured between two
consecutive predicts on the SAME obs, so image content doesn't matter.
(This probe is a diagnostic client, not a server-side warmup.)

Usage:
    python -m deploy.probe_rtc --server=http://127.0.0.1:8080 [--n=5]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from deploy import protocol
from deploy.protocol import http_get_json, http_post

CONSUMED = 10  # simulated rows executed between predicts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=5, help="steady-state predicts to time")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    info = http_get_json(args.server + "/info")
    rtc = bool(info.get("rtc", False))
    horizon = int(info.get("rtc_execution_horizon", 10))
    print(f"server: {info['name']}  rtc={rtc}  horizon={horizon}")

    rng = np.random.default_rng(0)
    images = {
        key: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        for key in info["image_keys"]
    }
    state = np.zeros(info["state_dim"], dtype=np.float32)

    http_post(args.server + "/reset")

    def predict(consumed, delay):
        payload = protocol.encode_observation(images, state, "probe", consumed, delay)
        t0 = time.perf_counter()
        chunk = protocol.decode_chunk(http_post(args.server + "/predict", payload, args.timeout))
        return chunk, time.perf_counter() - t0

    prev, dt = predict(-1, 0)
    print(f"predict[cold #1, no leftover] {dt:6.2f}s")
    chunk, dt = predict(0, 5)
    print(f"predict[cold #2, guidance  ] {dt:6.2f}s")
    prev = chunk

    lat, dev = [], []
    for i in range(args.n):
        chunk, dt = predict(CONSUMED, 5)
        lat.append(dt)
        # New chunk row k should track prev row CONSUMED+k over the guidance window.
        d = np.abs(chunk[:horizon] - prev[CONSUMED:CONSUMED + horizon]).mean()
        dev.append(d)
        print(f"predict[steady #{i}] {dt:6.2f}s  prefix mean|Δ| = {d:.4f}")
        prev = chunk

    max_lat = max(lat)
    mean_dev = float(np.mean(dev))
    print(f"\nsteady-state: max latency {max_lat:.2f}s, mean prefix |Δ| {mean_dev:.4f}")
    ok = max_lat < 1.5
    print("LATENCY", "PASS" if ok else "FAIL (torch.compile re-capturing per call?)")
    if not rtc:
        print("(RTC off — prefix Δ above is the no-RTC baseline; rerun with RTC=1 to compare)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
