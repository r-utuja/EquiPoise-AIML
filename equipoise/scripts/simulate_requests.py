"""Write simulated API traffic to the request log, to demonstrate drift monitoring.

Draws held-out test patients (real or synthetic) and appends their baseline features to
`logs/requests.jsonl` in exactly the API's log format. A configurable share is shifted
(older, higher HbA1c, worse vision: `monitor.simulate` in config/default.yaml) so the drift
report has something to find. The traffic is SIMULATED and labelled as such
(endpoint "simulated"); it is not real usage.

Usage:
    python scripts/simulate_requests.py [--synthetic] [--log PATH]
    python -m equipoise monitor [--synthetic]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from equipoise.config import load_config
from equipoise.data.schema import ANALYSIS_COLUMNS
from equipoise.serving.service import append_request_log, held_out_patients, resolve_log_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true", help="use synthetic patients")
    ap.add_argument("--log", type=Path, default=None, help="request log (default: config)")
    args = ap.parse_args()
    cfg = load_config()
    sim = cfg.raw["monitor"]["simulate"]
    log = args.log or resolve_log_path(cfg)
    if log is None:
        raise SystemExit("request logging is disabled (EQUIPOISE_REQUEST_LOG=off)")
    pool = held_out_patients(cfg, args.synthetic)
    if pool.empty:
        raise SystemExit("no held-out patients found (build the data and split first)")
    rng = np.random.default_rng(cfg.seed)
    specs = {c.name: c for c in ANALYSIS_COLUMNS if c.role == "feature"}
    n = int(sim["n_requests"])
    shifted = rng.random(n) < float(sim["shifted_share"])
    for i in range(n):
        row = pool.iloc[int(rng.integers(len(pool)))]
        feats = {}
        for name, spec in specs.items():
            v = row[name]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                feats[name] = None
                continue
            v = v.item() if hasattr(v, "item") else v
            if shifted[i] and name in sim["shift"]:
                v = float(np.clip(v + float(sim["shift"][name]), spec.min, spec.max))
            feats[name] = v
        feats["X_va_below_69"] = int(feats["X_va_letters"] < 69)
        append_request_log(log, "simulated", feats)
    print(f"appended {n} simulated requests ({int(shifted.sum())} shifted) to {log}")


if __name__ == "__main__":
    main()
