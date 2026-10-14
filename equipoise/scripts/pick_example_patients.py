"""Score every held-out patient once, pick the app's three example patients, and write the
held-out "no clear winner" summary.

Writes:
  data/derived/app_examples.json                 features + precomputed predictions of the
                                                 three examples (patient-level, never committed)
  reports/tables/m8_heldout_no_clear_winner.csv/.md   aggregate summary (committed)

Usage (about 5 minutes on the real bundle):
    python scripts/pick_example_patients.py [--synthetic]
"""

from __future__ import annotations

import argparse
import warnings

from equipoise.config import for_data_source, load_config
from equipoise.decision.recommend import FEATURES, default_bundle_path
from equipoise.serving import examples as ex
from equipoise.serving.schemas import PredictRequest
from equipoise.serving.service import Service, held_out_patients
from equipoise.tracking import log_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic bundle/data")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = load_config()
    service = Service.from_path(default_bundle_path(cfg, args.synthetic), cfg, log_path=None)
    patients = held_out_patients(cfg, args.synthetic)
    if patients.empty:
        raise SystemExit("No held-out patients found (build the table and split first).")
    print(f"Scoring {len(patients)} held-out patients ...")
    scores = ex.score_rows(service, patients, FEATURES)

    out_cfg = for_data_source(cfg, args.synthetic)
    summary = ex.heldout_summary(scores, float(service.bundle.mcid_letters))
    csv_path, _ = log_table(out_cfg, "decision", summary, "heldout_no_clear_winner")
    print(summary.to_string(index=False))
    print(f"wrote {csv_path}")

    picks = ex.pick_by_scores(scores)
    out = {}
    for name, row in picks.items():
        feats = ex.feature_dict(patients.reset_index(drop=True).iloc[row], FEATURES)
        pred = service.predict(PredictRequest.model_validate({"patient": feats,
                                                              "explain": True}))
        out[name] = {"features": feats, "prediction": pred.model_dump(mode="json")}
        s = scores.iloc[row]
        print(f"{name:9s} P(>MCID)={s.p_gt_mcid:.2f} effect={s.effect:+.1f}")
    path = ex.write_examples(ex.examples_path(out_cfg), service.bundle.training_data_hash,
                             ex.prices_signature(service.prices(None)), out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
