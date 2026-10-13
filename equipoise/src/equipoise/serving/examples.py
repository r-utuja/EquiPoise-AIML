"""Example patients for the app, and the held-out "no clear winner" summary.

The app's three "Try an example" buttons use real held-out test patients. They are picked
by scoring every held-out patient once with `recommend()` and spanning the range of
P(best costlier drug beats bevacizumab by more than one line): the highest (poorer starting
vision), the one halfway between, and the lowest among patients who started at 20/40 or
better (good starting vision). Their features and precomputed predictions go to
`data/derived/` (patient-level, never committed); only the aggregate summary, which shows
how many held-out patients get "no clear winner", is written to `reports/tables/`.
Nothing here changes a model output.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equipoise.config import Config, Prices

EXAMPLE_IDS = ("poorer", "moderate", "good")
GOOD_VISION_LETTERS = 69          # 20/40 or better, the published subgroup cut-off


def prices_signature(prices: Prices) -> str:
    """Identifies the price set a precomputed prediction was made with."""
    return json.dumps(asdict(prices), sort_keys=True, default=str)


def score_rows(service: Any, patients: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per patient: vision, CST and the effect summaries of the best costlier drug
    (by estimated benefit), as returned by `Service.predict` (no explanation)."""
    from equipoise.serving.schemas import PredictRequest

    rows = []
    for i, r in patients.reset_index(drop=True).iterrows():
        feats = feature_dict(r, features)
        pred = service.predict(PredictRequest.model_validate({"patient": feats,
                                                              "explain": False}))
        comps = [e for e in pred.estimates if e.benefit_vs_reference is not None]
        best = max(comps, key=lambda e: e.benefit_vs_reference)
        rows.append({"row": i, "va": feats["X_va_letters"], "cst": feats.get("X_cst_um"),
                     "best_drug": best.drug, "effect": best.benefit_vs_reference,
                     "effect_low": best.benefit_low, "effect_high": best.benefit_high,
                     "p_gt_mcid": best.p_benefit_gt_mcid, "equipoise": pred.equipoise,
                     **{f"equipoise_mcid_{k}": v
                        for k, v in pred.equipoise_at_sensitivity.items()},
                     "recommendation": pred.recommendation})
    return pd.DataFrame(rows)


def feature_dict(row: pd.Series, features: list[str]) -> dict[str, Any]:
    out = {}
    for f in features:
        v = row[f]
        out[f] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    return out


def pick_by_scores(scores: pd.DataFrame) -> dict[str, int]:
    """Row numbers of the three examples (ties broken deterministically)."""
    s = scores.copy()
    s["cst_f"] = s["cst"].fillna(np.inf)
    poorer = s.sort_values(["p_gt_mcid", "va"], ascending=[False, True]).iloc[0]
    good_pool = s[s.va >= GOOD_VISION_LETTERS]
    good_pool = good_pool if not good_pool.empty else s
    good = good_pool.sort_values(["p_gt_mcid", "cst_f", "va"],
                                 ascending=[True, True, False]).iloc[0]
    mid_p = (poorer.p_gt_mcid + good.p_gt_mcid) / 2
    mid_va = (poorer.va + good.va) / 2
    rest = s[~s.row.isin([poorer.row, good.row])].copy()
    rest["d_p"] = (rest.p_gt_mcid - mid_p).abs().round(9)
    rest["d_va"] = (rest.va - mid_va).abs()
    moderate = rest.sort_values(["d_p", "d_va", "row"]).iloc[0]
    return {"poorer": int(poorer.row), "moderate": int(moderate.row), "good": int(good.row)}


def pick_by_features(patients: pd.DataFrame) -> dict[str, int]:
    """Fallback without scoring (e.g. synthetic demo): poorest vision among the thicker half
    of retinas, best vision among the thinner half, and vision closest to the middle."""
    d = patients.reset_index(drop=True)
    cst = d.X_cst_um.fillna(d.X_cst_um.median())
    thick, thin = d[cst >= cst.median()], d[cst <= cst.median()]
    poorer = int(thick.X_va_letters.idxmin())
    good = int(thin.X_va_letters.idxmax())
    mid = (d.X_va_letters.loc[poorer] + d.X_va_letters.loc[good]) / 2
    rest = d.drop(index=[poorer, good])
    moderate = int((rest.X_va_letters - mid).abs().idxmin())
    return {"poorer": poorer, "moderate": moderate, "good": good}


def heldout_summary(scores: pd.DataFrame, mcid: float) -> pd.DataFrame:
    """Aggregate table: how many held-out patients get "no clear winner", and how large and
    how certain the best costlier drug's estimated benefit is."""
    n = len(scores)
    rows: list[tuple[str, Any]] = [("n_heldout", n), ("mcid_letters", mcid),
                                   ("n_no_clear_winner_mcid_" + f"{mcid:g}",
                                    int(scores.equipoise.sum())),
                                   ("share_no_clear_winner_mcid_" + f"{mcid:g}",
                                    float(scores.equipoise.mean()))]
    for c in sorted(c for c in scores.columns if c.startswith("equipoise_mcid_")):
        m = c.removeprefix("equipoise_mcid_")
        rows += [(f"n_no_clear_winner_mcid_{m}", int(scores[c].sum())),
                 (f"share_no_clear_winner_mcid_{m}", float(scores[c].mean()))]
    for drug, cnt in scores.best_drug.value_counts().sort_index().items():
        rows.append((f"n_best_costlier_is_{drug}", int(cnt)))
    p = scores.p_gt_mcid
    width = scores.effect_high - scores.effect_low
    rows += [("p_gt_mcid_median", float(p.median())), ("p_gt_mcid_max", float(p.max())),
             ("n_p_gt_mcid_at_least_0.5", int((p >= 0.5).sum())),
             ("effect_letters_median", float(scores.effect.median())),
             ("effect_letters_min", float(scores.effect.min())),
             ("effect_letters_max", float(scores.effect.max())),
             ("effect_low_letters_max", float(scores.effect_low.max())),
             ("interval_width_letters_median", float(width.median()))]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def examples_path(cfg: Config) -> Path:
    return cfg.paths.derived_dir / cfg.raw["ui"]["examples_file"]


def write_examples(path: Path, bundle_hash: str, prices_sig: str,
                   examples: dict[str, dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bundle_hash": bundle_hash, "prices_signature": prices_sig,
                                "examples": examples}, indent=1), encoding="utf-8")
    return path


def read_examples(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
