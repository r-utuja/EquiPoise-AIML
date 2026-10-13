"""Early-responder forecasting: the ONE place post-baseline data is allowed as model input.

After a few injections, how a patient has responded so far is informative about week 52.
This model predicts week-52 letters gained from the baseline features plus the vision
change at weeks 4, 8 and 12 and the early slope (least-squares letters per week over
weeks 0-12), and is compared with the same model using baseline features only. Early
columns carry the `E_` prefix and pass the leakage gate only via `allow_early=True`, which
no other module uses. It answers "how is this patient doing?" after week 12, not "which drug
should they start?" - it is never used for the treatment decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equipoise.config import Config
from equipoise.data.splits import cv_folds
from equipoise.features import EARLY_PREFIX, assert_no_leakage
from equipoise.preprocess.encode import build_preprocessor
from equipoise.supervised.evaluate import regression_metrics


def early_features(visits: pd.DataFrame, analysis: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """E_ columns per patient: vision change at each early week label and the early slope."""
    weeks = [int(w) for w in cfg.longitudinal["early_weeks"]]
    base = analysis.set_index("pt_id")["X_va_letters"]
    out = pd.DataFrame({"pt_id": analysis["pt_id"]})
    for w in weeks:
        at = visits[visits.visit_label == f"{w} week"].drop_duplicates("pt_id")
        va = at.set_index("pt_id")["va_letters"]
        out[f"{EARLY_PREFIX}va_change_wk{w}"] = out.pt_id.map(va) - out.pt_id.map(base)
    early = visits[(visits.days_from_rand <= max(weeks) * 7 + 21) & visits.va_letters.notna()]
    slopes = {}
    for pid, g in early.groupby("pt_id"):
        if len(g) >= 2:
            slopes[pid] = float(np.polyfit(g.days_from_rand / 7, g.va_letters, 1)[0])
    out[f"{EARLY_PREFIX}early_slope"] = out.pt_id.map(slopes)
    return out


def _early_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(EARLY_PREFIX)]


def make_model(cfg: Config, with_early: bool, early_cols: list[str]) -> Pipeline:
    """Ridge on baseline features + arm (+ early vision when requested)."""
    pre = build_preprocessor(cfg, include_treatment=True)
    baseline_inputs = sorted({c for _, _, cols in pre.transformers for c in cols})
    blocks = [("baseline", pre, baseline_inputs)]
    if with_early:
        assert_no_leakage(early_cols, allow_early=True)
        blocks.append(("early", Pipeline([("impute", SimpleImputer(strategy="median")),
                                          ("scale", StandardScaler())]), early_cols))
    union = ColumnTransformer(blocks, remainder="drop")
    return Pipeline([("features", union),
                     ("model", Ridge(alpha=float(cfg.longitudinal["early_ridge_alpha"])))])


def compare(train: pd.DataFrame, test: pd.DataFrame, cfg: Config
            ) -> tuple[pd.DataFrame, Pipeline, dict[str, np.ndarray]]:
    """5-fold CV on training patients and one held-out score: baseline-only vs + early."""
    y = cfg.primary_outcome
    tr = train.dropna(subset=[y]).reset_index(drop=True)
    te = test.dropna(subset=[y]).reset_index(drop=True)
    cols = _early_columns(tr)
    folds = list(cv_folds(tr, cfg.n_folds, cfg.seed))
    rows, preds, final_early = [], {"y_test": te[y].to_numpy(float)}, None
    for name, with_early in (("Baseline features only", False),
                             ("Baseline + vision at weeks 4/8/12 + early slope", True)):
        model = make_model(cfg, with_early, cols)
        cv = []
        for i_tr, i_va in folds:
            m = clone(model).fit(tr.iloc[i_tr], tr[y].iloc[i_tr])
            cv.append(regression_metrics(tr[y].iloc[i_va].to_numpy(), m.predict(tr.iloc[i_va])))
        cvd = pd.DataFrame(cv)
        final = clone(model).fit(tr, tr[y])
        p = final.predict(te)
        held = regression_metrics(te[y].to_numpy(), p)
        row = {"Model": name, "n_train": len(tr), "n_test": len(te)}
        row.update({f"CV {k}": cvd[k].mean() for k in cvd.columns})
        row.update({f"CV {k} sd": cvd[k].std(ddof=1) for k in cvd.columns})
        row.update({f"Test {k}": v for k, v in held.items()})
        rows.append(row)
        preds[name] = p
        if with_early:
            final_early = final
    return pd.DataFrame(rows), final_early, preds
