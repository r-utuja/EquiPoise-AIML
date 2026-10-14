"""Subgroup checks: does the model work equally well for different kinds of patients?

For every patient we need a prediction from a model that never saw them: training patients
get out-of-fold predictions (5-fold cross-fitting, refitting the effect and outcome models
without their fold) and held-out patients get the final bundle's predictions. Within each
subgroup (sex, race, diabetes type, baseline vision) we report how well the letters model
fits (RMSE, R^2 under the drug actually received), how well the laser model ranks patients
(AUC), and calibration of the predicted aflibercept effect: the mean predicted effect next to
the observed aflibercept - bevacizumab difference with its 95% CI. Subgroups smaller than
the configured minimum are flagged as unreliable rather than interpreted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from equipoise.causal.data import arm_codes, prepare
from equipoise.causal.learners import base_learner
from equipoise.config import Config
from equipoise.data.splits import cv_folds
from equipoise.stats.replication import welch_ci
from equipoise.supervised.models import classification_models

RACE_GROUPS = {"White": "White", "Black/African American": "Black/African American"}


def subgroup_labels(frame: pd.DataFrame, col: str) -> pd.Series:
    """Readable subgroup label per patient for one subgroup variable."""
    s = frame[col]
    if col == "X_female":
        return s.map({1: "Female", 0: "Male"})
    if col == "X_va_below_69":
        return s.map({1: "Vision < 69 letters", 0: "Vision >= 69 letters"})
    if col == "X_race":
        return s.map(RACE_GROUPS).fillna("Other / not reported")
    return s.fillna("Not reported").astype(str)


def out_of_sample_predictions(train: pd.DataFrame, test: pd.DataFrame, bundle, factory,
                              cfg: Config) -> pd.DataFrame:
    """Per patient (complete cases): predicted letters under the received arm, predicted
    afl - bev effect and laser probability, all from models that did not see the patient."""
    gb_params = bundle.extras.get("tuned_gb_params", {})
    codes = arm_codes(cfg)
    y = cfg.primary_outcome
    tr_all = train.dropna(subset=[y]).reset_index(drop=True)
    parts = []
    for i_tr, i_va in cv_folds(tr_all, cfg.n_folds, cfg.seed):
        ftr, fva, _ = prepare(tr_all.iloc[i_tr], tr_all.iloc[i_va], cfg)
        effect = factory(cfg).fit(ftr.X, ftr.T, ftr.Y).effect(fva.X)
        mu = np.column_stack([base_learner("gb", cfg, gb_params).fit(
            ftr.X[ftr.T == a], ftr.Y[ftr.T == a]).predict(fva.X) for a in range(len(codes))])
        laser = classification_models(cfg)["Logistic regression"].fit(
            tr_all.iloc[i_tr], tr_all.iloc[i_tr]["Y_laser_yr1"].astype(int))
        parts.append(pd.DataFrame({
            "pt_id": fva.frame.pt_id, "source": "out-of-fold (training)",
            "pred_letters": mu[np.arange(len(fva)), fva.T], "pred_effect_afl": effect[:, 0],
            "p_laser": laser.predict_proba(fva.frame)[:, 1]}))
    te = test.dropna(subset=[y]).reset_index(drop=True)
    X = bundle.preprocessor.transform(te)
    T = te["treatment"].map(codes).to_numpy(int)
    mu = np.column_stack([m.predict(X) for m in bundle.outcome_models])
    parts.append(pd.DataFrame({
        "pt_id": te.pt_id, "source": "held-out (final bundle)",
        "pred_letters": mu[np.arange(len(te)), T],
        "pred_effect_afl": bundle.effect_model.effect(X)[:, 0],
        "p_laser": bundle.laser_model.predict_proba(te)[:, 1]}))
    return pd.concat(parts, ignore_index=True)


def subgroup_report(preds: pd.DataFrame, frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per subgroup variable x level."""
    e = cfg.explain
    level = float(cfg.stats["ci_level"])
    d = preds.merge(frame, on="pt_id", how="left")
    y = cfg.primary_outcome
    rows = []
    for col in e["subgroups"]:
        labels = subgroup_labels(d, col)
        for lab in sorted(labels.dropna().unique()):
            g = d[labels == lab]
            obs, pred = g[y].to_numpy(float), g.pred_letters.to_numpy(float)
            afl = g.loc[g.treatment == "Aflibercept", y]
            bev = g.loc[g.treatment == cfg.reference_arm, y]
            diff, lo, hi, _ = (welch_ci(afl, bev, level) if min(len(afl), len(bev)) >= 3
                               else (np.nan,) * 4)
            laser_y = g["Y_laser_yr1"].to_numpy(int)
            rows.append({
                "subgroup": col.replace("X_", ""), "level": lab, "n": len(g),
                "reliable": len(g) >= int(e["subgroup_min_n"]) and min(len(afl), len(bev)) >= 10,
                "letters_rmse": float(np.sqrt(np.mean((obs - pred) ** 2))),
                "letters_r2": float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean())
                                                                          ** 2)),
                "laser_auc": float(roc_auc_score(laser_y, g.p_laser))
                if len(np.unique(laser_y)) == 2 else np.nan,
                "mean_predicted_effect_afl": float(g.pred_effect_afl.mean()),
                "observed_effect_afl": diff, "observed_ci_low": lo, "observed_ci_high": hi,
                "n_afl": len(afl), "n_bev": len(bev)})
    return pd.DataFrame(rows)
