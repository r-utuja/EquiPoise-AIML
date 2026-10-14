"""M9 stage: SHAP for the effect model and subgroup (fairness) checks.

SHAP is computed for held-out patients with training patients as background; the waterfall
explains the 'typical patient' from M8 (built from training medians, not a real person).
Subgroup checks use out-of-fold predictions for training patients plus the final bundle's
predictions for held-out patients, so no prediction comes from a model that saw the patient.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from equipoise import tracking
from equipoise.causal.bundle import load_bundle
from equipoise.causal.run import _chosen_factory
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.splits import get_split
from equipoise.decision.recommend import default_bundle_path, patient_frame, serving_mode
from equipoise.decision.run import typical_patient
from equipoise.explain.fairness import out_of_sample_predictions, subgroup_report
from equipoise.explain.shap_effects import explain_effect, global_importance, raw_feature_shap
from equipoise.plotting import NEUTRAL, SERIES
from equipoise.serving.assets import build_serving_assets, save_assets

STAGE = "explain"


def _importance_plot(imp: pd.DataFrame, top: int = 15) -> plt.Figure:
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(t) + 1.4))
    ax.barh(t.feature.str.replace("X_", ""), t.mean_abs_shap_letters, color=SERIES[0], height=0.6)
    ax.set_xlabel("Mean |SHAP| (letters of predicted aflibercept - bevacizumab effect)")
    ax.set_title("What drives the predicted aflibercept advantage? (held-out patients)")
    fig.tight_layout()
    return fig


def _beeswarm(expl: shap.Explanation) -> plt.Figure:
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(expl, max_display=12, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (letters added to the predicted aflibercept effect)")
    fig.suptitle("SHAP summary, aflibercept - bevacizumab effect (held-out patients)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def _waterfall(expl: shap.Explanation) -> plt.Figure:
    plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(expl, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle("Why this predicted effect? Typical training patient (illustrative)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def _subgroup_plot(rep: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rep) + 1.5))
    y = np.arange(len(rep))[::-1]
    for yi, (_, r) in zip(y, rep.iterrows(), strict=True):
        color = SERIES[0] if r.reliable else NEUTRAL
        if np.isfinite(r.observed_effect_afl):
            ax.errorbar(r.observed_effect_afl, yi, xerr=[[r.observed_effect_afl
                                                          - r.observed_ci_low],
                                                         [r.observed_ci_high
                                                          - r.observed_effect_afl]],
                        fmt="o", color=color, capsize=3, ms=5)
        ax.scatter(r.mean_predicted_effect_afl, yi, marker="D", color=SERIES[1], s=30, zorder=3)
    ax.axvline(0, color=NEUTRAL, lw=1)
    ax.set_yticks(y, [f"{r.subgroup}: {r.level} (n={r.n}){'' if r.reliable else ' *'}"
                      for _, r in rep.iterrows()], fontsize=8)
    ax.set_xlabel("Aflibercept - bevacizumab (letters): observed (dot, 95% CI) vs mean "
                  "predicted (diamond)")
    ax.set_title("Effect calibration by subgroup (* = too small to interpret, grey)")
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    train, test = split.train(df), split.test(df)
    bundle = serving_mode(load_bundle(default_bundle_path(cfg, synthetic)))
    names = [n.split("__", 1)[1] for n in bundle.preprocessor.get_feature_names_out()]
    rng = np.random.default_rng(cfg.seed)
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"explain": cfg.explain, "effect_model": bundle.effect_model_name})

        # SHAP --------------------------------------------------------------------------
        Xtr = bundle.preprocessor.transform(train)
        bg = Xtr[rng.choice(len(Xtr), min(int(cfg.explain["shap_background"]), len(Xtr)),
                            replace=False)]
        Xte = bundle.preprocessor.transform(test)
        expl = explain_effect(bundle.effect_model, bg, Xte, names, cfg)
        pred = bundle.effect_model.effect(Xte)[:, 0]
        add_err = float(np.max(np.abs(expl.values.sum(axis=1) + expl.base_values - pred)))
        raw = raw_feature_shap(expl.values, names, cfg)
        imp = global_importance(raw)
        tracking.log_table(cfg, STAGE, imp, "shap_importance_aflibercept")
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "explainer": "permutation (model-agnostic) on effect(X)", "background": len(bg),
            "explained_patients": len(Xte), "base_value_letters": float(np.mean(expl.base_values)),
            "additivity_holds_within_1e-6": add_err < 1e-6}]), "shap_setup")
        tracking.log_figure(cfg, STAGE, _importance_plot(imp), "shap_importance")
        tracking.log_figure(cfg, STAGE, _beeswarm(expl), "shap_beeswarm")
        typ = patient_frame(typical_patient(train))
        one = explain_effect(bundle.effect_model, bg, bundle.preprocessor.transform(typ), names,
                             cfg)
        tracking.log_figure(cfg, STAGE, _waterfall(one[0]), "shap_waterfall_typical")
        # serving assets: aggregate SHAP background + form defaults (no patient rows)
        models_dir = cfg.paths.models_dir / ("synthetic" if synthetic else "")
        save_assets(build_serving_assets(train, bundle, cfg, typical_patient(train)), models_dir)
        top = imp.feature.iloc[0]
        metrics.update({"shap_top_feature_is_vision": float(top in ("X_va_letters",
                                                                     "X_va_below_69")),
                        "shap_additivity_ok": float(add_err < 1e-6),
                        "shap_rank_va_letters": float(imp.set_index("feature").loc[
                            "X_va_letters", "rank"])})

        # subgroups ---------------------------------------------------------------------
        factory = _chosen_factory(bundle.effect_model_name, cfg,
                                  bundle.extras.get("tuned_gb_params", {}))
        preds = out_of_sample_predictions(train, test, bundle, factory, cfg)
        rep = subgroup_report(preds, df, cfg)
        tracking.log_table(cfg, STAGE, rep, "subgroups")
        tracking.log_figure(cfg, STAGE, _subgroup_plot(rep), "subgroups")
        metrics["n_unreliable_subgroups"] = int((~rep.reliable).sum())
        metrics["n_subgroups"] = len(rep)
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
