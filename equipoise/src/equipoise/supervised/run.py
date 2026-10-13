"""M5 stage: outcome models for vision gain, gain >= 15 letters, laser and injections.

Models are compared by cross-validation on the training split; the held-out test set is
scored once per model for final reporting and is never used to choose anything. Every model
is logged as a nested MLflow run.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.splits import get_split
from equipoise.plotting import INK, INK_2, NEUTRAL, SEQUENTIAL, SERIES
from equipoise.preprocess.discretize import gain_item
from equipoise.supervised.evaluate import (
    evaluate,
    irls_check,
    knn_algorithm_timing,
    normal_equations_check,
)
from equipoise.supervised.models import (
    class_weight_variants,
    classification_models,
    count_models,
    regression_models,
)

STAGE = "supervised"
GAIN_COL = "gain_ge_15"


def _log_models(table: pd.DataFrame, task: str) -> None:
    for _, r in table.iterrows():
        metrics = {k.replace(" ", "_").replace("-", "_"): v for k, v in r.items()
                   if k.startswith(("CV ", "Test ")) and pd.notna(v)}
        with tracking.child_run(f"{task}: {r.Model}", {"task": task, "model": r.Model}):
            tracking.log_metrics(metrics)


def _regression_bar(table: pd.DataFrame) -> plt.Figure:
    t = table.sort_values("CV RMSE", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(t))
    ax.barh(y, t["CV RMSE"], xerr=t["CV RMSE sd"], color=SERIES[0], height=0.55,
            error_kw={"ecolor": INK_2, "capsize": 3}, label="5-fold CV RMSE (± SD)")
    ax.scatter(t["Test RMSE"], y, color=SERIES[1], zorder=3, s=40, label="Held-out RMSE")
    base = t.loc[t.Model.str.startswith("Baseline"), "CV RMSE"]
    if len(base):
        ax.axvline(float(base.iloc[0]), color=NEUTRAL, ls="--", lw=1)
    ax.set_yticks(y, t.Model)
    ax.set_xlabel("RMSE (letters) - lower is better; dashed = arm-mean baseline")
    ax.set_title("Predicting letters gained at week 52")
    ax.legend(loc="lower right")
    lo = float(min(t["CV RMSE"].min(), t["Test RMSE"].min()))
    ax.set_xlim(lo * 0.85, None)
    fig.tight_layout()
    return fig


def _pred_vs_obs(preds: dict[str, np.ndarray], best: str) -> plt.Figure:
    y = preds["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, name in zip(axes, ["Baseline: arm mean", best], strict=True):
        ax.scatter(y, preds[name], s=14, color=SERIES[0], alpha=0.6, linewidths=0)
        lim = [min(y.min(), preds[name].min()) - 2, max(y.max(), preds[name].max()) + 2]
        ax.plot(lim, lim, color=NEUTRAL, lw=1, ls="--")
        ax.set_title(name)
        ax.set_xlabel("Observed letters gained (held-out)")
    axes[0].set_ylabel("Predicted letters gained")
    fig.suptitle("Held-out predictions vs observations (dashed = perfect prediction)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


BASELINE_CLS = "Baseline: arm event rate"


def _curves(results: dict[str, tuple[dict[str, np.ndarray], list[str]]]) -> plt.Figure:
    """ROC and PR on held-out patients for the top models (chosen by CV AUC) + baseline."""
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 4.2 * len(results)))
    axes = np.atleast_2d(axes)
    for row, (task, (preds, top)) in enumerate(results.items()):
        y = preds["y_test"]
        for name, color, ls in [(BASELINE_CLS, NEUTRAL, ":"),
                                *[(n, SERIES[i], "-") for i, n in enumerate(top)]]:
            fpr, tpr, _ = roc_curve(y, preds[name])
            prec, rec, _ = precision_recall_curve(y, preds[name])
            axes[row, 0].plot(fpr, tpr, color=color, ls=ls, lw=1.8, label=name)
            axes[row, 1].plot(rec, prec, color=color, ls=ls, lw=1.8, label=name)
        axes[row, 0].plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1)
        axes[row, 1].axhline(y.mean(), color=NEUTRAL, ls="--", lw=1)
        axes[row, 0].set(xlabel="False positive rate (1 - specificity)",
                         ylabel="True positive rate (recall)", title=f"{task}: ROC (held-out)")
        axes[row, 1].set(xlabel="Recall", ylabel="Precision",
                         title=f"{task}: precision-recall (dashed = prevalence)")
        axes[row, 0].legend(fontsize=8, loc="lower right", title="top 3 by CV AUC + baseline",
                            title_fontsize=8)
    fig.tight_layout()
    return fig


def _confusions(preds: dict[str, np.ndarray], names: list[str], threshold: float, task: str
                ) -> plt.Figure:
    n = len(names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names, strict=False):
        cm = confusion_matrix(preds["y_test"], (preds[name] >= threshold).astype(int),
                              labels=[0, 1])
        ax.imshow(cm, cmap=SEQUENTIAL)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(name, fontsize=8)
        ax.grid(False)
    fig.suptitle(f"Confusion matrices, {task} (held-out, threshold {threshold})",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def _template_table(table: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The comparison layout the report template asks for."""
    if kind == "classification":
        cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC", "PR-AUC"]
    elif kind == "count":
        cols = ["Poisson deviance", "MAE", "RMSE"]
    else:
        cols = ["MSE", "RMSE", "MAE", "R2"]
    out = table[["Model"]].copy()
    for c in cols:
        out[f"CV {c}"] = table[f"CV {c}"].map("{:.3f}".format) + " ± " + \
            table[f"CV {c} sd"].map("{:.3f}".format)
    for c in cols:
        out[f"Test {c}"] = table[f"Test {c}"]
    return out


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M5 and return headline metrics."""
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    df = df.assign(**{GAIN_COL: gain_item(df, cfg)})
    train, test = split.train(df), split.test(df)
    thr = float(cfg.supervised["classification_threshold"])
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"supervised": cfg.supervised, "n_train": len(train),
                             "n_test": len(test)})

        # --- regression ------------------------------------------------------------------
        reg, reg_preds, _ = evaluate(regression_models(cfg), train, test,
                                     cfg.primary_outcome, "regression", cfg)
        _log_models(reg, "regression")
        tracking.log_table(cfg, STAGE, reg, "regression_full")
        tracking.log_table(cfg, STAGE, _template_table(reg, "regression"), "regression_comparison")
        tracking.log_figure(cfg, STAGE, _regression_bar(reg), "regression_rmse")
        models_only = reg[~reg.Model.str.startswith("Baseline")]
        best = models_only.loc[models_only["CV RMSE"].idxmin(), "Model"]  # chosen by CV only
        tracking.log_figure(cfg, STAGE, _pred_vs_obs(reg_preds, best), "regression_pred_vs_obs")
        base = reg.set_index("Model").loc["Baseline: arm mean"]
        bestrow = reg.set_index("Model").loc[best]
        metrics.update({"reg_baseline_cv_rmse": base["CV RMSE"], "reg_best_cv_rmse":
                        bestrow["CV RMSE"], "reg_best_test_rmse": bestrow["Test RMSE"],
                        "reg_best_test_r2": bestrow["Test R2"],
                        "reg_baseline_test_rmse": base["Test RMSE"]})
        tracking.log_table(cfg, STAGE, pd.DataFrame([{"best_by_cv": best}]), "regression_best")

        checks = pd.DataFrame([
            {"check": "OLS normal equations vs sklearn LinearRegression",
             **normal_equations_check(train, cfg.primary_outcome, cfg)},
            {"check": "IRLS logistic vs statsmodels Logit (gain >= 15)",
             **irls_check(train, train[GAIN_COL], cfg)},
            {"check": "IRLS logistic vs statsmodels Logit (laser year 1)",
             **irls_check(train, train["Y_laser_yr1"], cfg)},
        ])
        tracking.log_table(cfg, STAGE, checks, "scratch_checks")
        tracking.log_table(cfg, STAGE, knn_algorithm_timing(train, test, cfg.primary_outcome,
                                                            cfg), "knn_algorithms")

        cc = train.dropna(subset=[cfg.primary_outcome])
        fitted_lasso = regression_models(cfg)["Lasso (CV alpha)"].fit(cc, cc[cfg.primary_outcome])
        names = fitted_lasso.named_steps["pre"].get_feature_names_out()
        coef = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names],
                             "coefficient": fitted_lasso.named_steps["model"].coef_})
        coef = coef[coef.coefficient != 0].reindex(
            coef.coefficient.abs().sort_values(ascending=False).index).dropna()
        tracking.log_table(cfg, STAGE, coef.assign(
            alpha=fitted_lasso.named_steps["model"].alpha_), "lasso_coefficients")

        # --- classification --------------------------------------------------------------
        curve_data, cls_names = {}, list(classification_models(cfg))
        for task, col in [("gain >= 15 letters", GAIN_COL), ("laser by year 1", "Y_laser_yr1")]:
            tab, preds, _ = evaluate(classification_models(cfg), train, test, col,
                                     "classification", cfg)
            slug = "gain15" if col == GAIN_COL else "laser_yr1"
            _log_models(tab, f"classification {slug}")
            tracking.log_table(cfg, STAGE, tab, f"classification_{slug}_full")
            tracking.log_table(cfg, STAGE, _template_table(tab, "classification"),
                               f"classification_{slug}")
            tracking.log_figure(cfg, STAGE, _confusions(preds, cls_names, thr, task),
                                f"confusion_{slug}")
            ranked = tab[~tab.Model.str.startswith("Baseline")].sort_values(
                "CV AUC", ascending=False)
            curve_data[task] = (preds, list(ranked.Model.head(3)))
            prev = float(train[col].dropna().mean())
            t = tab.set_index("Model")
            metrics[f"{slug}_prevalence_train"] = prev
            metrics[f"{slug}_lr_cv_auc"] = t.loc["Logistic regression", "CV AUC"]
            metrics[f"{slug}_lr_test_auc"] = t.loc["Logistic regression", "Test AUC"]

            cw, _, _ = evaluate(class_weight_variants(cfg), train, test, col,
                                "classification", cfg)
            keep = ["Model"] + [c for c in cw.columns if c.startswith("CV ")
                                and not c.endswith(" sd")]
            tracking.log_table(cfg, STAGE, cw[keep], f"class_weight_{slug}")
        tracking.log_figure(cfg, STAGE, _curves(curve_data), "roc_pr_curves")

        # --- count model for injections --------------------------------------------------
        cnt, cnt_preds, fitted = evaluate(count_models(cfg), train, test, "Y_n_inj_yr1",
                                          "count", cfg)
        _log_models(cnt, "count")
        tracking.log_table(cfg, STAGE, cnt, "count_full")
        tracking.log_table(cfg, STAGE, _template_table(cnt, "count"), "count_comparison")
        glm = fitted["Poisson GLM"]
        mu = glm.predict(train)
        y = train["Y_n_inj_yr1"].to_numpy(float)
        p = len(glm.named_steps["model"].coef_) + 1
        dispersion = float(np.sum((y - mu) ** 2 / mu) / (len(y) - p))
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "model": "Poisson GLM", "pearson_dispersion": dispersion,
            "interpretation": "< 1: less variable than Poisson (counts are bounded by the "
                              "visit schedule); > 1: overdispersed"}]), "count_dispersion")
        t = cnt.set_index("Model")
        metrics.update({"count_glm_test_mae": t.loc["Poisson GLM", "Test MAE"],
                        "count_baseline_test_mae": t.loc["Baseline: arm mean", "Test MAE"],
                        "count_glm_dispersion": dispersion})

        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
