"""M2 stage: Table 1, trial replication, hypothesis tests and descriptive plots.

This is trial-level inference, so it uses all 660 randomized patients (completers at each
horizon), exactly like the published analysis; nothing here is fed to a model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis, load_cst_quality
from equipoise.plotting import ARM_COLORS, DIVERGING, INK, INK_2, NEUTRAL
from equipoise.preprocess.columns import CST, OCT_MACHINE
from equipoise.preprocess.cst_quality import apply_cst_quality
from equipoise.preprocess.harmonize import harmonizer_from_config
from equipoise.stats.inference import (
    interaction_regression,
    multiple_correlation,
    partial_correlation,
    variance_tests,
)
from equipoise.stats.replication import burden_table, hypothesis_tests, means_table, versus_targets
from equipoise.stats.table1 import n_significant, table1

STAGE = "stats"


def _jitter(n: int, rng: np.random.Generator, width: float = 0.12) -> np.ndarray:
    return rng.uniform(-width, width, n)


def _outcome_by_arm(df: pd.DataFrame, cfg: Config) -> plt.Figure:
    rng = np.random.default_rng(cfg.seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, [("Y_va_change_wk52", "Week 52"),
                                       ("Y_va_change_wk104", "Week 104")], strict=True):
        for i, arm in enumerate(cfg.arms):
            y = df.loc[df.treatment == arm, col].dropna()
            ax.scatter(i + _jitter(len(y), rng), y, s=9, color=ARM_COLORS[arm], alpha=0.35,
                       linewidths=0)
            ax.boxplot([y], positions=[i], widths=0.45, showfliers=False,
                       medianprops={"color": INK, "linewidth": 2},
                       boxprops={"color": INK_2}, whiskerprops={"color": INK_2},
                       capprops={"color": INK_2})
        ax.axhline(0, color=NEUTRAL, lw=1)
        ticks = []
        for arm in cfg.arms:
            y = df.loc[df.treatment == arm, col].dropna()
            ticks.append(f"{arm}\nmean {y.mean():.1f}, n={len(y)}")
        ax.set_xticks(range(len(cfg.arms)), ticks)
        ax.set_title(title)
    axes[0].set_ylabel("Change in visual acuity (ETDRS letters)")
    fig.suptitle("Letters gained by arm (completers at each visit)", fontweight="bold")
    fig.tight_layout()
    return fig


def _means_plot(means: pd.DataFrame, cfg: Config) -> plt.Figure:
    m = means[means.outcome == "letters"]
    labels = {"all": "All patients", "va_below_69": "Baseline 20/50 or worse",
              "va_69_or_better": "Baseline 20/32-20/40"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, h in zip(axes, (1, 2), strict=True):
        sub = m[m.horizon_years == h]
        for j, arm in enumerate(cfg.arms):
            s = sub[sub.arm == arm].set_index("subgroup").loc[list(labels)]
            x = np.arange(len(labels)) + (j - 1) * 0.22
            ax.errorbar(x, s["mean"], yerr=[s["mean"] - s.ci_low, s.ci_high - s["mean"]],
                        fmt="o", ms=7, capsize=3, color=ARM_COLORS[arm], label=arm)
        ax.set_xticks(range(len(labels)), list(labels.values()), fontsize=9)
        ax.set_title(f"Year {h}")
        ax.axhline(0, color=NEUTRAL, lw=1)
    axes[0].set_ylabel(f"Mean letters gained ({cfg.stats['ci_level']:.0%} CI)")
    axes[1].legend(title="Arm", loc="upper right")
    fig.suptitle("Replication: mean change in vision by arm and baseline-vision subgroup",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def _va_vs_gain(df: pd.DataFrame, cfg: Config) -> plt.Figure:
    d = df.dropna(subset=[cfg.primary_outcome])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    grid = np.linspace(d.X_va_letters.min(), d.X_va_letters.max(), 50)
    for arm in cfg.arms:
        s = d[d.treatment == arm]
        ax.scatter(s.X_va_letters, s[cfg.primary_outcome], s=10, alpha=0.3,
                   color=ARM_COLORS[arm], linewidths=0)
        b = np.polyfit(s.X_va_letters, s[cfg.primary_outcome], 1)
        ax.plot(grid, np.polyval(b, grid), color=ARM_COLORS[arm],
                label=f"{arm} (slope {b[0]:.2f})")
    ax.axvline(69, color=NEUTRAL, lw=1, ls="--")
    ax.text(69.3, ax.get_ylim()[1] * 0.92, "69 letters\n(20/50 cut)", fontsize=8, color=INK_2)
    ax.set_xlabel("Baseline visual acuity (ETDRS letters)")
    ax.set_ylabel("Letters gained at week 52")
    ax.set_title("Baseline vision vs letters gained, with a least-squares line per arm")
    ax.legend(title="Arm (least-squares line)")
    fig.tight_layout()
    return fig


def _corr_heatmap(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    cols = [c for c in df.columns if c.startswith("X_") and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() > 2] + ["Y_va_change_wk52", "Y_cst_change_wk52", "Y_n_inj_yr1"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    short = [c.replace("X_", "").replace("Y_", "Y: ") for c in cols]
    ax.set_xticks(range(len(cols)), short, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), short, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(corr.iat[i, j]) >= 0.3:
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.iat[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Correlation of numeric baseline features and outcomes\n(|r| >= 0.3 labelled)")
    fig.tight_layout()
    return fig, corr.reset_index(names="feature")


def _baseline_dists(df: pd.DataFrame, cfg: Config) -> plt.Figure:
    feats = [("X_va_letters", "Baseline vision (letters)"), ("X_cst_um", "Baseline CST (um)"),
             ("X_hba1c", "HbA1c (%)"), ("X_age", "Age (years)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (col, label) in zip(axes, feats, strict=True):
        bins = np.histogram_bin_edges(df[col].dropna(), bins=20)
        for arm in cfg.arms:
            ax.hist(df.loc[df.treatment == arm, col].dropna(), bins=bins, histtype="step",
                    lw=2, color=ARM_COLORS[arm], label=arm)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Patients")
    axes[-1].legend(title="Arm")
    fig.suptitle("Baseline distributions by arm (randomization check)", fontweight="bold")
    fig.tight_layout()
    return fig


def _injections_plot(df: pd.DataFrame, cfg: Config) -> plt.Figure:
    d = df[df.M_has_wk52 == 1]
    counts = np.arange(d.Y_n_inj_yr1.min(), d.Y_n_inj_yr1.max() + 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.27
    for j, arm in enumerate(cfg.arms):
        s = d.loc[d.treatment == arm, "Y_n_inj_yr1"]
        pct = s.value_counts(normalize=True).reindex(counts, fill_value=0) * 100
        ax.bar(counts + (j - 1) * w, pct, w * 0.92, color=ARM_COLORS[arm],
               label=f"{arm} (median {s.median():.0f})")
    ax.set_xticks(counts)
    ax.set_xlabel("Study-eye injections before the 1-year visit")
    ax.set_ylabel("% of patients in arm")
    ax.set_title("Injection burden in year 1 (week-52 completers)")
    ax.legend(title="Arm")
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M2 and return the headline metrics."""
    df = load_analysis(cfg, synthetic=synthetic)
    quality = load_cst_quality(cfg, df, synthetic)
    metrics: dict[str, float] = {}
    alpha = float(cfg.stats["alpha"])

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"stats": cfg.stats, "n_patients": len(df)})

        t1 = table1(df, cfg)
        tracking.log_table(cfg, STAGE, t1, "table1")
        n_sig, n_tested = n_significant(t1, alpha)
        metrics.update({"table1_n_p_below_alpha": n_sig, "table1_n_tested": n_tested})

        means = means_table(df, cfg)
        burden = burden_table(df, cfg)
        tests = hypothesis_tests(df, cfg)
        tracking.log_table(cfg, STAGE, means, "replication_means")
        tracking.log_table(cfg, STAGE, burden, "replication_burden")
        tracking.log_table(cfg, STAGE, tests, "replication_tests")
        if not synthetic:  # published numbers describe the real trial only
            tracking.log_table(cfg, STAGE, versus_targets(means, burden, cfg),
                               "replication_vs_published")

        sens = means_table(apply_cst_quality(df, quality, "missing"), cfg)
        cst = pd.concat([means[means.outcome == "cst_um"].assign(policy="keep (primary)"),
                         sens[sens.outcome == "cst_um"].assign(policy="missing (sensitivity)")])
        tracking.log_table(cfg, STAGE, cst, "cst_change_sensitivity")

        y1 = means[(means.horizon_years == 1) & (means.subgroup == "all")
                   & (means.outcome == "letters")].set_index("arm")["mean"]
        for arm in cfg.arms:
            metrics[f"mean_gain_yr1_{arm.lower()}"] = y1[arm]
        anova1 = tests[(tests.horizon_years == 1) & (tests.subgroup == "all")
                       & (tests.outcome == "letters") & (tests.test == "one-way ANOVA")]
        metrics["anova_yr1_p"] = float(anova1.p.iloc[0])

        tracking.log_table(cfg, STAGE, variance_tests(df, cfg), "variance_tests")

        h = harmonizer_from_config(cfg.preprocess["cst"], "zscore").fit(df[[CST, OCT_MACHINE]])
        dz = df.assign(cst_z=h.transform(df[[CST, OCT_MACHINE]])[:, 0])
        pc = partial_correlation(dz, "X_va_letters", cfg.primary_outcome, ["cst_z"])
        mc = multiple_correlation(df, cfg.primary_outcome,
                                  list(cfg.stats["multiple_correlation_features"]))
        tracking.log_table(cfg, STAGE, pd.DataFrame([
            {"analysis": "partial correlation: gain ~ baseline VA | CST (within-machine z)",
             **pc},
            {"analysis": "multiple correlation: gain ~ baseline features", **mc},
        ]), "correlations")
        metrics.update({"partial_r_va_gain": pc["partial_r"], "multiple_R": mc["R"]})

        coef_rows, int_rows = [], []
        for modifier in ("X_va_letters", "X_va_below_69"):
            coefs, test = interaction_regression(df, cfg, modifier)
            coef_rows.append(coefs)
            int_rows.append({"modifier": modifier, **test})
            metrics[f"p_interaction_{modifier.replace('X_', '')}"] = test["p_interaction"]
        tracking.log_table(cfg, STAGE, pd.concat(coef_rows), "interaction_regression")
        tracking.log_table(cfg, STAGE, pd.DataFrame(int_rows), "interaction_tests")

        tracking.log_figure(cfg, STAGE, _outcome_by_arm(df, cfg), "outcome_by_arm")
        tracking.log_figure(cfg, STAGE, _means_plot(means, cfg), "replication_means")
        tracking.log_figure(cfg, STAGE, _va_vs_gain(df, cfg), "baseline_va_vs_gain")
        fig, corr = _corr_heatmap(df)
        tracking.log_figure(cfg, STAGE, fig, "correlation_heatmap")
        tracking.log_table(cfg, STAGE, corr, "correlation_matrix")
        tracking.log_figure(cfg, STAGE, _baseline_dists(df, cfg), "baseline_distributions")
        tracking.log_figure(cfg, STAGE, _injections_plot(df, cfg), "injections_yr1")

        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
