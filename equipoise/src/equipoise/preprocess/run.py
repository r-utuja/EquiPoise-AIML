"""M1 stage: missing data, CST harmonization, outliers, encoding and discretization.

Trial-level estimates (the arm differences under each missing-data strategy) use all 660
randomized patients, as the trial's own analysis did. Everything that learns parameters for
later models (imputer scoring, harmonizer, outlier detectors, preprocessor) is fitted on the
training split only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis, load_cst_quality
from equipoise.data.splits import get_split
from equipoise.plotting import ARM_COLORS, INK, INK_2, NEUTRAL, SERIES
from equipoise.preprocess.columns import CST, OCT_MACHINE, feature_groups
from equipoise.preprocess.cst_quality import apply_cst_quality
from equipoise.preprocess.discretize import discretize
from equipoise.preprocess.encode import build_preprocessor, one_hot_sparse
from equipoise.preprocess.harmonize import harmonizer_from_config
from equipoise.preprocess.missing import (
    compare_strategies,
    missingness_table,
    score_feature_imputers,
)
from equipoise.preprocess.outliers import METHODS, detect_outliers, overlap_table

STAGE = "preprocess"


def _forest_plot(est: pd.DataFrame, cfg: Config) -> plt.Figure:
    comps = list(est["comparison"].unique())
    fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 3.2), sharex=True)
    for ax, comp in zip(np.atleast_1d(axes), comps, strict=True):
        sub = est[est.comparison == comp].reset_index(drop=True)
        arm = comp.split(" - ")[0]
        y = np.arange(len(sub))[::-1]
        ax.errorbar(sub.estimate, y, xerr=[sub.estimate - sub.ci_low, sub.ci_high - sub.estimate],
                    fmt="o", color=ARM_COLORS[arm], ecolor=ARM_COLORS[arm], capsize=4, ms=8)
        for yi, (_, r) in zip(y, sub.iterrows(), strict=True):
            ax.annotate(f"{r.estimate:.1f}", (r.estimate, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=INK_2)
        ax.axvline(0, color=NEUTRAL, lw=1)
        ax.set_ylim(-0.6, len(sub) - 0.3)
        ax.set_yticks(y, sub.strategy)
        ax.set_title(comp)
        level = f"{cfg.stats['ci_level']:.0%}"
        ax.set_xlabel(f"Difference in {cfg.primary_outcome} (letters, {level} CI)")
    fig.suptitle("Arm differences under three missing-data strategies", fontweight="bold")
    fig.tight_layout()
    return fig


def _imputer_plot(scores: pd.DataFrame) -> plt.Figure:
    piv = scores.pivot(index="feature", columns="imputer", values="nrmse")
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(piv.columns)
    x = np.arange(len(piv))
    for i, col in enumerate(piv.columns):
        ax.bar(x + i * width, piv[col], width * 0.92, label=col, color=SERIES[i])
    ax.axhline(1, color=NEUTRAL, lw=1, ls="--")
    ax.text(len(piv) - 0.5, 1.02, "= guessing the mean", ha="right", fontsize=8, color=INK_2)
    ax.set_xticks(x + width * (len(piv.columns) - 1) / 2,
                  [c.replace("X_", "") for c in piv.index], rotation=35, ha="right")
    ax.set_ylabel("Normalized RMSE (RMSE / SD)")
    ax.set_title("Recovering hidden baseline values (training patients)")
    ax.legend(title="Imputer")
    fig.tight_layout()
    return fig


def _cst_plot(train: pd.DataFrame, df: pd.DataFrame, cfg: Config
              ) -> tuple[plt.Figure, pd.DataFrame]:
    """CST by machine: raw, published median-offset, within-machine z-score."""
    cst_cfg = cfg.preprocess["cst"]
    machines = [m for m in cst_cfg["offset_um"] if (df[OCT_MACHINE] == m).any()]
    versions = {"Raw (machine units, um)": df[CST].to_numpy(float)}
    for method, label in (("offset", "Stratus-equivalent, median offset (um)"),
                          ("zscore", "Within-machine z-score (SD units)")):
        h = harmonizer_from_config(cst_cfg, method).fit(train[[CST, OCT_MACHINE]])
        versions[label] = h.transform(df[[CST, OCT_MACHINE]])[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows = []
    for ax, (label, values) in zip(axes, versions.items(), strict=True):
        data = [values[(df[OCT_MACHINE] == m).to_numpy() & ~np.isnan(values)] for m in machines]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops={"color": INK, "linewidth": 1.5},
                        flierprops={"markersize": 3, "markerfacecolor": NEUTRAL,
                                    "markeredgecolor": NEUTRAL})
        for patch, color in zip(bp["boxes"], SERIES[3:], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ticks = [f"{m.split()[-1]}\n(n={len(d)})" for m, d in zip(machines, data, strict=True)]
        ax.set_xticks(range(1, len(machines) + 1), ticks)
        ax.set_title(label, fontsize=10)
        for m, d in zip(machines, data, strict=True):
            rows.append({"version": label, "machine": m, "n": len(d), "mean": d.mean(),
                         "sd": d.std(ddof=1), "median": np.median(d)})
    axes[0].set_ylabel("Baseline central subfield thickness")
    fig.suptitle("Baseline CST by OCT machine, before and after harmonization (fit on training "
                 "patients)", fontweight="bold")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def _kdist_plot(k_dist: np.ndarray, eps: float, k: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(len(k_dist)), k_dist, color=SERIES[0])
    ax.axhline(eps, color=SERIES[1], lw=1.5, ls="--")
    ax.text(5, eps, f" eps = {eps:.2f}", va="bottom", color=INK_2, fontsize=9)
    ax.set_xlabel("Training patients, sorted by distance")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour\n(standardized feature space)")
    ax.set_title("DBSCAN k-distance curve")
    fig.tight_layout()
    return fig


def _overlap_plot(overlap: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(overlap))[::-1]
    ax.barh(y, overlap.n_flagged, color=SERIES[0], height=0.6)
    for yi, n in zip(y, overlap.n_flagged, strict=True):
        ax.text(n, yi, f" {n}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, overlap.methods.str.replace("_", " "))
    ax.set_xlabel("Patients flagged (all 660, detectors fit on training split)")
    ax.set_title("Outlier flags by method and their overlap")
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M1 and return the headline metrics."""
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    train = split.train(df)
    quality = load_cst_quality(cfg, df, synthetic)
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"preprocess": cfg.preprocess, "n_train": len(train),
                             "n_test": len(split.test_ids)})

        # --- missing data ------------------------------------------------------------
        tracking.log_table(cfg, STAGE, missingness_table(df, cfg), "missingness")
        est = compare_strategies(df, cfg)
        tracking.log_table(cfg, STAGE, est, "missing_strategies")
        tracking.log_figure(cfg, STAGE, _forest_plot(est, cfg), "missing_strategies")
        for _, r in est.iterrows():
            key = f"{r.strategy.split(' ')[0].lower()}_{r.comparison.split(' ')[0].lower()}"
            metrics[f"diff_{key}"] = r.estimate

        scores = score_feature_imputers(train, cfg)
        summary = scores.groupby("imputer", as_index=False)["nrmse"].mean()
        tracking.log_table(cfg, STAGE, scores, "imputer_scores")
        tracking.log_table(cfg, STAGE, summary, "imputer_scores_mean")
        tracking.log_figure(cfg, STAGE, _imputer_plot(scores), "imputer_scores")
        for _, r in summary.iterrows():
            metrics[f"nrmse_{r.imputer.split(' ')[0].lower()}"] = r.nrmse

        # --- CST harmonization and quality -------------------------------------------
        fig, cst_table = _cst_plot(train, df, cfg)
        tracking.log_figure(cfg, STAGE, fig, "cst_by_machine")
        tracking.log_table(cfg, STAGE, cst_table, "cst_harmonization")
        q = quality.drop(columns="pt_id")
        sens = apply_cst_quality(df, quality, "missing")
        qual = pd.DataFrame({
            "item": [*q.columns, "X_cst_um available (keep)", "X_cst_um available (missing)",
                     "Y_cst_change_wk52 available (keep)",
                     "Y_cst_change_wk52 available (missing)"],
            "n_patients": [*q.sum().astype(int), int(df[CST].notna().sum()),
                           int(sens[CST].notna().sum()), int(df.Y_cst_change_wk52.notna().sum()),
                           int(sens.Y_cst_change_wk52.notna().sum())],
        })
        tracking.log_table(cfg, STAGE, qual, "cst_quality")

        # --- outliers ----------------------------------------------------------------
        res = detect_outliers(train, df, cfg)
        overlap = overlap_table(res.flags)
        tracking.log_table(cfg, STAGE, overlap, "outlier_overlap")
        tracking.log_figure(cfg, STAGE, _overlap_plot(overlap), "outlier_overlap")
        tracking.log_figure(cfg, STAGE, _kdist_plot(
            res.k_distances, res.eps, int(cfg.preprocess["outliers"]["dbscan_min_samples"])),
            "dbscan_kdistance")
        cfg.paths.derived_dir.mkdir(parents=True, exist_ok=True)
        # patient-level flags belong in data/ (never committed), not in reports/
        flags_path = cfg.paths.derived_dir / "m1_outlier_flags.csv"
        res.flags.to_csv(flags_path, index=False, lineterminator="\n")
        for m in METHODS:
            metrics[f"outliers_{m}"] = int(res.flags[m].sum())
        metrics["dbscan_eps"] = res.eps

        # --- encoding, sparse one-hot, discretization --------------------------------
        pre = build_preprocessor(cfg).fit(train)
        g = feature_groups(cfg)
        names = list(pre.get_feature_names_out())
        group_of = {n: n.split("__")[0] for n in names}
        enc = pd.DataFrame({"output_feature": names, "block": [group_of[n] for n in names]})
        tracking.log_table(cfg, STAGE, enc, "preprocessor_features")
        X_sparse, oh_names = one_hot_sparse(train, cfg)
        dense_bytes = X_sparse.shape[0] * X_sparse.shape[1] * 8
        sparse_bytes = X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "nominal_features": len(g.nominal), "one_hot_columns": len(oh_names),
            "rows": X_sparse.shape[0], "non_zeros": X_sparse.nnz,
            "density": X_sparse.nnz / np.prod(X_sparse.shape),
            "dense_bytes": dense_bytes, "csr_bytes": sparse_bytes}]), "sparse_one_hot")
        bands = discretize(df, cfg)
        band_counts = pd.concat([
            bands[c].value_counts(sort=False).rename_axis("band").reset_index(name="n")
            .assign(feature=c.replace("_band", "")) for c in bands.columns
        ])[["feature", "band", "n"]]
        tracking.log_table(cfg, STAGE, band_counts, "bands")
        metrics["n_output_features"] = len(names)

        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
