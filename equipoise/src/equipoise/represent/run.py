"""M3 stage: PCA via SVD, clustering, per-cluster effects and association rules.

Everything here is exploratory and uses training patients only (the held-out test set stays
untouched). Features are the scaled baseline matrix from `build_preprocessor`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.splits import get_split
from equipoise.plotting import ARM_COLORS, INK_2, NEUTRAL, SEQUENTIAL, SERIES
from equipoise.preprocess.encode import build_preprocessor
from equipoise.represent.cluster import (
    cluster_profile,
    dbscan_labels,
    effects_by_cluster,
    kmeans_scan,
    ward_labels,
)
from equipoise.represent.pca import compare_with_sklearn, svd_pca
from equipoise.represent.rules import mine_rules

STAGE = "represent"


def _scree(pca, n: int, target: float) -> plt.Figure:
    k = min(n, len(pca.explained_variance_ratio))
    ratio = pca.explained_variance_ratio[:k] * 100
    cum = np.cumsum(pca.explained_variance_ratio) * 100
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(1, k + 1)
    ax.bar(x, ratio, color=SERIES[0], width=0.7, label="per component")
    ax.plot(x, cum[:k], color=SERIES[1], marker="o", ms=4, label="cumulative")
    ax.axhline(target * 100, color=NEUTRAL, lw=1, ls="--")
    reach = int(np.searchsorted(cum, target * 100) + 1)
    ax.text(k, target * 100 + 1.5, f"{target:.0%} reached at {reach} components", ha="right",
            fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot (SVD-PCA of scaled baseline features, training patients)")
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def _pca_maps(scores: np.ndarray, df: pd.DataFrame, pca, cfg: Config) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    pct = pca.explained_variance_ratio[:2] * 100
    for arm in cfg.arms:
        m = (df.treatment == arm).to_numpy()
        axes[0].scatter(scores[m, 0], scores[m, 1], s=12, alpha=0.6, color=ARM_COLORS[arm],
                        label=arm, linewidths=0)
    axes[0].legend(title="Arm")
    axes[0].set_title("Coloured by randomized arm")
    y = df[cfg.primary_outcome].to_numpy(float)
    ok = ~np.isnan(y)
    lim = np.nanpercentile(np.abs(y), 95)
    sc = axes[1].scatter(scores[ok, 0], scores[ok, 1], c=y[ok], cmap=SEQUENTIAL, s=12,
                         vmin=-lim / 2, vmax=lim, linewidths=0)
    axes[1].scatter(scores[~ok, 0], scores[~ok, 1], s=12, facecolors="none",
                    edgecolors=NEUTRAL, label="no week-52 vision")
    axes[1].legend(loc="lower right")
    fig.colorbar(sc, ax=axes[1], label="Letters gained at week 52")
    axes[1].set_title("Coloured by outcome")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pct[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({pct[1]:.1f}% of variance)")
    fig.suptitle("Patient map: first two principal components (training patients)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def _kscan_plot(scan: pd.DataFrame, k_best: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(scan.k, scan.inertia, marker="o", color=SERIES[0])
    axes[0].set_ylabel("Within-cluster sum of squares")
    axes[0].set_title("Elbow (k-means inertia)")
    axes[1].plot(scan.k, scan.silhouette, marker="o", color=SERIES[0])
    axes[1].axvline(k_best, color=SERIES[1], ls="--", lw=1.5)
    axes[1].text(k_best, scan.silhouette.max(), f" chosen k = {k_best}", color=INK_2,
                 fontsize=9, va="top")
    axes[1].set_ylabel("Mean silhouette (-1 to 1)")
    axes[1].set_title("Silhouette")
    for ax in axes:
        ax.set_xlabel("Number of clusters k")
        ax.set_xticks(scan.k)
    fig.tight_layout()
    return fig


def _dendrogram_plot(L: np.ndarray, truncate: int, k: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    set_link_color_palette(list(SERIES[:k]))
    cut = (L[-k, 2] + L[-k + 1, 2]) / 2 if k > 1 else L[-1, 2]
    dendrogram(L, truncate_mode="lastp", p=truncate, ax=ax, color_threshold=cut,
               above_threshold_color=NEUTRAL, leaf_font_size=8)
    ax.axhline(cut, color=SERIES[1], ls="--", lw=1.5)
    ax.text(ax.get_xlim()[1], cut, f"cut -> {k} clusters ", ha="right", va="bottom",
            fontsize=9, color=INK_2)
    ax.set_xlabel("Merged groups of patients (leaf labels = group size in brackets)")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Hierarchical clustering (Ward), training patients")
    ax.grid(False)
    fig.tight_layout()
    return fig


def _effects_plot(eff: pd.DataFrame, cfg: Config) -> plt.Figure:
    methods = list(eff.method.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), methods, strict=True):
        sub = eff[(eff.method == m) & eff.difference.notna()].reset_index(drop=True)
        labels, ys = [], []
        for i, (_, r) in enumerate(sub.iterrows()):
            arm = r.comparison.split(" - ")[0]
            y = len(sub) - i
            ax.errorbar(r.difference, y, xerr=[[r.difference - r.ci_low],
                                               [r.ci_high - r.difference]],
                        fmt="o", color=ARM_COLORS[arm], capsize=3, ms=6)
            labels.append(f"cluster {r.cluster} (n={r.n_cluster}): {arm[:3]}")
            ys.append(y)
        ax.axvline(0, color=NEUTRAL, lw=1)
        ax.set_yticks(ys, labels, fontsize=8)
        ax.set_title(m)
        ax.set_xlabel(f"Difference vs {cfg.reference_arm} (letters, 95% CI)")
    fig.suptitle("Treatment differences within clusters - EXPLORATORY, not pre-specified",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M3 and return headline metrics."""
    rp = cfg.represent
    df = load_analysis(cfg, synthetic=synthetic)
    train = get_split(cfg, df, synthetic).train(df)
    Z = build_preprocessor(cfg).fit_transform(train)
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"represent": rp, "n_train": len(train), "n_features": Z.shape[1]})

        pca = svd_pca(Z)
        check = compare_with_sklearn(Z, pca)
        tracking.log_table(cfg, STAGE, pd.DataFrame([check]), "svd_vs_sklearn")
        var = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance) + 1),
                            "variance": pca.explained_variance,
                            "pct_variance": 100 * pca.explained_variance_ratio,
                            "cumulative_pct": 100 * np.cumsum(pca.explained_variance_ratio)})
        tracking.log_table(cfg, STAGE, var, "pca_variance")
        names = build_preprocessor(cfg).fit(train).get_feature_names_out()
        loadings = pd.DataFrame(pca.components[:3].T, columns=["PC1", "PC2", "PC3"])
        loadings.insert(0, "feature", [n.split("__", 1)[1] for n in names])
        loadings = loadings.reindex(loadings.PC1.abs().sort_values(ascending=False).index)
        tracking.log_table(cfg, STAGE, loadings, "pca_loadings")
        tracking.log_figure(cfg, STAGE, _scree(pca, int(rp["n_components_plot"]),
                                               float(rp["variance_target"])), "scree")
        tracking.log_figure(cfg, STAGE, _pca_maps(pca.transform(Z, 2), train, pca, cfg),
                            "pca_map")
        target = float(rp["variance_target"])
        metrics["pca_components_for_target"] = int(
            np.searchsorted(np.cumsum(pca.explained_variance_ratio), target) + 1)
        metrics["pca_pc1_pct"] = 100 * float(pca.explained_variance_ratio[0])
        metrics["svd_sklearn_max_diff"] = max(check.values())

        scan = kmeans_scan(Z, tuple(rp["k_range"]), cfg.seed)
        k_best = int(scan.loc[scan.silhouette.idxmax(), "k"])
        tracking.log_table(cfg, STAGE, scan, "kmeans_scan")
        tracking.log_figure(cfg, STAGE, _kscan_plot(scan, k_best), "kmeans_scan")
        from sklearn.cluster import KMeans

        labels = {"k-means": KMeans(n_clusters=k_best, n_init=10,
                                    random_state=cfg.seed).fit_predict(Z)}
        L, labels["Ward"] = ward_labels(Z, k_best)
        tracking.log_figure(cfg, STAGE, _dendrogram_plot(L, int(rp["dendrogram_truncate"]),
                                                         k_best), "dendrogram")
        o = cfg.preprocess["outliers"]
        labels["DBSCAN"], eps, _ = dbscan_labels(Z, int(o["dbscan_min_samples"]),
                                                 float(o["dbscan_eps_quantile"]))
        eff = pd.concat([effects_by_cluster(train, lab, m, cfg) for m, lab in labels.items()])
        prof = pd.concat([cluster_profile(train, lab, m) for m, lab in labels.items()])
        tracking.log_table(cfg, STAGE, eff, "cluster_effects")
        tracking.log_table(cfg, STAGE, prof, "cluster_profiles")
        tracking.log_figure(cfg, STAGE, _effects_plot(eff, cfg), "cluster_effects")
        metrics.update({"k_best": k_best, "silhouette_best": float(scan.silhouette.max()),
                        "dbscan_clusters": int(len(set(labels["DBSCAN"]) - {-1})),
                        "dbscan_noise": int((labels["DBSCAN"] == -1).sum()),
                        "cluster_comparisons_p_below_0_05": int((eff.p < 0.05).sum()),
                        "cluster_comparisons": int(eff.p.notna().sum())})

        rules = mine_rules(train, cfg)
        tracking.log_table(cfg, STAGE, rules, "association_rules")
        metrics["n_rules_reported"] = len(rules)

        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
