"""M3: SVD-PCA matches sklearn, clustering helpers, association rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.preprocess.encode import build_preprocessor
from equipoise.represent.cluster import effects_by_cluster, kmeans_scan, ward_labels
from equipoise.represent.pca import compare_with_sklearn, svd_pca
from equipoise.represent.rules import GAIN_ITEM, baskets, mine_rules


def test_svd_pca_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) @ rng.normal(size=(6, 6))
    pca = svd_pca(X)
    diffs = compare_with_sklearn(X, pca)
    assert max(diffs.values()) < 1e-10
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(6), atol=1e-10)
    assert abs(pca.explained_variance_ratio.sum() - 1) < 1e-12
    assert np.all(np.diff(pca.explained_variance) <= 1e-12)


def test_clustering_and_effects(synthetic: pd.DataFrame, cfg: Config) -> None:
    train = get_split(cfg, synthetic, synthetic=True).train(synthetic)
    Z = build_preprocessor(cfg).fit_transform(train)
    scan = kmeans_scan(Z, (2, 4), cfg.seed)
    assert list(scan.k) == [2, 3, 4] and scan.inertia.is_monotonic_decreasing
    _, labels = ward_labels(Z, 3)
    assert set(labels) == {0, 1, 2}
    eff = effects_by_cluster(train, labels, "Ward", cfg)
    assert len(eff) == 3 * len(cfg.comparison_arms)
    assert (eff.note == "exploratory; not pre-specified").all()


def test_baskets_and_rules(synthetic: pd.DataFrame, cfg: Config) -> None:
    b = baskets(synthetic, cfg)
    assert len(b) == synthetic[cfg.primary_outcome].notna().sum()
    assert b.dtypes.eq(bool).all() and GAIN_ITEM in b.columns
    assert b.filter(like="va_letters=").sum(axis=1).eq(1).all()  # one vision band each
    rules = mine_rules(synthetic, cfg)
    assert not rules.empty and set(rules.arm) <= set(cfg.arms)
    assert (rules.lift > 0).all() and rules.confidence.between(0, 1).all()
