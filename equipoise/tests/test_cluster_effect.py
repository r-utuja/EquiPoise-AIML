"""Unsupervised cluster-then-look-up model: X-only clustering, the stage, interval sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equipoise.causal.cluster_effect import ClusterEffectModel
from equipoise.causal.cluster_run import PER_PATIENT, run
from equipoise.causal.data import prepare
from equipoise.config import Config
from equipoise.data.splits import get_split


def _data(synthetic: pd.DataFrame, cfg: Config):
    split = get_split(cfg, synthetic, synthetic=True)
    return prepare(split.train(synthetic), split.test(synthetic), cfg)


def test_clustering_uses_only_x_columns(synthetic: pd.DataFrame, cfg: Config) -> None:
    tr, _, pre = _data(synthetic, cfg)
    raw = {n.split("__", 1)[1].split("_", 1)[0] for n in pre.get_feature_names_out()}
    assert raw == {"X"}                                   # every model input is an X_ column
    a = ClusterEffectModel(cfg).fit(tr.X, tr.T, tr.Y)
    rng = np.random.default_rng(0)
    b = ClusterEffectModel(cfg).fit(tr.X, rng.permutation(tr.T), rng.permutation(tr.Y))
    assert (a.labels_ == b.labels_).all()                 # outcome and treatment don't move it


def test_intervals_contain_point_estimates(synthetic: pd.DataFrame, cfg: Config) -> None:
    tr, te, _ = _data(synthetic, cfg)
    m = ClusterEffectModel(cfg).fit(tr.X, tr.T, tr.Y)
    eff, (lo, hi) = m.predict_effect(te.X), m.effect_interval(te.X)
    assert eff.shape == lo.shape == hi.shape == (len(te), len(cfg.comparison_arms))
    assert (lo <= eff).all() and (eff <= hi).all()
    assert ((m.p_benefit(te.X, cfg.mcid_letters) >= 0) & (m.p_benefit(te.X, 5) <= 1)).all()


def test_stage_runs_on_synthetic(light: Config) -> None:
    metrics = run(light, synthetic=True)
    assert {"cluster_test_rmse", "cluster_test_r2", "verdict_agreement",
            "policy_cluster_verdict_ipw", "pehe_cluster_worst_case"} <= metrics.keys()
    per = pd.read_csv(light.paths.derived_dir / PER_PATIENT)
    for arm in light.comparison_arms:
        assert (per[f"low_{arm}"] <= per[f"effect_{arm}"]).all()
        assert (per[f"effect_{arm}"] <= per[f"high_{arm}"]).all()
    assert (light.paths.tables_dir / "m6_unsupervised_comparison.csv").is_file()
