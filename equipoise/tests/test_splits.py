"""Train/test split and CV folds are reproducible and stratified by arm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equipoise.config import Config
from equipoise.data.splits import cv_folds, load_split, make_split, save_split


def test_same_seed_same_split(synthetic: pd.DataFrame, cfg: Config) -> None:
    a = make_split(synthetic, cfg.test_fraction, cfg.seed)
    b = make_split(synthetic.sample(frac=1, random_state=0).sort_values("pt_id"),
                   cfg.test_fraction, cfg.seed)
    assert a == b


def test_different_seed_different_split(synthetic: pd.DataFrame, cfg: Config) -> None:
    a = make_split(synthetic, cfg.test_fraction, cfg.seed)
    b = make_split(synthetic, cfg.test_fraction, cfg.seed + 1)
    assert a.test_ids != b.test_ids


def test_split_is_disjoint_complete_and_sized(synthetic: pd.DataFrame, cfg: Config) -> None:
    s = make_split(synthetic, cfg.test_fraction, cfg.seed)
    assert set(s.train_ids).isdisjoint(s.test_ids)
    assert set(s.train_ids) | set(s.test_ids) == set(synthetic.pt_id)
    assert len(s.test_ids) == round(cfg.test_fraction * len(synthetic))


def test_split_stratified_by_arm(synthetic: pd.DataFrame, cfg: Config) -> None:
    s = make_split(synthetic, cfg.test_fraction, cfg.seed)
    full = synthetic.treatment.value_counts()
    test = s.test(synthetic).treatment.value_counts()
    for arm, n in full.items():
        assert abs(test[arm] - cfg.test_fraction * n) <= 1


def test_split_roundtrip(synthetic: pd.DataFrame, cfg: Config, tmp_path: Path) -> None:
    s = make_split(synthetic, cfg.test_fraction, cfg.seed)
    save_split(s, tmp_path / "splits.json")
    assert load_split(tmp_path / "splits.json") == s


def test_cv_folds_reproducible_and_stratified(synthetic: pd.DataFrame, cfg: Config) -> None:
    train = make_split(synthetic, cfg.test_fraction, cfg.seed).train(synthetic)
    f1 = list(cv_folds(train, cfg.n_folds, cfg.seed))
    f2 = list(cv_folds(train, cfg.n_folds, cfg.seed))
    assert len(f1) == cfg.n_folds
    for (a_tr, a_va), (b_tr, b_va) in zip(f1, f2, strict=True):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)
    all_val = np.concatenate([va for _, va in f1])
    assert sorted(all_val) == list(range(len(train)))
    share = train.treatment.value_counts(normalize=True)
    for _, va in f1:
        fold_share = train.iloc[va].treatment.value_counts(normalize=True)
        assert (fold_share - share).abs().max() < 0.03


def test_committed_synthetic_split_matches_config(synthetic: pd.DataFrame, cfg: Config) -> None:
    from equipoise.data.splits import get_split

    saved = get_split(cfg, synthetic, synthetic=True)
    assert saved == make_split(synthetic, cfg.test_fraction, cfg.seed)


def test_get_split_rejects_stale_or_missing(synthetic: pd.DataFrame, cfg: Config,
                                            tmp_path: Path) -> None:
    from dataclasses import replace

    import pytest

    from equipoise.data.splits import create_split, get_split

    tmp_cfg = replace(cfg, paths=replace(cfg.paths, synthetic_dir=tmp_path))
    with pytest.raises(FileNotFoundError, match="equipoise split --synthetic"):
        get_split(tmp_cfg, synthetic, synthetic=True)
    create_split(tmp_cfg, synthetic, synthetic=True)
    assert get_split(tmp_cfg, synthetic, synthetic=True).seed == cfg.seed
    with pytest.raises(ValueError, match="different settings"):
        get_split(replace(tmp_cfg, seed=cfg.seed + 1), synthetic, synthetic=True)
    with pytest.raises(ValueError, match="patient IDs"):
        get_split(tmp_cfg, synthetic.iloc[:-1], synthetic=True)
