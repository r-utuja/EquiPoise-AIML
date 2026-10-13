"""Fixed train/test split and arm-stratified cross-validation folds.

The held-out test set is 20% of patients, stratified by randomized arm, drawn once with the
configured seed and saved as patient IDs in `data/splits.json`. It is used only for final
reporting. All model selection uses `cv_folds`, a seeded `StratifiedKFold` over the
remaining 80%, stratified by arm so every fold keeps the trial's 1:1:1 balance.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from equipoise.config import Config


@dataclass(frozen=True)
class Split:
    """Patient IDs in the training and held-out test sets (sorted)."""

    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    seed: int
    test_fraction: float
    stratify_by: str

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["pt_id"].isin(self.train_ids)].reset_index(drop=True)

    def test(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["pt_id"].isin(self.test_ids)].reset_index(drop=True)


def make_split(
    df: pd.DataFrame, test_fraction: float, seed: int, stratify_by: str = "treatment"
) -> Split:
    """Draw the stratified train/test split by patient ID."""
    ids = df["pt_id"].to_numpy()
    train, test = train_test_split(
        ids, test_size=test_fraction, random_state=seed, stratify=df[stratify_by].to_numpy()
    )
    return Split(
        tuple(sorted(int(i) for i in train)),
        tuple(sorted(int(i) for i in test)),
        seed,
        test_fraction,
        stratify_by,
    )


def save_split(split: Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": split.seed,
        "test_fraction": split.test_fraction,
        "stratify_by": split.stratify_by,
        "train_ids": list(split.train_ids),
        "test_ids": list(split.test_ids),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_split(path: Path) -> Split:
    d = json.loads(path.read_text(encoding="utf-8"))
    return Split(
        tuple(d["train_ids"]), tuple(d["test_ids"]), d["seed"], d["test_fraction"],
        d["stratify_by"],
    )


def cv_folds(
    df: pd.DataFrame, n_folds: int, seed: int, stratify_by: str = "treatment"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_index, validation_index) positional arrays, stratified by arm."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(df)), df[stratify_by].to_numpy())


def splits_path(cfg: Config, synthetic: bool) -> Path:
    """data/splits.json for real data, data/synthetic/splits.json for synthetic data."""
    return cfg.paths.synthetic_dir / cfg.paths.splits_file.name if synthetic \
        else cfg.paths.splits_file


def create_split(cfg: Config, df: pd.DataFrame, synthetic: bool) -> Split:
    """Draw the split from config (seed, test fraction, stratum) and save it."""
    split = make_split(df, cfg.test_fraction, cfg.seed, cfg.stratify_by)
    save_split(split, splits_path(cfg, synthetic))
    return split


def get_split(cfg: Config, df: pd.DataFrame, synthetic: bool) -> Split:
    """Load the saved split and check it still matches the data and the config.

    Fails if the file is missing (run `python -m equipoise split`), if config changed
    (seed / test fraction / stratum) or if its patient IDs differ from `df`.
    """
    path = splits_path(cfg, synthetic)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run `python -m equipoise split"
            f"{' --synthetic' if synthetic else ''}`."
        )
    split = load_split(path)
    expected = (cfg.seed, cfg.test_fraction, cfg.stratify_by)
    if (split.seed, split.test_fraction, split.stratify_by) != expected:
        raise ValueError(f"{path} was made with different settings than config; re-run split.")
    if set(split.train_ids) | set(split.test_ids) != set(df["pt_id"]):
        raise ValueError(f"{path} patient IDs do not match the data; re-run split.")
    return split
