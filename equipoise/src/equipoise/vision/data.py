"""OCT dataset handling: manifest, patient-grouped validation split, preprocessed folders.

Each Kermany file name carries a patient ID (CLASS-patient-n.jpeg) and one patient can have
many scans. If scans of one patient landed in both training and validation, validation
accuracy would be inflated, so the validation split is drawn by patient
(GroupShuffleSplit). The official test split is kept as the held-out test set; the download
script already excludes its patients from the training subset, because the official Kermany
train/test split shares patients. Preprocessed
images are written as `<split>/<class>/<file>.png`, the folder layout both torchvision's
ImageFolder and Ultralytics' classification trainer read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from equipoise.config import Config
from equipoise.vision.preprocess import preprocess, read_gray


def raw_dir(cfg: Config) -> Path:
    return cfg.paths.root / cfg.raw["vision"]["raw_dir"]


def prepared_dir(cfg: Config) -> Path:
    return cfg.paths.derived_dir / "oct"


def load_manifest(cfg: Config) -> pd.DataFrame:
    path = raw_dir(cfg) / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `python scripts/download_oct_subset.py`")
    df = pd.read_csv(path, dtype={"patient_id": str})
    df["abs_path"] = [raw_dir(cfg) / p for p in df.path]
    return df


def assign_splits(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """train/val by patient inside the official train split; official test stays test."""
    out = df.copy()
    tr = out.index[out.split == "train"]
    groups = out.loc[tr, "patient_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    _, va = next(gss.split(tr, groups=groups))
    out["model_split"] = np.where(out.split == "test", "test", "train")
    out.loc[tr[va], "model_split"] = "val"
    return out


def patient_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Number of patients shared between each pair of splits (all should be 0)."""
    key = df.patient_id
    sets = {s: set(key[df.model_split == s]) for s in ("train", "val", "test")}
    return pd.DataFrame([{"splits": f"{a} & {b}", "shared_patients": len(sets[a] & sets[b])}
                         for a, b in (("train", "val"), ("train", "test"), ("val", "test"))])


def prepare(df: pd.DataFrame, cfg: Config, out: Path) -> Path:
    """Write preprocessed PNGs (skips files that already exist)."""
    p = cfg.raw["vision"]["preprocess"]
    for row in df.itertuples():
        dest = out / row.model_split / row.cls / (Path(row.path).stem + ".png")
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", preprocess(read_gray(row.abs_path), p))
        if not ok:
            raise ValueError(f"could not encode {row.path}")
        buf.tofile(str(dest))
    return out


def summary(df: pd.DataFrame) -> pd.DataFrame:
    t = df.pivot_table(index="model_split", columns="cls", values="path", aggfunc="count",
                       fill_value=0)
    t["total"] = t.sum(axis=1)
    t["patients"] = df.groupby("model_split").apply(
        lambda g: g.patient_id.nunique(), include_groups=False)
    return t.reindex(["train", "val", "test"]).reset_index()
