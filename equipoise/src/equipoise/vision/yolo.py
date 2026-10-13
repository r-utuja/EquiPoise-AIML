"""Ultralytics YOLO classification model (nano `-cls` checkpoint, fine-tuned).

YOLO's classification variant is a small convolutional backbone with a global-pooling
classifier head, pretrained on ImageNet. We fine-tune the nano checkpoint on the same
preprocessed folders as the ResNet, with the same image size, epochs and seed, so the two
models are compared fairly. Ultralytics' anonymous usage analytics and its own MLflow
logging are switched off (the stage run logs everything).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _quiet_settings() -> None:
    from ultralytics import settings

    settings.update({"sync": False,     # no anonymous usage analytics
                     "mlflow": False})  # our stage run does the MLflow logging


def train(data_dir: Path, p: dict[str, Any], image_size: int, seed: int, out_dir: Path):
    """Fine-tune on data_dir/{train,val}; returns the YOLO model loaded from best.pt."""
    _quiet_settings()
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(out_dir / p["checkpoint"]))   # downloads here, not into the repo root
    model.train(data=str(data_dir), epochs=int(p["epochs"]), imgsz=image_size,
                batch=int(p["batch_size"]), patience=int(p["patience"]), seed=seed,
                deterministic=True, workers=0, device="cpu", project=str(out_dir),
                name="yolo_cls", exist_ok=True, plots=False, verbose=False)
    best = out_dir / "yolo_cls" / "weights" / "best.pt"
    return YOLO(str(best)), best


def predict_folder(model, folder: Path, classes: list[str], image_size: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(true labels, class probabilities in `classes` order) for <folder>/<class>/*.png."""
    ys, ps = [], []
    names = model.names                                     # index -> class name
    order = [next(i for i, n in names.items() if n == c) for c in classes]
    for k, cls in enumerate(classes):
        files = sorted((folder / cls).glob("*.png"))
        for i in range(0, len(files), 64):
            res = model.predict([str(f) for f in files[i:i + 64]], imgsz=image_size,
                                device="cpu", verbose=False)
            for r in res:
                ps.append(r.probs.data.numpy()[order])
                ys.append(k)
    return np.array(ys), np.vstack(ps)



def history(out_dir: Path):
    """Per-epoch training loss and validation top-1 accuracy written by Ultralytics."""
    import pandas as pd

    df = pd.read_csv(out_dir / "yolo_cls" / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"train/loss": "train_loss", "val/loss": "val_loss",
                            "metrics/accuracy_top1": "val_acc"})
    return df[[c for c in ("epoch", "train_loss", "val_loss", "val_acc") if c in df.columns]]
