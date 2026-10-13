"""M10 OCT screening: preprocessing, patient-grouped split, ResNet training loop, Grad-CAM.

Uses small generated images and untrained networks, so no download is needed. Skipped when
the [cv] extra is not installed (CI installs dev + app only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equipoise.config import Config

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torchvision")


@pytest.fixture()
def p(cfg: Config) -> dict:
    return {**cfg.raw["vision"]["preprocess"], "image_size": 32}


def test_preprocess_shape_and_contrast(p: dict) -> None:
    from equipoise.vision.preprocess import clahe, denoise, preprocess

    rng = np.random.default_rng(0)
    img = (100 + 10 * rng.random((60, 80))).astype(np.uint8)        # low contrast
    out = preprocess(np.dstack([img] * 3), p)                        # colour input ok
    assert out.shape == (32, 32) and out.dtype == np.uint8
    assert clahe(img, p).std() > img.std()                          # CLAHE stretches contrast
    noisy = img.copy()
    noisy[rng.random(img.shape) < 0.05] = 255                        # salt noise
    assert np.abs(denoise(noisy, "median", p).astype(int) - img).mean() < \
        np.abs(noisy.astype(int) - img).mean()
    with pytest.raises(ValueError):
        denoise(img, "gaussian", p)


def test_split_groups_patients_and_keeps_official_test() -> None:
    from equipoise.vision.data import assign_splits, patient_overlap

    rows = [{"split": "train", "cls": c, "patient_id": str(i // 3), "path": f"{c}{i}"}
            for c in ("CNV", "DME") for i in range(60)]
    rows += [{"split": "test", "cls": "CNV", "patient_id": "900", "path": "t1"}]
    df = assign_splits(pd.DataFrame(rows), 0.25, seed=1)
    assert (df[df.split == "test"].model_split == "test").all()
    ov = patient_overlap(df).set_index("splits").shared_patients
    assert ov["train & val"] == 0
    assert 0.1 < (df.model_split == "val").mean() < 0.4


def _folder(root: Path, classes: list[str], n: int) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for k, c in enumerate(classes):
            d = root / split / c
            d.mkdir(parents=True)
            for i in range(n):
                img = (rng.random((32, 32)) * 60 + 120 * k).astype(np.uint8)
                cv2.imencode(".png", img)[1].tofile(str(d / f"{c}-{i}-1.png"))


def test_resnet_trains_predicts_and_gradcam(tmp_path: Path) -> None:
    from equipoise.vision.resnet import gradcam_overlay, predict_folder, train

    classes = ["CNV", "NORMAL"]
    _folder(tmp_path, classes, 6)
    params = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "weight_decay": 0.0,
              "freeze_backbone_epochs": 1}
    res = train(tmp_path, params, seed=0, pretrained=False)
    assert res.classes == classes and len(res.history) == 2
    assert res.history[0]["backbone_frozen"] and not res.history[1]["backbone_frozen"]
    y, prob, paths = predict_folder(res.model, tmp_path / "test", 4)
    assert prob.shape == (12, 2) and np.allclose(prob.sum(1), 1) and len(paths) == 12
    heat, overlay = gradcam_overlay(res.model, np.full((32, 32), 128, np.uint8))
    assert heat.shape == (32, 32) and 0 <= heat.min() <= heat.max() <= 1
    assert overlay.shape == (32, 32, 3)


def test_metrics_rows_known_values() -> None:
    from equipoise.vision.run import metrics_rows

    y = np.array([0, 0, 1, 1])
    prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    row, per = metrics_rows("m", y, prob, ["A", "B"])
    assert row["Accuracy"] == 0.75
    assert per.set_index("class").loc["A", "recall"] == 0.5
    assert per.set_index("class").loc["B", "precision"] == pytest.approx(2 / 3)


def test_vision_refuses_synthetic(cfg: Config) -> None:
    from equipoise.vision.run import run

    with pytest.raises(ValueError, match="no synthetic"):
        run(cfg, synthetic=True)
