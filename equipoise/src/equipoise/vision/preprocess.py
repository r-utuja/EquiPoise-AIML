"""OpenCV preprocessing for OCT B-scans: grayscale, denoise, CLAHE, resize.

OCT images carry speckle noise. A median filter replaces each pixel with the median of its
neighbourhood, which removes salt-and-pepper speckle and keeps edges. Non-local means
averages pixels whose surrounding *patches* look alike, which removes more noise but is
much slower. CLAHE (contrast-limited adaptive histogram equalization) stretches contrast
tile by tile, with a clip limit so noise is not amplified, which makes retinal layers and
fluid pockets easier to see. Images are finally resized to a fixed square input size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_gray(path: Path | str) -> np.ndarray:
    """Read an image as 8-bit grayscale (np.fromfile: safe with any Windows path)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not decode image {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        code = cv2.COLOR_RGBA2GRAY if img.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        img = cv2.cvtColor(img, code)
    return img.astype(np.uint8) if img.dtype != np.uint8 else img


def denoise(img: np.ndarray, method: str, p: dict[str, Any]) -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(img, int(p["median_kernel"]))
    if method == "nlm":
        return cv2.fastNlMeansDenoising(img, None, h=float(p["nlm_h"]))
    if method == "none":
        return img
    raise ValueError(f"unknown denoise method {method!r}")


def clahe(img: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    g = int(p["clahe_grid"])
    return cv2.createCLAHE(clipLimit=float(p["clahe_clip"]), tileGridSize=(g, g)).apply(img)


def preprocess(img: np.ndarray, p: dict[str, Any], method: str | None = None) -> np.ndarray:
    """Full pipeline: grayscale -> denoise -> CLAHE -> resize (uint8, size x size)."""
    size = int(p["image_size"])
    out = clahe(denoise(to_gray(img), method or p["denoise"], p), p)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def before_after_figure(paths: list[Path], labels: list[str], p: dict[str, Any]) -> plt.Figure:
    """One row per image: original, median, non-local means, median + CLAHE (model input)."""
    cols = ["original", "median filter", "non-local means", "median + CLAHE + resize"]
    fig, axes = plt.subplots(len(paths), 4, figsize=(13, 3.0 * len(paths)))
    axes = np.atleast_2d(axes)
    for r, (path, lab) in enumerate(zip(paths, labels, strict=True)):
        raw = read_gray(path)
        views = [raw, denoise(raw, "median", p), denoise(raw, "nlm", p),
                 preprocess(raw, p, "median")]
        for c, v in enumerate(views):
            ax = axes[r, c]
            ax.imshow(v, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
        axes[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle("OCT preprocessing (Kermany dataset; not Protocol T patients)",
                 fontweight="bold")
    fig.tight_layout()
    return fig
