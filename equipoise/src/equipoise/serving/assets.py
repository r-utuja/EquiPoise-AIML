"""Small serving assets written by M9 so the API and app never need patient-level data.

The per-patient SHAP waterfall needs background patients. Instead of shipping training rows,
we keep k-means centroids of the preprocessed training features (aggregate summaries). The
app form's defaults are the training medians and modes. Both are fitted on training patients
only and saved next to the bundle in `serving_assets.joblib`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from equipoise.config import Config

ASSETS_NAME = "serving_assets.joblib"


def build_serving_assets(train: pd.DataFrame, bundle: Any, cfg: Config,
                         defaults: dict[str, Any]) -> dict[str, Any]:
    """Centroids of preprocessed training X (SHAP background), feature names, form defaults."""
    X = bundle.preprocessor.transform(train)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
    k = min(int(cfg.raw["serving"]["shap_background_clusters"]), len(X))
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(X)
    names = [n.split("__", 1)[1] for n in bundle.preprocessor.get_feature_names_out()]
    clean = {key: (v.item() if isinstance(v, np.generic) else v) for key, v in defaults.items()}
    return {"shap_background": km.cluster_centers_, "feature_names": names,
            "defaults": clean, "n_training_patients": len(train),
            "training_data_hash": bundle.training_data_hash}


def save_assets(assets: dict[str, Any], models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / ASSETS_NAME
    joblib.dump(assets, path, compress=3)
    return path


def load_assets(models_dir: Path) -> dict[str, Any] | None:
    path = models_dir / ASSETS_NAME
    return joblib.load(path) if path.is_file() else None
