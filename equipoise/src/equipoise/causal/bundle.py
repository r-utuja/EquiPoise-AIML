"""The Equipoise model bundle: everything needed to score a new patient, in one file.

Contents: the fitted preprocessor, the chosen effect (CATE) model, a bootstrap ensemble of it
(for intervals and P(benefit > MCID)), per-arm outcome models for letters, year-1 injections
and laser, the feature schema, a hash of the training data and the config used. Letters under
each drug are anchored on the bevacizumab outcome model plus the estimated effect, so the
displayed letters and effects always agree. Saved with joblib and registered in MLflow as
`equipoise-cate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BUNDLE_NAME = "equipoise_bundle.joblib"
EARLY_MODEL_NAME = "early_response_model.joblib"   # written by M7, attached on load
REGISTERED_MODEL = "equipoise-cate"


@dataclass
class ModelBundle:
    arms: list[str]                    # code order: reference first
    reference_arm: str
    mcid_letters: float
    preprocessor: Any
    effect_model: Any
    effect_model_name: str
    bootstrap_models: list[Any]
    outcome_models: list[Any]          # per arm (code order), on preprocessed X
    injection_model: Any               # pipeline on raw df incl. treatment
    laser_model: Any                   # pipeline on raw df incl. treatment
    feature_schema: list[dict[str, Any]]
    training_data_hash: str
    config_snapshot: dict[str, Any]
    synthetic: bool
    extras: dict[str, Any] = field(default_factory=dict)
    early_response_model: Any = None   # M7: baseline + week 4/8/12 vision -> week-52 letters

    def _with_arm(self, df: pd.DataFrame, arm: str) -> pd.DataFrame:
        return df.assign(treatment=arm)

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-patient predictions under every arm (arrays in `self.arms` order)."""
        X = self.preprocessor.transform(df)
        eff = self.effect_model.effect(X)                      # (n, k) vs reference
        boot = np.stack([m.effect(X) for m in self.bootstrap_models]) \
            if self.bootstrap_models else eff[None]
        ref = self.outcome_models[0].predict(X)
        letters = np.column_stack([ref, ref[:, None] + eff]) if eff.ndim == 2 else ref
        inj = np.column_stack([self.injection_model.predict(self._with_arm(df, a))
                               for a in self.arms])
        laser = np.column_stack([self.laser_model.predict_proba(self._with_arm(df, a))[:, 1]
                                 for a in self.arms])
        return {
            "effect": eff,
            "effect_low": np.quantile(boot, 0.025, axis=0),
            "effect_high": np.quantile(boot, 0.975, axis=0),
            "p_benefit_gt_mcid": (boot > self.mcid_letters).mean(axis=0),
            "letters": letters,
            "letters_low": np.column_stack([ref, ref[:, None] + np.quantile(boot, 0.025, 0)]),
            "letters_high": np.column_stack([ref, ref[:, None] + np.quantile(boot, 0.975, 0)]),
            "injections_yr1": inj,
            "p_laser_yr1": laser,
        }


def save_bundle(bundle: ModelBundle, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / BUNDLE_NAME
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path) -> ModelBundle:
    """Load the bundle and attach M7's early-response model if it has been built.

    The early model lives in its own file (each DVC stage owns its outputs) but is exposed
    as `bundle.early_response_model`.
    """
    bundle: ModelBundle = joblib.load(path)
    early = path.parent / EARLY_MODEL_NAME
    if early.is_file():
        bundle.early_response_model = joblib.load(early)
    return bundle


def register_bundle(path: Path, run_name: str) -> str | None:
    """Log the bundle as an MLflow pyfunc model and register it as `equipoise-cate`.

    Returns the registered version, or None if registration is not possible (e.g. no run).
    """
    import mlflow
    import mlflow.pyfunc

    class EquipoisePyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context: Any) -> None:
            self.bundle = joblib.load(context.artifacts["bundle"])

        def predict(self, context: Any, model_input: pd.DataFrame,
                    params: dict | None = None) -> pd.DataFrame:
            out = self.bundle.predict(model_input)
            cols = {}
            for k, arm in enumerate(self.bundle.arms[1:]):
                cols[f"effect_{arm}"] = out["effect"][:, k]
                cols[f"p_benefit_gt_mcid_{arm}"] = out["p_benefit_gt_mcid"][:, k]
            return pd.DataFrame(cols)

    if mlflow.active_run() is None:
        return None
    info = mlflow.pyfunc.log_model(name="equipoise_bundle", python_model=EquipoisePyfunc(),
                                   artifacts={"bundle": str(path)},
                                   registered_model_name=REGISTERED_MODEL)
    return str(getattr(info, "registered_model_version", "") or "")
