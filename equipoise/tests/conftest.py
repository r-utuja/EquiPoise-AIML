"""Shared fixtures: config and the committed synthetic tables."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from equipoise.config import Config, load_config
from equipoise.data.load import load_analysis, load_visits


@pytest.fixture(autouse=True)
def _no_login(monkeypatch) -> None:
    """App tests start past the demo login page (tests/test_auth.py switches it back on)."""
    monkeypatch.setenv("EQUIPOISE_AUTH", "off")


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def _synthetic(cfg: Config) -> pd.DataFrame:
    return load_analysis(cfg, synthetic=True)


@pytest.fixture()
def synthetic(_synthetic: pd.DataFrame) -> pd.DataFrame:
    """A fresh copy of the synthetic analysis table (safe to mutate)."""
    return _synthetic.copy()


@pytest.fixture()
def synthetic_visits(cfg: Config) -> pd.DataFrame:
    return load_visits(cfg, synthetic=True)


@pytest.fixture(scope="session")
def real_analysis(cfg: Config) -> pd.DataFrame:
    """The real DRCR table; tests using it are skipped when it is absent (e.g. in CI)."""
    if not cfg.paths.analysis_table.is_file():
        pytest.skip("real DRCR analysis table not present")
    return pd.read_csv(cfg.paths.analysis_table, low_memory=False)


@pytest.fixture(scope="session")
def light(tmp_path_factory: pytest.TempPathFactory, cfg: Config) -> Config:
    """Config with light settings and every output in a temp folder; M6 already run."""
    tmp = tmp_path_factory.mktemp("pipeline")
    c = cfg.causal
    causal = {**c, "bootstrap_reps": 3, "automl": {**c["automl"], "enabled": False},
              "semi_synthetic": {**c["semi_synthetic"], "replications": 1},
              "tuning": {**c["tuning"], "trials": 3},
              "tarnet": {**c["tarnet"], "max_epochs": 5},
              "causal_forest": {**c["causal_forest"], "n_estimators": 100}}
    paths = replace(cfg.paths, figures_dir=tmp / "fig", tables_dir=tmp / "tab",
                    metrics_dir=tmp / "met", derived_dir=tmp / "derived",
                    models_dir=tmp / "models")
    out = replace(cfg, paths=paths, causal=causal, bootstrap_reps=20,
                  explain={**cfg.explain, "shap_background": 20},
                  mlflow_tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}",
                  mlflow_artifact_dir=tmp / "mlruns")
    from equipoise.causal.run import run

    assert run(out, synthetic=True)
    return out


@pytest.fixture(scope="session")
def bundle(light: Config):
    from equipoise.causal.bundle import load_bundle
    from equipoise.decision.recommend import default_bundle_path, serving_mode

    return serving_mode(load_bundle(default_bundle_path(light, synthetic=True)))
