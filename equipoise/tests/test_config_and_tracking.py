"""Config loading, price fallback, synthetic-data truth, tracking and CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from equipoise import tracking
from equipoise.cli import app
from equipoise.config import Config, load_prices


def test_config_values(cfg: Config) -> None:
    assert cfg.seed == 42 and cfg.n_folds == 5 and cfg.test_fraction == 0.2
    assert cfg.mcid_letters == 5 and cfg.bootstrap_reps == 200
    assert cfg.reference_arm == "Bevacizumab"
    assert cfg.comparison_arms == ("Aflibercept", "Ranibizumab")
    assert sum(cfg.arm_sizes.values()) == cfg.n_patients == 660
    assert cfg.paths.analysis_table.is_absolute()


def test_null_prices_fall_back_to_demo() -> None:
    with pytest.warns(UserWarning, match="DEMO"):
        p = load_prices()
    assert p.is_demo and "DEMO" in p.label
    assert p.per_injection["Bevacizumab"] < p.per_injection["Aflibercept"]


def test_price_override_wins() -> None:
    p = load_prices({"price_per_injection_inr": {"Aflibercept": 3.0, "Bevacizumab": 1.0,
                                                 "Ranibizumab": 2.0}, "laser_session_inr": 4})
    assert not p.is_demo and p.per_injection["Aflibercept"] == 3.0 and p.laser_session == 4.0


def test_partially_filled_costs_still_demo(tmp_path: Path, cfg: Config) -> None:
    raw = yaml.safe_load((cfg.paths.root / "config" / "costs.yaml").read_text(encoding="utf-8"))
    raw["price_per_injection_inr"]["Bevacizumab"] = 100
    path = tmp_path / "costs.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.warns(UserWarning):
        assert load_prices(path=path).is_demo


def test_synthetic_planted_effect_visible(synthetic: pd.DataFrame, cfg: Config) -> None:
    truth = pd.read_csv(cfg.paths.synthetic_truth_table)
    assert set(truth.pt_id) == set(synthetic.pt_id)
    d = synthetic.dropna(subset=["Y_va_change_wk52"])
    m = d.groupby(["X_va_below_69", "treatment"])["Y_va_change_wk52"].mean()
    gap_below = m[(1, "Aflibercept")] - m[(1, "Bevacizumab")]
    gap_above = m[(0, "Aflibercept")] - m[(0, "Bevacizumab")]
    assert gap_below > gap_above


def test_synthetic_visits_agree_with_outcomes(synthetic: pd.DataFrame,
                                              synthetic_visits: pd.DataFrame) -> None:
    v = synthetic_visits
    va52 = v[v.visit_label == "52 week"].set_index("pt_id")["va_letters"]
    derived = synthetic.pt_id.map(va52) - synthetic.X_va_letters
    pd.testing.assert_series_equal(derived, synthetic.Y_va_change_wk52, check_names=False)
    n_missing = int((synthetic.M_has_wk52 == 0).sum())
    assert 20 <= n_missing <= 70  # roughly the trial's 44


def test_tracking_logs_run(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import matplotlib.pyplot as plt

    paths = replace(cfg.paths, figures_dir=tmp_path / "fig", tables_dir=tmp_path / "tab")
    tmp_cfg = replace(cfg, paths=paths,
                      mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                      mlflow_artifact_dir=tmp_path / "mlruns")
    with tracking.start_run(tmp_cfg, "preprocess", cfg.paths.synthetic_analysis_table, True):
        tracking.log_params({"a": 1, "nested": {"b": 2}})
        tracking.log_metrics({"rmse": 1.5})
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        png = tracking.log_figure(tmp_cfg, "preprocess", fig, "demo")
        csv, md = tracking.log_table(tmp_cfg, "preprocess", pd.DataFrame({"x": [1.0]}), "demo")
        run_id = mlflow.active_run().info.run_id
    assert png.name == "m1_demo.png" and png.is_file()
    assert csv.is_file() and md.is_file()
    run = mlflow.get_run(run_id)
    assert run.data.params["nested.b"] == "2"
    assert run.data.metrics["rmse"] == 1.5
    assert run.data.tags["data_source"] == "synthetic"
    assert mlflow.get_experiment(run.info.experiment_id).name == "equipoise-m1-preprocess"
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "tables")}
    assert artifacts == {"tables/m1_demo.csv", "tables/m1_demo.md"}


def test_cli_validate_and_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    from equipoise import cli

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--synthetic"])
    assert result.exit_code == 0, result.output
    assert "analysis table OK" in result.output
    # a stage without an implementation prints a stub message (never runs a real stage here)
    monkeypatch.setattr(cli, "_runner", lambda stage: None)
    result = runner.invoke(app, ["decision", "--synthetic"])
    assert result.exit_code == 0 and "not implemented yet" in result.output


def light_causal(cfg: Config) -> dict:
    c = cfg.causal
    return {**c, "bootstrap_reps": 3, "automl": {**c["automl"], "enabled": False},
            "semi_synthetic": {**c["semi_synthetic"], "replications": 1},
            "tuning": {**c["tuning"], "trials": 3},
            "tarnet": {**c["tarnet"], "max_epochs": 5},
            "causal_forest": {**c["causal_forest"], "n_estimators": 100}}


STAGE_MODULES = {"preprocess": ("equipoise.preprocess.run", "m1"),
                 "stats": ("equipoise.stats.run", "m2"),
                 "represent": ("equipoise.represent.run", "m3"),
                 "supervised": ("equipoise.supervised.run", "m5"),
                 "search": ("equipoise.search.run", "m4"),
                 "longitudinal": ("equipoise.longitudinal.run", "m7")}


@pytest.mark.parametrize("stage", list(STAGE_MODULES))
def test_stage_runs_on_synthetic(stage: str, cfg: Config, tmp_path: Path) -> None:
    """Smoke-run a whole stage on synthetic data (light settings), writing into a temp folder."""
    from dataclasses import replace
    from importlib import import_module

    paths = replace(cfg.paths, figures_dir=tmp_path / "fig", tables_dir=tmp_path / "tab",
                    metrics_dir=tmp_path / "met", derived_dir=tmp_path / "derived",
                    models_dir=tmp_path / "models")
    sup = {**cfg.supervised, "knn_timing_repeats": 1, "lasso_n_alphas": 10,
           "random_forest": {**cfg.supervised["random_forest"], "n_estimators": 20},
           "gradient_boosting": {**cfg.supervised["gradient_boosting"], "n_estimators": 20}}
    small = replace(cfg, paths=paths, preprocess={**cfg.preprocess, "mice_imputations": 2},
                    supervised=sup, search={**cfg.search, "budget": 120, "seed_offsets": [0]},
                    causal=light_causal(cfg), bootstrap_reps=20,
                    mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                    mlflow_artifact_dir=tmp_path / "mlruns")
    module, code = STAGE_MODULES[stage]
    metrics = import_module(module).run(small, synthetic=True)
    assert metrics
    pngs = list((tmp_path / "fig").glob(f"{code}_*.png"))
    csvs = list((tmp_path / "tab").glob(f"{code}_*.csv"))
    assert pngs and csvs
    assert len(list((tmp_path / "tab").glob(f"{code}_*.md"))) == len(csvs)
    assert (tmp_path / "met" / f"{code}_{stage}.json").is_file()
    if stage == "longitudinal":
        import joblib

        from equipoise.causal.bundle import EARLY_MODEL_NAME

        assert (tmp_path / "models" / "synthetic" / EARLY_MODEL_NAME).is_file()
        assert joblib.load(tmp_path / "models" / "synthetic" / EARLY_MODEL_NAME) is not None
