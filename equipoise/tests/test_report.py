"""Phase 11: report tables are generated from the schema, the code and MLflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from equipoise.config import Config
from equipoise.data.schema import ANALYSIS_COLUMNS


def test_feature_description_covers_schema() -> None:
    from equipoise.report.tables import feature_description

    t = feature_description()
    assert list(t.Feature) == [c.name for c in ANALYSIS_COLUMNS]
    roles = dict(zip(t.Feature, t.Role, strict=True))
    assert roles["X_va_letters"].startswith("model input")
    assert roles["Y_va_change_wk52"].startswith("target only")
    assert roles["M_has_wk52"].startswith("bookkeeping")


def test_tech_stack_lists_real_imports_only(cfg: Config) -> None:
    from equipoise.report.tables import tech_stack

    t = tech_stack([cfg.paths.root / "src" / "equipoise"]).set_index("Library")
    assert {"numpy", "pandas", "scikit-learn", "econml", "mlflow"} <= set(t.index)
    assert not {"os", "json", "pathlib", "equipoise"} & set(t.index)
    assert t.loc["numpy", "Key functions used"]      # found via `np.` attribute use


def test_experiments_table_reads_mlflow(cfg: Config, tmp_path: Path) -> None:
    import mlflow

    from equipoise.report.tables import experiments_table
    from equipoise.tracking import experiment_name

    small = replace(cfg, mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'm.db').as_posix()}",
                    mlflow_artifact_dir=tmp_path / "mlruns")
    mlflow.set_tracking_uri(small.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name(small, "supervised"))
    with mlflow.start_run(tags={"data_source": "real"}):
        mlflow.log_param("supervised.random_forest.n_estimators", 300)
        for model, rmse in [("Baseline: arm mean", 10.9), ("Random forest", 9.7)]:
            with mlflow.start_run(nested=True):
                mlflow.log_params({"task": "regression", "model": model})
                mlflow.log_metrics({"CV_RMSE": rmse, "Test_RMSE": rmse + 0.1})
    with mlflow.start_run(tags={"data_source": "synthetic"}):    # ignored
        mlflow.log_metric("x", 1)
    t = experiments_table(small)
    assert list(t.Exp) == ["Exp-01", "Exp-02"]
    rf = t[t.Model.str.contains("Random forest")].iloc[0]
    assert "n_estimators=300" in rf.Parameters and "CV RMSE=9.7" in rf.Metrics
    assert "best CV RMSE" in rf.Remarks
    assert t[t.Model.str.contains("Baseline")].iloc[0].Remarks == "baseline"


def test_index_maps_every_generated_file(tmp_path: Path) -> None:
    from equipoise.report.assets_index import build_index

    (tmp_path / "figures").mkdir()
    (tmp_path / "tables").mkdir()
    for name in ("figures/m5_roc.png", "tables/m6_policy_values.md", "figures/flow_diagram.png"):
        (tmp_path / name).write_text("x")
    text, missing = build_index(tmp_path)
    assert "figures/m5_roc.png" in text.split("## 7.3")[1].split("## 7.4")[0]
    assert "flow_diagram.png" in text.split("## 5.")[1].split("## 6.")[0]
    assert "tables/m11_tech_stack.md" in missing and "**(missing" in text
    assert "Screenshots to take manually" in text


def test_diagrams_render() -> None:
    from equipoise.report.diagrams import causal_dag, flow_diagram

    for fig in (flow_diagram(), causal_dag()):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
