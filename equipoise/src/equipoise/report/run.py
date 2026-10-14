"""M11 report-assets stage: flow diagram, causal DAG, template tables, REPORT_ASSETS.md.

Everything here is generated from the code, the schema, the MLflow log and the files the
other stages wrote, so the report never carries a number the pipeline did not produce.
"""

from __future__ import annotations

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path
from equipoise.report.assets_index import build_index
from equipoise.report.diagrams import causal_dag, flow_diagram
from equipoise.report.tables import experiments_table, feature_description, tech_stack

STAGE = "report-assets"


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    root = cfg.paths.root
    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        fig = flow_diagram()
        path = cfg.paths.figures_dir / "flow_diagram.png"   # name fixed by the report spec
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        import mlflow

        plt.close(fig)
        mlflow.log_artifact(str(path), artifact_path="figures")
        tracking.log_figure(cfg, STAGE, causal_dag(), "causal_dag")
        tracking.log_table(cfg, STAGE, feature_description(), "feature_description")
        stack = tech_stack([root / "src" / "equipoise", root / "api", root / "app"])
        tracking.log_table(cfg, STAGE, stack, "tech_stack")
        exps = experiments_table(cfg)
        tracking.log_table(cfg, STAGE, exps, "experiments")
        text, missing = build_index(cfg.paths.figures_dir.parent)
        index = cfg.paths.figures_dir.parent / "REPORT_ASSETS.md"
        index.write_text(text + "\n", encoding="utf-8", newline="\n")
        mlflow.log_artifact(str(index))
        metrics = {"n_experiment_rows": float(len(exps)), "n_libraries": float(len(stack)),
                   "n_missing_assets": float(len(missing))}
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
