"""MLflow helpers shared by every stage.

Each module logs to its own experiment (`equipoise-m1-preprocess`, `equipoise-m2-stats`, ...).
`start_run` opens a run tagged with the git commit, the data source (real / synthetic) and a
short SHA-256 of the input table, so every number can be traced to code and data versions.
`log_figure` / `log_table` also write the report copies to `reports/figures/<module>_*.png`
and `reports/tables/<module>_*.csv` + `.md`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from equipoise.config import Config  # noqa: E402

# stage name (CLI) -> (module code used in file names, experiment suffix)
STAGES: dict[str, tuple[str, str]] = {
    "build": ("m0", "m0-build"),
    "preprocess": ("m1", "m1-preprocess"),
    "stats": ("m2", "m2-stats"),
    "represent": ("m3", "m3-represent"),
    "search": ("m4", "m4-search"),
    "supervised": ("m5", "m5-supervised"),
    "causal": ("m6", "m6-causal"),
    "longitudinal": ("m7", "m7-longitudinal"),
    "decision": ("m8", "m8-decision"),
    "explain": ("m9", "m9-explain"),
    "vision": ("m10", "m10-vision"),
    "report-assets": ("m11", "m11-report-assets"),
    "monitor": ("m11", "m11-monitor"),
}


def experiment_name(cfg: Config, stage: str) -> str:
    return f"{cfg.mlflow_experiment_prefix}-{STAGES[stage][1]}"


def module_code(stage: str) -> str:
    return STAGES[stage][0]


def tracking_uri(cfg: Config) -> str:
    """Tracking URI with a relative SQLite path made absolute (Windows and POSIX)."""
    uri = cfg.mlflow_tracking_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + (cfg.paths.root / uri[len(prefix):]).resolve().as_posix()
    return uri


def _set_experiment(cfg: Config, name: str) -> None:
    """Create the experiment with artifacts under the configured folder, then select it."""
    if mlflow.get_experiment_by_name(name) is None:
        location = (cfg.mlflow_artifact_dir / name).resolve().as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)


def git_commit(root: Path) -> str:
    """Current commit SHA with a `-dirty` suffix for uncommitted changes, or `unknown`."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def file_hash(path: Path, length: int = 12) -> str:
    """Short SHA-256 of a file, used as the data version tag."""
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextmanager
def start_run(
    cfg: Config,
    stage: str,
    data_file: Path,
    synthetic: bool,
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run in the stage's experiment with provenance tags."""
    mlflow.set_tracking_uri(tracking_uri(cfg))
    _set_experiment(cfg, experiment_name(cfg, stage))
    all_tags = {
        "stage": stage,
        "module": module_code(stage),
        "git_commit": git_commit(cfg.paths.root),
        "data_source": "synthetic" if synthetic else "real",
        "data_version": file_hash(data_file),
        "seed": str(cfg.seed),
        **(tags or {}),
    }
    with mlflow.start_run(run_name=run_name, tags=all_tags) as run:
        yield run


@contextmanager
def child_run(name: str, params: Mapping[str, Any] | None = None) -> Iterator[mlflow.ActiveRun]:
    """Nested MLflow run (e.g. one per model) inside the current stage run."""
    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tag("model", name)
        if params:
            log_params(params)
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    """Log (possibly nested) parameters; nested keys are joined with dots."""
    mlflow.log_params({k: str(v)[:500] for k, v in _flatten(params).items()})


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_figure(cfg: Config, stage: str, fig: plt.Figure, name: str) -> Path:
    """Save `fig` to reports/figures/<module>_<name>.png, log it, and close it."""
    path = cfg.paths.figures_dir / f"{module_code(stage)}_{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(path), artifact_path="figures")
    return path


def write_metrics(cfg: Config, stage: str, metrics: Mapping[str, float]) -> Path:
    """Write reports/metrics/<module>_<stage>.json (tracked by DVC) and log to MLflow."""
    path = cfg.paths.metrics_dir / f"{module_code(stage)}_{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: round(float(v), 6) for k, v in metrics.items()}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True) + chr(10), encoding="utf-8",
                    newline=chr(10))
    if mlflow.active_run() is not None:
        log_metrics(clean)
    return path


def log_table(cfg: Config, stage: str, df: pd.DataFrame, name: str, index: bool = False
              ) -> tuple[Path, Path]:
    """Write reports/tables/<module>_<name>.csv and .md, and log both."""
    base = cfg.paths.tables_dir / f"{module_code(stage)}_{name}"
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = base.with_suffix(".csv"), base.with_suffix(".md")
    # LF line endings so report files are byte-identical on Windows and Linux
    # 10 significant digits: hides last-bit noise from multithreaded sums (e.g. k-means
    # inertia), so identical results give identical files
    df.to_csv(csv_path, index=index, lineterminator="\n", float_format="%.10g")
    md_path.write_text(df.to_markdown(index=index, floatfmt=".3g") + "\n", encoding="utf-8",
                       newline="\n")
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(csv_path), artifact_path="tables")
        mlflow.log_artifact(str(md_path), artifact_path="tables")
    return csv_path, md_path
