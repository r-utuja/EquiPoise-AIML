"""M11 monitoring: Evidently data-drift report on the features sent to the API.

Every API/app request appends the patient's baseline features to `logs/requests.jsonl`.
Drift monitoring asks whether those incoming patients still look like the training patients
the model learned from. Evidently runs one statistical test per feature: Kolmogorov-Smirnov
or Wasserstein distance for numeric features, chi-square or Jensen-Shannon for categorical
ones, chosen by sample size. It flags the dataset as drifted when the share of drifted
features passes a threshold. Drift does not prove the model is wrong, but it does warn that
its estimates may not transfer.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.schema import ANALYSIS_COLUMNS
from equipoise.data.splits import get_split

STAGE = "monitor"
FEATURE_SPECS = [c for c in ANALYSIS_COLUMNS if c.role == "feature"]


def read_request_log(path: Path) -> pd.DataFrame:
    """Logged requests -> one row per request with every baseline feature."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line)["features"])
    return _typed(pd.DataFrame(rows, columns=[c.name for c in FEATURE_SPECS]))


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURE_SPECS:
        if c.categories:
            out[c.name] = out[c.name].astype(object).where(out[c.name].notna(), None)
        else:
            out[c.name] = pd.to_numeric(out[c.name], errors="coerce")
    return out


def _definition():
    from evidently import DataDefinition

    cat = [c.name for c in FEATURE_SPECS if c.categories or c.binary]
    num = [c.name for c in FEATURE_SPECS if c.name not in cat]
    return DataDefinition(numerical_columns=num, categorical_columns=cat)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: Config,
                 html: Path | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Evidently's DataDriftPreset; return a per-feature table and summary numbers."""
    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset

    m = cfg.raw["monitor"]
    d = _definition()
    ref = Dataset.from_pandas(_typed(reference), data_definition=d)
    cur = Dataset.from_pandas(_typed(current), data_definition=d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = Report([DataDriftPreset(drift_share=float(m["drift_share"]))]).run(
            current_data=cur, reference_data=ref)
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(html))
    rows, share = [], None
    for metric in snap.dict()["metrics"]:
        name, value = str(metric.get("metric_name", "")), metric.get("value")
        cfgm = metric.get("config", {})
        if name.startswith("DriftedColumnsCount"):
            share = float(value["share"])
        elif name.startswith("ValueDrift") and "column" in cfgm:
            method = str(cfgm.get("method", ""))
            thr = cfgm.get("threshold")
            score = float(value)
            # Evidently: p-value tests drift when score < threshold; distances when score >=
            is_pvalue = "p_value" in method or "test" in method.lower()
            drifted = score < thr if is_pvalue else score >= thr
            rows.append({"feature": cfgm["column"], "method": method, "score": score,
                         "threshold": thr, "drifted": bool(drifted)})
    table = pd.DataFrame(rows).sort_values(["drifted", "feature"], ascending=[False, True])
    summary = {"n_reference": float(len(reference)), "n_current": float(len(current)),
               "share_drifted_features": share if share is not None
               else float(table.drifted.mean()),
               "dataset_drift": float((share or 0.0) >= float(m["drift_share"]))}
    return table.reset_index(drop=True), summary


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Compare the training split's baseline features with the logged API requests."""
    from equipoise.serving.service import resolve_log_path

    log = resolve_log_path(cfg)
    if log is None or not log.is_file():
        raise FileNotFoundError(
            f"No request log at {log}. Send some requests through the API or app, or run "
            "`python scripts/simulate_requests.py` (add --synthetic for synthetic patients).")
    current = read_request_log(log)
    m = cfg.raw["monitor"]
    if len(current) < int(m["min_requests"]):
        raise ValueError(f"only {len(current)} logged requests; need >= {m['min_requests']}")
    df = load_analysis(cfg, synthetic=synthetic)
    reference = get_split(cfg, df, synthetic).train(df)[[c.name for c in FEATURE_SPECS]]
    out_dir = cfg.paths.figures_dir.parent / m["output_dir"]
    html = out_dir / "drift_report.html"
    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"monitor": m, "request_log": str(log)})
        table, summary = drift_report(reference, current, cfg, html)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "drift_by_feature.csv", index=False)
        table.to_markdown(out_dir / "drift_by_feature.md", index=False, floatfmt=".4g")
        tracking.log_metrics(summary)
        import mlflow

        mlflow.log_artifact(str(html), artifact_path="monitoring")
        mlflow.log_artifact(str(out_dir / "drift_by_feature.csv"), artifact_path="monitoring")
    return {**summary, "report": str(html)}
