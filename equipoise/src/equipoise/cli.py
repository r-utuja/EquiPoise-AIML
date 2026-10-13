"""Command-line entry point: `python -m equipoise <stage> [--synthetic]`.

Each stage is one module of the pipeline (README section 6.4). Stages that are not built yet
print which phase will implement them and exit cleanly, so `all` and DVC can be wired early.
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from equipoise.config import Config, for_data_source, load_config
from equipoise.reproducibility import seed_everything

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Equipoise pipeline (research prototype, not for clinical use).",
)

SyntheticOpt = typer.Option(
    False, "--synthetic", help="Run on data/synthetic/ instead of the DRCR data."
)

# Pipeline order used by `all` (README section 6.4).
PIPELINE = (
    "preprocess",
    "stats",
    "represent",
    "supervised",
    "search",
    "causal",
    "longitudinal",
    "decision",
    "explain",
    "report-assets",
)

# stage -> phase that implements it (shown by stubs)
PLANNED_PHASE = {
    "preprocess": 2,
    "stats": 2,
    "represent": 3,
    "supervised": 3,
    "search": 4,
    "causal": 5,
    "longitudinal": 6,
    "decision": 7,
    "explain": 7,
    "report-assets": 11,
    "monitor": 9,
    "vision": 10,
}

def _runner(stage: str) -> Callable[[Config, bool], object] | None:
    """Import the implemented stage lazily (keeps `--help` fast)."""
    if stage == "preprocess":
        from equipoise.preprocess.run import run
        return run
    if stage == "stats":
        from equipoise.stats.run import run
        return run
    if stage == "represent":
        from equipoise.represent.run import run
        return run
    if stage == "supervised":
        from equipoise.supervised.run import run
        return run
    if stage == "search":
        from equipoise.search.run import run
        return run
    if stage == "causal":
        from equipoise.causal.run import run
        return run
    if stage == "longitudinal":
        from equipoise.longitudinal.run import run
        return run
    if stage == "decision":
        from equipoise.decision.run import run
        return run
    if stage == "explain":
        from equipoise.explain.run import run
        return run
    if stage == "vision":
        from equipoise.vision.run import run
        return run
    if stage == "report-assets":
        from equipoise.report.run import run
        return run
    if stage == "monitor":
        from equipoise.monitor import run
        return run
    return None


def run_stage(stage: str, synthetic: bool) -> None:
    """Seed everything, then run one stage (synthetic outputs go to reports/synthetic/)."""
    cfg = for_data_source(load_config(), synthetic)
    seed_everything(cfg.seed)
    runner = _runner(stage)
    if runner is None:
        typer.echo(f"[{stage}] not implemented yet (planned for phase {PLANNED_PHASE[stage]}).")
        return
    source = ("Kermany OCT (not Protocol T)" if stage == "vision"
              else "synthetic" if synthetic else "real")
    typer.echo(f"[{stage}] running on {source} data ...")
    result = runner(cfg, synthetic)
    if isinstance(result, dict):
        for k, v in result.items():
            typer.echo(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")


def _register(stage: str, help_text: str) -> None:
    def command(synthetic: bool = SyntheticOpt) -> None:
        run_stage(stage, synthetic)

    command.__doc__ = help_text
    app.command(name=stage, help=help_text)(command)


for _stage, _help in [
    ("preprocess", "M1: missing data, CST harmonization, outliers, encoding."),
    ("stats", "M2: Table 1, trial replication, hypothesis tests."),
    ("represent", "M3: PCA/SVD, clustering, association rules."),
    ("supervised", "M5: regression, classification and count outcome models."),
    ("search", "M4: feature selection and search arena."),
    ("causal", "M6: counterfactual engine (CATE learners, policy value)."),
    ("longitudinal", "M7: trajectories, time series, HMM, early response."),
    ("decision", "M8: cost model, break-even, fuzzy system, expert system."),
    ("explain", "M9: SHAP and subgroup checks."),
    ("vision", "M10 (optional): OCT screening on the Kermany dataset (not Protocol T)."),
    ("report-assets", "M11: report figures, tables and docs."),
    ("monitor", "M11: Evidently drift report on logged API requests."),
]:
    _register(_stage, _help)


@app.command("all")
def run_all(synthetic: bool = SyntheticOpt) -> None:
    """Run every pipeline stage in order."""
    for stage in PIPELINE:
        run_stage(stage, synthetic)


@app.command("split")
def split(synthetic: bool = SyntheticOpt) -> None:
    """Draw and save the fixed held-out test split (patient IDs, stratified by arm)."""
    from equipoise.data.load import load_analysis
    from equipoise.data.splits import create_split, splits_path

    cfg = load_config()
    df = load_analysis(cfg, synthetic=synthetic)
    s = create_split(cfg, df, synthetic)
    test = s.test(df)["treatment"].value_counts().sort_index().to_dict()
    typer.echo(f"wrote {splits_path(cfg, synthetic)}: train {len(s.train_ids)}, "
               f"test {len(s.test_ids)} {test}")


@app.command("validate")
def validate(synthetic: bool = SyntheticOpt) -> None:
    """Validate the analysis (and, if present, visit) table against the schema."""
    from equipoise.data.load import load_analysis, load_visits, visits_path
    from equipoise.data.schema import validate_visits

    cfg = load_config()
    df = load_analysis(cfg, synthetic=synthetic)
    typer.echo(f"analysis table OK: {df.shape[0]} rows x {df.shape[1]} columns")
    typer.echo(f"arms: {df['treatment'].value_counts().sort_index().to_dict()}")
    if visits_path(cfg, synthetic).is_file():
        visits = load_visits(cfg, synthetic=synthetic, validate=False)
        validate_visits(visits, real=not synthetic, analysis=df)
        typer.echo(f"visit table OK: {visits.shape[0]} rows, {visits['pt_id'].nunique()} patients")
    else:
        typer.echo("visit table not built yet")


@app.command("promote")
def promote(version: str = typer.Option(None, help="Registered version (default: newest)."),
            alias: str = typer.Option("champion", help="Registry alias to set.")) -> None:
    """M11: mark a registered `equipoise-cate` version as the served model (MLflow alias)."""
    from equipoise.causal.bundle import load_bundle
    from equipoise.decision.recommend import default_bundle_path
    from equipoise.registry import promote_bundle
    from equipoise.tracking import tracking_uri

    cfg = load_config()
    path = default_bundle_path(cfg)
    bundle = load_bundle(path) if path.is_file() else None
    chosen = promote_bundle(tracking_uri(cfg), version, alias, bundle)
    typer.echo(f"equipoise-cate version {chosen} now has alias '{alias}'")


def main() -> None:
    app()
