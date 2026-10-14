"""Load the real or synthetic tables and validate them against the schema.

Every stage calls `load_analysis` / `load_visits` with the `--synthetic` flag from the CLI.
Real DRCR data lives in `data/` (never committed); synthetic data with the identical schema
lives in `data/synthetic/` (committed, used by tests and CI).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from equipoise.config import Config
from equipoise.data.schema import validate_analysis, validate_cst_quality, validate_visits


def analysis_path(cfg: Config, synthetic: bool) -> Path:
    return cfg.paths.synthetic_analysis_table if synthetic else cfg.paths.analysis_table


def visits_path(cfg: Config, synthetic: bool) -> Path:
    return cfg.paths.synthetic_visits_table if synthetic else cfg.paths.visits_table


def _read(path: Path, synthetic: bool, build_hint: str) -> pd.DataFrame:
    if not path.is_file():
        hint = (
            "run `python scripts/make_synthetic_data.py`"
            if synthetic
            else f"build it with `{build_hint}`, or pass --synthetic"
        )
        raise FileNotFoundError(f"{path} not found: {hint}.")
    return pd.read_csv(path, low_memory=False)


def load_analysis(cfg: Config, synthetic: bool = False, validate: bool = True) -> pd.DataFrame:
    """One row per patient (660 x 47 for the real trial)."""
    df = _read(analysis_path(cfg, synthetic), synthetic,
               "python scripts/build_protocolT_table.py <raw dir> data/protocolT_analysis.csv")
    return validate_analysis(df, real=not synthetic) if validate else df


def load_cst_quality(cfg: Config, analysis: pd.DataFrame, synthetic: bool = False
                     ) -> pd.DataFrame:
    """Per-patient reading-centre CST quality flags (M_ columns only)."""
    path = cfg.paths.synthetic_cst_quality_table if synthetic else cfg.paths.cst_quality_table
    df = _read(path, synthetic,
               "python scripts/build_cst_quality.py <raw dir> data/protocolT_cst_quality.csv")
    return validate_cst_quality(df, analysis)


def load_visits(cfg: Config, synthetic: bool = False, validate: bool = True) -> pd.DataFrame:
    """One row per patient x visit, study eye only."""
    df = _read(visits_path(cfg, synthetic), synthetic,
               "python scripts/build_protocolT_visits.py <raw dir> data/protocolT_visits.csv")
    return validate_visits(df, real=not synthetic) if validate else df
