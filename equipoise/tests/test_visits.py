"""Visit table: schema, cross-check against the analysis table, and the real build."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from equipoise.config import Config
from equipoise.data.crosscheck import cross_check
from equipoise.data.load import load_analysis, load_visits
from equipoise.data.schema import validate_visits

ROOT = Path(__file__).resolve().parents[1]


def _all_match(report: pd.DataFrame) -> None:
    bad = report[(report.n_missing_mismatch > 0) | (report.n_value_mismatch > 0)]
    assert bad.empty, f"cross-check mismatches:\n{bad}"


def test_synthetic_cross_check(synthetic: pd.DataFrame, synthetic_visits: pd.DataFrame) -> None:
    _all_match(cross_check(synthetic_visits, synthetic))


def test_cross_check_detects_a_changed_visit(synthetic: pd.DataFrame,
                                             synthetic_visits: pd.DataFrame) -> None:
    v = synthetic_visits.copy()
    row = v.index[(v.visit_label == "52 week")][0]
    v.loc[row, "va_letters"] += 3
    report = cross_check(v, synthetic).set_index("outcome")
    assert report.loc["Y_va_change_wk52", "n_value_mismatch"] == 1
    assert report.loc["Y_va_change_wk52", "max_abs_diff"] == 3


def test_synthetic_visit_labels_follow_drcr(synthetic_visits: pd.DataFrame) -> None:
    labels = set(synthetic_visits.visit_label)
    assert {"Baseline", "52 week", "104 week", "Post 52 week"} <= labels
    assert "56 week" not in labels


@pytest.fixture(scope="module")
def real_visits(cfg: Config) -> pd.DataFrame:
    if not cfg.paths.visits_table.is_file():
        pytest.skip("real visit table not built")
    return load_visits(cfg)


def test_real_visits_cross_check(cfg: Config, real_visits: pd.DataFrame) -> None:
    analysis = load_analysis(cfg)
    validate_visits(real_visits, real=True, analysis=analysis)
    assert real_visits.pt_id.nunique() == 660
    base = real_visits[real_visits.visit_label == "Baseline"]
    assert len(base) == 660 and (base.days_from_rand == 0).all()
    _all_match(cross_check(real_visits, analysis))


def test_real_build_reproduces_visit_table(cfg: Config, tmp_path: Path) -> None:
    """Re-running the build from raw data gives the committed-to-DVC table byte for byte."""
    if not cfg.paths.raw_dir.is_dir() or not cfg.paths.visits_table.is_file():
        pytest.skip("raw DRCR tables not present")
    out = tmp_path / "visits.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_protocolT_visits.py"),
         str(cfg.paths.raw_dir), str(out)],
        check=True, capture_output=True, cwd=ROOT,
    )
    assert out.read_bytes() == cfg.paths.visits_table.read_bytes()
