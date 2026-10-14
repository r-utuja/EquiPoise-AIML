"""M2: Table 1, replication helpers, inference, and the real-data replication targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equipoise.config import Config
from equipoise.stats.inference import (
    interaction_regression,
    multiple_correlation,
    partial_correlation,
    variance_tests,
)
from equipoise.stats.replication import (
    burden_table,
    hypothesis_tests,
    load_targets,
    mean_ci,
    means_table,
    welch_ci,
)
from equipoise.stats.table1 import table1

ARMS = ["Aflibercept", "Bevacizumab", "Ranibizumab"]


def test_table1_has_every_feature(synthetic: pd.DataFrame, cfg: Config) -> None:
    t1 = table1(synthetic, cfg)
    names = set(t1.Variable.str.strip())
    for c in [c for c in synthetic.columns if c.startswith("X_")]:
        assert c.replace("X_", "") in names
    used = set(t1.test.str.split(" \\(").str[0]) - {""}
    assert {"one-way ANOVA", "chi-square"} <= used <= {"one-way ANOVA", "Kruskal-Wallis",
                                                       "chi-square"}


def test_mean_ci_and_welch() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    m, sd, lo, hi = mean_ci(x, 0.95)
    assert m == 3.0 and sd == pytest.approx(np.sqrt(2.5))
    assert lo == pytest.approx(3 - 2.7764451 * np.sqrt(2.5) / np.sqrt(5), rel=1e-6)
    d, lo, hi, p = welch_ci(x + 2, x, 0.95)
    assert d == 2.0 and lo < 2 < hi and 0 < p < 1


def test_tests_table_holm_adjusts(synthetic: pd.DataFrame, cfg: Config) -> None:
    t = hypothesis_tests(synthetic, cfg)
    welch = t[t.test == "Welch t-test"]
    assert (welch.p_holm >= welch.p - 1e-12).all()
    assert len(welch) == 2 * 3 * 4  # 2 horizons x 3 pairs x (3 subgroups + CST)


def test_inference_on_synthetic(synthetic: pd.DataFrame, cfg: Config) -> None:
    v = variance_tests(synthetic, cfg)
    assert {"Levene (Brown-Forsythe)", "Bartlett"} <= set(v.test)
    pc = partial_correlation(synthetic, "X_va_letters", cfg.primary_outcome, ["X_cst_um"])
    assert -1 <= pc["partial_r"] <= 1 and pc["partial_r"] < 0  # worse vision -> more gain
    mc = multiple_correlation(synthetic, cfg.primary_outcome, ["X_va_letters", "X_age"])
    assert 0 <= mc["R"] <= 1 and mc["R2"] == pytest.approx(mc["R"] ** 2)


def test_interaction_detects_planted_effect(synthetic: pd.DataFrame, cfg: Config) -> None:
    """The synthetic data plant aflibercept x va_below_69; the classical test should see it."""
    _, test = interaction_regression(synthetic, cfg, "X_va_below_69")
    assert test["p_interaction"] < 0.05


def test_partial_correlation_removes_shared_driver() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    df = pd.DataFrame({"z": z, "x": z + rng.normal(size=2000), "y": z + rng.normal(size=2000)})
    pc = partial_correlation(df, "x", "y", ["z"])
    assert pc["raw_r"] > 0.4 and abs(pc["partial_r"]) < 0.06


# --- real-data replication targets (README section 9.1) --------------------------------------

def _arm_values(frame: pd.DataFrame, col: str) -> list[float]:
    s = frame.set_index("arm")[col]
    return [float(s[a]) for a in ARMS]


def test_replication_matches_readme(real_analysis: pd.DataFrame, cfg: Config) -> None:
    targets = load_targets(cfg)
    tol = float(cfg.stats["replication_tolerance_letters"])
    means, burden = means_table(real_analysis, cfg), burden_table(real_analysis, cfg)

    def letters(h: int, sg: str) -> list[float]:
        return _arm_values(means[(means.horizon_years == h) & (means.subgroup == sg)
                                 & (means.outcome == "letters")], "mean")

    y1, y2 = targets["year1"], targets["year2"]
    for key, sg in [("mean_letters_gained", "all"),
                    ("mean_letters_gained_va_below_69", "va_below_69"),
                    ("mean_letters_gained_va_69_or_better", "va_69_or_better")]:
        np.testing.assert_allclose(letters(1, sg), y1[key]["our_table"], atol=tol)
    np.testing.assert_allclose(letters(2, "all"), y2["mean_letters_gained"]["our_table"],
                               atol=tol)
    for h, block in ((1, y1), (2, y2)):
        b = burden[burden.horizon_years == h]
        assert [round(v) for v in _arm_values(b, "median_injections")] == \
            block["median_injections"]["our_table"]
        assert [round(v) for v in _arm_values(b, "laser_pct")] == block["laser_pct"]["our_table"]
    cst = _arm_values(means[(means.horizon_years == 1) & (means.outcome == "cst_um")], "mean")
    assert [round(v) for v in cst] == y1["mean_cst_change_um"]["our_table"]
