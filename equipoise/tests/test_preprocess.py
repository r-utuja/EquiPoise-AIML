"""M1: preprocessor, harmonizer, imputers fitted on train only, Rubin's rules, outliers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from equipoise.config import Config, for_data_source
from equipoise.data.splits import get_split
from equipoise.features import LeakageError
from equipoise.preprocess.columns import feature_groups
from equipoise.preprocess.cst_quality import apply_cst_quality
from equipoise.preprocess.discretize import discretize, gain_item
from equipoise.preprocess.encode import build_preprocessor
from equipoise.preprocess.harmonize import CSTHarmonizer
from equipoise.preprocess.missing import compare_strategies, rubin_pool
from equipoise.preprocess.outliers import detect_outliers, overlap_table


@pytest.fixture()
def split_frames(synthetic: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = get_split(cfg, synthetic, synthetic=True)
    return s.train(synthetic), s.test(synthetic)


def test_feature_groups_cover_every_x_column(synthetic: pd.DataFrame, cfg: Config) -> None:
    g = feature_groups(cfg)
    x_cols = {c for c in synthetic.columns if c.startswith("X_")}
    assert set(g.all) == x_cols
    assert len(g.all) == len(set(g.all))


@pytest.mark.parametrize("imputer", ["median", "knn", "iterative"])
def test_preprocessor_outputs_complete_matrix(split_frames, cfg: Config, imputer: str) -> None:
    train, test = split_frames
    pre = build_preprocessor(cfg, imputer=imputer).fit(train)
    Z = pre.transform(test)
    assert Z.shape[0] == len(test) and not np.isnan(Z).any()
    names = pre.get_feature_names_out()
    assert not any("Y_" in n or "M_" in n or "pt_id" in n for n in names)
    assert not any("treatment" in n for n in names)


def test_preprocessor_ignores_outcome_columns(split_frames, cfg: Config) -> None:
    """Changing every Y_/M_ value must not change the transformed features."""
    train, test = split_frames
    pre = build_preprocessor(cfg).fit(train)
    scrambled = test.copy()
    for c in [c for c in test.columns if c.startswith(("Y_", "M_"))]:
        scrambled[c] = scrambled[c].sample(frac=1, random_state=0).to_numpy()
    np.testing.assert_array_equal(pre.transform(test), pre.transform(scrambled))


def test_treatment_only_on_request(split_frames, cfg: Config) -> None:
    train, _ = split_frames
    names = build_preprocessor(cfg, include_treatment=True).fit(train).get_feature_names_out()
    trt = [n for n in names if "treatment" in n]
    assert len(trt) == 2 and not any(cfg.reference_arm in n for n in trt)


def test_imputer_fitted_on_train_only(split_frames, cfg: Config) -> None:
    """Imputation statistics must come from training rows: poisoning test rows changes nothing."""
    train, test = split_frames
    pre = build_preprocessor(cfg, imputer="median").fit(train)
    numeric = pre.named_transformers_["numeric"]
    medians = numeric.named_steps["impute"].statistics_.copy()
    means = numeric.named_steps["scale"].mean_.copy()
    poisoned = test.copy()
    poisoned["X_age"] = 999
    pre.transform(poisoned)  # transforming test rows must not update anything
    np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, medians)
    np.testing.assert_array_equal(numeric.named_steps["scale"].mean_, means)
    # and fitting on train vs train+test gives different statistics (the check is not vacuous)
    pre3 = build_preprocessor(cfg, imputer="median").fit(pd.concat([train, poisoned]))
    assert not np.allclose(pre3.named_transformers_["numeric"].named_steps["scale"].mean_,
                           means)


def test_harmonizer_zscore_and_offset() -> None:
    X = pd.DataFrame({"cst": [300.0, 400.0, 500.0, 343.0, 443.0, np.nan],
                      "machine": ["A", "A", "A", "B", "B", "B"]})
    z = CSTHarmonizer("zscore").fit(X).transform(X)[:, 0]
    np.testing.assert_allclose(z[:3], [-1.2247, 0.0, 1.2247], atol=1e-4)
    assert np.isnan(z[5])
    off = CSTHarmonizer("offset", offset_um={"B": 43.0}).fit(X).transform(X)[:, 0]
    np.testing.assert_allclose(off[3:5], [300.0, 400.0])
    with pytest.raises(ValueError, match="equations"):
        CSTHarmonizer("equations", equations={"B": {"intercept": None, "slope": None}}).fit(X)


def test_rubin_rules_known_values() -> None:
    est = np.array([1.0, 2.0, 3.0])
    var = np.array([0.5, 0.5, 0.5])
    q, se, dof, lam = rubin_pool(est, var, df_complete=1e9)
    assert q == 2.0
    # T = W + (1 + 1/m) B = 0.5 + 4/3 * 1 = 1.8333
    assert se == pytest.approx(np.sqrt(0.5 + 4 / 3))
    assert lam == pytest.approx((4 / 3) / (0.5 + 4 / 3))
    assert dof == pytest.approx(2 / lam**2, rel=1e-6)


def test_missing_strategies_run(synthetic: pd.DataFrame, cfg: Config) -> None:
    small = replace(cfg, preprocess={**cfg.preprocess, "mice_imputations": 3})
    out = compare_strategies(synthetic, small)
    assert set(out.strategy.str.split(" ").str[0]) == {"complete-case", "kNN", "MICE"}
    assert (out.ci_low < out.estimate).all() and (out.estimate < out.ci_high).all()
    cc = out[out.strategy == "complete-case"]
    assert (cc.n == synthetic[cfg.primary_outcome].notna().sum()).all()


def test_outliers_flag_but_do_not_drop(split_frames, synthetic: pd.DataFrame,
                                       cfg: Config) -> None:
    train, _ = split_frames
    res = detect_outliers(train, synthetic, cfg)
    assert len(res.flags) == len(synthetic)
    ov = overlap_table(res.flags).set_index("methods")
    assert ov.loc["any", "n_flagged"] >= ov.loc["iqr", "n_flagged"]


def test_discretize_bands(synthetic: pd.DataFrame, cfg: Config) -> None:
    bands = discretize(synthetic, cfg)
    va = bands["X_va_letters_band"]
    below = synthetic.X_va_letters < 69
    assert (va[below] != "20/32-20/40").all() and (va[~below] == "20/32-20/40").all()
    g = gain_item(synthetic, cfg)
    assert g.isna().sum() == synthetic[cfg.primary_outcome].isna().sum()


def test_cst_quality_policy(synthetic: pd.DataFrame, cfg: Config) -> None:
    from equipoise.data.load import load_cst_quality

    q = load_cst_quality(cfg, synthetic, synthetic=True)
    keep = apply_cst_quality(synthetic, q, "keep")
    pd.testing.assert_frame_equal(keep, synthetic)
    miss = apply_cst_quality(synthetic, q, "missing")
    flagged = synthetic.pt_id.isin(q.loc[q.M_cst_flag_baseline == 1, "pt_id"])
    assert miss.loc[flagged, "X_cst_um"].isna().all()
    assert miss.X_va_letters.equals(synthetic.X_va_letters)


def test_build_preprocessor_rejects_leaky_columns(cfg: Config, monkeypatch) -> None:
    from equipoise.preprocess import encode

    real = encode.feature_groups

    def leaky(c):
        g = real(c)
        return replace(g, continuous=(*g.continuous, "Y_va_change_wk52"))

    monkeypatch.setattr(encode, "feature_groups", leaky)
    with pytest.raises(LeakageError):
        encode.build_preprocessor(cfg)


def test_synthetic_runs_write_to_separate_reports(cfg: Config) -> None:
    syn = for_data_source(cfg, synthetic=True)
    assert "synthetic" in syn.paths.tables_dir.parts
    assert syn.paths.tables_dir != cfg.paths.tables_dir
