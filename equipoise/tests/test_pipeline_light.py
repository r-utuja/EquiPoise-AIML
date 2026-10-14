"""Light end-to-end run on synthetic data: causal (bundle) -> decision -> explain.

The session-scoped `light` / `bundle` fixtures (conftest.py) fit the light M6 pipeline once;
the decision layer, agent and explain stage are then tested against that bundle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equipoise.config import Config, load_prices


def _outputs(light: Config, code: str) -> tuple[list[Path], list[Path]]:
    return (list(light.paths.figures_dir.glob(f"{code}_*.png")),
            list(light.paths.tables_dir.glob(f"{code}_*.csv")))


def test_causal_outputs_and_bundle(light: Config, bundle, synthetic: pd.DataFrame) -> None:
    pngs, csvs = _outputs(light, "m6")
    assert pngs and csvs and (light.paths.metrics_dir / "m6_causal.json").is_file()
    out = bundle.predict(synthetic.head(5))
    assert out["effect"].shape == (5, 2) and out["letters"].shape == (5, 3)
    assert ((out["p_benefit_gt_mcid"] >= 0) & (out["p_benefit_gt_mcid"] <= 1)).all()
    assert (out["effect_low"] <= out["effect_high"]).all()
    assert out["injections_yr1"].shape == (5, 3) and (out["p_laser_yr1"] <= 1).all()


def test_recommend_and_agent(light: Config, bundle, synthetic: pd.DataFrame) -> None:
    from equipoise.decision.agent import EquipoiseAgent
    from equipoise.decision.recommend import recommend

    patient = {c: synthetic.iloc[0][c] for c in synthetic.columns if c.startswith("X_")}
    prices = load_prices()
    r = recommend(patient, prices, bundle, light)
    assert r.recommendation in {"Bevacizumab", "Aflibercept", "Ranibizumab", "Discuss"}
    assert [e.drug for e in r.estimates] == bundle.arms
    assert r.estimates[0].letters_low is None and r.estimates[1].benefit_low is not None
    assert r.prices_are_demo and any("DEMO" in w for w in r.warnings)
    assert all(0 <= v <= 100 for v in r.fuzzy_strength.values())
    assert r.rule_trace and "not clinical guidance" in r.rule_trace[0]
    assert recommend(patient, prices, bundle, light).rule_trace == r.rule_trace
    agent = EquipoiseAgent(light, prices, bundle).perceive({**patient, "X_hba1c": None})
    agent.decide()
    text = agent.explain()
    assert "Recommendation:" in text and "hba1c" in text      # imputed feature is reported
    with pytest.raises(ValueError, match="X_va_letters"):
        recommend({k: v for k, v in patient.items() if k != "X_va_letters"}, prices, bundle,
                  light)


def test_decision_stage(light: Config, bundle) -> None:
    from equipoise.decision.run import run

    metrics = run(light, synthetic=True)
    pngs, csvs = _outputs(light, "m8")
    assert pngs and csvs and 0 <= metrics["share_equipoise"] <= 1
    check = pd.read_csv(light.paths.tables_dir / "m8_newton_vs_brentq.csv").set_index("check")
    assert float(check.loc["benefit: newton vs brentq", "value"]) < 1e-6


def test_explain_stage(light: Config, bundle) -> None:
    from equipoise.explain.run import run

    metrics = run(light, synthetic=True)
    pngs, csvs = _outputs(light, "m9")
    assert pngs and csvs and metrics["shap_additivity_ok"] == 1.0
    rep = pd.read_csv(light.paths.tables_dir / "m9_subgroups.csv")
    assert {"female", "race", "diabetes_type", "va_below_69"} <= set(rep.subgroup)
    assert np.isfinite(rep.letters_rmse).all()
