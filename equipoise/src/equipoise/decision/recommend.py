"""`recommend(patient, prices)`: the single function the API and the app call.

It scores one patient with the model bundle (letters, injections and laser probability
under each drug, with bootstrap intervals), adds expected cost and cost per extra letter,
the break-even price and break-even benefit (Newton-Raphson), the fuzzy recommendation
strength and the expert-system reasoning trace, and returns everything as one validated
`Recommendation`. Decision support for a course project, not clinical guidance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from equipoise.causal.bundle import BUNDLE_NAME, ModelBundle, load_bundle
from equipoise.config import Config, Prices, load_config, load_prices
from equipoise.data.schema import ANALYSIS_COLUMNS
from equipoise.decision.breakeven import break_even
from equipoise.decision.cost import cost_per_letter, expected_cost
from equipoise.decision.expert import Engine, default_rules
from equipoise.decision.fuzzy import build as build_fuzzy
from equipoise.decision.fuzzy import strength as fuzzy_strength

DISCLAIMER = "Research prototype for a college AIML project. Not for clinical use."
FEATURES = [c.name for c in ANALYSIS_COLUMNS if c.role == "feature"]


class DrugEstimate(BaseModel):
    drug: str
    letters: float = Field(description="predicted letters gained at week 52")
    letters_low: float | None = Field(None, description="None for the reference arm: the "
                                      "bootstrap covers effects vs the reference only")
    letters_high: float | None = None
    benefit_vs_reference: float | None = None
    benefit_low: float | None = None
    benefit_high: float | None = None
    p_benefit_gt_mcid: float | None = None
    injections_yr1: float
    p_laser_yr1: float
    cost: float
    cost_per_extra_letter: float | None = None
    net_benefit: float | None = None


class BreakEvenOut(BaseModel):
    drug: str
    break_even_price: float | None
    break_even_benefit_letters: float | None
    newton_iterations_price: int
    newton_iterations_benefit: int
    message: str


class Recommendation(BaseModel):
    recommendation: str
    reason: str
    equipoise: bool
    equipoise_at_sensitivity: dict[str, bool]
    mcid_letters: float
    estimates: list[DrugEstimate]
    fuzzy_strength: dict[str, float]
    break_even: list[BreakEvenOut]
    rule_trace: list[str]
    published_rule_suggests: str
    prices_label: str
    prices_are_demo: bool
    warnings: list[str]
    model: str
    disclaimer: str = DISCLAIMER


def serving_mode(bundle: ModelBundle) -> ModelBundle:
    """Predict single patients fast: run forest predictions single-threaded (spinning up a
    parallel pool costs far more than scoring one row). Does not change any prediction."""
    for m in [bundle.effect_model, *bundle.bootstrap_models]:
        cate = getattr(getattr(m, "est_", None), "model_cate", None)
        for sub in getattr(cate, "estimators_", []):
            if hasattr(sub, "n_jobs"):
                sub.n_jobs = 1
    return bundle


@lru_cache(maxsize=4)
def _cached_bundle(path: str) -> ModelBundle:
    return serving_mode(load_bundle(Path(path)))


@lru_cache(maxsize=1)
def _cached_fuzzy(cfg_key: str):
    return build_fuzzy(load_config())


def default_bundle_path(cfg: Config, synthetic: bool = False) -> Path:
    return cfg.paths.models_dir / ("synthetic" if synthetic else "") / BUNDLE_NAME


def patient_frame(patient: dict[str, Any]) -> pd.DataFrame:
    """One-row frame with every baseline feature (missing ones as NaN); X_va_below_69 is
    derived from X_va_letters when not given."""
    row = {f: patient.get(f, np.nan) for f in FEATURES}
    if pd.isna(row["X_va_below_69"]) and not pd.isna(row["X_va_letters"]):
        row["X_va_below_69"] = int(float(row["X_va_letters"]) < 69)
    df = pd.DataFrame([row])
    for c in ANALYSIS_COLUMNS:
        if c.role != "feature":
            continue
        if c.dtype == "str":   # keep text columns as object even when the value is missing
            df[c.name] = df[c.name].astype(object).where(df[c.name].notna(), np.nan)
        else:
            df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
    return df


def recommend(patient: dict[str, Any], prices: Prices | None = None,
              bundle: ModelBundle | None = None, cfg: Config | None = None) -> Recommendation:
    """Score one patient under every drug and return the full, explained recommendation."""
    cfg = cfg or load_config()
    prices = prices or load_prices()
    bundle = bundle or _cached_bundle(str(default_bundle_path(cfg)))
    fs = _cached_fuzzy("default")
    d = cfg.decision
    k = float(d["value_of_vision_k"])
    mcid = float(bundle.mcid_letters)
    arms = list(bundle.arms)
    ref = arms[0]
    X = patient_frame(patient)
    if X.X_va_letters.isna().iloc[0]:
        raise ValueError("X_va_letters (baseline study-eye vision) is required")
    out = bundle.predict(X)
    letters = out["letters"]
    cost = expected_cost(out["injections_yr1"], out["p_laser_yr1"], arms, prices)
    cpl = cost_per_letter(cost, letters)
    wtp = prices.willingness_to_pay

    estimates, be_out, strengths = [], [], {}
    for j, arm in enumerate(arms):
        est = DrugEstimate(
            drug=arm, letters=float(letters[0, j]),
            letters_low=None if j == 0 else float(out["letters_low"][0, j]),
            letters_high=None if j == 0 else float(out["letters_high"][0, j]),
            injections_yr1=float(out["injections_yr1"][0, j]),
            p_laser_yr1=float(out["p_laser_yr1"][0, j]), cost=float(cost[0, j]),
            cost_per_extra_letter=None if np.isnan(cpl[0, j]) else float(cpl[0, j]),
            net_benefit=None if wtp is None else float(
                wtp * (1 - np.exp(-k * letters[0, j])) - cost[0, j]))
        if j > 0:
            e = j - 1
            est.benefit_vs_reference = float(out["effect"][0, e])
            est.benefit_low = float(out["effect_low"][0, e])
            est.benefit_high = float(out["effect_high"][0, e])
            est.p_benefit_gt_mcid = float(out["p_benefit_gt_mcid"][0, e])
            extra = max(0.0, (cost[0, j] - cost[0, 0]) / cost[0, j]) if cost[0, j] > 0 else 0.0
            strengths[arm] = fuzzy_strength(
                fs, est.benefit_vs_reference, est.p_benefit_gt_mcid, extra,
                float(out["injections_yr1"][0, j] - out["injections_yr1"][0, 0]))
            if wtp is not None:
                n = d["newton"]
                be = break_even(letters[0, 0], letters[0, j], out["injections_yr1"][0, 0],
                                out["injections_yr1"][0, j], out["p_laser_yr1"][0, 0],
                                out["p_laser_yr1"][0, j], prices.per_injection[ref],
                                prices.per_injection[arm], prices.laser_session, wtp, k,
                                float(n["tol"]), int(n["max_iter"]), float(n["fd_step"]))
                be_out.append(BreakEvenOut(
                    drug=arm, break_even_price=be.price,
                    break_even_benefit_letters=be.benefit_letters,
                    newton_iterations_price=be.price_newton.iterations,
                    newton_iterations_benefit=be.benefit_newton.iterations,
                    message=f"price: {be.price_newton.message}; benefit: "
                            f"{be.benefit_newton.message}"))
        estimates.append(est)

    # MCID sensitivity from the bootstrap intervals (lower bound vs each threshold)
    sens = {}
    for m in cfg.raw["effects"].get("mcid_sensitivity_letters", []):
        sens[f"{m:g}"] = bool(np.all(out["effect_low"][0] < float(m)))

    comps = estimates[1:]
    best = max(comps, key=lambda e: e.benefit_vs_reference)
    nb_gain = (None if best.net_benefit is None
               else best.net_benefit - estimates[0].net_benefit)
    be_best = next((b for b in be_out if b.drug == best.drug), None)
    facts: dict[str, Any] = {
        "mcid": mcid, "baseline_va_letters": float(X.X_va_letters.iloc[0]),
        "prices_are_demo": prices.is_demo, "best_drug": best.drug,
        "best_net_benefit_gain": nb_gain if nb_gain is not None else 0.0,
        "net_benefit_margin": float(d["rules"]["net_benefit_margin"]),
        "break_even_price": (f"{be_best.break_even_price:.3g}"
                             if be_best and be_best.break_even_price is not None else "n/a"),
        "equipoise_at_sensitivity": {float(m): v for m, v in sens.items()} or None,
    }
    for e in comps:
        facts[f"effect_low_{e.drug}"] = e.benefit_low
        facts[f"effect_high_{e.drug}"] = e.benefit_high
    engine = Engine(default_rules())
    wm = engine.run(facts)

    warnings = [DISCLAIMER]
    if prices.is_demo:
        warnings.append("DEMO prices in use: costs are relative units, not rupees.")
    if wtp is None:
        warnings.append("Willingness to pay not set: break-even outputs disabled.")
    warnings.append("Per-patient differences between drugs are weakly estimated (wide "
                    "intervals); see the published subgroup rule.")
    return Recommendation(
        recommendation=wm["recommendation"], reason=wm["reason"],
        equipoise=bool(wm.get("equipoise")), equipoise_at_sensitivity=sens, mcid_letters=mcid,
        estimates=estimates, fuzzy_strength=strengths, break_even=be_out,
        rule_trace=engine.trace, published_rule_suggests=wm["published_rule_suggests"],
        prices_label=prices.label, prices_are_demo=prices.is_demo, warnings=warnings,
        model=bundle.effect_model_name)
