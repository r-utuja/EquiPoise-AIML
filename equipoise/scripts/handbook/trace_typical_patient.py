"""Handbook helper: follow the "typical patient" through the whole serving path, printing every
intermediate value (handbook chapters 16 and 22).

The typical patient is built from training medians (numbers) and modes (categories), exactly
as `equipoise.decision.run.typical_patient` does: it is an aggregate, not a real person.
Steps printed: request validation (pydantic) -> one-row frame -> preprocessed features ->
bundle predictions (effects, bootstrap interval, letters, injections, laser) -> expected cost
-> fuzzy inputs, memberships, rule firing, centroid -> expert-system trace -> the app's
headline and cards (presenters). Read-only: it loads the trained bundle and changes nothing.

Usage:
    python scripts/handbook/trace_typical_patient.py              # real bundle (models/)
    python scripts/handbook/trace_typical_patient.py --synthetic  # synthetic bundle
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so `app` (the Streamlit package) is importable

from app import presenters  # noqa: E402
from equipoise.causal.bundle import load_bundle  # noqa: E402
from equipoise.config import load_config, load_prices  # noqa: E402
from equipoise.data.load import load_analysis  # noqa: E402
from equipoise.data.splits import get_split  # noqa: E402
from equipoise.decision import fuzzy  # noqa: E402
from equipoise.decision.cost import expected_cost  # noqa: E402
from equipoise.decision.recommend import (  # noqa: E402
    default_bundle_path,
    patient_frame,
    recommend,
    serving_mode,
)
from equipoise.decision.run import typical_patient  # noqa: E402
from equipoise.serving.schemas import PatientFeatures  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    df = load_analysis(cfg, synthetic=args.synthetic)
    train = get_split(cfg, df, args.synthetic).train(df)
    typ = {k: (v.item() if hasattr(v, "item") else v) for k, v in typical_patient(train).items()}

    section("1. typical patient (training medians / modes)")
    for k, v in typ.items():
        print(f"  {k}: {v}")

    section("2. pydantic validation (serving.schemas.PatientFeatures)")
    req = PatientFeatures(**typ)
    print(f"  valid; {len(req.provided())} fields provided")

    section("3. one-row frame -> preprocessor")
    bundle = serving_mode(load_bundle(default_bundle_path(cfg, args.synthetic)))
    frame = patient_frame(req.features())
    X = bundle.preprocessor.transform(frame)
    names = [n.split("__", 1)[1] for n in bundle.preprocessor.get_feature_names_out()]
    print(f"  preprocessed shape {X.shape}")
    for n, v in zip(names, np.ravel(X), strict=True):
        if n in ("X_va_letters", "X_cst_harmonized_zscore", "X_hba1c", "X_age"):
            print(f"  {n}: {v:.3f}")

    section("4. bundle.predict")
    out = bundle.predict(frame)
    arms = list(bundle.arms)
    print(f"  effect model: {bundle.effect_model_name}; bootstrap refits: "
          f"{len(bundle.bootstrap_models)}")
    for k, arm in enumerate(arms[1:]):
        print(f"  {arm} - {arms[0]}: effect {out['effect'][0, k]:.3f}, 95% interval "
              f"{out['effect_low'][0, k]:.3f} to {out['effect_high'][0, k]:.3f}, "
              f"P(>MCID) {out['p_benefit_gt_mcid'][0, k]:.3f}")
    for j, arm in enumerate(arms):
        print(f"  {arm}: letters {out['letters'][0, j]:.3f}, injections "
              f"{out['injections_yr1'][0, j]:.3f}, P(laser) {out['p_laser_yr1'][0, j]:.3f}")

    section("5. expected cost")
    prices = load_prices()
    cost = expected_cost(out["injections_yr1"], out["p_laser_yr1"], arms, prices)
    print(f"  prices: {prices.label}")
    for j, arm in enumerate(arms):
        print(f"  {arm}: {cost[0, j]:.3f}")

    section("6. fuzzy system for aflibercept")
    fs = fuzzy.build(cfg)
    j = arms.index("Aflibercept")
    inputs = {"benefit": float(out["effect"][0, j - 1]),
              "certainty": float(out["p_benefit_gt_mcid"][0, j - 1]),
              "extra_cost": max(0.0, (cost[0, j] - cost[0, 0]) / cost[0, j]),
              "burden": float(out["injections_yr1"][0, j] - out["injections_yr1"][0, 0])}
    for k, v in inputs.items():
        var = fs.variables[k]
        lo, hi = fs.bounds[k]
        x = float(np.clip(v, lo, hi))
        mem = {t: float(np.interp(x, var.universe, m.mf)) for t, m in var.terms.items()}
        print(f"  {k} = {v:.4f} -> memberships " + ", ".join(f"{t} {m:.3f}" for t, m in
                                                             mem.items()))
    s = fuzzy.strength(fs, **inputs)
    agg = fuzzy.aggregated_output(fs)
    out_var = fs.variables["strength"]
    for t, term in out_var.terms.items():
        print(f"  output '{t}' clipped at {float(term.membership_value[fs.sim]):.3f}")
    print(f"  aggregated area {np.trapezoid(agg, out_var.universe):.3f}; "
          f"centroid strength {s:.3f}")

    section("7. recommend(): expert-system trace")
    rec = recommend(req.features(), prices, bundle, cfg)
    print(f"  recommendation: {rec.recommendation} (equipoise {rec.equipoise}); "
          f"sensitivity {rec.equipoise_at_sensitivity}")
    for line in rec.rule_trace:
        print(f"  {line}")
    for b in rec.break_even:
        print(f"  break-even {b.drug}: price {b.break_even_price}, benefit "
              f"{b.break_even_benefit_letters} ({b.message})")

    section("8. presenters (what the app shows)")
    pred = rec.model_dump()
    ui = cfg.raw["ui"]
    head = presenters.headline(pred, ui, typ["X_va_letters"])
    print(f"  headline [{head.kind}]: {head.text} {head.confidence} "
          f"(level {head.confidence_level:.3f})")
    if head.secondary:
        print(f"  secondary: {head.secondary}")
    print(f"  trial finding: {head.trial_finding}")
    for c in presenters.cards(pred, ui):
        print(f"  card {c.name}: {c.improvement} ({c.improvement_letters}); {c.range_line}; "
              f"{c.injections}; tag={c.tag}; cost={c.cost}")

    section("9. as the app does it (default form patient, LocalClient, explain=True)")
    from equipoise.serving.client import LocalClient
    from equipoise.serving.service import Service

    svc = Service.from_path(default_bundle_path(cfg, args.synthetic), cfg, log_path=None)
    form = {k: v for k, v in svc.defaults().items() if k != "X_va_below_69"}
    snellen = presenters.nearest_snellen(form["X_va_letters"], ui)
    form["X_va_letters"] = presenters.snellen_to_letters(snellen)
    print(f"  form vision: 20/{snellen} -> {form['X_va_letters']} letters "
          f"(defaults had {svc.defaults()['X_va_letters']})")
    app_pred = LocalClient(svc).predict({"patient": form, "prices": None, "explain": True})
    e = {x["drug"]: x for x in app_pred["estimates"]}["Aflibercept"]
    print(f"  Eylea vs Avastin: {e['benefit_vs_reference']:.3f} "
          f"({e['benefit_low']:.3f} to {e['benefit_high']:.3f}), "
          f"P(>MCID) {e['p_benefit_gt_mcid']:.3f}; imputed: {app_pred['imputed_features']}")
    ex = app_pred["explanation"]
    print(f"  SHAP base {ex['base_value_letters']:.3f} -> prediction "
          f"{ex['prediction_letters']:.3f}; top contributions:")
    for c in ex["contributions"][:5]:
        print(f"    {c['feature']} = {c['value']}: {c['shap_letters']:+.3f}")
    head = presenters.headline(app_pred, ui, form["X_va_letters"])
    print(f"  headline: {head.text} {head.confidence}")
    print(f"  trial finding: {head.trial_finding}")
    for c in presenters.cards(app_pred, ui):
        print(f"  card {c.name}: {c.improvement} ({c.improvement_letters}); {c.range_line}; "
              f"{c.injections}; tag={c.tag}; cost={c.cost}")
    print(f"  chart summary: {presenters.chart(app_pred, ui).summary}")
    binary ={f["name"] for f in svc.metadata()["features"] if f["binary"]}
    w = presenters.why(app_pred, ui, form["X_va_letters"], svc.defaults(), binary)
    for f in w.factors:
        print(f"  factor: {f}")
    for d in w.decided:
        print(f"  decided: {d}")
    print(f"  strength: {w.strength}")


if __name__ == "__main__":
    main()
