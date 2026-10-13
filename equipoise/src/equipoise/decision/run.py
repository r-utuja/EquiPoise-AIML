"""M8 stage: apply the decision layer to held-out patients and illustrate each method.

Population summaries (recommendations, equipoise at each MCID, costs, break-even values,
Newton vs brentq agreement) use held-out patients and are reported only in aggregate. The
worked examples (fuzzy output, break-even curve, rule trace) use a 'typical patient' built
from training medians / modes - not any real individual.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equipoise import tracking
from equipoise.causal.bundle import load_bundle
from equipoise.config import Config, load_prices
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.schema import ANALYSIS_COLUMNS
from equipoise.data.splits import get_split
from equipoise.decision.agent import EquipoiseAgent
from equipoise.decision.breakeven import break_even, value_of_vision
from equipoise.decision.fuzzy import RULE_TEXT, aggregated_output, build, strength
from equipoise.decision.recommend import default_bundle_path, recommend, serving_mode
from equipoise.plotting import INK, INK_2, NEUTRAL, SERIES

STAGE = "decision"


def typical_patient(train: pd.DataFrame) -> dict:
    """Training medians (numeric) and modes (categorical / binary)."""
    out = {}
    for c in ANALYSIS_COLUMNS:
        if c.role != "feature":
            continue
        s = train[c.name].dropna()
        out[c.name] = (s.mode().iloc[0] if c.categories or c.binary else float(s.median()))
    out["X_va_below_69"] = int(out["X_va_letters"] < 69)
    return out


def _mf_plot(fs) -> plt.Figure:
    names = ["benefit", "certainty", "extra_cost", "burden", "strength"]
    units = {"benefit": "letters vs cheapest", "certainty": "P(benefit > MCID)",
             "extra_cost": "share of cost that is extra", "burden": "injections vs cheapest",
             "strength": "recommendation strength (0-100)"}
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.3))
    for ax, n in zip(axes, names, strict=True):
        var = fs.variables[n]
        for i, (term, mf) in enumerate(var.terms.items()):
            ax.plot(var.universe, mf.mf, color=SERIES[i], lw=2, label=term)
        ax.set_title(n.replace("_", " "))
        ax.set_xlabel(units[n], fontsize=8)
        ax.set_ylim(-0.02, 1.08)
        ax.legend(fontsize=7, loc="center right")
    axes[0].set_ylabel("Membership degree")
    fig.suptitle("Fuzzy membership functions", fontweight="bold")
    fig.tight_layout()
    return fig


def _worked_example(fs, inputs: dict) -> tuple[plt.Figure, float]:
    s = strength(fs, **inputs)
    agg = aggregated_output(fs)
    var = fs.variables["strength"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (term, mf) in enumerate(var.terms.items()):
        ax.plot(var.universe, mf.mf, color=SERIES[i], lw=1, ls="--", label=term)
    ax.fill_between(var.universe, 0, agg, color=SERIES[0], alpha=0.35, label="aggregated output")
    ax.axvline(s, color=INK, lw=2)
    ax.text(s, 1.02, f" centroid = {s:.0f}", color=INK, fontsize=9)
    ax.set_xlabel("Recommendation strength for the dearer drug (0-100)")
    ax.set_ylabel("Membership")
    txt = ", ".join(f"{k.replace('_', ' ')} = {v:.2f}" for k, v in inputs.items())
    ax.set_title(f"Worked example: {txt}", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig, s


def _breakeven_plot(rec, prices, k: float) -> plt.Figure:
    ref, afl = rec.estimates[0], next(e for e in rec.estimates if e.drug == "Aflibercept")
    wtp = prices.willingness_to_pay
    gain = wtp * (value_of_vision(afl.letters, k) - value_of_vision(ref.letters, k))
    p = np.linspace(0, 2 * prices.per_injection["Aflibercept"], 200)
    g = gain - (afl.injections_yr1 * p + afl.p_laser_yr1 * prices.laser_session - ref.cost)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(p, g, color=SERIES[0])
    axes[0].axhline(0, color=NEUTRAL, lw=1)
    be = next(b for b in rec.break_even if b.drug == "Aflibercept")
    if be.break_even_price is not None:
        axes[0].axvline(be.break_even_price, color=SERIES[1], ls="--")
        axes[0].text(be.break_even_price, axes[0].get_ylim()[1] * 0.9,
                     f" break-even = {be.break_even_price:.2f}", color=INK_2, fontsize=9)
    axes[0].axvline(prices.per_injection["Aflibercept"], color=NEUTRAL, ls=":")
    axes[0].set_xlabel(f"Aflibercept price per injection ({prices.label})")
    axes[0].set_ylabel("Net benefit: aflibercept minus bevacizumab")
    axes[0].set_title("Break-even price (linear in price: Newton exact in one step)")
    d = np.linspace(0, 40, 300)
    d_cost = afl.cost - ref.cost
    h = wtp * (value_of_vision(ref.letters + d, k) - value_of_vision(ref.letters, k)) - d_cost
    axes[1].plot(d, h, color=SERIES[0])
    axes[1].axhline(0, color=NEUTRAL, lw=1)
    r = break_even(ref.letters, afl.letters, ref.injections_yr1, afl.injections_yr1,
                   ref.p_laser_yr1, afl.p_laser_yr1, prices.per_injection["Bevacizumab"],
                   prices.per_injection["Aflibercept"], prices.laser_session, wtp, k)
    for i, x in enumerate(r.benefit_newton.path):
        y = wtp * (value_of_vision(ref.letters + x, k) - value_of_vision(ref.letters, k)) - d_cost
        axes[1].scatter([x], [y], color=SERIES[1], s=25, zorder=3)
        if i < 3:   # later iterates sit on top of the root
            axes[1].annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4),
                             fontsize=8)
    axes[1].set_xlabel("Extra letters with aflibercept vs bevacizumab")
    axes[1].set_ylabel("Value of extra letters minus extra cost")
    axes[1].set_title(f"Break-even benefit: Newton iterates (root "
                      f"{r.benefit_letters:.2f} letters)" if r.benefit_letters is not None
                      else "Break-even benefit: no root")
    fig.suptitle("Typical training patient (illustrative, DEMO prices)" if prices.is_demo
                 else "Typical training patient", fontweight="bold")
    fig.tight_layout()
    return fig


def _strength_plot(df: pd.DataFrame, mcid: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, drug in enumerate(["Aflibercept", "Ranibizumab"]):
        ax.scatter(df[f"benefit_{drug}"], df[f"strength_{drug}"], s=16, color=SERIES[i],
                   alpha=0.7, label=drug, linewidths=0)
    ax.axvline(mcid, color=NEUTRAL, ls="--", lw=1)
    ax.set_xlabel("Predicted benefit vs bevacizumab (letters; dashed = MCID)")
    ax.set_ylabel("Fuzzy recommendation strength")
    ax.set_title("Held-out patients: predicted benefit vs fuzzy strength")
    ax.legend()
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    train, test = split.train(df), split.test(df)
    prices = load_prices()
    bundle = serving_mode(load_bundle(default_bundle_path(cfg, synthetic)))
    k = float(cfg.decision["value_of_vision_k"])
    feats = [c.name for c in ANALYSIS_COLUMNS if c.role == "feature"]
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"decision": cfg.decision, "prices": prices.label,
                             "willingness_to_pay": prices.willingness_to_pay})
        fs = build(cfg)
        tracking.log_table(cfg, STAGE, pd.DataFrame({"rule": range(1, len(RULE_TEXT) + 1),
                                                     "text": RULE_TEXT}), "fuzzy_rules")
        tracking.log_figure(cfg, STAGE, _mf_plot(fs), "fuzzy_membership")

        # typical patient: worked examples ---------------------------------------------
        typ = typical_patient(train)
        agent = EquipoiseAgent(cfg, prices, bundle).perceive(typ)
        rec = agent.decide()
        afl = next(e for e in rec.estimates if e.drug == "Aflibercept")
        inputs = {"benefit": afl.benefit_vs_reference, "certainty": afl.p_benefit_gt_mcid,
                  "extra_cost": max(0.0, (afl.cost - rec.estimates[0].cost) / afl.cost),
                  "burden": afl.injections_yr1 - rec.estimates[0].injections_yr1}
        fig, s = _worked_example(fs, inputs)
        tracking.log_figure(cfg, STAGE, fig, "fuzzy_worked_example")
        if prices.willingness_to_pay is not None:
            tracking.log_figure(cfg, STAGE, _breakeven_plot(rec, prices, k), "break_even")
        tracking.log_table(cfg, STAGE, pd.DataFrame({"step": agent.explain().splitlines()}),
                           "typical_patient_explanation")
        tracking.log_table(cfg, STAGE, pd.DataFrame([{"feature": f, "value": v}
                                                     for f, v in typ.items()]),
                           "typical_patient_profile")

        # held-out patients (aggregate only) ------------------------------------------
        rows, nb = [], []
        for _, r in test.iterrows():
            out = recommend({f: r[f] for f in feats}, prices, bundle, cfg)
            row = {"recommendation": out.recommendation, "equipoise": out.equipoise,
                   **{f"equipoise_mcid_{m}": v for m, v in out.equipoise_at_sensitivity.items()},
                   "published_rule": out.published_rule_suggests}
            for e in out.estimates:
                row[f"cost_{e.drug}"] = e.cost
                row[f"letters_{e.drug}"] = e.letters
                row[f"injections_{e.drug}"] = e.injections_yr1
                row[f"laser_{e.drug}"] = e.p_laser_yr1
                if e.benefit_vs_reference is not None:
                    row[f"benefit_{e.drug}"] = e.benefit_vs_reference
                    row[f"strength_{e.drug}"] = out.fuzzy_strength[e.drug]
                    row[f"cost_per_letter_{e.drug}"] = e.cost_per_extra_letter
            for b in out.break_even:
                row[f"break_even_price_{b.drug}"] = b.break_even_price
                row[f"break_even_benefit_{b.drug}"] = b.break_even_benefit_letters
            rows.append(row)
            # Newton (analytic and finite difference) vs brentq, per patient and drug
            ests = {e.drug: e for e in out.estimates}
            ref = out.estimates[0]
            if prices.willingness_to_pay is not None:
                for drug in ("Aflibercept", "Ranibizumab"):
                    e = ests[drug]
                    be = break_even(ref.letters, e.letters, ref.injections_yr1,
                                    e.injections_yr1, ref.p_laser_yr1, e.p_laser_yr1,
                                    prices.per_injection[ref.drug], prices.per_injection[drug],
                                    prices.laser_session, prices.willingness_to_pay, k)
                    nb.append({"drug": drug, "price_newton": be.price_newton.root,
                               "price_newton_fd": be.price_newton_fd.root,
                               "price_brentq": be.price_brentq,
                               "price_iterations": be.price_newton.iterations,
                               "benefit_newton": be.benefit_newton.root,
                               "benefit_newton_fd": be.benefit_newton_fd.root,
                               "benefit_brentq": be.benefit_brentq,
                               "benefit_iterations": be.benefit_newton.iterations})
        pop = pd.DataFrame(rows)
        summary = {"n_heldout": len(pop), "prices": prices.label}
        for drug in ("Bevacizumab", "Aflibercept", "Ranibizumab", "Discuss"):
            summary[f"share_recommend_{drug}"] = float((pop.recommendation == drug).mean())
        summary["share_equipoise_mcid_5"] = float(pop.equipoise.mean())
        for c in [c for c in pop.columns if c.startswith("equipoise_mcid_")]:
            summary[f"share_{c}"] = float(pop[c].mean())
        summary["share_published_rule_aflibercept"] = float(
            (pop.published_rule == "Aflibercept").mean())
        for drug in ("Aflibercept", "Ranibizumab"):
            summary[f"median_strength_{drug}"] = float(pop[f"strength_{drug}"].median())
            summary[f"median_cost_per_letter_{drug}"] = float(
                pop[f"cost_per_letter_{drug}"].median())
            if f"break_even_price_{drug}" in pop:
                summary[f"median_break_even_price_{drug}"] = float(
                    pop[f"break_even_price_{drug}"].median())
                summary[f"share_break_even_benefit_no_root_{drug}"] = float(
                    pop[f"break_even_benefit_{drug}"].isna().mean())
        tracking.log_table(cfg, STAGE, pd.DataFrame([summary]).T.reset_index().rename(
            columns={"index": "quantity", 0: "value"}), "population_summary")
        per_drug = pd.DataFrame([{
            "drug": d, "mean_letters": pop[f"letters_{d}"].mean(),
            "mean_injections_yr1": pop[f"injections_{d}"].mean(),
            "mean_p_laser_yr1": pop[f"laser_{d}"].mean(), "mean_cost": pop[f"cost_{d}"].mean(),
            "prices": prices.label} for d in bundle.arms])
        tracking.log_table(cfg, STAGE, per_drug, "cost_by_drug")
        tracking.log_figure(cfg, STAGE, _strength_plot(pop, cfg.mcid_letters),
                            "strength_vs_benefit")
        if nb:
            nbd = pd.DataFrame(nb)

            def maxdiff(a: str, b: str) -> float:
                both = nbd[[a, b]].dropna()
                return float((both[a] - both[b]).abs().max()) if len(both) else float("nan")

            check = pd.DataFrame([{
                "comparisons": len(nbd),
                "price: newton vs brentq (max abs diff)": maxdiff("price_newton", "price_brentq"),
                "price: analytic vs finite-difference newton": maxdiff("price_newton",
                                                                       "price_newton_fd"),
                "benefit: newton vs brentq": maxdiff("benefit_newton", "benefit_brentq"),
                "benefit: analytic vs finite-difference newton": maxdiff("benefit_newton",
                                                                         "benefit_newton_fd"),
                "median newton iterations (price)": float(nbd.price_iterations.median()),
                "median newton iterations (benefit)": float(nbd.benefit_iterations.median()),
                "no-root cases (benefit)": int(nbd.benefit_newton.isna().sum()),
                "no break-even price >= 0": int(nbd.price_brentq.isna().sum())}]).T
            check = check.reset_index().rename(columns={"index": "check", 0: "value"})
            tracking.log_table(cfg, STAGE, check, "newton_vs_brentq")
            metrics["newton_brentq_max_diff_benefit"] = maxdiff("benefit_newton",
                                                                "benefit_brentq")
        metrics.update({"share_equipoise": summary["share_equipoise_mcid_5"],
                        "share_recommend_bevacizumab": summary["share_recommend_Bevacizumab"],
                        "typical_patient_fuzzy_strength_afl": s,
                        "prices_demo": float(prices.is_demo)})
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
