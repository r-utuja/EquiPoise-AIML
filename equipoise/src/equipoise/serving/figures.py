"""Figures drawn by the app from API responses (plain dicts), kept out of the Streamlit file
so they can be tested without a browser."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from equipoise.plotting import INK_2, NEUTRAL, SERIES, arm_color


def point_range(pred: dict[str, Any]) -> plt.Figure:
    """Predicted letters gained per drug with the bootstrap interval of the effect."""
    est = pred["estimates"]
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(est) + 1.4))
    y = np.arange(len(est))[::-1]
    for yi, e in zip(y, est, strict=True):
        c = arm_color(e["drug"])
        if e["letters_low"] is not None:
            ax.hlines(yi, e["letters_low"], e["letters_high"], color=c, lw=3, alpha=0.6)
        ax.plot(e["letters"], yi, "o", color=c, ms=9)
        ax.annotate(f"{e['letters']:+.1f}", (e["letters"], yi), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, [e["drug"] for e in est])
    ax.set_xlabel("Predicted letters gained at 1 year (ETDRS letters)")
    ax.set_title("Counterfactual vision under each drug (bar = 95% bootstrap interval)")
    ax.set_ylim(-0.7, len(est) - 0.3)
    fig.tight_layout()
    return fig


def strength_gauge(strengths: dict[str, float]) -> plt.Figure:
    """Fuzzy recommendation strength (0-100) for each dearer drug."""
    fig, ax = plt.subplots(figsize=(7, 0.6 * max(len(strengths), 1) + 1.1))
    names = list(strengths)
    y = np.arange(len(names))[::-1]
    ax.barh(y, [100] * len(names), color="#eeeeea", height=0.5)
    ax.barh(y, [strengths[n] for n in names], color=[arm_color(n) for n in names], height=0.5)
    for yi, n in zip(y, names, strict=True):
        ax.text(min(strengths[n] + 2, 92), yi, f"{strengths[n]:.0f}", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fuzzy recommendation strength for the dearer drug (0-100)")
    ax.set_title("How strongly the fuzzy layer favours paying more")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def waterfall(expl: dict[str, Any]) -> plt.Figure:
    """SHAP waterfall from the API's contribution list (letters)."""
    rows = [(f"{c['feature'].replace('X_', '')} = {c['value']}", c["shap_letters"])
            for c in expl["contributions"]]
    rows.append(("all other features", expl["other_features_letters"]))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.6))
    start = expl["base_value_letters"]
    y = np.arange(len(rows))[::-1]
    for yi, (_label, v) in zip(y, rows, strict=True):
        ax.barh(yi, v, left=start, color=SERIES[0] if v >= 0 else SERIES[1], height=0.6)
        ax.text(start + v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
        start += v
    ax.axvline(expl["base_value_letters"], color=NEUTRAL, lw=1, ls="--")
    ax.axvline(expl["prediction_letters"], color=INK_2, lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f"Predicted {expl['comparison']} effect (letters); dashed = average, "
                  "solid = this patient")
    ax.set_title("Why this predicted effect? (SHAP contributions)")
    fig.tight_layout()
    return fig


def breakeven_curve(be: dict[str, Any]) -> plt.Figure:
    """Net-benefit difference (dearer drug - reference) against the dearer drug's price."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    p = [c["price"] for c in be["curve"]]
    v = [c["net_benefit_difference"] for c in be["curve"]]
    ax.plot(p, v, color=arm_color(be["drug"]), label=f"{be['drug']} - {be['reference']}")
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.axvline(be["current_price"], color=INK_2, ls=":", lw=1.5, label="price in use")
    res = be.get("result") or {}
    if res.get("break_even_price") is not None:
        ax.axvline(res["break_even_price"], color=SERIES[3], ls="--", lw=1.5,
                   label=f"break-even {res['break_even_price']:.3g}")
    ax.set_xlabel(f"{be['drug']} price per injection ({be['prices_label']})")
    ax.set_ylabel("Net benefit difference (value units)")
    ax.set_title(f"Is {be['drug']} worth its price for this patient?")
    ax.legend()
    fig.tight_layout()
    return fig


def early_trajectory(er: dict[str, Any]) -> plt.Figure:
    """Observed early vision change and the week-52 forecast (with typical error band)."""
    pts = er["trajectory"]
    obs = [p for p in pts if not p.get("forecast")]
    fc = [p for p in pts if p.get("forecast")][0]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    c = arm_color(er["treatment"])
    ax.plot([p["week"] for p in obs], [p["va_change"] for p in obs], "o-", color=c,
            label="entered vision")
    ax.plot([obs[-1]["week"], fc["week"]], [obs[-1]["va_change"], fc["va_change"]], "--",
            color=c, alpha=0.7)
    err = er.get("typical_error_letters")
    ax.errorbar(fc["week"], fc["va_change"], yerr=err if err else None, fmt="D", color=c,
                capsize=4, label="week-52 forecast" + (" (± CV RMSE)" if err else ""))
    if er.get("baseline_model_va_change_wk52") is not None:
        ax.plot(fc["week"] + 1.5, er["baseline_model_va_change_wk52"], "s", color=NEUTRAL,
                label="baseline-only prediction")
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xlabel("Weeks since first injection")
    ax.set_ylabel("Change from baseline (ETDRS letters)")
    ax.set_title(f"Early response on {er['treatment']}")
    ax.legend()
    fig.tight_layout()
    return fig
