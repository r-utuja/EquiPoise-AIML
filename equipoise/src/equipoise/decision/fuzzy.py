"""Fuzzy inference: how strongly to recommend the dearer drug over the cheapest (0-100).

Crisp inputs are turned into degrees of membership in overlapping linguistic sets ("benefit
is moderate to degree 0.7"). Readable IF-THEN rules combine them (AND = minimum, OR =
maximum), each rule clips its output set, the clipped sets are merged (maximum), and the
centroid of the merged shape gives one crisp strength. Inputs: predicted benefit vs the
cheapest drug (letters), certainty P(benefit > MCID), extra cost (normalized 0-1: share of
the dearer drug's cost that is extra) and the injection-burden difference. Decision-support
logic for this project, not clinical guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from equipoise.config import Config

# Each rule's firing moves monotonically with benefit (weak rules fire less as benefit rises,
# moderate/strong rules fire more), which keeps the output monotone in benefit.
RULE_TEXT = [
    "IF benefit is negligible THEN strength is weak",
    "IF certainty is low THEN strength is weak",
    "IF benefit is NOT large AND extra cost is high THEN strength is weak",
    "IF benefit is NOT large AND burden is more THEN strength is weak",
    "IF benefit is NOT negligible AND certainty is medium AND extra cost is NOT high "
    "THEN strength is moderate",
    "IF benefit is NOT negligible AND benefit is NOT large AND certainty is high "
    "THEN strength is moderate",
    "IF benefit is large AND certainty is high AND extra cost is NOT high THEN strength is strong",
    "IF benefit is large AND certainty is high AND extra cost is high THEN strength is moderate",
    "IF benefit is large AND certainty is medium THEN strength is moderate",
    "IF benefit is large AND burden is fewer AND certainty is NOT low THEN strength is strong",
]


@dataclass
class FuzzySystem:
    sim: ctrl.ControlSystemSimulation
    variables: dict[str, ctrl.Antecedent | ctrl.Consequent]
    bounds: dict[str, tuple[float, float]]


def build(cfg: Config) -> FuzzySystem:
    f = cfg.decision["fuzzy"]
    res = int(f["output_resolution"])
    blo, bhi = f["benefit_universe"]
    ulo, uhi = f["burden_universe"]
    benefit = ctrl.Antecedent(np.linspace(blo, bhi, 201), "benefit")
    certainty = ctrl.Antecedent(np.linspace(0, 1, 101), "certainty")
    cost = ctrl.Antecedent(np.linspace(0, 1, 101), "extra_cost")
    burden = ctrl.Antecedent(np.linspace(ulo, uhi, 81), "burden")
    strength = ctrl.Consequent(np.linspace(0, 100, res), "strength", defuzzify_method="centroid")

    mcid = cfg.mcid_letters
    benefit["negligible"] = fuzz.trapmf(benefit.universe, [blo, blo, 0.4 * mcid, mcid])
    benefit["moderate"] = fuzz.trimf(benefit.universe, [0.4 * mcid, mcid, 2 * mcid])
    benefit["large"] = fuzz.trapmf(benefit.universe, [mcid, 2 * mcid, bhi, bhi])
    certainty["low"] = fuzz.trapmf(certainty.universe, [0, 0, 0.2, 0.45])
    certainty["medium"] = fuzz.trimf(certainty.universe, [0.25, 0.5, 0.75])
    certainty["high"] = fuzz.trapmf(certainty.universe, [0.55, 0.8, 1, 1])
    cost["low"] = fuzz.trapmf(cost.universe, [0, 0, 0.2, 0.4])
    cost["medium"] = fuzz.trimf(cost.universe, [0.25, 0.5, 0.75])
    cost["high"] = fuzz.trapmf(cost.universe, [0.6, 0.8, 1, 1])
    burden["fewer"] = fuzz.trapmf(burden.universe, [ulo, ulo, -1.5, 0])
    burden["similar"] = fuzz.trimf(burden.universe, [-1.5, 0, 1.5])
    burden["more"] = fuzz.trapmf(burden.universe, [0, 1.5, uhi, uhi])
    strength["weak"] = fuzz.trapmf(strength.universe, [0, 0, 20, 45])
    strength["moderate"] = fuzz.trimf(strength.universe, [25, 50, 75])
    strength["strong"] = fuzz.trapmf(strength.universe, [55, 80, 100, 100])

    b, c, x, u, s = benefit, certainty, cost, burden, strength
    rules = [
        ctrl.Rule(b["negligible"], s["weak"]),
        ctrl.Rule(c["low"], s["weak"]),
        ctrl.Rule(~b["large"] & x["high"], s["weak"]),
        ctrl.Rule(~b["large"] & u["more"], s["weak"]),
        ctrl.Rule(~b["negligible"] & c["medium"] & ~x["high"], s["moderate"]),
        ctrl.Rule(~b["negligible"] & ~b["large"] & c["high"], s["moderate"]),
        ctrl.Rule(b["large"] & c["high"] & ~x["high"], s["strong"]),
        ctrl.Rule(b["large"] & c["high"] & x["high"], s["moderate"]),
        ctrl.Rule(b["large"] & c["medium"], s["moderate"]),
        ctrl.Rule(b["large"] & u["fewer"] & ~c["low"], s["strong"]),
    ]
    assert len(rules) == len(RULE_TEXT)
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    variables = {"benefit": benefit, "certainty": certainty, "extra_cost": cost,
                 "burden": burden, "strength": strength}
    bounds = {"benefit": (blo, bhi), "certainty": (0.0, 1.0), "extra_cost": (0.0, 1.0),
              "burden": (ulo, uhi)}
    return FuzzySystem(sim, variables, bounds)


def strength(fs: FuzzySystem, benefit: float, certainty: float, extra_cost: float,
             burden: float) -> float:
    """Recommendation strength (0-100) for the dearer drug; inputs are clipped to range."""
    vals = {"benefit": benefit, "certainty": certainty, "extra_cost": extra_cost,
            "burden": burden}
    for k, v in vals.items():
        lo, hi = fs.bounds[k]
        fs.sim.input[k] = float(np.clip(v, lo, hi))
    fs.sim.compute()
    return float(fs.sim.output["strength"])


def aggregated_output(fs: FuzzySystem) -> np.ndarray:
    """Membership of the merged (max of clipped) output sets after the last compute()."""
    out = fs.variables["strength"]
    agg = np.zeros_like(out.universe)
    for term in out.terms.values():
        level = term.membership_value[fs.sim]
        agg = np.fmax(agg, np.fmin(level, term.mf))
    return agg
