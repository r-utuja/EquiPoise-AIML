"""EquipoiseAgent: the decision layer framed as an intelligent agent (perceive -> decide ->
explain).

The agent perceives a patient's baseline profile (its sensors are the input form / API
fields), checks it, decides by calling `recommend` (models + cost model + fuzzy system +
expert system), and acts by returning a recommendation with an explanation (its actuators
are the API response and the app screen). Its performance measure is the expected letters
gained at an acceptable cost, judged on held-out patients by policy value. It is a simple
reflex-plus-model agent: it keeps no memory between patients. PEAS details are in
docs/concepts/agent.md. Decision support for a course project, not clinical guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from equipoise.config import Config, Prices, load_config
from equipoise.decision.recommend import FEATURES, Recommendation, recommend


@dataclass
class EquipoiseAgent:
    cfg: Config = field(default_factory=load_config)
    prices: Prices | None = None
    bundle: Any = None
    percept: dict[str, Any] | None = None
    missing: list[str] = field(default_factory=list)
    decision: Recommendation | None = None

    def perceive(self, patient: dict[str, Any]) -> EquipoiseAgent:
        """Read the patient's baseline profile; note which features are missing."""
        self.percept = {k: patient[k] for k in FEATURES if k in patient}
        self.missing = [k for k in FEATURES if patient.get(k) is None]
        self.decision = None
        return self

    def decide(self) -> Recommendation:
        """Choose an action: the recommendation for the perceived patient."""
        if self.percept is None:
            raise RuntimeError("perceive(patient) must be called before decide()")
        self.decision = recommend(self.percept, self.prices, self.bundle, self.cfg)
        return self.decision

    def explain(self) -> str:
        """Human-readable explanation of the last decision (reasoning trace included)."""
        if self.decision is None:
            raise RuntimeError("decide() must be called before explain()")
        r = self.decision
        lines = [f"Recommendation: {r.recommendation} - {r.reason}.",
                 f"Equipoise (MCID {r.mcid_letters:g} letters): {'yes' if r.equipoise else 'no'}."]
        for e in r.estimates[1:]:
            lines.append(f"{e.drug} vs {r.estimates[0].drug}: {e.benefit_vs_reference:+.1f} "
                         f"letters (95% interval {e.benefit_low:+.1f} to {e.benefit_high:+.1f}), "
                         f"P(benefit > MCID) = {e.p_benefit_gt_mcid:.2f}, fuzzy strength "
                         f"{r.fuzzy_strength[e.drug]:.0f}/100.")
        if self.missing:
            names = ", ".join(m.replace("X_", "") for m in self.missing)
            lines.append(f"Imputed (not provided): {names}.")
        lines.append("Reasoning trace:")
        lines += [f"  {t}" for t in r.rule_trace]
        lines += [f"Note: {w}" for w in r.warnings]
        return "\n".join(lines)
