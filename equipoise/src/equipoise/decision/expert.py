"""A small forward-chaining rule engine (expert system), written from scratch.

Working memory holds facts (name -> value). Each rule has conditions on the facts and an
action that asserts new facts. In every cycle the engine finds all rules whose conditions
hold and that have not fired yet (the agenda), resolves the conflict by choosing the one with
the highest salience (priority), then the most conditions (most specific), then the earliest
defined, fires it, and records a line in the reasoning trace. It stops when no rule can fire.
Same facts in -> same trace out. Every rule is decision-support logic for this project only,
NOT clinical guidance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Facts = dict[str, Any]


@dataclass
class Rule:
    name: str
    conditions: list[Callable[[Facts], bool]]
    action: Callable[[Facts], Facts]
    explain: Callable[[Facts], str]
    salience: int = 0
    doc: str = "Decision-support logic for this project, not clinical guidance."


@dataclass
class Engine:
    rules: list[Rule]
    trace: list[str] = field(default_factory=list)

    def run(self, facts: Facts, max_cycles: int = 100) -> Facts:
        wm = dict(facts)
        fired: set[str] = set()
        self.trace = []
        for cycle in range(1, max_cycles + 1):
            agenda = [(i, r) for i, r in enumerate(self.rules)
                      if r.name not in fired and all(c(wm) for c in r.conditions)]
            if not agenda:
                break
            i, rule = max(agenda, key=lambda ir: (ir[1].salience, len(ir[1].conditions),
                                                  -ir[0]))
            new = rule.action(wm)
            wm.update(new)
            fired.add(rule.name)
            self.trace.append(f"{cycle:2d}. [{rule.name}] {rule.explain(wm)}")
        return wm


def _f(name: str) -> Callable[[Facts], bool]:
    return lambda wm: bool(wm.get(name))


def _not(name: str) -> Callable[[Facts], bool]:
    return lambda wm: not wm.get(name)


def default_rules() -> list[Rule]:
    """The Equipoise rule base. Each rule is decision-support logic for this project, not
    clinical guidance. Comparisons are 'drug vs bevacizumab'; the dearer drug of interest is
    the one with the larger predicted benefit."""
    r: list[Rule] = []

    def add(name: str, conds: list, action, explain, salience: int = 0) -> None:
        r.append(Rule(name, conds, action, explain, salience))

    add("demo-prices", [_f("prices_are_demo")], lambda wm: {"warn_demo_prices": True},
        lambda wm: "Prices are DEMO relative units: cost outputs are illustrative only.", 90)
    add("research-prototype", [lambda wm: True], lambda wm: {"warn_prototype": True},
        lambda wm: "Research prototype for a course project; not clinical guidance.", 100)
    add("weak-signal", [lambda wm: True], lambda wm: {"warn_weak_signal": True},
        lambda wm: "Per-patient differences between drugs are weakly estimated from 660 "
                   "trial patients; intervals are wide.", 80)
    add("published-subgroup", [lambda wm: wm.get("baseline_va_letters", 99) < 69],
        lambda wm: {"published_rule_suggests": "Aflibercept"},
        lambda wm: "Baseline vision < 69 letters (20/50 or worse): in Protocol T this subgroup "
                   "gained more with aflibercept (published rule).", 50)
    add("published-subgroup-better-vision", [lambda wm: wm.get("baseline_va_letters", 0) >= 69],
        lambda wm: {"published_rule_suggests": "Bevacizumab"},
        lambda wm: "Baseline vision >= 69 letters: in Protocol T the drugs gave similar "
                   "gains (published rule: no advantage for the dearer drugs).", 50)
    for drug in ("Aflibercept", "Ranibizumab"):
        add(f"clear-benefit-{drug}",
            [lambda wm, d=drug: wm[f"effect_low_{d}"] >= wm["mcid"]],
            lambda wm, d=drug: {f"clear_benefit_{d}": True},
            lambda wm, d=drug: f"{d}: whole 95% interval of the benefit vs bevacizumab "
                               f"({wm[f'effect_low_{d}']:.1f} to {wm[f'effect_high_{d}']:.1f}) is "
                               f">= MCID {wm['mcid']:.0f} letters.", 40)
        add(f"equipoise-{drug}",
            [lambda wm, d=drug: wm[f"effect_low_{d}"] < wm["mcid"]],
            lambda wm, d=drug: {f"equipoise_{d}": True},
            lambda wm, d=drug: f"{d}: the 95% interval of the benefit vs bevacizumab "
                               f"({wm[f'effect_low_{d}']:.1f} to {wm[f'effect_high_{d}']:.1f}) "
                               f"includes values below MCID {wm['mcid']:.0f} -> equipoise.", 40)
    add("best-drug-clear", [lambda wm: bool(wm.get(f"clear_benefit_{wm.get('best_drug')}"))],
        lambda wm: {"best_clear_benefit": True},
        lambda wm: f"The drug with the larger predicted benefit is {wm['best_drug']}, and its "
                   "benefit is clear.", 30)
    add("fallback-cheapest", [lambda wm: "recommendation" not in wm],
        lambda wm: {"recommendation": "Bevacizumab", "equipoise": True,
                    "reason": "no rule found a clear, cost-effective advantage"},
        lambda wm: "No clear, cost-effective advantage found -> cheapest option.", -10)
    add("all-equipoise", [_f("equipoise_Aflibercept"), _f("equipoise_Ranibizumab")],
        lambda wm: {"recommendation": "Bevacizumab", "equipoise": True,
                    "reason": "no drug shows a benefit beyond the MCID with confidence; the "
                              "cheapest option is reasonable"},
        lambda wm: "Equipoise for both dearer drugs -> recommend the cheapest (bevacizumab).",
        20)
    add("clear-benefit-cost-effective",
        [_f("best_clear_benefit"), lambda wm: wm.get("best_net_benefit_gain", 0) > wm.get(
            "net_benefit_margin", 0)],
        lambda wm: {"recommendation": wm["best_drug"], "equipoise": False,
                    "reason": "clear benefit beyond the MCID and positive net benefit at the "
                              "current prices"},
        lambda wm: f"{wm['best_drug']} shows a clear benefit and is worth its extra cost at "
                   f"current prices -> recommend {wm['best_drug']}.", 20)
    add("clear-benefit-not-cost-effective",
        [_f("best_clear_benefit"), lambda wm: wm.get("best_net_benefit_gain", 0) <= wm.get(
            "net_benefit_margin", 0)],
        lambda wm: {"recommendation": "Discuss", "equipoise": False,
                    "reason": "clear benefit, but not worth the extra cost at current prices"},
        lambda wm: f"{wm['best_drug']} shows a clear benefit but its net benefit is not "
                   f"positive at current prices -> discuss; break-even price "
                   f"{wm.get('break_even_price', 'n/a')}.", 20)
    add("model-vs-published", [_f("equipoise"), lambda wm: wm.get("published_rule_suggests")
                               == "Aflibercept"],
        lambda wm: {"note_published_rule": True},
        lambda wm: "Note: the model is uncertain for this patient, while the published "
                   "subgroup result favoured aflibercept; discuss both.", 10)
    add("mcid-sensitivity", [lambda wm: wm.get("equipoise_at_sensitivity") is not None],
        lambda wm: {},
        lambda wm: "Sensitivity: at MCID "
                   + ", ".join(f"{m:g}: {'equipoise' if e else 'clear benefit'}"
                               for m, e in wm["equipoise_at_sensitivity"].items()) + ".", 5)
    return r
