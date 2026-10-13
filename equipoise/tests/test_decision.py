"""M8 building blocks: Newton vs brentq, fuzzy range and monotonicity, rule engine."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from equipoise.config import Config, load_prices
from equipoise.decision.breakeven import break_even, brent_root, newton, value_of_vision
from equipoise.decision.cost import cost_per_letter, expected_cost
from equipoise.decision.expert import Engine, Rule, default_rules
from equipoise.decision.fuzzy import RULE_TEXT, build, strength


def test_newton_matches_brentq_on_nonlinear_function() -> None:
    def g(x: float) -> float:
        return x**3 - 2 * x - 5

    analytic = newton(g, 2.0, lambda x: 3 * x**2 - 2)
    fd = newton(g, 2.0)
    ref = brent_root(g, 0, 5)
    assert analytic.converged and fd.converged
    assert analytic.root == pytest.approx(ref, abs=1e-10) == fd.root
    assert brent_root(lambda x: x**2 + 1, -5, 5) is None          # no sign change: no root
    assert not newton(lambda x: x**2 + 1, 0.0, lambda x: 2 * x).converged


@pytest.mark.parametrize("seed", range(5))
def test_break_even_newton_matches_brentq(seed: int) -> None:
    rng = np.random.default_rng(seed)
    args = dict(letters_ref=rng.uniform(0, 15), letters_drug=rng.uniform(5, 25),
                inj_ref=rng.uniform(6, 12), inj_drug=rng.uniform(5, 11),
                laser_ref=rng.uniform(0, 0.6), laser_drug=rng.uniform(0, 0.6), price_ref=1.0,
                price_drug=10.0, laser_price=2.0, wtp=200.0, k=0.05)
    be = break_even(**args)
    if be.price_brentq is not None:
        assert be.price_newton.root == pytest.approx(be.price_brentq, abs=1e-8)
        assert be.price_newton.iterations <= 2                       # linear: exact in one step
    if be.benefit_brentq is not None:
        assert be.benefit_newton.root == pytest.approx(be.benefit_brentq, abs=1e-8)
        assert be.benefit_newton_fd.root == pytest.approx(be.benefit_brentq, abs=1e-6)
    else:
        assert be.benefit_letters is None and "no root" in be.benefit_newton.message


def test_break_even_no_root_when_cost_exceeds_value() -> None:
    be = break_even(10, 12, 9, 9, 0.3, 0.3, 1.0, 1000.0, 2.0, wtp=200.0, k=0.05)
    assert be.benefit_letters is None and "no root" in be.benefit_newton.message


def test_value_of_vision_is_concave_increasing() -> None:
    u = value_of_vision(np.arange(0, 60, 5.0), 0.05)
    assert np.all(np.diff(u) > 0) and np.all(np.diff(u, 2) < 0)


def test_cost_model() -> None:
    prices = load_prices({"price_per_injection_inr": {"Bevacizumab": 1, "Aflibercept": 10},
                          "laser_session_inr": 2})
    cost = expected_cost(np.array([[10.0, 8.0]]), np.array([[0.5, 0.25]]),
                         ["Bevacizumab", "Aflibercept"], prices)
    np.testing.assert_allclose(cost, [[11.0, 80.5]])
    np.testing.assert_allclose(cost_per_letter(cost, np.array([[8.0, 12.0]]))[0, 1], 69.5 / 4)
    assert np.isnan(cost_per_letter(cost, np.array([[8.0, 7.0]]))[0, 1])   # no gain


@pytest.fixture(scope="module")
def fs(cfg: Config):
    return build(cfg)


def test_fuzzy_output_in_range_and_monotone_in_benefit(fs) -> None:
    for c, x, u in itertools.product([0, 0.3, 0.6, 1], [0, 0.5, 1], [-4, 0, 4]):
        s = [strength(fs, b, c, x, u) for b in np.linspace(-5, 15, 21)]
        assert all(0 <= v <= 100 for v in s)
        assert np.all(np.diff(s) >= -1e-6), (c, x, u)
    assert 9 <= len(RULE_TEXT) <= 15


def test_fuzzy_clips_out_of_range_inputs(fs) -> None:
    assert strength(fs, 99, 2.0, -1, 50) == pytest.approx(strength(fs, 15, 1.0, 0, 4))


def _facts(low_afl: float, low_ran: float, va: float = 60) -> dict:
    return {"mcid": 5.0, "baseline_va_letters": va, "prices_are_demo": True,
            "best_drug": "Aflibercept", "best_net_benefit_gain": 10.0,
            "net_benefit_margin": 0.0, "break_even_price": "3",
            "effect_low_Aflibercept": low_afl, "effect_high_Aflibercept": low_afl + 4,
            "effect_low_Ranibizumab": low_ran, "effect_high_Ranibizumab": low_ran + 4,
            "equipoise_at_sensitivity": {3.0: False}}


def test_rule_engine_is_deterministic_and_reasons_correctly() -> None:
    eng = Engine(default_rules())
    wm1 = eng.run(_facts(1.0, 0.0))
    trace1 = list(eng.trace)
    wm2 = eng.run(_facts(1.0, 0.0))
    assert trace1 == eng.trace and wm1 == wm2
    assert wm1["recommendation"] == "Bevacizumab" and wm1["equipoise"]
    assert wm1.get("note_published_rule")              # vision < 69 and model equipoise
    clear = eng.run(_facts(6.0, 0.0))
    assert clear["recommendation"] == "Aflibercept"
    costly = eng.run({**_facts(6.0, 0.0), "best_net_benefit_gain": -5.0})
    assert costly["recommendation"] == "Discuss"
    assert all("not clinical guidance" in r.doc for r in default_rules())


def test_conflict_resolution_prefers_salience_then_specificity() -> None:
    always = [lambda wm: True]
    rules = [Rule("low", always, lambda wm: {"x": "low"}, lambda wm: "low", salience=1),
             Rule("specific", always * 2, lambda wm: {"x": "specific"}, lambda wm: "s",
                  salience=5),
             Rule("high", always, lambda wm: {"x": "high"}, lambda wm: "high", salience=5)]
    eng = Engine(rules)
    eng.run({})
    assert [t.split("[")[1].split("]")[0] for t in eng.trace] == ["specific", "high", "low"]
