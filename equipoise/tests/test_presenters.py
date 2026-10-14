"""Presentation layer (app/presenters.py): rounding, wording thresholds, money, and the
safety rules (no clear winner never suggests the costlier drug; nothing shown from DEMO
prices). Pure functions on hand-made prediction dicts: no model, no Streamlit."""

from __future__ import annotations

import math
import re

import pytest

from app import copy as C
from app import presenters as P
from equipoise.config import load_config

UI = load_config().raw["ui"]
COSTLIER_BRANDS = ("Eylea", "Lucentis")


def make_pred(*, equipoise: bool = True, rec: str = "Bevacizumab", p_afl: float = 0.1,
              p_ran: float = 0.0, eff_afl: float = 3.0, eff_ran: float = 1.0,
              demo: bool = True, strength: float = 17.0) -> dict:
    """A Prediction-shaped dict (same keys as the API's /predict response)."""
    ref = 8.0

    def est(drug, eff, p, lo, hi, inj, cost):
        return {"drug": drug, "letters": ref + (eff or 0.0),
                "letters_low": None if eff is None else ref + lo,
                "letters_high": None if eff is None else ref + hi,
                "benefit_vs_reference": eff, "benefit_low": None if eff is None else lo,
                "benefit_high": None if eff is None else hi, "p_benefit_gt_mcid": p,
                "injections_yr1": inj, "p_laser_yr1": 0.3, "cost": cost,
                "cost_per_extra_letter": None, "net_benefit": None}
    return {
        "recommendation": rec, "reason": "", "equipoise": equipoise,
        "equipoise_at_sensitivity": {"3": equipoise}, "mcid_letters": 5.0,
        "estimates": [est("Bevacizumab", None, None, 0, 0, 9.4, 18_000.0),
                      est("Aflibercept", eff_afl, p_afl, eff_afl - 2.5, eff_afl + 2.5, 8.7,
                          176_000.0),
                      est("Ranibizumab", eff_ran, p_ran, eff_ran - 2, eff_ran + 2, 9.2,
                          92_000.0)],
        "fuzzy_strength": {"Aflibercept": strength, "Ranibizumab": 17.0},
        "break_even": [], "rule_trace": [], "published_rule_suggests": "Aflibercept",
        "prices_label": "DEMO" if demo else "INR", "prices_are_demo": demo, "warnings": [],
        "model": "test", "imputed_features": [], "explanation": None,
    }


# ---------------------------------------------------------------- lines and letters
@pytest.mark.parametrize("letters,lines", [(0, 0), (1.2, 0), (1.3, 0.5), (5, 1), (7.4, 1.5),
                                           (7.5, 1.5), (8.7, 1.5), (8.8, 2), (10, 2), (12.6, 2.5),
                                           (-6, -1), (-1.24, 0)])
def test_letters_to_lines_rounds_to_half_lines(letters: float, lines: float) -> None:
    assert P.letters_to_lines(letters, UI) == lines


def test_gain_text_and_units() -> None:
    assert P.gain_text(10, UI) == "About 2 lines"
    assert P.gain_text(5, UI) == "About 1 line"
    assert P.gain_text(0.5, UI) == C.LITTLE_CHANGE
    assert P.gain_text(-5, UI) == "About 1 line worse"
    assert P.letters_text(9.6) == "≈ 10 letters"


def test_no_negative_zero_in_ranges() -> None:
    assert "-0" not in P.range_text(-0.3, 4.6, UI)
    assert P.range_text(-0.3, 4.6, UI) == "likely between 0 and 1 more line"
    assert P.range_text(-2.0, 3.7, UI) == "likely between 0.5 fewer and 0.5 more lines"
    assert P.range_text(2.2, 8.6, UI) == "likely between 0.5 and 1.5 more lines"
    assert P.range_text(-6, -3, UI) == "likely between 0.5 and 1 fewer line"


# ---------------------------------------------------------------- eye chart
def test_snellen_to_letters_formula_and_clamp() -> None:
    assert [P.snellen_to_letters(d) for d, _ in UI["snellen"]] == \
        [75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    assert P.snellen_to_letters(20) == 78          # 85 letters clamped to the trial range
    assert P.snellen_to_letters(800) == 24


def test_nearest_snellen_and_text() -> None:
    assert P.nearest_snellen(48, UI) == 100
    assert P.snellen_text(65, UI) == "20/50 (6/15)"
    assert P.snellen_text(90, UI).startswith("20/16")    # results can exceed the trial range
    assert P.is_snellen_value(70, UI) and not P.is_snellen_value(69, UI)


# ---------------------------------------------------------------- words from thresholds
@pytest.mark.parametrize("x,word", [(0.95, "fairly"), (0.8, "fairly"), (0.79, "somewhat"),
                                    (0.6, "somewhat"), (0.59, "close"), (0.0, "close")])
def test_confidence_thresholds(x: float, word: str) -> None:
    assert P.confidence_key(x, UI) == word
    assert P.confidence_word(x, UI) == C.CONFIDENCE[word]


def test_confidence_thresholds_come_from_config() -> None:
    ui = {**UI, "confidence": {"fairly": 0.9, "somewhat": 0.7}}
    assert P.confidence_key(0.85, ui) == "somewhat"


@pytest.mark.parametrize("s,word", [(17, "weak"), (39.9, "weak"), (40, "moderate"),
                                    (69, "moderate"), (70, "strong"), (100, "strong")])
def test_strength_words(s: float, word: str) -> None:
    assert P.strength_word(s, UI) == C.STRENGTH[word]


# ---------------------------------------------------------------- money
@pytest.mark.parametrize("amount,text", [(950, "₹950"), (4_504, "₹4,500"), (45_449, "₹45,400"),
                                         (99_960, "₹1,00,000"), (100_000, "₹1 lakh"),
                                         (123_456, "₹1.2 lakh"), (1_960_000, "₹19.6 lakh"),
                                         (25_000_000, "₹2.5 crore")])
def test_rupee_formatting(amount: float, text: str) -> None:
    assert P.format_inr(amount) == text


def test_value_of_a_line_converts_to_willingness_to_pay() -> None:
    k, lpl = 0.05, 5
    wtp = P.wtp_from_value_per_line(10_000, k, lpl)
    assert wtp * (1 - math.exp(-k * lpl)) == pytest.approx(10_000)


# ---------------------------------------------------------------- prices not set
def test_costs_hidden_while_prices_are_demo() -> None:
    vm = P.comparison(make_pred(demo=True), UI, va=60)
    assert all(c.cost is None for c in vm.cards)
    assert vm.cost_note == C.COSTS_HIDDEN_NOTE
    shown = " ".join([vm.headline.text, *(str(c) for c in vm.cards), vm.chart.summary])
    assert "₹" not in shown and "×" not in shown and "cost of" not in shown


def test_costs_shown_with_real_prices() -> None:
    vm = P.comparison(make_pred(demo=False), UI, va=60)
    assert vm.cost_note is None
    assert [c.cost for c in vm.cards] == ["About ₹18,000 in the first year",
                                          "About ₹92,000 in the first year",
                                          "About ₹1.8 lakh in the first year"]


def test_injections_always_shown() -> None:
    cards = P.cards(make_pred(demo=True), UI)
    assert [c.injections for c in cards] == ["About 9 injections in the first year"] * 3


# ---------------------------------------------------------------- headline rules
@pytest.mark.parametrize("rec", ["Bevacizumab", "Aflibercept", "Ranibizumab", "Discuss"])
@pytest.mark.parametrize("p_afl", [0.0, 0.3, 0.52, 0.9])
@pytest.mark.parametrize("demo", [True, False])
def test_no_clear_winner_never_suggests_a_costlier_drug(rec: str, p_afl: float,
                                                        demo: bool) -> None:
    pred = make_pred(equipoise=True, rec=rec, p_afl=p_afl, demo=demo)
    h = P.headline(pred, UI, va=50)
    assert h.kind == "no_clear_winner"
    assert h.suggested == "Bevacizumab"
    assert not any(b in h.text for b in COSTLIER_BRANDS)
    assert "Avastin" in h.text
    tagged = [c for c in P.cards(pred, UI) if c.tag == C.TAG_SUGGESTED]
    assert [c.drug for c in tagged] == ["Bevacizumab"]


def test_no_clear_winner_confidence_is_one_minus_p() -> None:
    h = P.headline(make_pred(p_afl=0.04), UI, va=70)
    assert h.confidence_level == pytest.approx(0.96)
    assert h.confidence == C.CONFIDENCE["fairly"]
    assert P.headline(make_pred(p_afl=0.24), UI).confidence == C.CONFIDENCE["somewhat"]
    assert P.headline(make_pred(p_afl=0.52), UI).confidence == C.CONFIDENCE["close"]


def test_may_benefit_line_when_no_clear_winner_but_p_at_least_half() -> None:
    h = P.headline(make_pred(equipoise=True, p_afl=0.52), UI, va=48)
    assert h.secondary == C.MAY_BENEFIT.format(drug="Eylea")
    assert h.kind == "no_clear_winner" and "Avastin" in h.text
    assert "reasonable choice" in h.text
    assert P.headline(make_pred(equipoise=True, p_afl=0.49), UI).secondary is None


def test_clear_winner_headline() -> None:
    pred = make_pred(equipoise=False, rec="Aflibercept", p_afl=0.97, eff_afl=7.0, demo=False)
    h = P.headline(pred, UI, va=50)
    assert h.kind == "winner" and h.suggested == "Aflibercept"
    assert h.text.startswith("Eylea (aflibercept) is likely to help this patient most")
    assert "about 1.5 lines more" in h.text
    assert h.confidence == C.CONFIDENCE["fairly"]


def test_discuss_needs_real_prices() -> None:
    real = make_pred(equipoise=False, rec="Discuss", p_afl=0.97, eff_afl=7.0, demo=False)
    assert P.headline(real, UI).kind == "discuss"
    demo = make_pred(equipoise=False, rec="Discuss", p_afl=0.97, eff_afl=7.0, demo=True)
    h = P.headline(demo, UI)
    assert h.kind == "winner" and "cost" not in h.text


def test_trial_finding_follows_starting_vision() -> None:
    assert P.headline(make_pred(), UI, va=60).trial_finding == C.TRIAL_FINDING_POOR
    assert P.headline(make_pred(), UI, va=70).trial_finding == C.TRIAL_FINDING_GOOD


# ---------------------------------------------------------------- cards and chart
def test_cards_label_ranges_as_differences_from_avastin() -> None:
    cards = {c.drug: c for c in P.cards(make_pred(eff_afl=3.0), UI)}
    assert [c.name for c in cards.values()] == ["Avastin (bevacizumab)",
                                                "Lucentis (ranibizumab)", "Eylea (aflibercept)"]
    assert cards["Bevacizumab"].range_line == C.CARD_REFERENCE_NOTE
    assert cards["Aflibercept"].range_line.startswith("Compared with Avastin: likely between")
    assert sum(c.tag is not None for c in cards.values()) <= 3
    assert all(re.fullmatch(r"(About [\d.]+ lines?( worse)?|Little change)", c.improvement)
               for c in cards.values())


def test_chart_band_only_for_drugs_with_a_range() -> None:
    vm = P.chart(make_pred(), UI)
    rows = {r.drug: r for r in vm.rows}
    assert rows["Bevacizumab"].low is None and rows["Aflibercept"].low is not None
    assert vm.axis == C.CHART_AXIS
    assert "None of the medicines is clearly ahead." in vm.summary


# ---------------------------------------------------------------- validation
def test_validation_messages_are_plain() -> None:
    ranges = {"X_cst_um": P.FieldRange(150, 1200), "X_log_uacr": P.FieldRange(0, 15)}
    msgs = P.validate({"X_cst_um": 90.0, "X_sbp": 80.0, "X_dbp": 90.0,
                       "X_log_uacr": math.log1p(50.0)}, ranges)
    assert msgs[0] == "Retinal thickness from OCT report should be between 150 and 1200 µm."
    assert msgs[-1] == C.VALIDATION_BP
    assert len(msgs) == 2


def test_kidney_test_round_trip() -> None:
    v = math.log1p(79.0)
    assert P.to_form_value("X_log_uacr", v) == pytest.approx(79.0)
    assert P.from_form_value("X_log_uacr", 79.0) == pytest.approx(v)
    assert P.from_form_value("X_cst_um", 400.0) == 400.0


# ---------------------------------------------------------------- progress check
def test_progress_view_model() -> None:
    er = {"treatment": "Aflibercept", "baseline_va_letters": 60.0,
          "predicted_va_change_wk52": 11.0, "predicted_va_wk52": 71.0,
          "typical_error_letters": 8.1, "baseline_model_va_change_wk52": 9.0,
          "trajectory": [{"week": 0.0, "va_change": 0.0}, {"week": 4.0, "va_change": 5.0},
                         {"week": 8.0, "va_change": 7.0}, {"week": 12.0, "va_change": 9.0},
                         {"week": 52.0, "va_change": 11.0, "forecast": 1.0}],
          "note": "", "disclaimer": ""}
    vm = P.progress(er, UI)
    assert vm.result == "On this path, expected vision at 1 year is about 20/40 (6/12) " \
                        "(≈ 71 letters)."
    assert vm.change == "That's about 2 lines better than at the start."
    assert vm.error == "This could easily be about 1.5 lines better or worse."
    assert [p.label for p in vm.observed] == ["Start", "1 month", "2 months", "3 months"]
    assert vm.forecast.label == "1 year"
    from app import charts

    spec = charts.progress(vm).to_dict()
    assert "layer" in spec
    assert "title" not in spec


def test_improvement_chart_spec_has_no_legend_or_title() -> None:
    from app import charts

    spec = charts.improvement(P.chart(make_pred(), UI)).to_dict()
    assert "title" not in spec
    assert '"legend": null' in __import__("json").dumps(spec)


# ---------------------------------------------------------------- step 3: why
TYPICAL = {"X_cst_um": 430.0, "X_va_letters": 69.0, "X_hba1c": 7.7, "X_log_uacr": 4.38}
BINARY = {"X_female", "X_insulin", "X_va_below_69"}


def with_explanation(pred: dict) -> dict:
    return {**pred, "explanation": {
        "comparison": "Aflibercept - Bevacizumab", "base_value_letters": 3.0,
        "prediction_letters": 4.5, "other_features_letters": 0.1, "note": "",
        "contributions": [
            {"feature": "X_hba1c", "value": "imputed", "shap_letters": -0.1},
            {"feature": "X_cst_um", "value": "768", "shap_letters": 1.2},
            {"feature": "X_va_letters", "value": "48", "shap_letters": 0.4},
            {"feature": "X_female", "value": "1", "shap_letters": 0.05}]}}


def test_factors_are_plain_sentences_ordered_by_size() -> None:
    f = P.factors(with_explanation(make_pred()), UI, TYPICAL, BINARY)
    assert f == ["A thicker retina (768 µm) made Eylea's advantage noticeably larger.",
                 "Poorer starting vision (20/100) made Eylea's advantage a little larger.",
                 "A typical value for HbA1c made Eylea's advantage slightly smaller."]


def test_factor_subjects() -> None:
    assert P.factor_subject("X_female", "1", None, True, UI) == "Sex (Female)"
    assert P.factor_subject("X_insulin", "0", None, True, UI) == "Uses insulin (No)"
    assert P.factor_subject("X_log_uacr", str(math.log1p(7792.1)), 4.38, False, UI) == \
        "A higher kidney-test result (7,792 mg/g)"
    assert P.factor_subject("X_oct_machine", "Zeiss Cirrus", None, False, UI) == \
        "OCT machine (Zeiss Cirrus)"


def test_factors_absent_without_explanation() -> None:
    vm = P.why(make_pred(), UI, 60, TYPICAL, BINARY)
    assert vm.factors == [] and vm.factors_intro is None


def test_decided_bullets() -> None:
    d = P.decided(make_pred(eff_afl=3.0), UI)
    assert 2 <= len(d) <= 4
    assert d[0] == C.DECIDE_ESTIMATE
    assert d[1].startswith("Eylea compared with Avastin: likely between")
    assert d[1].endswith("That's not clearly more than 1 line.")
    assert d[-1] == C.DECIDE_LOWER_BAR_SAME.format(bar=C.HALF_LINE)
    clear = P.decided(make_pred(equipoise=False, rec="Aflibercept", eff_afl=9.0, demo=False), UI)
    assert any(s.endswith("That's clearly more than 1 line.") for s in clear)
    assert C.DECIDE_WORTH in clear


def test_strength_hidden_until_real_prices() -> None:
    assert P.strength_line(make_pred(demo=True, strength=80), UI) == C.STRENGTH_HIDDEN
    assert P.strength_line(make_pred(demo=False, strength=17), UI) == \
        "Weak case for paying more for Eylea."
    assert P.strength_line(make_pred(demo=False, strength=75), UI) == \
        "Strong case for paying more for Eylea."


def _be(curve_at_zero: float, slope: float, bep: float | None) -> dict:
    return {"drug": "Aflibercept", "reference": "Bevacizumab", "current_price": 30_000.0,
            "result": {"drug": "Aflibercept", "break_even_price": bep,
                       "break_even_benefit_letters": None, "newton_iterations_price": 3,
                       "newton_iterations_benefit": 3, "message": ""},
            "curve": [{"price": p, "net_benefit_difference": curve_at_zero + slope * p}
                      for p in (0.0, 30_000.0, 60_000.0, 90_000.0)],
            "prices_label": "INR", "prices_are_demo": False, "warnings": []}


def test_break_even_sentences() -> None:
    vm = P.break_even(_be(100_000.0, -2.0, 50_000.0), 30_000.0)
    assert vm.sentence == ("If one Eylea injection costs more than about ₹50,000, it stops "
                           "being worth the extra cost for this patient.")
    assert vm.at_price == "At ₹30,000 per injection, Eylea is worth the extra cost."
    assert P.break_even(_be(100_000.0, -2.0, 50_000.0), 60_000.0).at_price.endswith(
        "isn't worth the extra cost.")
    assert P.break_even(_be(-5.0, -2.0, None), 0.0).sentence == C.BE_NEVER.format(drug="Eylea")
    assert P.break_even({**_be(1, 1, None), "curve": []}, 0.0) is None
