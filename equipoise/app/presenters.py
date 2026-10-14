"""Plain-language view models built from the API's `Prediction` (a plain dict).

Pure functions, no Streamlit: everything a page shows is computed here and unit-tested in
`tests/test_presenters.py`. Nothing here changes a model number; it only rounds, converts
(letters to eye-chart lines, letters to Snellen, rupees to lakh) and picks wording from
`app/copy.py` with thresholds from the `ui:` block of `config/default.yaml`.

Two rules are enforced here rather than in the pages:
* "No clear winner" never names a costlier drug as the suggestion.
* While prices are the DEMO placeholders, nothing derived from them is shown (no costs, no
  cost-based wording); injection counts are shown because they are model outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app import copy as C

REFERENCE = "Bevacizumab"
CARD_ORDER = ("Bevacizumab", "Ranibizumab", "Aflibercept")   # cheapest to dearest


# ---------------------------------------------------------------- numbers and units
def brand(drug: str) -> str:
    return C.DRUG_BRAND.get(drug, drug)


def full_name(drug: str) -> str:
    return C.DRUG_FULL.format(brand=brand(drug), generic=C.DRUG_GENERIC.get(drug, drug))


def round_step(x: float, step: float) -> float:
    """Round half away from zero to a multiple of `step` (1.25 -> 1.5, -0.25 -> -0.5)."""
    q = abs(x) / step
    return math.copysign(math.floor(q + 0.5) * step, x) + 0.0   # + 0.0 turns -0.0 into 0.0


def num(x: float) -> str:
    """1.0 -> '1', 1.5 -> '1.5' (one decimal at most)."""
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def letters_to_lines(letters: float, ui: dict[str, Any]) -> float:
    """Eye-chart lines (1 line = 5 letters), rounded to the configured step."""
    return round_step(letters / float(ui["letters_per_line"]), float(ui["lines_step"]))


def line_unit(n: float) -> str:
    return C.LINE if abs(n) == 1 else C.LINES


def lines_phrase(letters: float, ui: dict[str, Any]) -> str:
    """'2 lines', '1 line', '0.5 lines' (magnitude only)."""
    n = abs(letters_to_lines(letters, ui))
    return f"{num(n)} {line_unit(n)}"


def gain_text(letters: float, ui: dict[str, Any]) -> str:
    """'About 2 lines', 'About 1 line worse' or 'Little change'."""
    n = letters_to_lines(letters, ui)
    if n == 0:
        return C.LITTLE_CHANGE
    tpl = C.GAIN if n > 0 else C.LOSS
    return tpl.format(n=num(abs(n)), unit=line_unit(n))


def letters_text(letters: float) -> str:
    return C.LETTERS_APPROX.format(n=f"{round(letters):.0f}")


def range_text(low: float, high: float, ui: dict[str, Any]) -> str:
    """Likely range of a *difference* in letters, in lines: 'likely between 0 and 1 more line'."""
    lo, hi = letters_to_lines(low, ui), letters_to_lines(high, ui)
    if lo == 0 and hi == 0:
        return C.RANGE_NONE
    if lo >= 0:
        if lo == hi:
            return C.RANGE_ABOUT.format(n=num(hi), unit=line_unit(hi))
        return C.RANGE_MORE.format(lo=num(lo), hi=num(hi), unit=line_unit(hi))
    if hi <= 0:
        return C.RANGE_FEWER.format(lo=num(-hi), hi=num(-lo), unit=line_unit(lo))
    return C.RANGE_MIXED.format(lo=num(-lo), hi=num(hi), unit=line_unit(hi))


# ---------------------------------------------------------------- eye chart
def snellen_to_letters(denominator: float, lo: float = 24, hi: float = 78) -> float:
    """letters ~ 85 + 50 * log10(20 / denominator), rounded and clamped to the trial range."""
    v = round(85 + 50 * math.log10(20.0 / float(denominator)))
    return float(min(max(v, lo), hi))


def snellen_options(ui: dict[str, Any]) -> list[tuple[int, str]]:
    """[(denominator, '20/50 (6/15)'), ...] from config."""
    return [(int(d), C.VISION_OPTION.format(feet=f"20/{int(d)}", metric=m))
            for d, m in ui["snellen"]]


def nearest_snellen(letters: float, ui: dict[str, Any]) -> int:
    """Denominator of the dropdown entry closest to a letter score."""
    return min((int(d) for d, _ in ui["snellen"]),
               key=lambda d: (abs(snellen_to_letters(d) - letters), d))


def _letters_unclamped(denominator: float) -> float:
    return 85 + 50 * math.log10(20.0 / float(denominator))


def _display_lines(ui: dict[str, Any]) -> list[tuple[int, str]]:
    return sorted([(int(d), m) for d, m in [*ui["snellen"], *ui["snellen_display_extra"]]])


def snellen_text(letters: float, ui: dict[str, Any]) -> str:
    """'20/50 (6/15)' for the nearest chart line (also outside the trial's range, e.g. a
    forecast of 20/25)."""
    d, m = min(_display_lines(ui), key=lambda dm: abs(_letters_unclamped(dm[0]) - letters))
    return C.VISION_OPTION.format(feet=f"20/{d}", metric=m)


def snellen_feet(letters: float, ui: dict[str, Any]) -> str:
    """'20/50' for the nearest chart line."""
    d, _ = min(_display_lines(ui), key=lambda dm: abs(_letters_unclamped(dm[0]) - letters))
    return f"20/{d}"


def is_snellen_value(letters: float, ui: dict[str, Any]) -> bool:
    return snellen_to_letters(nearest_snellen(letters, ui)) == float(letters)


# ---------------------------------------------------------------- money
def format_inr(amount: float) -> str:
    """Rounded the way people say it: '₹950', '₹45,500', '₹1.2 lakh', '₹2.5 crore'."""
    a = float(amount)
    if a >= 1e7:
        return f"₹{num(round(a / 1e7, 1))} crore"
    if a >= 1e5:
        return f"₹{num(round(a / 1e5, 1))} lakh"
    if a >= 1000:
        r = int(round(a / 100.0) * 100)
        s = str(r)
        head, tail = s[:-3], s[-3:]      # Indian grouping: 12,34,500
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        return "₹" + ",".join([*groups, tail])
    return f"₹{int(round(a / 10.0) * 10)}"


def wtp_from_value_per_line(value_per_line: float, k: float, letters_per_line: float) -> float:
    """Willingness to pay per unit of U(letters) = 1 - exp(-k * letters) such that the first
    eye-chart line gained (0 -> 5 letters) is worth `value_per_line` rupees."""
    return float(value_per_line) / (1.0 - math.exp(-float(k) * float(letters_per_line)))


# ---------------------------------------------------------------- words from thresholds
def confidence_key(x: float, ui: dict[str, Any]) -> str:
    c = ui["confidence"]
    if x >= float(c["fairly"]):
        return "fairly"
    if x >= float(c["somewhat"]):
        return "somewhat"
    return "close"


def confidence_word(x: float, ui: dict[str, Any]) -> str:
    return C.CONFIDENCE[confidence_key(x, ui)]


def strength_key(s: float, ui: dict[str, Any]) -> str:
    t = ui["strength"]
    if s >= float(t["strong"]):
        return "strong"
    if s >= float(t["moderate"]):
        return "moderate"
    return "weak"


def strength_word(s: float, ui: dict[str, Any]) -> str:
    return C.STRENGTH[strength_key(s, ui)]


# ---------------------------------------------------------------- the comparison
def _estimates(pred: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["drug"]: e for e in pred["estimates"]}


def best_costlier(pred: dict[str, Any]) -> dict[str, Any]:
    """The costlier drug with the larger estimated benefit over the reference."""
    return max((e for e in pred["estimates"] if e["benefit_vs_reference"] is not None),
               key=lambda e: e["benefit_vs_reference"])


def prices_set(pred: dict[str, Any]) -> bool:
    return not pred["prices_are_demo"]


@dataclass(frozen=True)
class HeadlineVM:
    kind: str                     # no_clear_winner | winner | discuss
    text: str
    confidence: str
    confidence_level: float       # the number the confidence word was read from
    secondary: str | None
    trial_finding: str
    suggested: str                # drug that gets the "Suggested" tag
    uncertainty: str = C.UNCERTAINTY


def headline(pred: dict[str, Any], ui: dict[str, Any], va: float | None = None) -> HeadlineVM:
    """One sentence that answers the question, plus how sure we are, in words.

    Confidence is measured for the headline's own claim: P(benefit > 1 line) of the suggested
    drug when there is a winner, and 1 - P(...) of the best costlier drug when there is not.
    """
    est = _estimates(pred)
    ref = est[REFERENCE]
    best = best_costlier(pred)
    p_best = float(best["p_benefit_gt_mcid"] or 0.0)
    rec = pred["recommendation"]
    finding = C.TRIAL_FINDING_POOR if va is not None and va < 69 else C.TRIAL_FINDING_GOOD

    no_winner = bool(pred["equipoise"]) or rec == REFERENCE
    if no_winner:
        secondary = (C.MAY_BENEFIT.format(drug=brand(best["drug"]))
                     if p_best >= float(ui["may_benefit_p"]) else None)
        x = 1.0 - p_best
        return HeadlineVM("no_clear_winner",
                          C.HEAD_NO_WINNER.format(ref_full=full_name(ref["drug"])),
                          confidence_word(x, ui), x, secondary, finding, REFERENCE)

    drug = est.get(rec, best)        # "Discuss" -> the best costlier drug
    x = float(drug["p_benefit_gt_mcid"] or 0.0)
    gain = lines_phrase(drug["benefit_vs_reference"], ui)
    # Without real prices the cost-based "worth it / discuss" split rests on DEMO ratios:
    # say only what the benefit estimate supports.
    if rec == "Discuss" and prices_set(pred):
        return HeadlineVM("discuss", C.HEAD_DISCUSS.format(drug_full=full_name(drug["drug"])),
                          confidence_word(x, ui), x, None, finding, drug["drug"])
    return HeadlineVM("winner", C.HEAD_WINNER.format(drug_full=full_name(drug["drug"]),
                                                     gain=gain, ref=brand(REFERENCE)),
                      confidence_word(x, ui), x, None, finding, drug["drug"])


@dataclass(frozen=True)
class DrugCardVM:
    drug: str
    name: str                     # "Avastin (bevacizumab)"
    tag: str | None
    improvement: str              # "About 2 lines"
    improvement_letters: str      # "≈ 10 letters"
    range_line: str               # "Compared with Avastin: likely ..." or the reference note
    injections: str
    cost: str | None              # None while prices are not set


def cards(pred: dict[str, Any], ui: dict[str, Any]) -> list[DrugCardVM]:
    est = _estimates(pred)
    head = headline(pred, ui)   # the tag depends only on the suggested drug
    show_cost = prices_set(pred)
    out = []
    for drug in CARD_ORDER:
        e = est[drug]
        if drug == head.suggested:
            tag = C.TAG_SUGGESTED
        elif drug == REFERENCE:
            tag = C.TAG_LOWEST_COST
        else:
            tag = None
        if e["benefit_vs_reference"] is None:
            rng = C.CARD_REFERENCE_NOTE
        else:
            rng = (C.RANGE_PREFIX.format(ref=brand(REFERENCE))
                   + range_text(e["benefit_low"], e["benefit_high"], ui))
        out.append(DrugCardVM(
            drug=drug, name=full_name(drug), tag=tag,
            improvement=gain_text(e["letters"], ui), improvement_letters=letters_text(e["letters"]),
            range_line=rng,
            injections=C.CARD_INJECTIONS.format(n=f"{round(e['injections_yr1']):.0f}"),
            cost=C.CARD_COST.format(amount=format_inr(e["cost"])) if show_cost else None))
    return out


@dataclass(frozen=True)
class ChartRow:
    drug: str
    label: str                    # brand
    lines: float                  # unrounded, for plotting
    low: float | None             # band (Avastin estimate + likely difference), lines
    high: float | None
    text: str                     # direct label: "About 2 lines"


@dataclass(frozen=True)
class ChartVM:
    rows: list[ChartRow]
    axis: str
    summary: str


def chart(pred: dict[str, Any], ui: dict[str, Any]) -> ChartVM:
    """Expected lines per drug; band = Avastin's estimate plus the likely difference."""
    est = _estimates(pred)
    lpl = float(ui["letters_per_line"])
    rows = []
    for drug in CARD_ORDER:
        e = est[drug]
        lo = None if e["letters_low"] is None else e["letters_low"] / lpl
        hi = None if e["letters_high"] is None else e["letters_high"] / lpl
        rows.append(ChartRow(drug, brand(drug), e["letters"] / lpl, lo, hi,
                             gain_text(e["letters"], ui)))
    parts = [C.CHART_SUMMARY_DRUG.format(drug=brand(d), gain=_lower_first(
        gain_text(est[d]["letters"], ui))) for d in CARD_ORDER]
    banded = [brand(d) for d in CARD_ORDER if est[d]["letters_low"] is not None]
    parts.append(C.CHART_SUMMARY_BANDS.format(drugs=f" {C.AND} ".join(banded),
                                              ref=brand(REFERENCE)))
    head = headline(pred, ui)
    parts.append(C.CHART_SUMMARY_NONE_AHEAD if head.kind == "no_clear_winner"
                 else C.CHART_SUMMARY_AHEAD.format(drug=brand(head.suggested)))
    return ChartVM(rows, C.CHART_AXIS, " ".join(parts))


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


# ---------------------------------------------------------------- form validation
@dataclass(frozen=True)
class FieldRange:
    lo: float
    hi: float


def validate(patient: dict[str, Any], ranges: dict[str, FieldRange]) -> list[str]:
    """Plain messages for out-of-range numbers and an impossible blood pressure."""
    msgs = []
    for name, r in ranges.items():
        v = patient.get(name)
        if v is None:
            continue
        shown, lo, hi = _display_value(name, float(v)), r.lo, r.hi
        if name == "X_log_uacr":
            lo, hi = math.expm1(lo), math.expm1(hi)
        if not (lo <= shown <= hi):
            f = C.FIELDS.get(name, {"label": name})
            unit = f" {f['unit']}" if f.get("unit") else ""
            msgs.append(C.VALIDATION_RANGE.format(label=f["label"], lo=num(round(lo, 1)),
                                                  hi=num(round(hi, 1)), unit=unit))
    sbp, dbp = patient.get("X_sbp"), patient.get("X_dbp")
    if sbp is not None and dbp is not None and float(sbp) <= float(dbp):
        msgs.append(C.VALIDATION_BP)
    return msgs


def _display_value(name: str, v: float) -> float:
    return math.expm1(v) if name == "X_log_uacr" else v


def to_form_value(name: str, v: Any) -> float | None:
    """Model value -> what the form shows (the kidney test is stored as log(1 + ratio))."""
    return None if v is None else float(_display_value(name, float(v)))


def from_form_value(name: str, v: float | None) -> float | None:
    """What the form shows -> model value."""
    if v is None:
        return None
    return math.log1p(float(v)) if name == "X_log_uacr" else float(v)


@dataclass(frozen=True)
class ComparisonVM:
    headline: HeadlineVM
    cards: list[DrugCardVM]
    chart: ChartVM
    cost_note: str | None = field(default=None)


def comparison(pred: dict[str, Any], ui: dict[str, Any], va: float | None) -> ComparisonVM:
    return ComparisonVM(headline(pred, ui, va), cards(pred, ui), chart(pred, ui),
                        None if prices_set(pred) else C.COSTS_HIDDEN_NOTE)


# ---------------------------------------------------------------- progress check
@dataclass(frozen=True)
class ProgressPoint:
    label: str
    week: float
    lines: float                  # lines gained since the start (unrounded, for plotting)
    text: str = ""


@dataclass(frozen=True)
class ProgressVM:
    drug: str
    observed: list[ProgressPoint]
    forecast: ProgressPoint
    error_lines: float | None
    axis: str
    label_expr: str               # Vega expression naming the visit weeks
    result: str
    change: str
    error: str | None
    summary: str


def progress(er: dict[str, Any], ui: dict[str, Any]) -> ProgressVM:
    """Plain words and chart data for the early-response forecast."""
    lpl = float(ui["letters_per_line"])
    base = float(er["baseline_va_letters"])
    traj = er["trajectory"]
    final_week = max(p["week"] for p in traj)
    names = dict(zip((0.0, 4.0, 8.0, 12.0, float(final_week)), C.PROGRESS_TIMES, strict=True))
    observed = [ProgressPoint(names.get(float(p["week"]), ""), float(p["week"]),
                              p["va_change"] / lpl) for p in traj if not p.get("forecast")]
    change = float(er["predicted_va_change_wk52"])
    forecast = ProgressPoint(names[float(final_week)], float(final_week), change / lpl,
                             gain_text(change, ui))
    n = letters_to_lines(change, ui)
    if n > 0:
        change_text = C.PROGRESS_BETTER.format(gain=lines_phrase(change, ui))
    elif n < 0:
        change_text = C.PROGRESS_WORSE.format(gain=lines_phrase(change, ui))
    else:
        change_text = C.PROGRESS_SAME
    err = er.get("typical_error_letters")
    expr = " : ".join(f"datum.value == {w:g} ? '{t}'" for w, t in names.items()) + " : ''"
    last = base + traj[[i for i, p in enumerate(traj) if not p.get("forecast")][-1]]["va_change"]
    return ProgressVM(
        drug=er["treatment"], observed=observed, forecast=forecast,
        error_lines=None if err is None else float(err) / lpl, axis=C.PROGRESS_AXIS,
        label_expr=expr,
        result=C.PROGRESS_RESULT.format(vision=snellen_text(er["predicted_va_wk52"], ui),
                                        letters=letters_text(er["predicted_va_wk52"])),
        change=change_text,
        error=None if err is None else C.PROGRESS_ERROR.format(gain=lines_phrase(err, ui)),
        summary=C.PROGRESS_SUMMARY.format(start=snellen_feet(base, ui),
                                          last=snellen_feet(last, ui)))


# ---------------------------------------------------------------- step 3: why
@dataclass(frozen=True)
class WhyVM:
    recap: str
    factors_intro: str | None
    factors: list[str]            # empty -> show C.WHAT_MATTERED_UNAVAILABLE
    decided: list[str]
    strength: str                 # strength sentence, or the "appears once prices" note


def _field_value_text(name: str, raw: str, ui: dict[str, Any]) -> str:
    """How a feature value from the explanation is shown in a sentence."""
    f = C.FIELDS.get(name, {})
    try:
        v = float(raw)
    except ValueError:
        return raw
    if name == "X_va_letters":
        return snellen_feet(v, ui)
    if name == "X_log_uacr":
        v = math.expm1(v)
    shown = f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else num(round(v, 1))
    unit = f.get("unit")
    if unit == "%":
        return f"{shown}%"
    return f"{shown} {unit}" if unit else shown


def factor_subject(name: str, raw: str, typical: Any, binary: bool,
                   ui: dict[str, Any]) -> str:
    """'A thicker retina (768 µm)', 'Uses insulin (Yes)', 'A typical value for HbA1c'."""
    f = C.FIELDS.get(name, {"label": name})
    if raw == "imputed":
        return C.SUBJECT_TYPICAL.format(label=f["label"])
    if binary:
        on = raw in ("1", "1.0")
        val = C.SEX_OPTIONS[int(on)] if name == "X_female" else (C.YES if on else C.NO)
        return C.SUBJECT_VALUE.format(phrase=f["label"], value=val)
    try:
        v = float(raw)
    except ValueError:                       # category
        return C.SUBJECT_VALUE.format(phrase=f["label"], value=raw)
    phrase = f["label"]
    if typical is not None and "more" in f:
        t = float(typical)
        if v > t:
            phrase = f["more"]
        elif v < t:
            phrase = f["less"]
    return C.SUBJECT_VALUE.format(phrase=phrase, value=_field_value_text(name, raw, ui))


def _size_word(shap_letters: float, ui: dict[str, Any]) -> str:
    a, t = abs(shap_letters), ui["factor_size_letters"]
    if a < float(t["small"]):
        return C.FACTOR_SIZE["small"]
    if a < float(t["medium"]):
        return C.FACTOR_SIZE["medium"]
    return C.FACTOR_SIZE["large"]


def factors(pred: dict[str, Any], ui: dict[str, Any], typical: dict[str, Any],
            binary: set[str], n: int = 3) -> list[str]:
    """Top-n contributions of the explanation (Eylea vs Avastin) as sentences."""
    expl = pred.get("explanation")
    if not expl:
        return []
    drug = brand(expl["comparison"].split(" - ")[0])
    top = sorted(expl["contributions"], key=lambda c: abs(c["shap_letters"]), reverse=True)[:n]
    out = []
    for c in top:
        subject = factor_subject(c["feature"], c["value"], typical.get(c["feature"]),
                                 c["feature"] in binary, ui)
        out.append(C.FACTOR.format(
            subject=subject, drug=drug, size=_size_word(c["shap_letters"], ui),
            direction=C.FACTOR_LARGER if c["shap_letters"] >= 0 else C.FACTOR_SMALLER))
    return out


def decided(pred: dict[str, Any], ui: dict[str, Any]) -> list[str]:
    """The decision rules in 2-4 short sentences."""
    mcid = float(pred["mcid_letters"])
    out = []
    comps = [e for e in pred["estimates"] if e["benefit_vs_reference"] is not None]
    for e in sorted(comps, key=lambda e: CARD_ORDER.index(e["drug"]), reverse=True):
        rng = range_text(e["benefit_low"], e["benefit_high"], ui)
        tpl = C.DECIDE_CLEAR if e["benefit_low"] >= mcid else C.DECIDE_NOT_CLEAR
        out.append(tpl.format(drug=brand(e["drug"]), ref=brand(REFERENCE), range=rng))
    for m, still_none in pred["equipoise_at_sensitivity"].items():
        lines = letters_to_lines(float(m), ui)
        bar = C.HALF_LINE if lines == 0.5 else f"{num(lines)} {line_unit(lines)}"
        out.append((C.DECIDE_LOWER_BAR_SAME if still_none else C.DECIDE_LOWER_BAR_DIFF)
                   .format(bar=bar))
    if prices_set(pred) and pred["recommendation"] != REFERENCE:
        out.append(C.DECIDE_WORTH if pred["recommendation"] != "Discuss"
                   else C.DECIDE_NOT_WORTH)
    if len(out) < 4:
        out.insert(0, C.DECIDE_ESTIMATE)
    return out[:4]


def strength_line(pred: dict[str, Any], ui: dict[str, Any]) -> str:
    """Strength of the case for paying more; hidden while prices are DEMO placeholders
    (the fuzzy layer uses the cost difference)."""
    if not prices_set(pred) or not pred.get("fuzzy_strength"):
        return C.STRENGTH_HIDDEN
    best = best_costlier(pred)["drug"]
    s = pred["fuzzy_strength"].get(best)
    if s is None:
        return C.STRENGTH_HIDDEN
    return C.STRENGTH_LINE.format(strength=strength_word(float(s), ui), drug=brand(best))


def why(pred: dict[str, Any], ui: dict[str, Any], va: float | None,
        typical: dict[str, Any], binary: set[str]) -> WhyVM:
    expl = pred.get("explanation")
    intro = None
    if expl:
        a, b = expl["comparison"].split(" - ")
        intro = C.WHAT_MATTERED_INTRO.format(drug=brand(a), ref=brand(b))
    return WhyVM(recap=headline(pred, ui, va).text, factors_intro=intro,
                 factors=factors(pred, ui, typical, binary), decided=decided(pred, ui),
                 strength=strength_line(pred, ui))


# ---------------------------------------------------------------- is the extra cost worth it?
@dataclass(frozen=True)
class BreakEvenVM:
    sentence: str
    at_price: str
    slider_max: float
    current: float


def _nbd_at(curve: list[dict[str, float]], price: float) -> float:
    """Read the returned curve at a price (it is linear in price, so this is exact)."""
    xs = [c["price"] for c in curve]
    ys = [c["net_benefit_difference"] for c in curve]
    return float(np.interp(price, xs, ys))


def break_even(be: dict[str, Any], price: float) -> BreakEvenVM | None:
    """Plain sentences from the /breakeven response; None if there is no curve (no value of a
    line set)."""
    curve = be.get("curve") or []
    if not curve:
        return None
    drug = brand(be["drug"])
    res = be.get("result") or {}
    bep = res.get("break_even_price")
    top = float(curve[-1]["price"])
    if _nbd_at(curve, 0.0) <= 0:
        sentence = C.BE_NEVER.format(drug=drug)
    elif bep is not None and bep > 0:
        sentence = C.BE_SENTENCE.format(drug=drug, price=format_inr(bep))
    else:
        sentence = C.BE_ALWAYS.format(drug=drug, price=format_inr(top))
    worth = _nbd_at(curve, price) > 0
    at = (C.BE_AT_PRICE_WORTH if worth else C.BE_AT_PRICE_NOT).format(
        price=format_inr(price), drug=drug)
    return BreakEvenVM(sentence, at, top, float(be["current_price"]))


# ---------------------------------------------------------------- about the trial
def _triple(s: str) -> tuple[float, float, float] | None:
    """'13.3 / 9.4 / 11.2' (Eylea / Avastin / Lucentis, as in the replication table)."""
    try:
        a, b, c = (float(x) for x in str(s).split("/"))
    except ValueError:
        return None
    return a, b, c


def about_findings(table: dict[str, str], ui: dict[str, Any]) -> list[str] | None:
    """Plain sentences from the 1-year rows of reports/tables/m2_replication_vs_published
    (our replication of the published results); None if the table is not available."""
    rows = {k: _triple(v) for k, v in table.items()}
    keys = ("mean_letters_gained", "mean_letters_gained_va_below_69",
            "mean_letters_gained_va_69_or_better", "median_injections")
    if any(rows.get(k) is None for k in keys):
        return None

    def lines3(k: str) -> dict[str, str]:
        afl, bev, ran = rows[k]
        return {"afl": lines_phrase(afl, ui), "bev": lines_phrase(bev, ui),
                "ran": lines_phrase(ran, ui)}

    inj = sorted({int(round(x)) for x in rows["median_injections"]})
    inj_text = num(inj[0]) if len(inj) == 1 else f"{inj[0]} {C.TO} {inj[-1]}"
    return [C.ABOUT_FOUND_ALL.format(**lines3("mean_letters_gained")),
            C.ABOUT_FOUND_POOR.format(**lines3("mean_letters_gained_va_below_69")),
            C.ABOUT_FOUND_GOOD.format(**lines3("mean_letters_gained_va_69_or_better")),
            C.ABOUT_FOUND_INJ.format(inj=inj_text)]
