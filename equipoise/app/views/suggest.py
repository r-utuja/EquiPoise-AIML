"""Get a suggestion: a three-step wizard (patient details -> comparison -> why).

All state lives in st.session_state (see app/services.py), so going back never loses input.
Layout only: wording comes from app/copy.py and numbers from app/presenters.py.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import charts, services, widgets
from app import copy as C
from app import presenters as P
from app.pdf import summary_pdf
from app.theme import swatch
from equipoise.serving.service import ATTRIBUTION

UI = services.UI
SS = st.session_state
W = services.WIDGET_PREFIX

FORMATS = {"X_hba1c": "%.1f", "X_bmi": "%.1f"}          # everything else: whole numbers


# ---------------------------------------------------------------- shared bits
def progress_header(step: int) -> None:
    st.markdown(f'<p class="eq-step">{C.STEP_LABEL.format(n=step, name=C.STEP_NAMES[step - 1])}'
                "</p>", unsafe_allow_html=True)
    st.progress(step / 3)


def onboarding_tip() -> None:
    if SS.get(services.TIP_DISMISSED):
        return
    with st.container(border=True):
        st.markdown(f"**{C.TIP_TITLE}**")
        st.markdown("  \n".join(C.TIP_LINES))
        st.button(C.TIP_DISMISS, key="tip_ok", type="tertiary",
                  on_click=lambda: SS.__setitem__(services.TIP_DISMISSED, True))


# ---------------------------------------------------------------- step 1: form widgets
def _seed(name: str, value: Any) -> str:
    key = W + name
    if key not in SS:
        SS[key] = value
    return key


def _label(name: str) -> str:
    f = C.FIELDS[name]
    return f"{f['label']} ({f['unit']})" if f.get("unit") else f["label"]


def field(name: str, value: Any) -> Any:
    """One input bound to session state; returns the model-ready value."""
    spec = services.feature_specs()[name]
    f = C.FIELDS[name]
    if spec["categories"]:
        opts = [*spec["categories"], C.NOT_KNOWN]
        key = _seed(name, value if value is not None else C.NOT_KNOWN)
        v = st.selectbox(_label(name), opts, key=key, help=f["help"])
        return None if v == C.NOT_KNOWN else v
    if spec["binary"]:
        opts = [C.NO, C.YES, C.NOT_KNOWN]
        key = _seed(name, C.NOT_KNOWN if value is None else (C.YES if value else C.NO))
        v = st.selectbox(_label(name), opts, key=key, help=f["help"])
        return None if v == C.NOT_KNOWN else int(v == C.YES)
    shown = P.to_form_value(name, value)          # e.g. the kidney test is asked as the ratio
    key = _seed(name, shown)
    v = st.number_input(_label(name), key=key, help=f["help"], step=1.0,
                        format=FORMATS.get(name, "%.0f"), placeholder=C.NOT_KNOWN)
    return P.from_form_value(name, v)


def sex_field(value: Any) -> Any:
    key = _seed("X_female", None if value is None else C.SEX_OPTIONS[int(value)])
    v = st.segmented_control(_label("X_female"), C.SEX_OPTIONS, key=key,
                             help=C.FIELDS["X_female"]["help"])
    return None if v is None else C.SEX_OPTIONS.index(v)


def vision_field(letters: float) -> float:
    exact_key = _seed("va_exact", bool(SS.get(services.VA_EXACT, False)))
    if SS[exact_key]:
        v = st.number_input(C.EXACT_LABEL, key=_seed("va_letters", float(letters)), step=1.0,
                            format="%.0f", help=C.EXACT_HELP)
        out = float(v) if v is not None else letters
        st.caption(C.EXACT_NEAREST.format(vision=P.snellen_text(out, UI)))
    else:
        opts = dict(P.snellen_options(UI))
        d = st.selectbox(C.VISION_LABEL, list(opts), format_func=opts.__getitem__,
                         key=_seed("va_snellen", P.nearest_snellen(letters, UI)),
                         help=C.VISION_HELP)
        out = P.snellen_to_letters(d)
    st.toggle(C.EXACT_TOGGLE, key=exact_key, help=C.EXACT_TOGGLE_HELP,
              on_change=_exact_changed)
    return out


def _exact_changed() -> None:
    """Keep the shown vision in step when switching between the chart and the exact score."""
    SS[services.VA_EXACT] = SS[W + "va_exact"]
    SS.pop(W + "va_letters", None)
    SS.pop(W + "va_snellen", None)


def step1() -> None:
    progress_header(1)
    onboarding_tip()
    patient = services.patient()

    ex = services.examples()
    if ex:
        st.markdown(f"**{C.EXAMPLES_LABEL}**")
        row = st.container(horizontal=True)
        for k in UI["examples"]:
            if k in ex:
                row.button(C.EXAMPLE_NAMES[k], key=f"ex_{k}", help=C.EXAMPLE_HELP[k],
                           on_click=services.load_example, args=(k,),
                           icon=":material/person:")
        if SS.get(services.EXAMPLE) in ex:
            st.caption(C.EXAMPLE_LOADED.format(name=C.EXAMPLE_NAMES[SS[services.EXAMPLE]]))

    new: dict[str, Any] = {}
    with st.container(border=True):
        st.markdown(f"**{C.ESSENTIALS_TITLE}**")
        new["X_va_letters"] = vision_field(patient["X_va_letters"])
        pairs = [("X_cst_um", "X_oct_machine"), ("X_hba1c", "X_diabetes_duration_yrs"),
                 ("X_age", "X_female"), ("X_sbp", "X_dbp")]
        for a, b in pairs:
            c1, c2 = st.columns(2)
            with c1:
                new[a] = field(a, patient.get(a))
            with c2:
                new[b] = sex_field(patient.get(b)) if b == "X_female" else field(b, patient.get(b))

    with st.expander(C.MORE_TITLE):
        st.caption(C.MORE_NOTE)
        for group, names in C.OPTIONAL_GROUPS.items():
            st.markdown(f"**{group}**")
            cols = st.columns(2)
            for i, n in enumerate(names):
                with cols[i % 2]:
                    new[n] = field(n, patient.get(n))
    SS[services.PATIENT] = services.clean_features({**patient, **new})

    if SS.get("errors"):
        st.error("\n".join([C.VALIDATION_TITLE, *[f"- {m}" for m in SS["errors"]]]))

    bar = st.container(horizontal=True, horizontal_alignment="right")
    bar.button(C.BTN_START_OVER, on_click=services.start_over)
    if bar.button(C.BTN_COMPARE, type="primary", icon=":material/arrow_forward:"):
        compare()


def compare() -> None:
    feats = SS[services.PATIENT]
    ranges = {n: P.FieldRange(s["min"], s["max"]) for n, s in services.feature_specs().items()
              if s["min"] is not None and s["max"] is not None}
    errors = P.validate(feats, ranges)
    SS["errors"] = errors
    if errors:
        st.rerun()
    try:
        with st.spinner(C.SPINNER_COMPARE):
            services.predict(feats, explain=False)
    except services.ClientError:
        st.error(C.ERROR_REJECTED)
        return
    except services.AppUnavailable:
        st.error(C.ERROR_MODEL)
        return
    services.set_step(2)
    st.rerun()


# ---------------------------------------------------------------- step 2: the comparison
def current_prediction(explain: bool = False) -> dict[str, Any] | None:
    feats = SS.get(services.PATIENT)
    if feats is None:
        return None
    spinner = C.SPINNER_WHY if explain else C.SPINNER_COMPARE
    try:
        if services.cached_prediction(feats, explain) is None:
            with st.spinner(spinner):
                return services.predict(feats, explain)
        return services.predict(feats, explain)
    except (services.ClientError, services.AppUnavailable):
        st.error(C.ERROR_MODEL)
        return None


def headline_card(h: P.HeadlineVM) -> None:
    with st.container(border=True):
        st.markdown(f'<p class="eq-headline">{h.text}</p>', unsafe_allow_html=True)
        st.markdown(f"**{h.confidence}**")
        if h.secondary:
            st.markdown(h.secondary)
        st.caption(h.trial_finding)


def drug_card(card: P.DrugCardVM) -> None:
    with st.container(border=True, height="stretch"):
        st.markdown(f"{swatch(card.drug)}**{card.name}**", unsafe_allow_html=True)
        if card.tag:      # same row height with or without a tag, so the cards line up
            color = "blue" if card.tag == C.TAG_SUGGESTED else "gray"
            st.markdown(f":{color}-badge[{card.tag}]")
        else:
            st.markdown("&nbsp;")
        st.caption(C.CARD_IMPROVE_LABEL)
        st.markdown(f'<p class="eq-big">{card.improvement}</p>', unsafe_allow_html=True)
        st.caption(card.improvement_letters)
        st.markdown(card.range_line)
        st.markdown(card.injections)
        if card.cost:
            st.markdown(card.cost)


def step2() -> None:
    progress_header(2)
    pred = current_prediction()
    if pred is None:
        services.set_step(1)
        st.rerun()
    vm = P.comparison(pred, UI, SS[services.PATIENT]["X_va_letters"])
    headline_card(vm.headline)
    cols = st.columns(3)
    for col, card in zip(cols, vm.cards, strict=True):
        with col:
            drug_card(card)
    st.altair_chart(charts.improvement(vm.chart, services.chart_ink()), width="stretch")
    st.caption(vm.chart.summary)
    if vm.cost_note:
        st.caption(vm.cost_note)
    st.caption(vm.headline.uncertainty)
    if services.health()["synthetic_bundle"]:
        st.caption(C.DEMO_MODEL)

    bar = st.container(horizontal=True, horizontal_alignment="right")
    bar.button(C.BTN_EDIT, on_click=services.set_step, args=(1,), icon=":material/arrow_back:")
    bar.button(C.BTN_START_OVER, on_click=services.start_over)
    bar.button(C.BTN_SEE_WHY, type="primary", on_click=services.set_step, args=(3,),
               icon=":material/arrow_forward:")


# ---------------------------------------------------------------- step 3: why
def step3() -> None:
    progress_header(3)
    feats = SS[services.PATIENT]
    pred = current_prediction(explain=True)
    if pred is None:
        return
    va = feats["X_va_letters"]
    specs = services.feature_specs()
    binary = {n for n, s in specs.items() if s["binary"]} | {services.DERIVED_FEATURE}
    typical = services.meta()["defaults"]
    vm = P.why(pred, UI, va, typical, binary)
    st.markdown(f"**{vm.recap}**")

    with st.container(border=True):
        st.markdown(f"**{C.WHAT_MATTERED_TITLE}**")
        if vm.factors:
            st.markdown(vm.factors_intro)
            st.markdown("\n".join(f"- {f}" for f in vm.factors))
        else:
            st.caption(C.WHAT_MATTERED_UNAVAILABLE)
    with st.container(border=True):
        st.markdown(f"**{C.HOW_DECIDED_TITLE}**")
        st.markdown("\n".join(f"- {d}" for d in vm.decided))
        st.caption(vm.strength)
    widgets.worth_it(feats)

    cmp = P.comparison(pred, UI, va)
    cst = feats.get("X_cst_um")
    patient_line = C.PDF_PATIENT_LINE.format(
        vision=P.snellen_text(va, UI),
        cst=f"{cst:.0f} µm" if cst is not None else C.NOT_KNOWN.lower())
    bar = st.container(horizontal=True, horizontal_alignment="right")
    bar.button(C.BTN_BACK_COMPARISON, on_click=services.set_step, args=(2,),
               icon=":material/arrow_back:")
    bar.button(C.BTN_START_OVER, on_click=services.start_over)
    bar.download_button(C.PDF_BUTTON, data=summary_pdf(cmp, vm, patient_line, ATTRIBUTION),
                        file_name=C.PDF_FILE, mime="application/pdf", type="primary",
                        icon=":material/download:", on_click="ignore")


# ---------------------------------------------------------------- page
st.title(C.WIZARD_TITLE)
services.notice()
if services.model_ready():
    step = int(SS.get(services.STEP, 1)) if services.PATIENT in SS else 1
    {1: step1, 2: step2, 3: step3}[step]()
