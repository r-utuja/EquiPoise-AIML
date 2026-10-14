"""Progress check: how is a patient on treatment doing so far? (early-response forecast).

A different question from the suggestion; it uses vision measured after treatment started.
"""

from __future__ import annotations

import streamlit as st

from app import charts, services
from app import copy as C
from app import presenters as P

UI = services.UI
SS = st.session_state

st.title(C.PROGRESS_TITLE)
services.notice()


def vision_input(label: str, key: str, letters: float, exact: bool) -> float:
    wkey = services.WIDGET_PREFIX + key
    if exact:
        if wkey + "_l" not in SS:
            SS[wkey + "_l"] = float(letters)
        v = st.number_input(label, key=wkey + "_l", step=1.0, format="%.0f",
                            min_value=0.0, max_value=100.0, help=C.EXACT_HELP)
        return float(v)
    opts = dict(P.snellen_options(UI))
    if wkey + "_s" not in SS:
        SS[wkey + "_s"] = P.nearest_snellen(letters, UI)
    d = st.selectbox(label, list(opts), format_func=opts.__getitem__, key=wkey + "_s")
    return P.snellen_to_letters(d)


if services.model_ready():
    st.markdown(C.PROGRESS_INTRO)
    with st.container(border=True):
        if "w_progress_drug" not in SS:
            SS["w_progress_drug"] = SS.get("progress_drug", P.REFERENCE)
        drug = st.segmented_control(C.PROGRESS_DRUG, P.CARD_ORDER, format_func=P.full_name,
                                    key="w_progress_drug") or P.REFERENCE
        SS["progress_drug"] = drug
        known = SS.get(services.PATIENT)
        if "w_progress_exact" not in SS:    # an exact start score keeps visits exact too
            SS["w_progress_exact"] = bool(known) and not P.is_snellen_value(
                known["X_va_letters"], UI)
        exact = st.toggle(C.PROGRESS_EXACT, key="w_progress_exact")
        if services.PATIENT in SS:
            patient = dict(SS[services.PATIENT])
            start = float(patient["X_va_letters"])
            st.caption(C.PROGRESS_START_FROM.format(vision=P.snellen_text(start, UI)))
        else:
            patient = services.default_patient()
            start = vision_input(C.PROGRESS_START, "progress_start", patient["X_va_letters"],
                                 exact)
            patient["X_va_letters"] = start
        cols = st.columns(3)
        visits = []
        for i, (col, label) in enumerate(zip(cols, C.VISIT_LABELS, strict=True)):
            with col:
                visits.append(vision_input(label, f"progress_v{i}", start, exact))
        if st.button(C.PROGRESS_BTN, type="primary", icon=":material/trending_up:"):
            SS["progress_payload"] = {"patient": patient, "treatment": drug,
                                      "va_wk4": visits[0], "va_wk8": visits[1],
                                      "va_wk12": visits[2]}

    payload = SS.get("progress_payload")
    if payload:
        try:
            with st.spinner(C.SPINNER_PROGRESS):
                er = services.early_response(payload)
        except services.ClientError:
            st.info(C.PROGRESS_UNAVAILABLE)
        except services.AppUnavailable:
            st.error(C.ERROR_MODEL)
        else:
            vm = P.progress(er, UI)
            with st.container(border=True):
                st.markdown(f'<p class="eq-headline">{vm.result}</p>', unsafe_allow_html=True)
                st.markdown(vm.change)
                if vm.error:
                    st.caption(vm.error)
            st.altair_chart(charts.progress(vm, services.chart_ink()), width="stretch")
            st.caption(vm.summary)
