"""Price check: is a costlier medicine worth it for the patient already entered?

Shows nothing derived from the DEMO placeholder prices: until the team's prices are in
config/costs.yaml, the page offers an optional form for prices used in this session only.
"""

from __future__ import annotations

import streamlit as st

from app import copy as C
from app import presenters as P
from app import services, widgets

SS = st.session_state

st.title(C.PRICE_TITLE)
services.notice()


def session_prices_form() -> None:
    with st.container(border=True):
        st.markdown(f"**{C.SESSION_PRICES_TITLE}**")
        st.caption(C.SESSION_PRICES_HELP)
        with st.form("session_prices", border=False):
            cols = st.columns(2)
            vals = {}
            for i, drug in enumerate(P.CARD_ORDER):
                vals[drug] = cols[i % 2].number_input(C.PRICE_FIELD.format(drug=P.brand(drug)),
                                                      min_value=0.0, step=500.0, format="%.0f",
                                                      value=None, placeholder="0")
            laser = cols[1].number_input(C.LASER_FIELD, min_value=0.0, step=500.0,
                                         format="%.0f", value=None, placeholder="0")
            if st.form_submit_button(C.BTN_USE_PRICES, type="primary"):
                if any(not v for v in vals.values()):
                    st.error(C.PRICES_INVALID)
                else:
                    SS[services.SESSION_PRICES] = {"per_injection": vals,
                                                   "laser": float(laser or 0.0)}
                    st.rerun()


if services.model_ready():
    st.markdown(C.PRICE_INTRO)
    if services.team_prices_set():
        st.caption(C.PRICES_TEAM)
    elif SS.get(services.SESSION_PRICES):
        row = st.container(horizontal=True, vertical_alignment="center")
        row.caption(C.PRICES_SESSION)
        if row.button(C.BTN_CLEAR_PRICES, type="tertiary"):
            SS.pop(services.SESSION_PRICES, None)
            st.rerun()
    else:
        st.info(C.COSTS_HIDDEN_NOTE)
        session_prices_form()

    if services.PATIENT not in SS:
        st.caption(C.NEED_PATIENT)
        if st.button(C.BTN_GO_STEP1, icon=":material/arrow_forward:"):
            services.go_to_wizard(step=1)
    elif services.prices_set():
        widgets.worth_it(SS[services.PATIENT], show_link=False)
