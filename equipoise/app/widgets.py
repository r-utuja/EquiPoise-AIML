"""Page pieces shared by more than one main-flow page (step 3 and Price check).
Wording from app/copy.py, numbers from app/presenters.py."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import copy as C
from app import presenters as P
from app import services

COSTLIER = ("Aflibercept", "Ranibizumab")
PRICE_PAGE = "views/price_check.py"


def _persisted(widget_key: str, state_key: str, default: Any) -> str:
    """Seed a widget from a non-widget session key so it survives page changes."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(state_key, default)
    return widget_key


def worth_it(features: dict[str, Any], show_link: bool = True) -> None:
    """'Is the extra cost worth it?': one price slider and one sentence."""
    with st.container(border=True):
        st.markdown(f"**{C.WORTH_TITLE}**")
        if not services.prices_set():
            st.caption(C.COSTS_HIDDEN_NOTE)
            if show_link:
                st.page_link(PRICE_PAGE, label=C.BE_GO_PRICES, icon=":material/payments:")
            return
        key = _persisted("w_compare_with", "compare_with", COSTLIER[0])
        drug = st.segmented_control(C.COMPARE_WITH.format(ref=P.brand(P.REFERENCE)),
                                    COSTLIER, format_func=P.brand, key=key) or COSTLIER[0]
        st.session_state["compare_with"] = drug
        vkey = _persisted("w_value_per_line", services.VALUE_PER_LINE, None)
        vpl = st.number_input(C.VALUE_LINE_LABEL, min_value=0.0, step=1000.0, format="%.0f",
                              key=vkey, help=C.VALUE_LINE_HELP, placeholder="0")
        st.session_state[services.VALUE_PER_LINE] = vpl or None
        try:
            with st.spinner(C.SPINNER_PRICE):
                be = services.breakeven(features, drug)
        except (services.ClientError, services.AppUnavailable):
            st.error(C.ERROR_MODEL)
            return
        first = P.break_even(be, float(be["current_price"]))
        if first is None:
            st.caption(C.BE_NEED_VALUE)
            return
        step = max(1.0, round(first.slider_max / 200.0, -1))
        price = st.slider(C.PRICE_SLIDER.format(drug=P.brand(drug)), 0.0,
                          float(first.slider_max), float(min(first.current, first.slider_max)),
                          step=step, format="₹%d", key=f"w_price_{drug}")
        vm = P.break_even(be, price)
        st.markdown(f"**{vm.sentence}**")
        st.markdown(vm.at_price)
