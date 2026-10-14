"""Home: what the app does, how it works, and one button to start."""

from __future__ import annotations

import streamlit as st

from app import copy as C
from app import services

st.title(C.APP_NAME)
services.notice()
st.markdown(f"#### {C.HOME_TAGLINE}")
st.space("small")

st.subheader(C.HOW_IT_WORKS)
cols = st.columns(len(C.HOME_STEPS))
for i, (col, (icon, title, text)) in enumerate(zip(cols, C.HOME_STEPS, strict=True), start=1):
    with col.container(border=True, height="stretch"):
        st.markdown(f"{icon} **{i}. {title}**")
        st.caption(text)

st.space("small")
if st.button(C.HOME_CTA, type="primary", icon=":material/arrow_forward:"):
    services.go_to_wizard(step=1)
links = st.container(horizontal=True, gap="medium")
with links:
    st.page_link(services.SUGGEST_PAGE, label=C.HOME_EXAMPLE_LINK, icon=":material/person:")
    st.page_link("views/about.py", label=C.HOME_ABOUT_LINK, icon=":material/menu_book:")
