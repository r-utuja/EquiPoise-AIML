"""About the trial: what Protocol T was, why it allows a fair comparison, what it found
(from our replication table) and what the app can and can't tell you."""

from __future__ import annotations

import streamlit as st

from app import copy as C
from app import presenters as P
from app import services

st.title(C.ABOUT_TITLE)
services.notice()

st.subheader(C.ABOUT_WHAT_TITLE)
st.markdown(C.ABOUT_WHAT)

st.subheader(C.ABOUT_FAIR_TITLE)
st.markdown(C.ABOUT_FAIR)

st.subheader(C.ABOUT_FOUND_TITLE)
found = P.about_findings(services.replication_1yr(), services.UI)
if found:
    st.markdown("\n".join(f"- {s}" for s in found))
else:
    st.caption(C.ABOUT_FOUND_MISSING)

st.subheader(C.ABOUT_LIMITS_TITLE)
st.markdown("\n".join(f"- {s}" for s in C.ABOUT_LIMITS))
