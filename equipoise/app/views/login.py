"""Log in / sign up: the first page until someone is logged in (see app/auth.py)."""

from __future__ import annotations

import streamlit as st

from app import auth, services
from app import copy as C

# after a log-out the browser keeps the (now empty) sidebar open: hide it here
st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']"
            "{display:none}</style>", unsafe_allow_html=True)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.title(C.APP_NAME)
    services.notice()
    login_tab, signup_tab = st.tabs([C.AUTH_LOGIN, C.AUTH_SIGNUP])

    with login_tab, st.form("login_form"):
        email = st.text_input(C.AUTH_EMAIL)
        password = st.text_input(C.AUTH_PASSWORD, type="password")
        if st.form_submit_button(C.AUTH_LOGIN, type="primary", width="stretch"):
            user = auth.log_in(email, password)
            if user is None:
                st.error(C.AUTH_WRONG)
            else:
                auth.set_user(user)
                st.rerun()

    with signup_tab, st.form("signup_form"):
        name = st.text_input(C.AUTH_NAME)
        email = st.text_input(C.AUTH_EMAIL)
        password = st.text_input(C.AUTH_PASSWORD, type="password")
        if st.form_submit_button(C.AUTH_SIGNUP, type="primary", width="stretch"):
            err = auth.sign_up(name, email, password)
            if err:
                st.error(err)
            else:
                auth.set_user(auth.log_in(email, password))
                st.rerun()
