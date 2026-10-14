"""Equipoise clinician app (Streamlit). Research prototype, not for clinical use.

Run: `streamlit run app/streamlit_app.py`. When `API_URL` is set the app calls the FastAPI
service (and falls back to the local bundle if it can't be reached); otherwise it loads the
model bundle itself. This file only builds the navigation: pages live in `app/views/`, their
wording in `app/copy.py` and all model logic in the `equipoise` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # so `from app import ...` works under `streamlit run`
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import auth, services  # noqa: E402
from app import copy as C  # noqa: E402

VIEWS = Path(__file__).resolve().parent / "views"

st.set_page_config(page_title=C.APP_NAME, page_icon=":material/visibility:", layout="wide")
services.apply_theme()

if auth.enabled() and not auth.current_user():     # login page first, no navigation
    st.navigation([st.Page(VIEWS / "login.py", title=C.AUTH_LOGIN,
                           icon=":material/login:")], position="hidden").run()
    services.footer()
    st.stop()


@st.cache_data(show_spinner=False)
def oct_available() -> bool:
    """The OCT page appears only when the [cv] extra is installed and M10 has been trained."""
    try:
        from equipoise.vision.infer import available
    except ImportError:
        return False
    return available(services.CFG)


main = [
    st.Page(VIEWS / "home.py", title=C.NAV_HOME, icon=":material/home:", default=True),
    st.Page(VIEWS / "suggest.py", title=C.NAV_SUGGEST, icon=":material/compare_arrows:"),
    st.Page(VIEWS / "price_check.py", title=C.NAV_PRICE, icon=":material/payments:"),
    st.Page(VIEWS / "progress.py", title=C.NAV_PROGRESS, icon=":material/trending_up:"),
    st.Page(VIEWS / "about.py", title=C.NAV_ABOUT, icon=":material/menu_book:"),
]
review = []
if oct_available():
    review.append(st.Page(VIEWS / "oct.py", title=C.NAV_OCT, icon=":material/image:"))
sections = {C.NAV_SECTION_MAIN: main}
if review:
    sections[C.NAV_SECTION_REVIEW] = review
page = st.navigation(sections)
auth.sidebar_profile()
page.run()
services.footer()
