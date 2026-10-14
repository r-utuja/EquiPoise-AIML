"""Demo login: accounts file, password hashing, and the login gate in the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import copy as C

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture()
def users_file(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "users.json"
    monkeypatch.setenv("EQUIPOISE_USERS_FILE", str(p))
    monkeypatch.setenv("EQUIPOISE_AUTH", "on")
    return p


def test_sign_up_and_log_in(users_file: Path) -> None:
    from app import auth

    assert auth.sign_up("Asha", "Asha@Example.com", "secret1") is None
    stored = json.loads(users_file.read_text(encoding="utf-8"))
    assert "secret1" not in users_file.read_text(encoding="utf-8")   # only a hash is kept
    assert set(stored["asha@example.com"]) == {"name", "salt", "hash"}
    assert auth.log_in("asha@example.com", "secret1") == {"name": "Asha",
                                                          "email": "asha@example.com"}
    assert auth.log_in("asha@example.com", "wrong") is None
    assert auth.log_in("nobody@example.com", "secret1") is None
    assert auth.sign_up("A", "asha@example.com", "secret1") == C.AUTH_EXISTS
    assert auth.sign_up("A", "not-an-email", "secret1") == C.AUTH_BAD_EMAIL
    assert auth.sign_up("A", "b@example.com", "123") == C.AUTH_SHORT_PASSWORD.format(n=6)
    assert auth.sign_up("", "b@example.com", "secret1") == C.AUTH_FILL_ALL


def test_app_login_gate_sign_up_and_log_out(users_file: Path, monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("EQUIPOISE_REQUEST_LOG", "off")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert C.NAV_HOME not in [t.value for t in at.title]          # login page, not Home
    assert [t.label for t in at.tabs] == [C.AUTH_LOGIN, C.AUTH_SIGNUP]

    # wrong login
    at.text_input[0].set_value("asha@example.com")
    at.text_input[1].set_value("secret1")
    next(b for b in at.button if b.label == C.AUTH_LOGIN).click().run()
    assert C.AUTH_WRONG in [e.value for e in at.error]

    # sign up -> logged in, name in the sidebar
    at.text_input[2].set_value("Asha")
    at.text_input[3].set_value("asha@example.com")
    at.text_input[4].set_value("secret1")
    next(b for b in at.button if b.label == C.AUTH_SIGNUP).click().run()
    assert not at.exception, at.exception
    assert at.session_state["auth_user"]["name"] == "Asha"
    assert any("Asha" in m.value for m in at.sidebar.markdown)

    # log out -> back to the login page; log in again with the new account
    next(b for b in at.sidebar.button if b.label == C.AUTH_LOGOUT).click().run()
    assert "auth_user" not in at.session_state
    at.text_input[0].set_value("asha@example.com")
    at.text_input[1].set_value("secret1")
    next(b for b in at.button if b.label == C.AUTH_LOGIN).click().run()
    assert not at.exception, at.exception
    assert at.session_state["auth_user"]["email"] == "asha@example.com"
