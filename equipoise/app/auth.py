"""Minimal email + password accounts for the Streamlit app (a demo login, not real security).

Accounts live in a local JSON file under data/ (git-ignored, never shipped in Docker images).
Passwords are stored only as salted PBKDF2-SHA256 hashes from the standard library. The
logged-in user is kept in the Streamlit session, so reloading the page logs out.
Set `EQUIPOISE_AUTH=off` to skip the login (tests, scripted screenshots).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

import streamlit as st

from app import copy as C
from app import services

ROOT = Path(__file__).resolve().parents[1]
AUTH = services.UI["auth"]
USER = "auth_user"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def enabled() -> bool:
    return os.environ.get(AUTH["env_var"], "on").strip().lower() not in {"0", "off", "false",
                                                                         "no"}


def users_path() -> Path:
    p = Path(os.environ.get(AUTH["users_file_env_var"]) or AUTH["users_file"])
    return p if p.is_absolute() else ROOT / p


def _load() -> dict[str, dict[str, Any]]:
    p = users_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(users: dict[str, dict[str, Any]]) -> None:
    p = users_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(users, indent=1), encoding="utf-8")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt),
                               int(AUTH["hash_iterations"])).hex()


def sign_up(name: str, email: str, password: str) -> str | None:
    """Create an account; returns an error message, or None on success."""
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        return C.AUTH_FILL_ALL
    if not EMAIL_RE.match(email):
        return C.AUTH_BAD_EMAIL
    if len(password) < int(AUTH["min_password_length"]):
        return C.AUTH_SHORT_PASSWORD.format(n=int(AUTH["min_password_length"]))
    users = _load()
    if email in users:
        return C.AUTH_EXISTS
    salt = secrets.token_hex(16)
    users[email] = {"name": name, "salt": salt, "hash": _hash(password, salt)}
    _save(users)
    return None


def log_in(email: str, password: str) -> dict[str, str] | None:
    """The user ({'name', 'email'}) if the password matches, else None."""
    email = email.strip().lower()
    u = _load().get(email)
    if u is None or not hmac.compare_digest(u["hash"], _hash(password, u["salt"])):
        return None
    return {"name": u["name"], "email": email}


def current_user() -> dict[str, str] | None:
    return st.session_state.get(USER)


def set_user(user: dict[str, str]) -> None:
    st.session_state[USER] = user


def log_out() -> None:
    st.session_state.pop(USER, None)


def sidebar_profile() -> None:
    """Profile icon, name and a log-out button at the bottom of the navigation."""
    user = current_user()
    if not user:
        return
    with st.sidebar:
        st.divider()
        row = st.container(horizontal=True, vertical_alignment="center")
        row.markdown(f":material/account_circle: **{user['name']}**")
        if row.button(C.AUTH_LOGOUT, icon=":material/logout:", type="tertiary"):
            log_out()
            st.rerun()
