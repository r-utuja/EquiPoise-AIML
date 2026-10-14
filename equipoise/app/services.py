"""Model calls, caching and session state for the Streamlit pages.

The pages never talk to the model directly. `client()` returns one shared client that uses
the FastAPI service when `API_URL` is set and quietly falls back to the local model bundle if
the API can't be reached; only when both fail does a page show an error. Results are cached
per (patient, prices) so moving between steps or pages never recomputes anything.
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from app import copy as C
from app import theme
from equipoise.config import load_config
from equipoise.serving.client import ApiClient, ClientError, LocalClient
from equipoise.serving.service import ATTRIBUTION, Service

CFG = load_config()
UI = CFG.raw["ui"]


class AppUnavailable(RuntimeError):
    """Neither the API nor the local model bundle could answer."""


class FallbackClient:
    """API first (when configured), then the local bundle. Validation errors are re-raised
    as `ClientError`; anything else from the API switches to the local bundle for good."""

    def __init__(self) -> None:
        s = CFG.raw["serving"]
        url = os.environ.get(s["api_url_env_var"])
        self._api = ApiClient(url, float(s["request_timeout_s"])) if url else None
        self._local: LocalClient | None = None

    @property
    def mode(self) -> str:
        return self._api.mode if self._api is not None else LocalClient.mode

    def _local_client(self) -> LocalClient:
        if self._local is None:
            try:
                self._local = LocalClient(Service.from_env(CFG))
            except Exception as e:   # missing bundle, unreadable file, ...
                raise AppUnavailable(str(e)) from e
        return self._local

    def call(self, name: str, *args: Any) -> dict[str, Any]:
        if self._api is not None:
            try:
                return getattr(self._api, name)(*args)
            except ClientError:
                raise
            except Exception:        # API unreachable: use the bundle from now on
                self._api = None
        return getattr(self._local_client(), name)(*args)


@st.cache_resource(show_spinner=C.LOADING_MODEL)
def client() -> FallbackClient:
    c = FallbackClient()
    c.call("health")                 # loads the bundle now, under the spinner
    return c


@st.cache_data(show_spinner=False)
def meta() -> dict[str, Any]:
    return client().call("metadata")


@st.cache_data(show_spinner=False)
def health() -> dict[str, Any]:
    return client().call("health")


def model_ready() -> bool:
    """Load the model (with a spinner); show the plain error and return False if impossible."""
    try:
        client()
        return True
    except AppUnavailable:
        st.error(C.ERROR_MODEL)
        return False


# ---------------------------------------------------------------- page furniture
def notice() -> None:
    st.markdown(f'<p class="eq-notice">{C.NOTICE}</p>', unsafe_allow_html=True)


def footer() -> None:
    st.divider()
    st.caption(ATTRIBUTION)


def apply_theme() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)


SUGGEST_PAGE = "views/suggest.py"


def go_to_wizard(step: int = 1) -> None:
    st.session_state["step"] = step
    st.switch_page(SUGGEST_PAGE)


def chart_ink() -> str:
    """Text colour for chart labels in the viewer's current light/dark theme."""
    try:
        return theme.ink(st.context.theme.type)
    except Exception:   # noqa: BLE001 - older Streamlit / bare mode
        return theme.ink(None)


def canonical(obj: Any) -> str:
    """Stable JSON key for caching (floats rounded so widget round-trips still match)."""
    def norm(v: Any) -> Any:
        if isinstance(v, float):
            return int(v) if v.is_integer() else round(v, 6)
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items()) if k != DERIVED_FEATURE}
        return v
    return json.dumps(norm(obj), sort_keys=True)


# ---------------------------------------------------------------- patient state
DERIVED_FEATURE = "X_va_below_69"     # derived from vision by the model; never a form field
PATIENT, STEP, EXAMPLE, VA_EXACT = "patient", "step", "example", "va_exact"
TIP_DISMISSED, SESSION_PRICES, VALUE_PER_LINE = "tip_dismissed", "session_prices", "value_per_line"
WIDGET_PREFIX = "w_"


def feature_specs() -> dict[str, dict[str, Any]]:
    return {f["name"]: f for f in meta()["features"] if f["name"] != DERIVED_FEATURE}


def clean_features(values: dict[str, Any]) -> dict[str, Any]:
    """Model-ready values for every form feature (binary -> 0/1, int fields -> int)."""
    out: dict[str, Any] = {}
    for name, spec in feature_specs().items():
        v = values.get(name)
        if v is None or (isinstance(v, float) and v != v):
            out[name] = None
        elif spec["binary"]:
            out[name] = int(round(float(v)))
        elif spec["categories"]:
            out[name] = str(v)
        elif spec["dtype"] == "int64":
            out[name] = int(round(float(v)))
        else:
            out[name] = float(v)
    return out


def default_patient() -> dict[str, Any]:
    """Typical training values, with vision moved to the nearest eye-chart line."""
    from app import presenters as P

    p = clean_features(meta()["defaults"])
    p["X_va_letters"] = P.snellen_to_letters(P.nearest_snellen(p["X_va_letters"], UI))
    return p


def patient() -> dict[str, Any]:
    if PATIENT not in st.session_state:
        st.session_state[PATIENT] = default_patient()
    return st.session_state[PATIENT]


def clear_widgets() -> None:
    for k in [k for k in st.session_state if str(k).startswith(WIDGET_PREFIX)]:
        del st.session_state[k]


def load_example(example_id: str) -> None:
    """Callback: put an example patient in the form (exact vision kept as recorded)."""
    from app import presenters as P

    feats = clean_features(examples()[example_id])
    st.session_state[PATIENT] = feats
    st.session_state[EXAMPLE] = example_id
    st.session_state[VA_EXACT] = not P.is_snellen_value(feats["X_va_letters"], UI)
    clear_widgets()


def start_over() -> None:
    """Callback: forget the patient and every result; keep prices and the dismissed tip."""
    for k in (PATIENT, EXAMPLE, VA_EXACT, "errors", "compare_with"):
        st.session_state.pop(k, None)
    st.session_state[STEP] = 1
    clear_widgets()


def set_step(step: int) -> None:
    st.session_state[STEP] = step


# ---------------------------------------------------------------- prices
def team_prices_set() -> bool:
    return not health()["prices_are_demo"]


def prices_set() -> bool:
    return bool(st.session_state.get(SESSION_PRICES)) or team_prices_set()


def prices_payload() -> dict[str, Any] | None:
    """Prices for the request: session prices, else the team's prices from config (with the
    user's value of a line when given). None = config as is (DEMO when prices are null; the
    presenters then hide everything derived from them)."""
    import warnings

    from app import presenters as P
    from equipoise.config import load_prices

    vpl = st.session_state.get(VALUE_PER_LINE)
    wtp = (P.wtp_from_value_per_line(vpl, float(CFG.decision["value_of_vision_k"]),
                                     float(UI["letters_per_line"])) if vpl else None)
    sp = st.session_state.get(SESSION_PRICES)
    if sp:
        return {"price_per_injection": dict(sp["per_injection"]),
                "laser_session": float(sp["laser"]), "willingness_to_pay": wtp}
    if team_prices_set() and wtp is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = load_prices()
        return {"price_per_injection": dict(p.per_injection),
                "laser_session": p.laser_session, "willingness_to_pay": wtp}
    return None


# ---------------------------------------------------------------- results (cached)
@st.cache_resource(show_spinner=False)
def _results() -> dict[tuple[str, bool], dict[str, Any]]:
    """(patient + prices key, with explanation?) -> prediction; shared by all sessions."""
    return {}


def _result_key(features: dict[str, Any], prices: dict[str, Any] | None) -> str:
    return canonical({"patient": features, "prices": prices})


def cached_prediction(features: dict[str, Any], explain: bool = False
                      ) -> dict[str, Any] | None:
    store, k = _results(), _result_key(features, prices_payload())
    return store.get((k, True)) or (None if explain else store.get((k, False)))


def predict(features: dict[str, Any], explain: bool = False) -> dict[str, Any]:
    """The prediction for this patient at the current prices (instant when cached)."""
    hit = cached_prediction(features, explain)
    if hit is not None:
        return hit
    prices = prices_payload()
    res = client().call("predict", {"patient": features, "prices": prices, "explain": explain})
    _results()[(_result_key(features, prices), explain)] = res
    return res


def breakeven(features: dict[str, Any], drug: str) -> dict[str, Any]:
    """Break-even curve for a costlier drug at the current prices (cached; the slider then
    reads the returned curve, so moving it never calls the model)."""
    prices = prices_payload()
    key = ("breakeven:" + drug + ":" + _result_key(features, prices), False)
    store = _results()
    if key not in store:
        store[key] = client().call("breakeven", {"patient": features, "prices": prices,
                                                 "drug": drug})
    return store[key]


def early_response(payload: dict[str, Any]) -> dict[str, Any]:
    key = ("early:" + canonical(payload), False)
    store = _results()
    if key not in store:
        store[key] = client().call("early_response", payload)
    return store[key]


# ---------------------------------------------------------------- example patients
@st.cache_resource(show_spinner=False)
def examples() -> dict[str, dict[str, Any]]:
    """id -> features of the three example patients.

    Uses `data/derived/app_examples.json` (written by scripts/pick_example_patients.py) when
    it was made with the bundle being served, and seeds the result cache with its
    precomputed predictions so the examples show instantly. Otherwise picks held-out
    patients by their features (synthetic data inside Docker / CI)."""
    import warnings

    from equipoise.config import for_data_source, load_prices
    from equipoise.decision.recommend import FEATURES
    from equipoise.serving import examples as ex
    from equipoise.serving.service import held_out_patients

    h = health()
    synthetic = bool(h["synthetic_bundle"])
    data = ex.read_examples(ex.examples_path(for_data_source(CFG, synthetic)))
    if data and data.get("bundle_hash") == h["training_data_hash"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            same_prices = data.get("prices_signature") == ex.prices_signature(load_prices())
        out = {}
        for k in ex.EXAMPLE_IDS:
            e = data["examples"].get(k)
            if not e:
                continue
            out[k] = clean_features(e["features"])
            if same_prices and e.get("prediction"):
                _results()[(_result_key(out[k], None), True)] = e["prediction"]
        if out:
            return out
    tp = held_out_patients(CFG, synthetic)
    if tp.empty and not synthetic:
        tp = held_out_patients(CFG, True)
    if tp.empty:
        return {}
    picks = ex.pick_by_features(tp)
    rows = tp.reset_index(drop=True)
    return {k: clean_features(ex.feature_dict(rows.iloc[picks[k]], FEATURES))
            for k in ex.EXAMPLE_IDS}


# ---------------------------------------------------------------- trial results (About page)
@st.cache_data(show_spinner=False)
def replication_1yr() -> dict[str, str]:
    """1-year rows of our replication table ('this run' column). Real results only: the
    About page never shows synthetic numbers as the trial's."""
    import csv

    path = CFG.paths.tables_dir / "m2_replication_vs_published.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["result"]: r["this run"] for r in csv.DictReader(fh)
                if r["horizon_years"] == "1"}
