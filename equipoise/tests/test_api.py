"""Phase 8: request schemas, the FastAPI service, request logging, app client and figures.

Runs against the light synthetic bundle from conftest.py, with serving assets and an early-
response model fitted on synthetic training patients, so no real data is needed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.decision.recommend import FEATURES, recommend
from equipoise.serving.schemas import PatientFeatures


@pytest.fixture(scope="module")
def service(light: Config, bundle, tmp_path_factory: pytest.TempPathFactory):
    from equipoise.data.load import load_analysis, load_visits
    from equipoise.decision.run import typical_patient
    from equipoise.longitudinal.early_response import compare, early_features
    from equipoise.serving.assets import build_serving_assets
    from equipoise.serving.service import Service

    df = load_analysis(light, synthetic=True)
    split = get_split(light, df, True)
    early = early_features(load_visits(light, synthetic=True), df, light)
    both = df.merge(early, on="pt_id", how="left")
    b = copy.copy(bundle)
    b.early_response_model = compare(split.train(both), split.test(both), light)[1]
    train = split.train(df)
    assets = build_serving_assets(train, b, light, typical_patient(train))
    log = tmp_path_factory.mktemp("logs") / "requests.jsonl"
    return Service(light, b, assets, log, early_typical_error=8.0)


@pytest.fixture(scope="module")
def api(service):
    from equipoise.serving.api import create_app

    with TestClient(create_app(service)) as c:
        yield c


@pytest.fixture()
def patient(synthetic: pd.DataFrame) -> dict:
    row = synthetic.iloc[3]
    out = {}
    for f in FEATURES:
        v = row[f]
        out[f] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    return out


# ---------------------------------------------------------------- schemas
def test_patient_schema_is_generated_from_feature_schema() -> None:
    assert set(PatientFeatures.model_fields) == set(FEATURES)
    assert PatientFeatures.model_fields["X_va_letters"].is_required()
    assert sum(f.is_required() for f in PatientFeatures.model_fields.values()) == 1


@pytest.mark.parametrize("extra", [{"Y_va_change_wk52": 5.0}, {"M_has_wk52": 1},
                                   {"treatment": "Aflibercept"}, {"pt_id": 12},
                                   {"E_va_change_wk4": 3.0}])
def test_patient_schema_rejects_non_baseline_fields(extra: dict) -> None:
    with pytest.raises(ValidationError):
        PatientFeatures.model_validate({"X_va_letters": 60, **extra})


@pytest.mark.parametrize("bad", [{"X_va_letters": 10}, {"X_va_letters": 60, "X_hba1c": 40},
                                 {"X_va_letters": 60, "X_race": "Martian"},
                                 {"X_va_letters": 60, "X_female": 2},
                                 {"X_va_letters": 60, "X_va_below_69": 0}, {}])
def test_patient_schema_rejects_invalid_values(bad: dict) -> None:
    with pytest.raises(ValidationError):
        PatientFeatures.model_validate(bad)


# ---------------------------------------------------------------- endpoints
def test_health_and_metadata(api) -> None:
    h = api.get("/health").json()
    assert h["status"] == "ok" and h["synthetic_bundle"] and h["early_response_available"]
    assert h["shap_available"] and h["prices_are_demo"]
    m = api.get("/metadata").json()
    assert m["reference_arm"] == "Bevacizumab" and len(m["features"]) == len(FEATURES)
    assert set(m["defaults"]) == set(FEATURES) and "DRCR" in m["attribution"]


def test_predict_matches_recommend_and_explanation_adds_up(api, service, patient) -> None:
    r = api.post("/predict", json={"patient": patient})
    assert r.status_code == 200
    body = r.json()
    direct = recommend(patient, service.prices(None), service.bundle, service.cfg)
    assert body["rule_trace"] == direct.rule_trace
    assert body["recommendation"] == direct.recommendation
    np.testing.assert_allclose([e["letters"] for e in body["estimates"]],
                               [e.letters for e in direct.estimates])
    assert body["prices_are_demo"] and any("DEMO" in w for w in body["warnings"])
    assert "not for clinical use" in body["disclaimer"].lower()
    ex = body["explanation"]
    total = ex["base_value_letters"] + sum(c["shap_letters"] for c in ex["contributions"]) \
        + ex["other_features_letters"]
    assert total == pytest.approx(ex["prediction_letters"], abs=1e-6)
    assert ex["prediction_letters"] == pytest.approx(
        body["estimates"][1]["benefit_vs_reference"], abs=1e-6)


def test_predict_with_missing_features_and_custom_prices(api) -> None:
    prices = {"price_per_injection": {"Aflibercept": 30.0, "Bevacizumab": 2.0,
                                      "Ranibizumab": 15.0},
              "laser_session": 5.0, "willingness_to_pay": 500.0}
    r = api.post("/predict", json={"patient": {"X_va_letters": 55}, "prices": prices,
                                   "explain": False}).json()
    assert not r["prices_are_demo"] and r["prices_label"] == "override"
    assert r["explanation"] is None and "X_hba1c" in r["imputed_features"]
    assert r["estimates"][0]["cost"] == pytest.approx(
        2.0 * r["estimates"][0]["injections_yr1"] + 5.0 * r["estimates"][0]["p_laser_yr1"])
    bad = {**prices, "price_per_injection": {"Aflibercept": 30.0}}
    assert api.post("/predict", json={"patient": {"X_va_letters": 55},
                                      "prices": bad}).status_code == 422


def test_breakeven_curve_crosses_zero_at_break_even_price(api, patient) -> None:
    body = api.post("/breakeven", json={"patient": patient, "drug": "Aflibercept"}).json()
    curve = pd.DataFrame(body["curve"])
    assert len(curve) > 2 and curve.net_benefit_difference.is_monotonic_decreasing
    be = body["result"]["break_even_price"]
    if be is not None:
        at_be = np.interp(be, curve.price, curve.net_benefit_difference)
        assert at_be == pytest.approx(0.0, abs=1e-6)
    assert api.post("/breakeven", json={"patient": patient,
                                        "drug": "Bevacizumab"}).status_code == 422


def test_early_response(api, service, patient) -> None:
    base = patient["X_va_letters"]
    req = {"patient": patient, "treatment": "Aflibercept", "va_wk4": base + 4,
           "va_wk8": base + 6, "va_wk12": None}
    body = api.post("/early-response", json=req).json()
    assert np.isfinite(body["predicted_va_change_wk52"])
    assert body["predicted_va_wk52"] == pytest.approx(base + body["predicted_va_change_wk52"])
    assert [p["week"] for p in body["trajectory"]] == [0, 4, 8, 52]
    assert api.post("/early-response", json={**req, "va_wk4": None,
                                             "va_wk8": None}).status_code == 422
    from equipoise.serving.api import create_app
    no_early = copy.copy(service)
    no_early.bundle = copy.copy(service.bundle)
    no_early.bundle.early_response_model = None
    with TestClient(create_app(no_early)) as c:
        assert c.post("/early-response", json=req).status_code == 503


def test_request_log_holds_baseline_features_only(api, service, patient) -> None:
    log: Path = service.log_path
    before = len(log.read_text(encoding="utf-8").splitlines()) if log.is_file() else 0
    api.post("/predict", json={"patient": patient, "explain": False})
    api.post("/predict", json={"patient": {"X_va_letters": 200}})        # rejected: not logged
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == before + 1
    row = json.loads(lines[-1])
    assert set(row) == {"ts", "endpoint", "features"}
    assert set(row["features"]) == set(FEATURES)
    assert row["features"]["X_va_letters"] == patient["X_va_letters"]


def test_local_client_matches_api(api, service, patient) -> None:
    from equipoise.serving.client import ClientError, LocalClient

    local = LocalClient(service)
    payload = {"patient": patient, "explain": False}
    assert local.predict(payload) == api.post("/predict", json=payload).json()
    with pytest.raises(ClientError):
        local.predict({"patient": {"X_va_letters": 60, "Y_laser_yr1": 1}})


def test_app_figures_render(api, patient) -> None:
    from equipoise.serving import figures

    pred = api.post("/predict", json={"patient": patient}).json()
    be = api.post("/breakeven", json={"patient": patient}).json()
    er = api.post("/early-response", json={"patient": patient, "treatment": "Bevacizumab",
                                           "va_wk12": patient["X_va_letters"] + 5}).json()
    for fig in (figures.point_range(pred), figures.strength_gauge(pred["fuzzy_strength"]),
                figures.waterfall(pred["explanation"]), figures.breakeven_curve(be),
                figures.early_trajectory(er)):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_streamlit_app_runs(light: Config, bundle, monkeypatch, service) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from equipoise.decision.recommend import default_bundle_path

    monkeypatch.setenv("EQUIPOISE_BUNDLE", str(default_bundle_path(light, synthetic=True)))
    monkeypatch.setenv("EQUIPOISE_REQUEST_LOG", "off")
    monkeypatch.delenv("API_URL", raising=False)
    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    assert any("not for clinical use" in m.value.lower() for m in at.markdown)
    at.switch_page("views/suggest.py").run()                # wizard step 1 (typical patient)
    next(b for b in at.button if b.label == "Compare the medicines").click().run()
    assert not at.exception and at.session_state["step"] == 2   # a prediction was made
    assert any('class="eq-headline"' in m.value for m in at.markdown)
