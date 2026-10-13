"""Serving layer shared by the FastAPI service and the Streamlit app.

`Service` loads the model bundle once (plus M7's early-response model and M9's serving
assets when they exist) and answers the three questions the app asks: the full
recommendation for a patient (`recommend()` plus a per-patient SHAP explanation), the
break-even price of a dearer drug with its net-benefit curve, and the early-response
forecast. Every request's baseline features (never identifiers) are appended to
`logs/requests.jsonl`, which the drift monitor compares with the training features.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equipoise.causal.bundle import ModelBundle, load_bundle
from equipoise.config import Config, Prices, for_data_source, load_config, load_prices
from equipoise.decision.breakeven import value_of_vision
from equipoise.decision.recommend import (
    DISCLAIMER,
    FEATURES,
    default_bundle_path,
    patient_frame,
    recommend,
    serving_mode,
)
from equipoise.features import EARLY_PREFIX
from equipoise.serving.assets import load_assets
from equipoise.serving.schemas import (
    FEATURE_SPECS,
    BreakEvenRequest,
    BreakEvenResponse,
    Contribution,
    CurvePoint,
    EarlyResponse,
    EarlyResponseRequest,
    Explanation,
    Health,
    Prediction,
    PredictRequest,
    PricesIn,
)

ATTRIBUTION = ("Data: DRCR Retina Network, Protocol T (FAIN UG1EY014231), "
               "https://public.jaeb.org/drcrnet. The analyses, content and conclusions are "
               "solely the responsibility of the authors and have not been reviewed or approved "
               "by the DRCR Retina Network.")
EARLY_REQUEST_WEEKS = (4, 8, 12)   # fields va_wk4 / va_wk8 / va_wk12 of EarlyResponseRequest


_LOG_LOCK = threading.Lock()


def append_request_log(path: Path, endpoint: str, features: dict[str, Any]) -> None:
    """One JSON line: timestamp, endpoint and the baseline features only (no identifiers,
    no outcomes, no client address)."""
    row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "endpoint": endpoint,
           "features": {k: features.get(k) for k in FEATURES}}
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


class ServiceUnavailable(RuntimeError):
    """A requested capability needs an artifact that has not been built."""


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_bundle_path(cfg: Config) -> Path:
    """Bundle file: `$EQUIPOISE_BUNDLE`, else models/synthetic/ when `$EQUIPOISE_SYNTHETIC`
    is set (public demo), else the real bundle in models/."""
    s = cfg.raw["serving"]
    explicit = os.environ.get(s["bundle_env_var"])
    if explicit:
        return Path(explicit).resolve()
    return default_bundle_path(cfg, synthetic=_truthy(os.environ.get(s["synthetic_env_var"])))


def resolve_log_path(cfg: Config) -> Path | None:
    """Request log: `$EQUIPOISE_REQUEST_LOG` ("off" disables), else logs/requests.jsonl."""
    s = cfg.raw["serving"]
    env = os.environ.get(s["request_log_env_var"])
    if env is not None:
        return None if env.strip().lower() in {"", "off", "none"} else Path(env).resolve()
    return cfg.paths.logs_dir / s["request_log"]


@dataclass
class Service:
    cfg: Config
    bundle: ModelBundle
    assets: dict[str, Any] | None = None
    log_path: Path | None = None
    early_typical_error: float | None = None

    # construction ---------------------------------------------------------------------
    @classmethod
    def from_path(cls, bundle_path: Path, cfg: Config | None = None,
                  log_path: Path | None = None) -> Service:
        cfg = cfg or load_config()
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"No model bundle at {bundle_path}. Build it with `python -m equipoise causal` "
                "(add --synthetic for the demo bundle) or set EQUIPOISE_BUNDLE.")
        bundle = serving_mode(load_bundle(bundle_path))
        weeks = tuple(int(w) for w in cfg.longitudinal["early_weeks"])
        if weeks != EARLY_REQUEST_WEEKS:
            raise ValueError(f"early-response API expects weeks {EARLY_REQUEST_WEEKS}, config "
                             f"has {weeks}")
        metrics = for_data_source(cfg, bundle.synthetic).paths.metrics_dir
        early_err = None
        m7 = metrics / "m7_longitudinal.json"
        if m7.is_file():
            early_err = json.loads(m7.read_text(encoding="utf-8")).get("early_cv_rmse_with_early")
        return cls(cfg, bundle, load_assets(bundle_path.parent), log_path, early_err)

    @classmethod
    def from_env(cls, cfg: Config | None = None) -> Service:
        cfg = cfg or load_config()
        return cls.from_path(resolve_bundle_path(cfg), cfg, resolve_log_path(cfg))

    # helpers --------------------------------------------------------------------------
    def prices(self, given: PricesIn | None) -> Prices:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # demo use is flagged in the output
            return load_prices(given.as_override()) if given is not None else load_prices()

    def log_request(self, endpoint: str, features: dict[str, Any]) -> None:
        """Append baseline features (no identifiers) to the request log."""
        if self.log_path is not None:
            append_request_log(self.log_path, endpoint, features)

    def defaults(self) -> dict[str, Any]:
        """Form defaults: training medians / modes from M9's assets, else only vision set."""
        if self.assets and self.assets.get("defaults"):
            return dict(self.assets["defaults"])
        lo, hi = self.cfg.va_eligibility_letters
        return {f: None for f in FEATURES} | {"X_va_letters": float((lo + hi) // 2)}

    def metadata(self) -> dict[str, Any]:
        """Arms, MCID, form defaults and the feature schema (drives the app's form)."""
        return {"arms": list(self.bundle.arms), "reference_arm": self.bundle.reference_arm,
                "mcid_letters": float(self.bundle.mcid_letters), "defaults": self.defaults(),
                "features": [{"name": f.name, "dtype": f.dtype, "min": f.min, "max": f.max,
                              "categories": list(f.categories) if f.categories else None,
                              "binary": f.binary, "nullable": f.nullable,
                              "description": f.description} for f in FEATURE_SPECS],
                "synthetic_bundle": bool(self.bundle.synthetic),
                "disclaimer": DISCLAIMER, "attribution": ATTRIBUTION}

    def health(self) -> Health:
        p = self.prices(None)
        return Health(status="ok", model=self.bundle.effect_model_name,
                      synthetic_bundle=bool(self.bundle.synthetic),
                      training_data_hash=self.bundle.training_data_hash,
                      early_response_available=self.bundle.early_response_model is not None,
                      shap_available=self.assets is not None,
                      prices_label=p.label, prices_are_demo=p.is_demo)

    # explanation ----------------------------------------------------------------------
    def explain(self, features: dict[str, Any]) -> Explanation | None:
        """Permutation SHAP of the aflibercept - bevacizumab effect for one patient, summed
        back to raw features; background = k-means centroids of training patients."""
        if self.assets is None:
            return None
        from equipoise.explain.shap_effects import explain_effect, raw_feature_shap

        names = self.assets["feature_names"]
        X = self.bundle.preprocessor.transform(patient_frame(features))
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
        expl = explain_effect(self.bundle.effect_model, self.assets["shap_background"], X,
                              names, self.cfg)
        raw = raw_feature_shap(expl.values, names, self.cfg).iloc[0]
        order = raw.abs().sort_values(ascending=False).index
        top = int(self.cfg.raw["serving"]["shap_top_features"])
        contribs = [Contribution(feature=f, value=_fmt(features.get(f)),
                                 shap_letters=float(raw[f])) for f in order[:top]]
        base = float(np.ravel(expl.base_values)[0])
        arms = self.bundle.arms
        return Explanation(
            comparison=f"{arms[1]} - {arms[0]}", base_value_letters=base,
            prediction_letters=base + float(raw.sum()), contributions=contribs,
            other_features_letters=float(raw[order[top:]].sum()),
            note="Contributions (letters) add up to this patient's predicted effect, starting "
                 "from the average over training-patient cluster centres. They explain the "
                 "model, not the biology.")

    # endpoints ------------------------------------------------------------------------
    def predict(self, req: PredictRequest) -> Prediction:
        feats = req.patient.features()
        rec = recommend(feats, self.prices(req.prices), self.bundle, self.cfg)
        imputed = [k for k in FEATURES if feats.get(k) is None and k != "X_va_below_69"]
        expl = self.explain(feats) if req.explain else None
        self.log_request("predict", feats)
        return Prediction(**rec.model_dump(), imputed_features=imputed, explanation=expl)

    def breakeven(self, req: BreakEvenRequest) -> BreakEvenResponse:
        feats = req.patient.features()
        prices = self.prices(req.prices)
        rec = recommend(feats, prices, self.bundle, self.cfg)
        est = {e.drug: e for e in rec.estimates}
        ref, drug = est[self.bundle.arms[0]], est[req.drug]
        result = next((b for b in rec.break_even if b.drug == req.drug), None)
        price_now = prices.per_injection[req.drug]
        curve: list[CurvePoint] = []
        wtp = prices.willingness_to_pay
        if wtp is not None:
            s, k = self.cfg.raw["serving"], float(self.cfg.decision["value_of_vision_k"])
            gain = wtp * (value_of_vision(drug.letters, k) - value_of_vision(ref.letters, k))
            be = result.break_even_price if result and result.break_even_price else 0.0
            top = float(s["breakeven_curve_max_multiple"]) * max(price_now, be, 1e-9)
            for p in np.linspace(0.0, top, int(s["breakeven_curve_points"])):
                nbd = gain - (drug.injections_yr1 * p + drug.p_laser_yr1 * prices.laser_session
                              - ref.cost)
                curve.append(CurvePoint(price=float(p), net_benefit_difference=float(nbd)))
        self.log_request("breakeven", feats)
        return BreakEvenResponse(drug=req.drug, reference=ref.drug, current_price=price_now,
                                 result=result, curve=curve, prices_label=prices.label,
                                 prices_are_demo=prices.is_demo, warnings=rec.warnings)

    def early_response(self, req: EarlyResponseRequest) -> EarlyResponse:
        model = self.bundle.early_response_model
        if model is None:
            raise ServiceUnavailable("early-response model not built: run "
                                     "`python -m equipoise longitudinal`")
        feats = req.patient.features()
        X = patient_frame(feats).assign(treatment=req.treatment)
        base = float(X.X_va_letters.iloc[0])
        given = dict(zip(EARLY_REQUEST_WEEKS, (req.va_wk4, req.va_wk8, req.va_wk12),
                         strict=True))
        points = [(0.0, base)]
        for w, va in given.items():
            X[f"{EARLY_PREFIX}va_change_wk{w}"] = np.nan if va is None else va - base
            if va is not None:
                points.append((float(w), float(va)))
        wk, va = np.array(points).T
        X[f"{EARLY_PREFIX}early_slope"] = float(np.polyfit(wk, va, 1)[0])
        pred = float(model.predict(X)[0])
        arm = self.bundle.arms.index(req.treatment)
        baseline_pred = float(self.bundle.predict(patient_frame(feats))["letters"][0, arm])
        traj = [{"week": float(w), "va_change": float(v - base)} for w, v in points]
        weeks_final = float(self.cfg.primary_outcome.rsplit("wk", 1)[1])
        traj.append({"week": weeks_final, "va_change": pred, "forecast": 1.0})
        self.log_request("early-response", feats)
        return EarlyResponse(
            treatment=req.treatment, baseline_va_letters=base, predicted_va_change_wk52=pred,
            predicted_va_wk52=base + pred, typical_error_letters=self.early_typical_error,
            baseline_model_va_change_wk52=baseline_pred, trajectory=traj,
            note="Uses vision measured after treatment started (weeks 4-12): it tracks how "
                 "this patient is responding; it is not used to choose the drug.",
            disclaimer=DISCLAIMER)


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "imputed"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.4g}"
    return str(v)


def held_out_patients(cfg: Config, synthetic: bool) -> pd.DataFrame:
    """Held-out test patients (baseline features plus the arm they actually received and
    their observed week-52 change, for context only). Empty when the data is not present."""
    from equipoise.data.load import analysis_path, load_analysis
    from equipoise.data.splits import get_split, splits_path

    if not analysis_path(cfg, synthetic).is_file() or not splits_path(cfg, synthetic).is_file():
        return pd.DataFrame()
    df = load_analysis(cfg, synthetic=synthetic)
    test = get_split(cfg, df, synthetic).test(df).reset_index(drop=True)
    return test[[*FEATURES, "treatment", cfg.primary_outcome]]
