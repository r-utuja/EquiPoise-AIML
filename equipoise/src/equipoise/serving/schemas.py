"""Request and response schemas for the API, generated from the feature schema.

`PatientFeatures` is built field by field from `data.schema.ANALYSIS_COLUMNS`: every `X_`
baseline feature becomes an optional field with the same type, range and category levels the
analysis table is validated against (only baseline vision is required). Unknown fields are
rejected, so identifiers, outcomes (`Y_`), bookkeeping (`M_`) or the treatment arm can never
reach the model through the API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from equipoise.data.schema import ANALYSIS_COLUMNS, ARMS, ColumnSpec
from equipoise.decision.recommend import BreakEvenOut, Recommendation

REQUIRED_FEATURES = ("X_va_letters",)
FEATURE_SPECS: tuple[ColumnSpec, ...] = tuple(c for c in ANALYSIS_COLUMNS if c.role == "feature")
COMPARISON_ARMS = tuple(a for a in ARMS if a != "Bevacizumab")


def _field(spec: ColumnSpec) -> tuple[Any, Any]:
    """(annotation, FieldInfo) for one feature column."""
    if spec.categories:
        typ: Any = Literal[spec.categories]  # type: ignore[valid-type]
    elif spec.binary:
        typ = Literal[0, 1]
    elif spec.integer and spec.dtype == "int64":
        typ = int
    else:
        typ = float
    kwargs: dict[str, Any] = {"description": spec.description}
    if not spec.categories and not spec.binary:
        if spec.min is not None:
            kwargs["ge"] = spec.min
        if spec.max is not None:
            kwargs["le"] = spec.max
    if spec.name in REQUIRED_FEATURES:
        return typ, Field(..., **kwargs)
    return typ | None, Field(None, **kwargs)


class _PatientBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _consistent_subgroup(self) -> _PatientBase:
        va, below = self.X_va_letters, self.X_va_below_69  # type: ignore[attr-defined]
        if below is not None and int(below) != int(va < 69):
            raise ValueError("X_va_below_69 contradicts X_va_letters (< 69 letters)")
        return self

    def features(self) -> dict[str, Any]:
        """Feature dict for `recommend` (missing features stay None and are imputed)."""
        return self.model_dump()

    def provided(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v is not None]


PatientFeatures: type[_PatientBase] = create_model(  # type: ignore[call-overload]
    "PatientFeatures", __base__=_PatientBase,
    **{s.name: _field(s) for s in FEATURE_SPECS},
)
PatientFeatures.__doc__ = ("Baseline (pre-randomization) features of one patient. Only "
                           "X_va_letters is required; missing features are imputed.")


class PricesIn(BaseModel):
    """Prices for this request; overrides config/costs.yaml and the DEMO set."""

    model_config = ConfigDict(extra="forbid")
    price_per_injection: dict[str, float] = Field(
        ..., description="price per injection for every drug")
    laser_session: float = Field(..., ge=0)
    willingness_to_pay: float | None = Field(
        None, gt=0, description="value of one unit of U(letters); enables break-even outputs")

    @model_validator(mode="after")
    def _all_arms(self) -> PricesIn:
        if set(self.price_per_injection) != set(ARMS):
            raise ValueError(f"price_per_injection needs exactly these drugs: {list(ARMS)}")
        if any(v < 0 for v in self.price_per_injection.values()):
            raise ValueError("prices must be >= 0")
        return self

    def as_override(self) -> dict[str, Any]:
        return {"price_per_injection_inr": self.price_per_injection,
                "laser_session_inr": self.laser_session,
                "willingness_to_pay": self.willingness_to_pay}


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient: PatientFeatures  # type: ignore[valid-type]
    prices: PricesIn | None = None
    explain: bool = Field(True, description="add the per-patient SHAP contributions")


class Contribution(BaseModel):
    feature: str
    value: str
    shap_letters: float


class Explanation(BaseModel):
    comparison: str
    base_value_letters: float
    prediction_letters: float
    contributions: list[Contribution]
    other_features_letters: float
    note: str


class Prediction(Recommendation):
    """`recommend()` output plus the SHAP explanation and which features were imputed."""

    imputed_features: list[str] = []
    explanation: Explanation | None = None


class BreakEvenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient: PatientFeatures  # type: ignore[valid-type]
    prices: PricesIn | None = None
    drug: Literal[COMPARISON_ARMS] = "Aflibercept"  # type: ignore[valid-type]


class CurvePoint(BaseModel):
    price: float
    net_benefit_difference: float


class BreakEvenResponse(BaseModel):
    drug: str
    reference: str
    current_price: float
    result: BreakEvenOut | None
    curve: list[CurvePoint]
    prices_label: str
    prices_are_demo: bool
    warnings: list[str]


class EarlyResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient: PatientFeatures  # type: ignore[valid-type]
    treatment: Literal[ARMS]  # type: ignore[valid-type]
    va_wk4: float | None = Field(None, ge=0, le=100, description="study-eye letters at week 4")
    va_wk8: float | None = Field(None, ge=0, le=100)
    va_wk12: float | None = Field(None, ge=0, le=100)

    @model_validator(mode="after")
    def _some_follow_up(self) -> EarlyResponseRequest:
        if all(v is None for v in (self.va_wk4, self.va_wk8, self.va_wk12)):
            raise ValueError("give vision at week 4, 8 or 12 (at least one)")
        return self


class EarlyResponse(BaseModel):
    treatment: str
    baseline_va_letters: float
    predicted_va_change_wk52: float
    predicted_va_wk52: float
    typical_error_letters: float | None
    baseline_model_va_change_wk52: float | None
    trajectory: list[dict[str, float]]
    note: str
    disclaimer: str


class Health(BaseModel):
    status: str
    model: str
    synthetic_bundle: bool
    training_data_hash: str
    early_response_available: bool
    shap_available: bool
    prices_label: str
    prices_are_demo: bool
