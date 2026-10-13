"""Baseline feature groups, derived from the schema and config.

Every X_ column is put into exactly one group, which decides how it is encoded:
nominal categories are one-hot encoded, ordered categories ordinal-encoded, 0/1 flags passed
through, skewed labs log-transformed, CST harmonized across OCT machines, and all other
continuous features standardized. A test checks the groups cover every X_ column.
"""

from __future__ import annotations

from dataclasses import dataclass

from equipoise.config import Config
from equipoise.data.schema import ANALYSIS_COLUMNS, DR_SEVERITY_CLINICAL

NOMINAL = ("X_race", "X_diabetes_type", "X_oct_machine")
ORDINAL = {"X_dr_severity_clinical": DR_SEVERITY_CLINICAL}
CST = "X_cst_um"
OCT_MACHINE = "X_oct_machine"


@dataclass(frozen=True)
class FeatureGroups:
    """Column names per encoding group (all X_)."""

    nominal: tuple[str, ...]
    ordinal: tuple[str, ...]
    binary: tuple[str, ...]
    log: tuple[str, ...]
    continuous: tuple[str, ...]
    cst: str

    @property
    def all(self) -> tuple[str, ...]:
        return (*self.nominal, *self.ordinal, *self.binary, *self.log, *self.continuous,
                self.cst)


def feature_groups(cfg: Config) -> FeatureGroups:
    """Assign every X_ column of the schema to one encoding group."""
    features = [c for c in ANALYSIS_COLUMNS if c.role == "feature"]
    log = tuple(cfg.preprocess["log_features"])
    binary = tuple(c.name for c in features if c.binary)
    taken = {*NOMINAL, *ORDINAL, *binary, *log, CST}
    continuous = tuple(c.name for c in features if c.name not in taken)
    return FeatureGroups(NOMINAL, tuple(ORDINAL), binary, log, continuous, CST)
