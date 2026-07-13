"""Pure-Python DRC rules (milestone 1).

Interface kept minimal and data-driven so a klayout-backed checker can be
swapped in later behind the same ``check`` signature.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from picfix.core.config import DRCRulesConfig
from picfix.core.params import DesignParams


class DRCViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    message: str
    value: float
    limit: float


def check(params: DesignParams, rules: DRCRulesConfig) -> list[DRCViolation]:
    violations: list[DRCViolation] = []
    if params.gap_nm < rules.min_gap_nm:
        violations.append(
            DRCViolation(
                rule="min_gap",
                message=f"gap {params.gap_nm:.1f} nm below minimum {rules.min_gap_nm:.1f} nm",
                value=params.gap_nm,
                limit=rules.min_gap_nm,
            )
        )
    if params.width_nm < rules.min_width_nm:
        violations.append(
            DRCViolation(
                rule="min_width",
                message=f"width {params.width_nm:.1f} nm below minimum {rules.min_width_nm:.1f} nm",
                value=params.width_nm,
                limit=rules.min_width_nm,
            )
        )
    if params.width_nm > rules.max_width_nm:
        violations.append(
            DRCViolation(
                rule="max_width",
                message=f"width {params.width_nm:.1f} nm above maximum {rules.max_width_nm:.1f} nm",
                value=params.width_nm,
                limit=rules.max_width_nm,
            )
        )
    if params.length_um < rules.min_length_um:
        violations.append(
            DRCViolation(
                rule="min_length",
                message=f"length {params.length_um:.2f} um below minimum {rules.min_length_um:.2f} um",
                value=params.length_um,
                limit=rules.min_length_um,
            )
        )
    if params.length_um > rules.max_length_um:
        violations.append(
            DRCViolation(
                rule="max_length",
                message=f"length {params.length_um:.2f} um above maximum {rules.max_length_um:.2f} um",
                value=params.length_um,
                limit=rules.max_length_um,
            )
        )
    return violations
