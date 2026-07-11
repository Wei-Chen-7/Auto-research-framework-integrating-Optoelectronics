"""Frozen analytical-model constants.

Written once by ``scripts/calibrate.py`` and treated as a meta-immutable
asset afterwards (Gate hash manifest covers the config file section).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AnalyticalConstants(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kappa0_per_um: float      # coupling coefficient at (gap_ref, width_ref, lambda0)
    gap_ref_nm: float
    gap_decay_nm: float       # kappa ~ exp(-(gap - gap_ref)/gap_decay)
    width_ref_nm: float
    width_sens_per_nm: float  # kappa *= 1 + s_w * (width_ref - width)
    lambda0_nm: float
    lambda_sens_per_nm: float  # kappa *= 1 + s_l * (lambda - lambda0)
    il_a_db_per_um: float     # length-proportional loss
    il_b_db: float            # scattering loss amplitude
    il_g0_nm: float           # scattering decay length
