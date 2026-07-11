"""Root-cause ground-truth labels.

Emitted by the analytical backend alongside every evaluation, but routed
only to the metrics layer — agents never see these labels in any arm.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RootCause(StrEnum):
    PARAM_OUT_OF_BOUNDS = "param_out_of_bounds"
    COUPLING_LENGTH_MISMATCH = "coupling_length_mismatch"
    GAP_TOO_SMALL_SCATTERING = "gap_too_small_scattering"
    WAVELENGTH_DRIFT = "wavelength_drift"
    INSERTION_LOSS_EXCESS = "insertion_loss_excess"
    DRC_VIOLATION = "drc_violation"


class GroundTruth(BaseModel):
    """Noise-free root-cause labels for one evaluated design.

    Empty ``causes`` means the design truly meets the frozen spec.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    causes: frozenset[RootCause]

    @property
    def is_failure(self) -> bool:
        return bool(self.causes)
