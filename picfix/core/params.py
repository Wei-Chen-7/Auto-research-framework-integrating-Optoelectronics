"""Design parameters for the directional coupler."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DesignParams(BaseModel):
    """Geometry of a directional coupler candidate design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_nm: float
    length_um: float
    width_nm: float

    def perturbed(self, dgap_nm: float = 0.0, dwidth_nm: float = 0.0) -> "DesignParams":
        return DesignParams(
            gap_nm=self.gap_nm + dgap_nm,
            length_um=self.length_um,
            width_nm=self.width_nm + dwidth_nm,
        )
