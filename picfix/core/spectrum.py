"""Simulation results: spectra and their agent-facing summaries."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from picfix.core.params import DesignParams


class Spectrum(BaseModel):
    """Cross-port split ratio and insertion loss sampled on a wavelength grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wavelengths_nm: tuple[float, ...]
    cross_ratio: tuple[float, ...]      # power fraction in cross port, 0..1
    insertion_loss_db: tuple[float, ...]

    def worst_ratio_dev(self, target: float) -> float:
        return max(abs(r - target) for r in self.cross_ratio)

    def max_il_db(self) -> float:
        return max(self.insertion_loss_db)


class SimResult(BaseModel):
    """What one simulator call returns to the caller (agent-visible part)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    params: DesignParams
    spectrum: Spectrum
    backend: str
    noisy: bool
    seed: int | None
    elapsed_s: float
