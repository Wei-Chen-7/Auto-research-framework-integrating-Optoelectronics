"""Analytical directional-coupler model (milestone 1 simulation stand-in).

Physics:
    kappa(gap, width, lambda) = kappa0
        * exp(-(gap - gap_ref) / gap_decay)
        * (1 + width_sens * (width_ref - width))
        * (1 + lambda_sens * (lambda - lambda0))
    cross ratio           = sin^2(kappa * L)
    insertion loss  IL_dB = a * L + b * exp(-gap / g0)

The loss model makes both escape routes costly: growing L to fix the split
ratio pays the a*L term, shrinking gap to strengthen coupling pays the
scattering term — which also runs into the DRC minimum gap.
"""
from __future__ import annotations

import math
import time

import numpy as np

from picfix.core.config import DRCRulesConfig
from picfix.core.constants import AnalyticalConstants
from picfix.core.params import DesignParams
from picfix.core.spec import TaskSpec
from picfix.core.spectrum import SimResult, Spectrum
from picfix.core.truth import GroundTruth, RootCause
from picfix.layout import drc
from picfix.simulators.base import SimulatorBackend


class AnalyticalBackend(SimulatorBackend):
    name = "analytical"

    def __init__(
        self,
        constants: AnalyticalConstants,
        spec: TaskSpec,
        drc_rules: DRCRulesConfig,
        sigma_ratio: float,
        sigma_il_db: float,
    ) -> None:
        self._c = constants
        self._spec = spec
        self._drc = drc_rules
        self._sigma_ratio = sigma_ratio
        self._sigma_il = sigma_il_db

    # -- physics ---------------------------------------------------------

    def kappa_per_um(self, gap_nm: float, width_nm: float, wavelength_nm: float) -> float:
        c = self._c
        k = c.kappa0_per_um * math.exp(-(gap_nm - c.gap_ref_nm) / c.gap_decay_nm)
        k *= 1.0 + c.width_sens_per_nm * (c.width_ref_nm - width_nm)
        k *= 1.0 + c.lambda_sens_per_nm * (wavelength_nm - c.lambda0_nm)
        return max(k, 0.0)

    def cross_ratio(self, params: DesignParams, wavelength_nm: float) -> float:
        k = self.kappa_per_um(params.gap_nm, params.width_nm, wavelength_nm)
        return math.sin(k * params.length_um) ** 2

    def scattering_loss_db(self, gap_nm: float) -> float:
        return self._c.il_b_db * math.exp(-gap_nm / self._c.il_g0_nm)

    def length_loss_db(self, length_um: float) -> float:
        return self._c.il_a_db_per_um * length_um

    def insertion_loss_db(self, params: DesignParams) -> float:
        return self.length_loss_db(params.length_um) + self.scattering_loss_db(params.gap_nm)

    # -- SimulatorBackend --------------------------------------------------

    def simulate(
        self,
        params: DesignParams,
        wavelengths_nm: tuple[float, ...],
        *,
        noisy: bool,
        seed: int | None,
    ) -> SimResult:
        t0 = time.perf_counter()
        ratios = [self.cross_ratio(params, wl) for wl in wavelengths_nm]
        il = self.insertion_loss_db(params)
        ils = [il for _ in wavelengths_nm]
        if noisy:
            rng = np.random.default_rng(seed)
            ratios = [
                float(np.clip(r + rng.normal(0.0, self._sigma_ratio), 0.0, 1.0)) for r in ratios
            ]
            ils = [max(float(v + rng.normal(0.0, self._sigma_il)), 0.0) for v in ils]
        spectrum = Spectrum(
            wavelengths_nm=wavelengths_nm,
            cross_ratio=tuple(ratios),
            insertion_loss_db=tuple(ils),
        )
        return SimResult(
            params=params,
            spectrum=spectrum,
            backend=self.name,
            noisy=noisy,
            seed=seed if noisy else None,
            elapsed_s=time.perf_counter() - t0,
        )

    def ground_truth(self, params: DesignParams) -> GroundTruth:
        """Root-cause labels mirroring the frozen judge's criteria, noise-free."""
        spec = self._spec
        causes: set[RootCause] = set()

        if params.gap_nm <= 0 or params.length_um <= 0 or params.width_nm <= 0:
            return GroundTruth(causes=frozenset({RootCause.PARAM_OUT_OF_BOUNDS}))

        if drc.check(params, self._drc):
            causes.add(RootCause.DRC_VIOLATION)

        grid = spec.hidden_grid()
        center = self._c.lambda0_nm
        d = spec.corner_offset_nm
        corners = [(0.0, 0.0)] + [(sg * d, sw * d) for sg in (-1, 1) for sw in (-1, 1)]

        center_dev = abs(self.cross_ratio(params, center) - spec.split_target)
        nominal_edge_dev = max(
            abs(self.cross_ratio(params, wl) - spec.split_target) for wl in grid
        )
        worst_dev = max(
            abs(self.cross_ratio(params.perturbed(dg, dw), wl) - spec.split_target)
            for dg, dw in corners
            for wl in grid
        )
        if worst_dev > spec.split_tolerance:
            if center_dev > spec.split_tolerance:
                causes.add(RootCause.COUPLING_LENGTH_MISMATCH)
            elif nominal_edge_dev > spec.split_tolerance:
                causes.add(RootCause.WAVELENGTH_DRIFT)
            else:
                # only process corners fail: insufficient coupling margin
                causes.add(RootCause.COUPLING_LENGTH_MISMATCH)

        worst_il = max(self.insertion_loss_db(params.perturbed(dg, dw)) for dg, dw in corners)
        if worst_il > spec.il_max_db:
            scatter = self.scattering_loss_db(params.gap_nm - d)
            length_term = self.length_loss_db(params.length_um)
            if scatter >= spec.il_max_db / 2 or scatter >= length_term:
                causes.add(RootCause.GAP_TOO_SMALL_SCATTERING)
            if length_term >= spec.il_max_db / 2 and length_term > scatter:
                causes.add(RootCause.INSERTION_LOSS_EXCESS)

        return GroundTruth(causes=frozenset(causes))
