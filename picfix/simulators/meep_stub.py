"""Meep FDTD backend stub (milestone 2).

TODO(milestone-2): invoke Meep across a process boundary (CLI/subprocess)
to keep the GPL boundary outside the Python import graph, and calibrate it
against the analytical backend before use (DESIGN.md §3.2, §13).
"""
from __future__ import annotations

from picfix.core.params import DesignParams
from picfix.core.spectrum import SimResult
from picfix.core.truth import GroundTruth
from picfix.simulators.base import SimulatorBackend


class MeepBackend(SimulatorBackend):
    name = "meep"

    def simulate(
        self,
        params: DesignParams,
        wavelengths_nm: tuple[float, ...],
        *,
        noisy: bool,
        seed: int | None,
    ) -> SimResult:
        raise NotImplementedError("MeepBackend is a milestone-2 stub; use AnalyticalBackend")

    def ground_truth(self, params: DesignParams) -> GroundTruth:
        raise NotImplementedError("MeepBackend is a milestone-2 stub; use AnalyticalBackend")
