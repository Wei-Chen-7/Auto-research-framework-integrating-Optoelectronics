"""Simulator backend abstraction.

``simulate`` returns the agent-visible :class:`SimResult`;
``ground_truth`` computes the noise-free root-cause labels and is wired
only into the judge/metrics layer — agent code never receives it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from picfix.core.params import DesignParams
from picfix.core.spectrum import SimResult
from picfix.core.truth import GroundTruth


class SimulatorBackend(ABC):
    name: str

    @abstractmethod
    def simulate(
        self,
        params: DesignParams,
        wavelengths_nm: tuple[float, ...],
        *,
        noisy: bool,
        seed: int | None,
    ) -> SimResult:
        """Evaluate a design on a wavelength grid.

        ``noisy=True`` adds seeded Gaussian noise (agent-visible feedback);
        ``noisy=False`` is bit-for-bit deterministic (judge evaluations).
        """

    @abstractmethod
    def ground_truth(self, params: DesignParams) -> GroundTruth:
        """Noise-free root-cause labels. Metrics-layer only; hidden from agents."""
