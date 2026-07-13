"""Agent-visible failure evidence handed to the diagnosers.

Built strictly from what the agent itself can see: the noisy simulation
result and the working-verifier report. Ground-truth labels never enter.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from picfix.core.params import DesignParams
from picfix.core.verifier import VerifierReport


class FailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    task_id: str
    params: DesignParams
    center_ratio: float
    edge_ratio_dev: float       # worst |ratio-target| at the band edges
    worst_ratio_dev: float
    max_il_db: float
    report: VerifierReport
    drc_rule_names: tuple[str, ...]

    def ratio_failed(self, tolerance: float) -> bool:
        return self.worst_ratio_dev > tolerance

    def il_failed(self, il_max_db: float) -> bool:
        return self.max_il_db > il_max_db
