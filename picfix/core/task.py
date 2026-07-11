"""Task instances and per-task outcome records."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from picfix.core.params import DesignParams
from picfix.core.proposal import GateDecision, Proposal
from picfix.core.truth import RootCause


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    init_params: DesignParams
    noise_seed: int


class JudgeVerdict(BaseModel):
    """Frozen judge output: per-criterion results + overall decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    visible_grid_ok: bool
    hidden_grid_ok: bool
    corners_ok: bool
    drc_ok: bool
    worst_ratio_dev: float
    max_il_db: float
    failed_criteria: tuple[str, ...]


class ProposalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: Proposal
    gate_decision: GateDecision | None  # None in the unguarded arm
    deployed: bool


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    arm: str
    repeat: int
    task_index: int                     # position in the (shuffled) sequence
    init_params: DesignParams
    final_params: DesignParams
    sim_calls: int
    wall_time_s: float
    verifier_passed: bool               # working-copy visible verification
    judge_passed: bool                  # frozen judge (ground truth of success)
    judge_failed_criteria: tuple[str, ...]
    truth_causes: tuple[RootCause, ...]  # ground truth for the final design
    tokens_in: int
    tokens_out: int
    r3_triggered: bool
    proposal_events: list[ProposalEvent]
    sds_values: list[float]
    rollbacks: int
    verifier_version: int
    workflow_version: int
    prior_version: int
