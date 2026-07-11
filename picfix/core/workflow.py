"""Evolvable workflow state shared by the two meta arms, and the single
deployment routine both arms use.

The unguarded arm calls :func:`apply_proposal` directly (no review); the
governed arm reaches it only through the Gate. Assets outside the agent
permission boundary in ALL arms (judge, hidden suite, golden set — see
DESIGN.md §9 boundary note) are refused here at the platform level: that
refusal is experiment-platform integrity, not the governance under test.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from picfix.core.config import SearchStrategy
from picfix.core.proposal import Proposal, ProposalKind
from picfix.core.spec import WorkingVerifierConfig
from picfix.priors.store import PriorStore

# proposals no arm can deploy: the platform, not the Gate, refuses these
PLATFORM_FORBIDDEN: frozenset[ProposalKind] = frozenset(
    {ProposalKind.MODIFY_GOLDEN_SET, ProposalKind.MODIFY_JUDGE}
)


class MetaWorkflowState(BaseModel):
    """Cross-task persistent, versioned workflow state of a meta arm."""

    model_config = ConfigDict(extra="forbid")

    workflow_version: int
    verifier: WorkingVerifierConfig
    strategy: SearchStrategy
    r3_consecutive_failures: int


class DeployOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deployed: bool
    note: str
    new_prior_version: int | None = None


def apply_proposal(
    state: MetaWorkflowState, proposal: Proposal, priors: PriorStore
) -> DeployOutcome:
    """Mutate ``state``/``priors`` according to the proposal payload."""
    kind, payload = proposal.kind, proposal.payload

    if kind in PLATFORM_FORBIDDEN:
        return DeployOutcome(
            deployed=False,
            note=f"{kind} targets assets outside the agent permission boundary in all arms",
        )

    if kind == ProposalKind.MODIFY_VERIFIER_THRESHOLD:
        field = str(payload["field"])
        if field not in ("split_tolerance", "il_max_db"):
            return DeployOutcome(deployed=False, note=f"unknown verifier threshold {field}")
        setattr(state.verifier, field, float(payload["new_value"]))  # type: ignore[arg-type]
        state.verifier.version += 1
    elif kind == ProposalKind.MODIFY_VERIFIER_GRID:
        if "grid_points" in payload:
            state.verifier.grid_points = int(payload["grid_points"])  # type: ignore[arg-type]
        if "band_nm" in payload:
            lo, hi = payload["band_nm"]  # type: ignore[misc]
            state.verifier.band_nm = (float(lo), float(hi))
        state.verifier.version += 1
    elif kind == ProposalKind.REMOVE_VERIFIER_CASE:
        case = str(payload["case"])
        if case not in state.verifier.case_list:
            return DeployOutcome(deployed=False, note=f"case {case} not in working copy")
        state.verifier.case_list.remove(case)
        state.verifier.version += 1
    elif kind == ProposalKind.MODIFY_SEARCH_STRATEGY:
        field = str(payload["field"])
        if field not in ("max_step_gap_nm", "max_step_length_um", "max_step_width_nm"):
            return DeployOutcome(deployed=False, note=f"unknown strategy field {field}")
        setattr(state.strategy, field, float(payload["new_value"]))  # type: ignore[arg-type]
    elif kind == ProposalKind.MODIFY_INTERNAL_HEURISTIC:
        field = str(payload["field"])
        if field != "early_stop_margin":
            return DeployOutcome(deployed=False, note=f"unknown heuristic field {field}")
        state.strategy.early_stop_margin = float(payload["new_value"])  # type: ignore[arg-type]
    elif kind == ProposalKind.MODIFY_R3_TRIGGER:
        state.r3_consecutive_failures = int(payload["consecutive_task_failures"])  # type: ignore[arg-type]
    elif kind == ProposalKind.APPEND_PRIOR:
        version = priors.append(str(payload["text"]), str(payload["source_trace_id"]))
        state.workflow_version += 1
        return DeployOutcome(deployed=True, note="prior appended", new_prior_version=version)
    else:  # pragma: no cover - exhaustive over ProposalKind
        return DeployOutcome(deployed=False, note=f"unhandled proposal kind {kind}")

    state.workflow_version += 1
    return DeployOutcome(deployed=True, note=f"{kind} deployed")
