"""R3 proposals and gate decisions.

Proposals are the only mechanism through which a meta arm may change its
own workflow. They are structured (strict JSON schema, ``extra="forbid"``);
free-form runtime hot-patching is not representable.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ProposalKind(StrEnum):
    MODIFY_VERIFIER_THRESHOLD = "modify_verifier_threshold"
    MODIFY_VERIFIER_GRID = "modify_verifier_grid"
    REMOVE_VERIFIER_CASE = "remove_verifier_case"
    MODIFY_SEARCH_STRATEGY = "modify_search_strategy"
    MODIFY_INTERNAL_HEURISTIC = "modify_internal_heuristic"
    MODIFY_R3_TRIGGER = "modify_r3_trigger"
    APPEND_PRIOR = "append_prior"
    MODIFY_GOLDEN_SET = "modify_golden_set"
    MODIFY_JUDGE = "modify_judge"


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    kind: ProposalKind
    payload: dict[str, object]
    rationale: str
    source_trace_ids: tuple[str, ...]
    sds: float | None = None  # SDS of the diagnosis round that produced this


class GateVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"


class RiskLevel(StrEnum):
    HARD_VIOLATION = "hard_violation"
    HIGH = "high"
    LOW = "low"


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    verdict: GateVerdict
    risk: RiskLevel
    reasons: tuple[str, ...]
    golden_regression_passed: bool | None = None
    deployed_version: int | None = None
