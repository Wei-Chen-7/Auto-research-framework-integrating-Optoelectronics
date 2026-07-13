"""Constitutional Guard — deterministic, two-tier proposal review.

Never consults an LLM: every check is a hash comparison, a policy table
lookup, a numeric direction test, or the deterministic golden regression.
An LLM signing off in place of the human approver is structurally
impossible here — there is no code path that could ask one.

Mounted on the governed (RAE) arm only. Tier (a): hard compliance —
proposals touching meta-immutable assets are rejected and recorded.
Tier (b): risk grading — loosening internal heuristics and prior-library
writes are high risk (needs_human_approval; recorded and skipped in batch
runs), everything else runs the golden regression and deploys versioned,
with automatic rollback if the live state later degrades.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from picfix.core.audit import AuditLog
from picfix.core.config import DRCRulesConfig
from picfix.core.proposal import GateDecision, GateVerdict, Proposal, ProposalKind, RiskLevel
from picfix.core.workflow import MetaWorkflowState, apply_proposal
from picfix.gate.integrity import HashManifest
from picfix.gate.regression import golden_regression_passes, load_regression_starts
from picfix.judge.frozen_judge import FrozenJudge
from picfix.priors.store import PriorStore
from picfix.simulators.analytical import AnalyticalBackend

_POLICY_PATH = Path(__file__).parent / "policy.yaml"


class ConstitutionalGuard:
    def __init__(
        self,
        repo_root: Path,
        config_path: Path,
        audit: AuditLog,
        backend: AnalyticalBackend,
        judge: FrozenJudge,
        drc_rules: DRCRulesConfig,
        golden_path: Path,
        sds_threshold: float,
        center_nm: float,
        policy_path: Path = _POLICY_PATH,
    ) -> None:
        with policy_path.open("r", encoding="utf-8") as f:
            policy = yaml.safe_load(f)
        self._hard_kinds = {ProposalKind(k) for k in policy["hard_reject_kinds"]}
        self._high_risk_kinds = {ProposalKind(k) for k in policy["high_risk_kinds"]}
        self._loosening_fields = set(policy["loosening_when_increased"])
        self._manifest = HashManifest(
            repo_root, policy["protected_files"], config_path, policy["protected_config_sections"]
        )
        self._audit = audit
        self._backend = backend
        self._judge = judge
        self._drc = drc_rules
        self._starts = load_regression_starts(golden_path)
        self._sds_threshold = sds_threshold
        self._center_nm = center_nm
        self._history: list[MetaWorkflowState] = []

    # -- integrity ---------------------------------------------------------

    def verify_integrity(self) -> None:
        """Abort the experiment if any meta-immutable asset changed."""
        self._manifest.verify()
        self._audit.verify_chain()

    # -- review ------------------------------------------------------------

    def review_and_deploy(
        self, state: MetaWorkflowState, proposal: Proposal, priors: PriorStore
    ) -> GateDecision:
        self.verify_integrity()

        if proposal.kind in self._hard_kinds:
            return self._record(
                GateDecision(
                    proposal_id=proposal.proposal_id,
                    verdict=GateVerdict.REJECTED,
                    risk=RiskLevel.HARD_VIOLATION,
                    reasons=(f"{proposal.kind} touches a meta-immutable asset",),
                )
            )

        if proposal.kind == ProposalKind.APPEND_PRIOR:
            if proposal.sds is not None and proposal.sds > self._sds_threshold:
                return self._record(
                    GateDecision(
                        proposal_id=proposal.proposal_id,
                        verdict=GateVerdict.REJECTED,
                        risk=RiskLevel.HIGH,
                        reasons=(
                            f"SDS {proposal.sds:.2f} > {self._sds_threshold:.2f}: "
                            "prior write frozen, needs_human_review",
                        ),
                    )
                )
            return self._record(
                GateDecision(
                    proposal_id=proposal.proposal_id,
                    verdict=GateVerdict.NEEDS_HUMAN_APPROVAL,
                    risk=RiskLevel.HIGH,
                    reasons=("prior-library write requires human approval; skipped in batch run",),
                )
            )

        if proposal.kind == ProposalKind.MODIFY_INTERNAL_HEURISTIC:
            field = str(proposal.payload.get("field"))
            if field in self._loosening_fields:
                current = getattr(state.strategy, field)
                if float(proposal.payload["new_value"]) > current:  # type: ignore[arg-type]
                    return self._record(
                        GateDecision(
                            proposal_id=proposal.proposal_id,
                            verdict=GateVerdict.NEEDS_HUMAN_APPROVAL,
                            risk=RiskLevel.HIGH,
                            reasons=(
                                f"loosening internal heuristic {field} requires human approval",
                            ),
                        )
                    )

        # low risk: trial-deploy on a copy, then golden regression
        candidate = state.model_copy(deep=True)
        outcome = apply_proposal(candidate, proposal, priors)
        if not outcome.deployed:
            return self._record(
                GateDecision(
                    proposal_id=proposal.proposal_id,
                    verdict=GateVerdict.REJECTED,
                    risk=RiskLevel.LOW,
                    reasons=(outcome.note,),
                )
            )
        regression_ok = golden_regression_passes(
            candidate.strategy, self._starts, self._backend, self._judge, self._drc, self._center_nm
        )
        if not regression_ok:
            return self._record(
                GateDecision(
                    proposal_id=proposal.proposal_id,
                    verdict=GateVerdict.REJECTED,
                    risk=RiskLevel.LOW,
                    reasons=("golden-set regression degraded",),
                    golden_regression_passed=False,
                )
            )

        self._history.append(state.model_copy(deep=True))
        _copy_state(candidate, into=state)
        return self._record(
            GateDecision(
                proposal_id=proposal.proposal_id,
                verdict=GateVerdict.APPROVED,
                risk=RiskLevel.LOW,
                reasons=("golden-set regression passed",),
                golden_regression_passed=True,
                deployed_version=state.workflow_version,
            )
        )

    # -- post-deploy monitoring --------------------------------------------

    def check_and_rollback(self, state: MetaWorkflowState) -> bool:
        """Golden check on the LIVE state; roll back to the previous version
        on degradation. Returns True if a rollback happened."""
        self.verify_integrity()
        ok = golden_regression_passes(
            state.strategy, self._starts, self._backend, self._judge, self._drc, self._center_nm
        )
        if ok or not self._history:
            return False
        previous = self._history.pop()
        rolled_back_from = state.workflow_version
        _copy_state(previous, into=state)
        self._audit.append(
            "rollback",
            {"from_version": rolled_back_from, "to_version": state.workflow_version},
        )
        return True

    def _record(self, decision: GateDecision) -> GateDecision:
        self._audit.append("gate_decision", decision.model_dump(mode="json"))
        return decision


def _copy_state(source: MetaWorkflowState, *, into: MetaWorkflowState) -> None:
    into.workflow_version = source.workflow_version
    into.verifier = source.verifier.model_copy(deep=True)
    into.strategy = source.strategy.model_copy(deep=True)
    into.r3_consecutive_failures = source.r3_consecutive_failures
