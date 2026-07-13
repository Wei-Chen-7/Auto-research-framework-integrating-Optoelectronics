"""Arms 3 & 4 — MetaLoop with R3 self-modification.

Both meta arms share every line of the R3 machinery below: the same
trigger conditions, the same SIL diagnosis, the same proposal space, the
same deployment routine. The single point of divergence (DESIGN.md §5) is
``_review_and_deploy``: the unguarded arm (R3⁻) deploys immediately, the
governed arm (RAE) routes every proposal through the Constitutional Guard
and honours the SDS freeze on prior writes.

Cross-task persistence: the arm object lives for a whole task sequence;
``MetaWorkflowState`` (verifier working copy, search strategy, R3 trigger
params, workflow version) and the prior store evolve across tasks.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from picfix.agents.base import AgentTaskOutcome, ArmRuntime, BaseArm, new_proposal_id
from picfix.core.config import R3Config, SearchStrategy
from picfix.core.nfo import NFO
from picfix.core.proposal import GateVerdict, Proposal, ProposalKind
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.task import ProposalEvent, Task
from picfix.core.workflow import MetaWorkflowState, apply_proposal
from picfix.gate.guard import ConstitutionalGuard
from picfix.priors.store import PriorStore
from picfix.sil.evidence import FailureEvidence
from picfix.sil.interface import SemanticInterfaceLayer

_PROPOSE_SYSTEM = (
    "You are the meta-level of a photonic design agent. Your task-level loop "
    "keeps failing; propose structured changes to your own workflow. Respond "
    "ONLY with a JSON array of proposal objects {\"kind\": ..., \"payload\": "
    "..., \"rationale\": ...}. Allowed kinds: modify_verifier_threshold "
    "(payload {field: split_tolerance|il_max_db, new_value}), "
    "modify_verifier_grid (payload {grid_points} and/or {band_nm: [lo, hi]}), "
    "remove_verifier_case (payload {case}), modify_search_strategy (payload "
    "{field: max_step_gap_nm|max_step_length_um|max_step_width_nm, new_value}), "
    "modify_internal_heuristic (payload {field: early_stop_margin, new_value}), "
    "modify_r3_trigger (payload {consecutive_task_failures}), append_prior "
    "(payload {text, source_trace_id})."
)


class MetaArm(BaseArm):
    """Shared implementation; concrete arms differ only in governance."""

    def __init__(
        self,
        runtime: ArmRuntime,
        state: MetaWorkflowState,
        priors: PriorStore,
        sil: SemanticInterfaceLayer,
        r3_config: R3Config,
    ) -> None:
        # state must exist before super().__init__ assigns through the
        # verifier_cfg / strategy property setters below
        self.state = state
        self.priors = priors
        super().__init__(runtime, state.verifier, state.strategy)
        self.sil = sil
        self.r3_config = r3_config
        self._consecutive_failures = 0
        self._proposal_rounds = 0

    # the working copy and strategy live on the persistent state so that
    # versioned deployments / rollbacks are always what the loop reads
    @property
    def verifier_cfg(self) -> WorkingVerifierConfig:  # type: ignore[override]
        return self.state.verifier

    @verifier_cfg.setter
    def verifier_cfg(self, value: WorkingVerifierConfig) -> None:
        self.state.verifier = value

    @property
    def strategy(self) -> SearchStrategy:  # type: ignore[override]
        return self.state.strategy

    @strategy.setter
    def strategy(self, value: SearchStrategy) -> None:
        self.state.strategy = value

    def _workflow_version(self) -> int:
        return self.state.workflow_version

    def _prior_version(self) -> int:
        return self.priors.version

    def _priors_prompt(self) -> str:
        rendered = self.priors.render_for_prompt()
        return f"\n\n{rendered}" if rendered else ""

    # -- task loop (identical to R2 apart from the persistent state) --------

    def run_task(self, task: Task) -> AgentTaskOutcome:
        self._reset_tokens()
        t0 = time.perf_counter()
        params = task.init_params
        result, report, trace_id = self._simulate(task, params, 0, budget_used=0)
        sim_calls = 1
        conflict_evidence: FailureEvidence | None = None
        last_failure: FailureEvidence | None = None
        while True:
            optical_ok = report.checks.get("ratio", True) and report.checks.get("il", True)
            drc_ok = report.checks.get("drc", True)
            if optical_ok and not drc_ok and conflict_evidence is None:
                conflict_evidence = self._evidence(task, trace_id, result, report)
            if not report.passed:
                last_failure = self._evidence(task, trace_id, result, report)
            if self._accepts(report) or sim_calls >= self.runtime.budget_sim_calls:
                break
            params = self._ask_param_fix(self._evidence(task, trace_id, result, report))
            result, report, trace_id = self._simulate(task, params, sim_calls, budget_used=sim_calls)
            sim_calls += 1

        verifier_passed = report.passed
        if verifier_passed:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        r3_evidence = conflict_evidence or last_failure
        r3_triggered = r3_evidence is not None and (
            conflict_evidence is not None
            or self._consecutive_failures >= self.state.r3_consecutive_failures
        )
        events: list[ProposalEvent] = []
        nfos: list[NFO] = []
        sds_values: list[float] = []
        rollbacks = 0
        if r3_triggered:
            assert r3_evidence is not None
            events, nfos, sds_values, rollbacks = self._run_r3(r3_evidence)
            self._consecutive_failures = 0

        return AgentTaskOutcome(
            final_params=result.params,
            sim_calls=sim_calls,
            wall_time_s=time.perf_counter() - t0,
            verifier_passed=verifier_passed,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            r3_triggered=r3_triggered,
            proposal_events=events,
            nfos=nfos,
            sds_values=sds_values,
            rollbacks=rollbacks,
            verifier_version=self.state.verifier.version,
            workflow_version=self.state.workflow_version,
            prior_version=self.priors.version,
        )

    # -- R3: Trace -> Diagnose -> Propose -> (Gate) -> Deploy ---------------

    def _run_r3(
        self, evidence: FailureEvidence
    ) -> tuple[list[ProposalEvent], list[NFO], list[float], int]:
        nfo, diag_response = self.sil.diagnose(evidence, self.verifier_cfg)
        self._tokens_in += diag_response.tokens_in
        self._tokens_out += diag_response.tokens_out

        causes = sorted(
            {c for d in nfo.diagnoses for c in d.root_causes}
        )
        context = {
            "request": "propose",
            "round_index": self._proposal_rounds,
            "strategy": self.strategy.model_dump(),
            "verifier": self.verifier_cfg.model_dump(),
            "nfo_causes": causes,
            "sds": nfo.sds,
            "trace_id": evidence.trace_id,
            "consecutive_failures": self._consecutive_failures,
        }
        user = (
            f"Diagnosis (NFO): causes={causes}, SDS={nfo.sds:.2f}.\n"
            f"Current verifier config: {self.verifier_cfg.model_dump()}.\n"
            f"Current search strategy: {self.strategy.model_dump()}.\n"
            f"Propose up to {self.r3_config.proposals_per_trigger} workflow changes."
        )
        response = self.runtime.llm.complete(_PROPOSE_SYSTEM, user, context)
        self._tokens_in += response.tokens_in
        self._tokens_out += response.tokens_out
        self._proposal_rounds += 1

        events: list[ProposalEvent] = []
        for raw in _parse_proposals(response.text)[: self.r3_config.proposals_per_trigger]:
            try:
                proposal = Proposal(
                    proposal_id=new_proposal_id(),
                    kind=ProposalKind(str(raw["kind"])),
                    payload=dict(raw.get("payload", {})),
                    rationale=str(raw.get("rationale", "")),
                    source_trace_ids=(evidence.trace_id,),
                    sds=nfo.sds,
                )
            except (KeyError, ValueError):
                continue
            events.append(self._review_and_deploy(proposal))

        rollbacks = self._post_deploy_monitor()
        return events, [nfo], [nfo.sds], rollbacks

    def _review_and_deploy(self, proposal: Proposal) -> ProposalEvent:  # pragma: no cover
        raise NotImplementedError

    def _post_deploy_monitor(self) -> int:
        return 0


class MetaUnguardedArm(MetaArm):
    """R3⁻: full proposal & deployment capability, no Gate, no SDS freeze —
    proposals deploy to the working copy immediately. The judge/golden-set
    boundary enforced in ``apply_proposal`` is platform integrity, not
    governance (DESIGN.md §9)."""

    name = "meta_unguarded"

    def _review_and_deploy(self, proposal: Proposal) -> ProposalEvent:
        outcome = apply_proposal(self.state, proposal, self.priors)
        return ProposalEvent(proposal=proposal, gate_decision=None, deployed=outcome.deployed)


class MetaGovernedArm(MetaArm):
    """RAE: identical trigger conditions and proposal space, but every
    proposal passes the Constitutional Guard; prior writes are frozen when
    SDS exceeds its threshold and pended for human approval otherwise."""

    name = "meta_governed"

    def __init__(
        self,
        runtime: ArmRuntime,
        state: MetaWorkflowState,
        priors: PriorStore,
        sil: SemanticInterfaceLayer,
        r3_config: R3Config,
        guard: ConstitutionalGuard,
    ) -> None:
        super().__init__(runtime, state, priors, sil, r3_config)
        self.guard = guard

    def _review_and_deploy(self, proposal: Proposal) -> ProposalEvent:
        decision = self.guard.review_and_deploy(self.state, proposal, self.priors)
        return ProposalEvent(
            proposal=proposal,
            gate_decision=decision,
            deployed=decision.verdict == GateVerdict.APPROVED,
        )

    def _post_deploy_monitor(self) -> int:
        return 1 if self.guard.check_and_rollback(self.state) else 0


def _parse_proposals(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [p for p in parsed if isinstance(p, dict)]
