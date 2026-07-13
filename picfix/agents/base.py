"""Shared agent scaffold.

All four arms use the same base model client, the same system-prompt
skeleton, the same hard simulation budget and the same trace-append
channel. The ONLY differences between arms are the governance mechanisms
(DESIGN.md §5): everything shared lives here.

The judge is deliberately absent from this module: agents return an
:class:`AgentTaskOutcome` and the experiment runner (platform code)
evaluates it against the frozen judge afterwards.
"""
from __future__ import annotations

import json
import re
import time
import uuid

from pydantic import BaseModel, ConfigDict

from picfix.agents.llm import LLMClient
from picfix.core.config import DRCRulesConfig, SearchStrategy
from picfix.core.nfo import NFO
from picfix.core.params import DesignParams
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.spectrum import SimResult
from picfix.core.task import ProposalEvent, Task
from picfix.core.trace import SpectrumSummary, TraceRecord, TraceWriter
from picfix.core.verifier import VerifierReport, verify
from picfix.layout import drc as drc_mod
from picfix.simulators.base import SimulatorBackend
from picfix.sil.evidence import FailureEvidence

SYSTEM_SKELETON = (
    "You are a photonic integrated circuit design agent fixing a directional "
    "coupler. Target: {target}:{target2} split (cross-port ratio {split_target} "
    "± {split_tolerance}) across the visible wavelength grid, insertion loss "
    "below {il_max_db} dB, and DRC clean (min gap {min_gap} nm, width "
    "{min_width}-{max_width} nm, length {min_length}-{max_length} um). "
    "You control three parameters: gap_nm, length_um, width_nm. Respond ONLY "
    "with a JSON object {{\"gap_nm\": <float>, \"length_um\": <float>, "
    "\"width_nm\": <float>, \"reasoning\": <string>}}.{priors_section}"
)


class BudgetExhausted(RuntimeError):
    pass


class AgentTaskOutcome(BaseModel):
    """What an arm reports back to the platform after one task."""

    model_config = ConfigDict(extra="forbid")

    final_params: DesignParams
    sim_calls: int
    wall_time_s: float
    verifier_passed: bool
    tokens_in: int
    tokens_out: int
    r3_triggered: bool = False
    proposal_events: list[ProposalEvent] = []
    nfos: list[NFO] = []
    sds_values: list[float] = []
    rollbacks: int = 0
    verifier_version: int = 1
    workflow_version: int = 1
    prior_version: int = 0


class ArmRuntime:
    """Platform services handed to every arm (identical across arms)."""

    def __init__(
        self,
        backend: SimulatorBackend,
        drc_rules: DRCRulesConfig,
        trace: TraceWriter,
        llm: LLMClient,
        budget_sim_calls: int,
    ) -> None:
        self.backend = backend
        self.drc_rules = drc_rules
        self.trace = trace
        self.llm = llm
        self.budget_sim_calls = budget_sim_calls


class BaseArm:
    name = "base"

    def __init__(self, runtime: ArmRuntime, verifier_cfg: WorkingVerifierConfig,
                 strategy: SearchStrategy) -> None:
        self.runtime = runtime
        self.verifier_cfg = verifier_cfg
        self.strategy = strategy
        self._tokens_in = 0
        self._tokens_out = 0

    # -- lifecycle ---------------------------------------------------------

    def run_task(self, task: Task) -> AgentTaskOutcome:  # pragma: no cover - abstract
        raise NotImplementedError

    def _reset_tokens(self) -> None:
        self._tokens_in = 0
        self._tokens_out = 0

    # -- shared machinery ----------------------------------------------------

    def _simulate(
        self, task: Task, params: DesignParams, call_index: int, budget_used: int
    ) -> tuple[SimResult, VerifierReport, str]:
        """One budgeted, traced, noisy simulation + working verification."""
        if budget_used >= self.runtime.budget_sim_calls:
            raise BudgetExhausted(f"hard budget of {self.runtime.budget_sim_calls} reached")
        seed = task.noise_seed * 1000 + call_index
        result = self.runtime.backend.simulate(
            params, self.verifier_cfg.grid(), noisy=True, seed=seed
        )
        report = verify(result, self.verifier_cfg, self.runtime.drc_rules)
        trace_id = f"{task.task_id}-c{call_index}"
        ratios = result.spectrum.cross_ratio
        self.runtime.trace.append(
            TraceRecord(
                trace_id=trace_id,
                task_id=task.task_id,
                arm=self.name,
                call_index=call_index,
                params=params,
                summary=SpectrumSummary(
                    worst_ratio_dev=report.worst_ratio_dev,
                    max_il_db=report.max_il_db,
                    center_ratio=ratios[len(ratios) // 2],
                ),
                backend=result.backend,
                seed=seed,
                elapsed_s=result.elapsed_s,
                agent_version=1,
                workflow_version=self._workflow_version(),
                verifier_version=self.verifier_cfg.version,
                prior_version=self._prior_version(),
            )
        )
        return result, report, trace_id

    def _workflow_version(self) -> int:
        return 1

    def _prior_version(self) -> int:
        return 0

    def _priors_prompt(self) -> str:
        return ""

    def _evidence(
        self, task: Task, trace_id: str, result: SimResult, report: VerifierReport
    ) -> FailureEvidence:
        ratios = result.spectrum.cross_ratio
        target = self.verifier_cfg.split_target
        return FailureEvidence(
            trace_id=trace_id,
            task_id=task.task_id,
            params=result.params,
            center_ratio=ratios[len(ratios) // 2],
            edge_ratio_dev=max(abs(ratios[0] - target), abs(ratios[-1] - target)),
            worst_ratio_dev=report.worst_ratio_dev,
            max_il_db=report.max_il_db,
            report=report,
            drc_rule_names=tuple(
                v.rule for v in drc_mod.check(result.params, self.runtime.drc_rules)
            ),
        )

    def _ask_param_fix(self, evidence: FailureEvidence) -> DesignParams:
        """One LLM round proposing new parameters, clamped to the arm's
        current search-strategy step limits (the strategy is part of the
        workflow, so it binds in both mock and API modes)."""
        system = SYSTEM_SKELETON.format(
            target=50,
            target2=50,
            split_target=self.verifier_cfg.split_target,
            split_tolerance=self.verifier_cfg.split_tolerance,
            il_max_db=self.verifier_cfg.il_max_db,
            min_gap=self.runtime.drc_rules.min_gap_nm,
            min_width=self.runtime.drc_rules.min_width_nm,
            max_width=self.runtime.drc_rules.max_width_nm,
            min_length=self.runtime.drc_rules.min_length_um,
            max_length=self.runtime.drc_rules.max_length_um,
            priors_section=self._priors_prompt(),
        )
        user = (
            f"Current design: gap={evidence.params.gap_nm:.2f} nm, "
            f"length={evidence.params.length_um:.4f} um, width={evidence.params.width_nm:.2f} nm.\n"
            f"Noisy feedback: centre ratio={evidence.center_ratio:.4f}, "
            f"worst deviation={evidence.worst_ratio_dev:.4f}, "
            f"band-edge deviation={evidence.edge_ratio_dev:.4f}, "
            f"max IL={evidence.max_il_db:.3f} dB.\n"
            f"Verifier checks: {evidence.report.checks}.\n"
            f"DRC violations: {', '.join(evidence.drc_rule_names) or 'none'}.\n"
            "Propose the next parameters."
        )
        context = {
            "request": "param_fix",
            "gap_nm": evidence.params.gap_nm,
            "length_um": evidence.params.length_um,
            "width_nm": evidence.params.width_nm,
            "center_ratio": evidence.center_ratio,
            "ratio_failed": not evidence.report.checks.get("ratio", True),
            "il_failed": not evidence.report.checks.get("il", True),
            "drc_violations": list(evidence.drc_rule_names),
            "strategy": self.strategy.model_dump(),
            "drc": self.runtime.drc_rules.model_dump(),
        }
        response = self.runtime.llm.complete(system, user, context)
        self._tokens_in += response.tokens_in
        self._tokens_out += response.tokens_out
        return self._clamp_to_strategy(evidence.params, _parse_params(response.text, evidence.params))

    def _clamp_to_strategy(self, old: DesignParams, new: DesignParams) -> DesignParams:
        def clamp(prev: float, nxt: float, max_step: float) -> float:
            delta = max(-max_step, min(max_step, nxt - prev))
            return prev + delta

        return DesignParams(
            gap_nm=clamp(old.gap_nm, new.gap_nm, self.strategy.max_step_gap_nm),
            length_um=clamp(old.length_um, new.length_um, self.strategy.max_step_length_um),
            width_nm=clamp(old.width_nm, new.width_nm, self.strategy.max_step_width_nm),
        )

    def _accepts(self, report: VerifierReport) -> bool:
        """Internal early-stop heuristic: with a positive margin the agent
        stops on near-misses (the verifier verdict itself is unaffected)."""
        if report.passed:
            return True
        margin = self.strategy.early_stop_margin
        if margin <= 0:
            return False
        ratio_ok = report.worst_ratio_dev <= self.verifier_cfg.split_tolerance + margin
        il_ok = report.max_il_db <= self.verifier_cfg.il_max_db + margin
        return ratio_ok and il_ok and not report.drc_violations


def _parse_params(text: str, fallback: DesignParams) -> DesignParams:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return fallback
    try:
        payload = json.loads(match.group(0))
        return DesignParams(
            gap_nm=float(payload["gap_nm"]),
            length_um=float(payload["length_um"]),
            width_nm=float(payload["width_nm"]),
        )
    except (KeyError, TypeError, ValueError):
        return fallback


def new_proposal_id() -> str:
    return f"prop-{uuid.uuid4().hex[:12]}"


def elapsed_since(t0: float) -> float:
    return time.perf_counter() - t0


def make_task_id() -> str:
    return f"task-{uuid.uuid4().hex[:8]}"
