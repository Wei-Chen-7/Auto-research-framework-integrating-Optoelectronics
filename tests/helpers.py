"""Shared builders for agent-level tests."""
from __future__ import annotations

from pathlib import Path

from picfix.agents.base import ArmRuntime
from picfix.agents.llm import MockLLM
from picfix.agents.meta import MetaGovernedArm, MetaUnguardedArm
from picfix.core.audit import AuditLog
from picfix.core.config import ExperimentConfig
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.trace import TraceWriter
from picfix.core.workflow import MetaWorkflowState
from picfix.gate.guard import ConstitutionalGuard
from picfix.judge.frozen_judge import FrozenJudge
from picfix.priors.store import PriorStore
from picfix.simulators.analytical import AnalyticalBackend
from picfix.sil.interface import SemanticInterfaceLayer
from tests.conftest import CONFIG_PATH, GOLDEN_PATH, REPO


def make_runtime(cfg: ExperimentConfig, backend: AnalyticalBackend, tmp: Path) -> ArmRuntime:
    return ArmRuntime(
        backend=backend,
        drc_rules=cfg.drc,
        trace=TraceWriter(tmp / "trace.jsonl"),
        llm=MockLLM(cfg.llm),
        budget_sim_calls=cfg.experiment.budget_sim_calls,
    )


def make_state(cfg: ExperimentConfig) -> MetaWorkflowState:
    return MetaWorkflowState(
        workflow_version=1,
        verifier=WorkingVerifierConfig.from_spec(cfg.spec),
        strategy=cfg.search_strategy.model_copy(deep=True),
        r3_consecutive_failures=cfg.r3.consecutive_task_failures,
    )


def make_unguarded(cfg: ExperimentConfig, backend: AnalyticalBackend, tmp: Path) -> MetaUnguardedArm:
    runtime = make_runtime(cfg, backend, tmp)
    return MetaUnguardedArm(
        runtime,
        make_state(cfg),
        PriorStore(tmp / "priors.json"),
        SemanticInterfaceLayer(runtime.llm, cfg.r3.sds_threshold),
        cfg.r3,
    )


def make_governed(
    cfg: ExperimentConfig, backend: AnalyticalBackend, tmp: Path
) -> tuple[MetaGovernedArm, AuditLog]:
    runtime = make_runtime(cfg, backend, tmp)
    audit = AuditLog(tmp / "audit.jsonl")
    judge = FrozenJudge(cfg.spec, cfg.drc, backend)
    guard = ConstitutionalGuard(
        repo_root=REPO,
        config_path=CONFIG_PATH,
        audit=audit,
        backend=backend,
        judge=judge,
        drc_rules=cfg.drc,
        golden_path=GOLDEN_PATH,
        sds_threshold=cfg.r3.sds_threshold,
        center_nm=cfg.frozen_constants().lambda0_nm,
    )
    arm = MetaGovernedArm(
        runtime,
        make_state(cfg),
        PriorStore(tmp / "priors.json"),
        SemanticInterfaceLayer(runtime.llm, cfg.r3.sds_threshold),
        cfg.r3,
        guard,
    )
    return arm, audit
