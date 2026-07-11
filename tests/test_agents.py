"""M9 acceptance: budget cap, baseline shape, R3 trigger conditions."""
from __future__ import annotations

from pathlib import Path

import pytest

from picfix.agents.base import BudgetExhausted
from picfix.agents.baseline import BaselineArm
from picfix.agents.fixed_loop import FixedLoopArm
from picfix.core.config import ExperimentConfig
from picfix.core.params import DesignParams
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.task import Task
from picfix.simulators.analytical import AnalyticalBackend
from tests.helpers import make_runtime, make_state, make_unguarded


def _task(params: DesignParams, seed: int = 1) -> Task:
    return Task(task_id=f"t-seed{seed}", init_params=params, noise_seed=seed)


def test_baseline_simulates_at_most_twice(
    cfg: ExperimentConfig, backend: AnalyticalBackend, tmp_path: Path
) -> None:
    golden = cfg.golden_params()
    runtime = make_runtime(cfg, backend, tmp_path)
    arm = BaselineArm(runtime, WorkingVerifierConfig.from_spec(cfg.spec),
                      cfg.search_strategy.model_copy(deep=True))
    failing = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 0.75,
                           width_nm=golden.width_nm)
    outcome = arm.run_task(_task(failing))
    assert outcome.sim_calls == 2
    assert outcome.final_params != failing
    assert len(runtime.trace.read_all()) == 2


def test_hard_budget_is_enforced(
    cfg: ExperimentConfig, backend: AnalyticalBackend, tmp_path: Path
) -> None:
    golden = cfg.golden_params()
    runtime = make_runtime(cfg, backend, tmp_path)
    strategy = cfg.search_strategy.model_copy(deep=True)
    strategy.max_step_length_um = 0.01  # crippled: cannot traverse the error
    arm = FixedLoopArm(runtime, WorkingVerifierConfig.from_spec(cfg.spec), strategy)
    far_off = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um + 6.0,
                           width_nm=golden.width_nm)
    outcome = arm.run_task(_task(far_off))
    assert outcome.sim_calls == cfg.experiment.budget_sim_calls  # exactly the cap
    assert not outcome.verifier_passed
    with pytest.raises(BudgetExhausted):
        arm._simulate(_task(far_off), far_off, 99, budget_used=cfg.experiment.budget_sim_calls)


def test_r3_triggers_after_three_consecutive_failures(
    cfg: ExperimentConfig, backend: AnalyticalBackend, tmp_path: Path
) -> None:
    golden = cfg.golden_params()
    arm = make_unguarded(cfg, backend, tmp_path)
    arm.state.strategy.max_step_length_um = 0.01  # force visible failures
    far_off = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um + 6.0,
                           width_nm=golden.width_nm)
    outcomes = [arm.run_task(_task(far_off, seed=i)) for i in range(1, 4)]
    assert [o.r3_triggered for o in outcomes] == [False, False, True]
    assert outcomes[2].proposal_events and outcomes[2].sds_values


def test_r3_triggers_on_optical_pass_drc_fail_conflict(
    cfg: ExperimentConfig, backend: AnalyticalBackend, tmp_path: Path
) -> None:
    golden = cfg.golden_params()
    arm = make_unguarded(cfg, backend, tmp_path)
    # width below DRC minimum but optics tuned to pass: kappa gains
    # (1 + 0.0006*(500-380)) so the 50:50 length shortens accordingly
    kappa_factor = 1.0 + 0.0006 * (500.0 - 380.0)
    conflicted = DesignParams(
        gap_nm=golden.gap_nm,
        length_um=golden.length_um / kappa_factor,
        width_nm=380.0,
    )
    outcome = arm.run_task(_task(conflicted))
    assert outcome.r3_triggered  # conflict fires even if the task later succeeds


def test_cross_task_persistence_and_prior_injection(
    cfg: ExperimentConfig, backend: AnalyticalBackend, tmp_path: Path
) -> None:
    golden = cfg.golden_params()
    arm = make_unguarded(cfg, backend, tmp_path)
    arm.state.strategy.max_step_length_um = 0.01
    far_off = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um + 6.0,
                           width_nm=golden.width_nm)
    for i in range(1, 4):
        outcome = arm.run_task(_task(far_off, seed=i))
    # round 0 of the mock proposal policy appends a prior and widens steps
    assert arm.priors.version >= 1
    assert "length" in arm._priors_prompt() or "gap" in arm._priors_prompt()
    assert arm.state.workflow_version > 1
    assert outcome.prior_version == arm.priors.version
    # the strategy change persists into the next task's prompt context
    assert arm.state.strategy.max_step_length_um != 0.01
