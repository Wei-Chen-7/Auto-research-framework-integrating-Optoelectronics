"""Config-driven four-arm experiment runner.

    python -m picfix.experiments.run --config configs/coupler_v1.yaml --arm all

Per repeat k: one seeded, shuffled task sequence shared by all arms (paired
comparison). Meta arms keep their evolvable state across the sequence and
reset between repeats; baseline/R2 carry no state at all. Every success
judgement comes from the frozen judge; agents never see it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from picfix.agents.base import ArmRuntime, BaseArm
from picfix.agents.baseline import BaselineArm
from picfix.agents.fixed_loop import FixedLoopArm
from picfix.agents.llm import AnthropicClient, LLMClient, MockLLM, OpenAICompatibleClient
from picfix.agents.meta import MetaGovernedArm, MetaUnguardedArm
from picfix.core.audit import AuditLog
from picfix.core.config import ExperimentConfig, load_config
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.task import Task, TaskResult
from picfix.core.trace import TraceWriter
from picfix.core.workflow import MetaWorkflowState
from picfix.experiments.tasks import generate_tasks
from picfix.gate.guard import ConstitutionalGuard
from picfix.judge.frozen_judge import FrozenJudge
from picfix.metrics.compute import ArmMetrics, arm_metrics, write_csv
from picfix.metrics.plots import plot_comparison
from picfix.priors.store import PriorStore
from picfix.simulators.analytical import AnalyticalBackend
from picfix.sil.interface import SemanticInterfaceLayer

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "golden" / "coupler_golden.json"

ARM_NAMES = ("baseline", "fixed_loop", "meta_unguarded", "meta_governed")


def build_llm(cfg: ExperimentConfig) -> LLMClient:
    if cfg.llm.mode == "api":
        if cfg.llm.provider == "openai_compatible":
            return OpenAICompatibleClient(cfg.llm)
        return AnthropicClient(cfg.llm)
    return MockLLM(cfg.llm)


def build_arm(
    arm_name: str,
    cfg: ExperimentConfig,
    backend: AnalyticalBackend,
    judge: FrozenJudge,
    llm: LLMClient,
    out_dir: Path,
    config_path: Path,
    repeat: int,
) -> BaseArm:
    runtime = ArmRuntime(
        backend=backend,
        drc_rules=cfg.drc,
        trace=TraceWriter(out_dir / f"trace_{arm_name}.jsonl"),
        llm=llm,
        budget_sim_calls=cfg.experiment.budget_sim_calls,
    )
    if arm_name == "baseline":
        return BaselineArm(runtime, WorkingVerifierConfig.from_spec(cfg.spec),
                           cfg.search_strategy.model_copy(deep=True))
    if arm_name == "fixed_loop":
        return FixedLoopArm(runtime, WorkingVerifierConfig.from_spec(cfg.spec),
                            cfg.search_strategy.model_copy(deep=True))

    state = MetaWorkflowState(
        workflow_version=1,
        verifier=WorkingVerifierConfig.from_spec(cfg.spec),
        strategy=cfg.search_strategy.model_copy(deep=True),
        r3_consecutive_failures=cfg.r3.consecutive_task_failures,
    )
    priors = PriorStore(out_dir / f"priors_{arm_name}_r{repeat}.json")
    sil = SemanticInterfaceLayer(llm, cfg.r3.sds_threshold)
    if arm_name == "meta_unguarded":
        return MetaUnguardedArm(runtime, state, priors, sil, cfg.r3)
    if arm_name == "meta_governed":
        guard = ConstitutionalGuard(
            repo_root=REPO_ROOT,
            config_path=config_path,
            audit=AuditLog(out_dir / f"audit_{arm_name}_r{repeat}.jsonl"),
            backend=backend,
            judge=judge,
            drc_rules=cfg.drc,
            golden_path=GOLDEN_PATH,
            sds_threshold=cfg.r3.sds_threshold,
            center_nm=cfg.frozen_constants().lambda0_nm,
        )
        return MetaGovernedArm(runtime, state, priors, sil, cfg.r3, guard)
    raise ValueError(f"unknown arm {arm_name}")


def run_arm_sequence(
    arm: BaseArm,
    tasks: list[Task],
    judge: FrozenJudge,
    backend: AnalyticalBackend,
    repeat: int,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for index, task in enumerate(tasks):
        outcome = arm.run_task(task)
        verdict = judge.evaluate(outcome.final_params)          # frozen judge decides
        truth = backend.ground_truth(outcome.final_params)      # metrics-only channel
        results.append(
            TaskResult(
                task_id=task.task_id,
                arm=arm.name,
                repeat=repeat,
                task_index=index,
                init_params=task.init_params,
                final_params=outcome.final_params,
                sim_calls=outcome.sim_calls,
                wall_time_s=outcome.wall_time_s,
                verifier_passed=outcome.verifier_passed,
                judge_passed=verdict.passed,
                judge_failed_criteria=verdict.failed_criteria,
                truth_causes=tuple(sorted(truth.causes)),
                tokens_in=outcome.tokens_in,
                tokens_out=outcome.tokens_out,
                r3_triggered=outcome.r3_triggered,
                proposal_events=outcome.proposal_events,
                nfos=outcome.nfos,
                sds_values=outcome.sds_values,
                rollbacks=outcome.rollbacks,
                verifier_version=outcome.verifier_version,
                workflow_version=outcome.workflow_version,
                prior_version=outcome.prior_version,
            )
        )
    return results


def run_experiment(
    config_path: Path, arms: tuple[str, ...], llm_mode: str | None = None,
    tasks_override: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    if llm_mode is not None:
        cfg.llm.mode = llm_mode
    backend = AnalyticalBackend(
        cfg.frozen_constants(), cfg.spec, cfg.drc,
        sigma_ratio=cfg.noise.sigma_ratio, sigma_il_db=cfg.noise.sigma_il_db,
    )
    judge = FrozenJudge(cfg.spec, cfg.drc, backend)

    # refuse to run sick: the calibrated golden device must pass the judge
    if not judge.evaluate(cfg.golden_params()).passed:
        print("FATAL: golden device fails the frozen judge — recalibrate first", file=sys.stderr)
        raise SystemExit(1)

    llm = build_llm(cfg)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / cfg.experiment.output_dir / cfg.experiment.name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config_snapshot.yaml")

    tasks_per_arm = tasks_override or cfg.experiment.tasks_per_arm
    all_results: dict[str, list[TaskResult]] = {a: [] for a in arms}
    for repeat in range(cfg.experiment.repeats):
        rng = np.random.default_rng(cfg.experiment.seed + repeat)
        tasks = generate_tasks(cfg, judge, tasks_per_arm, rng)
        for arm_name in arms:
            arm = build_arm(arm_name, cfg, backend, judge, llm, out_dir, config_path, repeat)
            results = run_arm_sequence(arm, tasks, judge, backend, repeat)
            all_results[arm_name].extend(results)
            ok = sum(r.judge_passed for r in results)
            print(f"[repeat {repeat}] {arm_name}: {ok}/{len(results)} judge-true successes")

    with (out_dir / "task_results.jsonl").open("w", encoding="utf-8") as f:
        for arm_name in arms:
            for r in all_results[arm_name]:
                f.write(r.model_dump_json() + "\n")

    metrics: list[ArmMetrics] = []
    for arm_name in arms:
        trace_path = out_dir / f"trace_{arm_name}.jsonl"
        traces = TraceWriter(trace_path).read_all() if trace_path.exists() else []
        metrics.append(arm_metrics(arm_name, all_results[arm_name], traces, backend))

    write_csv(metrics, out_dir / "metrics.csv")
    plot_comparison(metrics, out_dir / "comparison.png")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in metrics], f, indent=2)

    print(f"\nresults written to {out_dir}")
    for m in metrics:
        far = "n/a" if m.false_accept_rate is None else f"{m.false_accept_rate:.2f}"
        print(
            f"  {m.arm:16s} success={m.success_rate:.2f} FAR={far} "
            f"r3={m.r3_triggers} deployed={m.proposals_deployed} "
            f"rejected={m.proposals_rejected} pending={m.proposals_pending_human} "
            f"rollbacks={m.rollbacks}"
        )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "coupler_v1.yaml")
    parser.add_argument("--arm", default="all", help="all | baseline | fixed_loop | meta_unguarded | meta_governed")
    parser.add_argument("--tasks", type=int, default=None, help="override tasks per arm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock-llm", action="store_true", help="force the scripted mock model")
    mode.add_argument("--api-llm", action="store_true", help="force the Anthropic API model")
    args = parser.parse_args(argv)

    arms = ARM_NAMES if args.arm == "all" else (args.arm,)
    if any(a not in ARM_NAMES for a in arms):
        parser.error(f"unknown arm {args.arm}")
    llm_mode = "mock" if args.mock_llm else "api" if args.api_llm else None
    run_experiment(args.config, arms, llm_mode=llm_mode, tasks_override=args.tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
