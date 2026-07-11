"""Metric computation from task results + trace log + root-cause ground truth.

Everything here consumes platform-side data only: frozen-judge verdicts,
the append-only trace, and the backend's truth channel. Nothing the agent
self-reports enters a primary endpoint.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from picfix.core.proposal import GateVerdict
from picfix.core.task import TaskResult
from picfix.core.trace import TraceRecord
from picfix.core.truth import RootCause
from picfix.simulators.base import SimulatorBackend
from picfix.sil.sds import jaccard


class ArmMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: str
    tasks: int
    # primary endpoints
    success_rate: float                # frozen judge + hidden suite
    false_accept_rate: float | None    # None when nothing passed visible checks
    visible_passes: int
    false_accepts: int
    # secondary
    repeated_failure_rate: float | None
    diagnose_accuracy: float | None
    rollback_frequency: float | None
    rollbacks: int
    time_to_fix_median_s: float | None
    time_to_fix_p90_s: float | None
    tokens_per_success: float | None
    sim_calls_per_success: float | None
    total_sim_calls: int
    total_tokens: int
    sds_mean: float | None
    sds_values: list[float]
    convergence_rounds: float | None   # mean over repeats that converged
    r3_triggers: int
    proposals_total: int
    proposals_deployed: int
    proposals_rejected: int
    proposals_pending_human: int


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    idx = (len(ordered) - 1) * q
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def false_accepts(results: list[TaskResult]) -> tuple[int, int]:
    """(false accepts, visible passes): a false accept is a design that
    cleared every visible check of the working copy but was vetoed by the
    frozen judge / hidden suite."""
    visible = [r for r in results if r.verifier_passed]
    fa = [r for r in visible if not r.judge_passed]
    return len(fa), len(visible)


def repeated_failure_rate(results: list[TaskResult]) -> float | None:
    """Recurrence of same-root-cause failures across tasks, computed on the
    backend's ground-truth labels per repeat sequence."""
    repeats_hit = 0
    opportunities = 0
    for rep in sorted({r.repeat for r in results}):
        seq = sorted((r for r in results if r.repeat == rep), key=lambda r: r.task_index)
        seen: set[RootCause] = set()
        for r in seq:
            if r.judge_passed:
                continue
            causes = set(r.truth_causes)
            if seen:
                opportunities += 1
                if causes & seen:
                    repeats_hit += 1
            seen |= causes
    return repeats_hit / opportunities if opportunities else None


def diagnose_accuracy(
    results: list[TaskResult], traces: list[TraceRecord], backend: SimulatorBackend
) -> float | None:
    """Mean Jaccard agreement between each diagnoser's root-cause labels and
    the ground truth of the design the diagnosed trace recorded."""
    by_id = {t.trace_id: t for t in traces}
    scores: list[float] = []
    for result in results:
        for nfo in result.nfos:
            trace = by_id.get(nfo.trace_id)
            if trace is None:
                continue
            truth = frozenset(c.value for c in backend.ground_truth(trace.params).causes)
            for diagnosis in nfo.diagnoses:
                scores.append(jaccard(frozenset(diagnosis.root_causes), truth))
    return sum(scores) / len(scores) if scores else None


def convergence_rounds(results: list[TaskResult]) -> float | None:
    """Per repeat: proposal rounds from the first R3 trigger until the start
    of 3 consecutive judge-true successes. Mean over repeats that converged."""
    rounds_per_repeat: list[int] = []
    for rep in sorted({r.repeat for r in results}):
        seq = sorted((r for r in results if r.repeat == rep), key=lambda r: r.task_index)
        first_trigger = next((i for i, r in enumerate(seq) if r.r3_triggered), None)
        if first_trigger is None:
            continue
        for j in range(first_trigger, len(seq) - 2):
            if all(seq[k].judge_passed for k in (j, j + 1, j + 2)):
                rounds_per_repeat.append(
                    sum(1 for r in seq[first_trigger : j + 1] if r.r3_triggered)
                )
                break
    return sum(rounds_per_repeat) / len(rounds_per_repeat) if rounds_per_repeat else None


def arm_metrics(
    arm: str,
    results: list[TaskResult],
    traces: list[TraceRecord],
    backend: SimulatorBackend,
) -> ArmMetrics:
    n = len(results)
    successes = [r for r in results if r.judge_passed]
    fa, visible = false_accepts(results)

    events = [e for r in results for e in r.proposal_events]
    deployed = sum(1 for e in events if e.deployed)
    rejected = sum(
        1
        for e in events
        if e.gate_decision is not None and e.gate_decision.verdict == GateVerdict.REJECTED
    )
    pending = sum(
        1
        for e in events
        if e.gate_decision is not None
        and e.gate_decision.verdict == GateVerdict.NEEDS_HUMAN_APPROVAL
    )
    rollbacks = sum(r.rollbacks for r in results)
    sds_values = [s for r in results for s in r.sds_values]
    fix_times = [r.wall_time_s for r in successes]
    total_tokens = sum(r.tokens_in + r.tokens_out for r in results)
    total_sim_calls = sum(r.sim_calls for r in results)

    return ArmMetrics(
        arm=arm,
        tasks=n,
        success_rate=len(successes) / n if n else 0.0,
        false_accept_rate=(fa / visible) if visible else None,
        visible_passes=visible,
        false_accepts=fa,
        repeated_failure_rate=repeated_failure_rate(results),
        diagnose_accuracy=diagnose_accuracy(results, traces, backend),
        rollback_frequency=(rollbacks / deployed) if deployed else None,
        rollbacks=rollbacks,
        time_to_fix_median_s=_percentile(fix_times, 0.5) if fix_times else None,
        time_to_fix_p90_s=_percentile(fix_times, 0.9) if fix_times else None,
        tokens_per_success=(total_tokens / len(successes)) if successes else None,
        sim_calls_per_success=(total_sim_calls / len(successes)) if successes else None,
        total_sim_calls=total_sim_calls,
        total_tokens=total_tokens,
        sds_mean=(sum(sds_values) / len(sds_values)) if sds_values else None,
        sds_values=sds_values,
        convergence_rounds=convergence_rounds(results),
        r3_triggers=sum(1 for r in results if r.r3_triggered),
        proposals_total=len(events),
        proposals_deployed=deployed,
        proposals_rejected=rejected,
        proposals_pending_human=pending,
    )


def write_csv(metrics: list[ArmMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f for f in ArmMetrics.model_fields if f != "sds_values"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metrics:
            row = m.model_dump(exclude={"sds_values"})
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
