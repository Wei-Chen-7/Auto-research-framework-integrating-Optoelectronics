"""Arm 1 — Baseline: simulate → read failure → one parameter fix → re-simulate → stop."""
from __future__ import annotations

import time

from picfix.agents.base import AgentTaskOutcome, BaseArm
from picfix.core.task import Task


class BaselineArm(BaseArm):
    name = "baseline"

    def run_task(self, task: Task) -> AgentTaskOutcome:
        self._reset_tokens()
        t0 = time.perf_counter()
        result, report, trace_id = self._simulate(task, task.init_params, 0, budget_used=0)
        sim_calls = 1
        if not report.passed:
            fixed = self._ask_param_fix(self._evidence(task, trace_id, result, report))
            result, report, _ = self._simulate(task, fixed, 1, budget_used=1)
            sim_calls = 2
        return AgentTaskOutcome(
            final_params=result.params,
            sim_calls=sim_calls,
            wall_time_s=time.perf_counter() - t0,
            verifier_passed=report.passed,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
        )
