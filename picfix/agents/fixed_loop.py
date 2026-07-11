"""Arm 2 — FixedLoop (R2): fixed adjust→simulate→read loop to budget
exhaustion. Prompts, verification logic and search strategy are locked:
no component of the workflow can be self-modified, and no state persists
across tasks."""
from __future__ import annotations

import time

from picfix.agents.base import AgentTaskOutcome, BaseArm
from picfix.core.task import Task


class FixedLoopArm(BaseArm):
    name = "fixed_loop"

    def run_task(self, task: Task) -> AgentTaskOutcome:
        self._reset_tokens()
        t0 = time.perf_counter()
        params = task.init_params
        result, report, trace_id = self._simulate(task, params, 0, budget_used=0)
        sim_calls = 1
        while not self._accepts(report) and sim_calls < self.runtime.budget_sim_calls:
            params = self._ask_param_fix(self._evidence(task, trace_id, result, report))
            result, report, trace_id = self._simulate(task, params, sim_calls, budget_used=sim_calls)
            sim_calls += 1
        return AgentTaskOutcome(
            final_params=result.params,
            sim_calls=sim_calls,
            wall_time_s=time.perf_counter() - t0,
            verifier_passed=report.passed,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
        )
