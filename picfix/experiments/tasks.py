"""Task generation: random offsets from the golden parameters.

Feasibility by construction: the golden design passes the frozen judge
(verified at runner startup), so a solution always exists. Initial designs
that already pass the judge are rejected and regenerated — a task must
start from a genuine failure.
"""
from __future__ import annotations

import numpy as np

from picfix.core.config import ExperimentConfig
from picfix.core.params import DesignParams
from picfix.core.task import Task
from picfix.judge.frozen_judge import FrozenJudge

_MAX_ATTEMPTS_PER_TASK = 100


def generate_tasks(
    cfg: ExperimentConfig, judge: FrozenJudge, count: int, rng: np.random.Generator
) -> list[Task]:
    golden = cfg.golden_params()
    gen = cfg.task_generation
    tasks: list[Task] = []
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * _MAX_ATTEMPTS_PER_TASK:
            raise RuntimeError("task generation failed to find failing initial designs")
        params = DesignParams(
            gap_nm=golden.gap_nm + _signed(rng, gen.gap_offset_nm),
            length_um=golden.length_um + _signed(rng, gen.length_offset_um),
            width_nm=golden.width_nm + _signed(rng, gen.width_offset_nm),
        )
        if judge.evaluate(params).passed:
            continue  # trivial: nothing to fix
        tasks.append(
            Task(
                task_id=f"task-{len(tasks):03d}",
                init_params=params,
                noise_seed=int(rng.integers(1, 2**31 - 1)),
            )
        )
    return tasks


def _signed(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    magnitude = float(rng.uniform(bounds[0], bounds[1]))
    return magnitude if rng.random() < 0.5 else -magnitude
