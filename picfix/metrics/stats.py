"""Significance tests for the primary endpoints (DESIGN §11).

Two-proportion z-test for a quick pooled comparison, plus a task-clustered
paired bootstrap that respects the design's dependence structure: within
one repeat all four arms run the SAME task sequence (paired), and repeats
use different seeds. The independent resampling unit is therefore the task
instance identified by (repeat, task_index); resampling a cluster carries
every arm's outcome on that task together, so the arm difference is paired.

n=60/arm (20 tasks x k=3) is only powered for large effects — report the
CI, don't over-claim.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from picfix.core.task import TaskResult


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class ZTestResult:
    p_a: float
    p_b: float
    diff: float
    z: float
    p_value: float  # two-sided

    def as_dict(self) -> dict[str, float]:
        return {"p_a": self.p_a, "p_b": self.p_b, "diff": self.diff, "z": self.z, "p_value": self.p_value}


def two_proportion_ztest(x_a: int, n_a: int, x_b: int, n_b: int) -> ZTestResult:
    """Pooled two-sided two-proportion z-test for successes x out of n."""
    if n_a == 0 or n_b == 0:
        return ZTestResult(math.nan, math.nan, math.nan, math.nan, math.nan)
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        z = 0.0 if p_a == p_b else math.copysign(math.inf, p_a - p_b)
    else:
        z = (p_a - p_b) / se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z))) if math.isfinite(z) else 0.0
    return ZTestResult(p_a=p_a, p_b=p_b, diff=p_a - p_b, z=z, p_value=p_value)


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lo: float
    hi: float
    level: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_lo": self.lo, "ci_hi": self.hi, "level": self.level}


# a cluster holds, per arm, the counts this task instance contributed
# (num, den) so both success rate (den=1 per task) and FAR (den=visible pass)
# use one code path
_Cluster = dict[str, tuple[int, int]]


def _success_cluster(results: list[TaskResult]) -> dict[tuple[int, int], _Cluster]:
    clusters: dict[tuple[int, int], _Cluster] = defaultdict(dict)
    for r in results:
        clusters[(r.repeat, r.task_index)][r.arm] = (int(r.judge_passed), 1)
    return clusters


def _far_cluster(results: list[TaskResult]) -> dict[tuple[int, int], _Cluster]:
    clusters: dict[tuple[int, int], _Cluster] = defaultdict(dict)
    for r in results:
        den = int(r.verifier_passed)                       # in the FAR denominator?
        num = int(r.verifier_passed and not r.judge_passed)  # false accept
        clusters[(r.repeat, r.task_index)][r.arm] = (num, den)
    return clusters


def _rate(clusters: list[_Cluster], arm: str) -> float:
    num = sum(c[arm][0] for c in clusters if arm in c)
    den = sum(c[arm][1] for c in clusters if arm in c)
    return num / den if den else math.nan


def clustered_bootstrap_diff(
    clusters: dict[tuple[int, int], _Cluster],
    arm_a: str,
    arm_b: str,
    *,
    iterations: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Paired task-clustered bootstrap CI for rate(arm_a) - rate(arm_b)."""
    keys = list(clusters)
    cluster_list = [clusters[k] for k in keys]
    point = _rate(cluster_list, arm_a) - _rate(cluster_list, arm_b)

    rng = np.random.default_rng(seed)
    n = len(keys)
    diffs = np.empty(iterations)
    idx_all = np.arange(n)
    for i in range(iterations):
        sample = [cluster_list[j] for j in rng.choice(idx_all, size=n, replace=True)]
        diffs[i] = _rate(sample, arm_a) - _rate(sample, arm_b)
    diffs = diffs[np.isfinite(diffs)]
    alpha = 1.0 - level
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), level=level)


def endpoint_totals(results: list[TaskResult], arm: str) -> dict[str, int]:
    ar = [r for r in results if r.arm == arm]
    visible = [r for r in ar if r.verifier_passed]
    return {
        "n": len(ar),
        "successes": sum(r.judge_passed for r in ar),
        "visible_passes": len(visible),
        "false_accepts": sum(1 for r in visible if not r.judge_passed),
    }
