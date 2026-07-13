"""Significance-test unit checks: z-test known values, paired bootstrap."""
from __future__ import annotations

import math

from picfix.core.params import DesignParams
from picfix.core.task import TaskResult
from picfix.metrics.stats import (
    _far_cluster,
    _success_cluster,
    clustered_bootstrap_diff,
    two_proportion_ztest,
)


def _r(arm: str, repeat: int, idx: int, *, verifier: bool, judge: bool) -> TaskResult:
    p = DesignParams(gap_nm=200.0, length_um=15.7, width_nm=500.0)
    return TaskResult(
        task_id=f"t{idx}", arm=arm, repeat=repeat, task_index=idx, init_params=p,
        final_params=p, sim_calls=3, wall_time_s=0.1, verifier_passed=verifier,
        judge_passed=judge, judge_failed_criteria=(), truth_causes=(), tokens_in=0,
        tokens_out=0, r3_triggered=False, proposal_events=[], nfos=[], sds_values=[],
        rollbacks=0, verifier_version=1, workflow_version=1, prior_version=0,
    )


def test_ztest_known_value() -> None:
    # 40/100 vs 60/100: pooled p=0.5, se=sqrt(0.25*0.02)=0.070710..., z=-2.828
    res = two_proportion_ztest(40, 100, 60, 100)
    assert abs(res.diff + 0.2) < 1e-9
    assert abs(res.z + 2.8284271) < 1e-4
    assert abs(res.p_value - 0.004678) < 1e-4  # two-sided


def test_ztest_no_difference_is_insignificant() -> None:
    res = two_proportion_ztest(30, 60, 30, 60)
    assert res.z == 0.0
    assert res.p_value == 1.0


def test_ztest_handles_empty() -> None:
    assert math.isnan(two_proportion_ztest(0, 0, 5, 10).z)


def test_bootstrap_ci_brackets_point_and_paired() -> None:
    # arm A fails judge on every visible-pass task, arm B never does -> FAR
    # diff should be strongly positive with a CI clear of zero
    results = []
    for rep in range(3):
        for idx in range(20):
            results.append(_r("A", rep, idx, verifier=True, judge=False))  # all FA
            results.append(_r("B", rep, idx, verifier=True, judge=True))   # no FA
    clusters = _far_cluster(results)
    ci = clustered_bootstrap_diff(clusters, "A", "B", iterations=2000, seed=1)
    assert ci.point == 1.0            # FAR_A=1.0, FAR_B=0.0
    assert ci.lo > 0.5 and ci.hi <= 1.0


def test_bootstrap_ci_straddles_zero_for_identical_arms() -> None:
    results = []
    for rep in range(3):
        for idx in range(20):
            j = (idx % 2 == 0)
            results.append(_r("A", rep, idx, verifier=True, judge=j))
            results.append(_r("B", rep, idx, verifier=True, judge=j))
    clusters = _success_cluster(results)
    ci = clustered_bootstrap_diff(clusters, "A", "B", iterations=2000, seed=2)
    assert ci.point == 0.0
    assert ci.lo <= 0.0 <= ci.hi     # identical arms -> CI contains 0


def test_success_cluster_pairs_by_task() -> None:
    results = [
        _r("A", 0, 0, verifier=True, judge=True),
        _r("B", 0, 0, verifier=True, judge=False),
    ]
    clusters = _success_cluster(results)
    assert set(clusters[(0, 0)]) == {"A", "B"}  # same task carries both arms
