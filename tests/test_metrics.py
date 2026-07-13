"""M10 acceptance:每个指标在合成 fixture 上与手算对照。"""
from __future__ import annotations

from pathlib import Path

from picfix.core.nfo import NFO, DiagnosisLabels
from picfix.core.params import DesignParams
from picfix.core.task import TaskResult
from picfix.core.trace import SpectrumSummary, TraceRecord
from picfix.core.truth import RootCause
from picfix.metrics.compute import (
    arm_metrics,
    convergence_rounds,
    false_accepts,
    repeated_failure_rate,
    write_csv,
)
from picfix.simulators.analytical import AnalyticalBackend


def _result(
    idx: int,
    *,
    verifier: bool,
    judge: bool,
    causes: tuple[RootCause, ...] = (),
    r3: bool = False,
    repeat: int = 0,
    nfos: list[NFO] | None = None,
    params: DesignParams | None = None,
) -> TaskResult:
    p = params or DesignParams(gap_nm=200.0, length_um=15.7, width_nm=500.0)
    return TaskResult(
        task_id=f"t{idx}", arm="x", repeat=repeat, task_index=idx, init_params=p,
        final_params=p, sim_calls=5, wall_time_s=1.0 + idx, verifier_passed=verifier,
        judge_passed=judge, judge_failed_criteria=(), truth_causes=causes,
        tokens_in=100, tokens_out=50, r3_triggered=r3, proposal_events=[], nfos=nfos or [],
        sds_values=[0.4] if r3 else [], rollbacks=0, verifier_version=1,
        workflow_version=1, prior_version=0,
    )


def test_false_accept_counting() -> None:
    results = [
        _result(0, verifier=True, judge=True),    # true success
        _result(1, verifier=True, judge=False),   # false accept
        _result(2, verifier=False, judge=False),  # honest failure (not in denominator)
    ]
    assert false_accepts(results) == (1, 2)
    assert false_accepts([_result(0, verifier=False, judge=False)]) == (0, 0)


def test_repeated_failure_rate_uses_ground_truth() -> None:
    seq = [
        _result(0, verifier=False, judge=False, causes=(RootCause.COUPLING_LENGTH_MISMATCH,)),
        _result(1, verifier=False, judge=False, causes=(RootCause.COUPLING_LENGTH_MISMATCH,)),
        _result(2, verifier=True, judge=True),
        _result(3, verifier=False, judge=False, causes=(RootCause.GAP_TOO_SMALL_SCATTERING,)),
    ]
    # failures after the first: t1 (repeat of t0's cause), t3 (new cause) -> 1/2
    assert repeated_failure_rate(seq) == 0.5
    assert repeated_failure_rate([seq[0]]) is None


def test_convergence_rounds() -> None:
    seq = [
        _result(0, verifier=False, judge=False),
        _result(1, verifier=False, judge=False, r3=True),   # first trigger
        _result(2, verifier=False, judge=False, r3=True),   # second round
        _result(3, verifier=True, judge=True),              # streak starts here
        _result(4, verifier=True, judge=True),
        _result(5, verifier=True, judge=True),
    ]
    assert convergence_rounds(seq) == 2.0  # two proposal rounds before the streak
    assert convergence_rounds([_result(0, verifier=True, judge=True)]) is None


def test_diagnose_accuracy_against_truth(backend: AnalyticalBackend, cfg) -> None:
    golden = cfg.golden_params()
    bad = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 0.7,
                       width_nm=golden.width_nm)
    trace = TraceRecord(
        trace_id="tr1", task_id="t0", arm="x", call_index=0, params=bad,
        summary=SpectrumSummary(worst_ratio_dev=0.2, max_il_db=0.2, center_ratio=0.3),
        backend="analytical", seed=1, elapsed_s=0.0, agent_version=1,
        workflow_version=1, verifier_version=1, prior_version=0,
    )
    # truth for `bad` is exactly {coupling_length_mismatch}
    perfect = DiagnosisLabels(diagnoser="a", root_causes=("coupling_length_mismatch",),
                              affected_params=(), suggested_actions=())
    wrong = DiagnosisLabels(diagnoser="b", root_causes=("wavelength_drift",),
                            affected_params=(), suggested_actions=())
    nfo = NFO(nfo_id="n1", task_id="t0", trace_id="tr1", diagnoses=(perfect, wrong),
              sds=0.5, needs_human_review=False)
    results = [_result(0, verifier=False, judge=False, r3=True, nfos=[nfo], params=bad)]
    metrics = arm_metrics("x", results, [trace], backend)
    assert metrics.diagnose_accuracy == 0.5  # (1.0 + 0.0) / 2


def test_arm_metrics_aggregate_and_csv(tmp_path: Path, backend: AnalyticalBackend) -> None:
    results = [
        _result(0, verifier=True, judge=True),
        _result(1, verifier=True, judge=False),
        _result(2, verifier=False, judge=False, r3=True),
    ]
    m = arm_metrics("meta_unguarded", results, [], backend)
    assert m.success_rate == 1 / 3
    assert m.false_accept_rate == 0.5
    assert m.r3_triggers == 1
    assert m.sds_mean == 0.4
    assert m.tokens_per_success == 450.0  # 3 tasks x 150 tokens / 1 success
    assert m.sim_calls_per_success == 15.0

    out = tmp_path / "metrics.csv"
    write_csv([m], out)
    header, row = out.read_text().strip().splitlines()
    assert "false_accept_rate" in header and "sds_values" not in header
    assert row.startswith("meta_unguarded,3,")
