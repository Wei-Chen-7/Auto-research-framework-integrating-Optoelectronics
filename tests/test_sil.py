"""M8 acceptance: SDS known cases, rule classifier vs ground truth,
LLM diagnoser schema compliance via the mock client."""
from __future__ import annotations

from picfix.agents.llm import MockLLM
from picfix.core.config import ExperimentConfig
from picfix.core.params import DesignParams
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.truth import RootCause
from picfix.core.verifier import verify
from picfix.simulators.analytical import AnalyticalBackend
from picfix.sil.evidence import FailureEvidence
from picfix.sil.interface import SemanticInterfaceLayer
from picfix.sil.rule_classifier import classify
from picfix.sil.sds import jaccard, sds


def make_evidence(
    cfg: ExperimentConfig, backend: AnalyticalBackend, params: DesignParams
) -> tuple[FailureEvidence, WorkingVerifierConfig]:
    verifier_cfg = WorkingVerifierConfig.from_spec(cfg.spec)
    result = backend.simulate(params, verifier_cfg.grid(), noisy=True, seed=11)
    report = verify(result, verifier_cfg, cfg.drc)
    ratios = result.spectrum.cross_ratio
    center = ratios[len(ratios) // 2]
    edge_dev = max(
        abs(ratios[0] - verifier_cfg.split_target), abs(ratios[-1] - verifier_cfg.split_target)
    )
    from picfix.layout import drc as drc_mod

    evidence = FailureEvidence(
        trace_id="t-test",
        task_id="task-test",
        params=params,
        center_ratio=center,
        edge_ratio_dev=edge_dev,
        worst_ratio_dev=result.spectrum.worst_ratio_dev(verifier_cfg.split_target),
        max_il_db=result.spectrum.max_il_db(),
        report=report,
        drc_rule_names=tuple(v.rule for v in drc_mod.check(params, cfg.drc)),
    )
    return evidence, verifier_cfg


def test_sds_known_cases() -> None:
    a = frozenset({"cause:x", "param:p", "action:z"})
    assert sds([a, a]) == 0.0
    assert sds([a, frozenset({"cause:y", "param:q"})]) == 1.0
    half = frozenset({"cause:x", "param:p", "action:w", "param:q"})
    expected = 1.0 - jaccard(a, half)
    assert abs(sds([a, half]) - expected) < 1e-12
    assert sds([frozenset(), frozenset()]) == 0.0  # two empty sets agree


def test_rule_classifier_matches_truth(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    golden = cfg.golden_params()
    cases = [
        DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 0.7, width_nm=golden.width_nm),
        DesignParams(gap_nm=80.0, length_um=golden.length_um, width_nm=golden.width_nm),
        DesignParams(gap_nm=golden.gap_nm, length_um=45.0, width_nm=golden.width_nm),
    ]
    hits = 0
    for params in cases:
        evidence, verifier_cfg = make_evidence(cfg, backend, params)
        labels = classify(evidence, verifier_cfg)
        truth = {c.value for c in backend.ground_truth(params).causes}
        if set(labels.root_causes) & truth:
            hits += 1
    assert hits == len(cases)  # at least one correct cause per synthetic failure


def test_sil_produces_nfo_with_mock_llm(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    golden = cfg.golden_params()
    bad = DesignParams(gap_nm=120.0, length_um=golden.length_um * 0.8, width_nm=golden.width_nm)
    evidence, verifier_cfg = make_evidence(cfg, backend, bad)
    sil = SemanticInterfaceLayer(MockLLM(cfg.llm), sds_threshold=cfg.r3.sds_threshold)
    nfo, response = sil.diagnose(evidence, verifier_cfg)

    assert len(nfo.diagnoses) == 2
    assert {d.diagnoser for d in nfo.diagnoses} == {"rule_classifier", "llm_diagnoser"}
    assert 0.0 <= nfo.sds <= 1.0
    assert response.tokens_in > 0 and response.tokens_out > 0
    llm_labels = next(d for d in nfo.diagnoses if d.diagnoser == "llm_diagnoser")
    allowed = {c.value for c in RootCause}
    assert set(llm_labels.root_causes) <= allowed
    assert set(llm_labels.affected_params) <= {"gap_nm", "length_um", "width_nm"}


def test_mock_diagnosers_diverge_partially(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    """The two diagnosers are heterogeneous by design: over varied failures
    SDS must not saturate at 0 or 1."""
    golden = cfg.golden_params()
    sil = SemanticInterfaceLayer(MockLLM(cfg.llm), sds_threshold=cfg.r3.sds_threshold)
    scores = []
    for params in [
        DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 0.75, width_nm=golden.width_nm),
        DesignParams(gap_nm=90.0, length_um=golden.length_um, width_nm=golden.width_nm),
        DesignParams(gap_nm=130.0, length_um=golden.length_um * 1.2, width_nm=golden.width_nm),
    ]:
        evidence, verifier_cfg = make_evidence(cfg, backend, params)
        nfo, _ = sil.diagnose(evidence, verifier_cfg)
        scores.append(nfo.sds)
    assert any(s > 0.0 for s in scores)
    assert all(s < 1.0 for s in scores)
