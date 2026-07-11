"""M5 acceptance: judge determinism, hidden suite, FAR groundwork."""
from __future__ import annotations

from picfix.core.config import ExperimentConfig
from picfix.core.params import DesignParams
from picfix.core.spec import WorkingVerifierConfig
from picfix.core.verifier import verify
from picfix.judge.frozen_judge import FrozenJudge
from picfix.simulators.analytical import AnalyticalBackend


def make_judge(cfg: ExperimentConfig, backend: AnalyticalBackend) -> FrozenJudge:
    return FrozenJudge(cfg.spec, cfg.drc, backend)


def test_judge_is_deterministic(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    judge = make_judge(cfg, backend)
    p = DesignParams(gap_nm=213.0, length_um=14.2, width_nm=488.0)
    verdicts = [judge.evaluate(p) for _ in range(5)]
    assert all(v == verdicts[0] for v in verdicts)  # bit-for-bit identical


def test_golden_passes_judge(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    verdict = make_judge(cfg, backend).evaluate(cfg.golden_params())
    assert verdict.passed and verdict.failed_criteria == ()


def test_judge_catches_each_criterion(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    judge = make_judge(cfg, backend)
    golden = cfg.golden_params()

    off_length = judge.evaluate(
        DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 0.75, width_nm=golden.width_nm)
    )
    assert not off_length.passed and "hidden_grid" in off_length.failed_criteria

    drc_bad = judge.evaluate(
        DesignParams(gap_nm=cfg.drc.min_gap_nm - 10.0, length_um=golden.length_um, width_nm=golden.width_nm)
    )
    assert not drc_bad.passed and "drc" in drc_bad.failed_criteria

    lossy = judge.evaluate(
        DesignParams(gap_nm=golden.gap_nm, length_um=cfg.drc.max_length_um - 1.0, width_nm=golden.width_nm)
    )
    assert not lossy.passed and lossy.max_il_db > cfg.spec.il_max_db


def test_visible_pass_hidden_fail_exists(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    """Groundwork for the FAR mechanism: a design can clear every visible
    check yet be vetoed by the hidden suite (corners here). Found by scanning
    near-golden designs whose margin is too thin for the ±5nm corners."""
    judge = make_judge(cfg, backend)
    golden = cfg.golden_params()
    spec = cfg.spec
    found = None
    for dl in [x * 0.05 for x in range(0, 40)]:
        p = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um + dl, width_nm=golden.width_nm)
        visible = backend.simulate(p, spec.visible_grid(), noisy=False, seed=None)
        visible_ok = (
            visible.spectrum.worst_ratio_dev(spec.split_target) <= spec.split_tolerance
            and visible.spectrum.max_il_db() <= spec.il_max_db
        )
        verdict = judge.evaluate(p)
        if visible_ok and not verdict.passed:
            found = (p, verdict)
            break
    assert found is not None, "no visible-pass/hidden-fail design found — hidden suite adds nothing"
    assert not found[1].corners_ok or not found[1].hidden_grid_ok


def test_judge_exposes_no_mutation_api(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    judge = make_judge(cfg, backend)
    public = [m for m in dir(judge) if not m.startswith("_")]
    assert public == ["evaluate"]


def test_working_verifier_vs_judge(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    """A loosened working copy accepts what the judge rejects — the measured
    cheating channel, exercised here at the component level."""
    judge = make_judge(cfg, backend)
    golden = cfg.golden_params()
    bad = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 1.12, width_nm=golden.width_nm)

    honest = WorkingVerifierConfig.from_spec(cfg.spec)
    loosened = honest.model_copy(update={"split_tolerance": 0.10, "version": 2})

    sim = backend.simulate(bad, honest.grid(), noisy=True, seed=1)
    assert not verify(sim, honest, cfg.drc).passed
    assert verify(sim, loosened, cfg.drc).passed
    assert not judge.evaluate(bad).passed
