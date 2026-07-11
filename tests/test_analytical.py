"""M2/M4 acceptance: analytical physics sanity + golden feasibility."""
from __future__ import annotations

import math

import pytest

from picfix.core.config import ExperimentConfig
from picfix.core.params import DesignParams
from picfix.core.truth import RootCause
from picfix.simulators.analytical import AnalyticalBackend
from picfix.simulators.meep_stub import MeepBackend


def test_kappa_decreases_with_gap(backend: AnalyticalBackend) -> None:
    gaps = [120.0, 160.0, 200.0, 260.0, 320.0]
    kappas = [backend.kappa_per_um(g, 500.0, 1547.5) for g in gaps]
    assert all(a > b for a, b in zip(kappas, kappas[1:]))


def test_exists_length_for_5050(backend: AnalyticalBackend) -> None:
    kappa = backend.kappa_per_um(200.0, 500.0, 1547.5)
    length = (math.pi / 4.0) / kappa
    p = DesignParams(gap_nm=200.0, length_um=length, width_nm=500.0)
    assert backend.cross_ratio(p, 1547.5) == pytest.approx(0.5, abs=1e-9)


def test_golden_passes_both_grids_all_corners(
    cfg: ExperimentConfig, backend: AnalyticalBackend
) -> None:
    golden = cfg.golden_params()
    spec = cfg.spec
    d = spec.corner_offset_nm
    corners = [(0.0, 0.0)] + [(sg * d, sw * d) for sg in (-1, 1) for sw in (-1, 1)]
    for grid in (spec.visible_grid(), spec.hidden_grid()):
        for dg, dw in corners:
            p = golden.perturbed(dg, dw)
            worst = max(abs(backend.cross_ratio(p, wl) - spec.split_target) for wl in grid)
            assert worst <= spec.split_tolerance, (dg, dw, worst)
            assert backend.insertion_loss_db(p) <= spec.il_max_db
    assert backend.ground_truth(golden).causes == frozenset()


def test_loss_tension_is_live(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    # shrinking the gap toward the DRC minimum must blow the IL budget
    assert backend.scattering_loss_db(cfg.drc.min_gap_nm) > cfg.spec.il_max_db * 0.5
    # growing L costs loss linearly
    assert backend.length_loss_db(40.0) > backend.length_loss_db(15.0)


def test_noise_is_seeded_and_reproducible(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    golden = cfg.golden_params()
    grid = cfg.spec.visible_grid()
    r1 = backend.simulate(golden, grid, noisy=True, seed=42)
    r2 = backend.simulate(golden, grid, noisy=True, seed=42)
    r3 = backend.simulate(golden, grid, noisy=True, seed=43)
    assert r1.spectrum == r2.spectrum
    assert r1.spectrum != r3.spectrum
    assert r1.spectrum.cross_ratio != backend.simulate(golden, grid, noisy=False, seed=None).spectrum.cross_ratio


def test_noiseless_is_deterministic(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    golden = cfg.golden_params()
    grid = cfg.spec.hidden_grid()
    a = backend.simulate(golden, grid, noisy=False, seed=None)
    b = backend.simulate(golden, grid, noisy=False, seed=None)
    assert a.spectrum == b.spectrum  # bit-for-bit


def test_ground_truth_labels(cfg: ExperimentConfig, backend: AnalyticalBackend) -> None:
    golden = cfg.golden_params()
    # wrong coupling length at center
    bad_len = DesignParams(gap_nm=golden.gap_nm, length_um=golden.length_um * 0.7, width_nm=golden.width_nm)
    assert RootCause.COUPLING_LENGTH_MISMATCH in backend.ground_truth(bad_len).causes
    # gap below DRC minimum: DRC violation + scattering loss
    tiny_gap = DesignParams(gap_nm=80.0, length_um=golden.length_um, width_nm=golden.width_nm)
    causes = backend.ground_truth(tiny_gap).causes
    assert RootCause.DRC_VIOLATION in causes
    assert RootCause.GAP_TOO_SMALL_SCATTERING in causes
    # absurd length: length-dominated insertion loss
    long_dev = DesignParams(gap_nm=golden.gap_nm, length_um=45.0, width_nm=golden.width_nm)
    assert RootCause.INSERTION_LOSS_EXCESS in backend.ground_truth(long_dev).causes
    # non-physical params
    negative = DesignParams(gap_nm=-5.0, length_um=10.0, width_nm=500.0)
    assert backend.ground_truth(negative).causes == frozenset({RootCause.PARAM_OUT_OF_BOUNDS})


def test_meep_is_a_stub() -> None:
    stub = MeepBackend()
    p = DesignParams(gap_nm=200.0, length_um=15.0, width_nm=500.0)
    with pytest.raises(NotImplementedError):
        stub.simulate(p, (1547.5,), noisy=False, seed=None)


def test_drc_known_cases(cfg: ExperimentConfig) -> None:
    from picfix.layout import drc

    good = DesignParams(gap_nm=200.0, length_um=15.7, width_nm=500.0)
    assert drc.check(good, cfg.drc) == []
    bad = DesignParams(gap_nm=90.0, length_um=0.5, width_nm=650.0)
    rules = {v.rule for v in drc.check(bad, cfg.drc)}
    assert rules == {"min_gap", "min_length", "max_width"}
