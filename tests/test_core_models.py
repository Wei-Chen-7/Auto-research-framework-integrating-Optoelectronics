"""M1 acceptance: model round-trips, strict schemas, config loading."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from picfix.core.config import load_config
from picfix.core.params import DesignParams
from picfix.core.proposal import Proposal, ProposalKind
from picfix.core.spec import TaskSpec, WorkingVerifierConfig, wavelength_grid

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "coupler_v1.yaml"


def test_config_loads_and_is_typed() -> None:
    cfg = load_config(CONFIG)
    assert cfg.spec.band_nm == (1530.0, 1565.0)
    assert cfg.experiment.budget_sim_calls == 15
    assert set(cfg.experiment.arms) == {"baseline", "fixed_loop", "meta_unguarded", "meta_governed"}


def test_uncalibrated_config_refuses_constants() -> None:
    cfg = load_config(CONFIG)
    if not cfg.calibration.calibrated:
        with pytest.raises(RuntimeError, match="calibrate"):
            cfg.frozen_constants()


def test_proposal_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Proposal(
            proposal_id="p1",
            kind=ProposalKind.APPEND_PRIOR,
            payload={},
            rationale="",
            source_trace_ids=(),
            hot_patch="import os",  # type: ignore[call-arg]
        )


def test_params_roundtrip_and_frozen() -> None:
    p = DesignParams(gap_nm=200.0, length_um=15.7, width_nm=500.0)
    assert DesignParams.model_validate_json(p.model_dump_json()) == p
    with pytest.raises(ValidationError):
        p.gap_nm = 100.0  # type: ignore[misc]


def test_working_verifier_starts_as_faithful_copy() -> None:
    cfg = load_config(CONFIG)
    wv = WorkingVerifierConfig.from_spec(cfg.spec)
    assert wv.split_tolerance == cfg.spec.split_tolerance
    assert wv.grid() == cfg.spec.visible_grid()
    assert wv.case_list == ["ratio", "il", "drc"]


def test_wavelength_grid_endpoints() -> None:
    g = wavelength_grid((1530.0, 1565.0), 15)
    assert len(g) == 15 and g[0] == 1530.0 and g[-1] == 1565.0
    g61 = wavelength_grid((1530.0, 1565.0), 61)
    assert len(g61) == 61


def test_spec_is_frozen() -> None:
    cfg = load_config(CONFIG)
    with pytest.raises(ValidationError):
        cfg.spec.split_tolerance = 0.5  # type: ignore[misc]
