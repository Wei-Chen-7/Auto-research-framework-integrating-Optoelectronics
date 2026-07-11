from __future__ import annotations

from pathlib import Path

import pytest

from picfix.core.config import ExperimentConfig, load_config
from picfix.simulators.analytical import AnalyticalBackend

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "coupler_v1.yaml"
GOLDEN_PATH = REPO / "golden" / "coupler_golden.json"


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    config = load_config(CONFIG_PATH)
    if not config.calibration.calibrated:
        pytest.fail("config not calibrated — run: uv run python scripts/calibrate.py")
    return config


@pytest.fixture(scope="session")
def backend(cfg: ExperimentConfig) -> AnalyticalBackend:
    return AnalyticalBackend(
        cfg.frozen_constants(),
        cfg.spec,
        cfg.drc,
        sigma_ratio=cfg.noise.sigma_ratio,
        sigma_il_db=cfg.noise.sigma_il_db,
    )
