"""Experiment configuration: YAML → typed models."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from picfix.core.constants import AnalyticalConstants
from picfix.core.params import DesignParams
from picfix.core.spec import TaskSpec


class DRCRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_gap_nm: float
    min_width_nm: float
    max_width_nm: float
    min_length_um: float
    max_length_um: float


class NoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sigma_ratio: float
    sigma_il_db: float


class TaskGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_offset_nm: tuple[float, float]
    length_offset_um: tuple[float, float]
    width_offset_nm: tuple[float, float]


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "api"]
    provider: Literal["anthropic", "openai_compatible"] = "anthropic"
    base_url: str | None = None    # e.g. https://api.deepseek.com for DeepSeek
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str
    temperature: float
    max_tokens: int
    seed: int


class R3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    consecutive_task_failures: int
    sds_threshold: float
    proposals_per_trigger: int


class SearchStrategy(BaseModel):
    """Evolvable search strategy (versioning handled by the meta arms)."""

    model_config = ConfigDict(extra="forbid")

    max_step_gap_nm: float
    max_step_length_um: float
    max_step_width_nm: float
    early_stop_margin: float


class ExperimentSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    seed: int
    tasks_per_arm: int
    repeats: int
    budget_sim_calls: int
    arms: tuple[str, ...]
    output_dir: str


class CalibrationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibrated: bool
    frozen_constants: dict[str, float]


class GoldenSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: dict[str, float]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentSection
    spec: TaskSpec
    drc: DRCRulesConfig
    noise: NoiseConfig
    task_generation: TaskGenerationConfig
    llm: LLMConfig
    r3: R3Config
    search_strategy: SearchStrategy
    calibration: CalibrationSection
    golden: GoldenSection

    def frozen_constants(self) -> AnalyticalConstants:
        if not self.calibration.calibrated:
            raise RuntimeError(
                "config is not calibrated — run scripts/calibrate.py first "
                "(refusing to run with unfrozen analytical constants)"
            )
        return AnalyticalConstants.model_validate(self.calibration.frozen_constants)

    def golden_params(self) -> DesignParams:
        if not self.golden.params:
            raise RuntimeError("golden params missing — run scripts/calibrate.py first")
        return DesignParams.model_validate(self.golden.params)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
