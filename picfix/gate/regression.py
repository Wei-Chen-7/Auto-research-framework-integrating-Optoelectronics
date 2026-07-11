"""Golden-set automatic regression for low-risk proposals.

Deterministic and LLM-free: from each canned regression start, run the
platform's reference repair routine under the CANDIDATE search strategy
(noiseless simulation) and require every start to end judge-green within
a fixed iteration budget. A strategy change that cripples the search
(e.g. absurd step limits) fails here and the proposal is rejected.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from picfix.core.config import DRCRulesConfig, SearchStrategy
from picfix.core.params import DesignParams
from picfix.judge.frozen_judge import FrozenJudge
from picfix.simulators.analytical import AnalyticalBackend

MAX_REPAIR_ITERS = 12
_CLAMP_MARGIN_NM = 10.0
_CLAMP_MARGIN_UM = 0.5
_TARGET_KL = math.pi / 4.0


def load_regression_starts(golden_path: Path) -> list[DesignParams]:
    with golden_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    return [DesignParams.model_validate(s) for s in doc["regression_starts"]]


def _step(value: float, target: float, max_step: float) -> float:
    delta = target - value
    return value + math.copysign(min(abs(delta), max_step), delta)


def repair_step(
    params: DesignParams,
    strategy: SearchStrategy,
    backend: AnalyticalBackend,
    drc_rules: DRCRulesConfig,
    center_nm: float,
) -> DesignParams:
    """One deterministic repair move: clamp into DRC, then invert L for a
    50:50 split at band center, then relieve scattering loss via the gap."""
    gap, length, width = params.gap_nm, params.length_um, params.width_nm

    gap = max(gap, drc_rules.min_gap_nm + _CLAMP_MARGIN_NM)
    width = min(max(width, drc_rules.min_width_nm + _CLAMP_MARGIN_NM), drc_rules.max_width_nm - _CLAMP_MARGIN_NM)
    length = min(max(length, drc_rules.min_length_um + _CLAMP_MARGIN_UM), drc_rules.max_length_um - _CLAMP_MARGIN_UM)

    if backend.scattering_loss_db(gap) > 0.1:
        gap = _step(gap, gap + strategy.max_step_gap_nm, strategy.max_step_gap_nm)

    ratio = backend.cross_ratio(
        DesignParams(gap_nm=gap, length_um=length, width_nm=width), center_nm
    )
    ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
    kl_est = math.asin(math.sqrt(ratio))
    target_length = length * _TARGET_KL / kl_est
    length = _step(length, target_length, strategy.max_step_length_um)

    return DesignParams(gap_nm=gap, length_um=length, width_nm=width)


def golden_regression_passes(
    strategy: SearchStrategy,
    starts: list[DesignParams],
    backend: AnalyticalBackend,
    judge: FrozenJudge,
    drc_rules: DRCRulesConfig,
    center_nm: float,
) -> bool:
    for start in starts:
        params = start
        for _ in range(MAX_REPAIR_ITERS):
            if judge.evaluate(params).passed:
                break
            params = repair_step(params, strategy, backend, drc_rules, center_nm)
        if not judge.evaluate(params).passed:
            return False
    return True
