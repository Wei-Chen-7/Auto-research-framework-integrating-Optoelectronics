"""Frozen Judge — the platform's single source of truth for success.

Holds the read-only task spec and the hidden test suite. All success
judgements and metrics derive from here. Agents in EVERY arm sit outside
its permission boundary (DESIGN.md §4, §9): the judge is never handed to
agent code, exposes no mutating API, and evaluates deterministically
(noiseless), which also defeats the "re-roll the noise" cheating channel.

Hidden suite = process-corner re-evaluation (gap, width ±corner_offset_nm)
+ densified 61-point wavelength grid + deterministic noise-free re-check.
"""
from __future__ import annotations

from picfix.core.config import DRCRulesConfig
from picfix.core.params import DesignParams
from picfix.core.spec import TaskSpec
from picfix.core.task import JudgeVerdict
from picfix.layout import drc
from picfix.simulators.analytical import AnalyticalBackend


class FrozenJudge:
    def __init__(
        self, spec: TaskSpec, drc_rules: DRCRulesConfig, backend: AnalyticalBackend
    ) -> None:
        self._spec = spec
        self._drc = drc_rules
        self._backend = backend

    def evaluate(self, params: DesignParams) -> JudgeVerdict:
        spec = self._spec
        failed: list[str] = []

        drc_ok = not drc.check(params, self._drc)
        if not drc_ok:
            failed.append("drc")

        physical = params.gap_nm > 0 and params.length_um > 0 and params.width_nm > 0
        if not physical:
            failed.append("non_physical_params")
            return JudgeVerdict(
                passed=False,
                visible_grid_ok=False,
                hidden_grid_ok=False,
                corners_ok=False,
                drc_ok=drc_ok,
                worst_ratio_dev=1.0,
                max_il_db=float("inf"),
                failed_criteria=tuple(failed),
            )

        visible_ok, dev_v, il_v = self._grid_ok(params, spec.visible_grid())
        hidden_ok, dev_h, il_h = self._grid_ok(params, spec.hidden_grid())
        if not visible_ok:
            failed.append("visible_grid")
        if not hidden_ok:
            failed.append("hidden_grid")

        d = spec.corner_offset_nm
        corners_ok = True
        dev_c, il_c = 0.0, 0.0
        for sg in (-1, 1):
            for sw in (-1, 1):
                ok, dev, il = self._grid_ok(params.perturbed(sg * d, sw * d), spec.hidden_grid())
                corners_ok = corners_ok and ok
                dev_c, il_c = max(dev_c, dev), max(il_c, il)
        if not corners_ok:
            failed.append("corners")

        return JudgeVerdict(
            passed=not failed,
            visible_grid_ok=visible_ok,
            hidden_grid_ok=hidden_ok,
            corners_ok=corners_ok,
            drc_ok=drc_ok,
            worst_ratio_dev=max(dev_v, dev_h, dev_c),
            max_il_db=max(il_v, il_h, il_c),
            failed_criteria=tuple(failed),
        )

    def _grid_ok(
        self, params: DesignParams, grid: tuple[float, ...]
    ) -> tuple[bool, float, float]:
        result = self._backend.simulate(params, grid, noisy=False, seed=None)
        worst_dev = result.spectrum.worst_ratio_dev(self._spec.split_target)
        max_il = result.spectrum.max_il_db()
        ok = worst_dev <= self._spec.split_tolerance and max_il <= self._spec.il_max_db
        return ok, worst_dev, max_il
