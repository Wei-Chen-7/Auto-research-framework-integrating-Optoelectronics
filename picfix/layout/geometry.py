"""gdsfactory geometry generation (optional dependency).

Install with ``uv sync --extra layout``. The experiment pipeline itself is
parameter-level and does not require GDS output; this exists so designs can
be exported for inspection and to keep the interface ready for real PDK flows.
"""
from __future__ import annotations

from typing import Any

from picfix.core.params import DesignParams


def build_coupler_component(params: DesignParams) -> Any:
    try:
        import gdsfactory as gf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gdsfactory is not installed; install the optional extra: uv sync --extra layout"
        ) from exc

    return gf.components.coupler(
        gap=params.gap_nm / 1000.0,
        length=params.length_um,
        cross_section=gf.cross_section.strip(width=params.width_nm / 1000.0),
    )
