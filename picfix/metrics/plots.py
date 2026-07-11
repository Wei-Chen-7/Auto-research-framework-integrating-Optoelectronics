"""Four-arm comparison figure (light theme PNG).

Colors follow the arm identity in fixed categorical order (validated:
worst adjacent CVD ΔE 24.2, all slots in band; the two sub-3:1 slots are
relieved by direct value labels on every bar).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from picfix.metrics.compute import ArmMetrics

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
# fixed categorical order — color follows the arm, never its rank
_ARM_COLORS = {
    "baseline": "#2a78d6",
    "fixed_loop": "#1baf7a",
    "meta_unguarded": "#eda100",
    "meta_governed": "#008300",
}
_ARM_LABELS = {
    "baseline": "Baseline",
    "fixed_loop": "FixedLoop\n(R2)",
    "meta_unguarded": "MetaLoop\n(R3⁻)",
    "meta_governed": "MetaLoop\n(RAE)",
}

_PANELS: list[tuple[str, str, str]] = [
    ("success_rate", "Success Rate (frozen judge)", "percent"),
    ("false_accept_rate", "False Accept Rate", "percent"),
    ("time_to_fix_median_s", "Time to Fix — median (s)", "number"),
    ("sim_calls_per_success", "Sim Calls per Success", "number"),
    ("sds_mean", "SDS (mean)", "number"),
    ("proposals_deployed", "R3 Proposals Deployed", "int"),
]


def _fmt(value: float, kind: str) -> str:
    if kind == "percent":
        return f"{value * 100:.0f}%"
    if kind == "int":
        return f"{value:.0f}"
    return f"{value:.3g}"


def plot_comparison(metrics: list[ArmMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), facecolor=_SURFACE)
    fig.suptitle(
        "PIC-fix: four-arm comparison (directional coupler, analytical backend)",
        color=_INK,
        fontsize=13,
        fontweight="bold",
    )

    for ax, (field, title, kind) in zip(axes.flat, _PANELS):
        ax.set_facecolor(_SURFACE)
        arms = [m.arm for m in metrics]
        values = [getattr(m, field) for m in metrics]
        xs = range(len(arms))
        for x, arm, value in zip(xs, arms, values):
            color = _ARM_COLORS.get(arm, _MUTED)
            if value is None:
                ax.text(x, 0, "n/a", ha="center", va="bottom", color=_MUTED, fontsize=9)
                continue
            ax.bar(x, value, width=0.55, color=color, zorder=3)
            ax.text(
                x,
                value,
                " " + _fmt(float(value), kind),
                ha="center",
                va="bottom",
                color=_INK_SECONDARY,
                fontsize=9,
            )
        ax.set_title(title, color=_INK, fontsize=10, pad=10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([_ARM_LABELS.get(a, a) for a in arms], color=_MUTED, fontsize=8.5)
        ax.tick_params(axis="y", colors=_MUTED, labelsize=8)
        ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_BASELINE)
        if kind == "percent":
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        else:
            ax.margins(y=0.18)
            ax.set_ylim(bottom=0)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
