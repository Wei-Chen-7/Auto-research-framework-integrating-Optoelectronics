"""Significance report for a completed run's primary endpoints.

    python -m picfix.metrics.report_stats runs/coupler_v1/<stamp>

Writes stats.json + stats.md into the run directory: two-proportion z-test
and task-clustered paired bootstrap 95% CI for the sharpest contrasts
(R3⁻ vs RAE, and RAE vs R2) on Success Rate and False Accept Rate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from picfix.core.task import TaskResult
from picfix.metrics.stats import (
    BootstrapCI,
    ZTestResult,
    _far_cluster,
    _success_cluster,
    clustered_bootstrap_diff,
    endpoint_totals,
    two_proportion_ztest,
)

_CONTRASTS = [
    ("meta_unguarded", "meta_governed", "R3⁻ vs RAE — the sharpest test (DESIGN §12)"),
    ("meta_governed", "fixed_loop", "RAE vs R2 — does governance cost capability?"),
    ("meta_unguarded", "fixed_loop", "R3⁻ vs R2 — is unguarded self-mod harmful?"),
]


def _load(run_dir: Path) -> list[TaskResult]:
    path = run_dir / "task_results.jsonl"
    return [TaskResult.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _success_contrast(results: list[TaskResult], a: str, b: str) -> tuple[ZTestResult, BootstrapCI]:
    ta, tb = endpoint_totals(results, a), endpoint_totals(results, b)
    z = two_proportion_ztest(ta["successes"], ta["n"], tb["successes"], tb["n"])
    ci = clustered_bootstrap_diff(_success_cluster(results), a, b)
    return z, ci


def _far_contrast(results: list[TaskResult], a: str, b: str) -> tuple[ZTestResult, BootstrapCI]:
    ta, tb = endpoint_totals(results, a), endpoint_totals(results, b)
    z = two_proportion_ztest(
        ta["false_accepts"], ta["visible_passes"], tb["false_accepts"], tb["visible_passes"]
    )
    ci = clustered_bootstrap_diff(_far_cluster(results), a, b)
    return z, ci


def build_report(run_dir: Path) -> dict:
    results = _load(run_dir)
    report: dict = {"run": run_dir.name, "contrasts": []}
    for a, b, label in _CONTRASTS:
        sz, sci = _success_contrast(results, a, b)
        fz, fci = _far_contrast(results, a, b)
        report["contrasts"].append(
            {
                "arm_a": a, "arm_b": b, "label": label,
                "success": {"ztest": sz.as_dict(), "bootstrap_ci": sci.as_dict()},
                "false_accept": {"ztest": fz.as_dict(), "bootstrap_ci": fci.as_dict()},
            }
        )
    return report


def _fmt_ci(ci: dict) -> str:
    return f"{ci['point']:+.3f} [95% CI {ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]"


def render_markdown(report: dict) -> str:
    lines = [
        f"# 统计检验 — {report['run']}",
        "",
        "主要终点两比例 z 检验（pooled，双侧）+ 按任务聚类的配对 bootstrap 95% CI",
        "（10000 次重采样；聚类单元 = (repeat, task_index)，同一 repeat 内四臂共用",
        "同一任务序列，故为配对）。n=60/臂仅足检出大效应，CI 是关键、p 值作参考。",
        "",
    ]
    for c in report["contrasts"]:
        lines.append(f"## {c['label']}")
        lines.append(f"（{c['arm_a']} − {c['arm_b']}）")
        lines.append("")
        s, f = c["success"], c["false_accept"]
        lines.append("| 终点 | 差值 [95% CI] | z | p |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| Success Rate | {_fmt_ci(s['bootstrap_ci'])} | "
            f"{s['ztest']['z']:.2f} | {s['ztest']['p_value']:.4f} |"
        )
        lines.append(
            f"| False Accept Rate | {_fmt_ci(f['bootstrap_ci'])} | "
            f"{f['ztest']['z']:.2f} | {f['ztest']['p_value']:.4f} |"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m picfix.metrics.report_stats <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    report = build_report(run_dir)
    (run_dir / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    md = render_markdown(report)
    (run_dir / "stats.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
