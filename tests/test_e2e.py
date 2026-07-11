"""M11 acceptance: four arms end-to-end in mock mode, all artifacts produced."""
from __future__ import annotations

import csv
import json

from picfix.experiments.run import ARM_NAMES, run_experiment
from tests.conftest import CONFIG_PATH


def test_four_arms_end_to_end_mock() -> None:
    out_dir = run_experiment(CONFIG_PATH, ARM_NAMES, llm_mode="mock", tasks_override=3)

    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "comparison.png").exists()
    assert (out_dir / "config_snapshot.yaml").exists()

    with (out_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["arm"] for r in rows] == list(ARM_NAMES)
    for row in rows:
        assert int(row["tasks"]) == 3
        assert 0.0 <= float(row["success_rate"]) <= 1.0

    results = [
        json.loads(line)
        for line in (out_dir / "task_results.jsonl").read_text().splitlines()
    ]
    assert len(results) == 3 * len(ARM_NAMES)
    # every arm respected the hard budget
    assert all(r["sim_calls"] <= 15 for r in results)
    # traces were appended for every arm
    for arm in ARM_NAMES:
        assert (out_dir / f"trace_{arm}.jsonl").stat().st_size > 0
