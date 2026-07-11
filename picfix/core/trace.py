"""Append-only trace log (JSONL).

Every simulator call is recorded. Agents receive only the ``append``
capability; there is deliberately no update or delete interface, and the
file is opened in append mode on every write.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from picfix.core.params import DesignParams


class SpectrumSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worst_ratio_dev: float
    max_il_db: float
    center_ratio: float


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    task_id: str
    arm: str
    call_index: int
    params: DesignParams
    summary: SpectrumSummary
    backend: str
    seed: int | None
    elapsed_s: float
    agent_version: int
    workflow_version: int
    verifier_version: int
    prior_version: int


class TraceWriter:
    """Append-only writer. The only mutation exposed is :meth:`append`."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: TraceRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[TraceRecord]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return [TraceRecord.model_validate(json.loads(line)) for line in f if line.strip()]
