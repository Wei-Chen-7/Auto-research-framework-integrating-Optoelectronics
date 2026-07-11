"""Normalized Failure Object (NFO) — output of the semantic interface layer."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DiagnosisLabels(BaseModel):
    """Structured label set produced by one diagnoser for one failure trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnoser: str
    root_causes: tuple[str, ...]
    affected_params: tuple[str, ...]
    suggested_actions: tuple[str, ...]

    def label_set(self) -> frozenset[str]:
        return frozenset(
            [f"cause:{c}" for c in self.root_causes]
            + [f"param:{p}" for p in self.affected_params]
            + [f"action:{a}" for a in self.suggested_actions]
        )


class NFO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nfo_id: str
    task_id: str
    trace_id: str
    diagnoses: tuple[DiagnosisLabels, ...]
    sds: float
    needs_human_review: bool
