"""Versioned prior library.

Natural-language design priors accumulated across tasks by the meta arms
and injected into the agent system prompt. The only mutation is append
(via R3 proposals); every entry carries its source trace ID and the
version at which it was added. In the governed arm appends additionally
require passing the Gate and are frozen when SDS exceeds its threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PriorEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    source_trace_id: str
    added_at_version: int


class PriorStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"version": 0, "entries": []})

    @property
    def version(self) -> int:
        return int(self._read()["version"])

    def entries(self) -> list[PriorEntry]:
        return [PriorEntry.model_validate(e) for e in self._read()["entries"]]

    def append(self, text: str, source_trace_id: str) -> int:
        """Append one prior; returns the new store version."""
        doc = self._read()
        new_version = int(doc["version"]) + 1
        entry = PriorEntry(text=text, source_trace_id=source_trace_id, added_at_version=new_version)
        doc["entries"].append(entry.model_dump())
        doc["version"] = new_version
        self._write(doc)
        return new_version

    def render_for_prompt(self) -> str:
        entries = self.entries()
        if not entries:
            return ""
        lines = [f"- {e.text} (v{e.added_at_version})" for e in entries]
        return "Design priors learned from earlier tasks:\n" + "\n".join(lines)

    def _read(self) -> dict:
        with self._path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, doc: dict) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
