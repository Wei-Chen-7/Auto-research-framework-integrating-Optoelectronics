"""Tamper-evident append-only audit log.

Records form a hash chain: each entry stores the SHA-256 of the previous
entry's hash plus its own canonical payload. The public API exposes only
``append`` and read/verify operations — there is no update or delete.
The Gate's integrity check calls :meth:`verify_chain` every round and the
experiment aborts on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_GENESIS = "0" * 64


def _entry_hash(index: int, prev_hash: str, kind: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"index": index, "prev": prev_hash, "kind": kind, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditTamperedError(RuntimeError):
    """Raised when the audit chain fails verification."""


class AuditLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, kind: str, payload: dict[str, Any]) -> None:
        entries = self._raw_entries()
        index = len(entries)
        prev_hash = entries[-1]["hash"] if entries else _GENESIS
        entry = {
            "index": index,
            "prev": prev_hash,
            "kind": kind,
            "payload": payload,
            "hash": _entry_hash(index, prev_hash, kind, payload),
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        self.verify_chain()
        return self._raw_entries()

    def verify_chain(self) -> None:
        prev_hash = _GENESIS
        for i, entry in enumerate(self._raw_entries()):
            expected = _entry_hash(i, prev_hash, entry.get("kind", ""), entry.get("payload", {}))
            if entry.get("index") != i or entry.get("prev") != prev_hash or entry.get("hash") != expected:
                raise AuditTamperedError(f"audit log chain broken at entry {i}: {self._path}")
            prev_hash = entry["hash"]

    def _raw_entries(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
