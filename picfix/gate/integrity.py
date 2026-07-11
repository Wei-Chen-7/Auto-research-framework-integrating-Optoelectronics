"""Hash manifest over meta-immutable assets.

Baseline hashes are taken at experiment start; every Gate round re-hashes
and any difference aborts the experiment. Config sections are hashed on
their canonical JSON so a "loosen the spec in the YAML file" edit is
caught just like a code edit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


class ExperimentIntegrityError(RuntimeError):
    """A meta-immutable asset changed — the experiment must abort."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_obj(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class HashManifest:
    def __init__(
        self,
        root: Path,
        file_globs: list[str],
        config_path: Path | None,
        config_sections: list[str],
    ) -> None:
        self._root = root
        self._globs = file_globs
        self._config_path = config_path
        self._sections = config_sections
        self._baseline = self._snapshot()
        if not self._baseline:
            raise ExperimentIntegrityError("hash manifest is empty — nothing protected")

    def _snapshot(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for pattern in self._globs:
            for path in sorted(self._root.glob(pattern)):
                if path.is_file():
                    hashes[str(path.relative_to(self._root))] = _sha256_file(path)
        if self._config_path is not None and self._config_path.exists():
            with self._config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            for section in self._sections:
                hashes[f"config:{section}"] = _sha256_obj(raw.get(section))
        return hashes

    def verify(self) -> None:
        current = self._snapshot()
        if current != self._baseline:
            changed = sorted(
                set(self._baseline) ^ set(current)
                | {k for k in self._baseline if current.get(k) != self._baseline[k]}
            )
            raise ExperimentIntegrityError(
                f"meta-immutable assets changed, aborting experiment: {changed}"
            )
