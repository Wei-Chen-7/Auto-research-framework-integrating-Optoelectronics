"""M6 acceptance: trace/audit/priors expose no mutation beyond append,
and tampering with the audit chain is detected."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from picfix.core.audit import AuditLog, AuditTamperedError
from picfix.core.params import DesignParams
from picfix.core.trace import SpectrumSummary, TraceRecord, TraceWriter
from picfix.priors.store import PriorStore


def _record(i: int) -> TraceRecord:
    return TraceRecord(
        trace_id=f"t{i}",
        task_id="task1",
        arm="fixed_loop",
        call_index=i,
        params=DesignParams(gap_nm=200.0, length_um=15.7, width_nm=500.0),
        summary=SpectrumSummary(worst_ratio_dev=0.01, max_il_db=0.2, center_ratio=0.5),
        backend="analytical",
        seed=1,
        elapsed_s=0.001,
        agent_version=1,
        workflow_version=1,
        verifier_version=1,
        prior_version=0,
    )


def test_trace_writer_api_is_append_only(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "trace.jsonl")
    mutators = [
        m for m in dir(writer)
        if not m.startswith("_") and any(k in m.lower() for k in ("delete", "remove", "update", "edit", "truncate", "clear", "pop", "write"))
    ]
    assert mutators == []
    writer.append(_record(0))
    writer.append(_record(1))
    assert [r.trace_id for r in writer.read_all()] == ["t0", "t1"]


def test_audit_log_api_is_append_only(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    mutators = [
        m for m in dir(log)
        if not m.startswith("_") and any(k in m.lower() for k in ("delete", "remove", "update", "edit", "truncate", "clear", "pop", "write"))
    ]
    assert mutators == []


def test_audit_chain_detects_edit(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("gate_decision", {"proposal_id": "p1", "verdict": "rejected"})
    log.append("gate_decision", {"proposal_id": "p2", "verdict": "approved"})
    lines = path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["payload"]["verdict"] = "approved"  # falsify history
    path.write_text("\n".join([json.dumps(entry, sort_keys=True)] + lines[1:]) + "\n")
    with pytest.raises(AuditTamperedError):
        log.verify_chain()


def test_audit_chain_detects_deletion(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append("event", {"i": i})
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")  # drop the middle record
    with pytest.raises(AuditTamperedError):
        log.verify_chain()


def test_prior_store_append_only_and_versioned(tmp_path: Path) -> None:
    store = PriorStore(tmp_path / "priors.json")
    assert store.version == 0
    v1 = store.append("scattering loss rises sharply once gap drops below ~150 nm", "t42")
    v2 = store.append("fix split ratio with length before touching the gap", "t43")
    assert (v1, v2) == (1, 2)
    entries = store.entries()
    assert entries[0].source_trace_id == "t42" and entries[1].added_at_version == 2
    mutators = [
        m for m in dir(store)
        if not m.startswith("_") and any(k in m.lower() for k in ("delete", "remove", "update", "edit", "clear", "pop", "set"))
    ]
    assert mutators == []
    assert "150 nm" in store.render_for_prompt()
