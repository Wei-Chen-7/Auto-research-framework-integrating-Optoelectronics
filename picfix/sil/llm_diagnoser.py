"""LLM-backed root-cause diagnoser (diagnoser #2)."""
from __future__ import annotations

import json
import re

from picfix.agents.llm import LLMClient, LLMResponse
from picfix.core.nfo import DiagnosisLabels
from picfix.core.spec import WorkingVerifierConfig
from picfix.sil.evidence import FailureEvidence

DIAGNOSER_NAME = "llm_diagnoser"

_SYSTEM = (
    "You are a photonic integrated circuit design assistant diagnosing why a "
    "directional coupler design failed verification. Answer ONLY with a JSON "
    "object: {\"root_causes\": [...], \"affected_params\": [...], "
    "\"suggested_actions\": [...]}. Allowed root_causes: param_out_of_bounds, "
    "coupling_length_mismatch, gap_too_small_scattering, wavelength_drift, "
    "insertion_loss_excess, drc_violation. Allowed affected_params: gap_nm, "
    "length_um, width_nm. Allowed suggested_actions: adjust_length, "
    "increase_gap, decrease_gap, reduce_length, fix_drc."
)


def _render_user(evidence: FailureEvidence, verifier: WorkingVerifierConfig) -> str:
    return (
        f"Design: gap={evidence.params.gap_nm:.1f} nm, length={evidence.params.length_um:.3f} um, "
        f"width={evidence.params.width_nm:.1f} nm.\n"
        f"Visible feedback (noisy): centre cross ratio={evidence.center_ratio:.4f} "
        f"(target {verifier.split_target}±{verifier.split_tolerance}), "
        f"worst ratio deviation={evidence.worst_ratio_dev:.4f}, "
        f"band-edge deviation={evidence.edge_ratio_dev:.4f}, "
        f"max insertion loss={evidence.max_il_db:.3f} dB (limit {verifier.il_max_db}).\n"
        f"DRC violations: {', '.join(evidence.drc_rule_names) or 'none'}.\n"
        "Diagnose the root causes."
    )


def parse_labels(response: LLMResponse) -> DiagnosisLabels:
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    payload = json.loads(match.group(0)) if match else {}
    return DiagnosisLabels(
        diagnoser=DIAGNOSER_NAME,
        root_causes=tuple(sorted({str(c) for c in payload.get("root_causes", [])})),
        affected_params=tuple(sorted({str(p) for p in payload.get("affected_params", [])})),
        suggested_actions=tuple(sorted({str(a) for a in payload.get("suggested_actions", [])})),
    )


def diagnose(
    llm: LLMClient, evidence: FailureEvidence, verifier: WorkingVerifierConfig
) -> tuple[DiagnosisLabels, LLMResponse]:
    context = {
        "request": "diagnose",
        "gap_nm": evidence.params.gap_nm,
        "length_um": evidence.params.length_um,
        "width_nm": evidence.params.width_nm,
        "center_ratio": evidence.center_ratio,
        "ratio_failed": evidence.ratio_failed(verifier.split_tolerance),
        "il_failed": evidence.il_failed(verifier.il_max_db),
        "drc_violations": list(evidence.drc_rule_names),
    }
    response = llm.complete(_SYSTEM, _render_user(evidence, verifier), context)
    return parse_labels(response), response
