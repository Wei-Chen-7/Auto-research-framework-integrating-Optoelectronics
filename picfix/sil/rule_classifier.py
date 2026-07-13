"""Rule-based root-cause classifier (diagnoser #1).

Operates only on agent-visible evidence. Its label taxonomy matches the
ground-truth RootCause values so Diagnose Accuracy can be computed, but it
never reads the truth channel.
"""
from __future__ import annotations

from picfix.core.nfo import DiagnosisLabels
from picfix.core.spec import WorkingVerifierConfig
from picfix.sil.evidence import FailureEvidence

DIAGNOSER_NAME = "rule_classifier"
_SMALL_GAP_NM = 150.0


def classify(evidence: FailureEvidence, verifier: WorkingVerifierConfig) -> DiagnosisLabels:
    causes: list[str] = []
    params: list[str] = []
    actions: list[str] = []

    if evidence.drc_rule_names:
        causes.append("drc_violation")
        actions.append("fix_drc")
        if "min_gap" in evidence.drc_rule_names:
            params.append("gap_nm")
        if any(r in evidence.drc_rule_names for r in ("min_width", "max_width")):
            params.append("width_nm")
        if any(r in evidence.drc_rule_names for r in ("min_length", "max_length")):
            params.append("length_um")

    center_dev = abs(evidence.center_ratio - verifier.split_target)
    if evidence.ratio_failed(verifier.split_tolerance):
        if center_dev > verifier.split_tolerance:
            causes.append("coupling_length_mismatch")
            params.append("length_um")
            actions.append("adjust_length")
        elif evidence.edge_ratio_dev > verifier.split_tolerance:
            causes.append("wavelength_drift")
            params.append("length_um")
            actions.append("adjust_length")

    if evidence.il_failed(verifier.il_max_db):
        if evidence.params.gap_nm < _SMALL_GAP_NM:
            causes.append("gap_too_small_scattering")
            params.append("gap_nm")
            actions.append("increase_gap")
        else:
            causes.append("insertion_loss_excess")
            params.append("length_um")
            actions.append("reduce_length")

    return DiagnosisLabels(
        diagnoser=DIAGNOSER_NAME,
        root_causes=tuple(sorted(set(causes))),
        affected_params=tuple(sorted(set(params))),
        suggested_actions=tuple(sorted(set(actions))),
    )
