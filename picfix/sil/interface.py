"""Semantic Interface Layer orchestrator: evidence → NFO + SDS."""
from __future__ import annotations

import uuid

from picfix.agents.llm import LLMClient, LLMResponse
from picfix.core.nfo import NFO
from picfix.core.spec import WorkingVerifierConfig
from picfix.sil import llm_diagnoser, rule_classifier
from picfix.sil.evidence import FailureEvidence
from picfix.sil.sds import sds


class SemanticInterfaceLayer:
    def __init__(self, llm: LLMClient, sds_threshold: float) -> None:
        self._llm = llm
        self._sds_threshold = sds_threshold

    def diagnose(
        self, evidence: FailureEvidence, verifier: WorkingVerifierConfig
    ) -> tuple[NFO, LLMResponse]:
        rule_labels = rule_classifier.classify(evidence, verifier)
        llm_labels, response = llm_diagnoser.diagnose(self._llm, evidence, verifier)
        score = sds([rule_labels.label_set(), llm_labels.label_set()])
        nfo = NFO(
            nfo_id=f"nfo-{uuid.uuid4().hex[:12]}",
            task_id=evidence.task_id,
            trace_id=evidence.trace_id,
            diagnoses=(rule_labels, llm_labels),
            sds=score,
            needs_human_review=score > self._sds_threshold,
        )
        return nfo, response
