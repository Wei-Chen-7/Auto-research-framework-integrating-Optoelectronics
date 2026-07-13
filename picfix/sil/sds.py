"""Semantic Divergence Score.

SDS = 1 - mean pairwise Jaccard agreement between the structured label
sets that heterogeneous diagnosers produced for the SAME failure trace.

It is deliberately NOT computed between the optical criterion and the DRC
criterion: those check different properties, single-sided failures would
disagree by construction, and the score would saturate (DESIGN.md §7).
"""
from __future__ import annotations

from itertools import combinations


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def sds(label_sets: list[frozenset[str]]) -> float:
    if len(label_sets) < 2:
        raise ValueError("SDS needs at least two diagnoser label sets")
    pairs = list(combinations(label_sets, 2))
    return 1.0 - sum(jaccard(a, b) for a, b in pairs) / len(pairs)
