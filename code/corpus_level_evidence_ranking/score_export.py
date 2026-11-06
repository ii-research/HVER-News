from __future__ import annotations

import json
import math
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finite_document_score_export(
    ranked_docs: list[str],
    reranked_docs: list[str],
    reranker_scores: list[float],
    first_stage_score_by_doc: dict[str, float],
) -> tuple[list[float], list[float]]:
    """Return finite, rank-aligned scores without mixing model score scales."""
    if len(reranked_docs) != len(reranker_scores):
        raise ValueError(
            "reranked documents and reranker scores have different lengths"
        )
    if not reranker_scores or not all(
        math.isfinite(score) for score in reranker_scores
    ):
        raise ValueError("reranker scores must be non-empty and finite")
    if set(ranked_docs) != set(first_stage_score_by_doc):
        raise ValueError("first-stage and final document sets differ")

    fallback_start = min(reranker_scores) - 1.0
    fallback_count = len(ranked_docs) - len(reranked_docs)
    fallback_scores = [fallback_start - rank * 1e-6 for rank in range(fallback_count)]
    document_scores = list(reranker_scores) + fallback_scores
    first_stage_scores = [
        float(first_stage_score_by_doc[doc_id]) for doc_id in ranked_docs
    ]
    if not all(math.isfinite(score) for score in document_scores + first_stage_scores):
        raise ValueError("non-finite score in document export")
    return document_scores, first_stage_scores
