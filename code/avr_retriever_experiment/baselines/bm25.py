from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import List, Sequence

from avr_retriever_experiment.config import ExperimentConfig
from avr_retriever_experiment.data import BenchmarkData, candidate_doc_ids
from avr_retriever_experiment.evaluation import _aggregate, _per_query

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tok(s: str) -> List[str]:
    return _TOKEN_RE.findall(str(s).lower())


def _bm25_scores(
    query_tokens: List[str], docs: List[List[str]], k1: float, b: float
) -> List[float]:
    n = len(docs)
    if n == 0:
        return []
    lengths = [len(x) for x in docs]
    avgdl = sum(lengths) / max(n, 1)
    tfs = [Counter(x) for x in docs]
    df = Counter()
    for tf in tfs:
        for term in tf:
            df[term] += 1
    scores = [0.0] * n
    for term in query_tokens:
        dft = df.get(term, 0)
        if dft == 0:
            continue
        idf = math.log(1.0 + (n - dft + 0.5) / (dft + 0.5))
        for i, tf in enumerate(tfs):
            f = tf.get(term, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1.0 - b + b * lengths[i] / max(avgdl, 1e-9))
            scores[i] += idf * (f * (k1 + 1.0) / denom)
    return scores


def evaluate_bm25(
    data: BenchmarkData, query_ids: Sequence[str], cfg: ExperimentConfig
) -> tuple[dict, List[dict]]:
    per, preds = [], []
    groups_qt, groups_ul = defaultdict(list), defaultdict(list)
    for qid in query_ids:
        cands = candidate_doc_ids(data, qid, cfg.retrieval_scope)
        docs = [tok(str(data.corpus[d].get("text", ""))) for d in cands]
        scores = _bm25_scores(
            tok(data.queries[qid]["query"]), docs, cfg.bm25_k1, cfg.bm25_b
        )
        # Save the complete article-local ranking for offline relation-change
        # diagnostics; _per_query applies the configured metric cutoffs.
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
        ranked = [cands[i] for i in order]
        row = _per_query(ranked, data.qrels.get(qid, {}), cfg)
        per.append(row)
        inst = data.instances[qid]
        groups_qt[str(inst.get("query_type", "unknown"))].append(row)
        groups_ul[str(inst.get("unit_level", "unknown"))].append(row)
        preds.append(
            {
                "query_id": qid,
                "candidate_count": len(cands),
                "ranked_doc_ids": ranked,
                "scores": [float(scores[i]) for i in order],
                "ranking_is_complete": True,
                "llm_calls": 0,
            }
        )
    metrics = {
        "coverage": {
            "expected_queries": len(query_ids),
            "successful_queries": len(per),
            "failed_queries": 0,
        },
        "overall": _aggregate(per),
        "by_query_type": {k: _aggregate(v) for k, v in sorted(groups_qt.items())},
        "by_unit_level": {k: _aggregate(v) for k, v in sorted(groups_ul.items())},
        "efficiency": {"llm_calls_per_query": 0.0},
    }
    return metrics, preds
