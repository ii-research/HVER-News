from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean


KS = (1, 3, 5, 10)


def dcg(relevances: list[int]) -> float:
    return sum(
        (2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances)
    )


def per_query(ranked: list[str], qrels: dict[str, int]) -> dict[str, float]:
    direct = {doc_id for doc_id, rel in qrels.items() if rel >= 2}
    result: dict[str, float] = {}
    for k in KS:
        top = ranked[:k]
        result[f"recall@{k}"] = (
            len(direct.intersection(top)) / len(direct) if direct else 0.0
        )
        ideal = sorted(qrels.values(), reverse=True)[:k]
        denom = dcg(ideal)
        result[f"ndcg@{k}"] = (
            dcg([qrels.get(x, 0) for x in top]) / denom if denom else 0.0
        )
    result["mrr@10"] = next(
        (1.0 / rank for rank, x in enumerate(ranked[:10], 1) if x in direct), 0.0
    )
    return result


def aggregate(rows: list[dict[str, float]]) -> dict:
    if not rows:
        return {"queries": 0}
    return {
        "queries": len(rows),
        **{key: mean(row[key] for row in rows) for key in rows[0]},
    }


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    query_types: dict[str, str],
) -> dict:
    rows: list[dict[str, float]] = []
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    skipped_no_direct = 0
    for qid in sorted(set(rankings).intersection(qrels)):
        if not any(rel >= 2 for rel in qrels[qid].values()):
            skipped_no_direct += 1
            continue
        row = per_query(rankings[qid], qrels[qid])
        rows.append(row)
        groups[query_types.get(qid, "unknown")].append(row)
    return {
        "overall": aggregate(rows),
        "by_query_type": {
            name: aggregate(group) for name, group in sorted(groups.items())
        },
        "skipped_without_direct_qrel": skipped_no_direct,
    }
