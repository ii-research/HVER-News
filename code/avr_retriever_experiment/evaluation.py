from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from avr_retriever_experiment.config import ExperimentConfig
from avr_retriever_experiment.data import BenchmarkData, candidate_doc_ids
from avr_retriever_experiment.model import DualEncoder, resolve_device


def _dcg(rels: Sequence[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def _per_query(ranked: List[str], qrels: Dict[str, int], cfg: ExperimentConfig) -> dict:
    direct = {d for d, r in qrels.items() if r >= cfg.binary_relevance_threshold}
    out = {}
    for k in cfg.metric_ks:
        top = ranked[:k]
        hits = len(direct.intersection(top))
        recall = hits / len(direct) if direct else 0.0
        precision = hits / max(k, 1)
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        out[f"recall@{k}"] = recall
        out[f"precision@{k}"] = precision
        out[f"f1@{k}"] = f1
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        rels = [qrels.get(d, 0) for d in top]
        ideal = sorted(qrels.values(), reverse=True)[:k]
        idcg = _dcg(ideal)
        out[f"ndcg@{k}"] = _dcg(rels) / idcg if idcg > 0 else 0.0
    rr = 0.0
    # The benchmark reports MRR@10, aligned with the largest configured k.
    # Rankings may contain all candidates for offline diagnostics.
    for rank, did in enumerate(ranked[: max(cfg.metric_ks)], 1):
        if did in direct:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def _aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {"queries": 0}
    return {
        "queries": len(rows),
        **{k: mean(float(x[k]) for x in rows) for k in rows[0]},
    }


@torch.no_grad()
def encode_texts(
    model: DualEncoder,
    texts: List[str],
    is_query: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="encode-query" if is_query else "encode-corpus",
        leave=False,
    ):
        batch = texts[i : i + batch_size]
        emb = (
            model.encode_queries(batch, device)
            if is_query
            else model.encode_passages(batch, device)
        )
        # NumPy has no native bfloat16 dtype. Qwen3-Embedding loads in BF16,
        # so normalize the exported evaluation matrix to FP32 first.
        chunks.append(emb.detach().float().cpu().numpy())
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def evaluate_model(
    model: DualEncoder,
    data: BenchmarkData,
    query_ids: Sequence[str],
    cfg: ExperimentConfig,
    save_predictions: bool = False,
) -> tuple[dict, List[dict]]:
    device = resolve_device(cfg.device)
    model.to(device)

    doc_ids = list(data.corpus.keys())
    doc_texts = [str(data.corpus[d].get("text", "")) for d in doc_ids]
    doc_emb = encode_texts(model, doc_texts, False, cfg.eval_batch_size, device)
    doc_index = {d: i for i, d in enumerate(doc_ids)}

    qids = list(query_ids)
    q_texts = [data.queries[q]["query"] for q in qids]
    q_emb = encode_texts(model, q_texts, True, cfg.eval_batch_size, device)

    per, preds = [], []
    groups_qt, groups_ul = defaultdict(list), defaultdict(list)
    max_k = max(cfg.metric_ks)

    for qi, qid in enumerate(qids):
        cands = candidate_doc_ids(data, qid, cfg.retrieval_scope)
        idxs = np.array([doc_index[d] for d in cands], dtype=np.int64)
        scores = doc_emb[idxs] @ q_emb[qi]
        # Preserve every article-local candidate score.  Metrics still use
        # top-k internally, while downstream relation-sensitivity analyses
        # require scores for evidence sentences that may fall below Top-10.
        order = np.argsort(-scores)
        ranked = [cands[int(i)] for i in order]
        row = _per_query(ranked, data.qrels.get(qid, {}), cfg)
        per.append(row)
        inst = data.instances[qid]
        groups_qt[str(inst.get("query_type", "unknown"))].append(row)
        groups_ul[str(inst.get("unit_level", "unknown"))].append(row)
        if save_predictions:
            preds.append(
                {
                    "query_id": qid,
                    "query": data.queries[qid]["query"],
                    "query_type": inst.get("query_type"),
                    "unit_level": inst.get("unit_level"),
                    "candidate_count": len(cands),
                    "ranked_doc_ids": ranked,
                    "scores": [float(scores[int(i)]) for i in order],
                    "ranking_is_complete": True,
                    "llm_calls": 0,
                }
            )

    metrics = {
        "coverage": {
            "expected_queries": len(qids),
            "successful_queries": len(per),
            "failed_queries": 0,
        },
        "overall": _aggregate(per),
        "by_query_type": {k: _aggregate(v) for k, v in sorted(groups_qt.items())},
        "by_unit_level": {k: _aggregate(v) for k, v in sorted(groups_ul.items())},
        "efficiency": {"llm_calls_per_query": 0.0},
    }
    return metrics, preds
