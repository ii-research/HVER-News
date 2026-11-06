from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import CorpusData, write_jsonl
from .metrics import evaluate_rankings


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(
        self, identifiers: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75
    ):
        self.identifiers = identifiers
        self.k1, self.b = k1, b
        self.lengths = np.asarray(
            [len(tokens(text)) for text in texts], dtype=np.float32
        )
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 1.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, text in enumerate(texts):
            for term, freq in Counter(tokens(text)).items():
                postings[term].append((index, freq))
        count = len(identifiers)
        self.postings = postings
        self.idf = {
            term: math.log(1.0 + (count - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in postings.items()
        }

    def rank(
        self, query: str, limit: int | None = None
    ) -> tuple[list[str], list[float]]:
        scores = np.zeros(len(self.identifiers), dtype=np.float32)
        for term in set(tokens(query)):
            for index, freq in self.postings.get(term, ()):
                norm = freq + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / self.avgdl
                )
                scores[index] += self.idf[term] * freq * (self.k1 + 1.0) / norm
        if limit is not None and limit < len(scores):
            chosen = np.argpartition(-scores, limit - 1)[:limit]
            local = np.lexsort((np.asarray(self.identifiers)[chosen], -scores[chosen]))
            order = chosen[local]
        else:
            order = np.lexsort((np.asarray(self.identifiers), -scores))
        return [self.identifiers[i] for i in order], [float(scores[i]) for i in order]


def rank_bm25(data: CorpusData, qids: list[str]) -> list[dict]:
    article_ids = sorted(data.articles)
    article_texts = [
        data.articles[x].get("title", "")
        + "\n"
        + " ".join(s["text"] for s in data.articles[x]["sentences"])
        for x in article_ids
    ]
    sentence_ids = sorted(data.sentences)
    sentence_texts = [
        data.sentences[x].get("title", "") + "\n" + data.sentences[x]["text"]
        for x in sentence_ids
    ]
    article_index = BM25Index(article_ids, article_texts)
    sentence_index = BM25Index(sentence_ids, sentence_texts)
    rows = []
    for qid in qids:
        query = data.queries[qid]["query"]
        docs, doc_scores = article_index.rank(query)
        sentences, sentence_scores = sentence_index.rank(query, 100)
        rows.append(
            prediction_row(data, qid, docs, doc_scores, sentences, sentence_scores)
        )
    return rows


def encode(model, texts: list[str], query: bool, batch_size: int, device) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            value = (
                model.encode_queries(batch, device)
                if query
                else model.encode_passages(batch, device)
            )
        chunks.append(value.detach().float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)


def rank_dense(
    data: CorpusData,
    qids: list[str],
    model_path: str,
    batch_size: int,
    device_name: str,
) -> list[dict]:
    from document_level_evidence_ranking.model import DualEncoder, resolve_device

    device = resolve_device(device_name)
    checkpoint = Path(model_path)
    model = (
        DualEncoder.load(checkpoint)
        if checkpoint.is_dir() and (checkpoint / "retriever_config.json").exists()
        else DualEncoder(model_path)
    )
    model.to(device)
    return rank_dense_model(data, qids, model, batch_size, device)


def rank_dense_model(
    data: CorpusData, qids: list[str], model, batch_size: int, device
) -> list[dict]:
    sentence_ids = sorted(data.sentences)
    passages = [
        data.sentences[x].get("title", "") + "\n" + data.sentences[x]["text"]
        for x in sentence_ids
    ]
    sentence_emb = encode(model, passages, False, batch_size, device)
    query_emb = encode(
        model, [data.queries[q]["query"] for q in qids], True, batch_size, device
    )
    article_ids = sorted(data.articles)
    sentence_index = {sid: index for index, sid in enumerate(sentence_ids)}
    article_indexes = {
        aid: np.asarray(
            [sentence_index[sid] for sid in data.article_sentence_ids[aid]],
            dtype=np.int64,
        )
        for aid in article_ids
    }
    rows = []
    for offset, qid in enumerate(qids):
        scores = sentence_emb @ query_emb[offset]
        sentence_limit = min(100, len(scores))
        chosen = np.argpartition(-scores, sentence_limit - 1)[:sentence_limit]
        sentence_local_order = np.lexsort(
            (np.asarray(sentence_ids)[chosen], -scores[chosen])
        )
        sentence_order = chosen[sentence_local_order]
        document_scores = np.asarray(
            [float(scores[idxs].max()) for idxs in article_indexes.values()],
            dtype=np.float32,
        )
        document_order = np.lexsort((np.asarray(article_ids), -document_scores))
        rows.append(
            prediction_row(
                data,
                qid,
                [article_ids[i] for i in document_order],
                [float(document_scores[i]) for i in document_order],
                [sentence_ids[i] for i in sentence_order],
                [float(scores[i]) for i in sentence_order],
            )
        )
    return rows


def prediction_row(
    data: CorpusData,
    qid: str,
    docs: list[str],
    doc_scores: list[float],
    sentences: list[str],
    sentence_scores: list[float],
) -> dict:
    return {
        "query_id": qid,
        "query": data.queries[qid]["query"],
        "query_type": data.queries[qid].get("query_type"),
        "relation_id": data.queries[qid].get("relation_id"),
        "source_article_id": data.source_article(qid),
        "ranked_document_ids": docs,
        "document_scores": doc_scores,
        "document_ranking_is_complete": True,
        "ranked_sentence_ids": sentences[:100],
        "sentence_scores": sentence_scores[:100],
        "sentence_ranking_depth": min(100, len(sentences)),
    }


def metrics_from_predictions(data: CorpusData, rows: list[dict]) -> dict:
    """Score CER in full; cross_only_* excludes each query's source article.

    document/sentence name the ranking unit, not DER/CER task scopes.
    Cross-document-only is a diagnostic subset, not a third task.
    """
    query_types = {
        qid: str(row.get("query_type", "unknown")) for qid, row in data.queries.items()
    }
    doc_rankings = {row["query_id"]: row["ranked_document_ids"] for row in rows}
    sentence_rankings = {row["query_id"]: row["ranked_sentence_ids"] for row in rows}
    cross_docs, cross_sentences = {}, {}
    cross_doc_qrels, cross_sentence_qrels = {}, {}
    for qid in doc_rankings:
        source = data.source_article(qid)
        dq = {
            did: rel
            for did, rel in data.document_qrels.get(qid, {}).items()
            if did != source
        }
        sq = {
            sid: rel
            for sid, rel in data.sentence_qrels.get(qid, {}).items()
            if data.sentence_article[sid] != source
        }
        if any(rel >= 2 for rel in dq.values()):
            cross_docs[qid] = [did for did in doc_rankings[qid] if did != source]
            cross_doc_qrels[qid] = dq
        if any(rel >= 2 for rel in sq.values()):
            cross_sentences[qid] = [
                sid
                for sid in sentence_rankings[qid]
                if data.sentence_article[sid] != source
            ]
            cross_sentence_qrels[qid] = sq
    return {
        "document": evaluate_rankings(doc_rankings, data.document_qrels, query_types),
        "sentence": evaluate_rankings(
            sentence_rankings, data.sentence_qrels, query_types
        ),
        "cross_only_document": evaluate_rankings(
            cross_docs, cross_doc_qrels, query_types
        ),
        "cross_only_sentence": evaluate_rankings(
            cross_sentences, cross_sentence_qrels, query_types
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-level evidence ranking (CER): source and other articles in the split.")
    parser.add_argument("--setting", required=True, choices=("bm25", "dense"))
    parser.add_argument("--model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset/corpus-level",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    if args.setting == "dense" and not args.model:
        parser.error("--model is required for dense")
    started = time.time()
    data = CorpusData.load(args.data_dir, args.split)
    qids = sorted(data.queries)
    if args.max_queries is not None:
        qids = qids[: args.max_queries]
    rows = (
        rank_bm25(data, qids)
        if args.setting == "bm25"
        else rank_dense(data, qids, args.model, args.eval_batch_size, args.device)
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.results_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite non-empty results directory {args.results_dir}: "
            f"{[path.name for path in existing[:5]]}"
        )
    write_jsonl(args.results_dir / "predictions.jsonl", rows)
    metrics = metrics_from_predictions(data, rows)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = vars(args) | {
        "data_dir": str(args.data_dir.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "query_count": len(qids),
        "elapsed_seconds": time.time() - started,
        "dataset_manifest": data.manifest(),
        "pid": os.getpid(),
    }
    (args.results_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
