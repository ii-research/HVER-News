from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import CorpusData
from ..evaluate import metrics_from_predictions
from ..score_export import finite_document_score_export


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@torch.inference_mode()
def score_pairs(
    model,
    tokenizer,
    query: str,
    passages: list[str],
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(passages), batch_size):
        pairs = [[query, passage] for passage in passages[start : start + batch_size]]
        inputs = tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            logits = model(**inputs, return_dict=True).logits.view(-1)
        scores.extend(logits.float().cpu().tolist())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--first-stage", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "dataset/corpus-level",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()

    started = time.time()
    data = CorpusData.load(args.data_dir, "test")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    partial = args.results_dir / "predictions.partial.jsonl"
    complete = args.results_dir / "predictions.jsonl"
    if complete.exists():
        raise FileExistsError(f"Completed predictions already exist: {complete}")
    finished: set[str] = set()
    if partial.exists():
        finished = {row["query_id"] for row in read_jsonl(partial)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    )
    written = len(finished)
    with partial.open("a", encoding="utf-8") as output:
        for index, base in enumerate(read_jsonl(args.first_stage)):
            if args.max_queries is not None and index >= args.max_queries:
                break
            qid = base["query_id"]
            if qid in finished:
                continue
            sentence_ids = base["ranked_sentence_ids"][: args.candidate_depth]
            passages = [
                data.sentences[sid].get("title", "")
                + "\n"
                + data.sentences[sid]["text"]
                for sid in sentence_ids
            ]
            scores = score_pairs(
                model, tokenizer, base["query"], passages, args.batch_size, device
            )
            sentence_order = sorted(
                range(len(sentence_ids)), key=lambda i: (-scores[i], sentence_ids[i])
            )
            ranked_sentences = [sentence_ids[i] for i in sentence_order]
            ranked_sentence_scores = [scores[i] for i in sentence_order]

            document_scores: dict[str, float] = defaultdict(lambda: float("-inf"))
            for sid, score in zip(sentence_ids, scores, strict=True):
                aid = data.sentence_article[sid]
                document_scores[aid] = max(document_scores[aid], float(score))
            reranked_docs = sorted(
                document_scores, key=lambda aid: (-document_scores[aid], aid)
            )
            remaining_docs = [
                aid for aid in base["ranked_document_ids"] if aid not in document_scores
            ]
            ranked_docs = reranked_docs + remaining_docs
            reranker_doc_scores = [document_scores[aid] for aid in reranked_docs]
            first_stage_score_by_doc = dict(
                zip(
                    base["ranked_document_ids"],
                    base["document_scores"],
                    strict=True,
                )
            )
            doc_scores, first_stage_doc_scores = finite_document_score_export(
                ranked_docs,
                reranked_docs,
                reranker_doc_scores,
                first_stage_score_by_doc,
            )
            row = {
                **{
                    key: base[key]
                    for key in (
                        "query_id",
                        "query",
                        "query_type",
                        "relation_id",
                        "source_article_id",
                    )
                },
                "ranked_document_ids": ranked_docs,
                "document_scores": doc_scores,
                "first_stage_document_scores": first_stage_doc_scores,
                "document_ranking_is_complete": True,
                "reranked_document_depth": len(reranked_docs),
                "document_score_semantics": (
                    "cross-encoder logits for reranked documents; finite rank-preserving "
                    "fallback below the minimum candidate logit for unscored documents"
                ),
                "first_stage_document_score_semantics": "original first-stage BGE-M3 score",
                "ranked_sentence_ids": ranked_sentences,
                "sentence_scores": ranked_sentence_scores,
                "sentence_ranking_depth": len(ranked_sentences),
                "first_stage": str(args.first_stage),
                "candidate_depth": args.candidate_depth,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            written += 1
            if written % 100 == 0:
                print(f"reranked {written}", flush=True)

    partial.replace(complete)
    rows = list(read_jsonl(complete))
    metrics = metrics_from_predictions(data, rows)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = vars(args) | {
        "first_stage": str(args.first_stage.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "queries": len(rows),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "pid": os.getpid(),
        "protocol": "rerank first-stage top sentence candidates; append noncandidate documents in first-stage order",
        "dataset_manifest": data.manifest(),
    }
    (args.results_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
