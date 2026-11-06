"""Evaluate document-level evidence ranking over all source-article sentences."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from prepare_document_level_baseline import load_corpus, read_jsonl, validate_instance


def evidence_ranking_metrics(row, scores, threshold=0.1):
    candidates = row["article_local_candidate_doc_ids"]
    if len(scores) != len(candidates) or not all(math.isfinite(x) for x in scores):
        raise ValueError(f"Invalid sentence probabilities: {row['query_id']}")
    ranked = sorted(zip(candidates, scores), key=lambda x: (-x[1], x[0]))
    direct = set(row["direct_doc_ids"])
    partial = set(row["partial_doc_ids"]) - direct
    minimal = set(row["minimal_sufficient_doc_ids"])
    result = {
        "mrr": next(
            (1 / i for i, (doc, _) in enumerate(ranked, 1) if doc in direct), 0.0
        )
    }
    grades = {
        doc: 2 if doc in direct else 1 if doc in partial else 0 for doc in candidates
    }
    for k in [1, 5, 10]:
        top = {doc for doc, _ in ranked[:k]}
        result[f"direct_recall@{k}"] = len(top & direct) / len(direct)
        result[f"minimal_complete@{k}"] = float(minimal <= top) if minimal else 0.0
        gains = [grades[doc] for doc, _ in ranked[:k]]
        ideal = sorted(grades.values(), reverse=True)[:k]
        dcg = lambda values: sum(
            (2**g - 1) / math.log2(i + 2) for i, g in enumerate(values)
        )
        result[f"graded_ndcg@{k}"] = dcg(gains) / dcg(ideal)
    kept = {doc for doc, score in ranked if score >= threshold}
    tp = len(kept & direct)
    result.update(
        precision=tp / len(kept) if kept else 0.0,
        recall=tp / len(direct),
        f2=5 * tp / (4 * len(direct) + len(kept)),
        sentence_retention=len(kept) / len(candidates),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[3] / "dataset/document-level"
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    corpus = load_corpus(args.source / args.split / "corpus.jsonl")
    totals: dict[str, float] = defaultdict(float)
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    with predictions_path.open("w") as handle, torch.inference_mode():
        for row in read_jsonl(args.source / args.split / "instances.jsonl"):
            if args.limit is not None and count >= args.limit:
                break
            validate_instance(row, corpus)
            sentences = [
                corpus[doc]["text"] for doc in row["article_local_candidate_doc_ids"]
            ]
            result = model.process(
                question=[row["query"]],
                context=[sentences],
                sentence_splitter=None,
                return_sentence_metrics=True,
                batch_size=args.batch_size,
                threshold=args.threshold,
                show_progress=False,
            )
            scores = result["sentence_probabilities"][0]
            metrics = evidence_ranking_metrics(row, scores, args.threshold)
            count += 1
            counts[row["query_type"]] += 1
            for key, value in metrics.items():
                totals[key] += value
                by_type[row["query_type"]][key] += value
            handle.write(
                json.dumps(
                    {
                        "query_id": row["query_id"],
                        "doc_ids": row["article_local_candidate_doc_ids"],
                        "scores": scores,
                    }
                )
                + "\n"
            )
            if count % 100 == 0:
                print(f"Evaluated {count} queries", flush=True)
    if count == 0:
        raise ValueError("No queries evaluated")
    summary = {
        "model": args.model,
        "split": args.split,
        "queries": count,
        "threshold": args.threshold,
        "limit": args.limit,
        "metrics": {k: v / count for k, v in totals.items()},
        "by_query_type": {
            t: {k: v / counts[t] for k, v in values.items()}
            for t, values in by_type.items()
        },
        "grades": {"direct": 2, "partial": 1, "other": 0},
        "averaging": "macro per query; binary metrics use direct evidence only",
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
