"""Validate document-level evidence data and export sentence-window pruning supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    corpus = {}
    for row in read_jsonl(path):
        if row["doc_id"] in corpus:
            raise ValueError(f"Duplicate document: {row['doc_id']}")
        if not row["text"].strip():
            raise ValueError(f"Empty document: {row['doc_id']}")
        corpus[row["doc_id"]] = row
    return corpus


def validate_instance(row, corpus):
    candidates = row["article_local_candidate_doc_ids"]
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"Empty or duplicate candidates: {row['query_id']}")
    if not row["query"].strip():
        raise ValueError(f"Empty query: {row['query_id']}")
    for doc_id in candidates:
        if doc_id not in corpus or corpus[doc_id]["guid"] != row["guid"]:
            raise ValueError(f"Missing or cross-article candidate: {doc_id}")
    for field in ["direct_doc_ids", "partial_doc_ids", "minimal_sufficient_doc_ids"]:
        if not set(row[field]) <= set(candidates):
            raise ValueError(f"{field} outside candidate set: {row['query_id']}")
    if not row["direct_doc_ids"]:
        raise ValueError(f"No direct evidence: {row['query_id']}")


def make_windows(row, corpus, tokenizer, max_length):
    """Keep original sentences and order; fail instead of silently truncating evidence."""
    ids = sorted(
        row["article_local_candidate_doc_ids"], key=lambda x: corpus[x]["sentence_id"]
    )
    groups: list[list[str]] = []
    start = 0
    while start < len(ids):
        # Find the largest fitting prefix without repeatedly tokenizing every prefix.
        low, high = start + 1, len(ids)
        best = start
        while low <= high:
            end = (low + high) // 2
            text = "\n".join(corpus[x]["text"] for x in ids[start:end])
            size = len(tokenizer(row["query"], text, truncation=False)["input_ids"])
            if size <= max_length:
                best = end
                low = end + 1
            else:
                high = end - 1
        if best == start:
            raise ValueError(
                f"Single sentence exceeds {max_length} tokens: {ids[start]}; increase max_length"
            )
        groups.append(ids[start:best])
        start = best
    direct = set(row["direct_doc_ids"])
    for index, group in enumerate(groups):
        text = ""
        spans = []
        for doc_id in group:
            if text:
                text += "\n"
            start = len(text)
            text += corpus[doc_id]["text"]
            spans.append([start, len(text)])
        mask = [int(x in direct) for x in group]
        yield {
            "id": f"{row['query_id']}::w{index}",
            "query_id": row["query_id"],
            "guid": row["guid"],
            "query": row["query"],
            "texts": [text],
            "labels": [int(any(mask))],
            "context_spans": [spans],
            "context_spans_relevance": [mask],
            "doc_ids": group,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[3] / "dataset/document-level"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/document-level/provence/data")
    )
    parser.add_argument(
        "--tokenizer", default="Alibaba-NLP/gte-reranker-modernbert-base"
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--limit", type=int, help="Seeded query sample per split for debugging"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    tokenizer = None
    if not args.audit_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report: dict[str, Any] = {
        "seed": args.seed,
        "max_length": args.max_length,
        "tokenizer": args.tokenizer,
        "label_policy": "direct_only",
        "splits": {},
    }
    seen_articles: set[str] = set()
    datasets = {}
    args.output.mkdir(parents=True)
    for split in ["train", "dev", "test"]:
        corpus = load_corpus(args.source / split / "corpus.jsonl")
        articles = {r["guid"] for r in corpus.values()}
        if articles & seen_articles:
            raise ValueError(f"Article leakage in {split}")
        seen_articles |= articles
        query_ids = set()
        count = 0
        selected = []
        rng = random.Random(args.seed)
        for row in read_jsonl(args.source / split / "instances.jsonl"):
            validate_instance(row, corpus)
            if row["split"] != split or row["query_id"] in query_ids:
                raise ValueError(f"Wrong split or duplicate query: {row['query_id']}")
            query_ids.add(row["query_id"])
            count += 1
            if args.limit is None:
                continue
            if len(selected) < args.limit:
                if not args.audit_only:
                    selected.append(row)
            elif not args.audit_only:
                pos = rng.randrange(count)
                if pos < args.limit:
                    selected[pos] = row
        info: dict[str, Any] = {
            "articles": len(articles),
            "sentences": len(corpus),
            "queries": count,
        }
        for name in ["corpus.jsonl", "instances.jsonl"]:
            digest = hashlib.sha256()
            with (args.source / split / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            info[name + "_sha256"] = digest.hexdigest()
        if tokenizer is not None:
            from datasets import Dataset

            out = args.output / f"{split}.jsonl"
            windows = 0
            with out.open("w", encoding="utf-8") as handle:
                for row in (
                    selected
                    if args.limit is not None
                    else read_jsonl(args.source / split / "instances.jsonl")
                ):
                    for converted in make_windows(
                        row, corpus, tokenizer, args.max_length
                    ):
                        handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        windows += 1
            datasets["validation" if split == "dev" else split] = Dataset.from_json(
                str(out)
            )
            info.update(
                selected_queries=len(selected) if args.limit is not None else count,
                windows=windows,
            )
        report["splits"][split] = info
        print(split, info, flush=True)
    if not args.audit_only:
        from datasets import DatasetDict

        DatasetDict(datasets).save_to_disk(str(args.output / "hf"))
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
