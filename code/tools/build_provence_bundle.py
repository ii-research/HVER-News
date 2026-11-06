#!/usr/bin/env python3
"""Build the immutable Provence/HVER input bundle for the document-level task."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--query-view-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()

    root = args.bundle_root.resolve()
    project = args.project_root.resolve()
    data_root = args.data_root.resolve()
    views = args.query_view_root.resolve()
    checkpoint = args.checkpoint.resolve()
    sys.path.insert(0, str(project))

    from avr_retriever_experiment.data import load_split
    from avr_retriever_experiment.sampling import select_experiment_query_ids

    shared = root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        shutil.copy2(
            views / f"{split}_relations_all.json",
            shared / f"{split}_relations_all.json",
        )

    # Freeze the same one-query-per-relation policy for the auxiliary test view.
    test = load_split(data_root / "test", "test")
    select_experiment_query_ids(
        test,
        root / "_test_view",
        "one_per_relation_balanced",
        42,
        True,
        (
            "entity_value_explicit",
            "entity_implicit_value_explicit",
            "entity_explicit_value_naturalized",
            "entity_value_implicit",
        ),
        None,
    )
    generated = (
        root
        / "_test_view/_query_views/one_per_relation_balanced_seed42/test_relations_all.json"
    )
    shutil.copy2(generated, shared / "test_relations_all.json")
    shutil.rmtree(root / "_test_view")

    # Freeze the all-Q1--Q4 test population used for the paper and analyses.
    records = []
    relation_ids = set()
    query_types = set()
    for qid in sorted(test.instances):
        row = test.instances[qid]
        records.append(
            {
                "relation_id": row["relation_id"],
                "query_id": qid,
                "query_type": row["query_type"],
                "query_variant": row.get("query_variant"),
            }
        )
        relation_ids.add(row["relation_id"])
        query_types.add(row["query_type"])
    four = {
        "split": "test",
        "selection": "all four formulations, no sampling",
        "relation_ids": sorted(relation_ids),
        "query_types": sorted(query_types),
        "records": records,
    }
    write_json(root / "four_query_manifest.json", four)

    train = load_split(data_root / "train", "train")
    train_manifest = json.loads((shared / "train_relations_all.json").read_text())
    sampled = []
    stats = Counter()
    for index, item in enumerate(train_manifest["records"]):
        qid = item["query_id"]
        row = train.instances[qid]
        qrels = train.qrels[qid]
        candidates = row["article_local_candidate_doc_ids"]
        direct = [d for d in candidates if qrels.get(d, 0) == 2]
        eligible = [d for d in candidates if qrels.get(d, 0) == 0]
        partial = [d for d in candidates if qrels.get(d, 0) == 1]
        if not direct or not eligible:
            sampled.append({"query_id": qid, "skip": "no_direct_or_negative"})
            stats["skipped"] += 1
            continue
        rng = random.Random(42 * 1_000_003 + index)
        positive = rng.choice(direct)
        n = min(3, len(eligible))
        negatives = rng.sample(eligible, n) if len(eligible) > n else list(eligible)
        sampled.append(
            {
                "query_id": qid,
                "relation_id": row["relation_id"],
                "query_type": row["query_type"],
                "positive_ids": [positive],
                "generic": [{"doc_id": d, "type": "random"} for d in negatives],
                "partial_ids": partial,
                "negative_count": n,
            }
        )
        stats["eligible"] += 1
        stats[f"negative_count_{n}"] += 1
    sampling = shared / "sampling_abcd.jsonl"
    with sampling.open("w") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(shared / "sampling_statistics.json", dict(stats))

    # The training program verifies these immutable inputs before every run.
    tracked = []
    for split in ("train", "dev", "test"):
        tracked.extend(data_root.joinpath(split).glob("*.jsonl"))
        tracked.append(data_root / split / "qrels.tsv")
    tracked.extend(shared.glob("*"))
    tracked.append(root / "four_query_manifest.json")
    tracked.extend((root / "scripts").rglob("*.py"))
    tracked.extend((project / "avr_retriever_experiment").glob("*.py"))
    tracked.extend(checkpoint.glob("*"))
    hashes = {}
    for path in tracked:
        if path.is_file():
            try:
                key = str(path.relative_to(root))
            except ValueError:
                key = str(path)
            hashes[key] = sha256(path)
    write_json(root / "input_hashes.json", hashes)
    write_json(
        root / "bundle_manifest.json",
        {
            "task": "deduplicated document-level evidence ranking",
            "query_view_seed": 42,
            "training_queries": train_manifest["query_count"],
            "test_relations": len(relation_ids),
            "test_queries": len(records),
            "sampling_sha256": sha256(sampling),
            "four_query_manifest_sha256": sha256(root / "four_query_manifest.json"),
            "data_manifest_sha256": sha256(data_root / "manifest.json"),
            "metric_semantics": "Direct-only Recall/MRR; Direct=2 Partial=1 graded nDCG",
        },
    )
    print(json.dumps(json.loads((root / "bundle_manifest.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
