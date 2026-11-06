from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from document_level_evidence_ranking.support.io import read_jsonl


@dataclass
class BenchmarkData:
    split: str
    corpus: Dict[str, Dict[str, Any]]
    queries: Dict[str, Dict[str, Any]]
    instances: Dict[str, Dict[str, Any]]
    qrels: Dict[str, Dict[str, int]]

    @property
    def query_ids(self) -> List[str]:
        return [
            qid for qid in self.queries if qid in self.instances and qid in self.qrels
        ]


def load_split(base: Path, split: str) -> BenchmarkData:
    required = ["corpus.jsonl", "queries.jsonl", "instances.jsonl", "qrels.tsv"]
    missing = [x for x in required if not (base / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase-5 files under {base}: {missing}")

    corpus = {str(x["doc_id"]): x for x in read_jsonl(base / "corpus.jsonl")}
    queries = {}
    for x in read_jsonl(base / "queries.jsonl"):
        row = dict(x)
        row["query"] = str(row.get("query") or row.get("text") or "").strip()
        queries[str(row["query_id"])] = row
    instances = {str(x["query_id"]): x for x in read_jsonl(base / "instances.jsonl")}

    qrels: Dict[str, Dict[str, int]] = {}
    with (base / "qrels.tsv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(
                row["relevance"]
            )

    return BenchmarkData(
        split=split, corpus=corpus, queries=queries, instances=instances, qrels=qrels
    )


def candidate_doc_ids(data: BenchmarkData, query_id: str, scope: str) -> List[str]:
    if scope == "corpus":
        return list(data.corpus.keys())
    if scope != "article_local":
        raise ValueError(f"Unknown retrieval scope: {scope}")

    inst = data.instances[query_id]
    ids = [
        str(x)
        for x in inst.get("article_local_candidate_doc_ids", [])
        if str(x) in data.corpus
    ]
    if ids:
        return ids
    guid = str(inst.get("guid", ""))
    ids = [did for did, row in data.corpus.items() if str(row.get("guid", "")) == guid]
    if not ids:
        raise ValueError(f"No article-local candidates for {query_id}")
    return ids
