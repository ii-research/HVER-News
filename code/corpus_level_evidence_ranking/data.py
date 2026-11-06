from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["query_id"]][row["doc_id"]] = int(row["relevance"])
    return dict(out)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CorpusData:
    root: Path
    split: str
    queries: dict[str, dict]
    articles: dict[str, dict]
    sentences: dict[str, dict]
    article_sentence_ids: dict[str, list[str]]
    sentence_article: dict[str, str]
    instances: dict[str, list[dict]]
    document_qrels: dict[str, dict[str, int]]
    sentence_qrels: dict[str, dict[str, int]]

    @classmethod
    def load(cls, data_root: Path | str, split: str) -> "CorpusData":
        root = Path(data_root) / split
        required = (
            "queries.jsonl",
            "articles.jsonl",
            "corpus.jsonl",
            "instances.jsonl",
            "qrels.documents.tsv",
            "qrels.sentences.tsv",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{root}: missing {missing}")

        queries = {row["query_id"]: row for row in read_jsonl(root / "queries.jsonl")}
        articles = {row["doc_id"]: row for row in read_jsonl(root / "articles.jsonl")}
        sentences = {row["doc_id"]: row for row in read_jsonl(root / "corpus.jsonl")}
        article_sentence_ids: dict[str, list[str]] = defaultdict(list)
        sentence_article: dict[str, str] = {}
        for sid, row in sentences.items():
            aid = str(row.get("guid") or sid.rsplit("::s", 1)[0])
            article_sentence_ids[aid].append(sid)
            sentence_article[sid] = aid
        for ids in article_sentence_ids.values():
            ids.sort(key=lambda x: int(sentences[x].get("sentence_id", 0)))

        instances: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(root / "instances.jsonl"):
            instances[row["query_id"]].append(row)

        data = cls(
            root=root,
            split=split,
            queries=queries,
            articles=articles,
            sentences=sentences,
            article_sentence_ids=dict(article_sentence_ids),
            sentence_article=sentence_article,
            instances=dict(instances),
            document_qrels=read_qrels(root / "qrels.documents.tsv"),
            sentence_qrels=read_qrels(root / "qrels.sentences.tsv"),
        )
        data.validate()
        return data

    def validate(self) -> None:
        qids = set(self.queries)
        article_ids = set(self.articles)
        sentence_ids = set(self.sentences)
        for name, values in (
            ("instances", self.instances),
            ("document_qrels", self.document_qrels),
            ("sentence_qrels", self.sentence_qrels),
        ):
            extra = set(values) - qids
            if extra:
                raise ValueError(f"{self.split}/{name}: {len(extra)} unknown query IDs")
        for qid, qrels in self.document_qrels.items():
            unknown = set(qrels) - article_ids
            if unknown:
                raise ValueError(
                    f"{self.split}/{qid}: unknown article IDs: {list(unknown)[:3]}"
                )
        for qid, qrels in self.sentence_qrels.items():
            unknown = set(qrels) - sentence_ids
            if unknown:
                raise ValueError(
                    f"{self.split}/{qid}: unknown sentence IDs: {list(unknown)[:3]}"
                )

    def source_article(self, qid: str) -> str:
        return str(
            self.queries[qid].get("source_article_id") or self.queries[qid].get("guid")
        )

    def manifest(self) -> dict:
        files = sorted(self.root.glob("*"))
        unresolved = sum(
            1
            for rows in self.instances.values()
            for row in rows
            if row.get("document_relevance") is None
        )
        return {
            "root": str(self.root.resolve()),
            "split": self.split,
            "queries": len(self.queries),
            "articles": len(self.articles),
            "sentences": len(self.sentences),
            "instances": sum(map(len, self.instances.values())),
            "unresolved_instances_skipped": unresolved,
            "sha256": {p.name: sha256(p) for p in files if p.is_file()},
        }


def balanced_one_per_relation(data: CorpusData, seed: int = 42) -> list[str]:
    """Choose one Q1--Q4 query for each relation, balanced by query type."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid, row in data.queries.items():
        grouped[str(row["relation_id"])].append(qid)
    rng = random.Random(seed)
    relation_ids = sorted(grouped)
    rng.shuffle(relation_ids)
    counts: dict[str, int] = defaultdict(int)
    chosen: list[str] = []
    for relation_id in relation_ids:
        candidates = grouped[relation_id]
        min_count = min(counts[str(data.queries[q]["query_type"])] for q in candidates)
        pool = [
            q
            for q in candidates
            if counts[str(data.queries[q]["query_type"])] == min_count
        ]
        qid = rng.choice(sorted(pool))
        chosen.append(qid)
        counts[str(data.queries[qid]["query_type"])] += 1
    return sorted(chosen)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
