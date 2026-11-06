from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from avr_retriever_experiment.data import BenchmarkData
from avr_retriever_experiment.support.io import read_json, write_json


DEFAULT_QUERY_TYPES: Tuple[str, ...] = (
    "entity_value_explicit",
    "entity_implicit_value_explicit",
    "entity_explicit_value_naturalized",
    "entity_value_implicit",
)


def _relation_id(data: BenchmarkData, qid: str) -> str:
    inst = data.instances[qid]
    return str(inst.get("relation_id") or qid)


def _query_type(data: BenchmarkData, qid: str) -> str:
    return str(data.instances[qid].get("query_type", "unknown"))


def _has_direct(data: BenchmarkData, qid: str) -> bool:
    return any(int(r) >= 2 for r in data.qrels.get(qid, {}).values())


def _group_relation_variants(
    data: BenchmarkData,
    query_types: Sequence[str],
) -> Dict[str, Dict[str, List[str]]]:
    allowed = set(query_types)
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for qid in sorted(data.query_ids):
        qtype = _query_type(data, qid)
        if qtype not in allowed:
            continue
        if not _has_direct(data, qid):
            # A retrieval relation without Direct evidence is invalid for the
            # current binary R@k/MRR and train-positive definition. Skip the
            # bad variant; normally all four variants share the same qrels.
            continue
        grouped[_relation_id(data, qid)][qtype].append(qid)

    return grouped


def _balanced_one_per_relation(
    data: BenchmarkData,
    seed: int,
    query_types: Sequence[str],
    max_relations: Optional[int],
) -> tuple[List[str], List[dict]]:
    """Select exactly one available query per relation.

    Relations are shuffled once using `seed`. For each relation we choose the
    currently least-used available query type, with deterministic random
    tie-breaking. This keeps the four formulations as balanced as possible
    without inventing missing variants.
    """

    grouped = _group_relation_variants(data, query_types)
    relation_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(relation_ids)

    if max_relations is not None and max_relations > 0:
        relation_ids = relation_ids[: min(max_relations, len(relation_ids))]

    type_counts = Counter({qt: 0 for qt in query_types})
    chosen: List[str] = []
    records: List[dict] = []

    for rid in relation_ids:
        available_types = [qt for qt in query_types if grouped[rid].get(qt)]
        if not available_types:
            continue

        min_count = min(type_counts[qt] for qt in available_types)
        tied = [qt for qt in available_types if type_counts[qt] == min_count]
        qtype = rng.choice(tied)
        qid = rng.choice(sorted(grouped[rid][qtype]))

        chosen.append(qid)
        type_counts[qtype] += 1
        records.append(
            {
                "relation_id": rid,
                "query_id": qid,
                "query_type": qtype,
                "available_query_types": available_types,
                "guid": str(data.instances[qid].get("guid", "")),
                "unit_level": str(data.instances[qid].get("unit_level", "")),
            }
        )

    # Stable ordering for downstream reproducibility independent of Python dict
    # insertion order. The manifest retains the actual selection records.
    chosen = sorted(chosen)
    records = sorted(records, key=lambda x: x["relation_id"])
    return chosen, records


def _manifest_path(
    results_dir: Path,
    split: str,
    mode: str,
    seed: int,
    max_relations: Optional[int],
) -> Path:
    suffix = (
        "all" if max_relations is None or max_relations <= 0 else str(max_relations)
    )
    return (
        results_dir
        / "_query_views"
        / f"{mode}_seed{seed}"
        / f"{split}_relations_{suffix}.json"
    )


def select_experiment_query_ids(
    data: BenchmarkData,
    results_dir: Path,
    mode: str,
    seed: int,
    reuse: bool,
    query_types: Sequence[str] = DEFAULT_QUERY_TYPES,
    max_relations: Optional[int] = None,
) -> List[str]:
    """Return the frozen query IDs for one official split.

    Main-paper mode is `one_per_relation_balanced`. A manifest is always
    written and, when `reuse=True`, reused across BM25/E5/BGE-M3/Generic FT/
    across different training seeds.
    """

    if mode != "one_per_relation_balanced":
        raise ValueError(
            f"Unknown query-view mode={mode!r}. " "Use 'one_per_relation_balanced'."
        )

    manifest = _manifest_path(results_dir, data.split, mode, seed, max_relations)

    valid_ids = set(data.query_ids)
    if reuse and manifest.exists():
        saved = read_json(manifest, {}) or {}
        records = list(saved.get("records", []) or [])
        chosen = [str(x.get("query_id", "")) for x in records]

        # Validate the manifest before silently reusing it.
        if chosen and all(qid in valid_ids for qid in chosen):
            relation_ids = [_relation_id(data, qid) for qid in chosen]
            if len(relation_ids) == len(set(relation_ids)):
                return sorted(chosen)

    chosen, records = _balanced_one_per_relation(
        data=data,
        seed=seed,
        query_types=query_types,
        max_relations=max_relations,
    )

    if not chosen:
        raise RuntimeError(
            f"No eligible relation/query pairs selected for split={data.split}."
        )

    qtype_counts = Counter(_query_type(data, qid) for qid in chosen)
    unit_counts = Counter(
        str(data.instances[qid].get("unit_level", "unknown")) for qid in chosen
    )

    write_json(
        manifest,
        {
            "split": data.split,
            "selection_policy": mode,
            "seed": seed,
            "max_relations": max_relations,
            "query_types": list(query_types),
            "relation_count": len(chosen),
            "query_count": len(chosen),
            "query_type_counts": dict(sorted(qtype_counts.items())),
            "unit_level_counts": dict(sorted(unit_counts.items())),
            "records": records,
        },
    )
    return sorted(chosen)
