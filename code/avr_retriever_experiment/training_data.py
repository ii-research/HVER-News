from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from avr_retriever_experiment.data import BenchmarkData
from avr_retriever_experiment.negative_mining import RelationNegativeMiner


@dataclass(frozen=True)
class TrainingRecipe:
    """Ordinary-negative Generic FT; Partial is excluded from supervision."""

    name: str = "generic"


class RetrievalTrainingDataset(Dataset):
    def __init__(
        self,
        data: BenchmarkData,
        query_ids: Sequence[str],
        recipe: TrainingRecipe,
        seed: int,
        generic_negatives_per_query: int,
    ):
        self.data = data
        self.query_ids = list(query_ids)
        self.recipe = recipe
        self.seed = seed
        self.generic_k = generic_negatives_per_query
        self.miner = RelationNegativeMiner(data, seed=seed)

    def __len__(self) -> int:
        return len(self.query_ids)

    def _sample(self, xs: List[str], k: int, rng: random.Random) -> List[str]:
        if not xs or k <= 0:
            return []
        if len(xs) <= k:
            return list(xs)
        return rng.sample(xs, k)

    def _rng(self, idx: int) -> random.Random:
        return random.Random(self.seed * 1000003 + idx)

    def sampling_plan(self, idx: int) -> dict:
        """Return the exact positive and explicit-negative IDs for an item."""
        qid = self.query_ids[idx]
        rng = self._rng(idx)
        qrels = self.data.qrels[qid]
        direct = [d for d, r in qrels.items() if r >= 2 and d in self.data.corpus]
        if not direct:
            raise ValueError(f"Query {qid} has no direct evidence (rel>=2)")
        pos_id = rng.choice(direct)
        pool = self.miner.build_pool(qid)

        negatives: List[Tuple[str, str]] = []
        for did in self._sample(pool.generic, self.generic_k, rng):
            negatives.append((did, "generic"))
        return {"query_id": qid, "positive_id": pos_id, "negatives": negatives}

    def __getitem__(self, idx: int) -> dict:
        qid = self.query_ids[idx]
        inst = self.data.instances[qid]
        plan = self.sampling_plan(idx)
        pos_id = plan["positive_id"]
        pool = self.miner.build_pool(qid)
        partial_id = None
        return {
            "query_id": qid,
            "relation_id": str(inst.get("relation_id") or qid),
            "query": self.data.queries[qid]["query"],
            "positive_id": pos_id,
            "positive": str(self.data.corpus[pos_id].get("text", "")),
            "negatives": [
                {
                    "doc_id": did,
                    "type": typ,
                    "text": str(self.data.corpus[did].get("text", "")),
                }
                for did, typ in plan["negatives"]
                if did in self.data.corpus
            ],
            "partial_id": partial_id,
            "partial": (
                str(self.data.corpus[partial_id].get("text", ""))
                if partial_id
                else None
            ),
            "query_type": inst.get("query_type", "unknown"),
            "unit_level": inst.get("unit_level", "unknown"),
        }


def collate_training(batch: Sequence[dict]) -> dict:
    return {
        "query_ids": [x["query_id"] for x in batch],
        "relation_ids": [x["relation_id"] for x in batch],
        "queries": [x["query"] for x in batch],
        "positives": [x["positive"] for x in batch],
        "negative_lists": [x["negatives"] for x in batch],
        "partials": [x["partial"] for x in batch],
    }
