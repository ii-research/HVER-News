from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from avr_retriever_experiment.data import BenchmarkData


@dataclass
class NegativePool:
    actor: List[str]
    value: List[str]
    direction: List[str]
    generic: List[str]
    partial: List[str]


class RelationNegativeMiner:
    """Mine negatives from NEVU relation metadata.

    Structured negatives are deliberately limited to three interpretable types:
      * actor: same value/direction, different actor
      * value: same actor/direction, different value
      * direction: same actor/value, opposite direction

    rel=1 partial evidence is NOT treated as a hard negative; it is reserved for
    graded evidence learning (direct > partial > wrong relation).
    """

    def __init__(self, data: BenchmarkData, seed: int = 42):
        self.data = data
        self.seed = seed
        self.by_guid = defaultdict(list)
        seen_relation = set()
        for qid, inst in data.instances.items():
            rid = str(inst.get("relation_id") or qid)
            if rid in seen_relation:
                continue
            seen_relation.add(rid)
            self.by_guid[str(inst.get("guid", ""))].append(inst)

    @staticmethod
    def _actor(inst: dict) -> str:
        return str(inst.get("actor_id") or inst.get("actor_name") or "")

    @staticmethod
    def _value(inst: dict) -> str:
        return str(
            inst.get("value_id")
            if inst.get("value_id") is not None
            else inst.get("value_name", "")
        )

    @staticmethod
    def _direction(inst: dict) -> str:
        return str(inst.get("direction") or "").lower()

    def build_pool(self, query_id: str) -> NegativePool:
        inst = self.data.instances[query_id]
        qrels = self.data.qrels.get(query_id, {})
        direct = {d for d, r in qrels.items() if r >= 2}
        partial = [d for d, r in qrels.items() if r == 1]
        excluded: Set[str] = set(direct) | set(partial)

        actor, value, direction = (
            self._actor(inst),
            self._value(inst),
            self._direction(inst),
        )
        actor_negs, value_negs, direction_negs = [], [], []

        for other in self.by_guid.get(str(inst.get("guid", "")), []):
            if str(other.get("relation_id")) == str(inst.get("relation_id")):
                continue
            odocs = [
                str(x)
                for x in other.get("direct_doc_ids", [])
                if str(x) in self.data.corpus
            ]
            if not odocs:
                continue
            oa, ov, od = self._actor(other), self._value(other), self._direction(other)

            # Isolate the changed semantic factor whenever possible.
            if oa != actor and ov == value and od == direction:
                actor_negs.extend(odocs)
            if oa == actor and ov != value and od == direction:
                value_negs.extend(odocs)
            if oa == actor and ov == value and od != direction:
                direction_negs.extend(odocs)

        def uniq(xs: Sequence[str]) -> List[str]:
            out, seen = [], set(excluded)
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        actor_negs = uniq(actor_negs)
        value_negs = uniq(value_negs)
        direction_negs = uniq(direction_negs)

        candidate_ids = [
            str(x)
            for x in inst.get("article_local_candidate_doc_ids", [])
            if str(x) in self.data.corpus and str(x) not in excluded
        ]
        structured = set(actor_negs) | set(value_negs) | set(direction_negs)
        generic = [x for x in candidate_ids if x not in structured]

        return NegativePool(
            actor=actor_negs,
            value=value_negs,
            direction=direction_negs,
            generic=generic,
            partial=[x for x in partial if x in self.data.corpus],
        )

    def statistics(self, query_ids: Sequence[str]) -> dict:
        rows = [self.build_pool(qid) for qid in query_ids]
        n = max(len(rows), 1)
        return {
            "queries": len(rows),
            "actor_negative_available_rate": sum(bool(x.actor) for x in rows) / n,
            "value_negative_available_rate": sum(bool(x.value) for x in rows) / n,
            "direction_negative_available_rate": sum(bool(x.direction) for x in rows)
            / n,
            "partial_evidence_available_rate": sum(bool(x.partial) for x in rows) / n,
            "avg_actor_negatives": sum(len(x.actor) for x in rows) / n,
            "avg_value_negatives": sum(len(x.value) for x in rows) / n,
            "avg_direction_negatives": sum(len(x.direction) for x in rows) / n,
            "avg_partial_evidence": sum(len(x.partial) for x in rows) / n,
        }
