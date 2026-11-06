#!/usr/bin/env python3
from __future__ import annotations

from quality_dataset_editor import QualityDatasetEdits, RevisionConflict, current_score, revision
import argparse
import json
import mimetypes
import os
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

ANNOTATOR_RE = re.compile(
    r"^(?=.{1,60}$)[A-Za-z0-9._-]+(?:[ ]+[A-Za-z0-9._-]+)*$"
)
GROUP_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
MAX_BODY_BYTES = 25 * 1024 * 1024
QUERY_TYPES = (
    "entity_value_explicit",
    "entity_implicit_value_explicit",
    "entity_explicit_value_naturalized",
    "entity_value_implicit",
)
QUERY_STATUS = ("keep", "surface_edit", "semantic_edit", "exclude")
STAGE1_DIMS = (
    "direct_evidence_validity",
    "partial_evidence_validity",
    "evidence_set_sufficiency",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def to_int(x):
    if x in (None, ""):
        return None
    try:
        v = int(x)
        return v if v in (0, 1, 2) else None
    except Exception:
        return None


def normalize_status(x) -> str:
    s = str(x or "keep").lower()
    if s == "edited":
        s = "semantic_edit"
    if s == "drop":
        s = "exclude"
    return s if s in QUERY_STATUS else "keep"


def parse_time(x: str) -> float:
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def score_summary(values):
    vals = [int(v) for v in values if v is not None]
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "strict_2_rate": None,
            "soft_ge1_rate": None,
            "counts": {"0": 0, "1": 0, "2": 0},
        }
    c = Counter(vals)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "strict_2_rate": sum(v == 2 for v in vals) / len(vals),
        "soft_ge1_rate": sum(v >= 1 for v in vals) / len(vals),
        "counts": {str(k): c.get(k, 0) for k in (0, 1, 2)},
    }


class AnnotationStore:
    """Append-only JSONL annotation store shared by all annotators."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    @staticmethod
    def _validate_identity(annotator: str, group: str) -> tuple[str, str]:
        annotator = str(annotator or "").strip()
        group = str(group or "").strip()
        if not ANNOTATOR_RE.fullmatch(annotator):
            raise ValueError("invalid annotator")
        if not GROUP_RE.fullmatch(group):
            raise ValueError("invalid group")
        return annotator, group

    def _append_records(self, records: list[dict]) -> None:
        if not records:
            return
        lines = "".join(
            json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n"
            for x in records
        )
        with self._thread_lock:
            with self.path.open("a", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(lines)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save_one(
        self,
        *,
        annotator: str,
        group: str,
        relation_id: str,
        state: dict,
        protocol: str = "",
        client_updated_at: str = "",
        source: str = "annotation_ui",
        admin_reason: str = "",
    ) -> dict:
        annotator, group = self._validate_identity(annotator, group)
        relation_id = str(relation_id or "").strip()
        if not relation_id:
            raise ValueError("relation_id is required")
        if not isinstance(state, dict):
            raise ValueError("state must be a JSON object")

        record = {
            "record_type": "annotation_state",
            "annotator": annotator,
            "group": group,
            "relation_id": relation_id,
            "client_updated_at": str(client_updated_at or state.get("_client_updated_at", "")),
            "server_saved_at": now_iso(),
            "protocol": str(protocol or ""),
            "source": str(source or "annotation_ui"),
            "state": state,
        }
        if admin_reason:
            record["admin_reason"] = str(admin_reason)
        self._append_records([record])
        return record

    def save_batch(self, *, annotator: str, group: str, states: dict, protocol: str = "") -> int:
        annotator, group = self._validate_identity(annotator, group)
        if not isinstance(states, dict):
            raise ValueError("states must be an object")
        records = []
        saved_at = now_iso()
        for rid, state in states.items():
            if not rid or not isinstance(state, dict):
                continue
            records.append(
                {
                    "record_type": "annotation_state",
                    "annotator": annotator,
                    "group": group,
                    "relation_id": str(rid),
                    "client_updated_at": str(state.get("_client_updated_at", "")),
                    "server_saved_at": saved_at,
                    "protocol": str(protocol or ""),
                    "source": "annotation_ui_batch",
                    "state": state,
                }
            )
        self._append_records(records)
        return len(records)

    def load_all_latest(self) -> dict[tuple[str, str, str], dict]:
        latest = {}
        with self._thread_lock:
            with self.path.open("r", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except Exception:
                            continue
                        if rec.get("record_type") != "annotation_state":
                            continue
                        key = (
                            str(rec.get("annotator", "")),
                            str(rec.get("group", "")),
                            str(rec.get("relation_id", "")),
                        )
                        if all(key) and isinstance(rec.get("state"), dict):
                            latest[key] = rec
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return latest

    def load_latest(self, *, annotator: str, group: str) -> dict:
        annotator, group = self._validate_identity(annotator, group)
        return {
            rid: rec
            for (a, g, rid), rec in self.load_all_latest().items()
            if a == annotator and g == group
        }


class BenchmarkIndex:
    """Original benchmark metadata used by the admin dashboard."""

    def __init__(self, verification_dir: Path, group_size: int = 25):
        self.verification_dir = verification_dir.resolve()
        self.group_size = group_size
        self.sample_path = self.verification_dir / "sample.jsonl"
        self.claude_path = self.verification_dir / "claude_annotations.jsonl"
        self.relations = {}
        self.relation_order = []
        self.group_to_relations = defaultdict(list)
        self.claude_latest = {}
        self.reload()

    def reload(self):
        self.relations = {}
        self.relation_order = []
        self.group_to_relations = defaultdict(list)
        rows = read_jsonl(self.sample_path)
        grouped = defaultdict(list)
        seen = set()
        for row in rows:
            rid = str(row.get("relation_id", ""))
            if not rid:
                continue
            grouped[rid].append(row)
            if rid not in seen:
                seen.add(rid)
                self.relation_order.append(rid)
        for idx, rid in enumerate(self.relation_order):
            rr = grouped[rid]
            first = rr[0]
            group = str(idx // self.group_size + 1)
            rr.sort(key=lambda x: QUERY_TYPES.index(x.get("query_type")) if x.get("query_type") in QUERY_TYPES else 999)
            relation = {
                "relation_id": rid,
                "group": group,
                "guid": str(first.get("guid", "")),
                "article_title": str(first.get("article_title", "")),
                "unit_level": str(first.get("unit_level", "")),
                "unit_id": str(first.get("unit_id", "")),
                "unit_text": str(first.get("unit_text", "")),
                "actor_name": str(first.get("actor_name", "")),
                "value_name": str(first.get("value_name", "")),
                "direction": str(first.get("direction", "")),
                "nevu_explanation": str(first.get("nevu_explanation", "")),
                "nevu_confidence": str(first.get("nevu_confidence", "")),
                "direct_doc_ids": [str(x) for x in first.get("direct_doc_ids", []) or []],
                "partial_doc_ids": [str(x) for x in first.get("partial_doc_ids", []) or []],
                "article_sentences": list(first.get("article_sentences", []) or []),
                "queries": [
                    {
                        "query_id": str(x.get("query_id", "")),
                        "query_type": str(x.get("query_type", "")),
                        "query": str(x.get("query", "")),
                    }
                    for x in rr
                ],
            }
            self.relations[rid] = relation
            self.group_to_relations[group].append(rid)

        self.claude_latest = {}
        for rec in read_jsonl(self.claude_path):
            rid = str(rec.get("relation_id", ""))
            if rid:
                self.claude_latest[rid] = rec

    def group_ids(self):
        return sorted(self.group_to_relations, key=lambda x: int(x) if x.isdigit() else x)

    def get_relation(self, rid: str):
        return self.relations.get(str(rid))

    def expected_count_for_groups(self, groups):
        return sum(len(self.group_to_relations.get(str(g), [])) for g in groups)


class AdminAnalytics:
    def __init__(self, store: AnnotationStore, benchmark: BenchmarkIndex, assignments: dict | None = None):
        self.store = store
        self.benchmark = benchmark
        self.assignments = assignments or {}

    def groups_for_user(self, user: str, observed_groups: list[str]) -> list[str]:
        raw = self.assignments.get(user, self.assignments.get("*"))
        if raw in ("all", "*"):
            groups = self.benchmark.group_ids()
        elif isinstance(raw, list):
            groups = [str(x) for x in raw]
        else:
            groups = list(observed_groups)
        return sorted(set(groups), key=lambda x: int(x) if x.isdigit() else 999999)

    @staticmethod
    def _stage1_scores(state: dict):
        snap = state.get("original_assessment_snapshot") or {}
        direct = to_int(snap.get("direct_evidence_validity", state.get("direct_evidence_validity")))
        partial = to_int(snap.get("partial_evidence_validity", state.get("partial_evidence_validity")))
        suff = to_int(snap.get("evidence_set_sufficiency", state.get("evidence_set_sufficiency")))
        q_scores = {}
        sq = snap.get("queries") or {}
        for qid, qstate in (state.get("queries") or {}).items():
            qsnap = sq.get(str(qid), {})
            sem = to_int(qsnap.get("semantic_faithfulness", qsnap.get("query_quality", qstate.get("semantic_faithfulness", qstate.get("query_quality")))))
            con = to_int(qsnap.get("query_evidence_consistency", qstate.get("query_evidence_consistency")))
            q_scores[str(qid)] = {"semantic": sem, "consistency": con}
        return direct, partial, suff, q_scores

    @staticmethod
    def _final_evidence(state: dict):
        return {
            str(k): str(v)
            for k, v in (state.get("evidence") or {}).items()
            if str(v) in {"direct", "partial"}
        }

    def _relation_corrections(self, relation: dict, state: dict):
        orig_d = set(relation.get("direct_doc_ids", []))
        orig_p = set(relation.get("partial_doc_ids", []))
        final = self._final_evidence(state)
        fin_d = {d for d, lab in final.items() if lab == "direct"}
        fin_p = {d for d, lab in final.items() if lab == "partial"}
        universe = {str(x.get("doc_id", "")) for x in relation.get("article_sentences", []) if x.get("doc_id") is not None}
        orig_none = universe - orig_d - orig_p
        return {
            "evidence_changed": orig_d != fin_d or orig_p != fin_p,
            "d_to_d": len(orig_d & fin_d),
            "d_to_p": len(orig_d & fin_p),
            "d_to_none": len(orig_d - fin_d - fin_p),
            "p_to_d": len(orig_p & fin_d),
            "p_to_p": len(orig_p & fin_p),
            "p_to_none": len(orig_p - fin_d - fin_p),
            "none_to_d": len(orig_none & fin_d),
            "none_to_p": len(orig_none & fin_p),
        }

    def _summarize_records(self, records: list[dict], expected_relations: int):
        dv, pv, es, qf, qc = [], [], [], [], []
        statuses = Counter()
        locked = 0
        evidence_changed = 0
        transitions = Counter()
        last_saved = ""
        last_ts = 0.0
        relation_rows = []

        for rec in records:
            state = rec.get("state") or {}
            rid = str(rec.get("relation_id", ""))
            relation = self.benchmark.get_relation(rid)
            if not relation:
                continue
            if state.get("locked"):
                locked += 1
                d, p, s, qs = self._stage1_scores(state)
                if d is not None:
                    dv.append(d)
                if p is not None:
                    pv.append(p)
                if s is not None:
                    es.append(s)
                for vals in qs.values():
                    if vals["semantic"] is not None:
                        qf.append(vals["semantic"])
                    if vals["consistency"] is not None:
                        qc.append(vals["consistency"])
            for q in (state.get("queries") or {}).values():
                statuses[normalize_status(q.get("status"))] += 1
            corr = self._relation_corrections(relation, state)
            if corr["evidence_changed"]:
                evidence_changed += 1
            for k, v in corr.items():
                if k != "evidence_changed":
                    transitions[k] += v
            ts = parse_time(rec.get("server_saved_at", ""))
            if ts >= last_ts:
                last_ts = ts
                last_saved = str(rec.get("server_saved_at", ""))
            relation_rows.append(
                {
                    "relation_id": rid,
                    "group": str(rec.get("group", relation.get("group", ""))),
                    "scope": relation.get("unit_level", ""),
                    "actor": relation.get("actor_name", ""),
                    "value": relation.get("value_name", ""),
                    "direction": relation.get("direction", ""),
                    "locked": bool(state.get("locked")),
                    "evidence_changed": corr["evidence_changed"],
                    "material_queries": sum(normalize_status(q.get("status")) in {"semantic_edit", "exclude"} for q in (state.get("queries") or {}).values()),
                    "surface_queries": sum(normalize_status(q.get("status")) == "surface_edit" for q in (state.get("queries") or {}).values()),
                    "updated_at": rec.get("server_saved_at", ""),
                }
            )

        touched = len(records)
        material = statuses["semantic_edit"] + statuses["exclude"]
        decided = sum(statuses.values())
        return {
            "expected_relations": expected_relations,
            "touched_relations": touched,
            "locked_relations": locked,
            "completion_rate": (locked / expected_relations) if expected_relations else None,
            "touch_rate": (touched / expected_relations) if expected_relations else None,
            "last_saved_at": last_saved,
            "stage1": {
                "direct_evidence_validity": score_summary(dv),
                "partial_evidence_validity": score_summary(pv),
                "evidence_set_sufficiency": score_summary(es),
                "query_semantic_faithfulness": score_summary(qf),
                "query_evidence_consistency": score_summary(qc),
            },
            "stage2": {
                "query_status_counts": {k: statuses.get(k, 0) for k in QUERY_STATUS},
                "material_query_count": material,
                "material_query_rate": (material / decided) if decided else None,
                "evidence_changed_relations": evidence_changed,
                "evidence_changed_rate": (evidence_changed / touched) if touched else None,
                "evidence_transitions": dict(transitions),
            },
            "relations": relation_rows,
        }

    def overview(self):
        latest = self.store.load_all_latest()
        by_user = defaultdict(list)
        all_records = []
        all_groups = set()
        for (annotator, group, _rid), rec in latest.items():
            by_user[annotator].append(rec)
            all_records.append(rec)
            if group.isdigit():
                all_groups.add(group)

        annotators = []
        configured_users = {str(x) for x in self.assignments.keys() if str(x) != "*"}
        users = sorted(set(by_user) | configured_users)
        for user in users:
            records = by_user.get(user, [])
            observed = sorted({str(r.get("group", "")) for r in records}, key=lambda x: int(x) if x.isdigit() else 999999)
            groups = self.groups_for_user(user, observed)
            expected = self.benchmark.expected_count_for_groups([g for g in groups if g.isdigit()])
            stats = self._summarize_records(records, expected)
            annotators.append(
                {
                    "annotator": user,
                    "groups": groups,
                    **{k: stats[k] for k in ("expected_relations", "touched_relations", "locked_relations", "completion_rate", "touch_rate", "last_saved_at")},
                    "material_query_rate": stats["stage2"]["material_query_rate"],
                    "evidence_changed_rate": stats["stage2"]["evidence_changed_rate"],
                }
            )

        overall_expected = len(self.benchmark.relations)
        overall = self._summarize_records(all_records, overall_expected)
        # Overall completion is annotation-instance completion, not unique relation completion.
        if annotators:
            denom = sum(x["expected_relations"] for x in annotators)
            num = sum(x["locked_relations"] for x in annotators)
            overall["annotation_instance_expected"] = denom
            overall["annotation_instance_locked"] = num
            overall["annotation_instance_completion_rate"] = num / denom if denom else None
        else:
            overall["annotation_instance_expected"] = 0
            overall["annotation_instance_locked"] = 0
            overall["annotation_instance_completion_rate"] = None

        return {
            "generated_at": now_iso(),
            "benchmark": {
                "relations": len(self.benchmark.relations),
                "groups": len(self.benchmark.group_to_relations),
                "queries": sum(len(r.get("queries", [])) for r in self.benchmark.relations.values()),
            },
            "annotator_count": len(annotators),
            "annotators": annotators,
            "overall": overall,
        }

    def annotator_detail(self, annotator: str):
        latest = self.store.load_all_latest()
        records = [rec for (a, _g, _rid), rec in latest.items() if a == annotator]
        observed = sorted({str(r.get("group", "")) for r in records}, key=lambda x: int(x) if x.isdigit() else 999999)
        groups = self.groups_for_user(annotator, observed)
        expected = self.benchmark.expected_count_for_groups([g for g in groups if g.isdigit()])
        stats = self._summarize_records(records, expected)
        group_rows = []
        for g in groups:
            gr = [r for r in records if str(r.get("group", "")) == g]
            g_expected = len(self.benchmark.group_to_relations.get(g, [])) if g.isdigit() else len(gr)
            gs = self._summarize_records(gr, g_expected)
            group_rows.append(
                {
                    "group": g,
                    "expected_relations": g_expected,
                    "touched_relations": gs["touched_relations"],
                    "locked_relations": gs["locked_relations"],
                    "completion_rate": gs["completion_rate"],
                    "last_saved_at": gs["last_saved_at"],
                }
            )
        stats["annotator"] = annotator
        stats["groups"] = group_rows
        return stats

    def relation_detail(self, annotator: str, group: str, relation_id: str):
        latest = self.store.load_all_latest()
        rec = latest.get((annotator, group, relation_id))
        if not rec:
            raise KeyError("annotation state not found")
        relation = self.benchmark.get_relation(relation_id)
        if not relation:
            raise KeyError("benchmark relation not found")
        return {
            "annotator": annotator,
            "group": group,
            "relation": relation,
            "fable": self.benchmark.claude_latest.get(relation_id, {}),
            "state": rec.get("state", {}),
            "meta": {
                "server_saved_at": rec.get("server_saved_at", ""),
                "client_updated_at": rec.get("client_updated_at", ""),
                "source": rec.get("source", ""),
                "protocol": rec.get("protocol", ""),
                "admin_reason": rec.get("admin_reason", ""),
            },
        }



class PageBenchmark(BenchmarkIndex):
    """Use deployed page membership, including non-contiguous group IDs."""
    def __init__(self, directory):
        self.directory = directory
        self.dataset_edits = QualityDatasetEdits(directory.parent) if directory.parent.name == "document_quality" else None
        self.reload()

    def reload(self):
        relations, order, groups, advice = {}, [], defaultdict(list), {}
        decoder = json.JSONDecoder()
        for path in sorted(self.directory.glob("group_*.html")):
            group = str(int(path.stem.split("_")[-1]))
            content = path.read_text(encoding="utf-8")
            match = re.search(r"const RELATIONS\s*=\s*", content)
            if not match:
                raise ValueError(f"Missing embedded relations: {path.name}")
            rows, _ = decoder.raw_decode(content[match.end():])
            for row in rows:
                relation = dict(row["original"])
                rid = relation["relation_id"]
                if rid in relations:
                    raise ValueError(f"Duplicate relation: {rid}")
                relation["group"] = group
                relations[rid] = relation
                order.append(rid)
                groups[group].append(rid)
                advice[rid] = row.get("claude", {})
        if self.dataset_edits:
            edits = self.dataset_edits.read()
            relations = {rid: self.dataset_edits.apply(r, edits) for rid, r in relations.items()}
        self.relations, self.relation_order = relations, order
        self.group_to_relations, self.claude_latest = groups, advice


class RemainingAdminStore:
    def __init__(self, directory):
        from latest_annotation_store import ShardedAnnotationStore
        self.writer = ShardedAnnotationStore(directory)
        self.path = directory

    def load_all_latest(self):
        latest = {}
        for path in self.path.glob("annotations_group_*_*_plus.jsonl"):
            for rec in read_jsonl(path):
                if rec.get("record_type") == "annotation_state" and isinstance(rec.get("state"), dict):
                    key = tuple(str(rec.get(k, "")) for k in ("annotator", "group", "relation_id"))
                    if all(key):
                        latest[key] = rec
        return latest

    def save_one(self, **kwargs):
        kwargs.pop("source", None)
        kwargs.pop("admin_reason", None)
        return self.writer.save_one(**kwargs)


def quality_complete(relation, state):
    def scored(value):
        return str(value) in {"0", "1", "2"}
    return (current_score(relation, state) and all(not relation.get(ids) or scored(state.get(field)) for ids, field in
                (("direct_doc_ids", "direct_evidence_validity"), ("partial_doc_ids", "partial_evidence_validity")))
            and scored(state.get("evidence_set_sufficiency"))
            and all(scored((state.get("queries", {}).get(q["query_id"], {})).get(field))
                    for q in relation["queries"] for field in ("query_quality", "query_evidence_consistency")))


class ModuleAnalytics(AdminAnalytics):
    def __init__(self, store, benchmark, module):
        super().__init__(store, benchmark)
        self.module = module

    def relation_detail(self, annotator, group, relation_id):
        result = super().relation_detail(annotator, group, relation_id)
        if self.module == "quality" and not current_score(result["relation"], result["state"]):
            result["meta"]["outdated_scores"] = True
            result["state"] = {"queries": {}, "_dataset_revision": revision(result["relation"]), "completed": False}
        return result

    def _relation_corrections(self, relation, state):
        if self.module == "quality" or not state.get("human_confirmed") or state.get("valid") is False:
            return {"evidence_changed": False}
        return super()._relation_corrections(relation, state)

    def _summarize_records(self, records, expected_relations):
        adapted = []
        for rec in records:
            relation = self.benchmark.get_relation(rec.get("relation_id"))
            if not relation or str(rec.get("group")) != str(relation["group"]):
                continue
            state = dict(rec.get("state") or {})
            state["locked"] = (quality_complete(relation, state) if self.module == "quality"
                               else state.get("human_confirmed") is True)
            if self.module == "remaining":
                # No Stage 1 scores in this module; only confirmed valid decisions count.
                for key in ("original_assessment_snapshot", "direct_evidence_validity", "partial_evidence_validity", "evidence_set_sufficiency"):
                    state.pop(key, None)
                state["queries"] = ({qid: {k: v for k, v in q.items() if k in {"status", "final_query"}}
                                     for qid, q in state.get("queries", {}).items()}
                                    if state["locked"] and state.get("valid") is not False else {})
            adapted.append({**rec, "state": state})
        result = super()._summarize_records(adapted, expected_relations)
        result["completed_relations"] = result["locked_relations"]
        result["module"] = self.module
        if self.module == "remaining":
            valid = sum(r["state"]["locked"] and r["state"].get("valid") is not False for r in adapted)
            result["invalid_relations"] = sum(r["state"]["locked"] and r["state"].get("valid") is False for r in adapted)
            result["stage2"]["evidence_changed_rate"] = result["stage2"]["evidence_changed_relations"] / valid if valid else None
        return result


def setup_modules(server, directory):
    from quality_annotation_store import QualityAnnotationStore
    server.modules = {"original": (server.store, server.benchmark, server.admin)}
    for module, folder in (("quality", "document_quality"), ("remaining", "document_annotation")):
        root = directory / folder
        benchmark = PageBenchmark(root / "pages")
        store = (RemainingAdminStore(root / "server_data") if module == "remaining" else
                 QualityAnnotationStore(root / "server_data" / "quality_annotations.jsonl"))
        server.modules[module] = (store, benchmark, ModuleAnalytics(store, benchmark, module))

class AdminServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler, *, pages_dir: Path, store: AnnotationStore, benchmark: BenchmarkIndex, admin_token: str, assignments: dict | None = None):
        super().__init__(server_address, handler)
        self.pages_dir = pages_dir.resolve()
        self.store = store
        self.benchmark = benchmark
        self.admin = AdminAnalytics(store, benchmark, assignments=assignments)
        self.admin_token = admin_token
        self.dashboard_file = None


class Handler(BaseHTTPRequestHandler):
    server_version = "NEVUAdminServer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}")

    def _send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n <= 0:
            raise ValueError("empty request body")
        if n > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        obj = json.loads(self.rfile.read(n).decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("JSON body must be an object")
        return obj

    def _admin_authorized(self) -> bool:
        token = self.server.admin_token
        if not token:
            return False
        header = self.headers.get("X-Admin-Token", "")
        auth = self.headers.get("Authorization", "")
        if header == token:
            return True
        if auth.startswith("Bearer ") and auth[7:] == token:
            return True
        return False

    def _require_admin(self) -> bool:
        if self._admin_authorized():
            return True
        if not self.server.admin_token:
            self._send_json({"ok": False, "error": "admin API disabled: configure NEVU_ADMIN_TOKEN or --admin-token"}, HTTPStatus.SERVICE_UNAVAILABLE)
        else:
            self._send_json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
        return False

    def _module_context(self, parsed):
        module = (parse_qs(parsed.query).get("module") or ["original"])[0]
        if module not in self.server.modules:
            raise ValueError("Unknown annotation module")
        return (module,) + self.server.modules[module]

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/admin/dataset/save":
                if not self._require_admin(): return
                module, store, benchmark, admin = self._module_context(parsed)
                if module != "quality": raise ValueError("Only module 02 supports dataset editing")
                result = benchmark.dataset_edits.save(group=body.get("group"), relation_id=body.get("relation_id"),
                    expected_revision=body.get("expected_revision"), changes=body.get("changes"), reason=body.get("reason"))
                if result["changed"]: benchmark.reload()
                self._send_json({"ok":True,"data":result})
                return
            if parsed.path == "/api/admin/save-state":
                if not self._require_admin():
                    return
                module, store, benchmark, admin = self._module_context(parsed)
                annotator = str(body.get("annotator", ""))
                group = str(body.get("group", ""))
                rid = str(body.get("relation_id", ""))
                state = body.get("state")
                reason = str(body.get("reason", "")).strip()
                if not reason:
                    raise ValueError("admin edit reason is required")
                if not isinstance(state, dict):
                    raise ValueError("state is required")
                relation = benchmark.get_relation(rid)
                if not relation or str(relation["group"]) != str(group):
                    raise ValueError("Relation does not belong to this module/group")
                if module == "quality":
                    benchmark.dataset_edits.validate_submission(group, {rid: state})
                    state["completed"] = quality_complete(relation, state)
                state = json.loads(json.dumps(state))
                state["_client_updated_at"] = now_iso()
                history = state.get("_admin_edit_history")
                if not isinstance(history, list):
                    history = []
                history.append({"edited_at": now_iso(), "reason": reason})
                state["_admin_edit_history"] = history[-50:]
                rec = store.save_one(
                    annotator=annotator,
                    group=group,
                    relation_id=rid,
                    state=state,
                    protocol="admin_dashboard_edit",
                    client_updated_at=state["_client_updated_at"],
                    source="admin_dashboard",
                    admin_reason=reason,
                )
                self._send_json({"ok": True, "server_saved_at": rec["server_saved_at"]})
                return

            self._send_json({"ok": False, "error": "unknown API endpoint"}, HTTPStatus.NOT_FOUND)
        except RevisionConflict as exc:
            self._send_json({"ok":False,"error":str(exc)}, HTTPStatus.CONFLICT)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"server error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json({"ok": True, "data_file": str(self.server.store.path), "relations": len(self.server.benchmark.relations)})
            return

        if parsed.path.startswith("/api/admin/"):
            if not self._require_admin():
                return
            qs = parse_qs(parsed.query)
            try:
                module, store, benchmark, admin = self._module_context(parsed)
                if parsed.path.startswith("/api/admin/dataset/"):
                    if module != "quality": raise ValueError("Only module 02 supports dataset editing")
                    if parsed.path == "/api/admin/dataset/relation":
                        rid = (qs.get("relation_id") or [""])[0]
                        relation = benchmark.get_relation(rid)
                        if not relation: raise KeyError("Relation not found")
                        history = benchmark.dataset_edits.read()["relations"].get(rid, {}).get("history", [])
                        self._send_json({"ok":True,"data":{"relation":relation,"history":[{"edited_at":h["edited_at"],"reason":h["reason"]} for h in history]}})
                        return
                    if parsed.path == "/api/admin/dataset/export":
                        self._send_json({"ok":True,"data":{"dataset":"document_quality_effective","exported_at":now_iso(),"relations":list(benchmark.relations.values())}})
                        return
                    if parsed.path == "/api/admin/dataset/list":
                        group = (qs.get("group") or [""])[0]
                        search = (qs.get("q") or [""])[0].casefold().strip()
                        page = max(1, int((qs.get("page") or ["1"])[0]))
                        rows = [r for r in benchmark.relations.values() if (not group or str(r["group"])==group)
                                and (not search or search in json.dumps(r,ensure_ascii=False).casefold())]
                        rows.sort(key=lambda r:(int(r["group"]),r["relation_id"]))
                        self._send_json({"ok":True,"data":{"total":len(rows),"page":page,"groups":benchmark.group_ids(),
                            "rows":[{k:r.get(k,"") for k in ("group","relation_id","guid","actor_name","value_name","unit_level","_dataset_revision")}
                                    for r in rows[(page-1)*50:page*50]]}})
                        return
                if parsed.path == "/api/admin/overview":
                    self._send_json({"ok": True, "data": admin.overview()})
                    return
                if parsed.path == "/api/admin/annotator":
                    annotator = (qs.get("annotator") or [""])[0]
                    self._send_json({"ok": True, "data": admin.annotator_detail(annotator)})
                    return
                if parsed.path == "/api/admin/relation":
                    annotator = (qs.get("annotator") or [""])[0]
                    group = (qs.get("group") or [""])[0]
                    rid = (qs.get("relation_id") or [""])[0]
                    self._send_json({"ok": True, "data": admin.relation_detail(annotator, group, rid)})
                    return
                if parsed.path == "/api/admin/reload-benchmark":
                    benchmark.reload()
                    self._send_json({"ok": True, "relations": len(benchmark.relations)})
                    return
                self._send_json({"ok": False, "error": "unknown admin endpoint"}, HTTPStatus.NOT_FOUND)
            except KeyError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self._serve_static(parsed.path)

    def _serve_static(self, url_path: str):
        # Standalone admin server intentionally serves only the admin dashboard.
        clean = unquote(url_path).split("?", 1)[0]
        if clean not in {"/", "/admin_dashboard.html", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        candidate = self.server.dashboard_file
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(description="Standalone NEVU admin dashboard server. It does not replace or serve the live annotation UI.")
    ap.add_argument("--dashboard", type=Path, default=None, help="Path to admin_dashboard.html. Default: <script_dir>/admin_dashboard.html")
    ap.add_argument("--data-file", type=Path, default=None)
    ap.add_argument("--verification-dir", type=Path, default=None)
    ap.add_argument("--assignments", type=Path, default=None, help="Optional JSON mapping annotator -> group list, e.g. {\"alice\":[1,2],\"bob\":\"all\"}.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--admin-token", default=os.environ.get("NEVU_ADMIN_TOKEN", ""))
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    verification_dir = (
        args.verification_dir.resolve() if args.verification_dir
        else (script_dir / "example_data").resolve()
    )
    dashboard_file = args.dashboard.resolve() if args.dashboard else (script_dir / "admin_dashboard.html").resolve()
    data_file = args.data_file.resolve() if args.data_file else (verification_dir / "server_data" / "annotations.jsonl").resolve()

    if not dashboard_file.exists():
        raise FileNotFoundError(f"Admin dashboard HTML does not exist: {dashboard_file}")
    if not (verification_dir / "sample.jsonl").exists():
        raise FileNotFoundError(f"Missing sample.jsonl under: {verification_dir}")

    assignments = {}
    if args.assignments:
        assignments_path = args.assignments.resolve()
        if not assignments_path.exists():
            raise FileNotFoundError(f"Assignments file does not exist: {assignments_path}")
        assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
        if not isinstance(assignments, dict):
            raise ValueError("Assignments JSON must be an object mapping annotator to group list or 'all'.")

    store = AnnotationStore(data_file)
    benchmark = BenchmarkIndex(verification_dir)
    server = AdminServer((args.host, args.port), Handler, pages_dir=dashboard_file.parent, store=store, benchmark=benchmark, admin_token=args.admin_token, assignments=assignments)
    setup_modules(server, verification_dir)
    server.dashboard_file = dashboard_file

    print("=== NEVU Standalone Admin Server ===")
    print("Verification:", verification_dir)
    print("Dashboard   :", dashboard_file)
    print("Data file   :", data_file)
    print("Relations   :", len(benchmark.relations))
    print("URL         :", f"http://{args.host}:{args.port}/")
    print("Admin API   :", "enabled" if args.admin_token else "DISABLED (set NEVU_ADMIN_TOKEN)")
    print("Assignments :", str(args.assignments.resolve()) if args.assignments else "not configured; completion uses observed groups")
    print("Live annotation server is NOT touched by this process.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
