"""Run an OpenAI LLM document-level evidence ranking baseline on the full Q1--Q4 test.

This uses the Batch API so 22,156 independent Responses requests can be
submitted cheaply and resumed safely.  The model may only rank candidate
sentence IDs supplied in the prompt; it cannot generate evidence text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from document_level_evidence_ranking.data import candidate_doc_ids, load_split
from document_level_evidence_ranking.sampling import DEFAULT_QUERY_TYPES
from document_level_evidence_ranking.support.io import write_json, write_jsonl


from document_level_evidence_ranking.api_prompts import (SYSTEM_PROMPT, OUTPUT_SCHEMA, format_ranking_prompt, ranked_sentence_ids, resolve_sentence_ids, ranking_for_evaluation, PROMPT_FINGERPRINT)


def build_complete_four_query_view(data) -> tuple[list[str], list[dict], dict]:
    """Freeze the same complete Q1--Q4 view as the formal full-test runner."""
    grouped = defaultdict(lambda: defaultdict(list))
    for qid in sorted(data.query_ids):
        instance = data.instances[qid]
        query_type = str(instance.get("query_type", "unknown"))
        if query_type not in DEFAULT_QUERY_TYPES:
            continue
        if not any(
            int(relevance) >= 2 for relevance in data.qrels.get(qid, {}).values()
        ):
            continue
        relation_id = str(instance.get("relation_id") or qid)
        grouped[relation_id][query_type].append(qid)
    records, query_ids, failures = [], [], Counter()
    excluded_incomplete = 0
    for relation_id in sorted(grouped):
        variants = grouped[relation_id]
        if not all(variants.get(query_type) for query_type in DEFAULT_QUERY_TYPES):
            excluded_incomplete += 1
            continue
        selected = {
            query_type: sorted(variants[query_type])[0]
            for query_type in DEFAULT_QUERY_TYPES
        }
        qids = list(selected.values())
        reference_qrels = data.qrels[qids[0]]
        reference_candidates = list(
            map(str, data.instances[qids[0]].get("article_local_candidate_doc_ids", []))
        )
        reference_guid = str(data.instances[qids[0]].get("guid", ""))
        for qid in qids[1:]:
            if data.qrels[qid] != reference_qrels:
                failures["qrels_mismatch"] += 1
            if (
                list(
                    map(
                        str,
                        data.instances[qid].get("article_local_candidate_doc_ids", []),
                    )
                )
                != reference_candidates
            ):
                failures["candidate_set_or_order_mismatch"] += 1
            if str(data.instances[qid].get("guid", "")) != reference_guid:
                failures["article_guid_mismatch"] += 1
        records.append(
            {
                "relation_id": relation_id,
                "guid": reference_guid,
                "query_ids": selected,
                "direct_doc_ids": sorted(
                    doc for doc, rel in reference_qrels.items() if rel >= 2
                ),
                "partial_doc_ids": sorted(
                    doc for doc, rel in reference_qrels.items() if rel == 1
                ),
                "candidate_count": len(reference_candidates),
            }
        )
        query_ids.extend(qids)
    audit = {
        "selection_policy": "complete_four_query_types_per_relation",
        "query_types": list(DEFAULT_QUERY_TYPES),
        "relation_count": len(records),
        "query_count": len(query_ids),
        "queries_per_relation": 4,
        "query_type_counts": dict(
            Counter(str(data.instances[qid].get("query_type")) for qid in query_ids)
        ),
        "excluded_incomplete_relations": excluded_incomplete,
        "pairing_failure_counts": dict(failures),
        "strict_pairing_passed": not failures,
    }
    if failures:
        raise RuntimeError(f"Four-query pairing audit failed: {dict(failures)}")
    return sorted(query_ids), records, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prompt_for(data, qid: str, candidates: list[str]) -> str:
    return format_ranking_prompt(data.queries[qid]['query'], candidates, data.corpus)


def make_request(model: str, qid: str, prompt: str, max_output_tokens: int) -> dict:
    return {
        "custom_id": qid,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "reasoning": {"effort": "none"},
            "max_output_tokens": max_output_tokens,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "evidence_ranking",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },
        },
    }


def extract_output_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    texts = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(
                content.get("text"), str
            ):
                texts.append(content["text"])
    if not texts:
        raise ValueError("Responses body contains no output_text")
    return "".join(texts)


def prepare(args) -> None:
    data = load_split(args.project_root.parent / "dataset/document-level/test", "test")
    query_ids, records, audit = build_complete_four_query_view(data)
    selection_source = "full_complete_q1_q4_view"
    if args.match_predictions:
        matched = [row["query_id"] for row in read_jsonl(args.match_predictions)]
        unknown = sorted(set(matched) - set(query_ids))
        if unknown:
            raise ValueError(
                f"match-predictions contains {len(unknown)} unknown query IDs"
            )
        if len(matched) != len(set(matched)):
            raise ValueError("match-predictions contains duplicate query IDs")
        query_ids = matched
        selection_source = str(args.match_predictions)
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]
        selection_source += f"; first_{args.max_queries}"
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "batch_inputs").mkdir()
    candidates_by_query = {
        qid: candidate_doc_ids(data, qid, "article_local") for qid in query_ids
    }
    shard_paths = []
    for start in range(0, len(query_ids), args.shard_size):
        shard_qids = query_ids[start : start + args.shard_size]
        path = (
            output / "batch_inputs" / f"requests_{start // args.shard_size:04d}.jsonl"
        )
        with path.open("w", encoding="utf-8") as handle:
            for qid in shard_qids:
                request = make_request(
                    args.model,
                    qid,
                    prompt_for(data, qid, candidates_by_query[qid]),
                    args.max_output_tokens,
                )
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        shard_paths.append(str(path))
    write_json(output / "query_manifest.json", {**audit, "records": records})
    write_jsonl(
        output / "candidate_manifest.jsonl",
        [
            {"query_id": qid, "candidate_doc_ids": candidates_by_query[qid]}
            for qid in query_ids
        ],
    )
    files = {}
    for name in ("corpus.jsonl", "queries.jsonl", "instances.jsonl", "qrels.tsv"):
        path = args.project_root.parent / "dataset/document-level/test" / name
        files[name] = {"path": str(path), "sha256": sha256(path)}
    write_json(
        output / "run_config.json",
        {
            "stage": "prepared",
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "evaluation_only": True,
            "training_seed": None,
            "model": args.model,
            "endpoint": "/v1/responses",
            "processing": "batch",
            "reasoning_effort": "none",
            "temperature": None,
            "retrieval_scope": "article_local",
            "query_view_seed": 42,
            "query_count": len(query_ids),
            "relation_count": audit["relation_count"],
            "selection_source": selection_source,
            "query_types": audit["query_types"],
            "shard_size": args.shard_size,
            "batch_input_files": shard_paths,
            "dataset_files": files,
            "metric_semantics": {
                "recall_and_mrr": "Direct only (relevance>=2)",
                "ndcg": "graded Direct=2 Partial=1",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"Prepared {len(query_ids)} requests in {len(shard_paths)} shards under {output}"
    )


def client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Install the SDK first: python -m pip install 'openai>=1.0'"
        ) from exc
    return OpenAI()


def submit(args) -> None:
    api = client()
    config = json.loads((args.output / "run_config.json").read_text())
    jobs_path = args.output / "batch_jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    submitted_paths = {job["input_path"] for job in jobs}
    newly_submitted = 0
    for value in config["batch_input_files"]:
        path = Path(value)
        if str(path) in submitted_paths:
            continue
        if newly_submitted >= args.max_new_jobs:
            break
        with path.open("rb") as handle:
            uploaded = api.files.create(file=handle, purpose="batch")
        batch = api.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"experiment": "gpt6_luna_full_q1_q4", "shard": path.stem},
        )
        jobs.append(
            {
                "input_path": str(path),
                "input_file_id": uploaded.id,
                "batch_id": batch.id,
                "status": batch.status,
            }
        )
        write_json(jobs_path, jobs)
        newly_submitted += 1
        print(path.name, batch.id, batch.status)
    print(f"New batches submitted: {newly_submitted}; total recorded: {len(jobs)}")


def realtime(args) -> None:
    """Run prepared requests through the synchronous Responses API.

    This is intended for a small smoke test. The full 22,156-query run should
    continue to use Batch so it remains resumable and less expensive.
    """
    api = client()
    candidate_rows = {
        row["query_id"]: row["candidate_doc_ids"]
        for row in read_jsonl(args.output / "candidate_manifest.jsonl")
    }
    existing_predictions = {}
    for existing_path in (
        args.output / "predictions.jsonl",
        args.output / "realtime_predictions.partial.jsonl",
    ):
        if existing_path.exists():
            for row in read_jsonl(existing_path):
                existing_predictions[row["query_id"]] = row
    requests = []
    for path in sorted((args.output / "batch_inputs").glob("requests_*.jsonl")):
        requests.extend(read_jsonl(path))
    requests = [
        request
        for request in requests
        if request["custom_id"] not in existing_predictions
    ]
    print(
        f"Realtime resume: existing={len(existing_predictions)} pending={len(requests)}",
        flush=True,
    )

    def invoke(request: dict) -> tuple[dict | None, dict | None]:
        qid = request["custom_id"]
        last_error = None
        for attempt in range(1, args.realtime_retries + 1):
            try:
                response = api.responses.create(**request["body"])
                body = response.model_dump()
                parsed = json.loads(extract_output_text(body))
                ranked = ranked_sentence_ids(parsed)
                allowed = set(candidate_rows[qid])
                dropped, cleaned, seen = [], [], set()
                for returned_id in ranked:
                    try:
                        doc_id = resolve_sentence_ids([returned_id], candidate_rows[qid])[0]
                    except ValueError:
                        dropped.append(returned_id)
                        continue
                    if doc_id not in allowed or doc_id in seen:
                        dropped.append(doc_id)
                        continue
                    cleaned.append(doc_id)
                    seen.add(doc_id)
                if not cleaned:
                    raise ValueError("response contains no valid sentence IDs")
                # Deterministic recovery for an otherwise usable response: keep
                # the model's valid order, then backfill from article order.
                for doc_id in candidate_rows[qid]:
                    if len(cleaned) >= min(10, len(candidate_rows[qid])):
                        break
                    if doc_id not in seen:
                        cleaned.append(doc_id)
                        seen.add(doc_id)
                return (
                    {
                        "query_id": qid,
                        "ranked_sen_ids": cleaned[:10],
                        "scores": [],
                        "llm_calls": 1,
                        "response_id": response.id,
                        "sanitized_invalid_doc_ids": dropped,
                    },
                    None,
                )
            except Exception as exc:
                last_error = exc
                if attempt < args.realtime_retries:
                    # A large synchronous run can hit minute-level RPM/TPM
                    # windows. Back off long enough for that window to reset.
                    time.sleep(min(5 * (2 ** (attempt - 1)), 60) + random.random())
        return None, {
            "query_id": qid,
            "reason": str(last_error),
            "error_type": type(last_error).__name__,
            "attempts": args.realtime_retries,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

    predictions, failures = list(existing_predictions.values()), []
    partial_path = args.output / "realtime_predictions.partial.jsonl"
    failure_partial_path = args.output / "realtime_failures.partial.jsonl"
    with partial_path.open(
        "a", encoding="utf-8"
    ) as partial_handle, failure_partial_path.open(
        "a", encoding="utf-8"
    ) as failure_handle:
        with ThreadPoolExecutor(max_workers=args.realtime_workers) as pool:
            futures = {
                pool.submit(invoke, request): request["custom_id"]
                for request in requests
            }
            for completed, future in enumerate(as_completed(futures), 1):
                prediction, failure = future.result()
                if prediction:
                    predictions.append(prediction)
                    partial_handle.write(
                        json.dumps(prediction, ensure_ascii=False) + "\n"
                    )
                    partial_handle.flush()
                if failure:
                    failures.append(failure)
                    failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failure_handle.flush()
                print(
                    f"Realtime progress {completed}/{len(requests)} total_success={len(predictions)} failures={len(failures)}",
                    flush=True,
                )

    predictions = list({row["query_id"]: row for row in predictions}.values())
    predictions.sort(key=lambda row: row["query_id"])
    failures.sort(key=lambda row: row["query_id"])
    write_jsonl(args.output / "predictions.jsonl", predictions)
    write_jsonl(args.output / "failures.jsonl", failures)
    config_path = args.output / "run_config.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "stage": "realtime_completed",
            "processing": "realtime",
            "realtime_workers": args.realtime_workers,
            "successful_queries": len(predictions),
            "failed_queries": len(failures),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(config_path, config)
    print(f"Realtime predictions={len(predictions)} failures={len(failures)}")


def status(args) -> None:
    api = client()
    jobs_path = args.output / "batch_jobs.json"
    jobs = json.loads(jobs_path.read_text())
    for job in jobs:
        batch = api.batches.retrieve(job["batch_id"])
        job.update(
            {
                "status": batch.status,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "request_counts": (
                    batch.request_counts.model_dump() if batch.request_counts else None
                ),
            }
        )
        print(job["batch_id"], batch.status, job["request_counts"])
    write_json(jobs_path, jobs)


def download(args) -> None:
    api = client()
    jobs_path = args.output / "batch_jobs.json"
    jobs = json.loads(jobs_path.read_text())
    raw_dir = args.output / "batch_outputs"
    raw_dir.mkdir(exist_ok=True)
    candidate_rows = {
        row["query_id"]: row["candidate_doc_ids"]
        for row in read_jsonl(args.output / "candidate_manifest.jsonl")
    }
    predictions, failures = [], []
    for index, job in enumerate(jobs):
        batch = api.batches.retrieve(job["batch_id"])
        if batch.status != "completed" or not batch.output_file_id:
            failures.append(
                {"batch_id": batch.id, "reason": f"batch_status={batch.status}"}
            )
            continue
        raw_path = raw_dir / f"responses_{index:04d}.jsonl"
        if not raw_path.exists():
            api.files.content(batch.output_file_id).write_to_file(raw_path)
        for row in read_jsonl(raw_path):
            qid = row["custom_id"]
            try:
                response = row["response"]
                if int(response["status_code"]) != 200:
                    raise ValueError(f"HTTP {response['status_code']}")
                parsed = json.loads(extract_output_text(response["body"]))
                ranked = resolve_sentence_ids(ranked_sentence_ids(parsed), candidate_rows[qid])
                allowed = set(candidate_rows[qid])
                if (
                    not ranked
                    or len(ranked) != len(set(ranked))
                    or any(doc_id not in allowed for doc_id in ranked)
                ):
                    raise ValueError("invalid, duplicate, or unknown sentence IDs")
                predictions.append(
                    {
                        "query_id": qid,
                        "ranked_sen_ids": ranked[:10],
                        "scores": [],
                        "llm_calls": 1,
                    }
                )
            except Exception as exc:
                failures.append({"query_id": qid, "reason": str(exc)})
    predictions.sort(key=lambda row: row["query_id"])
    write_jsonl(args.output / "predictions.jsonl", predictions)
    write_jsonl(args.output / "failures.jsonl", failures)
    print(f"Parsed predictions={len(predictions)} failures={len(failures)}")


def evaluate(args) -> None:
    from document_level_evidence_ranking.config import ExperimentConfig
    from document_level_evidence_ranking.evaluation import _aggregate, _per_query

    data = load_split(args.project_root.parent / "dataset/document-level/test", "test")
    query_ids, _, audit = build_complete_four_query_view(data)
    predictions = {
        row["query_id"]: row for row in read_jsonl(args.output / "predictions.jsonl")
    }
    missing = sorted(set(query_ids) - predictions.keys())
    if missing:
        raise SystemExit(
            f"Refusing incomplete evaluation: {len(missing)} predictions missing; see failures.jsonl"
        )
    cfg = ExperimentConfig(
        project_root=args.project_root,
        phase5_dir=args.project_root.parent / "dataset/document-level",
        results_dir=args.output,
        retrieval_scope="article_local",
        metric_ks=(1, 3, 5, 10),
        binary_relevance_threshold=2,
    )
    rows, enriched = [], []
    by_type = {}
    for qid in query_ids:
        metric = _per_query(ranking_for_evaluation(predictions[qid], candidate_doc_ids(data, qid, "article_local")), data.qrels[qid], cfg)
        rows.append(metric)
        qtype = str(data.instances[qid].get("query_type", "unknown"))
        by_type.setdefault(qtype, []).append(metric)
        enriched.append(
            {
                "query_id": qid,
                "relation_id": str(data.instances[qid].get("relation_id") or qid),
                "query_type": qtype,
                "metrics": metric,
                "ranked_sen_ids": ranking_for_evaluation(predictions[qid], candidate_doc_ids(data, qid, "article_local")),
            }
        )
    metrics = {
        "coverage": {
            "expected_queries": len(query_ids),
            "successful_queries": len(rows),
            "failed_queries": 0,
        },
        "overall": _aggregate(rows),
        "by_query_type": {
            key: _aggregate(value) for key, value in sorted(by_type.items())
        },
        "efficiency": {"llm_calls_per_query": 1.0},
    }
    write_json(args.output / "metrics.json", metrics)
    write_jsonl(args.output / "per_query_metrics.jsonl", enriched)
    print(json.dumps(metrics["overall"], indent=2))


def evaluate_smoke(args) -> None:
    from document_level_evidence_ranking.config import ExperimentConfig
    from document_level_evidence_ranking.evaluation import _aggregate, _per_query

    data = load_split(args.project_root.parent / "dataset/document-level/test", "test")
    predictions = list(read_jsonl(args.output / "predictions.jsonl"))
    if not predictions:
        raise SystemExit("No smoke predictions found")
    cfg = ExperimentConfig(
        project_root=args.project_root,
        phase5_dir=args.project_root.parent / "dataset/document-level",
        results_dir=args.output,
        retrieval_scope="article_local",
        metric_ks=(1, 3, 5, 10),
        binary_relevance_threshold=2,
    )
    rows, by_type = [], {}
    for prediction in predictions:
        qid = prediction["query_id"]
        metric = _per_query(ranking_for_evaluation(prediction, candidate_doc_ids(data, qid, "article_local")), data.qrels[qid], cfg)
        rows.append(metric)
        query_type = str(data.instances[qid].get("query_type", "unknown"))
        by_type.setdefault(query_type, []).append(metric)
    report = {
        "status": "smoke_test_only_not_formal",
        "model": args.model,
        "queries": len(predictions),
        "full_test_queries": len(data.query_ids),
        "coverage_fraction": len(predictions) / len(data.query_ids),
        "overall": _aggregate(rows),
        "by_query_type": {
            name: _aggregate(values) for name, values in sorted(by_type.items())
        },
        "warning": "Matched DeepSeek smoke subset; pipeline validation only, not a formal benchmark result.",
    }
    write_json(args.output / "smoke_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "realtime",
            "submit",
            "status",
            "download",
            "evaluate-smoke",
            "evaluate",
        ),
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=1,
        help="Submit at most this many new shards (protects low-tier Batch queue limits).",
    )
    parser.add_argument("--realtime-workers", type=int, default=4)
    parser.add_argument("--realtime-retries", type=int, default=3)
    parser.add_argument(
        "--match-predictions",
        type=Path,
        help="Use exactly the query IDs in this predictions JSONL (for matched smoke tests).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Optional smoke-test limit applied during prepare.",
    )
    args = parser.parse_args()
    globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    main()
