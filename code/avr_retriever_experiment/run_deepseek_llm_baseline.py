"""Run the DeepSeek article-local LLM baseline on the full Q1--Q4 test.

DeepSeek exposes an OpenAI-compatible Responses API but not an equivalent
documented Batch API. This runner therefore uses bounded async concurrency,
retry/backoff, immediate append-only persistence, and resumable execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from avr_retriever_experiment.api_prompts import (ranked_sentence_ids, resolve_sentence_ids, ranking_for_evaluation, PROMPT_FINGERPRINT)
from avr_retriever_experiment.data import candidate_doc_ids, load_split
from avr_retriever_experiment.run_openai_llm_baseline import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    build_complete_four_query_view,
    evaluate as formal_evaluate,
    prompt_for,
    read_jsonl,
    sha256,
)
from avr_retriever_experiment.support.io import write_json, write_jsonl


def prepare(args) -> None:
    data = load_split(args.project_root.parent / "dataset/document-level/test", "test")
    query_ids, records, audit = build_complete_four_query_view(data)
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = {
        qid: candidate_doc_ids(data, qid, "article_local") for qid in query_ids
    }
    write_json(args.output / "query_manifest.json", {**audit, "records": records})
    write_jsonl(
        args.output / "candidate_manifest.jsonl",
        [{"query_id": qid, "candidate_doc_ids": candidates[qid]} for qid in query_ids],
    )
    files = {}
    for name in ("corpus.jsonl", "queries.jsonl", "instances.jsonl", "qrels.tsv"):
        path = args.project_root.parent / "dataset/document-level/test" / name
        files[name] = {"path": str(path), "sha256": sha256(path)}
    write_json(
        args.output / "run_config.json",
        {
            "stage": "prepared",
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "evaluation_only": True,
            "training_seed": None,
            "provider": "deepseek",
            "base_url": args.base_url,
            "model": args.model,
            "endpoint": "/responses",
            "execution": "bounded_async_resumable",
            "reasoning_effort": "none",
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "retrieval_scope": "article_local",
            "query_view_seed": 42,
            "query_count": len(query_ids),
            "relation_count": audit["relation_count"],
            "query_types": audit["query_types"],
            "dataset_files": files,
            "metric_semantics": {
                "recall_and_mrr": "Direct only (relevance>=2)",
                "ndcg": "graded Direct=2 Partial=1",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"Prepared DeepSeek baseline: relations={audit['relation_count']} queries={len(query_ids)}"
    )


def parse_response(response, candidates: list[str]) -> tuple[list[str], list[dict]]:
    payload = json.loads(response.output_text)
    ranked = ranked_sentence_ids(payload)
    if not ranked:
        raise ValueError("empty sentence ranking")
    allowed = set(candidates)
    normalized, mappings, unknown, seen = [], [], [], set()
    for value in ranked:
        try:
            doc_id = resolve_sentence_ids([value], candidates, allow_indices=True)[0]
        except ValueError:
            unknown.append(value)
            continue
        if doc_id != value:
            mappings.append({"returned_id": value, "mapped_sentence_id": doc_id})
        if doc_id not in allowed:
            unknown.append(value)
            continue
        if doc_id in seen:
            continue
        normalized.append(doc_id)
        seen.add(doc_id)
    if unknown:
        raise ValueError(f"unknown candidate IDs: {unknown[:3]}")
    for doc_id in candidates:
        if len(normalized) >= min(10, len(candidates)):
            break
        if doc_id not in seen:
            normalized.append(doc_id)
            seen.add(doc_id)
    return normalized[:10], mappings


async def run_async(args) -> None:
    config = json.loads((args.output / "run_config.json").read_text())
    if config.get("prompt_fingerprint"):
        if config["prompt_fingerprint"] != PROMPT_FINGERPRINT:
            raise ValueError("Prompt differs from prepared run; use its original prompt or a new output directory")
    else:
        raise ValueError("Run has no prompt fingerprint; prepare a new output directory")
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit(
            "Install the SDK first: python -m pip install 'openai>=1.0'"
        ) from exc
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    api = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        max_retries=0,
        timeout=args.timeout_seconds,
    )
    data = load_split(args.project_root.parent / "dataset/document-level/test", "test")
    expected, _, _ = build_complete_four_query_view(data)
    candidates = {
        row["query_id"]: row["candidate_doc_ids"]
        for row in read_jsonl(args.output / "candidate_manifest.jsonl")
    }
    raw_path = args.output / "responses.jsonl"
    completed = (
        {row["query_id"] for row in read_jsonl(raw_path)}
        if raw_path.exists()
        else set()
    )
    pending = [qid for qid in expected if qid not in completed]
    if args.max_queries is not None:
        pending = pending[: args.max_queries]
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters = Counter(completed=len(completed))

    async def one(qid: str) -> None:
        allowed = set(candidates[qid])
        prompt = prompt_for(data, qid, candidates[qid])
        last_error = None
        async with semaphore:
            for attempt in range(args.max_retries + 1):
                try:
                    response = await api.responses.create(
                        model=args.model,
                        reasoning={"effort": "none"},
                        input=[
                            {
                                "role": "developer",
                                "content": [
                                    {"type": "input_text", "text": SYSTEM_PROMPT}
                                ],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": prompt}],
                            },
                        ],
                        max_output_tokens=args.max_output_tokens,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "evidence_ranking",
                                "strict": True,
                                "schema": OUTPUT_SCHEMA,
                            }
                        },
                    )
                    ranked, index_mappings = parse_response(response, candidates[qid])
                    usage = response.usage.model_dump() if response.usage else None
                    row = {
                        "query_id": qid,
                        "ranked_sen_ids": ranked,
                        "scores": [],
                        "llm_calls": attempt + 1,
                        "response_id": response.id,
                        "usage": usage,
                        "index_mappings": index_mappings,
                    }
                    async with write_lock:
                        with raw_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                        counters["succeeded"] += 1
                    return
                except (
                    Exception
                ) as exc:  # API/network/schema failures are retryable here.
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_retries:
                        await asyncio.sleep(min(60.0, (2**attempt) + random.random()))
            async with write_lock:
                with (args.output / "failures.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps(
                            {"query_id": qid, "reason": last_error}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    handle.flush()
                counters["failed"] += 1

    for start in range(0, len(pending), args.task_window):
        window = pending[start : start + args.task_window]
        await asyncio.gather(*(one(qid) for qid in window))
        print(
            f"progress={min(start + len(window), len(pending))}/{len(pending)} "
            f"succeeded={counters['succeeded']} failed={counters['failed']}",
            flush=True,
        )
    await api.close()
    rows = list(read_jsonl(raw_path)) if raw_path.exists() else []
    latest = {row["query_id"]: row for row in rows}
    write_jsonl(
        args.output / "predictions.jsonl", [latest[qid] for qid in sorted(latest)]
    )
    missing = len(set(expected) - latest.keys())
    print(f"Saved unique predictions={len(latest)} missing={missing}")


def run(args) -> None:
    asyncio.run(run_async(args))


def evaluate(args) -> None:
    formal_evaluate(args)


def evaluate_smoke(args) -> None:
    """Score only completed smoke queries; never label this as a formal result."""
    from avr_retriever_experiment.config import ExperimentConfig
    from avr_retriever_experiment.evaluation import _aggregate, _per_query

    data = load_split(args.project_root.parent / "dataset/document-level/test", "test")
    expected, _, _ = build_complete_four_query_view(data)
    predictions = {
        row["query_id"]: row for row in read_jsonl(args.output / "predictions.jsonl")
    }
    qids = [qid for qid in expected if qid in predictions]
    if not qids:
        raise SystemExit("No completed smoke predictions found")
    cfg = ExperimentConfig(
        project_root=args.project_root,
        phase5_dir=args.project_root.parent / "dataset/document-level",
        results_dir=args.output,
        retrieval_scope="article_local",
        metric_ks=(1, 3, 5, 10),
        binary_relevance_threshold=2,
    )
    rows, by_type = [], {}
    for qid in qids:
        metric = _per_query(ranking_for_evaluation(predictions[qid], candidate_doc_ids(data, qid, "article_local")), data.qrels[qid], cfg)
        rows.append(metric)
        qtype = str(data.instances[qid].get("query_type", "unknown"))
        by_type.setdefault(qtype, []).append(metric)
    report = {
        "status": "smoke_test_only_not_formal",
        "model": args.model,
        "queries": len(qids),
        "full_test_queries": len(expected),
        "coverage_fraction": len(qids) / len(expected),
        "overall": _aggregate(rows),
        "by_query_type": {
            name: _aggregate(values) for name, values in sorted(by_type.items())
        },
        "warning": "This tiny prefix subset is for pipeline validation only and is not a comparable benchmark result.",
    }
    write_json(args.output / "smoke_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "run", "evaluate-smoke", "evaluate")
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--task-window", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Smoke-test limit; omit for the formal full run.",
    )
    args = parser.parse_args()
    command = args.command.replace("-", "_")
    globals()[command](args)


if __name__ == "__main__":
    main()
