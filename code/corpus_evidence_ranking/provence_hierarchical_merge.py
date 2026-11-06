from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import CorpusData, read_jsonl, sha256, write_jsonl
from .evaluate import metrics_from_predictions


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and merge sharded hierarchical Provence predictions"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    if len(args.shard_dir) < 2:
        parser.error("at least two --shard-dir values are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    data = CorpusData.load(args.data_dir, args.split)
    expected = sorted(data.queries)
    expected_set = set(expected)
    configs: list[dict] = []
    rows_by_qid: dict[str, dict] = {}
    shard_records = []
    for shard_dir in args.shard_dir:
        config_path = shard_dir / "run_config.json"
        completion_path = shard_dir / "completion.json"
        predictions_path = shard_dir / "predictions.jsonl"
        if not (
            config_path.is_file()
            and completion_path.is_file()
            and predictions_path.is_file()
        ):
            raise FileNotFoundError(f"Incomplete shard output: {shard_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not completion.get("completed"):
            raise RuntimeError(f"Shard is not complete: {shard_dir}")
        rows = read_jsonl(predictions_path)
        if len(rows) != int(completion["query_count"]):
            raise ValueError(f"Shard row count mismatch: {shard_dir}")
        for row in rows:
            qid = str(row["query_id"])
            if qid in rows_by_qid:
                raise ValueError(f"Duplicate query ID across shards: {qid}")
            rows_by_qid[qid] = row
        configs.append(config)
        shard_records.append(
            {
                "path": str(shard_dir.resolve()),
                "query_count": len(rows),
                "predictions_sha256": sha256(predictions_path),
                "run_config_sha256": sha256(config_path),
            }
        )

    invariant_keys = (
        "split",
        "variant",
        "seed",
        "candidate_depth",
        "block_batch_size",
        "candidates_sha256",
        "trained_checkpoint_sha256",
        "training_run_config_sha256",
        "data_manifest",
    )
    reference = configs[0]
    for index, config in enumerate(configs[1:], 1):
        for key in invariant_keys:
            if config.get(key) != reference.get(key):
                raise ValueError(f"Shard {index} differs for invariant {key}")

    actual_set = set(rows_by_qid)
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if missing or extra or len(rows_by_qid) != len(expected):
        raise ValueError(
            f"Invalid shard coverage: expected={len(expected)} actual={len(rows_by_qid)} "
            f"missing={len(missing)} extra={len(extra)}"
        )

    rows = [rows_by_qid[qid] for qid in expected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    metrics = metrics_from_predictions(data, rows)
    save_json(args.output_dir / "metrics.json", metrics)
    save_json(
        args.output_dir / "run_config.json",
        {
            **reference,
            "output_dir": str(args.output_dir.resolve()),
            "query_count": len(rows),
            "query_ids": "all sorted queries in split; verified union of shards",
            "merged_from_shards": shard_records,
            "merge_validation": {
                "expected_queries": len(expected),
                "actual_queries": len(rows),
                "duplicate_queries": 0,
                "missing_queries": 0,
                "extra_queries": 0,
            },
        },
    )
    save_json(
        args.output_dir / "completion.json",
        {
            "completed": True,
            "query_count": len(rows),
            "predictions_sha256": sha256(predictions_path),
            "metrics": metrics,
            "merge_validation": {
                "shards": len(args.shard_dir),
                "duplicates": 0,
                "missing": 0,
                "extra": 0,
            },
        },
    )


if __name__ == "__main__":
    main()
