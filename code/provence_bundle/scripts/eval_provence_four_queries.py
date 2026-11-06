"""Evaluate all four formulations on the same frozen test relation population."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from provence_finetune import METRICS, benchmark, evaluate, write_json
from provence_zero_shot import Scorer, sha256


def validate_rows(data, manifest):
    records = manifest["records"]
    ids = [r["query_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(data.queries):
        raise ValueError("Full test query population differs from the frozen manifest")
    grouped = defaultdict(list)
    for record in records:
        row = data.instances[record["query_id"]]
        if (row["relation_id"], row["query_type"]) != (
            record["relation_id"],
            record["query_type"],
        ):
            raise ValueError("Query metadata changed")
        grouped[row["relation_id"]].append(row)
    if set(grouped) != set(manifest["relation_ids"]):
        raise ValueError("Test relation population changed")
    for rows in grouped.values():
        if len(rows) != 4 or {r["query_type"] for r in rows} != set(
            manifest["query_types"]
        ):
            raise ValueError("Every relation must have exactly four query types")
        first = rows[0]
        for row in rows:
            for key in (
                "guid",
                "actor_name",
                "value_name",
                "direction",
                "article_local_candidate_doc_ids",
            ):
                if row[key] != first[key]:
                    raise ValueError(f"Query formulations differ in {key}")
            if data.qrels[row["query_id"]] != data.qrels[first["query_id"]]:
                raise ValueError(
                    "Qrels differ between formulations of the same relation"
                )
    return [data.instances[q] for q in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["zero_shot", "generic_seed42"], required=True
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    for filename, expected in cfg["data_sha256"].items():
        if sha256(Path(cfg["source"]) / filename) != expected:
            raise ValueError(f"Frozen test input changed: {filename}")
    manifest_path = Path(cfg["manifest"])
    if sha256(manifest_path) != cfg["manifest_sha256"]:
        raise ValueError("Four-query manifest changed")
    modules = benchmark(cfg["benchmark_root"])
    data = modules[0].load_split(Path(cfg["source"]), "test")
    manifest = json.loads(manifest_path.read_text())
    rows = validate_rows(data, manifest)
    output = Path(cfg["output_root"]) / args.model
    output.mkdir(parents=True, exist_ok=False)
    scorer = Scorer(cfg["base_checkpoint"], args.device, 8)
    if args.model == "generic_seed42":
        weights = Path(cfg["generic_checkpoint"]) / "model.safetensors"
        if sha256(weights) != cfg["generic_weights_sha256"]:
            raise ValueError("Wrong Generic FT best checkpoint")
        scorer.model.load_state_dict(load_file(str(weights)), strict=True)
    scorer.model.requires_grad_(False)
    scorer.model.eval()
    write_json(
        output / "run_config.json", {**cfg, "model": args.model, "training": False}
    )
    evaluate(
        scorer,
        data,
        rows[:32],
        modules,
        output / "smoke32",
        manifest_path,
        Path(cfg["source"]),
        args.model,
    )
    if args.smoke_only:
        return
    result = evaluate(
        scorer,
        data,
        rows,
        modules,
        output / "test",
        manifest_path,
        Path(cfg["source"]),
        args.model,
    )
    eval_cfg = modules[3].ExperimentConfig(
        project_root=Path(cfg["source"]).parent,
        phase5_dir=Path(cfg["source"]).parent,
        results_dir=output,
    )
    with (output / "relation_query_metrics.csv").open("w", newline="") as table:
        writer = csv.writer(table)
        writer.writerow(["relation_id", "query_id", "query_type", *METRICS])
        for line in (output / "test/predictions.jsonl").open():
            pred = json.loads(line)
            row = data.instances[pred["query_id"]]
            order = np.argsort(-np.asarray(pred["scores"], dtype=np.float32))[:10]
            ranked = [pred["doc_ids"][int(i)] for i in order]
            scores = modules[2]._per_query(
                ranked, data.qrels[row["query_id"]], eval_cfg
            )
            scores["MRR@10"] = scores["mrr"]
            writer.writerow(
                [
                    row["relation_id"],
                    row["query_id"],
                    row["query_type"],
                    *[scores[k] for k in METRICS],
                ]
            )
    write_json(
        output / "four_query_analysis.json",
        {
            "relation_count": len(manifest["relation_ids"]),
            "query_count": len(rows),
            "by_query_type": result["by_query_type"],
            "note": "Each query type evaluates the exact same relations, candidates and qrels.",
        },
    )
    (output / "COMPLETED").write_text(
        "Full four-query evaluation and analysis completed.\n"
    )


if __name__ == "__main__":
    main()
