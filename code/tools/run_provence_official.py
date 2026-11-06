#!/usr/bin/env python3
"""Run official Provence zero-shot and Generic FT on the fixed document-level view."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def config(args):
    prepared = args.bundle_root / "prepared/provence_generic"
    return {
        "benchmark_root": str(args.project_root),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.project_root.parent / "dataset/document-level"),
        "effective_batch_size": 16,
        "epochs": 3,
        "eval_window_batch_size": 8,
        "gradient_checkpointing": True,
        "learning_rate": 3e-6,
        "max_grad_norm": 1.0,
        "output_root": str(args.results_root / "G3/provence_generic_ft"),
        "precision": "bf16",
        "prepared": str(prepared),
        "query_views": str(args.bundle_root / "shared"),
        "smoke_tag": "within_generic",
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "zero_shot_metrics": str(
            args.bundle_root / "prepared/provence_zero_reference/formal_metrics.json"
        ),
        "allow_oversized_sentence_fragments": True,
    }


def evaluate_all(args, checkpoint=None, name="Provence zero-shot"):
    import provence_finetune as ft
    from eval_provence_four_queries import validate_rows
    from provence_zero_shot import Scorer

    modules = ft.benchmark(args.project_root)
    data_root = args.project_root.parent / "dataset/document-level/test"
    data = modules[0].load_split(data_root, "test")
    manifest = args.bundle_root / "four_query_manifest.json"
    rows = validate_rows(data, json.loads(manifest.read_text()))
    scorer = Scorer(str(args.checkpoint), "cuda", 8)
    if checkpoint:
        scorer.model.load_state_dict(load_file(str(checkpoint)), strict=True)
    scorer.model.requires_grad_(False)
    scorer.model.eval()
    out = args.results_root / (
        "G3/provence_zero_shot/seed_na"
        if checkpoint is None
        else f"G3/provence_generic_ft/seed_{args.seed}/full_test"
    )
    result = ft.evaluate(scorer, data, rows, modules, out, manifest, data_root, name)
    ft.write_json(
        out / "run_config.json",
        {
            "model": name,
            "training_seed": None if checkpoint is None else args.seed,
            "query_view_seed": 42,
            "checkpoint": str(args.checkpoint if checkpoint is None else checkpoint),
            "test_queries": len(rows),
            "all_candidate_scores_saved": True,
            "metric_semantics": "Direct-only Recall/MRR; graded Direct=2 Partial=1 nDCG",
        },
    )
    print(json.dumps(result["overall"], indent=2))


def zero_reference(args):
    import provence_finetune as ft
    from provence_zero_shot import Scorer, load_fixed_queries

    modules = ft.benchmark(args.project_root)
    data_root = args.project_root.parent / "dataset/document-level/test"
    data = modules[0].load_split(data_root, "test")
    manifest = args.bundle_root / "shared/test_relations_all.json"
    rows = load_fixed_queries(manifest, data_root, "test")
    scorer = Scorer(str(args.checkpoint), "cuda", 8)
    scorer.model.requires_grad_(False)
    out = args.bundle_root / "prepared/provence_zero_reference"
    ft.evaluate(
        scorer, data, rows, modules, out, manifest, data_root, "Provence zero-shot"
    )


def generic_smoke(cfg):
    """Validate the official Generic FT training objective on a small batch."""
    import provence_finetune as ft

    contract = ft.verify_contract(cfg)
    modules = ft.benchmark(cfg["benchmark_root"])
    data = modules[0].load_split(Path(cfg["data_root"]) / "train", "train")
    records = [
        json.loads(line) for line in (Path(cfg["prepared"]) / "sampling.jsonl").open()
    ]
    records = [row for row in records if "skip" not in row][:2]
    if not records:
        raise ValueError("No trainable Generic FT records")
    torch.manual_seed(42)
    scorer = ft.make_scorer(cfg, "cuda")
    scorer.model.train()
    scorer.model.zero_grad(set_to_none=True)
    losses = []
    for record in records:
        qid = record["query_id"]
        loss, windows = ft.backward_query(
            scorer,
            data.instances[qid],
            data.corpus,
            record,
            "generic",
            data.qrels[qid],
            len(records),
            cfg["precision"],
        )
        losses.append({"query_id": qid, "loss": loss, "windows": windows})
    encoder = ft.gradient_norm(scorer.model.deberta)
    head = ft.gradient_norm(scorer.model.token_classifier)
    if not (
        math.isfinite(encoder) and encoder > 0 and math.isfinite(head) and head > 0
    ):
        raise ValueError("Invalid Generic FT smoke gradients")
    report = {
        "passed": True,
        "arm": "generic",
        "records": losses,
        "encoder_gradient_norm": encoder,
        "pruning_gradient_norm": head,
        "sampling_sha256": contract["sampling_sha256"],
    }
    path = Path(cfg["prepared"]) / f"smoke_cuda_{cfg.get('smoke_tag', 'generic')}.json"
    ft.write_json(path, report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "zero-reference",
            "zero-full",
            "prepare",
            "preflight",
            "smoke",
            "train",
            "eval-full",
        ],
    )
    p.add_argument("--bundle-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=[13, 42, 87], default=42)
    args = p.parse_args()
    sys.path[:0] = [str(args.bundle_root / "scripts"), str(args.project_root)]
    import provence_finetune as ft

    cfg = config(args)
    if args.command == "zero-reference":
        zero_reference(args)
    elif args.command == "zero-full":
        evaluate_all(args)
    elif args.command == "prepare":
        ft.prepare(cfg)
    elif args.command == "preflight":
        ft.preflight(cfg)
    elif args.command == "smoke":
        generic_smoke(cfg)
    elif args.command == "train":
        ft.train(cfg, "generic", args.seed, "cuda")
    else:
        weights = (
            args.results_root
            / f"G3/provence_generic_ft/generic_seed{args.seed}/best_checkpoint/model.safetensors"
        )
        evaluate_all(args, weights, f"Provence Generic FT seed {args.seed}")


if __name__ == "__main__":
    main()
