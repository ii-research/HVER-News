#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=[13, 42, 87], required=True)
    args = p.parse_args()
    root = args.bundle_root.resolve()
    data = args.project_root.resolve().parent / "dataset/document-level"
    common = {
        "seed": args.seed,
        "epochs": 3,
        "learning_rate": 3e-6,
        "head_learning_rate": 3e-5,
        "effective_batch_size": 16,
        "precision": "bf16",
        "eval_precision": "fp32",
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "sentence_width": 256,
        "negative_policy": "fixed ordinary qrel0 up to3; sampling seed42; shared by HVER and ablation",
        "loss_normalization": "global query mean; equal supervised-class means within query",
        "test_policy": "all 22156 document-level Q1-Q4 queries",
        "checkpoint_selection": "full fixed dev2506 nDCG@10",
        "initialization": "official Provence checkpoint plus freshly initialized sentence modules",
        "bundle_root": str(root),
        "data_root": str(data),
        "benchmark_root": str(args.project_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "sampling": str(root / "shared/sampling_abcd.jsonl"),
        "full_manifest_sha256": sha(root / "four_query_manifest.json"),
        "query_sampling_seed": 42,
    }
    configs = {
        "without_pe": {
            "experiment": f"HVER-News-seed{args.seed}",
            "sentence_interaction": True,
            "sentence_position_encoding": False,
            "output": str(args.results_root / "G3/hver_news" / f"seed_{args.seed}"),
        },
        "adapted_provence": {
            "experiment": f"HVER-News-w/o-SI-seed{args.seed}",
            "sentence_interaction": False,
            "sentence_position_encoding": False,
            "output": str(
                args.results_root / "G3/hver_news_wo_si" / f"seed_{args.seed}"
            ),
        },
    }
    for arm, extra in configs.items():
        formal = {**common, "arm": arm, **extra}
        dump(root / "configs" / f"{arm}_{args.seed}.json", formal)
        dump(
            root / "configs" / f"{arm}_{args.seed}_smoke.json",
            {**formal, "output": str(root / "output" / f"smoke_{arm}")},
        )


if __name__ == "__main__":
    main()
