"""Formal runner for the deduplicated, document-level benchmark.

Train/dev use the frozen one-query-per-relation view.  Test always evaluates
all four query variants so Q1--Q4 analyses are available from the same run.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from document_level_evidence_ranking.config import ExperimentConfig
from document_level_evidence_ranking.data import load_split
from document_level_evidence_ranking.runner import run_setting
from document_level_evidence_ranking.sampling import select_experiment_query_ids
from document_level_evidence_ranking.settings import all_settings
from document_level_evidence_ranking.support.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parent / "dataset/document-level"
DEFAULT_RESULTS = Path("results/document-level")
DEFAULT_MODELS = Path("models/document-level")


def parse_args():
    parser = argparse.ArgumentParser(description="Document-level evidence ranking (DER): rank source-article sentences.")
    parser.add_argument(
        "--setting",
        required=True,
        choices=(
            "bm25",
            "e5_zero_shot",
            "bge_m3_zero_shot",
            "qwen3_zero_shot",
            "e5_generic_ft",
            "qwen3_generic_ft",
        ),
    )
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-view-seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    data_dir = args.data_dir.resolve()
    train = load_split(data_dir / "train", "train")
    dev = load_split(data_dir / "dev", "dev")
    test = load_split(data_dir / "test", "test")

    if args.results_dir.is_absolute():
        results_dir = args.results_dir
    else:
        results_dir = ROOT / args.results_dir
    if args.models_dir.is_absolute():
        models_dir = args.models_dir
    else:
        models_dir = ROOT / args.models_dir

    cfg = ExperimentConfig(
        project_root=ROOT,
        phase5_dir=(
            data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir
        ),
        results_dir=(
            results_dir.relative_to(ROOT)
            if results_dir.is_relative_to(ROOT)
            else results_dir
        ),
        models_dir=(
            models_dir.relative_to(ROOT)
            if models_dir.is_relative_to(ROOT)
            else models_dir
        ),
        retrieval_scope="article_local",
        seed=args.train_seed,
        query_view_mode="one_per_relation_balanced",
        query_view_seed=args.query_view_seed,
        reuse_query_view_manifest=True,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        epochs=3,
        batch_size=16,
        grad_accum=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        temperature=0.05,
        checkpoint_metric="ndcg@10",
    )
    seed_everything(cfg.seed)

    train_ids = select_experiment_query_ids(
        train,
        results_dir,
        cfg.query_view_mode,
        cfg.query_view_seed,
        True,
        cfg.query_types,
        None,
    )
    dev_ids = select_experiment_query_ids(
        dev,
        results_dir,
        cfg.query_view_mode,
        cfg.query_view_seed,
        True,
        cfg.query_types,
        None,
    )
    # Formal reporting and Q1--Q4 analysis use every retained test query.
    test_ids = sorted(test.query_ids)

    registry = all_settings(cfg.e5_model, cfg.bge_model, cfg.train_backbone)
    if args.setting in {"bm25", "e5_zero_shot", "bge_m3_zero_shot", "qwen3_zero_shot"}:
        setting = registry[args.setting]
    elif args.setting == "e5_generic_ft":
        setting = replace(
            registry["generic_ft"],
            name="e5_generic_ft",
            model_name="intfloat/e5-base-v2",
        )
    else:
        setting = replace(
            registry["generic_ft"],
            name="qwen3_generic_ft",
            model_name="Qwen/Qwen3-Embedding-0.6B",
        )

    print(f"data={data_dir}")
    print(f"setting={setting.name} train_seed={args.train_seed}")
    print(f"train/dev/test queries={len(train_ids)}/{len(dev_ids)}/{len(test_ids)}")
    try:
        run_setting(cfg, setting, train, dev, test, train_ids, dev_ids, test_ids)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
