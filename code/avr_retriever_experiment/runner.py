from __future__ import annotations

from dataclasses import asdict
from typing import Sequence
import os

import torch.distributed as dist

from avr_retriever_experiment.baselines.bm25 import evaluate_bm25
from avr_retriever_experiment.config import ExperimentConfig
from avr_retriever_experiment.data import BenchmarkData
from avr_retriever_experiment.evaluation import evaluate_model
from avr_retriever_experiment.model import DualEncoder
from avr_retriever_experiment.negative_mining import RelationNegativeMiner
from avr_retriever_experiment.settings import ExperimentSetting
from avr_retriever_experiment.support.io import write_json, write_jsonl
from avr_retriever_experiment.train import train_setting
from avr_retriever_experiment.training_data import (
    RetrievalTrainingDataset,
)


def run_setting(
    cfg: ExperimentConfig,
    setting: ExperimentSetting,
    train_data: BenchmarkData,
    dev_data: BenchmarkData,
    test_data: BenchmarkData,
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    test_ids: Sequence[str],
) -> None:
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    is_main = not distributed or dist.get_rank() == 0
    # Fine-tuned runs are seed-specific.  Zero-shot/BM25 have no training seed
    # and are written once under seed_na; query-view randomness is recorded
    # separately and must not be presented as model-training randomness.
    seed_label = str(cfg.seed) if setting.kind == "train" else "na"
    out_dir = cfg.abs_results_dir / "G3" / setting.name / f"seed_{seed_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setting_dict = dict(setting.__dict__)
    if setting.recipe is not None:
        setting_dict["recipe"] = asdict(setting.recipe)

    if is_main:
        write_json(
            out_dir / "run_config.json",
            {
                "experiment_config": cfg.snapshot(),
                "setting": setting_dict,
                "training_seed": cfg.seed if setting.kind == "train" else None,
                "query_view_seed": cfg.query_view_seed,
                "query_view_mode": cfg.query_view_mode,
                "train_query_count": len(train_ids),
                "dev_query_count": len(dev_ids),
                "test_query_count": len(test_ids),
                "retrieval_unit": "source_sentence",
                "candidate_scope": cfg.retrieval_scope,
                "metric_semantics": {
                    "binary_metrics_and_mrr": "Direct evidence only (relevance >= 2)",
                    "ndcg": "graded qrels: Direct=2, Partial/contextual=1, None=0",
                },
                "distributed_training": {
                    "enabled": distributed,
                    "world_size": dist.get_world_size() if distributed else 1,
                    "global_batch_size": cfg.batch_size,
                    "per_rank_batch_size": (
                        cfg.batch_size // dist.get_world_size()
                        if distributed
                        else cfg.batch_size
                    ),
                    "in_batch_negatives": "gathered across all ranks",
                },
            },
        )

    if setting.kind == "bm25":
        metrics, preds = evaluate_bm25(test_data, test_ids, cfg)

    elif setting.kind == "zero_shot":
        model = DualEncoder(
            setting.model_name,
            cfg.query_max_length,
            cfg.passage_max_length,
        )
        metrics, preds = evaluate_model(
            model,
            test_data,
            test_ids,
            cfg,
            save_predictions=True,
        )

    elif setting.kind == "train":
        miner = RelationNegativeMiner(train_data, seed=cfg.seed)
        if is_main:
            write_json(
                out_dir / "negative_statistics.json",
                miner.statistics(train_ids),
            )

        best_dir = train_setting(
            cfg,
            setting,
            train_data,
            dev_data,
            train_ids,
            dev_ids,
            out_dir,
        )

        if not is_main:
            return

        model = DualEncoder.load(
            best_dir,
            cfg.query_max_length,
            cfg.passage_max_length,
        )
        metrics, preds = evaluate_model(
            model,
            test_data,
            test_ids,
            cfg,
            save_predictions=True,
        )

    else:
        raise ValueError(f"Unknown setting kind: {setting.kind}")

    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", preds)

    print(
        f"[{setting.name} | seed={cfg.seed if setting.kind == 'train' else 'N/A'}] "
        f"R@5={metrics['overall'].get('recall@5', 0):.4f} "
        f"nDCG@10={metrics['overall'].get('ndcg@10', 0):.4f} "
        f"MRR={metrics['overall'].get('mrr', 0):.4f}"
    )
