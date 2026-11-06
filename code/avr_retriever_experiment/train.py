from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from avr_retriever_experiment.config import ExperimentConfig
from avr_retriever_experiment.data import BenchmarkData
from avr_retriever_experiment.evaluation import evaluate_model
from avr_retriever_experiment.losses import (
    distributed_inbatch_contrastive_loss,
    inbatch_contrastive_loss,
    pairwise_negative_loss,
)
from avr_retriever_experiment.model import DualEncoder, resolve_device
from avr_retriever_experiment.settings import ExperimentSetting
from avr_retriever_experiment.support.io import write_json
from avr_retriever_experiment.training_data import (
    RetrievalTrainingDataset,
    collate_training,
)


def _amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = (
        torch.bfloat16
        if (amp == "bf16" or amp == "auto" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_setting(
    cfg: ExperimentConfig,
    setting: ExperimentSetting,
    train_data: BenchmarkData,
    dev_data: BenchmarkData,
    train_query_ids: Sequence[str],
    dev_query_ids: Sequence[str],
    out_dir: Path,
) -> Path:
    if setting.recipe is None or setting.model_name is None:
        raise ValueError("Training setting requires recipe/model_name")

    distributed = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    if distributed:
        local_rank = int(torch.cuda.current_device())
        device = torch.device("cuda", local_rank)
        if cfg.batch_size % world_size:
            raise ValueError(
                f"Global batch size {cfg.batch_size} must be divisible by "
                f"DDP world size {world_size}"
            )
        local_batch_size = cfg.batch_size // world_size
    else:
        device = resolve_device(cfg.device)
        local_batch_size = cfg.batch_size
    model = DualEncoder(
        setting.model_name, cfg.query_max_length, cfg.passage_max_length
    ).to(device)
    if distributed:
        model.encoder = DistributedDataParallel(
            model.encoder,
            device_ids=[device.index],
            output_device=device.index,
            # Encoder pooler parameters are intentionally unused because all
            # retrievers pool last_hidden_state explicitly.
            find_unused_parameters=True,
        )
    ds = RetrievalTrainingDataset(
        train_data,
        train_query_ids,
        setting.recipe,
        cfg.seed,
        cfg.generic_negatives_per_query,
    )
    sampler = (
        DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=local_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_training,
        drop_last=False,
    )

    optimizer = AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    update_steps_per_epoch = math.ceil(len(loader) / max(cfg.grad_accum, 1))
    total_steps = max(1, cfg.epochs * update_steps_per_epoch)
    warmup = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    history = []
    best_metric = float("-inf")
    best_dir = out_dir / "best_model"
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch - 1)
        model.train()
        running = {"loss": 0.0, "contrastive": 0.0, "negative": 0.0, "steps": 0}
        bar = tqdm(loader, desc=f"{setting.name}-epoch{epoch}", disable=not is_main)
        for step, batch in enumerate(bar, 1):
            with _amp_context(device, cfg.amp):
                q = model.encode_queries(batch["queries"], device)
                pos = model.encode_passages(batch["positives"], device)
                if distributed:
                    l_con = distributed_inbatch_contrastive_loss(
                        q, pos, cfg.temperature, batch["relation_ids"]
                    )
                else:
                    l_con = inbatch_contrastive_loss(
                        q, pos, cfg.temperature, batch["relation_ids"]
                    )

                neg_texts, neg_owners = [], []
                anchor_texts, anchor_owners = [], []
                for i, negs in enumerate(batch["negative_lists"]):
                    for neg in negs:
                        neg_texts.append(neg["text"])
                        neg_owners.append(i)
                        if i not in anchor_owners:
                            anchor_texts.append(neg["text"])
                            anchor_owners.append(i)
                neg_emb = (
                    model.encode_passages(neg_texts, device) if neg_texts else None
                )
                # Preserve the encoder/dropout RNG consumption of the reported runs.
                if anchor_texts:
                    model.encode_passages(anchor_texts, device)
                l_neg = pairwise_negative_loss(
                    q, pos, neg_emb, neg_owners, cfg.temperature
                )

                loss = l_con + cfg.hard_negative_weight * l_neg
                loss = loss / max(cfg.grad_accum, 1)

            loss.backward()
            if step % cfg.grad_accum == 0 or step == len(loader):
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running["loss"] += float(loss.detach().cpu()) * max(cfg.grad_accum, 1)
            running["contrastive"] += float(l_con.detach().cpu())
            running["negative"] += float(l_neg.detach().cpu())
            running["steps"] += 1
            if is_main:
                bar.set_postfix(loss=f"{running['loss']/running['steps']:.4f}")

        if is_main:
            dev_ids = list(dev_query_ids)
            if cfg.dev_eval_max_queries and len(dev_ids) > cfg.dev_eval_max_queries:
                dev_ids = dev_ids[: cfg.dev_eval_max_queries]
            ddp_encoder = model.encoder if distributed else None
            if distributed:
                model.encoder = ddp_encoder.module
            dev_metrics, _ = evaluate_model(
                model, dev_data, dev_ids, cfg, save_predictions=False
            )
            score = float(dev_metrics["overall"].get(cfg.checkpoint_metric, 0.0))
            row = {
                "epoch": epoch,
                "train": {
                    k: (v / running["steps"] if k != "steps" else v)
                    for k, v in running.items()
                },
                "dev": dev_metrics,
                "checkpoint_metric": cfg.checkpoint_metric,
                "checkpoint_score": score,
            }
            history.append(row)
            write_json(out_dir / "training_history.json", history)
            print(
                f"[{setting.name}] epoch={epoch} dev {cfg.checkpoint_metric}={score:.4f}"
            )
            if score > best_metric:
                best_metric = score
                model.save(best_dir)
                write_json(
                    out_dir / "best_checkpoint.json",
                    {"epoch": epoch, "metric": cfg.checkpoint_metric, "score": score},
                )
            if distributed:
                model.encoder = ddp_encoder
        if distributed:
            dist.barrier()

    return best_dir
