from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import get_linear_schedule_with_warmup

from document_level_evidence_ranking.losses import (
    distributed_inbatch_contrastive_loss,
    inbatch_contrastive_loss,
    pairwise_negative_loss,
)
from document_level_evidence_ranking.model import DualEncoder, resolve_device
from .data import CorpusData, balanced_one_per_relation, write_jsonl
from .evaluate import metrics_from_predictions, rank_dense_model


class TrainingDataset(Dataset):
    """Direct positives and ordinary negatives over the whole article corpus.

    Partial evidence is excluded from both positive and negative sampling. A
    negative is either from an explicitly unsupported document (rel=0), or a
    non-evidence sentence inside a judged relevant document.
    """

    def __init__(self, data: CorpusData, qids: list[str], seed: int):
        self.data, self.seed = data, seed
        self.items = []
        for qid in qids:
            positive_ids = [
                sid for sid, rel in data.sentence_qrels.get(qid, {}).items() if rel >= 2
            ]
            if not positive_ids:
                continue
            evidence = {
                sid for sid, rel in data.sentence_qrels.get(qid, {}).items() if rel >= 1
            }
            negative_ids: list[str] = []
            for article_id, rel in data.document_qrels.get(qid, {}).items():
                if rel == 0:
                    negative_ids.extend(data.article_sentence_ids.get(article_id, ()))
                elif rel >= 1:
                    negative_ids.extend(
                        sid
                        for sid in data.article_sentence_ids.get(article_id, ())
                        if sid not in evidence
                    )
            negative_ids = sorted(set(negative_ids) - evidence)
            if negative_ids:
                self.items.append((qid, sorted(positive_ids), negative_ids))
        if not self.items:
            raise ValueError(
                "No trainable queries with both Direct positives and legal negatives"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        qid, positives, negatives = self.items[index]
        rng = random.Random(self.seed * 1_000_003 + index)
        positive_id = rng.choice(positives)
        negative_id = rng.choice(negatives)
        query = self.data.queries[qid]
        return {
            "query_id": qid,
            "relation_id": str(query["relation_id"]),
            "query": query["query"],
            "positive_id": positive_id,
            "positive": self.data.sentences[positive_id]["text"],
            "negative_id": negative_id,
            "negative": self.data.sentences[negative_id]["text"],
        }


def collate(rows: list[dict]) -> dict:
    return {key: [row[key] for row in rows] for key in rows[0]}


def amp_context(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    dtype = (
        torch.bfloat16
        if mode in ("auto", "bf16") and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.autocast("cuda", dtype=dtype)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def setup_distributed() -> tuple[bool, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return True, dist.get_rank(), world, torch.device("cuda", local_rank)
    return False, 0, 1, resolve_device("auto")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dense model for corpus-level evidence ranking (CER).")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset/corpus-level",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--query-view-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16, help="global batch size")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp", default="auto", choices=("auto", "bf16", "fp16", "none")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    distributed, rank, world, device = setup_distributed()
    main_process = rank == 0
    if args.batch_size % world:
        raise ValueError(
            f"global batch size {args.batch_size} is not divisible by world size {world}"
        )
    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_data = CorpusData.load(args.data_dir, "train")
    dev_data = CorpusData.load(args.data_dir, "dev")
    test_data = (
        CorpusData.load(args.data_dir, "test")
        if not args.smoke and main_process
        else None
    )
    train_qids = balanced_one_per_relation(train_data, args.query_view_seed)
    dev_qids = balanced_one_per_relation(dev_data, args.query_view_seed)
    if args.smoke:
        train_qids, dev_qids = train_qids[:64], dev_qids[:16]
        args.epochs = 1
    dataset = TrainingDataset(train_data, train_qids, args.seed)
    # Keep every relation. DistributedSampler may repeat at most world-1 rows
    # to equalize rank lengths; dropping would silently omit relations.
    sampler = (
        DistributedSampler(
            dataset, world, rank, shuffle=True, seed=args.seed, drop_last=False
        )
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size // world,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = DualEncoder(args.model, 128, 256).to(device)
    if distributed:
        model.encoder = DistributedDataParallel(
            model.encoder,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    update_steps = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, update_steps * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    if main_process:
        save_json(
            args.output_dir / "run_config.json",
            vars(args)
            | {
                "data_dir": str(args.data_dir.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "world_size": world,
                "global_batch_size": args.batch_size,
                "distributed_tail_policy": "retain all rows; DistributedSampler may pad at most world_size-1 rows",
                "train_queries": len(dataset),
                "dev_queries": len(dev_qids),
                "partial_supervision": False,
                "ranking_loss": False,
                "negative_policy": "explicit document rel=0 or non-evidence sentence in judged relevant doc; exclude rel>=1",
                "train_manifest": train_data.manifest(),
                "dev_manifest": dev_data.manifest(),
            },
        )
        write_jsonl(
            args.output_dir / "train_query_manifest.jsonl",
            ({"query_id": qid} for qid in train_qids),
        )
        write_jsonl(
            args.output_dir / "dev_query_manifest.jsonl",
            ({"query_id": qid} for qid in dev_qids),
        )
        audit = [dataset[i] for i in range(min(len(dataset), 256))]
        save_json(
            args.output_dir / "negative_sampling_audit.json",
            {
                "checked": len(audit),
                "direct_or_partial_leaks": sum(
                    row["negative_id"]
                    in {
                        sid
                        for sid, rel in train_data.sentence_qrels[
                            row["query_id"]
                        ].items()
                        if rel >= 1
                    }
                    for row in audit
                ),
            },
        )

    best_score, history = float("-inf"), []
    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "contrastive": 0.0, "negative": 0.0, "steps": 0}
        for step, batch in enumerate(loader, 1):
            with amp_context(device, args.amp):
                query_emb = model.encode_queries(batch["query"], device)
                positive_emb = model.encode_passages(batch["positive"], device)
                negative_emb = model.encode_passages(batch["negative"], device)
                contrastive = (
                    distributed_inbatch_contrastive_loss(
                        query_emb, positive_emb, args.temperature, batch["relation_id"]
                    )
                    if distributed
                    else inbatch_contrastive_loss(
                        query_emb, positive_emb, args.temperature, batch["relation_id"]
                    )
                )
                owners = list(range(len(batch["query"])))
                negative = pairwise_negative_loss(
                    query_emb, positive_emb, negative_emb, owners, args.temperature
                )
                loss = (
                    contrastive + args.hard_negative_weight * negative
                ) / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(loader):
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            sums["loss"] += float(loss.detach()) * args.grad_accum
            sums["contrastive"] += float(contrastive.detach())
            sums["negative"] += float(negative.detach())
            sums["steps"] += 1

        if main_process:
            ddp = model.encoder if distributed else None
            if ddp is not None:
                model.encoder = ddp.module
            eval_qids = dev_qids[:16] if args.smoke else dev_qids
            predictions = rank_dense_model(
                dev_data, eval_qids, model, args.eval_batch_size, device
            )
            metrics = metrics_from_predictions(dev_data, predictions)
            score = float(metrics["document"]["overall"]["ndcg@10"])
            row = {
                "epoch": epoch,
                "train": {
                    k: (v / sums["steps"] if k != "steps" else v)
                    for k, v in sums.items()
                },
                "dev": metrics,
                "checkpoint_metric": "document.ndcg@10",
                "checkpoint_score": score,
            }
            history.append(row)
            save_json(args.output_dir / "training_history.json", history)
            epoch_dir = args.output_dir / "checkpoints" / f"epoch_{epoch}"
            model.save(epoch_dir)
            if score > best_score:
                best_score = score
                model.save(args.output_dir / "best_model")
                save_json(
                    args.output_dir / "best_checkpoint.json",
                    {"epoch": epoch, "metric": "document.ndcg@10", "score": score},
                )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "dev_document_ndcg@10": score,
                        "train_loss": row["train"]["loss"],
                    }
                )
            )
            if ddp is not None:
                model.encoder = ddp
        if distributed:
            dist.barrier()

    if main_process and test_data is not None:
        best = DualEncoder.load(args.output_dir / "best_model").to(device)
        rows = rank_dense_model(
            test_data, sorted(test_data.queries), best, args.eval_batch_size, device
        )
        write_jsonl(args.output_dir / "predictions.jsonl", rows)
        save_json(
            args.output_dir / "metrics.json", metrics_from_predictions(test_data, rows)
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
