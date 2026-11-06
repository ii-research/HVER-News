from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import load_file, save_file
from transformers import get_linear_schedule_with_warmup

from .data import CorpusData, balanced_one_per_relation, read_jsonl, sha256, write_jsonl
from .evaluate import metrics_from_predictions
from .provence_hierarchical_model import HierarchicalProvence
from .baselines.provence import load_modules, pseudo_row
from .provence_train import eligible_documents, save_json


VARIANTS = {
    "no_interaction": (False, False),
    "sentence_only": (True, False),
    "document_only": (False, True),
    "hierarchical": (True, True),
}


def known_documents(data: CorpusData, qid: str) -> set[str]:
    return {
        str(row["doc_id"])
        for row in data.instances.get(qid, [])
        if row.get("annotation_status") != "unknown"
    }


def explicit_negative_documents(data: CorpusData, qid: str) -> list[str]:
    known = known_documents(data, qid)
    return sorted(
        doc_id
        for doc_id, relevance in data.document_qrels.get(qid, {}).items()
        if relevance == 0 and doc_id in known
    )


def sentence_labels(data: CorpusData, qid: str, ordered: list[str]) -> list[int]:
    qrels = data.sentence_qrels.get(qid, {})
    labels = [int(qrels.get(sid, 0)) for sid in ordered]
    if any(label not in (0, 1, 2) for label in labels):
        raise ValueError(f"Invalid sentence label: {qid}")
    return labels


def optimizer_groups(
    model: HierarchicalProvence, base_lr: float, head_lr: float, interaction_lr: float
):
    groups = {"encoder": [], "heads": [], "interaction": [], "gates": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("official.deberta."):
            groups["encoder"].append(parameter)
        elif name.endswith("gate_logit"):
            groups["gates"].append(parameter)
        elif "interaction" in name or "document_to_sentence" in name:
            groups["interaction"].append(parameter)
        else:
            groups["heads"].append(parameter)
    return [
        {"params": groups["encoder"], "lr": base_lr, "name": "encoder"},
        {"params": groups["heads"], "lr": head_lr, "name": "heads"},
        {"params": groups["interaction"], "lr": interaction_lr, "name": "interaction"},
        {"params": groups["gates"], "lr": head_lr, "name": "gates"},
    ]


@torch.inference_mode()
def evaluate(
    model: HierarchicalProvence,
    scorer,
    data: CorpusData,
    qids: list[str],
    candidates: dict[str, dict],
    depth: int,
    block_batch_size: int,
    output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    article_ids = sorted(data.articles)
    sentence_ids = sorted(data.sentences)
    rows = []
    model.eval()
    for offset, qid in enumerate(qids, 1):
        raw_docs = candidates[qid]["ranked_document_ids"]
        selected = raw_docs[:depth]
        prepared = [
            scorer.prepare(pseudo_row(data, qid, aid), data.sentences)
            for aid in selected
        ]
        result = model(prepared, block_batch_size, scorer.tokenizer.pad_token_id)
        document_scores = result.document_scores.cpu().tolist()
        scored_sentences = [
            (sid, float(score))
            for (ordered, _), values in zip(
                prepared, result.sentence_scores, strict=True
            )
            for sid, score in zip(ordered, values.cpu().tolist(), strict=True)
        ]
        doc_order = np.lexsort(
            (np.asarray(selected), -np.asarray(document_scores, dtype=np.float32))
        )
        reranked = [selected[int(index)] for index in doc_order]
        reranked_scores = [float(document_scores[int(index)]) for index in doc_order]
        selected_set = set(selected)
        remainder = [doc_id for doc_id in raw_docs if doc_id not in selected_set]
        if len(raw_docs) < len(article_ids):
            seen = set(raw_docs)
            remainder.extend(doc_id for doc_id in article_ids if doc_id not in seen)
        scored_sentences.sort(key=lambda item: (-item[1], item[0]))
        ranked_sentences = [sid for sid, _ in scored_sentences]
        seen_sentences = set(ranked_sentences)
        ranked_sentences.extend(
            sid for sid in sentence_ids if sid not in seen_sentences
        )
        rows.append(
            {
                "query_id": qid,
                "query": data.queries[qid]["query"],
                "query_type": data.queries[qid].get("query_type"),
                "relation_id": data.queries[qid].get("relation_id"),
                "source_article_id": data.source_article(qid),
                "ranked_document_ids": reranked + remainder,
                "document_scores": reranked_scores,
                "document_ranking_is_complete": True,
                "reranked_document_depth": len(selected),
                "ranked_sentence_ids": ranked_sentences[:100],
                "sentence_scores": [
                    float(value) for _, value in scored_sentences[:100]
                ],
                "sentence_ranking_depth": min(100, len(scored_sentences)),
            }
        )
        if offset % 50 == 0:
            print(
                json.dumps({"evaluation_queries": offset, "total": len(qids)}),
                flush=True,
            )
    write_jsonl(output / "predictions.jsonl", rows)
    metrics = metrics_from_predictions(data, rows)
    save_json(output / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hierarchical Provence variants with a fixed query view"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-view-seed", type=int, default=42)
    parser.add_argument("--dev-candidates", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--query-set-batch-size", type=int, default=4)
    parser.add_argument("--block-batch-size", type=int, default=8)
    parser.add_argument("--negative-documents", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--head-learning-rate", type=float, default=3e-5)
    parser.add_argument("--interaction-learning-rate", type=float, default=3e-6)
    parser.add_argument("--document-loss-weight", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=240,
        help="NCCL collective timeout; must cover rank-0 full development evaluation",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    is_main = rank == 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.smoke and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Formal development runs require a BF16 CUDA device")
    torch.set_num_threads(4)

    train = CorpusData.load(args.data_dir, "train")
    dev = CorpusData.load(args.data_dir, "dev")
    train_view = balanced_one_per_relation(train, args.query_view_seed)
    dev_qids = balanced_one_per_relation(dev, args.query_view_seed)
    positives = {qid: eligible_documents(train, qid) for qid in train_view}
    negatives = {qid: explicit_negative_documents(train, qid) for qid in train_view}
    train_qids = [qid for qid in train_view if positives[qid] and negatives[qid]]
    if args.smoke:
        train_qids, dev_qids, args.epochs = train_qids[:8], dev_qids[:8], 1
    if not train_qids:
        raise ValueError(
            "No queries have both a positive document and explicit negative document"
        )

    dev_candidates = {row["query_id"]: row for row in read_jsonl(args.dev_candidates)}
    missing = set(dev_qids) - set(dev_candidates)
    if missing:
        raise ValueError(f"Dev candidates miss {len(missing)} queries")

    TrainingScorer, _ = load_modules(args.bundle)
    scorer = TrainingScorer(args.checkpoint, "cpu", 1)
    scorer.allow_oversized_sentence_fragments = True
    sys.path.insert(0, str(args.bundle / "scripts"))
    from provence_final_suite.model import SentenceEvidenceModel  # type: ignore  # noqa:F401

    sentence_si, document_si = VARIANTS[args.variant]
    model = HierarchicalProvence(
        scorer.model,
        scorer.tokenizer.all_special_ids,
        use_sentence_interaction=sentence_si,
        use_document_interaction=document_si,
    ).to(device)
    model.official.deberta.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups(
            model,
            args.learning_rate,
            args.head_learning_rate,
            args.interaction_learning_rate,
        ),
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    steps_per_epoch = math.ceil(
        len(train_qids) / (args.query_set_batch_size * world_size)
    )
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        0 if args.smoke else math.ceil(total_steps * args.warmup_ratio),
        total_steps,
    )

    if is_main:
        save_json(
            args.output_dir / "run_config.json",
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "sentence_interaction": sentence_si,
                "document_interaction": document_si,
                "sentence_position_encoding": False,
                "ranking_loss": False,
                "partial_policy": "excluded from CE; never converted to negative",
                "unknown_policy": "excluded; only explicit known document relevance 0 is negative",
                "checkpoint_metric": "dev sentence nDCG@10",
                "train_query_view_count": len(train_view),
                "train_query_count_with_explicit_document_negatives": len(train_qids),
                "distributed_world_size": world_size,
                "per_gpu_query_set_batch_size": args.query_set_batch_size,
                "effective_query_set_batch_size": args.query_set_batch_size
                * world_size,
                "train_manifest": train.manifest(),
                "dev_manifest": dev.manifest(),
                "dev_candidates_sha256": sha256(args.dev_candidates),
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

    train_model = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if distributed
        else model
    )

    best, history, global_step = -math.inf, [], 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_qids)))
        random.Random(args.seed + epoch).shuffle(order)
        total_items = steps_per_epoch * args.query_set_batch_size * world_size
        if len(order) < total_items:
            order.extend(order[: total_items - len(order)])
        order = order[rank:total_items:world_size]
        train_model.train()
        started = time.time()
        epoch_loss = 0.0
        for begin in range(0, len(order), args.query_set_batch_size):
            group = order[begin : begin + args.query_set_batch_size]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            last_parts = {}
            for index in group:
                qid = train_qids[index]
                rng = random.Random(args.seed * 1_000_003 + epoch * 100_003 + index)
                positive = rng.choice(positives[qid])
                pool = negatives[qid]
                count = min(args.negative_documents, len(pool))
                chosen_negatives = (
                    rng.sample(pool, count) if len(pool) > count else list(pool)
                )
                documents = [positive, *chosen_negatives]
                prepared = [
                    scorer.prepare(pseudo_row(train, qid, doc_id), train.sentences)
                    for doc_id in documents
                ]
                labels = [
                    sentence_labels(train, qid, ordered) for ordered, _ in prepared
                ]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    result = train_model(
                        prepared, args.block_batch_size, scorer.tokenizer.pad_token_id
                    )
                    loss, last_parts = model.loss(
                        result,
                        labels,
                        [1, *([0] * len(chosen_negatives))],
                        args.document_loss_weight,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss: {qid}")
                (loss / len(group)).backward()
                group_loss += float(loss.detach()) / len(group)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += group_loss
            if is_main and (global_step % 25 == 0 or args.smoke):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "queries": begin + len(group),
                            "loss": group_loss,
                            "grad_norm": float(grad_norm),
                            "parts": last_parts,
                            "elapsed": time.time() - started,
                        }
                    ),
                    flush=True,
                )

        if distributed:
            dist.barrier()
        if is_main:
            metrics = evaluate(
                model,
                scorer,
                dev,
                dev_qids,
                dev_candidates,
                args.candidate_depth,
                args.block_batch_size,
                args.output_dir / f"dev_epoch_{epoch}",
            )
            score = float(metrics["sentence"]["overall"]["ndcg@10"])
            selected = score > best
            gates = {
                "sentence": float(model.sentence_interaction.gate_logit.detach().cpu()),
                "document": float(model.document_gate_logit.detach().cpu()),
                "feedback": float(model.feedback_gate_logit.detach().cpu()),
            }
            history.append(
                {
                    "epoch": epoch,
                    "train_loss_sum_rank0": epoch_loss,
                    "dev_sentence_ndcg@10": score,
                    "dev_document_ndcg@10": float(
                        metrics["document"]["overall"]["ndcg@10"]
                    ),
                    "selected_best": selected,
                    "gates": gates,
                }
            )
            epoch_dir = args.output_dir / "checkpoints" / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                },
                str(epoch_dir / "model.safetensors"),
            )
            if selected:
                best = score
                best_dir = args.output_dir / "best_checkpoint"
                best_dir.mkdir(parents=True, exist_ok=True)
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in model.state_dict().items()
                    },
                    str(best_dir / "model.safetensors"),
                )
                save_json(
                    args.output_dir / "best.json",
                    {"epoch": epoch, "dev_sentence_ndcg@10": score, "gates": gates},
                )
            save_json(args.output_dir / "history.json", history)
        if distributed:
            dist.barrier()

    if args.smoke and is_main:
        save_json(
            args.output_dir / "SMOKE_PASSED.json",
            {
                "finite_loss": True,
                "queries": len(train_qids),
                "position_encoding_disabled": True,
                "unknown_as_negative": False,
                "partial_as_negative": False,
            },
        )
    elif not args.smoke and is_main:
        (args.output_dir / "DEV_COMPLETED").write_text(
            f"training seed {args.seed}; development checkpoint selection only; test not accessed.\n",
            encoding="utf-8",
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
