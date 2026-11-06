from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import get_linear_schedule_with_warmup
from .data import CorpusData, balanced_one_per_relation, read_jsonl, sha256, write_jsonl
from .evaluate import metrics_from_predictions
from .provence_rerank import load_modules, pseudo_row, score_query_finetuned


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def eligible_documents(data: CorpusData, qid: str) -> list[str]:
    """Judged-positive documents containing Direct and ordinary sentence negatives.

    The project's evaluator treats missing sentence qrels as relevance zero. We
    apply that existing default only inside a known-positive, known-status
    document. Unknown query-document pairs are never considered here.
    """
    result = []
    qrels = data.sentence_qrels.get(qid, {})
    known = {
        row["doc_id"]: row
        for row in data.instances.get(qid, [])
        if row.get("annotation_status") != "unknown"
    }
    for article_id, document_rel in data.document_qrels.get(qid, {}).items():
        if document_rel <= 0 or article_id not in known:
            continue
        labels = [qrels.get(sid, 0) for sid in data.article_sentence_ids[article_id]]
        if 2 in labels and 0 in labels:
            result.append(article_id)
    return sorted(result)


def labels_for_document(
    data: CorpusData, qid: str, ordered: list[str]
) -> dict[int, int]:
    qrels = data.sentence_qrels.get(qid, {})
    labels = {index: int(qrels.get(sid, 0)) for (index, sid) in enumerate(ordered)}
    if 2 not in labels.values() or 0 not in labels.values():
        raise ValueError(
            f"Training document needs Direct and ordinary negatives: {qid}"
        )
    if any((value not in (0, 1, 2) for value in labels.values())):
        raise ValueError(f"Invalid ternary label: {qid}")
    return labels


def evaluate(
    model,
    scorer,
    data: CorpusData,
    qids: list[str],
    candidates: dict[str, dict],
    depth: int,
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
        values = score_query_finetuned(
            model, prepared, 8, scorer.tokenizer.pad_token_id
        )
        (doc_scores, sentence_scores) = ([], [])
        for aid, (ordered, _), scores in zip(selected, prepared, values, strict=True):
            doc_scores.append(max(scores))
            sentence_scores.extend(zip(ordered, scores))
        order = np.lexsort(
            (np.asarray(selected), -np.asarray(doc_scores, dtype=np.float32))
        )
        reranked = [selected[int(i)] for i in order]
        reranked_scores = [float(doc_scores[int(i)]) for i in order]
        selected_set = set(selected)
        remainder = [aid for aid in raw_docs if aid not in selected_set]
        if len(raw_docs) < len(article_ids):
            seen = set(raw_docs)
            remainder.extend((aid for aid in article_ids if aid not in seen))
        sentence_scores.sort(key=lambda item: (-item[1], item[0]))
        ranked_sentences = [sid for (sid, _) in sentence_scores]
        seen_sentences = set(ranked_sentences)
        ranked_sentences.extend(
            (sid for sid in sentence_ids if sid not in seen_sentences)
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
                "sentence_scores": [float(v) for (_, v) in sentence_scores[:100]],
                "sentence_ranking_depth": min(100, len(sentence_scores)),
            }
        )
        if offset % 50 == 0:
            print(
                json.dumps(
                    {"evaluation": output.name, "queries": offset, "total": len(qids)}
                ),
                flush=True,
            )
    write_jsonl(output / "predictions.jsonl", rows)
    metrics = metrics_from_predictions(data, rows)
    save_json(output / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=("adapted_provence",))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--query-view-seed", type=int, default=42)
    parser.add_argument("--dev-candidates", type=Path, required=True)
    parser.add_argument("--test-candidates", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-06)
    parser.add_argument("--head-learning-rate", type=float, default=3e-05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke and (args.seed not in (13, 42, 87) or args.epochs != 3):
        raise ValueError("Formal runs require seed 13/42/87 and exactly 3 epochs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if not args.smoke and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Formal Provence training requires a BF16 CUDA device")
    torch.set_num_threads(4)
    train = CorpusData.load(args.data_dir, "train")
    dev = CorpusData.load(args.data_dir, "dev")
    test = CorpusData.load(args.data_dir, "test")
    train_qids_all = balanced_one_per_relation(train, args.query_view_seed)
    dev_qids = balanced_one_per_relation(dev, args.query_view_seed)
    choices = {qid: eligible_documents(train, qid) for qid in train_qids_all}
    train_qids = [qid for qid in train_qids_all if choices[qid]]
    skipped = [qid for qid in train_qids_all if not choices[qid]]
    if args.smoke:
        (train_qids, dev_qids) = (train_qids[:16], dev_qids[:8])
        args.epochs = 1
    dev_candidates = {r["query_id"]: r for r in read_jsonl(args.dev_candidates)}
    test_candidates = {r["query_id"]: r for r in read_jsonl(args.test_candidates)}
    for name, qids, rows in (
        ("dev", dev_qids, dev_candidates),
        ("test", sorted(test.queries), test_candidates),
    ):
        missing = set(qids) - set(rows)
        if missing:
            raise ValueError(
                f"{name} first-stage candidates miss {len(missing)} queries"
            )
    (TrainingScorer, SentenceEvidenceModel) = load_modules(args.bundle)
    scorer = TrainingScorer(args.checkpoint, "cpu", 1)
    scorer.allow_oversized_sentence_fragments = True
    model = SentenceEvidenceModel(
        scorer.model, args.variant, scorer.tokenizer.all_special_ids
    ).to(device)
    initial_interaction_gate = None
    model.official.deberta.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    inference_equivalence = None
    if args.smoke:
        audit_qid = dev_qids[0]
        audit_docs = dev_candidates[audit_qid]["ranked_document_ids"][:3]
        audit_prepared = [
            scorer.prepare(pseudo_row(dev, audit_qid, aid), dev.sentences)
            for aid in audit_docs
        ]
        model.eval()
        with torch.inference_mode():
            reference = [
                model(blocks, len(ordered))[0].float().cpu()
                for (ordered, blocks) in audit_prepared
            ]
        batched = score_query_finetuned(
            model, audit_prepared, 8, scorer.tokenizer.pad_token_id
        )
        max_error = max(
            (
                float((expected - torch.tensor(actual)).abs().max())
                for (expected, actual) in zip(reference, batched, strict=True)
            )
        )
        if max_error > 1e-05:
            raise RuntimeError(f"Batched inference differs from reference: {max_error}")
        inference_equivalence = {
            "query_id": audit_qid,
            "documents": len(audit_docs),
            "max_absolute_error": max_error,
            "passed": True,
        }
    sys.path.insert(0, str(args.bundle / "scripts"))
    from provence_final_suite.model import optimizer_groups

    groups = optimizer_groups(model, args.learning_rate)
    groups[1]["lr"] = args.head_learning_rate
    optimizer = torch.optim.AdamW(
        groups, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08
    )
    steps_per_epoch = math.ceil(len(train_qids) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        0 if args.smoke else math.ceil(total_steps * args.warmup_ratio),
        total_steps,
    )
    save_json(
        args.output_dir / "run_config.json",
        {
            **{
                k: str(v) if isinstance(v, Path) else v for (k, v) in vars(args).items()
            },
            "train_query_count_before_eligibility": len(train_qids_all),
            "train_query_count": len(train_qids),
            "skipped_without_direct_and_ordinary_negative": len(skipped),
            "dev_query_count": len(dev_qids),
            "test_query_count": len(test.queries),
            "query_document_sampling": "one judged-positive document per query per epoch; deterministic by training seed/epoch/query index",
            "sentence_default_qrel": "missing sentence qrel in a known-positive document is relevance 0 (project evaluator rule)",
            "partial_policy": "label 1 retained in audit/model input but excluded from final-supervision CE",
            "unknown_policy": "annotation_status=unknown query-document pairs excluded; never treated as negative",
            "ranking_loss": False,
            "sentence_interaction": False,
            "sentence_position_encoding": False,
            "initialization": "pinned official Provence checkpoint; fresh task modules for every seed",
            "train_manifest": train.manifest(),
            "dev_manifest": dev.manifest(),
            "test_manifest": test.manifest(),
            "dev_candidates_sha256": sha256(args.dev_candidates),
            "test_candidates_sha256": sha256(args.test_candidates),
        },
    )
    write_jsonl(
        args.output_dir / "train_query_manifest.jsonl",
        ({"query_id": qid} for qid in train_qids_all),
    )
    write_jsonl(
        args.output_dir / "dev_query_manifest.jsonl",
        ({"query_id": qid} for qid in dev_qids),
    )
    save_json(
        args.output_dir / "training_eligibility_audit.json",
        {
            "eligible_queries": len(train_qids),
            "skipped_queries": skipped,
            "unknown_instances_total": sum(
                (
                    row.get("annotation_status") == "unknown"
                    for rows in train.instances.values()
                    for row in rows
                )
            ),
            "partial_is_negative": False,
        },
    )
    (best, history, global_step) = (-math.inf, [], 0)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_qids)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        started = time.time()
        epoch_loss = 0.0
        for begin in range(0, len(order), args.batch_size):
            group = order[begin : begin + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for index in group:
                qid = train_qids[index]
                rng = random.Random(args.seed * 1000003 + epoch * 100003 + index)
                article_id = rng.choice(choices[qid])
                (ordered, blocks) = scorer.prepare(
                    pseudo_row(train, qid, article_id), train.sentences
                )
                labels = labels_for_document(train, qid, ordered)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    (scores, loss, counts) = model(blocks, len(ordered), labels)
                if (
                    not torch.isfinite(loss)
                    or not torch.isfinite(scores).all()
                    or (not torch.all(counts > 0))
                ):
                    raise RuntimeError(f"Nonfinite train result: {qid}/{article_id}")
                (loss / len(group)).backward()
                group_loss += float(loss.detach()) / len(group)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += group_loss
            if global_step % 25 == 0 or args.smoke:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "queries": begin + len(group),
                            "loss": group_loss,
                            "grad_norm": float(norm),
                            "elapsed": time.time() - started,
                        }
                    ),
                    flush=True,
                )
        metrics = evaluate(
            model,
            scorer,
            dev,
            dev_qids,
            dev_candidates,
            args.candidate_depth,
            args.output_dir / f"dev_epoch_{epoch}",
        )
        score = float(metrics["document"]["overall"]["ndcg@10"])
        selected = score > best
        current_gate = None
        history.append(
            {
                "epoch": epoch,
                "train_loss_sum": epoch_loss,
                "dev_document_ndcg@10": score,
                "selected_best": selected,
            }
        )
        epoch_dir = args.output_dir / "checkpoints" / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for (k, v) in model.state_dict().items()},
            str(epoch_dir / "model.safetensors"),
        )
        if selected:
            best = score
            best_dir = args.output_dir / "best_checkpoint"
            best_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for (k, v) in model.state_dict().items()
                },
                str(best_dir / "model.safetensors"),
            )
            save_json(best_dir / "config.json", vars(args))
            save_json(
                args.output_dir / "best.json",
                {"epoch": epoch, "dev_document_ndcg@10": score},
            )
        save_json(args.output_dir / "history.json", history)
    if args.smoke:
        save_json(
            args.output_dir / "SMOKE_PASSED.json",
            {
                "queries": len(train_qids),
                "finite_loss": True,
                "unknown_excluded": True,
                "partial_not_negative": True,
                "position_encoding_disabled": not model.use_sentence_position_encoding,
                "batched_inference_equivalence": inference_equivalence,
            },
        )
        return
    model.load_state_dict(
        load_file(str(args.output_dir / "best_checkpoint/model.safetensors"))
    )
    evaluate(
        model,
        scorer,
        test,
        sorted(test.queries),
        test_candidates,
        args.candidate_depth,
        args.output_dir / "test",
    )
    (args.output_dir / "COMPLETED").write_text(
        "3 epochs; best full-dev document nDCG@10 checkpoint; fixed full Q1-Q4 test.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
