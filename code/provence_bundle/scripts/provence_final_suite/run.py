from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import SentenceEvidenceModel, optimizer_groups  # noqa: E402

from provence_finetune import TrainingScorer, benchmark, write_json  # noqa: E402
from provence_zero_shot import load_fixed_queries, sha256  # noqa: E402


def barrier(world):
    if world > 1:
        dist.barrier()


def sum_gradients(parameters, world):
    """Losses are already divided by GLOBAL query batch size; reduce SUM, not mean."""
    if world == 1:
        return
    bucket, count = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in bucket:
            p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if count + p.numel() > 8_000_000:
            flush()
            bucket, count = [], 0
        bucket.append(p)
        count += p.numel()
    flush()


def labels_for(record, ordered, qrels):
    positives = record["positive_ids"]
    negatives = [r["doc_id"] for r in record["generic"]]
    partials = record["partial_ids"]
    assert len(negatives) == len(set(negatives)) <= 3
    assert all(qrels.get(d, 0) == 2 for d in positives)
    assert all(qrels.get(d, 0) == 0 for d in negatives)
    assert all(qrels.get(d, 0) == 1 for d in partials)
    lookup = {d: i for i, d in enumerate(ordered)}
    return {
        lookup[d]: label
        for docs, label in [(positives, 2), (negatives, 0), (partials, 1)]
        for d in docs
    }


def gradient_norm(module):
    return math.sqrt(
        sum(
            float(p.grad.detach().float().square().sum())
            for p in module.parameters()
            if p.grad is not None
        )
    )


def parameter_fingerprint(model):
    # Compare ranks' full parameter sums, plus squared sums, in FP64 after one update.
    device = next(model.parameters()).device
    result = torch.zeros(2, device=device, dtype=torch.float64)
    for p in model.parameters():
        if p.requires_grad:
            flat = p.detach().double()
            result[0] += flat.sum()
            result[1] += flat.square().sum()
    return result


def eval_rows(model, scorer, data, rows, output, modules, cfg, rank, world, manifest):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    part = output / f"predictions.rank{rank}.jsonl"
    with torch.no_grad(), part.open("w") as handle:
        for index in range(rank, len(rows), world):
            row = rows[index]
            ordered, blocks = scorer.prepare(row, data.corpus)
            scores, _, counts = model(blocks, len(ordered))
            assert len(ordered) == len(set(ordered)) and torch.all(counts > 0)
            assert set(ordered) == set(row["article_local_candidate_doc_ids"])
            assert torch.isfinite(scores).all()
            handle.write(
                json.dumps(
                    {
                        "query_id": row["query_id"],
                        "doc_ids": ordered,
                        "scores": scores.float().cpu().tolist(),
                        "probabilities": model.last_probabilities.cpu().tolist(),
                        "windows": len(blocks),
                        "input_tokens": sum(len(b.input_ids) for b in blocks),
                    }
                )
                + "\n"
            )
            if index % 256 < world:
                print(f"eval {output.name} rank={rank} {index}/{len(rows)}", flush=True)
    barrier(world)
    result = None
    if rank == 0:
        predictions = {}
        for path in sorted(output.glob("predictions.rank*.jsonl")):
            for line in path.open():
                item = json.loads(line)
                assert item["query_id"] not in predictions
                predictions[item["query_id"]] = item
        assert set(predictions) == {r["query_id"] for r in rows}
        ev, config_mod = modules[2], modules[3]
        metric_cfg = config_mod.ExperimentConfig(
            project_root=Path(cfg["data_root"]),
            phase5_dir=Path("."),
            results_dir=output,
        )
        metrics, groups = [], {}
        with (output / "predictions.jsonl").open("w") as handle:
            for row in rows:
                r = predictions[row["query_id"]]
                mapping = dict(zip(r["doc_ids"], r["scores"], strict=True))
                candidates = row["article_local_candidate_doc_ids"]
                values = np.asarray([mapping[d] for d in candidates], dtype=np.float32)
                ranked = [candidates[int(i)] for i in np.argsort(-values)[:10]]
                m = ev._per_query(ranked, data.qrels[row["query_id"]], metric_cfg)
                m["MRR@10"] = m.pop("mrr")
                metrics.append(m)
                groups.setdefault(row["query_type"], []).append(m)
                handle.write(json.dumps(r) + "\n")
        result = {
            "experiment": f"Provence-{cfg['arm']}",
            "training_seed": cfg["seed"],
            "overall": ev._aggregate(metrics),
            "by_query_type": {k: ev._aggregate(v) for k, v in groups.items()},
            "coverage": {
                "expected_queries": len(rows),
                "successful_queries": len(rows),
                "failed_queries": 0,
            },
            "query_manifest_sha256": sha256(manifest),
            "qrels_sha256": sha256(
                Path(
                    cfg.get(
                        "evaluation_qrels",
                        str(
                            Path(cfg["data_root"])
                            / (
                                "test"
                                if output.name in {"test", "full_test"}
                                else "dev"
                            )
                            / "qrels.tsv"
                        ),
                    )
                )
            ),
            "metric_definition": "original evaluator top10; graded nDCG gain2**rel-1; Direct-only recall/MRR@10",
        }
        write_json(output / "formal_metrics.json", result)
        for path in output.glob("predictions.rank*.jsonl"):
            path.unlink()
    barrier(world)
    if world > 1:
        box = [result]
        dist.broadcast_object_list(box, src=0)
        result = box[0]
    return result


def full_test(model, scorer, modules, cfg, rank, world):
    from eval_provence_four_queries import validate_rows

    root = Path(cfg["bundle_root"])
    manifest = root / "four_query_manifest.json"
    assert sha256(manifest) == cfg["full_manifest_sha256"]
    data = modules[0].load_split(Path(cfg["data_root"]) / "test", "test")
    rows = validate_rows(data, json.loads(manifest.read_text()))
    output = Path(cfg["output"]) / "full_test"
    eval_rows(model, scorer, data, rows, output, modules, cfg, rank, world, manifest)



def evaluate_external_test(
    model, scorer, modules, cfg, rank, world, dataset_root, output
):
    """Evaluate a saved checkpoint on a fixed external test directory only."""
    dataset_root = Path(dataset_root)
    output = Path(output)
    data = modules[0].load_split(dataset_root, "test")
    rows = [data.instances[q] for q in sorted(data.instances)]
    manifest = output / "query_manifest.json"
    if rank == 0:
        write_json(
            manifest,
            {
                "split": "test",
                "query_ids": [row["query_id"] for row in rows],
                "source": str(dataset_root),
            },
        )
    barrier(world)
    eval_rows(
        model,
        scorer,
        data,
        rows,
        output,
        modules,
        {**cfg, "evaluation_qrels": str(dataset_root / "qrels.tsv")},
        rank,
        world,
        manifest,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights")
    parser.add_argument("--evaluation-data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert cfg["arm"] in {
        "adapted_provence",
        "without_pe",
    }
    assert cfg["learning_rate"] == 3e-6 and cfg["head_learning_rate"] == 3e-5
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=datetime.timedelta(hours=4),
        )
    if not args.smoke:
        assert device.type == "cuda" and torch.cuda.is_bf16_supported()
        assert (
            cfg["epochs"] == 3
            and cfg["seed"] in {42, 13, 87}
            and cfg["effective_batch_size"] == 16
        )
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root = Path(cfg["bundle_root"])
    for relative, expected in json.loads(
        (root / "input_hashes.json").read_text()
    ).items():
        assert sha256(root / relative) == expected, f"Input changed: {relative}"
    output = Path(cfg["output"])
    code_hashes = {
        str(p.relative_to(root)): sha256(p) for p in (root / "scripts").rglob("*.py")
    }
    if not args.smoke and not args.resume and not args.eval_only:
        proof = json.loads(
            (
                root / "output" / ("smoke_" + cfg["arm"]) / "SMOKE_PASSED.json"
            ).read_text()
        )
        assert proof["arm"] == cfg["arm"] and proof["world_size"] == world
        assert proof["code_sha256"] == code_hashes
        assert proof["input_hashes_sha256"] == sha256(root / "input_hashes.json")
    if rank == 0:
        if args.smoke:
            output.mkdir(parents=True, exist_ok=True)
        elif not args.resume:
            output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / ("smoke_config.json" if args.smoke else "run_config.json"),
            {
                **cfg,
                "world_size": world,
                "torch": torch.__version__,
                "device": str(device),
                "training_hardware": (
                    torch.cuda.get_device_name()
                    if device.type == "cuda"
                    else str(device)
                ),
                "code_sha256": {
                    str(p.relative_to(root)): sha256(p)
                    for p in (root / "scripts").rglob("*.py")
                },
            },
        )
    barrier(world)
    modules = benchmark(cfg["benchmark_root"])
    data_root = Path(cfg["data_root"])
    train = modules[0].load_split(data_root / "train", "train")
    records = [json.loads(line) for line in Path(cfg["sampling"]).open()]
    records = [r for r in records if "skip" not in r]
    # Smoke uses fixed common records plus the longest article, never test labels.
    if args.smoke:
        largest = max(
            records,
            key=lambda r: sum(
                len(train.corpus[d]["text"])
                for d in train.instances[r["query_id"]][
                    "article_local_candidate_doc_ids"
                ]
            ),
        )
        partial = next(r for r in records if r["partial_ids"])
        chosen = records[: max(8, world * 2)] + [largest, partial]
        records = list({r["query_id"]: r for r in chosen}.values())
    scorer = TrainingScorer(cfg["checkpoint"], "cpu", 1)
    scorer.allow_oversized_sentence_fragments = True
    model = SentenceEvidenceModel(
        scorer.model, cfg["arm"], scorer.tokenizer.all_special_ids
    ).to(device)
    model.official.deberta.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if args.eval_only:
        if not args.weights:
            raise ValueError("--weights required")
        model.load_state_dict(load_file(args.weights))
        if rank == 0:
            write_json(
                output / "source_checkpoint.json",
                {"path": args.weights, "sha256": sha256(Path(args.weights))},
            )
        if args.evaluation_data:
            evaluate_external_test(
                model, scorer, modules, cfg, rank, world, args.evaluation_data, output
            )
        else:
            full_test(model, scorer, modules, cfg, rank, world)
        if rank == 0:
            evaluated = args.evaluation_data or "configured full test"
            (output / "COMPLETED").write_text(
                f"Existing checkpoint evaluation complete: {evaluated}\n"
            )
        if world > 1:
            dist.destroy_process_group()
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, cfg["learning_rate"]),
        lr=cfg["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    batch_size = cfg["effective_batch_size"]
    steps_per_epoch = math.ceil(len(records) / batch_size)
    total_steps = steps_per_epoch * (1 if args.smoke else cfg["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0 if args.smoke else math.ceil(total_steps * 0.06), total_steps
    )
    best, history, start_epoch, start_batch, step = -math.inf, [], 1, 0, 0
    latest = output / "latest.pt"
    if args.resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, history, start_epoch, start_batch, step = (
            state[k] for k in ["best", "history", "epoch", "next_batch", "step"]
        )
        torch.set_rng_state(state["rng"][rank]["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rng"][rank]["cuda"], device)

    def checkpoint(epoch, next_batch):
        rng = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, rng)
        else:
            rngs = [rng]
        if rank == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best": best,
                    "history": history,
                    "epoch": epoch,
                    "next_batch": next_batch,
                    "step": step,
                    "rng": rngs,
                },
                output / "latest.tmp",
            )
            (output / "latest.tmp").replace(latest)
        barrier(world)

    smoke_gradients = None
    epochs = 1 if args.smoke else cfg["epochs"]
    for epoch in range(start_epoch, epochs + 1):
        order = list(range(len(records)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        model.train()
        began = time.monotonic()
        for begin in range(
            start_batch if epoch == start_epoch else 0, len(order), batch_size
        ):
            group = order[begin : begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            component_totals = torch.zeros(1, device=device)
            for index in group[rank::world]:
                record = records[index]
                row = train.instances[record["query_id"]]
                ordered, blocks = scorer.prepare(row, train.corpus)
                labels = labels_for(record, ordered, train.qrels[record["query_id"]])
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    scores, loss, counts = model(blocks, len(ordered), labels)
                assert torch.isfinite(loss) and torch.isfinite(scores).all()
                (loss / len(group)).backward()
                total_loss += loss.detach() / len(group)
                component_totals += torch.stack(
                    [model.last_losses[k] for k in ["ternary"]]
                ) / len(group)
            sum_gradients(parameters, world)
            if world > 1:
                dist.all_reduce(total_loss)
                dist.all_reduce(component_totals)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            if args.smoke:
                smoke_gradients = {"encoder": gradient_norm(model.official.deberta)}
                if cfg["arm"] in {
                    "adapted_provence",
                    "without_pe",
                }:
                    smoke_gradients["evidence_head"] = gradient_norm(
                        model.evidence_head
                    )
                if model.use_sentence_interaction:
                    smoke_gradients["cross_window"] = gradient_norm(
                        model.sentence_transformer
                    )
                assert all(v > 0 and math.isfinite(v) for v in smoke_gradients.values())
                assert all(
                    p.grad is None for p in model.official.classifier.parameters()
                )
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0:
                record = {
                    "epoch": epoch,
                    "step": step,
                    "queries": begin + len(group),
                    "loss": float(total_loss),
                    "grad_norm": float(norm),
                    "lr": scheduler.get_last_lr()[0],
                    "learning_rates": dict(
                        zip(
                            [g["name"] for g in optimizer.param_groups],
                            scheduler.get_last_lr(),
                            strict=True,
                        )
                    ),
                    "loss_components": dict(
                        zip(
                            ["ternary"],
                            component_totals.tolist(),
                            strict=True,
                        )
                    ),
                    "seconds": time.monotonic() - began,
                }
                with (output / "train.jsonl").open("a") as log:
                    log.write(json.dumps(record) + "\n")
                if step % 10 == 0 or args.smoke:
                    print(record, flush=True)
            if not args.smoke and step % 200 == 0:
                checkpoint(epoch, begin + len(group))
        if args.smoke:
            if world > 1:
                local = parameter_fingerprint(model)
                copies = [torch.empty_like(local) for _ in range(world)]
                dist.all_gather(copies, local)
                assert all(torch.equal(local, v) for v in copies), "Ranks diverged"
            dev = modules[0].load_split(data_root / "dev", "dev")
            rows = load_fixed_queries(
                root / "shared/dev_relations_all.json", data_root / "dev", "dev"
            )[:8]
            eval_rows(
                model,
                scorer,
                dev,
                rows,
                output / "smoke_eval",
                modules,
                cfg,
                rank,
                world,
                root / "shared/dev_relations_all.json",
            )
            if rank == 0:
                write_json(
                    output / "SMOKE_PASSED.json",
                    {
                        "arm": cfg["arm"],
                        "world_size": world,
                        "gradient_norms": smoke_gradients,
                        "queries": len(records),
                        "optimizer_base_lrs": [
                            g["initial_lr"] for g in optimizer.param_groups
                        ],
                        "coverage_verified": True,
                        "code_sha256": code_hashes,
                        "input_hashes_sha256": sha256(root / "input_hashes.json"),
                    },
                )
            barrier(world)
            break
        dev = modules[0].load_split(data_root / "dev", "dev")
        dev_manifest = root / "shared/dev_relations_all.json"
        dev_rows = load_fixed_queries(dev_manifest, data_root / "dev", "dev")
        metrics = eval_rows(
            model,
            scorer,
            dev,
            dev_rows,
            output / f"dev_epoch{epoch}",
            modules,
            cfg,
            rank,
            world,
            dev_manifest,
        )
        ndcg = metrics["overall"]["ndcg@10"]
        selected = ndcg > best
        history.append(
            {"epoch": epoch, "dev": metrics["overall"], "selected_best": selected}
        )
        if selected:
            best = ndcg
            if rank == 0:
                (output / "best_checkpoint").mkdir(exist_ok=True)
                save_file(
                    {
                        k: v.detach().cpu().contiguous()
                        for k, v in model.state_dict().items()
                    },
                    str(output / "best_checkpoint/model.safetensors"),
                )
                write_json(output / "best_checkpoint/config.json", cfg)
                scorer.tokenizer.save_pretrained(output / "best_checkpoint")
                write_json(output / "best.json", {"epoch": epoch, "dev_ndcg@10": ndcg})
        if rank == 0:
            write_json(output / "history.json", history)
        checkpoint(epoch + 1, 0)
        del dev
    if not args.smoke:
        model.load_state_dict(
            load_file(str(output / "best_checkpoint/model.safetensors"))
        )
        test = modules[0].load_split(data_root / "test", "test")
        manifest = root / "shared/test_relations_all.json"
        rows = load_fixed_queries(manifest, data_root / "test", "test")
        eval_rows(
            model,
            scorer,
            test,
            rows,
            output / "test",
            modules,
            cfg,
            rank,
            world,
            manifest,
        )
        full_test(model, scorer, modules, cfg, rank, world)
        if rank == 0:
            (output / "COMPLETED").write_text(
                "3 epochs; best full-dev checkpoint; fixed test.\n"
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
