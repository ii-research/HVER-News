"""Controlled task adaptation of the exact official Provence pruning checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from provence_zero_shot import (
    MODEL,
    REVISION,
    Block,
    Scorer,
    load_fixed_queries,
    normalize_query,
    sha256,
)

NAMES = {"generic": "Provence + Generic FT"}
METRICS = ["recall@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "MRR@10"]


def pack_oversized_sentences(query_ids, sentences, cls_id, sep_id, max_length=512):
    """Exceptional oversized sentences only; retain every token and original sentence ID."""
    prefix = [cls_id, *query_ids, sep_id]
    budget = max_length - len(prefix) - 1
    if not query_ids or budget <= 0:
        raise ValueError("Empty or oversized query")
    blocks = []
    ids, owners = list(prefix), [-1] * len(prefix)
    for index, tokens in enumerate(sentences):
        if not tokens:
            raise ValueError("Empty sentence")
        if len(tokens) > budget:
            if len(ids) > len(prefix):
                blocks.append(Block(ids + [sep_id], owners + [-1]))
                ids, owners = list(prefix), [-1] * len(prefix)
            for begin in range(0, len(tokens), budget):
                fragment = tokens[begin : begin + budget]
                blocks.append(
                    Block(
                        prefix + fragment + [sep_id],
                        [-1] * len(prefix) + [index] * len(fragment) + [-1],
                    )
                )
        else:
            if len(ids) + len(tokens) + 1 > max_length:
                blocks.append(Block(ids + [sep_id], owners + [-1]))
                ids, owners = list(prefix), [-1] * len(prefix)
            ids.extend(tokens)
            owners.extend([index] * len(tokens))
    if len(ids) > len(prefix):
        blocks.append(Block(ids + [sep_id], owners + [-1]))
    return blocks


class TrainingScorer(Scorer):
    allow_oversized_sentence_fragments = False

    def prepare(self, row, corpus):
        try:
            return super().prepare(row, corpus)
        except ValueError as exc:
            if (
                not self.allow_oversized_sentence_fragments
                or "cannot preserve whole sentence" not in str(exc)
            ):
                raise
        ordered = sorted(
            row["article_local_candidate_doc_ids"],
            key=lambda d: corpus[d]["sentence_id"],
        )
        sentences = [
            self.cache[("first:" if i == 0 else "next:") + doc]
            for i, doc in enumerate(ordered)
        ]
        query_ids = self.tokenizer.encode(
            normalize_query(row["query"]), add_special_tokens=False
        )
        return ordered, pack_oversized_sentences(
            query_ids,
            sentences,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def benchmark(root):
    sys.path.insert(0, str(Path(root).resolve()))
    return tuple(
        importlib.import_module("avr_retriever_experiment." + name)
        for name in ("data", "negative_mining", "evaluation", "config")
    )


def sample_query(data, miner, row, index, seed=42):
    """Match the original Generic FT positive rule; freeze the ordinary negative sample."""
    qid = row["query_id"]
    qrels = data.qrels.get(qid, {})
    candidates = row["article_local_candidate_doc_ids"]
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"Duplicate article candidate: {qid}")
    if any(d not in data.corpus for d in candidates):
        raise ValueError(f"Missing article candidate: {qid}")
    if any(str(data.corpus[d]["guid"]) != str(row["guid"]) for d in candidates):
        raise ValueError(f"Cross-document candidate: {qid}")
    if any(r not in (0, 1, 2) for r in qrels.values()):
        raise ValueError("Expected ternary qrels")
    # Preserve insertion order and the exact RNG/idx rule of RetrievalTrainingDataset.
    direct = [d for d, r in qrels.items() if r == 2 and d in data.corpus]
    if not direct:
        return {"query_id": qid, "skip": "no_direct"}
    if any(d not in candidates for d in direct):
        raise ValueError(f"Direct evidence outside article: {qid}")
    rng = random.Random(seed * 1000003 + index)
    positive = rng.choice(direct)
    eligible = [d for d in candidates if qrels.get(d, 0) == 0]
    if not eligible:
        return {"query_id": qid, "skip": "no_negative"}
    n = min(3, len(eligible))
    generic = rng.sample(eligible, n) if len(eligible) > n else list(eligible)
    return {
        "query_id": qid,
        "relation_id": row["relation_id"],
        "query_type": row["query_type"],
        "positive_ids": [positive],
        "generic": [{"doc_id": d, "type": "random"} for d in generic],
        "negative_count": n,
        "statistics": {},
    }


def validate_reference_population(rows, reference):
    """Match this dataset version's zero-shot population, not an older run's size."""
    expected = reference["coverage"]["expected_queries"]
    if len(rows) != expected or reference["coverage"]["successful_queries"] != expected:
        raise ValueError("Incomplete or mismatched zero-shot reference population")
    expected_types = {
        key: value["queries"] for key, value in reference["by_query_type"].items()
    }
    if dict(Counter(row["query_type"] for row in rows)) != expected_types:
        raise ValueError("Query type counts differ from the zero-shot reference")


def prepare(cfg):
    target = Path(cfg["prepared"])
    if (target / "contract.json").exists():
        verify_contract(cfg)
        print("Existing immutable sampling manifest verified", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    data_mod, mining, evaluation, config_mod = benchmark(cfg["benchmark_root"])
    reference = json.loads(Path(cfg["zero_shot_metrics"]).read_text())
    files = {}
    populations = {}
    article_sets = {}
    stats = Counter()
    sample_rows = []
    for split in ("train", "dev", "test"):
        source = Path(cfg["data_root"]) / split
        manifest = Path(cfg["query_views"]) / f"{split}_relations_all.json"
        rows = load_fixed_queries(manifest, source, split)
        for path in [
            manifest,
            *(
                source / f
                for f in (
                    "corpus.jsonl",
                    "queries.jsonl",
                    "instances.jsonl",
                    "qrels.tsv",
                )
            ),
        ]:
            files[str(path.resolve())] = sha256(path)
        shutil.copy2(manifest, target / manifest.name)
        article_sets[split] = {str(r["guid"]) for r in rows}
        populations[split] = {
            "queries": len(rows),
            "types": dict(Counter(r["query_type"] for r in rows)),
        }
        if split == "test":
            validate_reference_population(rows, reference)
            assert sha256(manifest) == reference["query_manifest_sha256"]
            assert sha256(source / "qrels.tsv") == reference["qrels_sha256"]
        if split == "train":
            data = data_mod.load_split(source, split)
            miner = mining.RelationNegativeMiner(data, seed=42)
            for i, row in enumerate(rows):
                sampled = sample_query(data, miner, row, i)
                sample_rows.append(sampled)
                if "skip" in sampled:
                    stats["skipped_" + sampled["skip"]] += 1
                else:
                    stats["eligible_queries"] += 1
                    stats["negative_slots_per_arm"] += sampled["negative_count"]
                    stats.update(sampled["statistics"])
                    stats[f"queries_with_{sampled['negative_count']}_negatives"] += 1
            del data, miner
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not article_sets[a] & article_sets[b], f"Article leakage {a}/{b}"
    for module in (data_mod, mining, evaluation, config_mod):
        files[str(Path(str(module.__file__)).resolve())] = sha256(module.__file__)
    assert sha256(evaluation.__file__) == reference["existing_evaluator_sha256"]
    out = target / "sampling.jsonl"
    with out.open("w") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = stats["eligible_queries"]
    if not n:
        raise ValueError("No trainable queries")
    summary: dict[str, Any] = dict(stats)
    summary.update(
        {
            "sampling_seed": 42,
            "generic_negative_slots": stats["negative_slots_per_arm"],
            "queries": n,
        }
    )
    write_json(target / "negative_statistics.json", summary)
    write_json(
        target / "contract.json",
        {
            "files": files,
            "populations": populations,
            "sampling_sha256": sha256(out),
            "sampling_seed": 42,
            "query_view_seed": 42,
            "zero_shot_reference": reference,
            "protocol": "one shared Direct positive, unique qrel=0 negatives; Partial ignored; immutable across epochs/seeds",
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def verify_contract(cfg):
    prepared = Path(cfg["prepared"])
    contract = json.loads((prepared / "contract.json").read_text())
    for path, digest in contract["files"].items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed after sampling: {path}")
    if sha256(prepared / "sampling.jsonl") != contract["sampling_sha256"]:
        raise ValueError("Sampling manifest changed")
    for split in ("train", "dev", "test"):
        original = Path(cfg["query_views"]) / f"{split}_relations_all.json"
        if sha256(prepared / original.name) != sha256(original):
            raise ValueError("Snapshot differs from original fixed query manifest")
    return contract


def preflight(cfg):
    contract = verify_contract(cfg)
    data_mod = benchmark(cfg["benchmark_root"])[0]
    scorer = object.__new__(TrainingScorer)
    scorer.allow_oversized_sentence_fragments = cfg.get(
        "allow_oversized_sentence_fragments", False
    )
    scorer.tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    scorer.cache = {}
    special_ids = set(scorer.tokenizer.all_special_ids)
    records = {
        r["query_id"]: r
        for r in (
            json.loads(line)
            for line in (Path(cfg["prepared"]) / "sampling.jsonl").open()
        )
        if "skip" not in r
    }
    stats = {}
    for split in ("train", "dev", "test"):
        source = Path(cfg["data_root"]) / split
        data = data_mod.load_split(source, split)
        rows = load_fixed_queries(
            Path(cfg["prepared"]) / f"{split}_relations_all.json", source, split
        )
        counts = Counter()
        for row in rows:
            ordered, blocks = scorer.prepare(row, data.corpus)
            seen = Counter()
            for block in blocks:
                owners = {
                    s
                    for s, t in zip(block.sentence_indices, block.input_ids)
                    if s >= 0 and t not in special_ids
                }
                seen.update(owners)
            if set(seen) != set(range(len(ordered))):
                raise ValueError(
                    f"Incomplete or overlapping sentence windows: {row['query_id']}"
                )
            recovered = defaultdict(list)
            for block in blocks:
                for owner, token in zip(block.sentence_indices, block.input_ids):
                    if owner >= 0:
                        recovered[owner].append(token)
            for i, doc in enumerate(ordered):
                expected = scorer.cache[("first:" if i == 0 else "next:") + doc]
                if recovered[i] != expected:
                    raise ValueError(
                        f"Lost/repeated/reordered body tokens: {row['query_id']}/{doc}"
                    )
            counts["fragmented_sentence_occurrences"] += sum(
                n > 1 for n in seen.values()
            )
            counts["queries_with_fragmented_sentence"] += int(
                any(n > 1 for n in seen.values())
            )
            counts["queries"] += 1
            counts["windows"] += len(blocks)
            counts["candidates"] += len(ordered)
            counts["max_windows"] = max(counts["max_windows"], len(blocks))
            if split == "train" and row["query_id"] in records:
                for arm in NAMES:
                    supervision(
                        (ordered, blocks),
                        records[row["query_id"]],
                        arm,
                        data.qrels[row["query_id"]],
                        special_ids,
                    )
            if counts["queries"] % 10000 == 0:
                print(f"preflight {split}: {dict(counts)}", flush=True)
        if split == "test" and counts["queries_with_fragmented_sentence"]:
            raise ValueError(
                "Fixed test must retain the exact original zero-shot window protocol"
            )
        stats[split] = dict(counts)
        del data
    result = {
        "passed": True,
        "sampling_sha256": contract["sampling_sha256"],
        "coverage": stats,
        "token_overlap": False,
        "oversized_sentence_fragments": cfg.get(
            "allow_oversized_sentence_fragments", False
        ),
    }
    write_json(Path(cfg["prepared"]) / "preflight.json", result)
    print(json.dumps(result, indent=2), flush=True)


def block_loss(
    logits,
    owners,
    input_ids,
    attention_mask,
    labels,
    special_ids,
    class_counts,
    sentence_token_counts=None,
):
    """A block's contribution to a query-normalized, class-balanced sentence loss."""
    valid = attention_mask.bool() & (owners >= 0)
    for special in special_ids:
        valid &= input_ids != special
    terms = []
    for sentence, label in labels.items():
        mask = valid & (owners == sentence)
        if mask.any():
            targets = torch.full(
                (int(mask.sum()),), label, dtype=torch.long, device=logits.device
            )
            terms.append(
                F.cross_entropy(logits[mask].float(), targets)
                * (0.5 / class_counts[label])
                * (
                    int(mask.sum()) / sentence_token_counts[sentence]
                    if sentence_token_counts is not None
                    else 1.0
                )
            )
    return torch.stack(terms).sum() if terms else None


def supervision(prepared, record, arm, qrels, special_ids):
    ordered, blocks = prepared
    positives = record["positive_ids"]
    negatives = [r["doc_id"] for r in record[arm]]
    assert set(positives).isdisjoint(negatives)
    assert len(negatives) == len(set(negatives)) == record["negative_count"]
    assert all(qrels.get(d, 0) == 2 for d in positives)
    assert all(qrels.get(d, 0) == 0 for d in negatives)
    labels = {ordered.index(d): 1 for d in positives}
    labels.update({ordered.index(d): 0 for d in negatives})
    seen = Counter()
    jobs = []
    for block in blocks:
        owners = {
            s
            for s, t in zip(block.sentence_indices, block.input_ids)
            if s >= 0 and t not in special_ids
        }
        for s in owners:
            seen[s] += 1
        if owners & labels.keys():
            jobs.append(block)
    # Token spans never overlap; oversized sentences may occupy multiple windows.
    if set(seen) != set(range(len(ordered))):
        raise ValueError("Missing/repeated sentence in zero-shot window mapping")
    return labels, {0: len(negatives), 1: len(positives)}, jobs


def backward_query(scorer, row, corpus, record, arm, qrels, divisor, precision):
    prepared = scorer.prepare(row, corpus)
    labels, counts, jobs = supervision(
        prepared, record, arm, qrels, scorer.tokenizer.all_special_ids
    )
    special_ids = set(scorer.tokenizer.all_special_ids)
    token_counts = Counter(
        s
        for block in jobs
        for s, t in zip(block.sentence_indices, block.input_ids)
        if s in labels and t not in special_ids
    )
    value = 0.0
    for block in jobs:
        ids = torch.tensor([block.input_ids], device=scorer.device)
        owners = torch.tensor([block.sentence_indices], device=scorer.device)
        attention = torch.ones_like(ids)
        with torch.autocast(
            device_type=scorer.device, dtype=torch.bfloat16, enabled=precision == "bf16"
        ):
            logits = scorer.model(
                input_ids=ids, attention_mask=attention
            ).compression_logits
            loss = block_loss(
                logits,
                owners,
                ids,
                attention,
                labels,
                scorer.tokenizer.all_special_ids,
                counts,
                token_counts,
            )
        if loss is None or not torch.isfinite(loss):
            raise ValueError("Invalid training loss")
        value += loss.detach().item()
        (loss / divisor).backward()
    return value, len(jobs)


def evaluate(scorer, data, rows, modules, output, manifest, source, name):
    data_mod, _, evaluation, config_mod = modules
    cfg = config_mod.ExperimentConfig(
        project_root=source.parent, phase5_dir=source.parent, results_dir=output
    )
    assert cfg.metric_ks == (1, 3, 5, 10) and cfg.binary_relevance_threshold == 2
    output.mkdir(parents=True, exist_ok=True)
    scorer.model.eval()
    per, by_type = [], defaultdict(list)
    with (output / "predictions.jsonl").open("w") as handle:
        for begin in range(0, len(rows), 8):
            batch = rows[begin : begin + 8]
            scores = scorer.score([scorer.prepare(row, data.corpus) for row in batch])
            for row, (mapping, counts) in zip(batch, scores):
                qid = row["query_id"]
                candidates = data_mod.candidate_doc_ids(data, qid, "article_local")
                if set(mapping) != set(candidates) or any(
                    n <= 0 for n in counts.values()
                ):
                    raise ValueError(f"Incomplete evaluation candidates: {qid}")
                values = np.asarray([mapping[d] for d in candidates], dtype=np.float32)
                ranked = [candidates[int(i)] for i in np.argsort(-values)[:10]]
                metrics = evaluation._per_query(ranked, data.qrels.get(qid, {}), cfg)
                per.append(metrics)
                by_type[row["query_type"]].append(metrics)
                handle.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "doc_ids": candidates,
                            "scores": values.tolist(),
                        }
                    )
                    + "\n"
                )
            if begin % 1024 == 0:
                print(f"eval {output.name} {begin}/{len(rows)}", flush=True)

    def aggregate(items):
        result = evaluation._aggregate(items)
        result["MRR@10"] = result.pop("mrr")
        return result

    result = {
        "experiment": name,
        "overall": aggregate(per),
        "by_query_type": {k: aggregate(v) for k, v in sorted(by_type.items())},
        "coverage": {
            "expected_queries": len(rows),
            "successful_queries": len(per),
            "failed_queries": 0,
        },
        "query_manifest_sha256": sha256(manifest),
        "qrels_sha256": sha256(source / "qrels.tsv"),
        "existing_evaluator_sha256": sha256(evaluation.__file__),
        "metric_definition": "original _per_query with np.argsort(float32) top10; graded gain 2**rel-1, Direct-only recall/MRR@10",
    }
    write_json(output / "formal_metrics.json", result)
    return result


def make_scorer(cfg, device):
    scorer = TrainingScorer(cfg["checkpoint"], device, cfg["eval_window_batch_size"])
    scorer.allow_oversized_sentence_fragments = cfg.get(
        "allow_oversized_sentence_fragments", False
    )
    scorer.model.deberta.requires_grad_(True)
    scorer.model.token_classifier.requires_grad_(True)
    if cfg["gradient_checkpointing"]:
        scorer.model.deberta.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return scorer


def gradient_norm(module):
    return math.sqrt(
        sum(
            float(p.grad.detach().float().square().sum())
            for p in module.parameters()
            if p.grad is not None
        )
    )


def train(cfg, arm, seed, device):
    contract = verify_contract(cfg)
    preflight_result = json.loads(
        (Path(cfg["prepared"]) / "preflight.json").read_text()
    )
    if (
        not preflight_result["passed"]
        or preflight_result["sampling_sha256"] != contract["sampling_sha256"]
    ):
        raise ValueError("Full sentence coverage preflight required")
    if preflight_result["oversized_sentence_fragments"] != cfg.get(
        "allow_oversized_sentence_fragments", False
    ):
        raise ValueError("Window configuration differs from full preflight")
    smoke_path = (
        Path(cfg["prepared"]) / f"smoke_{device}_{cfg.get('smoke_tag', 'generic')}.json"
    )
    if (
        not smoke_path.exists()
        or json.loads(smoke_path.read_text())["sampling_sha256"]
        != contract["sampling_sha256"]
    ):
        raise ValueError(
            "Run smoke on this device and sampling manifest before formal training"
        )
    if cfg["precision"] == "bf16" and (
        device != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Formal bf16 configuration requires a supported CUDA GPU")
    output = Path(cfg["output_root"]) / f"{arm}_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    modules = benchmark(cfg["benchmark_root"])
    data_mod = modules[0]
    source = Path(cfg["data_root"])
    prepared = Path(cfg["prepared"])
    train_data = data_mod.load_split(source / "train", "train")
    dev_data = data_mod.load_split(source / "dev", "dev")
    dev_rows = load_fixed_queries(
        prepared / "dev_relations_all.json", source / "dev", "dev"
    )
    records = [json.loads(line) for line in (prepared / "sampling.jsonl").open()]
    records = [r for r in records if "skip" not in r]
    version = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob("*provence*.py")}
    for p in Path(__file__).parent.glob("*provence*.py"):
        (output / "code").mkdir(exist_ok=True)
        shutil.copy2(p, output / "code" / p.name)
    for p in prepared.iterdir():
        if p.is_file():
            shutil.copy2(p, output / p.name)
    write_json(
        output / "run_config.json",
        {
            **cfg,
            "arm": arm,
            "experiment": NAMES[arm],
            "training_seed": seed,
            "initial_checkpoint": MODEL,
            "revision": REVISION,
            "initial_weights_sha256": sha256(
                Path(cfg["checkpoint"]) / "model.safetensors"
            ),
            "git_head": version,
            "code_sha256": code,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
        },
    )
    scorer = make_scorer(cfg, device)
    parameters = [p for p in scorer.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        eps=1e-8,
        betas=(0.9, 0.999),
    )
    steps = math.ceil(len(records) / cfg["effective_batch_size"]) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(steps * cfg["warmup_ratio"]), steps
    )
    best = -math.inf
    history = []
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        scorer.model.train()
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        loss_sum, window_sum = 0.0, 0
        start = time.monotonic()
        with (output / f"train_epoch{epoch}.jsonl").open("w") as log:
            for begin in range(0, len(order), cfg["effective_batch_size"]):
                group = order[begin : begin + cfg["effective_batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for idx in group:
                    record = records[idx]
                    qid = record["query_id"]
                    loss, windows = backward_query(
                        scorer,
                        train_data.instances[qid],
                        train_data.corpus,
                        record,
                        arm,
                        train_data.qrels[qid],
                        len(group),
                        cfg["precision"],
                    )
                    group_loss += loss
                    window_sum += windows
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, cfg["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                step += 1
                loss_sum += group_loss
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "queries": len(group),
                            "loss": group_loss / len(group),
                            "lr": scheduler.get_last_lr()[0],
                            "gradient_norm": float(norm),
                        }
                    )
                    + "\n"
                )
                if step % 25 == 0:
                    log.flush()
                    print(
                        f"{arm} epoch={epoch} queries={begin + len(group)}/{len(records)} loss={group_loss / len(group):.5f}",
                        flush=True,
                    )
        dev = evaluate(
            scorer,
            dev_data,
            dev_rows,
            modules,
            output / f"dev_epoch{epoch}",
            prepared / "dev_relations_all.json",
            source / "dev",
            NAMES[arm],
        )
        selected = dev["overall"]["ndcg@10"] > best
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(records),
                "supervised_windows": window_sum,
                "seconds": time.monotonic() - start,
                "dev": dev["overall"],
                "selected_best": selected,
            }
        )
        write_json(output / "history.json", history)
        if selected:
            best = dev["overall"]["ndcg@10"]
            scorer.model.save_pretrained(output / "best_checkpoint")
            scorer.tokenizer.save_pretrained(output / "best_checkpoint")
            write_json(
                output / "best.json",
                {
                    "epoch": epoch,
                    "dev_ndcg@10": best,
                    "selection": "full fixed dev only",
                },
            )
    del optimizer, scheduler, parameters, train_data, dev_data
    from safetensors.torch import load_file

    state = load_file(str(output / "best_checkpoint" / "model.safetensors"))
    scorer.model.load_state_dict(state, strict=True)
    del state
    test_data = data_mod.load_split(source / "test", "test")
    test_rows = load_fixed_queries(
        prepared / "test_relations_all.json", source / "test", "test"
    )
    result = evaluate(
        scorer,
        test_data,
        test_rows,
        modules,
        output / "test",
        prepared / "test_relations_all.json",
        source / "test",
        NAMES[arm],
    )
    reference = contract["zero_shot_reference"]
    assert all(
        result[k] == reference[k]
        for k in ("query_manifest_sha256", "qrels_sha256", "existing_evaluator_sha256")
    )
    write_json(
        output / "checkpoint_identity.json",
        {
            "best_weights_sha256": sha256(
                output / "best_checkpoint" / "model.safetensors"
            ),
            "parent_revision": REVISION,
        },
    )
    (output / "COMPLETED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
