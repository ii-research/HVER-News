"""Official frozen Provence pruning probabilities adapted to sentence ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from eval_retrieval_baseline import retrieval_metrics
from prepare_retrieval_baseline import load_corpus, read_jsonl, validate_instance

MODEL = "naver/provence-reranker-debertav3-v1"
REVISION = "ef49e233e3c6e50efc476c68f1390f8a63add4d4"
WEIGHTS_SHA256 = "227b473f30fa66d9163c678264082a023f39f632f493bed560964e016ab39bf2"
EXPERIMENT = "Provence (zero-shot, adapted for ranking)"


def normalize_query(text: str) -> str:
    """Match the checkpoint's official query preprocessing (not the saved query)."""
    return " ".join(
        "".join(c for c in text.lower() if c not in string.punctuation).split()
    )


@dataclass
class Block:
    input_ids: list[int]
    sentence_indices: list[int]


def pack_sentences(query_ids, sentences, *, cls_id, sep_id, max_length=512):
    """One unbroken token span per original sentence, no truncation or overlap."""
    prefix = [cls_id, *query_ids, sep_id]
    budget = max_length - len(prefix) - 1
    if not query_ids or budget <= 0:
        raise ValueError("Empty or oversized query")
    blocks = []
    ids = list(prefix)
    owners = [-1] * len(prefix)
    for index, tokens in enumerate(sentences):
        if not tokens or len(tokens) > budget:
            raise ValueError(
                f"Sentence {index} has {len(tokens)} tokens, budget {budget}; cannot preserve whole sentence"
            )
        if len(ids) + len(tokens) + 1 > max_length:
            blocks.append(Block(ids + [sep_id], owners + [-1]))
            ids, owners = list(prefix), [-1] * len(prefix)
        ids.extend(tokens)
        owners.extend([index] * len(tokens))
    if len(ids) > len(prefix):
        blocks.append(Block(ids + [sep_id], owners + [-1]))
    return blocks


def pool_probabilities(
    logits, sentence_indices, attention_mask, input_ids, special_ids, n_sentences
):
    """Softmax class 1, then mean over valid context tokens only."""
    if logits.shape != (*input_ids.shape, 2):
        raise ValueError(f"Unexpected pruning logits shape: {logits.shape}")
    probabilities = logits.float().softmax(dim=-1)[..., 1]
    valid = attention_mask.bool() & (sentence_indices >= 0)
    for token_id in special_ids:
        valid &= input_ids != token_id
    sums = torch.zeros(n_sentences, dtype=torch.float32, device=logits.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, sentence_indices[valid], probabilities[valid])
    counts.scatter_add_(
        0, sentence_indices[valid], torch.ones_like(probabilities[valid])
    )
    return sums.cpu(), counts.cpu()


class Scorer:
    def __init__(self, checkpoint: str, device: str, batch_size: int):
        weight_path = (
            Path(checkpoint) / "model.safetensors"
            if Path(checkpoint).is_dir()
            else Path(hf_hub_download(MODEL, "model.safetensors", revision=REVISION))
        )
        if sha256(weight_path) != WEIGHTS_SHA256:
            raise ValueError(
                "Weights do not match the pinned official Provence checkpoint"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=REVISION)
        self.model = AutoModel.from_pretrained(
            checkpoint, revision=REVISION, trust_remote_code=True
        )
        self.model.requires_grad_(False)
        self.model.eval().to(device)
        self.device = device
        self.batch_size = batch_size
        self.cache: dict[str, list[int]] = {}
        if self.model.config.max_position_embeddings != 512:
            raise ValueError("Expected official 512-token checkpoint")

    def prepare(self, row, corpus):
        candidates = row["article_local_candidate_doc_ids"]
        ordered = sorted(candidates, key=lambda doc: corpus[doc]["sentence_id"])
        sentence_tokens = []
        for i, doc in enumerate(ordered):
            key = ("first:" if i == 0 else "next:") + doc
            if key not in self.cache:
                # Follow official sentence-wise encoding; never re-split dataset sentences.
                self.cache[key] = self.tokenizer.encode(
                    (" " if i else "") + corpus[doc]["text"], add_special_tokens=False
                )
            sentence_tokens.append(self.cache[key])
        query_ids = self.tokenizer.encode(
            normalize_query(row["query"]), add_special_tokens=False
        )
        blocks = pack_sentences(
            query_ids,
            sentence_tokens,
            cls_id=self.tokenizer.cls_token_id,
            sep_id=self.tokenizer.sep_token_id,
        )
        return ordered, blocks

    @torch.inference_mode()
    def score(self, prepared):
        """Batch blocks across queries while preserving their original sentence IDs."""
        jobs = []
        offsets = []
        total = 0
        for ordered, blocks in prepared:
            offsets.append(total)
            for block in blocks:
                owners = [x + total if x >= 0 else -1 for x in block.sentence_indices]
                jobs.append((block.input_ids, owners))
            total += len(ordered)
        sums = torch.zeros(total)
        counts = torch.zeros(total)
        for begin in range(0, len(jobs), self.batch_size):
            batch = jobs[begin : begin + self.batch_size]
            length = max(len(ids) for ids, _ in batch)
            ids = torch.full(
                (len(batch), length), self.tokenizer.pad_token_id, dtype=torch.long
            )
            owners = torch.full_like(ids, -1)
            mask = torch.zeros_like(ids)
            for i, (tokens, indices) in enumerate(batch):
                ids[i, : len(tokens)] = torch.tensor(tokens)
                owners[i, : len(tokens)] = torch.tensor(indices)
                mask[i, : len(tokens)] = 1
            ids, owners, mask = (t.to(self.device) for t in (ids, owners, mask))
            output = self.model(input_ids=ids, attention_mask=mask)
            batch_sums, batch_counts = pool_probabilities(
                output.compression_logits,
                owners,
                mask,
                ids,
                self.tokenizer.all_special_ids,
                total,
            )
            sums += batch_sums
            counts += batch_counts
        if (counts <= 0).any() or not torch.isfinite(sums).all():
            raise ValueError(
                "Missing valid tokens or nonfinite pruning score; refusing partial coverage"
            )
        scores = (sums / counts).tolist()
        return [
            (
                dict(zip(ordered, scores[offset : offset + len(ordered)])),
                dict(
                    zip(ordered, counts[offset : offset + len(ordered)].int().tolist())
                ),
            )
            for (ordered, _), offset in zip(prepared, offsets)
        ]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fixed_queries(path: Path, source: Path, split: str):
    """Read the existing selection only; never call a sampler or regenerate a view."""
    payload = json.loads(path.read_text())
    relation_records = {}
    if isinstance(payload, dict):
        if payload.get("split", split) != split:
            raise ValueError("Fixed query manifest has the wrong split")
        if "seed" in payload and payload["seed"] != 42:
            raise ValueError("Expected the existing seed-42 query selection")
        if "query_ids" in payload:
            ids = payload["query_ids"]
        elif "records" in payload:
            records = payload["records"]
            # The existing formal select_experiment_query_ids returns sorted IDs.
            ids = sorted(record["query_id"] for record in records)
            relation_records = {r["query_id"]: r.get("relation_id") for r in records}
        else:
            raise ValueError("Expected query_ids or records in the fixed manifest")
    else:
        ids = payload
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(q, str) and q for q in ids)
    ):
        raise ValueError("Expected a nonempty list of query ID strings")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query ID in fixed manifest")
    if isinstance(payload, dict):
        for field in ("query_count", "relation_count"):
            if field in payload and payload[field] != len(ids):
                raise ValueError(f"Fixed manifest {field} does not match ID count")
    wanted = set(ids)
    rows = {}
    for row in read_jsonl(source / "instances.jsonl"):
        qid = row["query_id"]
        if qid in wanted:
            if qid in rows:
                raise ValueError(f"Duplicate source query: {qid}")
            rows[qid] = row
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"Fixed query IDs missing from source: {sorted(missing)[:10]}")
    relations = set()
    for qid in ids:
        row = rows[qid]
        relation = row.get("relation_id")
        if row["split"] != split or not relation:
            raise ValueError(f"Wrong split or missing relation: {qid}")
        if relation in relations:
            raise ValueError(f"Multiple fixed queries for relation: {relation}")
        relations.add(relation)
        if relation_records.get(qid) not in (None, relation):
            raise ValueError(f"Manifest/source relation mismatch: {qid}")
    return [rows[qid] for qid in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("dataset/retrieval_dataset")
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--checkpoint", default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query-id-manifest",
        type=Path,
        required=True,
        help="Existing frozen seed-42 query IDs; never regenerated",
    )
    parser.add_argument(
        "--limit", type=int, help="First N IDs of the fixed view, for smoke testing"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.query_batch_size, args.limit or 1) <= 0:
        parser.error("Batch sizes and limit must be positive")
    source = args.source / args.split
    fixed_rows = load_fixed_queries(args.query_id_manifest, source, args.split)
    selected_rows = fixed_rows[: args.limit] if args.limit is not None else fixed_rows
    manifest = {
        "experiment": EXPERIMENT,
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "query_selection": "reuse existing one-query-per-relation manifest; no sampling",
        "query_selection_seed": 42,
        "query_manifest_sha256": sha256(args.query_id_manifest),
        "fixed_query_count": len(fixed_rows),
        "evaluated_query_ids_sha256": hashlib.sha256(
            "\n".join(r["query_id"] for r in selected_rows).encode()
        ).hexdigest(),
        "smoke_selection": "fixed-list prefix" if args.limit is not None else None,
        "split": args.split,
        "limit": args.limit,
        "device": args.device,
        "dtype": "float32",
        "inference_script_sha256": sha256(Path(__file__)),
        "metrics_script_sha256": sha256(
            Path(__file__).with_name("eval_retrieval_baseline.py")
        ),
        "max_length": 512,
        "batch_size": args.batch_size,
        "query_batch_size": args.query_batch_size,
        "pooling": "mean softmax(compression_logits)[1] over valid sentence tokens",
        "query_preprocessing": "official lowercase, remove ASCII punctuation, collapse whitespace",
        "source_sha256": {
            f: sha256(source / f)
            for f in ["instances.jsonl", "corpus.jsonl", "queries.jsonl"]
        },
    }
    if args.output.exists():
        if not args.resume:
            parser.error("Output exists; choose a new directory or explicitly --resume")
        if json.loads((args.output / "manifest.json").read_text()) != manifest:
            parser.error("Resume manifest differs from current experiment")
    else:
        args.output.mkdir(parents=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (args.output / "fixed_query_manifest.json").write_bytes(
            args.query_id_manifest.read_bytes()
        )
        (args.output / "evaluated_query_ids.json").write_text(
            json.dumps([row["query_id"] for row in selected_rows], indent=2)
        )
    corpus = load_corpus(source / "corpus.jsonl")
    scorer = Scorer(args.checkpoint, args.device, args.batch_size)
    predictions = args.output / "predictions.jsonl"
    done = {}
    if predictions.exists():
        for record in read_jsonl(predictions):
            if record["query_id"] in done:
                raise ValueError("Duplicate predictions")
            done[record["query_id"]] = record
    start = time.perf_counter()
    totals: dict[str, float] = defaultdict(float)
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    type_counts: dict[str, int] = defaultdict(int)
    count = 0
    candidate_count = 0
    pending = []
    seen = set()

    def accumulate(row, record):
        nonlocal count, candidate_count
        if record["doc_ids"] != row["article_local_candidate_doc_ids"]:
            raise ValueError("Candidate IDs or order changed")
        scores = record["scores"]
        if (
            len(record["valid_token_counts"]) != len(scores)
            or min(record["valid_token_counts"]) < 1
        ):
            raise ValueError("Incomplete token coverage")
        metrics = retrieval_metrics(row, scores)
        for k, value in metrics.items():
            totals[k] += value
            by_type[row["query_type"]][k] += value
        type_counts[row["query_type"]] += 1
        count += 1
        candidate_count += len(scores)

    with predictions.open("a", encoding="utf-8") as out:

        def flush_pending():
            if not pending:
                return
            prepared = [scorer.prepare(row, corpus) for row in pending]
            outputs = scorer.score(prepared)
            for row, (scores, counts), (_, blocks) in zip(pending, outputs, prepared):
                candidates = row["article_local_candidate_doc_ids"]
                ranked = sorted(candidates, key=lambda doc: (-scores[doc], doc))
                record = {
                    "query_id": row["query_id"],
                    "doc_ids": candidates,
                    "scores": [scores[doc] for doc in candidates],
                    "sentence_ids": [corpus[doc]["sentence_id"] for doc in candidates],
                    "valid_token_counts": [counts[doc] for doc in candidates],
                    "ranked_doc_ids": ranked,
                    "blocks": len(blocks),
                }
                accumulate(row, record)
                out.write(json.dumps(record) + "\n")
            out.flush()
            pending.clear()
            print(
                json.dumps(
                    {
                        "queries": count,
                        "candidates_scored": candidate_count,
                        "elapsed_seconds": round(time.perf_counter() - start, 2),
                    }
                ),
                flush=True,
            )

        for row in selected_rows:
            validate_instance(row, corpus)
            if row["split"] != args.split or row["query_id"] in seen:
                raise ValueError("Wrong split or duplicate query")
            seen.add(row["query_id"])
            if row["query_id"] in done:
                accumulate(row, done[row["query_id"]])
            else:
                pending.append(row)
                if len(pending) >= args.query_batch_size:
                    flush_pending()
        flush_pending()
    if count != len(selected_rows) or set(done) - seen:
        raise ValueError("Empty evaluation or resume contains unexpected queries")
    summary = {
        **manifest,
        "queries": count,
        "candidates_scored": candidate_count,
        "all_candidates_scored": True,
        "frozen": not any(p.requires_grad for p in scorer.model.parameters()),
        "metrics": {k: v / count for k, v in totals.items()},
        "by_query_type": {
            t: {k: v / type_counts[t] for k, v in m.items()} for t, m in by_type.items()
        },
        "query_type_counts": dict(type_counts),
        "elapsed_seconds_this_invocation": time.perf_counter() - start,
        "grades": {"direct": 2, "partial": 1, "other": 0},
        "averaging": "macro per query; binary metrics use direct evidence only",
        "threshold_metrics_only": 0.1,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "run.trec").open("w") as handle:
        for record in read_jsonl(predictions):
            scores = dict(zip(record["doc_ids"], record["scores"]))
            for rank, doc in enumerate(record["ranked_doc_ids"], 1):
                handle.write(
                    f"{record['query_id']} Q0 {doc} {rank} {scores[doc]:.9g} provence_zero_shot\n"
                )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
