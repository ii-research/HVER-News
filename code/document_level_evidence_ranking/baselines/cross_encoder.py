#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


class Reranker:
    def __init__(self, model_name: str, device: str, dtype: str, max_length: int):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_length = max_length
        print("Loading:", model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {}
        if self.device.type == "cuda":
            kwargs["torch_dtype"] = self.dtype
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs
        )
        self.model.to(self.device).eval()
        nlab = int(getattr(self.model.config, "num_labels", 1))
        if nlab not in (1, 2):
            raise ValueError(f"Expected 1- or 2-label reranker, got num_labels={nlab}")
        print(
            "Device:",
            self.device,
            "dtype:",
            next(self.model.parameters()).dtype,
            "labels:",
            nlab,
        )

    @torch.inference_mode()
    def score(self, pairs, batch_size: int) -> np.ndarray:
        outs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            qs = [x[0] for x in batch]
            ds = [x[1] for x in batch]
            toks = self.tok(
                qs,
                ds,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            logits = self.model(**toks).logits
            if logits.ndim == 1:
                s = logits
            elif logits.shape[-1] == 1:
                s = logits[:, 0]
            elif logits.shape[-1] == 2:
                s = logits[:, 1] - logits[:, 0]
            else:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            outs.append(s.float().cpu().numpy())
        return np.concatenate(outs) if outs else np.empty((0,), dtype=np.float32)


def dcg(rels: Sequence[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def per_query(ranked: List[str], qrels: Dict[str, int], ks=(1, 3, 5, 10)):
    direct = {d for d, r in qrels.items() if r >= 2}
    out = {}
    for k in ks:
        top = ranked[:k]
        hits = len(direct.intersection(top))
        rec = hits / len(direct) if direct else 0.0
        prec = hits / k
        out[f"recall@{k}"] = rec
        out[f"precision@{k}"] = prec
        out[f"f1@{k}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        rels = [qrels.get(d, 0) for d in top]
        ideal = sorted(qrels.values(), reverse=True)[:k]
        idcg = dcg(ideal)
        out[f"ndcg@{k}"] = dcg(rels) / idcg if idcg else 0.0

    # Standard full-ranking MRR.
    out["mrr_full"] = 0.0
    for rank, did in enumerate(ranked, 1):
        if did in direct:
            out["mrr_full"] = 1.0 / rank
            break

    # Compatibility with current document_level_evidence_ranking/evaluation.py:
    # it truncates ranking to max(k) before MRR, so existing paper "MRR" is MRR@10.
    max_k = max(ks)
    out["mrr"] = 0.0
    for rank, did in enumerate(ranked[:max_k], 1):
        if did in direct:
            out["mrr"] = 1.0 / rank
            break
    return out


def aggregate(rows):
    if not rows:
        return {"queries": 0}
    return {
        "queries": len(rows),
        **{k: mean(float(x[k]) for x in rows) for k in rows[0]},
    }


def main():
    script = Path(__file__).resolve()
    default_root = script.parents[1] if script.parent.name == "analysis" else Path.cwd()

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=default_root)
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument(
        "--reference-predictions",
        type=Path,
        default=None,
        help="Strongly recommended: predictions.jsonl from an existing formal 22,156-query run.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--query-block-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument(
        "--max-queries", type=int, default=None, help="Smoke test, e.g. 200"
    )
    args = ap.parse_args()

    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from document_level_evidence_ranking.data import load_split, candidate_doc_ids

    test_dir = (
        args.test_dir.expanduser().resolve()
        if args.test_dir
        else root.parent / "dataset" / "document-level" / "test"
    )
    data = load_split(test_dir, "test")

    if args.reference_predictions:
        ref = args.reference_predictions.expanduser().resolve()
        qids = [str(x["query_id"]) for x in read_jsonl(ref)]
        print("Using exact query IDs from:", ref)
    else:
        qids = list(data.query_ids)
        print(
            "WARNING: no --reference-predictions supplied; evaluating ALL accepted test queries."
        )
        print(
            "For the paper, use predictions.jsonl from an existing 22,156-query formal run."
        )

    seen = set()
    qids = [q for q in qids if not (q in seen or seen.add(q))]
    valid = set(data.query_ids)
    missing = [q for q in qids if q not in valid]
    if missing:
        raise ValueError(
            f"{len(missing)} query IDs not present in Phase-5 test. First: {missing[:5]}"
        )
    if args.max_queries:
        qids = qids[: args.max_queries]

    print("Queries:", len(qids))
    print(
        "Query types:",
        dict(
            Counter(str(data.instances[q].get("query_type", "unknown")) for q in qids)
        ),
    )

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (root / "results" / "cross_encoder" / args.model_name.replace("/", "__"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Reranker(args.model_name, args.device, args.dtype, args.max_length)
    ks = (1, 3, 5, 10)

    per, preds = [], []
    by_qt, by_ul = defaultdict(list), defaultdict(list)
    pair_count = 0
    t0 = time.time()

    pbar = tqdm(total=len(qids), desc="reranking", unit="query")
    for bs in range(0, len(qids), args.query_block_size):
        block = qids[bs : bs + args.query_block_size]
        pairs, meta = [], []

        for qid in block:
            q = data.queries[qid]["query"]
            cands = candidate_doc_ids(data, qid, "article_local")
            st = len(pairs)
            pairs.extend((q, str(data.corpus[d].get("text", ""))) for d in cands)
            meta.append((qid, cands, st, len(pairs)))

        scores = model.score(pairs, args.batch_size)
        pair_count += len(pairs)

        for qid, cands, st, ed in meta:
            s = scores[st:ed]
            order = np.argsort(-s)
            ranked = [cands[int(i)] for i in order]
            row = per_query(ranked, data.qrels.get(qid, {}), ks)
            per.append(row)
            inst = data.instances[qid]
            qt = str(inst.get("query_type", "unknown"))
            ul = str(inst.get("unit_level", "unknown"))
            by_qt[qt].append(row)
            by_ul[ul].append(row)
            preds.append(
                {
                    "query_id": qid,
                    "relation_id": str(inst.get("relation_id") or qid),
                    "query": data.queries[qid]["query"],
                    "query_type": qt,
                    "unit_level": ul,
                    "guid": inst.get("guid"),
                    "candidate_count": len(cands),
                    # Keep the complete article-local ranking.  Standard metrics still
                    # use top-k, while downstream relation-change diagnostics need
                    # scores for evidence sentences that may fall outside the top 10.
                    "ranked_doc_ids": [cands[int(i)] for i in order],
                    "scores": [float(s[int(i)]) for i in order],
                }
            )
        pbar.update(len(block))
    pbar.close()

    elapsed = time.time() - t0
    metrics = {
        "coverage": {
            "expected_queries": len(qids),
            "successful_queries": len(per),
            "failed_queries": 0,
        },
        "overall": aggregate(per),
        "by_query_type": {k: aggregate(v) for k, v in sorted(by_qt.items())},
        "by_unit_level": {k: aggregate(v) for k, v in sorted(by_ul.items())},
        "efficiency": {
            "pairs_scored": pair_count,
            "seconds": elapsed,
            "queries_per_second": len(qids) / elapsed if elapsed else None,
            "pairs_per_second": pair_count / elapsed if elapsed else None,
        },
    }

    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", preds)
    write_json(
        out_dir / "run_config.json",
        {
            "model_name": args.model_name,
            "evaluation_scope": "article_local",
            "task": "document_level_evidence_ranking",
            "query_count": len(qids),
            "reference_predictions": (
                str(args.reference_predictions) if args.reference_predictions else None
            ),
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "prediction_score_coverage": "all article-local candidate sentences",
            "note": "mrr matches current truncated-at-10 evaluation; mrr_full is standard full-ranking MRR.",
        },
    )

    print("\nDONE")
    for key in (
        "hit@1",
        "recall@1",
        "recall@5",
        "recall@10",
        "ndcg@5",
        "ndcg@10",
        "mrr",
        "mrr_full",
    ):
        print(f'{key:12s}: {metrics["overall"][key]:.6f}')
    print("Saved:", out_dir / "metrics.json")
    print("Saved:", out_dir / "predictions.jsonl")


if __name__ == "__main__":
    main()
