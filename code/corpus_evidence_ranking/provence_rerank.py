from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from .data import CorpusData, read_jsonl, sha256, write_jsonl
from .evaluate import metrics_from_predictions


def load_candidate_rows(path: Path) -> dict[str, dict]:
    rows = {row["query_id"]: row for row in read_jsonl(path)}
    if not rows:
        raise ValueError(f"Empty first-stage predictions: {path}")
    return rows


def pseudo_row(data: CorpusData, qid: str, article_id: str) -> dict:
    sentence_ids = data.article_sentence_ids[article_id]
    if not sentence_ids:
        raise ValueError(f"Article has no sentences: {article_id}")
    return {
        "query_id": qid,
        "query": data.queries[qid]["query"],
        "article_local_candidate_doc_ids": sentence_ids,
    }


def load_modules(bundle: Path):
    scripts = bundle / "scripts"
    sys.path.insert(0, str(scripts))
    from provence_finetune import TrainingScorer
    from provence_final_suite.model import SentenceEvidenceModel

    return (TrainingScorer, SentenceEvidenceModel)


def score_query_official(
    scorer, prepared: list[tuple[list[str], list]]
) -> list[list[float]]:
    results = scorer.score(prepared)
    return [
        [float(scores[sid]) for sid in ordered]
        for ((ordered, _), (scores, _)) in zip(prepared, results, strict=True)
    ]


def _sinusoidal(length: int, width: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device).float()[:, None]
    scale = torch.exp(
        torch.arange(0, width, 2, device=device).float() * (-np.log(10000.0) / width)
    )
    output = torch.zeros(length, width, device=device)
    output[:, 0::2] = torch.sin(positions * scale)
    output[:, 1::2] = torch.cos(positions * scale)
    return output


@torch.inference_mode()
def score_query_finetuned(
    model, prepared: list[tuple[list[str], list]], batch_size: int = 8, pad_id: int = 0
) -> list[list[float]]:
    """Score independent candidate documents while batching DeBERTa blocks.

    Sentence interaction and position encoding remain document-local. This is
    algebraically identical to calling ``model.forward`` once per document,
    but avoids thousands of batch-size-one encoder calls during corpus reranking.
    """
    device = next(model.parameters()).device
    hidden_width = model.official.config.hidden_size
    sums = [
        torch.zeros((len(ordered), hidden_width), device=device)
        for (ordered, _) in prepared
    ]
    counts = [torch.zeros(len(ordered), device=device) for (ordered, _) in prepared]
    jobs = [
        (doc_index, block.input_ids, block.sentence_indices)
        for (doc_index, (_, blocks)) in enumerate(prepared)
        for block in blocks
    ]
    for begin in range(0, len(jobs), batch_size):
        batch = jobs[begin : begin + batch_size]
        width = max((len(ids) for (_, ids, _) in batch))
        input_ids = torch.full(
            (len(batch), width), pad_id, dtype=torch.long, device=device
        )
        owners = torch.full_like(input_ids, -1)
        mask = torch.zeros_like(input_ids)
        for row, (_, ids, sentence_indices) in enumerate(batch):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, device=device)
            owners[row, :length] = torch.tensor(sentence_indices, device=device)
            mask[row, :length] = 1
        hidden = model.official.deberta(input_ids=input_ids, attention_mask=mask)[
            0
        ].float()
        for row, (doc_index, _, _) in enumerate(batch):
            valid = (owners[row] >= 0) & mask[row].bool()
            for special in model.special_ids:
                valid &= input_ids[row] != special
            chosen_owners = owners[row, valid]
            sums[doc_index].index_add_(0, chosen_owners, hidden[row, valid])
            counts[doc_index].index_add_(
                0, chosen_owners, torch.ones(len(chosen_owners), device=device)
            )
    values = []
    weights = torch.tensor([0.0, 1.0 / 3.0, 1.0], device=device)
    for total, count in zip(sums, counts, strict=True):
        if not torch.all(count > 0):
            raise ValueError("Missing sentence body tokens")
        features = model.projection(total / count[:, None])
        if model.use_sentence_position_encoding:
            features = features + _sinusoidal(len(count), features.shape[-1], device)
        if model.use_sentence_interaction:
            features = model.sentence_transformer(features[None])[0]
        logits = model.evidence_head(features)
        scores = (F.softmax(logits.float(), dim=-1) * weights).sum(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite Provence score")
        values.append(scores.cpu().tolist())
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank a fixed whole-corpus document pool with Provence"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Pinned official Provence checkpoint"
    )
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Official zero-shot block batch size"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variant", choices=("official", "adapted_provence"), default="official"
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    if args.variant == "official" and args.weights:
        parser.error("--weights is only valid for a fine-tuned variant")
    if args.variant != "official" and (not args.weights):
        parser.error("--weights is required for a fine-tuned variant")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output: {args.results_dir}"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (TrainingScorer, SentenceEvidenceModel) = load_modules(args.bundle)
    data = CorpusData.load(args.data_dir, args.split)
    candidates = load_candidate_rows(args.candidates)
    qids = sorted(data.queries)
    missing = sorted(set(qids) - set(candidates))
    if missing:
        raise ValueError(
            f"First-stage predictions miss {len(missing)} queries, e.g. {missing[:3]}"
        )
    if args.max_queries is not None:
        qids = qids[: args.max_queries]
    scorer = TrainingScorer(
        args.checkpoint,
        args.device if args.variant == "official" else "cpu",
        args.batch_size,
    )
    scorer.allow_oversized_sentence_fragments = True
    model = None
    if args.variant != "official":
        model = SentenceEvidenceModel(
            scorer.model, args.variant, scorer.tokenizer.all_special_ids
        ).to(args.device)
        model.load_state_dict(load_file(str(args.weights)))
        model.eval()
    article_ids = sorted(data.articles)
    sentence_ids = sorted(data.sentences)
    started = time.time()
    outputs = []
    for offset, qid in enumerate(qids, 1):
        stage = candidates[qid]
        raw_docs = stage["ranked_document_ids"]
        if len(raw_docs) != len(set(raw_docs)) or any(
            (d not in data.articles for d in raw_docs)
        ):
            raise ValueError(f"Invalid first-stage document list: {qid}")
        selected = raw_docs[: args.candidate_depth]
        prepared = [
            scorer.prepare(pseudo_row(data, qid, aid), data.sentences)
            for aid in selected
        ]
        score_lists = (
            score_query_official(scorer, prepared)
            if model is None
            else score_query_finetuned(
                model, prepared, args.batch_size, scorer.tokenizer.pad_token_id
            )
        )
        scored_sentences = []
        document_scores = []
        for aid, (ordered, _), values in zip(
            selected, prepared, score_lists, strict=True
        ):
            if len(ordered) != len(values):
                raise ValueError(f"Sentence score length mismatch: {qid}/{aid}")
            document_scores.append(max(values))
            scored_sentences.extend(zip(ordered, values))
        doc_order = np.lexsort(
            (np.asarray(selected), -np.asarray(document_scores, dtype=np.float32))
        )
        reranked = [selected[int(i)] for i in doc_order]
        reranked_scores = [float(document_scores[int(i)]) for i in doc_order]
        remainder = [d for d in raw_docs if d not in set(selected)]
        if len(raw_docs) < len(article_ids):
            seen = set(raw_docs)
            remainder.extend((d for d in article_ids if d not in seen))
        ranked_docs = reranked + remainder
        scored_sentences.sort(key=lambda item: (-item[1], item[0]))
        ranked_sentences = [sid for (sid, _) in scored_sentences]
        seen_sentences = set(ranked_sentences)
        ranked_sentences.extend(
            (sid for sid in sentence_ids if sid not in seen_sentences)
        )
        outputs.append(
            {
                "query_id": qid,
                "query": data.queries[qid]["query"],
                "query_type": data.queries[qid].get("query_type"),
                "relation_id": data.queries[qid].get("relation_id"),
                "source_article_id": data.source_article(qid),
                "ranked_document_ids": ranked_docs,
                "document_scores": reranked_scores,
                "document_ranking_is_complete": True,
                "reranked_document_depth": len(selected),
                "ranked_sentence_ids": ranked_sentences[:100],
                "sentence_scores": [float(v) for (_, v) in scored_sentences[:100]],
                "sentence_ranking_depth": min(100, len(scored_sentences)),
            }
        )
        if offset % 50 == 0:
            print(
                json.dumps(
                    {
                        "queries": offset,
                        "total": len(qids),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    write_jsonl(args.results_dir / "predictions.jsonl", outputs)
    metrics = metrics_from_predictions(data, outputs)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    config = {
        **{k: str(v) if isinstance(v, Path) else v for (k, v) in vars(args).items()},
        "query_count": len(qids),
        "elapsed_seconds": time.time() - started,
        "first_stage_sha256": sha256(args.candidates),
        "weights_sha256": sha256(args.weights) if args.weights else None,
        "dataset_manifest": data.manifest(),
        "document_score": "maximum reranked sentence evidence score",
        "candidate_policy": "fixed first-stage top depth; identical across Provence variants",
        "tail_policy": "documents retain first-stage order; sentences use canonical-id order",
        "unknown_policy": "unjudged pairs are never converted to negative; test contains none",
        "sentence_interaction": False,
        "sentence_position_encoding": False,
    }
    (args.results_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
