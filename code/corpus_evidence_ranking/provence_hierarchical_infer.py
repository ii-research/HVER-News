from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from .data import CorpusData, read_jsonl, sha256
from .provence_hierarchical_model import HierarchicalProvence
from .provence_hierarchical_train import VARIANTS, evaluate
from .provence_rerank import load_modules
from .provence_train import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained hierarchical Provence checkpoint"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Pinned official Provence initialization"
    )
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--block-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, num-shards)")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.trained_checkpoint.is_file():
        raise FileNotFoundError(args.trained_checkpoint)

    training_config_path = args.training_run_dir / "run_config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if training_config.get("variant") != args.variant:
        raise ValueError("Inference variant differs from the training run")
    if int(training_config.get("seed")) != args.seed:
        raise ValueError("Inference seed differs from the training run")

    data = CorpusData.load(args.data_dir, args.split)
    all_qids = sorted(data.queries)
    if args.max_queries is not None:
        all_qids = all_qids[: args.max_queries]
    qids = all_qids[args.shard_id :: args.num_shards]
    candidates = {row["query_id"]: row for row in read_jsonl(args.candidates)}
    missing = set(qids) - set(candidates)
    if missing:
        raise ValueError(f"Candidates miss {len(missing)} {args.split} queries")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal hierarchical inference requires a BF16 CUDA device")
    torch.set_num_threads(4)

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
    state = load_file(str(args.trained_checkpoint))
    model.load_state_dict(state, strict=True)

    save_json(
        args.output_dir / "run_config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "query_count": len(qids),
            "full_query_count_before_sharding": len(all_qids),
            "query_ids": (
                f"sorted queries with zero-based index congruent to {args.shard_id} "
                f"modulo {args.num_shards}"
            ),
            "sharding": {
                "strategy": "sorted query IDs, interleaved by zero-based index",
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
            },
            "data_manifest": data.manifest(),
            "candidates_sha256": sha256(args.candidates),
            "trained_checkpoint_sha256": sha256(args.trained_checkpoint),
            "training_run_config_sha256": sha256(training_config_path),
            "training_run_config": training_config,
            "checkpoint_selection": "best development sentence nDCG@10",
            "test_used_for_selection": False,
        },
    )
    metrics = evaluate(
        model,
        scorer,
        data,
        qids,
        candidates,
        args.candidate_depth,
        args.block_batch_size,
        args.output_dir,
    )
    save_json(
        args.output_dir / "completion.json",
        {
            "completed": True,
            "query_count": len(qids),
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
