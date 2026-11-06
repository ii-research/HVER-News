from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class ExperimentConfig:
    project_root: Path
    phase5_dir: Path
    results_dir: Path
    models_dir: Optional[Path] = None

    experiment_mode: str = "baselines"
    selected_settings: Tuple[str, ...] = ()
    retrieval_scope: str = "article_local"

    # Model/training randomness. Keep separate from the frozen query-view seed.
    seed: int = 42

    # ------------------------------------------------------------------
    # Frozen experimental query view
    # ------------------------------------------------------------------
    # Main-paper view: one available query variant per relation, selected once
    # and written to a manifest. All models reuse exactly the same query IDs.
    query_view_mode: str = "one_per_relation_balanced"
    query_view_seed: int = 42
    reuse_query_view_manifest: bool = True

    # None = use every relation in the official split.
    # Set a positive integer only for quick debugging/prototyping.
    max_train_relations: Optional[int] = None
    max_dev_relations: Optional[int] = None
    max_test_relations: Optional[int] = None

    # Four benchmark query formulations. The selector greedily balances them
    # among the variants actually available for each relation.
    query_types: Tuple[str, ...] = (
        "entity_value_explicit",
        "entity_implicit_value_explicit",
        "entity_explicit_value_naturalized",
        "entity_value_implicit",
    )

    # ------------------------------------------------------------------
    # Backbones
    # ------------------------------------------------------------------
    e5_model: str = "intfloat/e5-base-v2"
    bge_model: str = "BAAI/bge-m3"
    train_backbone: str = "intfloat/e5-base-v2"
    query_max_length: int = 128
    passage_max_length: int = 256
    device: str = "auto"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    temperature: float = 0.05
    hard_negative_weight: float = 1.0

    generic_negatives_per_query: int = 1
    num_workers: int = 0
    max_grad_norm: float = 1.0
    amp: str = "auto"  # auto | bf16 | fp16 | none

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # R@k / MRR use rel>=2 (Direct evidence only).
    # nDCG uses the full graded qrels: Direct=2, Partial/contextual=1.
    eval_batch_size: int = 32
    metric_ks: Tuple[int, ...] = (1, 3, 5, 10)
    binary_relevance_threshold: int = 2
    checkpoint_metric: str = "ndcg@10"
    dev_eval_max_queries: Optional[int] = None

    # BM25 article-local sentence ranking.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    @property
    def train_dir(self) -> Path:
        return self.project_root / self.phase5_dir / "train"

    @property
    def dev_dir(self) -> Path:
        return self.project_root / self.phase5_dir / "dev"

    @property
    def test_dir(self) -> Path:
        return self.project_root / self.phase5_dir / "test"

    @property
    def abs_results_dir(self) -> Path:
        return self.project_root / self.results_dir

    @property
    def abs_models_dir(self) -> Path:
        return self.project_root / (self.models_dir or (self.results_dir / "models"))

    @property
    def query_view_dir(self) -> Path:
        return self.abs_results_dir / "_query_views"

    def snapshot(self) -> Dict:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, Path):
                out[k] = str(v)
        return out
