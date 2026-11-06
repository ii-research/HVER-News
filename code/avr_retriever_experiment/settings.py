from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from avr_retriever_experiment.training_data import TrainingRecipe


@dataclass(frozen=True)
class ExperimentSetting:
    name: str
    kind: str
    description: str
    model_name: Optional[str] = None
    recipe: Optional[TrainingRecipe] = None


def all_settings(
    e5: str, bge: str, train_backbone: Optional[str] = None
) -> Dict[str, ExperimentSetting]:
    train_model = train_backbone or e5
    settings = {
        "bm25": ExperimentSetting(
            "bm25", "bm25", "Sparse lexical sentence retrieval with article-local BM25."
        ),
        "e5_zero_shot": ExperimentSetting(
            "e5_zero_shot",
            "zero_shot",
            "E5 zero-shot dense sentence retrieval.",
            model_name=e5,
        ),
        "bge_m3_zero_shot": ExperimentSetting(
            "bge_m3_zero_shot",
            "zero_shot",
            "BGE-M3 zero-shot dense sentence retrieval (dense vector only).",
            model_name=bge,
        ),
        "qwen3_zero_shot": ExperimentSetting(
            "qwen3_zero_shot",
            "zero_shot",
            "Qwen3-Embedding-0.6B zero-shot dense sentence retrieval.",
            model_name="Qwen/Qwen3-Embedding-0.6B",
        ),
        "generic_ft": ExperimentSetting(
            "generic_ft",
            "train",
            "Generic dense FT: in-batch + generic same-article negatives.",
            model_name=train_model,
            recipe=TrainingRecipe(),
        ),
    }
    return settings
