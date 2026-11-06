from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


RETRIEVER_META = "retriever_config.json"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_family(model_name: str) -> str:
    lower = str(model_name).lower()
    if "e5" in lower:
        return "e5"
    if "bge" in lower:
        return "bge"
    if "qwen3-embedding" in lower:
        return "qwen3_embedding"
    return "generic"


def family_defaults(family: str) -> tuple[str, str, str]:
    """Return query_prefix, passage_prefix, pooling.

    E5-base-v2 follows its standard `query:` / `passage:` format and mean
    pooling. BGE/BGE-M3 dense retrieval uses normalized CLS embeddings and no
    mandatory text prefix for this benchmark setting.
    """
    if family == "e5":
        return "query: ", "passage: ", "mean"
    if family == "bge":
        return "", "", "cls"
    if family == "qwen3_embedding":
        return (
            "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: ",
            "",
            "last_token",
        )
    return "", "", "mean"


class DualEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        query_max_length: int = 128,
        passage_max_length: int = 256,
        *,
        family: str | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        pooling: str | None = None,
    ):
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.family = family or infer_family(self.model_name_or_path)
        d_q, d_p, d_pool = family_defaults(self.family)
        self.query_prefix = d_q if query_prefix is None else query_prefix
        self.passage_prefix = d_p if passage_prefix is None else passage_prefix
        self.pooling = d_pool if pooling is None else pooling

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.encoder = AutoModel.from_pretrained(self.model_name_or_path)
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

    @staticmethod
    def mean_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def _pool(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "mean":
            return self.mean_pool(last_hidden, attention_mask)
        if self.pooling == "cls":
            return last_hidden[:, 0]
        if self.pooling == "last_token":
            # Qwen3-Embedding uses the final non-padding token. This supports
            # both left- and right-padded tokenizer batches.
            if bool(torch.all(attention_mask[:, -1] == 1)):
                return last_hidden[:, -1]
            lengths = attention_mask.sum(dim=1) - 1
            rows = torch.arange(last_hidden.size(0), device=last_hidden.device)
            return last_hidden[rows, lengths]
        raise ValueError(f"Unknown pooling={self.pooling}")

    def _encode(
        self,
        texts: Sequence[str],
        max_length: int,
        prefix: str,
        device: torch.device,
    ) -> torch.Tensor:
        batch = self.tokenizer(
            [prefix + str(x) for x in texts],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = self.encoder(**batch)
        emb = self._pool(out.last_hidden_state, batch["attention_mask"])
        return F.normalize(emb, p=2, dim=1)

    def encode_queries(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        return self._encode(
            texts,
            self.query_max_length,
            self.query_prefix,
            device,
        )

    def encode_passages(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        return self._encode(
            texts,
            self.passage_max_length,
            self.passage_prefix,
            device,
        )

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        encoder = (
            self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        )
        encoder.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        meta = {
            "source_model_name_or_path": self.model_name_or_path,
            "family": self.family,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "pooling": self.pooling,
            "query_max_length": self.query_max_length,
            "passage_max_length": self.passage_max_length,
        }
        (out_dir / RETRIEVER_META).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        query_max_length: int = 128,
        passage_max_length: int = 256,
    ) -> "DualEncoder":
        meta_path = Path(path) / RETRIEVER_META
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # The explicit metadata is critical for E5 fine-tuned checkpoints:
        # after saving to `best_model/`, the directory name no longer contains
        # "e5", so inferring prefixes from the checkpoint path would be wrong.
        return cls(
            str(path),
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            family=meta.get("family"),
            query_prefix=meta.get("query_prefix"),
            passage_prefix=meta.get("passage_prefix"),
            pooling=meta.get("pooling"),
        )
