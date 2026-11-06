from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


def inbatch_contrastive_loss(
    q: torch.Tensor,
    pos: torch.Tensor,
    temperature: float,
    relation_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """InfoNCE with masking for same-relation query variants.

    Phase 5 can generate several query types for the same underlying relation.
    Without this mask, the positive passage of another query variant from the
    same relation would be treated as an in-batch negative (a false negative).
    """
    logits = (q @ pos.T) / temperature
    if relation_ids is not None:
        ids = [str(x) for x in relation_ids]
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j and ids[i] == ids[j]:
                    logits[i, j] = -1e4
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def distributed_inbatch_contrastive_loss(
    q: torch.Tensor,
    pos: torch.Tensor,
    temperature: float,
    relation_ids: Sequence[str],
) -> torch.Tensor:
    """DDP InfoNCE with the same global in-batch budget as single-GPU FT.

    Each rank owns the queries in its local slice, while positives from all
    ranks participate as in-batch negatives.  The differentiable gather keeps
    gradients for remote positive embeddings. DistributedSampler guarantees
    equal local batch sizes, including its padded final step.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return inbatch_contrastive_loss(q, pos, temperature, relation_ids)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pos_global = torch.cat(list(differentiable_all_gather(pos)), dim=0)
    relation_parts: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(relation_parts, [str(x) for x in relation_ids])
    global_relation_ids = [rid for part in relation_parts for rid in (part or [])]

    logits = (q @ pos_global.T) / temperature
    local_ids = [str(x) for x in relation_ids]
    local_n = q.size(0)
    labels = rank * local_n + torch.arange(local_n, device=q.device)
    for i, rid in enumerate(local_ids):
        for j, other in enumerate(global_relation_ids):
            if j != int(labels[i]) and rid == other:
                logits[i, j] = -1e4
    return F.cross_entropy(logits, labels)


def pairwise_negative_loss(
    q: torch.Tensor,
    pos: torch.Tensor,
    neg: Optional[torch.Tensor],
    owners: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    if neg is None or neg.numel() == 0 or not owners:
        return q.new_tensor(0.0)
    owner_t = torch.tensor(list(owners), dtype=torch.long, device=q.device)
    q_sel = q[owner_t]
    p_sel = pos[owner_t]
    pos_s = torch.sum(q_sel * p_sel, dim=-1) / temperature
    neg_s = torch.sum(q_sel * neg, dim=-1) / temperature
    return F.softplus(neg_s - pos_s).mean()
