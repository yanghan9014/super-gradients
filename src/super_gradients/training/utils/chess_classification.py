"""Factorized chess classification, with the existing 13-class inference contract."""

import torch
from torch import Tensor


def chess_joint_scores(type_logits: Tensor, color_logits: Tensor, objectness_logits: Tensor = None) -> Tensor:
    """Decode [..., 7] type/board and [..., 2] white/black logits.

    Without objectness logits, six sigmoid type scores retain background/quality
    confidence. With objectness logits, each piece score is sigmoid(objectness)
    times a six-way type softmax times a two-way color softmax. The twelve piece
    scores then sum to objectness. The board always uses its independent sigmoid.
    The output order remains B,K,N,P,Q,R,b,k,n,p,q,r,board. Keep this composition
    identical in inference and task-aligned assignment.
    """
    if objectness_logits is None:
        type_scores = type_logits.sigmoid()
    else:
        type_scores = torch.cat((objectness_logits.sigmoid() * type_logits[..., :6].softmax(-1),
                                 type_logits[..., 6:7].sigmoid()), dim=-1)
    color_scores = color_logits.softmax(dim=-1)
    return torch.cat(
        (type_scores[..., :6] * color_scores[..., :1],
         type_scores[..., :6] * color_scores[..., 1:2],
         type_scores[..., 6:7]),
        dim=-1,
    )
