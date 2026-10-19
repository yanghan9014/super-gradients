"""Factorized chess classification, with the existing 13-class inference contract."""

import torch
from torch import Tensor


def chess_joint_scores(type_logits: Tensor, color_logits: Tensor) -> Tensor:
    """Decode [..., 7] type/board and [..., 2] white/black logits.

    Six sigmoid type scores retain the detector's background/quality confidence.
    Color is conditional on a piece: a two-way softmax splits each type's score.
    The output order remains B,K,N,P,Q,R,b,k,n,p,q,r,board. The board is independent
    of color. Keep this same composition in inference and task-aligned assignment.
    """
    type_scores = type_logits.sigmoid()
    color_scores = color_logits.softmax(dim=-1)
    return torch.cat(
        (type_scores[..., :6] * color_scores[..., :1],
         type_scores[..., :6] * color_scores[..., 1:2],
         type_scores[..., 6:7]),
        dim=-1,
    )
