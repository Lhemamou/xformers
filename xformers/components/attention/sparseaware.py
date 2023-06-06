# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional, Union
from xformers import _has_cpp_library, _is_triton_available

if _has_cpp_library:
    from ._sputnik_sparse import SparseCS
import torch
import torch.nn as nn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    register_attention,
)
from xformers.components.attention.core import scaled_dot_product_attention


@dataclass
class sparseAwareAttentionConfig(AttentionConfig):
    causal: Optional[bool]
    force_sparsity: Optional[bool]


@register_attention("sparsemask", sparseAwareAttentionConfig)
class sparseAwareAttention(Attention):
    def __init__(
        self,
        dropout: float,
        *_,
        **__,
    ):
        r"""
        Sparse Aware Attention. This attention takes as input a sparse mask and apply scaled dot product attention.
        This implementation is sparse-aware, meaning that the empty attention parts will not be represented in memory.
        For the moment, this attention has only be tested with a batch size equal to 1.

        Args:
            dropout (float): probability of an element to be zeroed

        """
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=False)

        # Properties specific to this attention mechanism
        self.requires_same_k_q_dimensions = True
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[torch.Tensor, AttentionMask,"SparseCS"]] = None,
        *_,
        **__,
    ):
        """_summary_

        Args:
            q (torch.Tensor): query tensor
            k (torch.Tensor): key tensor
            v (torch.Tensor): value tensor
            att_mask (Optional[Union[torch.Tensor, AttentionMask,&quot;SparseCS&quot;]], optional): Attention Mask to apply. Recommanded to be a SparceCS attention matrix.


        Raises:
            NotImplemented: This attention is not attended to work without Attention Mask, otherwise, just use classic full attention

        Returns:
            _type_: scaled dot product attention using the attention mask.
        """

        # Mask-aware attention
        if att_mask is not None:
            mask = att_mask
        else :
            raise NotImplemented

        # Normal attention with the global tokens mask
        return scaled_dot_product_attention(
            q=q, k=k, v=v, att_mask=mask, dropout=self.attn_drop
        )
