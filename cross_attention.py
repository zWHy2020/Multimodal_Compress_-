"""Lightweight cross-attention used by video semantic guidance only."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Scaled dot-product cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        text_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.text_proj = nn.Linear(text_embed_dim, embed_dim, bias=bias) if text_embed_dim not in (None, embed_dim) else None
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        guide_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        if self.text_proj is not None:
            guide_vector = self.text_proj(guide_vector)

        guide_len = guide_vector.shape[1]
        q = self.q_proj(query)
        kv = self.kv_proj(guide_vector)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, guide_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, guide_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q.float() @ k.float().transpose(-2, -1) * self.scale).to(q.dtype)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, torch.finfo(attn_scores.dtype).min)

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        if return_attention:
            return output, attn_weights
        return output
