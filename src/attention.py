"""Attention building blocks, written from scratch for readability.

Shared by the notebooks in this repo. Every function works on PyTorch
tensors whose last two dimensions are (sequence_length, feature_dim),
so the same code handles a single head, many heads, or a batch.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute softmax(Q K^T / sqrt(d_k)) V.

    Args:
        q: queries, shape (..., seq_q, d_k)
        k: keys,    shape (..., seq_k, d_k)
        v: values,  shape (..., seq_k, d_v)
        mask: optional boolean tensor broadcastable to (..., seq_q, seq_k).
              True means "this query may look at this key", False means
              "hide this key". This matches PyTorch's boolean attn_mask.

    Returns:
        output:  shape (..., seq_q, d_v), the weighted mix of values
        weights: shape (..., seq_q, seq_k), the attention weights
                 (every row sums to 1)
    """
    d_k = q.size(-1)

    # 1. How well does each query match each key? (dot products)
    scores = q @ k.transpose(-2, -1)

    # 2. Scale so the softmax doesn't become too peaked as d_k grows.
    scores = scores / math.sqrt(d_k)

    # 3. Hide the keys we are not allowed to see.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 4. Turn scores into percentages that sum to 1 along each row.
    weights = torch.softmax(scores, dim=-1)

    # 5. Mix the values using those percentages.
    output = weights @ v
    return output, weights


def causal_mask(seq_len, device=None):
    """Lower-triangular boolean mask: token i can only see tokens 0..i."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class MultiHeadAttention(nn.Module):
    """Minimal multi-head self-attention.

    The model dimension is split into n_heads smaller heads. Each head runs
    its own scaled dot-product attention, then the results are joined back
    together and passed through an output projection.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_head)
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        # (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

    def forward(self, x, mask=None):
        """x: (batch, seq, d_model). Returns (output, weights per head)."""
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.out_proj(self._merge_heads(out)), weights
