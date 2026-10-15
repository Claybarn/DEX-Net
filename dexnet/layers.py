"""Transformer building blocks shared by the event models."""
import torch
from torch import nn


def causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    return mask.masked_fill(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))


def history_query_mask(seq_len: int, n_queries: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """History is causal; each query attends to all history and itself only."""
    n = seq_len + n_queries
    mask = torch.full((n, n), float("-inf"), device=device, dtype=dtype)
    mask[:seq_len, :seq_len] = causal_mask(seq_len, device, dtype)
    mask[seq_len:, :seq_len] = 0
    eye = torch.eye(n_queries, device=device, dtype=torch.bool)
    mask[seq_len:, seq_len:].masked_fill_(eye, 0)
    return mask


def _head(d_model: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_out))
