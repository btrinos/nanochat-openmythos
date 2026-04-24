"""KV cache utilities used by the OpenMythos attention tests and decode path."""

import torch


class KVCache:
    """Layered key/value cache in Flash Attention 3 layout: (layer, batch, time, head, dim)."""

    def __init__(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        num_layers,
        device,
        dtype,
        value_head_dim=None,
        index_head_dim=0,
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = head_dim if value_head_dim is None else value_head_dim
        self.index_head_dim = index_head_dim
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, self.value_head_dim, device=device, dtype=dtype)
        self.index_cache = None
        if self.index_head_dim > 0:
            self.index_cache = torch.zeros(num_layers, batch_size, seq_len, self.index_head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding = None

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.value_head_dim == other.value_head_dim and self.index_head_dim == other.index_head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        if self.index_cache is not None:
            self.index_cache[:, :, :other_pos, :] = other.index_cache[:, :, :other_pos, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()
