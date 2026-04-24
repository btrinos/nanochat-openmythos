"""
OpenMythos-style nanochat model.
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
- optional portable MLA + DSA attention path
- optional DeepSeek mHC-style manifold-constrained hyper-connections
- optional Apple-style multi-token prediction (MTP) auxiliary head
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn


def _disable_torch_compile(fn):
    # Keep mHC's Sinkhorn/einsum maps eager; on MPS compile can amplify their gradients to NaN.
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "disable"):
        return dynamo.disable(fn)
    return fn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Attention implementation. "gqa" is the original nanochat path.
    # "mla_dsa" enables a portable DeepSeek-V3.2-style MLA + DSA path.
    attn_type: str = "gqa"
    mla_q_lora_rank: int = 0  # 0 => direct query projection
    mla_kv_lora_rank: int = 0  # 0 => infer latent KV width from model dim
    mla_qk_nope_head_dim: int = 0  # 0 => infer from regular head dim
    mla_qk_rope_head_dim: int = 0  # 0 => infer even half-head RoPE width
    mla_v_head_dim: int = 0  # 0 => regular attention value head dim
    dsa_topk: int = 256  # <=0 disables sparse token selection
    dsa_index_n_heads: int = 0  # 0 => use n_head
    dsa_index_head_dim: int = 0  # 0 => infer from MLA NoPE width
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Optional OpenMythos-style recurrent-depth trunk. When enabled, n_layer is
    # interpreted as the default virtual depth: prelude + loop_iters + coda.
    use_looped_transformer: bool = False
    loop_prelude_layers: int = 2
    loop_coda_layers: int = 2
    loop_iters: int = 0  # 0 => infer from n_layer - prelude - coda
    loop_lora_rank: int = 0  # 0 disables depth-wise LoRA
    loop_act_threshold: float = 0.99  # <=0 disables ACT weighting
    loop_embed_fraction: float = 0.125
    loop_moe_num_experts: int = 0  # 0 disables recurrent-block MoE FFN
    loop_moe_shared_experts: int = 0
    loop_moe_experts_per_token: int = 2
    loop_moe_expert_dim: int = 0  # 0 => infer a fine-grained expert width
    loop_moe_bias_update_rate: float = 1e-3  # <=0 disables router-bias load balancing
    loop_moe_bias_update_clamp: float = 5.0
    loop_moe_router_z_loss_weight: float = 1e-4  # <=0 disables router logit z-loss
    # Optional DeepSeek mHC-style residual stream expansion. 1 keeps the
    # original nanochat residual path; >1 enables n parallel residual streams.
    mhc_num_streams: int = 1
    mhc_sinkhorn_iters: int = 20
    mhc_dynamic_init_scale: float = 0.01
    mhc_residual_init_strength: float = 2.0
    # Optional Apple-style MTP. The default is off; when enabled, scalar
    # training losses get extra future-token losses while NTP logits are unchanged.
    apple_mtp_num_future_tokens: int = 0
    apple_mtp_loss_weight: float = 0.2
    apple_mtp_consistency_weight: float = 0.0
    apple_mtp_gated_lora_rank: int = 0
    apple_mtp_sampler_hidden_mult: int = 1

    def loop_layout(self, loop_iters=None):
        """Return (prelude_layers, recurrent_iters, coda_layers) for this config."""
        if not self.use_looped_transformer:
            return self.n_layer, 0, 0
        prelude = min(max(self.loop_prelude_layers, 0), max(self.n_layer - 1, 0))
        coda_budget = max(self.n_layer - prelude - 1, 0)
        coda = min(max(self.loop_coda_layers, 0), coda_budget)
        if loop_iters is None:
            loop_iters = self.loop_iters
        if loop_iters is None or loop_iters <= 0:
            loop_iters = max(1, self.n_layer - prelude - coda)
        return prelude, loop_iters, coda

    def num_kv_layers(self, loop_iters=None):
        """Number of virtual attention layers needed by the KV cache."""
        prelude, recurrent_iters, coda = self.loop_layout(loop_iters)
        return prelude + recurrent_iters + coda

    def loop_moe_enabled(self):
        return self.use_looped_transformer and self.loop_moe_num_experts > 0

    def loop_moe_topk(self):
        if not self.loop_moe_enabled():
            return 0
        return min(max(self.loop_moe_experts_per_token, 1), self.loop_moe_num_experts)

    def loop_moe_width(self):
        if self.loop_moe_expert_dim > 0:
            return self.loop_moe_expert_dim
        topk = max(self.loop_moe_topk(), 1)
        return max(1, self.n_embd // max(self.loop_moe_num_experts // topk, 1))

    def loop_moe_router_z_loss_enabled(self):
        return self.loop_moe_enabled() and self.loop_moe_router_z_loss_weight > 0

    def attention_head_dim(self):
        return self.n_embd // self.n_head

    def use_mla_attention(self):
        return self.attn_type.lower() in {"mla", "mla_dsa"}

    def mla_kv_rank(self):
        return self.mla_kv_lora_rank if self.mla_kv_lora_rank > 0 else max(1, self.n_embd // 2)

    def mla_nope_dim(self):
        return self.mla_qk_nope_head_dim if self.mla_qk_nope_head_dim > 0 else self.attention_head_dim()

    def mla_rope_dim(self):
        if self.mla_qk_rope_head_dim > 0:
            return self.mla_qk_rope_head_dim
        rope_dim = max(2, self.attention_head_dim() // 2)
        return rope_dim - (rope_dim % 2)

    def mla_qk_head_dim(self):
        return self.mla_nope_dim() + self.mla_rope_dim()

    def mla_value_head_dim(self):
        return self.mla_v_head_dim if self.mla_v_head_dim > 0 else self.attention_head_dim()

    def dsa_index_dim(self):
        return self.dsa_index_head_dim if self.dsa_index_head_dim > 0 else self.mla_nope_dim()

    def dsa_index_heads(self):
        return self.dsa_index_n_heads if self.dsa_index_n_heads > 0 else self.n_head

    def kv_cache_dims(self):
        """Return cache payload widths. MLA stores compressed c_kv + rotated k_rope."""
        if self.use_mla_attention():
            index_dim = self.dsa_index_dim() if self.attn_type.lower() == "mla_dsa" and self.dsa_topk > 0 else 0
            return self.mla_kv_rank(), self.mla_rope_dim(), index_dim
        head_dim = self.attention_head_dim()
        return head_dim, head_dim, 0

    def kv_cache_heads(self):
        return 1 if self.use_mla_attention() else self.n_kv_head

    def rotary_dim(self):
        if self.use_mla_attention():
            return max(self.attention_head_dim(), self.mla_rope_dim())
        return self.attention_head_dim()

    def use_mhc(self):
        return self.mhc_num_streams > 1

    def apple_mtp_enabled(self):
        return self.apple_mtp_num_future_tokens > 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, use_value_embedding=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.attn_type = config.attn_type.lower()
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        assert self.attn_type in {"gqa", "mla", "mla_dsa"}, f"Unknown attn_type: {config.attn_type}"
        self.use_mla = config.use_mla_attention()
        if not self.use_mla:
            self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        else:
            self.q_lora_rank = config.mla_q_lora_rank
            self.kv_lora_rank = config.mla_kv_rank()
            self.qk_nope_head_dim = config.mla_nope_dim()
            self.qk_rope_head_dim = config.mla_rope_dim()
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.v_head_dim = config.mla_value_head_dim()
            self.dsa_topk = max(config.dsa_topk, 0) if self.attn_type == "mla_dsa" else 0
            assert self.qk_nope_head_dim > 0
            assert self.qk_rope_head_dim > 0 and self.qk_rope_head_dim % 2 == 0
            assert self.v_head_dim > 0
            if self.q_lora_rank > 0:
                self.q_a = Linear(self.n_embd, self.q_lora_rank, bias=False)
                self.q_b = Linear(self.q_lora_rank, self.n_head * self.qk_head_dim, bias=False)
            else:
                self.c_q = Linear(self.n_embd, self.n_head * self.qk_head_dim, bias=False)
            # DeepSeek-style MLA: latent KV content plus a decoupled RoPE key part.
            self.kv_down = Linear(self.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
            self.kv_up = Linear(self.kv_lora_rank, self.n_kv_head * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
            self.c_proj = Linear(self.n_head * self.v_head_dim, self.n_embd, bias=False)
            if self.attn_type == "mla_dsa" and self.dsa_topk > 0:
                self.index_n_heads = config.dsa_index_heads()
                self.index_head_dim = config.dsa_index_dim()
                assert self.index_n_heads > 0 and self.index_head_dim > 0
                self.index_q = Linear(self.n_embd, self.index_n_heads * self.index_head_dim, bias=False)
                self.index_k = Linear(self.n_embd, self.index_head_dim, bias=False)
                self.index_weights = Linear(self.n_embd, self.index_n_heads, bias=False)
                self.index_scale = self.index_head_dim ** -0.5
        self.ve_gate_channels = 12
        if use_value_embedding is None:
            use_value_embedding = has_ve(layer_idx, config.num_kv_layers())
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if use_value_embedding else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cache_layer_idx=None):
        if self.use_mla:
            return self._forward_mla_dsa(x, ve, cos_sin, window_size, kv_cache, cache_layer_idx)
        return self._forward_gqa(x, ve, cos_sin, window_size, kv_cache, cache_layer_idx)

    def _forward_gqa(self, x, ve, cos_sin, window_size, kv_cache, cache_layer_idx=None):
        B, T, C = x.size()
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            assert self.ve_gate is not None, "value embedding was passed to an attention layer without a gate"
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(cache_layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if cache_layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _forward_mla_dsa(self, x, ve, cos_sin, window_size, kv_cache, cache_layer_idx=None):
        B, T, C = x.size()
        cache_layer_idx = self.layer_idx if cache_layer_idx is None else cache_layer_idx

        if self.q_lora_rank > 0:
            q = self.q_b(norm(self.q_a(x)))
        else:
            q = self.c_q(x)
        q = q.view(B, T, self.n_head, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        c_kv = self.kv_down(x)
        c_kv, k_rope = torch.split(c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = norm(c_kv)

        cos, sin = cos_sin
        rotary_width = self.qk_rope_head_dim // 2
        rope_cos_sin = cos[..., :rotary_width], sin[..., :rotary_width]
        q_rope = apply_rotary_emb(q_rope, *rope_cos_sin)
        k_rope = apply_rotary_emb(k_rope.view(B, T, 1, self.qk_rope_head_dim), *rope_cos_sin)

        q = torch.cat([q_nope, q_rope], dim=-1)

        if ve is not None:
            raise AssertionError("MLA attention disables nanochat value embeddings so compressed KV decode is reconstructable")

        index_k_new = None
        index_scores = None
        if self.attn_type == "mla_dsa" and self.dsa_topk > 0:
            index_k_new = norm(self.index_k(x))
            if kv_cache is None:
                index_scores = self._dsa_index_scores(x, index_k_new)

        if kv_cache is None:
            k_nope, v = self._expand_mla_latent(c_kv)
            k_rope_heads = k_rope.expand(-1, -1, self.n_kv_head, -1)
            k = torch.cat([k_nope, k_rope_heads], dim=-1)
            q, k = self._norm_scale_mla_qk(q, k)
            y = self._manual_attention(q, k, v, window_size, 0, 0, index_scores=index_scores)
        else:
            pos = kv_cache.get_pos()
            end_pos = pos + T
            assert end_pos <= kv_cache.max_seq_len, f"KV cache too small: need {end_pos}, have {kv_cache.max_seq_len}"
            c_kv_cache, k_rope_cache = kv_cache.get_layer_cache(cache_layer_idx)
            assert c_kv_cache.size(2) == 1, f"MLA cache uses one latent pseudo-head, got {c_kv_cache.size(2)}"
            assert c_kv_cache.size(-1) == self.kv_lora_rank, f"MLA latent cache dim {c_kv_cache.size(-1)} != {self.kv_lora_rank}"
            assert k_rope_cache.size(-1) == self.qk_rope_head_dim, f"MLA RoPE cache dim {k_rope_cache.size(-1)} != {self.qk_rope_head_dim}"
            c_kv_cache[:B, pos:end_pos, 0, :].copy_(c_kv.to(dtype=c_kv_cache.dtype))
            k_rope_cache[:B, pos:end_pos, 0, :].copy_(k_rope.squeeze(2).to(dtype=k_rope_cache.dtype))
            if index_k_new is not None:
                assert kv_cache.index_cache is not None, "MLA+DSA KV cache requires index_head_dim > 0"
                assert kv_cache.index_cache.size(-1) == self.index_head_dim
                kv_cache.index_cache[cache_layer_idx, :B, pos:end_pos].copy_(index_k_new.to(dtype=kv_cache.index_cache.dtype))
                index_scores = self._dsa_index_scores(x, kv_cache.index_cache[cache_layer_idx, :B, :end_pos])
            c_kv_full = c_kv_cache[:B, :end_pos, 0, :]
            k_rope_full = k_rope_cache[:B, :end_pos, 0, :]
            if T == 1 and index_scores is not None and self.dsa_topk > 0:
                topk_idx = self._dsa_topk_indices(index_scores, T, end_pos, window_size, pos, 0)
                selected_c_kv = self._gather_sequence(c_kv_full, topk_idx)
                selected_k_rope = self._gather_sequence(k_rope_full, topk_idx)
                k_nope, v = self._expand_mla_latent(selected_c_kv)
                k_rope_heads = selected_k_rope.unsqueeze(3).expand(-1, -1, -1, self.n_kv_head, -1)
                k = torch.cat([k_nope, k_rope_heads], dim=-1)
                q, k = self._norm_scale_mla_qk(q, k)
                y = self._selected_attention(q, k, v)
            else:
                k_nope, v = self._expand_mla_latent(c_kv_full)
                k_rope_heads = k_rope_full.unsqueeze(2).expand(-1, -1, self.n_kv_head, -1)
                k = torch.cat([k_nope, k_rope_heads], dim=-1)
                q, k = self._norm_scale_mla_qk(q, k)
                y = self._manual_attention(
                    q, k, v,
                    window_size, pos, 0, index_scores=index_scores,
                )
            if cache_layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _expand_mla_latent(self, c_kv):
        *prefix, latent_dim = c_kv.shape
        assert latent_dim == self.kv_lora_rank
        kv = self.kv_up(c_kv.reshape(-1, latent_dim))
        kv = kv.view(*prefix, self.n_kv_head, self.qk_nope_head_dim + self.v_head_dim)
        return torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    def _norm_scale_mla_qk(self, q, k):
        q, k = norm(q), norm(k)
        # Keep nanochat's QK-norm sharpening to match the GQA path. This is a
        # deliberate nanochat deviation from canonical MLA.
        return q * 1.2, k * 1.2

    def _gather_sequence(self, x, idx):
        B, T, K = idx.shape
        D = x.size(-1)
        x = x.unsqueeze(1).expand(-1, T, -1, -1)
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        return x.gather(2, idx)

    def _repeat_kv(self, x):
        if self.n_kv_head == self.n_head:
            return x
        return x.repeat_interleave(self.n_head // self.n_kv_head, dim=2)

    def _attention_valid_mask(self, T, S, window_size, query_pos_start, key_pos_start, device):
        q_pos = torch.arange(query_pos_start, query_pos_start + T, device=device).view(T, 1)
        k_pos = torch.arange(key_pos_start, key_pos_start + S, device=device).view(1, S)
        valid = k_pos <= q_pos
        left = window_size[0]
        if left is not None and left >= 0:
            valid = valid & (k_pos >= q_pos - left)
        return valid.view(1, 1, T, S)

    def _manual_attention(self, q, k, v, window_size, query_pos_start, key_pos_start, index_scores=None):
        B, T, H, D = q.shape
        S = k.size(1)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * (D ** -0.5)
        valid = self._attention_valid_mask(T, S, window_size, query_pos_start, key_pos_start, q.device)
        min_score = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid, min_score)

        if self.dsa_topk > 0 and self.dsa_topk < S:
            selector = scores if index_scores is None else index_scores.float().unsqueeze(1).expand(B, H, T, S)
            selector = selector.masked_fill(~valid, min_score)
            topk_idx = selector.topk(self.dsa_topk, dim=-1).indices
            sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_idx, True)
            scores = scores.masked_fill(~sparse_mask, min_score)

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=v.dtype)
        return torch.einsum("bhts,bshd->bthd", attn, v)

    def _dsa_topk_indices(self, index_scores, T, S, window_size, query_pos_start, key_pos_start):
        valid = self._attention_valid_mask(T, S, window_size, query_pos_start, key_pos_start, index_scores.device)
        selector = index_scores.float().masked_fill(~valid.squeeze(1), torch.finfo(torch.float32).min)
        # This helper is used for one-token decode, where all batch elements share
        # the same valid window. Prefill and multi-token cache calls use the dense
        # masking helper above.
        valid_count = int(valid[0, 0, 0].sum().item())
        topk = min(self.dsa_topk, max(valid_count, 1))
        return selector.topk(topk, dim=-1).indices

    def _repeat_selected_kv(self, x):
        if self.n_kv_head == self.n_head:
            return x
        return x.repeat_interleave(self.n_head // self.n_kv_head, dim=3)

    def _selected_attention(self, q, k, v):
        B, T, H, D = q.shape
        k = self._repeat_selected_kv(k)
        v = self._repeat_selected_kv(v)
        scores = torch.einsum("bthd,btkhd->bhtk", q.float(), k.float()) * (D ** -0.5)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=v.dtype)
        return torch.einsum("bhtk,btkhd->bthd", attn, v)

    def _dsa_index_scores(self, query_x, index_k):
        B, T, _ = query_x.shape
        q = self.index_q(query_x).view(B, T, self.index_n_heads, self.index_head_dim)
        q = norm(q)
        head_scores = torch.einsum("btid,bsd->bits", q.float(), index_k.float()) * self.index_scale
        head_scores = F.relu(head_scores)
        weights = self.index_weights(query_x).float().transpose(1, 2).unsqueeze(-1)
        weights = weights * (self.index_n_heads ** -0.5)
        return (head_scores * weights).sum(dim=1)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class AppleMTPGatedLoRA(nn.Module):
    """Low-rank adapter used only on MTP paths, preserving ordinary NTP logits."""
    def __init__(self, dim, rank):
        super().__init__()
        self.down = Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x):
        return self.down(x) @ self.B.to(dtype=x.dtype)


class AppleMTPSamplerHead(nn.Module):
    """Two-block sampler head conditioned on hidden state and previous token."""
    def __init__(self, config):
        super().__init__()
        self.num_future_tokens = config.apple_mtp_num_future_tokens
        dim = config.n_embd
        hidden_dim = max(dim, dim * max(config.apple_mtp_sampler_hidden_mult, 1))
        self.mask_embeddings = nn.Embedding(self.num_future_tokens, dim)
        self.gated_lora = AppleMTPGatedLoRA(dim, config.apple_mtp_gated_lora_rank) if config.apple_mtp_gated_lora_rank > 0 else None
        self.fc1 = Linear(2 * dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = Linear(hidden_dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)

    def apply_mask_slot(self, anchor_hidden, slot_idx):
        slot_idx = min(slot_idx, self.num_future_tokens - 1)
        slot = torch.tensor(slot_idx, device=anchor_hidden.device)
        mask = self.mask_embeddings(slot).to(dtype=anchor_hidden.dtype)
        x = anchor_hidden + mask.view(1, 1, -1)
        if self.gated_lora is not None:
            x = x + self.gated_lora(x)
        return x

    def forward(self, anchor_hidden, prev_token_embed, lm_head, vocab_size):
        x = torch.cat([anchor_hidden, prev_token_embed], dim=-1)
        x = self.fc1(x)
        x = self.norm1(F.silu(x).float()).to(dtype=anchor_hidden.dtype)
        x = self.fc2(x)
        x = self.norm2(F.silu(x).float()).to(dtype=anchor_hidden.dtype)
        logits = lm_head(x)[..., :vocab_size].float()
        softcap = 15
        return softcap * torch.tanh(logits / softcap)


class MoEExpert(nn.Module):
    """OpenMythos-style SwiGLU expert used inside routed/shared MoE FFNs."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = Linear(dim, hidden_dim, bias=False)
        self.up = Linear(dim, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RoutedSharedMoEFFN(nn.Module):
    """Fine-grained routed experts plus always-on shared experts, OpenMythos style."""
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.loop_moe_num_experts
        self.n_shared = max(config.loop_moe_shared_experts, 0)
        self.topk = config.loop_moe_topk()
        self.bias_update_rate = config.loop_moe_bias_update_rate
        self.bias_update_clamp = config.loop_moe_bias_update_clamp
        expert_dim = config.loop_moe_width()
        assert self.n_experts > 0
        assert 1 <= self.topk <= self.n_experts

        self.router = Linear(config.n_embd, self.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(self.n_experts))
        self.register_buffer("router_usage", torch.zeros(self.n_experts), persistent=False)
        self.register_buffer("router_usage_total", torch.zeros(()), persistent=False)
        self.routed_experts = nn.ModuleList([
            MoEExpert(config.n_embd, expert_dim) for _ in range(self.n_experts)
        ])
        self.shared_experts = nn.ModuleList([
            MoEExpert(config.n_embd, expert_dim * self.topk) for _ in range(self.n_shared)
        ])

    def forward(self, x, return_router_z_loss=False):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        logits = self.router(flat).float() + self.router_bias
        router_z_loss = logits.logsumexp(dim=-1).square().mean() if return_router_z_loss else None
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = scores.topk(self.topk, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.training and self.bias_update_rate > 0:
            self._accumulate_router_usage(topk_idx)

        # Keep dispatch shapes static for torch.compile: evaluate the small fine-grained
        # expert bank densely, then gather only the routed top-k outputs for each token.
        expert_outputs = torch.stack([expert(flat) for expert in self.routed_experts], dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        out = expert_outputs.gather(1, gather_idx)
        out = out * topk_scores.to(dtype=flat.dtype).unsqueeze(-1)
        out = out.sum(dim=1)

        for expert in self.shared_experts:
            out = out + expert(flat)

        out = out.view(B, T, D)
        if return_router_z_loss:
            return out, router_z_loss
        return out

    @torch.no_grad()
    def _accumulate_router_usage(self, topk_idx):
        usage = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=(0, 1))
        usage = usage.to(device=self.router_usage.device, dtype=self.router_usage.dtype)
        self.router_usage.add_(usage)
        self.router_usage_total.add_(usage.sum())

    @torch.no_grad()
    def update_router_bias(self):
        if self.bias_update_rate <= 0:
            self.router_usage.zero_()
            self.router_usage_total.zero_()
            return

        usage = self.router_usage.float().clone()
        total = self.router_usage_total.float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(usage, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        self.router_usage.zero_()
        self.router_usage_total.zero_()
        if total.item() <= 0:
            return

        target = total / self.n_experts
        delta = torch.sign(target - usage) * self.bias_update_rate
        self.router_bias.add_(delta.to(dtype=self.router_bias.dtype))
        if self.bias_update_clamp > 0:
            self.router_bias.clamp_(-self.bias_update_clamp, self.bias_update_clamp)

    def active_parameters(self):
        routed = self.topk * sum(p.numel() for p in self.routed_experts[0].parameters())
        shared = sum(p.numel() for expert in self.shared_experts for p in expert.parameters())
        router = self.router.weight.numel()
        return router + routed + shared


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_value_embedding=None, use_moe_mlp=False):
        super().__init__()
        if config.use_mla_attention():
            use_value_embedding = False
        self.attn = CausalSelfAttention(config, layer_idx, use_value_embedding=use_value_embedding)
        self.mlp = RoutedSharedMoEFFN(config) if use_moe_mlp else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cache_layer_idx=None, router_z_loss=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cache_layer_idx=cache_layer_idx)
        mlp_in = norm(x)
        if router_z_loss is not None and isinstance(self.mlp, RoutedSharedMoEFFN):
            mlp_out, z_loss = self.mlp(mlp_in, return_router_z_loss=True)
            router_z_loss = router_z_loss + z_loss
        else:
            mlp_out = self.mlp(mlp_in)
        x = x + mlp_out
        if router_z_loss is not None:
            return x, router_z_loss
        return x


class MHCHyperConnections(nn.Module):
    """DeepSeek mHC-style n-stream residual wrapper with Sinkhorn H_res maps."""
    def __init__(self, config, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_streams = config.mhc_num_streams
        self.n_embd = config.n_embd
        self.sinkhorn_iters = max(config.mhc_sinkhorn_iters, 1)
        self.dynamic_init_scale = config.mhc_dynamic_init_scale
        self.residual_init_strength = config.mhc_residual_init_strength
        coeff_dim = self.n_streams * self.n_streams + 2 * self.n_streams
        input_dim = self.n_streams * self.n_embd
        self.coeff_proj = nn.ModuleList([
            Linear(input_dim, coeff_dim, bias=False) for _ in range(n_layers)
        ])
        self.alpha = nn.Parameter(torch.ones(n_layers, 3))
        self.bias = nn.Parameter(torch.zeros(n_layers, coeff_dim))

    def initial_state(self, x):
        return x.unsqueeze(2).expand(-1, -1, self.n_streams, -1).contiguous()

    def readout(self, state):
        return state.mean(dim=2)

    def init_weights(self):
        input_dim = self.n_streams * self.n_embd
        bound = (3**0.5) * (input_dim ** -0.5) * self.dynamic_init_scale
        for proj in self.coeff_proj:
            torch.nn.init.uniform_(proj.weight, -bound, bound)
        torch.nn.init.ones_(self.alpha)

        pre_value = min(max(1.0 / self.n_streams, 1e-4), 1.0 - 1e-4)
        pre_bias = math.log(pre_value / (1.0 - pre_value))
        res_bias = torch.full(
            (self.n_streams, self.n_streams),
            -self.residual_init_strength,
            device=self.bias.device,
            dtype=self.bias.dtype,
        )
        diag = torch.arange(self.n_streams, device=self.bias.device)
        res_bias[diag, diag] = self.residual_init_strength
        for layer_bias in self.bias:
            layer_bias[:self.n_streams].fill_(pre_bias)
            layer_bias[self.n_streams:2 * self.n_streams].zero_()
            layer_bias[2 * self.n_streams:].copy_(res_bias.reshape(-1))

    def _layer_idx(self, layer_idx):
        return min(layer_idx, self.n_layers - 1)

    @_disable_torch_compile
    def mappings(self, state, layer_idx):
        B, T, N, D = state.shape
        assert N == self.n_streams and D == self.n_embd
        layer_idx = self._layer_idx(layer_idx)
        flat = state.reshape(B, T, N * D)
        # The mHC coefficient path is kept in fp32 for Sinkhorn stability.
        raw = F.linear(norm(flat).float(), self.coeff_proj[layer_idx].weight.float())
        pre_raw, post_raw, res_raw = torch.split(
            raw,
            [self.n_streams, self.n_streams, self.n_streams * self.n_streams],
            dim=-1,
        )
        bias = self.bias[layer_idx].float()
        pre_bias, post_bias, res_bias = torch.split(
            bias,
            [self.n_streams, self.n_streams, self.n_streams * self.n_streams],
            dim=-1,
        )
        alpha = self.alpha[layer_idx].float()
        h_pre = torch.sigmoid(alpha[0] * pre_raw + pre_bias)
        h_post = 2.0 * torch.sigmoid(alpha[1] * post_raw + post_bias)
        h_res_logits = alpha[2] * res_raw + res_bias
        h_res = self._sinkhorn(h_res_logits.view(B, T, self.n_streams, self.n_streams))
        return h_pre, h_post, h_res

    @_disable_torch_compile
    def apply_pre(self, state, h_pre):
        return torch.einsum("btn,btnd->btd", h_pre.to(dtype=state.dtype), state)

    @_disable_torch_compile
    def apply_post_res(self, state, residual, h_post, h_res):
        mixed = torch.einsum("btij,btjd->btid", h_res.to(dtype=state.dtype), state)
        return mixed + h_post.to(dtype=state.dtype).unsqueeze(-1) * residual.unsqueeze(2)

    @_disable_torch_compile
    def _sinkhorn(self, logits):
        matrix = torch.exp(logits.float().clamp(-20, 20))
        for _ in range(self.sinkhorn_iters):
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-12)
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return matrix


# Recurrent-depth components adapted from OpenMythos (MIT License),
# Copyright (c) 2026 Kye Gomez.
def loop_index_embedding(x, loop_idx, loop_dim, theta=10000.0):
    """Add a sinusoidal recurrent-depth signal to the first loop_dim channels."""
    if loop_dim <= 0:
        return x
    loop_dim = min(loop_dim, x.size(-1))
    loop_dim = loop_dim - (loop_dim % 2)
    if loop_dim <= 0:
        return x
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=x.device, dtype=x.dtype) / loop_dim))
    angles = loop_idx * freqs
    emb = torch.empty(loop_dim, device=x.device, dtype=x.dtype)
    emb[0::2] = angles.sin()
    emb[1::2] = angles.cos()
    return x + F.pad(emb, (0, x.size(-1) - loop_dim)).view(1, 1, -1)


class LTIInjection(nn.Module):
    """OpenMythos-style LTI-stable update h_next = A*h + B*e + transformer_out."""
    def __init__(self, dim):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self):
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h, encoded_input, transformer_out):
        A = self.get_A().to(dtype=h.dtype)
        B = self.B.to(dtype=h.dtype)
        return A * h + B * encoded_input + transformer_out


class DepthLoRAAdapter(nn.Module):
    """A small per-loop adapter for the shared recurrent block."""
    def __init__(self, dim, rank, max_loops):
        super().__init__()
        self.down = Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.zeros(rank, dim))
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x, loop_idx):
        idx = min(loop_idx, self.scale.num_embeddings - 1)
        scale = self.scale(torch.tensor(idx, device=x.device))
        return (self.down(x) * scale.to(dtype=x.dtype)) @ self.B.to(dtype=x.dtype)


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting probabilities for recurrent depth."""
    def __init__(self, dim):
        super().__init__()
        self.halt = Linear(dim, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.halt(x).float()).squeeze(-1)


class RecurrentBlock(nn.Module):
    """A single nanochat Block reused across virtual depth, OpenMythos style."""
    def __init__(self, config, layer_idx, max_loop_iters):
        super().__init__()
        self.config = config
        self.block = Block(
            config,
            layer_idx,
            use_value_embedding=True,
            use_moe_mlp=config.loop_moe_enabled(),
        )
        self.injection = LTIInjection(config.n_embd)
        self.lora = DepthLoRAAdapter(config.n_embd, config.loop_lora_rank, max_loop_iters) if config.loop_lora_rank > 0 else None
        self.act = ACTHalting(config.n_embd) if config.loop_act_threshold > 0 else None
        loop_dim = int(config.n_embd * config.loop_embed_fraction)
        self.loop_dim = loop_dim - (loop_dim % 2)

    def forward(
        self,
        h,
        encoded_input,
        x0,
        cos_sin,
        window_sizes,
        kv_cache,
        start_layer_idx,
        resid_lambdas,
        x0_lambdas,
        value_embeds,
        mhc=None,
        mhc_state=None,
        x0_mhc_state=None,
        router_z_loss=None,
    ):
        use_act = self.act is not None
        use_mhc = mhc is not None
        h_out = torch.zeros_like(h) if use_act else None
        mhc_state_out = torch.zeros_like(mhc_state) if use_act and use_mhc else None
        cumulative_p = None
        halted = None
        if use_mhc:
            assert mhc_state is not None and x0_mhc_state is not None
        if use_act:
            B, T, _ = h.shape
            cumulative_p = torch.zeros(B, T, device=h.device)
            halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)

        for loop_idx, window_size in enumerate(window_sizes):
            cache_layer_idx = start_layer_idx + loop_idx
            if use_mhc:
                layer_state = (
                    resid_lambdas[loop_idx].to(dtype=mhc_state.dtype) * mhc_state
                    + x0_lambdas[loop_idx].to(dtype=mhc_state.dtype) * x0_mhc_state
                )
                h_pre, h_post, h_res = mhc.mappings(layer_state, cache_layer_idx)
                h_in = mhc.apply_pre(layer_state, h_pre)
                h_loop = loop_index_embedding(h_in, loop_idx, self.loop_dim)
                combined = norm(h_loop + encoded_input)
                ve = value_embeds[loop_idx]
                transformer_out = self.block(combined, ve, cos_sin, window_size, kv_cache, cache_layer_idx=cache_layer_idx, router_z_loss=router_z_loss)
                if router_z_loss is not None:
                    transformer_out, router_z_loss = transformer_out
                if self.lora is not None:
                    transformer_out = transformer_out + self.lora(transformer_out, loop_idx)
                h_next = self.injection(h_in, encoded_input, transformer_out)
                mhc_state = mhc.apply_post_res(layer_state, h_next - h_in, h_post, h_res)
                h = mhc.readout(mhc_state)
            else:
                h_in = resid_lambdas[loop_idx] * h + x0_lambdas[loop_idx] * x0
                h_loop = loop_index_embedding(h_in, loop_idx, self.loop_dim)
                combined = norm(h_loop + encoded_input)
                ve = value_embeds[loop_idx]
                transformer_out = self.block(combined, ve, cos_sin, window_size, kv_cache, cache_layer_idx=cache_layer_idx, router_z_loss=router_z_loss)
                if router_z_loss is not None:
                    transformer_out, router_z_loss = transformer_out
                if self.lora is not None:
                    transformer_out = transformer_out + self.lora(transformer_out, loop_idx)
                h = self.injection(h, encoded_input, transformer_out)

            if use_act:
                p = self.act(h)
                still_running = ~halted
                remainder = (1.0 - cumulative_p).clamp(min=0.0)
                crosses = cumulative_p + p >= self.config.loop_act_threshold
                weight = torch.where(crosses, remainder, p) * still_running.float()
                h_out = h_out + weight.unsqueeze(-1).to(dtype=h.dtype) * h
                if use_mhc:
                    mhc_state_out = mhc_state_out + weight.unsqueeze(-1).unsqueeze(-1).to(dtype=mhc_state.dtype) * mhc_state
                new_cumulative_p = cumulative_p + p * still_running.float()
                just_halted = still_running & crosses
                cumulative_p = torch.where(just_halted, torch.ones_like(cumulative_p), new_cumulative_p)
                halted = halted | just_halted
                # With a KV cache, skipped virtual layers would leave cache holes.
                # Keep cache-backed inference dense; training and naive generation can short-circuit.
                if kv_cache is None and halted.all():
                    break

        if use_act:
            still_running = ~halted
            if still_running.any():
                remainder = (1.0 - cumulative_p).clamp(min=0.0) * still_running.float()
                h_out = h_out + remainder.unsqueeze(-1).to(dtype=h.dtype) * h
                if use_mhc:
                    mhc_state_out = mhc_state_out + remainder.unsqueeze(-1).unsqueeze(-1).to(dtype=mhc_state.dtype) * mhc_state
            if use_mhc:
                result = (h_out, mhc_state_out)
            else:
                result = h_out
            if router_z_loss is not None:
                return result, router_z_loss
            return result
        if use_mhc:
            result = (h, mhc_state)
            if router_z_loss is not None:
                return result, router_z_loss
            return result
        if router_z_loss is not None:
            return h, router_z_loss
        return h


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.use_looped_transformer = config.use_looped_transformer
        self.loop_prelude_layers, self.loop_iters, self.loop_coda_layers = config.loop_layout()
        self.effective_n_layer = config.num_kv_layers()
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config, self.effective_n_layer)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        transformer = {"wte": nn.Embedding(padded_vocab_size, config.n_embd)}
        if self.use_looped_transformer:
            recurrent_layer_idx = self.loop_prelude_layers
            coda_start_idx = self.loop_prelude_layers + self.loop_iters
            transformer["prelude"] = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(self.loop_prelude_layers)])
            transformer["recurrent"] = RecurrentBlock(config, recurrent_layer_idx, max(self.loop_iters, 1))
            transformer["coda"] = nn.ModuleList([Block(config, coda_start_idx + i) for i in range(self.loop_coda_layers)])
        else:
            transformer["h"] = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.transformer = nn.ModuleDict(transformer)
        self.mhc = MHCHyperConnections(config, self.effective_n_layer) if config.use_mhc() else None
        self.apple_mtp = AppleMTPSamplerHead(config) if config.apple_mtp_enabled() else None
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(self.effective_n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(self.effective_n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        if config.use_mla_attention():
            # MLA compressed decode reconstructs values from cached c_kv, so
            # nanochat's value embeddings are kept on the GQA path only.
            self.value_embeds = nn.ModuleDict()
        else:
            head_dim = config.attention_head_dim()
            kv_dim = config.n_kv_head * head_dim
            self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(self.effective_n_layer) if has_ve(i, self.effective_n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, config.rotary_dim())
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _unique_blocks(self):
        if self.use_looped_transformer:
            return list(self.transformer.prelude) + [self.transformer.recurrent.block] + list(self.transformer.coda)
        return list(self.transformer.h)

    def _loop_control_parameters(self):
        if not self.use_looped_transformer:
            return []
        recurrent = self.transformer.recurrent
        params = list(recurrent.injection.parameters())
        if recurrent.lora is not None:
            params.extend(recurrent.lora.parameters())
        if recurrent.act is not None:
            params.extend(recurrent.act.parameters())
        return params

    def _apple_mtp_parameters(self):
        return [] if self.apple_mtp is None else list(self.apple_mtp.parameters())

    def _active_block_matmul_params(self, block):
        total = sum(p.numel() for p in block.parameters())
        if hasattr(block.mlp, "active_parameters"):
            stored_mlp = sum(p.numel() for p in block.mlp.parameters())
            total = total - stored_mlp + block.mlp.active_parameters()
        return total

    def _mix_layer_input(self, x, x0, layer_idx):
        idx = min(layer_idx, self.resid_lambdas.numel() - 1)
        return self.resid_lambdas[idx] * x + self.x0_lambdas[idx] * x0

    def _mix_mhc_state(self, state, x0_state, layer_idx):
        idx = min(layer_idx, self.resid_lambdas.numel() - 1)
        return self.resid_lambdas[idx].to(dtype=state.dtype) * state + self.x0_lambdas[idx].to(dtype=state.dtype) * x0_state

    def _forward_mhc_block(self, state, x0_state, block, idx, layer_idx, cos_sin, window_size, kv_cache, router_z_loss=None):
        layer_state = self._mix_mhc_state(state, x0_state, layer_idx)
        h_pre, h_post, h_res = self.mhc.mappings(layer_state, layer_idx)
        block_in = self.mhc.apply_pre(layer_state, h_pre)
        ve = self._value_embedding(idx, layer_idx, block_in.dtype)
        block_out = block(block_in, ve, cos_sin, window_size, kv_cache, cache_layer_idx=layer_idx, router_z_loss=router_z_loss)
        if router_z_loss is not None:
            block_out, router_z_loss = block_out
        state = self.mhc.apply_post_res(layer_state, block_out - block_in, h_post, h_res)
        return state, self.mhc.readout(state), router_z_loss

    def _layer_lambdas(self, start_layer_idx, count):
        last_idx = self.resid_lambdas.numel() - 1
        resid_lambdas = [self.resid_lambdas[min(start_layer_idx + i, last_idx)] for i in range(count)]
        x0_lambdas = [self.x0_lambdas[min(start_layer_idx + i, last_idx)] for i in range(count)]
        return resid_lambdas, x0_lambdas

    def _value_embedding(self, idx, layer_idx, dtype):
        key = str(layer_idx)
        if key not in self.value_embeds:
            return None
        return self.value_embeds[key](idx).to(dtype)

    @torch.no_grad()
    def update_router_biases(self):
        for module in self.modules():
            if isinstance(module, RoutedSharedMoEFFN):
                module.update_router_bias()

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attention input projections: uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

        def init_mlp(mlp):
            if isinstance(mlp, MLP):
                torch.nn.init.uniform_(mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(mlp.c_proj.weight)
                return
            if isinstance(mlp, RoutedSharedMoEFFN):
                torch.nn.init.uniform_(mlp.router.weight, -s, s)
                for expert in list(mlp.routed_experts) + list(mlp.shared_experts):
                    torch.nn.init.uniform_(expert.gate.weight, -s * 0.4, s * 0.4)
                    torch.nn.init.uniform_(expert.up.weight, -s * 0.4, s * 0.4)
                    torch.nn.init.zeros_(expert.down.weight)
                return
            raise TypeError(f"Unknown MLP type: {type(mlp).__name__}")

        def init_attn(attn):
            for name in ("c_q", "c_k", "c_v", "q_a", "q_b", "kv_down", "kv_up", "index_q", "index_k", "index_weights"):
                module = getattr(attn, name, None)
                if module is not None:
                    torch.nn.init.uniform_(module.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.zeros_(attn.c_proj.weight) # projections are zero

        for block in self._unique_blocks():
            init_attn(block.attn)
            init_mlp(block.mlp)
        if self.mhc is not None:
            self.mhc.init_weights()
        if self.apple_mtp is not None:
            torch.nn.init.normal_(self.apple_mtp.mask_embeddings.weight, mean=0.0, std=0.8)
            if self.apple_mtp.gated_lora is not None:
                torch.nn.init.uniform_(self.apple_mtp.gated_lora.down.weight, -s, s)
                torch.nn.init.zeros_(self.apple_mtp.gated_lora.B)
            torch.nn.init.uniform_(self.apple_mtp.fc1.weight, -s, s)
            torch.nn.init.zeros_(self.apple_mtp.fc2.weight)
            torch.nn.init.ones_(self.apple_mtp.norm1.weight)
            torch.nn.init.zeros_(self.apple_mtp.norm1.bias)
            torch.nn.init.ones_(self.apple_mtp.norm2.weight)
            torch.nn.init.zeros_(self.apple_mtp.norm2.bias)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.effective_n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self._unique_blocks():
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # OpenMythos-inspired recurrent controls. Keep LoRA initially neutral
        # by zeroing B, while the down projection and loop scale can learn.
        if self.use_looped_transformer:
            recurrent = self.transformer.recurrent
            torch.nn.init.zeros_(recurrent.injection.log_A)
            torch.nn.init.zeros_(recurrent.injection.log_dt)
            torch.nn.init.constant_(recurrent.injection.B, 0.1)
            if recurrent.lora is not None:
                torch.nn.init.uniform_(recurrent.lora.down.weight, -s, s)
                torch.nn.init.zeros_(recurrent.lora.B)
                torch.nn.init.ones_(recurrent.lora.scale.weight)
            if recurrent.act is not None:
                torch.nn.init.zeros_(recurrent.act.halt.weight)

        # Rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.config.rotary_dim())
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)
            if self.apple_mtp is not None:
                self.apple_mtp.mask_embeddings.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config, n_layers=None):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        n_layers = config.n_layer if n_layers is None else n_layers
        for layer_idx in range(n_layers):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        apple_mtp_non_matmul = 0
        if self.apple_mtp is not None:
            apple_mtp_non_matmul = (
                self.apple_mtp.mask_embeddings.weight.numel()
                + self.apple_mtp.norm1.weight.numel() + self.apple_mtp.norm1.bias.numel()
                + self.apple_mtp.norm2.weight.numel() + self.apple_mtp.norm2.bias.numel()
            )
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() +
                          self.backout_lambda.numel() + apple_mtp_non_matmul)
        matmul_params = nparams - nparams_exclude
        if self.use_looped_transformer:
            prelude_params = sum(self._active_block_matmul_params(block) for block in self.transformer.prelude)
            recurrent_params = self._active_block_matmul_params(self.transformer.recurrent.block)
            coda_params = sum(self._active_block_matmul_params(block) for block in self.transformer.coda)
            matmul_params = self.lm_head.weight.numel() + prelude_params + self.loop_iters * recurrent_params + coda_params
            recurrent = self.transformer.recurrent
            if recurrent.lora is not None:
                matmul_params += self.loop_iters * (recurrent.lora.down.weight.numel() + recurrent.lora.B.numel())
            if recurrent.act is not None:
                matmul_params += self.loop_iters * recurrent.act.halt.weight.numel()
            if self.mhc is not None:
                matmul_params += sum(p.numel() for p in self.mhc.parameters())
        h = self.config.n_head
        q = self.config.mla_qk_head_dim() if self.config.use_mla_attention() else self.config.attention_head_dim()
        t = self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * matmul_params + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for block in self._unique_blocks() for p in block.parameters())
        loop_controls = sum(p.numel() for p in self._loop_control_parameters())
        mhc = sum(p.numel() for p in self.mhc.parameters()) if self.mhc is not None else 0
        apple_mtp = sum(p.numel() for p in self._apple_mtp_parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + loop_controls + mhc + apple_mtp + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'loop_controls': loop_controls,
            'mhc': mhc,
            'apple_mtp': apple_mtp,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = [p for block in self._unique_blocks() for p in block.parameters()]
        loop_control_params = self._loop_control_parameters()
        mhc_params = list(self.mhc.parameters()) if self.mhc is not None else []
        apple_mtp_params = self._apple_mtp_parameters()
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(loop_control_params) + len(mhc_params) + len(apple_mtp_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if loop_control_params:
            param_groups.append(dict(kind='adamw', params=loop_control_params, lr=scalar_lr * 0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if mhc_params:
            param_groups.append(dict(kind='adamw', params=mhc_params, lr=scalar_lr * 0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if apple_mtp_params:
            param_groups.append(dict(kind='adamw', params=apple_mtp_params, lr=scalar_lr * 0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _apple_mtp_slot_logits(self, anchor_hidden, prev_token_ids, slot_idx):
        assert self.apple_mtp is not None
        anchor_hidden = self.apple_mtp.apply_mask_slot(anchor_hidden, slot_idx)
        prev_token_ids = prev_token_ids.clamp_min(0)
        prev_emb = self.transformer.wte(prev_token_ids).to(dtype=anchor_hidden.dtype)
        return self.apple_mtp(anchor_hidden, prev_emb, self.lm_head, self.config.vocab_size)

    def _apple_mtp_aux_loss(self, hidden, logits, targets, loss_reduction):
        if self.apple_mtp is None or self.config.apple_mtp_loss_weight <= 0:
            return None
        B, T, _ = hidden.shape
        mtp_losses = []
        consistency_losses = []
        for slot_idx in range(self.config.apple_mtp_num_future_tokens):
            # Apple MTP keeps ordinary NTP for t+1. The MTP slots learn t+2, t+3, ...
            offset = slot_idx + 2
            if T <= offset:
                continue
            anchor = hidden[:, :T - offset]
            prev_ids = targets[:, offset - 2:T - 2]
            target_ids = targets[:, offset - 1:T - 1]
            mtp_logits = self._apple_mtp_slot_logits(anchor, prev_ids, slot_idx)
            flat_loss = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-1,
                reduction="none",
            ).view_as(target_ids)
            valid = target_ids != -1
            if loss_reduction == "sum":
                mtp_losses.append(flat_loss.sum())
            else:
                mtp_losses.append(flat_loss.sum() / valid.sum().clamp_min(1))

            if self.config.apple_mtp_consistency_weight > 0:
                ref_logits = logits[:, offset - 1:T - 1].detach()
                kl = F.kl_div(
                    F.log_softmax(mtp_logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                kl = kl.masked_fill(~valid, 0.0)
                if loss_reduction == "sum":
                    consistency_losses.append(kl.sum())
                else:
                    consistency_losses.append(kl.sum() / valid.sum().clamp_min(1))

        if not mtp_losses:
            return None
        mtp_loss = torch.stack(mtp_losses).sum() if loss_reduction == "sum" else torch.stack(mtp_losses).mean()
        total = self.config.apple_mtp_loss_weight * mtp_loss
        if consistency_losses:
            consistency_loss = torch.stack(consistency_losses).sum() if loss_reduction == "sum" else torch.stack(consistency_losses).mean()
            total = total + self.config.apple_mtp_consistency_weight * consistency_loss
        return total

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', loop_iters=None, return_hidden=False):
        B, T = idx.size()
        effective_n_layer = self.config.num_kv_layers(loop_iters)
        if kv_cache is not None:
            assert kv_cache.n_layers == effective_n_layer, f"KV cache has {kv_cache.n_layers} layers, model forward needs {effective_n_layer}"
        window_sizes = self.window_sizes if effective_n_layer == self.effective_n_layer else self._compute_window_sizes(self.config, effective_n_layer)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        mhc_state = self.mhc.initial_state(x) if self.mhc is not None else None
        x0_mhc_state = mhc_state
        router_z_loss = x.new_zeros(()) if (self.training and targets is not None and loss_reduction != 'none' and self.config.loop_moe_router_z_loss_enabled()) else None
        backout_layer = effective_n_layer // 2  # cache at halfway point
        x_backout = None

        if self.use_looped_transformer:
            layer_idx = 0
            prelude_layers, recurrent_iters, coda_layers = self.config.loop_layout(loop_iters)
            assert prelude_layers == self.loop_prelude_layers and coda_layers == self.loop_coda_layers
            for block in self.transformer.prelude:
                if self.mhc is not None:
                    mhc_state, x, router_z_loss = self._forward_mhc_block(mhc_state, x0_mhc_state, block, idx, layer_idx, cos_sin, window_sizes[layer_idx], kv_cache, router_z_loss=router_z_loss)
                else:
                    x = self._mix_layer_input(x, x0, layer_idx)
                    ve = self._value_embedding(idx, layer_idx, x.dtype)
                    x = block(x, ve, cos_sin, window_sizes[layer_idx], kv_cache, cache_layer_idx=layer_idx, router_z_loss=router_z_loss)
                    if router_z_loss is not None:
                        x, router_z_loss = x
                if layer_idx == backout_layer:
                    x_backout = x
                layer_idx += 1

            encoded_input = x
            recurrent_windows = window_sizes[layer_idx:layer_idx + recurrent_iters]
            recurrent_resid_lambdas, recurrent_x0_lambdas = self._layer_lambdas(layer_idx, recurrent_iters)
            recurrent_value_embeds = [self._value_embedding(idx, layer_idx + i, x.dtype) for i in range(recurrent_iters)]
            recurrent_out = self.transformer.recurrent(
                x, encoded_input, x0, cos_sin, recurrent_windows, kv_cache, layer_idx,
                recurrent_resid_lambdas, recurrent_x0_lambdas, recurrent_value_embeds,
                mhc=self.mhc, mhc_state=mhc_state, x0_mhc_state=x0_mhc_state,
                router_z_loss=router_z_loss,
            )
            if router_z_loss is not None:
                recurrent_out, router_z_loss = recurrent_out
            if self.mhc is not None:
                x, mhc_state = recurrent_out
            else:
                x = recurrent_out
            if layer_idx <= backout_layer < layer_idx + recurrent_iters:
                x_backout = x
            layer_idx += recurrent_iters

            for block in self.transformer.coda:
                if self.mhc is not None:
                    mhc_state, x, router_z_loss = self._forward_mhc_block(mhc_state, x0_mhc_state, block, idx, layer_idx, cos_sin, window_sizes[layer_idx], kv_cache, router_z_loss=router_z_loss)
                else:
                    x = self._mix_layer_input(x, x0, layer_idx)
                    ve = self._value_embedding(idx, layer_idx, x.dtype)
                    x = block(x, ve, cos_sin, window_sizes[layer_idx], kv_cache, cache_layer_idx=layer_idx, router_z_loss=router_z_loss)
                    if router_z_loss is not None:
                        x, router_z_loss = x
                if layer_idx == backout_layer:
                    x_backout = x
                layer_idx += 1
            assert layer_idx == effective_n_layer
        else:
            for i, block in enumerate(self.transformer.h):
                if self.mhc is not None:
                    mhc_state, x, router_z_loss = self._forward_mhc_block(mhc_state, x0_mhc_state, block, idx, i, cos_sin, window_sizes[i], kv_cache, router_z_loss=router_z_loss)
                else:
                    x = self._mix_layer_input(x, x0, i)
                    ve = self._value_embedding(idx, i, x.dtype)
                    x = block(x, ve, cos_sin, window_sizes[i], kv_cache, cache_layer_idx=i, router_z_loss=router_z_loss)
                    if router_z_loss is not None:
                        x, router_z_loss = x
                if i == backout_layer:
                    x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if router_z_loss is not None:
                loss = loss + self.config.loop_moe_router_z_loss_weight * router_z_loss
            if self.training and loss_reduction != 'none':
                apple_mtp_loss = self._apple_mtp_aux_loss(x, logits, targets, loss_reduction)
                if apple_mtp_loss is not None:
                    loss = loss + apple_mtp_loss
            return loss
        else:
            # inference: just return the logits directly
            if return_hidden:
                return logits, x
            return logits

    @torch.inference_mode()
    def predict_mtp(self, tokens, num_future_tokens=None):
        """Return next-token logits plus Apple-style MTP future logits/predictions."""
        assert self.apple_mtp is not None, "Apple MTP is disabled; set apple_mtp_num_future_tokens > 0"
        assert isinstance(tokens, list)
        device = self.get_device()
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, hidden = self.forward(idx, return_hidden=True)
        ntp_logits = logits[:, -1:, :]
        prev_ids = torch.argmax(ntp_logits, dim=-1)
        num_future_tokens = self.config.apple_mtp_num_future_tokens if num_future_tokens is None else min(num_future_tokens, self.config.apple_mtp_num_future_tokens)
        mtp_logits = []
        mtp_ids = []
        anchor = hidden[:, -1:, :]
        for slot_idx in range(num_future_tokens):
            slot_logits = self._apple_mtp_slot_logits(anchor, prev_ids, slot_idx)
            prev_ids = torch.argmax(slot_logits, dim=-1)
            mtp_logits.append(slot_logits)
            mtp_ids.append(prev_ids)
        mtp_logits = torch.cat(mtp_logits, dim=1) if mtp_logits else logits.new_empty(1, 0, self.config.vocab_size)
        mtp_ids = torch.cat(mtp_ids, dim=1) if mtp_ids else idx.new_empty(1, 0)
        return {
            "ntp_logits": ntp_logits,
            "mtp_logits": mtp_logits,
            "token_ids": torch.cat([torch.argmax(ntp_logits, dim=-1), mtp_ids], dim=1),
        }

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42, loop_iters=None):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids, loop_iters=loop_iters) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
