"""Small native nanochat GPT baseline used by the OpenMythos tests."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "L"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Linear layer that keeps fp32 weights and casts them for compute."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head
        assert config.n_head % config.n_kv_head == 0
        self.c_q = Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        batch, time, _ = x.shape
        cos, sin = cos_sin
        q = self.c_q(x).view(batch, time, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, time, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch, time, self.n_kv_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin)).transpose(1, 2)
        k = norm(apply_rotary_emb(k, cos, sin)).transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_head != self.n_kv_head)
        y = y.transpose(1, 2).contiguous().view(batch, time, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        bound = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -bound, bound)
            torch.nn.init.uniform_(block.attn.c_k.weight, -bound, bound)
            torch.nn.init.uniform_(block.attn.c_v.weight, -bound, bound)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * bound, 0.4 * bound)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len,
            head_dim,
            device=self.transformer.wte.weight.device,
        )
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        device = self.transformer.wte.weight.device if device is None else device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :].to(COMPUTE_DTYPE), sin[None, :, None, :].to(COMPUTE_DTYPE)

    def forward(self, idx, targets=None, loss_reduction="mean"):
        _, time = idx.shape
        assert time <= self.cos.size(1)
        cos_sin = self.cos[:, :time], self.sin[:, :time]
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
