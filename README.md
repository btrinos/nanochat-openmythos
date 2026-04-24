# nanochat-openmythos

This repository is a compact PyTorch experiment that ports
[OpenMythos](https://github.com/kyegomez/OpenMythos)-style looped transformer
ideas into a small nanochat-style GPT implementation. It also keeps opt-in
DeepSeek-inspired MLA/DSA attention, mHC residual streams, and an Apple-style
multi-token prediction head in the OpenMythos module so the pieces can be tested
together.

It is not a checkpoint loader, tokenizer package, training data bundle, or
production training stack.

## Attribution

This experiment is built from Andrej Karpathy's
[nanochat](https://github.com/karpathy/nanochat) codebase and keeps a small
native GPT baseline for comparison.

The recurrent-depth work incorporates ideas and adapted implementation pieces
from [OpenMythos](https://github.com/kyegomez/OpenMythos), including the looped
transformer topology and recurrent-depth control components. OpenMythos is MIT
licensed; see [NOTICE.md](NOTICE.md) for the attribution scope retained in this
repo.

The MLA/DSA and mHC paths are independent small-scale PyTorch implementations
inspired by public DeepSeek research/code descriptions. The MTP path is an
independent small-scale PyTorch implementation inspired by Apple's public
multi-token prediction paper.

## What Is Implemented

The main model is [nanochat/openmythos.py](nanochat/openmythos.py). A small
native baseline is kept in [nanochat/gpt.py](nanochat/gpt.py).

| file | nanochat surface touched | addition |
|---|---|---|
| `nanochat/openmythos.py` | `GPTConfig` recurrent knobs | Adds `use_looped_transformer`, `loop_prelude_layers`, `loop_coda_layers`, `loop_iters`, `loop_lora_rank`, `loop_act_threshold`, `loop_embed_fraction`, and recurrent MoE controls. Adds `loop_layout()` and `num_kv_layers()` so the model can distinguish physical blocks from virtual recurrent depth. |
| `nanochat/openmythos.py` | `GPTConfig` attention knobs | Adds `attn_type`, `mla_q_lora_rank`, `mla_kv_lora_rank`, `mla_qk_nope_head_dim`, `mla_qk_rope_head_dim`, `mla_v_head_dim`, `dsa_topk`, `dsa_index_n_heads`, and `dsa_index_head_dim`. The default remains `attn_type="gqa"`. |
| `nanochat/openmythos.py` | `GPTConfig` mHC knobs | Adds `mhc_num_streams`, `mhc_sinkhorn_iters`, `mhc_dynamic_init_scale`, and `mhc_residual_init_strength`. `mhc_num_streams=1` keeps the original residual path; values above 1 enable mHC. |
| `nanochat/openmythos.py` | `GPTConfig` Apple MTP knobs | Adds `apple_mtp_num_future_tokens`, `apple_mtp_loss_weight`, `apple_mtp_consistency_weight`, `apple_mtp_gated_lora_rank`, and `apple_mtp_sampler_hidden_mult`. `apple_mtp_num_future_tokens=0` keeps MTP disabled. |
| `nanochat/openmythos.py` | `CausalSelfAttention` | Keeps nanochat's GQA path as the default. Adds an opt-in portable MLA/DSA branch with low-rank query projection, latent KV projection, separate NoPE/RoPE query-key channels, a decoupled RoPE key, compressed `c_kv + k_rope` decode cache, value-head width control, and a lightweight DSA indexer that selects top-K cache tokens before the softmax. |
| `nanochat/openmythos.py` | `MHCHyperConnections` | Adds a DeepSeek mHC-inspired residual-stream wrapper. It expands hidden state into `n` streams, computes nonnegative `H_pre` and `H_post` maps, projects `H_res` onto the doubly stochastic manifold with Sinkhorn-Knopp iterations, applies the block to the `H_pre` readout, and writes the residual update back across streams. |
| `nanochat/openmythos.py` | `AppleMTPSamplerHead` | Adds a compact Apple-style MTP path with learned mask-slot embeddings, an MTP-only gated LoRA adapter, a two-block sampler head conditioned on the previous token embedding, future-token CE loss, optional detached-logit consistency KL, and `predict_mtp()` for emitting next-token logits plus additional MTP token logits. |
| `nanochat/openmythos.py` | recurrent OpenMythos components | Adds `RoutedSharedMoEFFN`, `MoEExpert`, `loop_index_embedding()`, `LTIInjection`, `DepthLoRAAdapter`, `ACTHalting`, and `RecurrentBlock`. These make one transformer block reusable across multiple virtual layers while retaining depth signals, optional per-loop adapters, stable input injection, ACT weighting, and routed/shared expert breadth. |
| `nanochat/openmythos.py` | MoE routing stability | `RoutedSharedMoEFFN` tracks top-K expert assignment counts during training and applies an out-of-graph `router_bias` update that nudges under-used routed experts up and over-used routed experts down after optimizer steps. It also exposes a router z-loss, `alpha * mean(logsumexp(router_logits)^2)`, to discourage unbounded router logits. |
| `nanochat/openmythos.py` | model construction | `GPT.__init__` builds either the original `transformer.h` stack or the opt-in `transformer.prelude`, `transformer.recurrent`, and `transformer.coda` modules. Baseline nanochat behavior is unchanged when `use_looped_transformer=False`. |
| `nanochat/openmythos.py` | forward pass | `GPT.forward(..., loop_iters=None)` routes through Prelude -> recurrent loop -> Coda when enabled. The recurrent path reuses the same block, advances virtual layer indices for rotary/window/value-embedding behavior, supports overriding loop count at call time, and can carry mHC stream state across virtual layers. |
| `nanochat/openmythos.py` | parameter accounting | `num_scaling_params()`, optimizer grouping, FLOP estimates, and initialization account for shared recurrent weights, active MoE parameters, loop-control parameters, mHC coefficient parameters, and Apple MTP sampler parameters instead of pretending each virtual layer has unique dense weights. |
| `nanochat/engine.py` | KV cache helper | Provides the compact `KVCache` utility used by tests and decode paths. Cache allocation asks the config for `num_kv_layers()` so generation has a cache slot for every virtual recurrent layer. MLA cache payloads store compressed latents and rotated RoPE state; DSA can keep a separate index-key cache. |
| `nanochat/gpt.py` | native baseline | Keeps a small nanochat-style GPT baseline separate from the OpenMythos experiment. |

## OpenMythos Technology

The OpenMythos-inspired path changes depth from "many unique transformer
blocks" into "a small number of unique blocks plus recurrent virtual depth":

```text
tokens
  -> nanochat embedding, norm, smear
  -> Prelude: unique nanochat transformer blocks
  -> Recurrent trunk: one shared block reused across virtual depth
       + loop embedding
       + stable LTI injection
       + optional loop-wise LoRA
       + optional ACT halting/weighted state updates
       + optional routed/shared MoE FFN
  -> Coda: unique nanochat transformer blocks
  -> final norm + lm_head
```

This keeps the model small in physical parameters while still giving it a notion
of virtual depth. The recurrent block receives a loop index, can inject the
original embedded input at each pass, can adapt per loop with small LoRA
adapters, and can learn to halt or blend recurrent states with ACT. The MoE
variant widens the recurrent FFN through routed experts plus always-on shared
experts while keeping routing balanced through `router_bias` updates and an
optional router z-loss.

## Configuration Examples

This compact repo exposes the old "flag" surface as `GPTConfig` fields. The
names map directly to the previous CLI flags, for example
`--apple-mtp-num-future-tokens` becomes `apple_mtp_num_future_tokens`.

### Looped Transformer / Mythos Mode

```python
from nanochat.openmythos import GPT, GPTConfig

config = GPTConfig(
    sequence_len=512,
    vocab_size=32768,
    n_layer=6,
    n_head=8,
    n_kv_head=8,
    n_embd=512,
    use_looped_transformer=True,
    loop_prelude_layers=1,
    loop_coda_layers=1,
    loop_iters=4,
    loop_lora_rank=8,
    loop_act_threshold=0.99,
    loop_moe_num_experts=8,
    loop_moe_shared_experts=1,
    loop_moe_experts_per_token=2,
    loop_moe_expert_dim=64,
    loop_moe_bias_update_rate=0.001,
    loop_moe_bias_update_clamp=5.0,
    loop_moe_router_z_loss_weight=0.0001,
)
model = GPT(config)
```

The example uses one prelude block, four recurrent passes through one shared
block, and one coda block. From the KV-cache and loss-scaling perspective that
is six virtual layers; from the parameter-sharing perspective only three
transformer blocks are unique.

### mHC Residual Streams

```python
from nanochat.openmythos import GPT, GPTConfig

config = GPTConfig(
    sequence_len=512,
    vocab_size=32768,
    n_layer=6,
    n_head=8,
    n_kv_head=8,
    n_embd=512,
    mhc_num_streams=4,
    mhc_sinkhorn_iters=20,
    mhc_dynamic_init_scale=0.01,
    mhc_residual_init_strength=2.0,
)
model = GPT(config)
```

`mhc_num_streams=1` is the default and keeps the original residual path. Values
above 1 enable the mHC wrapper around the prelude, recurrent virtual layers, and
coda.

### Apple MTP

```python
config = GPTConfig(
    sequence_len=512,
    vocab_size=32768,
    n_layer=6,
    n_head=8,
    n_kv_head=8,
    n_embd=512,
    apple_mtp_num_future_tokens=4,
    apple_mtp_loss_weight=0.2,
    apple_mtp_consistency_weight=0.05,
    apple_mtp_gated_lora_rank=8,
    apple_mtp_sampler_hidden_mult=1,
)
model = GPT(config)
```

The normal next-token path remains unchanged. Scalar training losses receive
extra teacher-forced future-token losses, and `GPT.predict_mtp(tokens)` returns
the next-token prediction plus the additional future-token logits.

### MLA/DSA Attention

```python
config = GPTConfig(
    sequence_len=512,
    vocab_size=32768,
    n_layer=6,
    n_head=8,
    n_kv_head=8,
    n_embd=512,
    attn_type="mla_dsa",
    mla_q_lora_rank=128,
    mla_kv_lora_rank=192,
    mla_qk_nope_head_dim=64,
    mla_qk_rope_head_dim=32,
    mla_v_head_dim=64,
    dsa_topk=128,
    dsa_index_n_heads=8,
    dsa_index_head_dim=64,
)
model = GPT(config)
```

The portable MLA/DSA path uses manual PyTorch attention and is intended for
experimentation and correctness checks, not production throughput benchmarking.

## Tests

Install the small dependency set:

```bash
python -m pip install torch pytest
```

Run tests:

```bash
python -m pytest -q
```

Target specific feature groups:

```bash
python -m pytest tests/test_looped_transformer.py -q
python -m pytest tests/test_mla_dsa_attention.py -q
python -m pytest tests/test_attention_fallback.py -q
```

The tests cover the unchanged baseline path, looped virtual depth with shared
recurrent weights, router-bias balancing, router z-loss scalar-loss wiring,
Apple MTP auxiliary-loss/prediction wiring, mHC Sinkhorn row/column constraints,
loop-control and mHC parameter accounting, FLOP estimates, KV-cache position
advancement, MLA/DSA forward/backward, MLA/DSA cached decoding, direct DSA
top-K sparsity, and the combined looped + mHC + MLA/DSA + routed/shared MoE +
ACT path.

## License

This repository is MIT licensed. The code is derived from Andrej Karpathy's
MIT-licensed nanochat codebase, and the upstream copyright notice is preserved
in [LICENSE](LICENSE). See [NOTICE.md](NOTICE.md) for attribution and scope.
