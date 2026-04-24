import torch

from nanochat.engine import KVCache
from nanochat.openmythos import CausalSelfAttention, GPT, GPTConfig, RoutedSharedMoEFFN


def build_model(config):
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


def tiny_mla_dsa_config(**overrides):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        attn_type="mla_dsa",
        mla_q_lora_rank=8,
        mla_kv_lora_rank=12,
        mla_qk_nope_head_dim=16,
        mla_qk_rope_head_dim=8,
        mla_v_head_dim=12,
        dsa_topk=2,
        dsa_index_n_heads=2,
        dsa_index_head_dim=10,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_mla_dsa_forward_backward_is_finite():
    config = tiny_mla_dsa_config()
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(idx)
    loss = model(idx, targets)
    loss.backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.transformer.h[0].attn.dsa_topk == 2
    assert len(model.value_embeds) == 0


def test_mla_dsa_kv_cache_uses_mla_and_index_dimensions():
    config = tiny_mla_dsa_config()
    model = build_model(config)
    key_dim, value_dim, index_dim = config.kv_cache_dims()
    cache = KVCache(
        batch_size=1,
        num_heads=config.kv_cache_heads(),
        seq_len=8,
        head_dim=key_dim,
        value_head_dim=value_dim,
        index_head_dim=index_dim,
        num_layers=config.num_kv_layers(),
        device="cpu",
        dtype=torch.float32,
    )
    idx = torch.randint(0, config.vocab_size, (1, 4))
    next_idx = torch.randint(0, config.vocab_size, (1, 1))

    logits = model(idx, kv_cache=cache)
    logits_next = model(next_idx, kv_cache=cache)

    assert logits.shape == (1, 4, config.vocab_size)
    assert logits_next.shape == (1, 1, config.vocab_size)
    assert cache.get_pos() == 5
    assert cache.n_heads == 1
    assert cache.k_cache.shape[-1] == config.mla_kv_rank()
    assert cache.v_cache.shape[-1] == config.mla_rope_dim()
    assert cache.index_cache is not None
    assert cache.index_cache.shape[-1] == config.dsa_index_dim()
    compressed = cache.n_heads * (cache.head_dim + cache.value_head_dim) + cache.index_head_dim
    expanded = config.n_kv_head * (config.mla_qk_head_dim() + config.mla_value_head_dim()) + config.dsa_index_dim()
    assert compressed < expanded


def test_dsa_topk_masks_attention_positions_directly():
    config = tiny_mla_dsa_config(
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16,
        mla_q_lora_rank=0,
        mla_kv_lora_rank=8,
        mla_qk_nope_head_dim=4,
        mla_qk_rope_head_dim=4,
        mla_v_head_dim=8,
        dsa_topk=2,
        dsa_index_n_heads=1,
        dsa_index_head_dim=4,
    )
    attn = CausalSelfAttention(config, layer_idx=0, use_value_embedding=False)
    T = 5
    q = torch.zeros(1, T, 1, 4)
    k = torch.zeros(1, T, 1, 4)
    v = torch.eye(T).view(1, T, 1, T)
    index_scores = torch.arange(T, dtype=torch.float32).view(1, 1, T).expand(1, T, T)

    y = attn._manual_attention(q, k, v, window_size=(-1, 0), query_pos_start=0, key_pos_start=0, index_scores=index_scores)
    probs = y[0, :, 0]

    for t in range(T):
        nonzero = probs[t] > 0
        expected = torch.zeros(T, dtype=torch.bool)
        top = torch.arange(max(0, t - config.dsa_topk + 1), t + 1)
        expected[top] = True
        assert torch.equal(nonzero, expected)


def test_looped_mla_dsa_moe_act_path_runs_end_to_end():
    config = tiny_mla_dsa_config(
        n_layer=5,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=3,
        loop_lora_rank=4,
        loop_act_threshold=0.99,
        loop_moe_num_experts=4,
        loop_moe_shared_experts=1,
        loop_moe_experts_per_token=2,
        loop_moe_expert_dim=16,
        loop_moe_bias_update_rate=0.01,
        mhc_num_streams=3,
        mhc_sinkhorn_iters=8,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(idx)
    loss = model(idx, targets)
    loss.backward()

    recurrent = model.transformer.recurrent
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.mhc is not None
    assert model.num_scaling_params()["mhc"] > 0
    assert recurrent.block.attn.use_mla
    assert isinstance(recurrent.block.mlp, RoutedSharedMoEFFN)
    assert recurrent.block.mlp.router_usage_total.item() > 0
