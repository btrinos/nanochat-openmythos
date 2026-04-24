import torch
import pytest

from nanochat.engine import KVCache
from nanochat.openmythos import (
    ACTHalting,
    DepthLoRAAdapter,
    GPT,
    GPTConfig,
    LTIInjection,
    MHCHyperConnections,
    RoutedSharedMoEFFN,
)


def tiny_config(**overrides):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def build_model(config):
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


def test_default_gpt_forward_still_works():
    model = build_model(tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))

    logits = model(idx)
    loss = model(idx, targets)

    assert logits.shape == (2, 8, model.config.vocab_size)
    assert torch.isfinite(loss)
    assert model.effective_n_layer == model.config.n_layer


def test_looped_transformer_uses_virtual_depth_with_shared_block():
    config = tiny_config(
        n_layer=5,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=3,
        loop_lora_rank=4,
        loop_act_threshold=0.99,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(idx)
    counts = model.num_scaling_params()
    recurrent = model.transformer.recurrent

    assert logits.shape == (2, 8, config.vocab_size)
    assert config.num_kv_layers() == 5
    assert model.effective_n_layer == 5
    assert len(model._unique_blocks()) == 3  # prelude + shared recurrent + coda
    assert isinstance(recurrent.injection, LTIInjection)
    assert isinstance(recurrent.lora, DepthLoRAAdapter)
    assert isinstance(recurrent.act, ACTHalting)
    assert counts["loop_controls"] > 0
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert model.estimate_flops() > 0


def test_looped_transformer_can_use_routed_shared_moe_ffn():
    config = tiny_config(
        n_layer=5,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=3,
        loop_moe_num_experts=4,
        loop_moe_shared_experts=1,
        loop_moe_experts_per_token=2,
        loop_moe_expert_dim=16,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(idx)
    mlp = model.transformer.recurrent.block.mlp
    param_names = {name for name, _ in mlp.named_parameters()}

    assert logits.shape == (2, 8, config.vocab_size)
    assert isinstance(mlp, RoutedSharedMoEFFN)
    assert mlp.topk == 2
    assert len(mlp.routed_experts) == 4
    assert len(mlp.shared_experts) == 1
    assert "router_bias" not in param_names
    assert "router_bias" in dict(mlp.named_buffers())
    assert mlp.active_parameters() < sum(p.numel() for p in mlp.parameters())


def test_mhc_forward_backward_uses_sinkhorn_streams():
    config = tiny_config(
        mhc_num_streams=3,
        mhc_sinkhorn_iters=20,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(idx)
    loss = model(idx, targets)
    loss.backward()
    counts = model.num_scaling_params()
    probe_state = model.mhc.initial_state(torch.randn(2, 4, config.n_embd))
    _, h_post, h_res = model.mhc.mappings(probe_state, 0)

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert isinstance(model.mhc, MHCHyperConnections)
    assert counts["mhc"] > 0
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert torch.all(h_post >= 0)
    assert torch.allclose(h_res.sum(dim=-1), torch.ones_like(h_res.sum(dim=-1)), atol=5e-4)
    assert torch.allclose(h_res.sum(dim=-2), torch.ones_like(h_res.sum(dim=-2)), atol=5e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="mHC compile regression is MPS-specific")
def test_mhc_compiled_training_step_stays_finite_on_mps():
    config = tiny_config(
        mhc_num_streams=3,
        mhc_sinkhorn_iters=8,
    )
    model = build_model(config).to("mps")
    compiled = torch.compile(model, dynamic=False)
    optimizer = model.setup_optimizer(
        unembedding_lr=1e-4,
        embedding_lr=1e-4,
        matrix_lr=1e-4,
        scalar_lr=1e-4,
        weight_decay=0.0,
    )

    for _ in range(3):
        idx = torch.randint(0, config.vocab_size, (2, 8), device="mps")
        targets = torch.randint(0, config.vocab_size, (2, 8), device="mps")
        loss = compiled(idx, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())


def test_apple_mtp_adds_future_loss_and_predicts_multiple_tokens():
    config = tiny_config(
        apple_mtp_num_future_tokens=2,
        apple_mtp_loss_weight=0.0,
        apple_mtp_consistency_weight=0.1,
        apple_mtp_gated_lora_rank=4,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    base_loss = model(idx, targets)
    model.config.apple_mtp_loss_weight = 1.0
    mtp_loss = model(idx, targets)
    mtp_loss.backward()
    counts = model.num_scaling_params()
    token_loss = model(idx, targets, loss_reduction="none")
    prediction = model.predict_mtp(idx[0].tolist(), num_future_tokens=2)

    assert mtp_loss > base_loss
    assert token_loss.shape == (idx.numel(),)
    assert model.apple_mtp is not None
    assert counts["apple_mtp"] > 0
    assert model.apple_mtp.fc1.weight.grad is not None
    assert model.apple_mtp.gated_lora.B.grad is not None
    assert prediction["ntp_logits"].shape == (1, 1, config.vocab_size)
    assert prediction["mtp_logits"].shape == (1, 2, config.vocab_size)
    assert prediction["token_ids"].shape == (1, 3)


def test_router_bias_update_balances_underused_experts():
    config = tiny_config(
        use_looped_transformer=True,
        loop_moe_num_experts=4,
        loop_moe_experts_per_token=2,
        loop_moe_expert_dim=16,
        loop_moe_bias_update_rate=0.01,
        loop_moe_bias_update_clamp=0.02,
    )
    mlp = RoutedSharedMoEFFN(config)
    mlp.router_usage.copy_(torch.tensor([8.0, 4.0, 0.0, 0.0]))
    mlp.router_usage_total.fill_(12.0)

    mlp.update_router_bias()

    assert torch.allclose(mlp.router_bias, torch.tensor([-0.01, -0.01, 0.01, 0.01]))
    assert mlp.router_usage.sum() == 0
    assert mlp.router_usage_total == 0

    mlp.router_usage.copy_(torch.tensor([0.0, 0.0, 12.0, 12.0]))
    mlp.router_usage_total.fill_(24.0)
    mlp.update_router_bias()
    mlp.router_usage.copy_(torch.tensor([0.0, 0.0, 12.0, 12.0]))
    mlp.router_usage_total.fill_(24.0)
    mlp.update_router_bias()

    assert torch.all(mlp.router_bias.abs() <= 0.02)


def test_router_usage_accumulates_during_training_forward():
    config = tiny_config(
        use_looped_transformer=True,
        loop_moe_num_experts=4,
        loop_moe_experts_per_token=2,
        loop_moe_expert_dim=16,
    )
    mlp = RoutedSharedMoEFFN(config)
    mlp.train()

    x = torch.randn(2, 5, config.n_embd)
    _ = mlp(x)

    assert mlp.router_usage_total.item() == 2 * 5 * mlp.topk
    assert mlp.router_usage.sum().item() == mlp.router_usage_total.item()


def test_router_z_loss_is_added_only_to_scalar_training_loss():
    config = tiny_config(
        n_layer=5,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=3,
        loop_moe_num_experts=4,
        loop_moe_experts_per_token=2,
        loop_moe_expert_dim=16,
        loop_moe_router_z_loss_weight=0.0,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    base_loss = model(idx, targets)
    model.config.loop_moe_router_z_loss_weight = 1.0
    z_loss = model(idx, targets)
    token_loss = model(idx, targets, loss_reduction="none")
    model.eval()
    eval_loss_with_weight = model(idx, targets)
    model.config.loop_moe_router_z_loss_weight = 0.0
    eval_loss_without_weight = model(idx, targets)

    assert z_loss > base_loss
    assert token_loss.shape == (idx.numel(),)
    assert torch.allclose(eval_loss_with_weight, eval_loss_without_weight)


def test_model_update_router_biases_reaches_recurrent_moe():
    config = tiny_config(
        n_layer=5,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=3,
        loop_moe_num_experts=4,
        loop_moe_experts_per_token=2,
        loop_moe_expert_dim=16,
        loop_moe_bias_update_rate=0.01,
    )
    model = build_model(config)
    mlp = model.transformer.recurrent.block.mlp
    mlp.router_usage.copy_(torch.tensor([8.0, 4.0, 0.0, 0.0]))
    mlp.router_usage_total.fill_(12.0)

    model.update_router_biases()

    assert torch.allclose(mlp.router_bias, torch.tensor([-0.01, -0.01, 0.01, 0.01]))


def test_act_halting_short_circuits_without_kv_cache():
    config = tiny_config(
        n_layer=6,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=4,
        loop_act_threshold=0.1,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    recurrent_calls = []

    handle = model.transformer.recurrent.block.register_forward_hook(
        lambda _module, _inputs, _output: recurrent_calls.append(1)
    )
    try:
        logits = model(idx)
    finally:
        handle.remove()

    assert logits.shape == (2, 8, config.vocab_size)
    assert len(recurrent_calls) == 1


def test_looped_transformer_kv_cache_advances_after_last_virtual_layer():
    config = tiny_config(
        n_layer=5,
        use_looped_transformer=True,
        loop_prelude_layers=1,
        loop_coda_layers=1,
        loop_iters=3,
    )
    model = build_model(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=8,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.num_kv_layers(),
        device="cpu",
        dtype=torch.float32,
    )

    logits = model(idx, kv_cache=cache)

    assert logits.shape == (1, 4, config.vocab_size)
    assert cache.get_pos() == idx.size(1)
