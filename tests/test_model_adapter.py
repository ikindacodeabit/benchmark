from types import SimpleNamespace

import torch
from torch import nn

from kvpress.model_adapter import (
    Qwen35ModelAdapter,
    StandardModelAdapter,
    get_model_adapter,
)


def _standard_model():
    layers = [SimpleNamespace(self_attn=object()), SimpleNamespace(self_attn=object())]
    language_model = SimpleNamespace(layers=layers)
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=512,
            head_dim=64,
        ),
        model=language_model,
        dtype=torch.float16,
    )


def _qwen35_model():
    layers = [
        SimpleNamespace(self_attn="full-0"),
        SimpleNamespace(linear_attn="linear-1"),
        SimpleNamespace(self_attn="full-2"),
    ]
    text_config = SimpleNamespace(
        layer_types=["full_attention", "linear_attention", "full_attention"],
        num_key_value_heads=2,
        head_dim=64,
    )
    return SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5", text_config=text_config),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
        dtype=torch.float16,
    )


def test_qwen3_uses_standard_adapter():
    model = _standard_model()
    adapter = get_model_adapter(model)
    assert isinstance(adapter, StandardModelAdapter)
    assert [idx for idx, _ in adapter.iter_kv_attention_layers(model)] == [0, 1]
    assert adapter.kv_bytes_per_token(model) == 2 * 2 * 2 * 64 * 2


def test_qwen35_only_exposes_full_attention_layers():
    model = _qwen35_model()
    adapter = get_model_adapter(model)
    assert isinstance(adapter, Qwen35ModelAdapter)
    assert list(adapter.iter_kv_attention_layers(model)) == [(0, "full-0"), (2, "full-2")]
    assert adapter.kv_bytes_per_token(model) == 2 * 2 * 2 * 64 * 2


def test_qwen35_cache_snapshot_restores_attention_and_recurrent_state():
    adapter = Qwen35ModelAdapter()
    cache = SimpleNamespace(
        key_cache=[torch.ones(1), None],
        value_cache=[torch.ones(1) * 2, None],
        recurrent_states=[torch.ones(2), torch.ones(3)],
        conv_states=[torch.ones(4), torch.ones(5)],
        layer_types=["full_attention", "linear_attention"],
        transformer_layers=[0],
        last_linear_layer=1,
    )
    snapshot = adapter.snapshot_cache_state(cache)
    cache.key_cache[0] = torch.zeros(1)
    cache.value_cache[0] = torch.zeros(1)
    cache.recurrent_states[1].zero_()
    cache.conv_states[1].zero_()
    adapter.restore_cache_state(cache, snapshot)
    assert torch.equal(cache.key_cache[0], torch.ones(1))
    assert torch.equal(cache.value_cache[0], torch.ones(1) * 2)
    assert torch.equal(cache.recurrent_states[1], torch.ones(3))
    assert torch.equal(cache.conv_states[1], torch.ones(5))

    # A restored state can itself be mutated during decode without corrupting
    # the master snapshot needed for the next question.
    cache.recurrent_states[1].zero_()
    cache.conv_states[1].zero_()
    adapter.restore_cache_state(cache, snapshot)
    assert torch.equal(cache.recurrent_states[1], torch.ones(3))
    assert torch.equal(cache.conv_states[1], torch.ones(5))


def test_qwen35_text_config_uses_hybrid_adapter():
    model = _qwen35_model()
    model.config = model.config.text_config
    model.config.model_type = "qwen3_5_text"
    model.model = SimpleNamespace(layers=model.model.language_model.layers)

    adapter = get_model_adapter(model)

    assert isinstance(adapter, Qwen35ModelAdapter)
    assert adapter.get_text_config(model) is model.config
    assert adapter.get_language_model(model) is model.model


def test_standard_adapter_leaves_cached_continuation_unchanged():
    adapter = StandardModelAdapter()
    model = _standard_model()
    original_attention = model.model.layers[0].self_attn

    with adapter.cached_continuation(model):
        assert model.model.layers[0].self_attn is original_attention


def test_qwen35_initial_prefill_remains_one_multitoken_call():
    class FakeLinearAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.received_lengths = []

        def forward(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            del cache_params, cache_position, attention_mask
            self.received_lengths.append(hidden_states.shape[1])
            return hidden_states

    linear_attention = FakeLinearAttention()
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_text", layer_types=["linear_attention"]),
        model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=linear_attention)]),
    )
    cache = SimpleNamespace(has_previous_state=False)

    with Qwen35ModelAdapter().cached_continuation(model):
        linear_attention(
            torch.zeros(1, 3, 1),
            cache_params=cache,
            cache_position=torch.arange(3),
        )

    assert linear_attention.received_lengths == [3]


def test_qwen35_cached_continuation_matches_full_sequence():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 0, 0],
            "mrope_interleaved": True,
        },
    )
    model = Qwen3_5ForCausalLM(config).eval()
    adapter = Qwen35ModelAdapter()
    context_ids = torch.tensor([[5, 7, 11, 13, 17]])
    question_ids = torch.tensor([[19, 23, 29]])

    with torch.inference_mode():
        full_logits = model(
            torch.cat((context_ids, question_ids), dim=1),
            use_cache=False,
        ).logits[:, -question_ids.shape[1] :]

        cache = adapter.create_cache(model)
        model(context_ids, past_key_values=cache, use_cache=True)
        with adapter.cached_continuation(model):
            cached_logits = model(
                question_ids,
                past_key_values=cache,
                use_cache=True,
            ).logits

    assert torch.allclose(full_logits, cached_logits, atol=2e-4, rtol=2e-4)
