from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from kvpress.model_adapter import Qwen35ModelAdapter
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import get_prerope_query_states


def test_qwen35_query_projection_ignores_gate_half():
    config = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
    )
    attention = Qwen3_5Attention(config, layer_idx=0)
    hidden_states = torch.randn(1, 3, config.hidden_size)

    query_states = get_prerope_query_states(attention, hidden_states)

    assert query_states.shape == (1, config.num_attention_heads, 3, config.head_dim)


def test_kvzip_supports_partial_rope_queries():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=32)
            self.head_dim = 8
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.layer_idx = 3

    attention = DummyAttention()
    press = KVzipPress(compression_ratio=0.25)
    press.context_length = 4
    press.start_idx = 1
    press.end_idx = 3
    press._score_layer_position = {3: 0}
    press._model_adapter = Qwen35ModelAdapter()
    press.score_val = torch.zeros(1, 1, 2, 4)

    keys, values = press.score_kvzip(
        attention,
        torch.randn(1, 4, 32),
        torch.randn(1, 2, 8, 8),
        torch.randn(1, 2, 8, 8),
        None,
        {"position_embeddings": (torch.randn(1, 4, 2), torch.randn(1, 4, 2))},
    )

    assert keys.shape[-2] == press.context_length
    assert values.shape[-2] == press.context_length
