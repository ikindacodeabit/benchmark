# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks import registry as benchmark_registry

from kvpress import (
    AdaKVPress,
    BlockPress,
    CAMPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    CURPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FastKVzipPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVComposePress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    LUKVPress,
    MergingPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)

DATASET_REGISTRY = benchmark_registry.DATASET_REGISTRY
SCORER_REGISTRY = benchmark_registry.SCORER_REGISTRY

# Presses remain specific to the KVPress inference path.
#
# Factories, not instances: _setup_press configures a press by MUTATING it
# (compression_ratio, threshold, ...), so shared module-level singletons let one
# matrix configuration's settings leak into the next, and presses that
# accumulate internal state carried it across runs. Building on lookup gives
# every configuration a clean press. It also stops every import of this module
# from constructing ~50 presses, which ran KVzipPress.__post_init__'s warning
# before a run had even started.
PRESS_REGISTRY = {
    "adakv_snapkv": lambda: AdaKVPress(SnapKVPress()),
    "block_keydiff": lambda: BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": lambda: ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": lambda: CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": lambda: CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": lambda: CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": lambda: CriticalKVPress(SnapKVPress()),
    "cur": CURPress,
    "duo_attention": DuoAttentionPress,
    "duo_attention_on_the_fly": lambda: DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": lambda: AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "fastkvzip": FastKVzipPress,
    "finch": FinchPress,
    "keydiff": KeyDiffPress,
    "kvcompose": KVComposePress,
    "kvcompose_unstructured": lambda: KVComposePress(structured=False),
    "kvzip": KVzipPress,
    "kvzip_plus": lambda: KVzipPress(kvzip_plus_normalization=True),
    "kvzap_linear": lambda: DMSPress(press=KVzapPress(model_type="linear")),
    "kvzap_mlp": lambda: DMSPress(press=KVzapPress(model_type="mlp")),
    "kvzap_mlp_head": lambda: KVzapPress(model_type="mlp"),
    "kvzap_mlp_layer": lambda: AdaKVPress(KVzapPress(model_type="mlp")),
    "lagkv": LagKVPress,
    "lukv": lambda: LUKVPress(ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1),
    "knorm": KnormPress,
    "observed_attention": ObservedAttentionPress,
    "pyramidkv": PyramidKVPress,
    "qfilter": QFilterPress,
    "random": RandomPress,
    "snap_think": lambda: ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress,
    "streaming_llm": StreamingLLMPress,
    "think": ThinKPress,
    "tova": TOVAPress,
    "compactor": CompactorPress,
    "adakv_compactor": lambda: AdaKVPress(CompactorPress()),
    "no_press": lambda: None,
    "cam_streaming_llm": lambda: CAMPress(base_press=StreamingLLMPress()),
    "cam_knorm": lambda: CAMPress(base_press=KnormPress()),
    "cam_adakv_snapkv": lambda: CAMPress(base_press=AdaKVPress(SnapKVPress())),
    "cam_tova": lambda: CAMPress(base_press=TOVAPress()),
    "decoding_knorm": lambda: DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": lambda: DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": lambda: DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": lambda: DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": lambda: DecodingPress(
        base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))
    ),
    "decoding_adakv_snapkv": lambda: DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": lambda: DecodingPress(base_press=KeyDiffPress()),
    # MergingPress: merge-on-evict during prefill (values-only merge preserves RoPE keys)
    "merging_knorm": lambda: MergingPress(KnormPress()),
    "merging_snapkv": lambda: MergingPress(SnapKVPress()),
    "merging_expected_attention": lambda: MergingPress(ExpectedAttentionPress(epsilon=1e-2)),
    "merging_kvzap_mlp": lambda: MergingPress(KVzapPress(model_type="mlp")),
}
