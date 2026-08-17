# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared memory-budget profiles for matrix evaluation."""

from dataclasses import dataclass
from typing import Optional


Budget = tuple[float, str]

STANDARD_KV_BUDGETS: tuple[Budget, ...] = (
    (256.0, "MB"),
    (512.0, "MB"),
    (1.0, "GB"),
    (2.0, "GB"),
)

EXTENDED_KV_BUDGETS: tuple[Budget, ...] = STANDARD_KV_BUDGETS + ((4.0, "GB"),)

SYNTHETIC_INDEPENDENT_BUDGETS: tuple[Budget, ...] = (
    (512.0, "MB"),
    (1.0, "GB"),
    (2.0, "GB"),
    (4.0, "GB"),
    (256.0, "MB"),
)

SYNTHETIC_EXTENDED_BUDGETS: tuple[Budget, ...] = (
    (512.0, "MB"),
    (1024.0, "MB"),
    (2048.0, "MB"),
    (4096.0, "MB"),
    (8192.0, "MB"),
)

SYNTHETIC_ALL_BUDGETS: tuple[Budget, ...] = (
    (256.0, "MB"),
    (512.0, "MB"),
    (1.0, "GB"),
    (2.0, "GB"),
    (4.0, "GB"),
    (8.0, "GB"),
)

RULER_32K_BUDGETS: tuple[Budget, ...] = (
    (0.2, "GB"),
    (0.4, "GB"),
    (0.6, "GB"),
    (0.8, "GB"),
    (1.0, "GB"),
)

RULER_32K_1GB_BUDGETS: tuple[Budget, ...] = ((1.0, "GB"),)

RULER_64K_BUDGETS: tuple[Budget, ...] = (
    (512.0, "MB"),
    (1.0, "GB"),
    (2.0, "GB"),
    (3.0, "GB"),
    (4.0, "GB"),
)


@dataclass(frozen=True)
class MatrixProfile:
    """Dataset checks and budgets that distinguish one matrix from another."""

    dataset: str
    memory_budgets: tuple[Budget, ...]
    baseline_compression_ratio: float
    baseline_press_name: Optional[str] = None
    task_suffix: Optional[str] = None
    exact_tasks: Optional[tuple[str, ...]] = None
    required_fraction: Optional[float] = None
    required_press_name: Optional[str] = None
    required_model_basename: Optional[str] = None
    require_non_quantized: bool = False
    validate_native_rope: bool = False
    yarn_factor: Optional[float] = None
    expected_max_position_embeddings: Optional[int] = None


MATRIX_PROFILES: dict[str, MatrixProfile] = {
    "loft32k": MatrixProfile(
        dataset="loft",
        memory_budgets=STANDARD_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        task_suffix="_32k",
    ),
    "loft128k": MatrixProfile(
        dataset="loft",
        memory_budgets=STANDARD_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        task_suffix="_128k",
    ),
    "loft32k-all": MatrixProfile(
        dataset="loft",
        memory_budgets=EXTENDED_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        task_suffix="_32k",
    ),
    "loft128k-all": MatrixProfile(
        dataset="loft",
        memory_budgets=EXTENDED_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        task_suffix="_128k",
    ),
    "loft32k-qwen35-9b-all-budgets": MatrixProfile(
        dataset="loft",
        memory_budgets=EXTENDED_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-9B",
        require_non_quantized=True,
    ),
    "loft128k-qwen35-9b-all-budgets": MatrixProfile(
        dataset="loft",
        memory_budgets=EXTENDED_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-9B",
        require_non_quantized=True,
    ),
    "loft32k-qwen35-27b-gptq-all-budgets": MatrixProfile(
        dataset="loft",
        memory_budgets=EXTENDED_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-27B-GPTQ-Int4",
    ),
    "loft128k-qwen35-27b-gptq-all-budgets": MatrixProfile(
        dataset="loft",
        memory_budgets=EXTENDED_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-27B-GPTQ-Int4",
    ),
    "ruler32k": MatrixProfile(
        dataset="ruler32k",
        memory_budgets=RULER_32K_BUDGETS,
        baseline_compression_ratio=0.0,
    ),
    "ruler32k-1gb": MatrixProfile(
        dataset="ruler32k",
        memory_budgets=RULER_32K_1GB_BUDGETS,
        baseline_compression_ratio=0.01,
    ),
    "ruler64k": MatrixProfile(
        dataset="ruler64k",
        memory_budgets=RULER_64K_BUDGETS,
        baseline_compression_ratio=0.01,
    ),
    "synthetic-kv-64k": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=STANDARD_KV_BUDGETS,
        baseline_compression_ratio=0.01,
        exact_tasks=("64k",),
    ),
    "synthetic-kv-64k-independent": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_INDEPENDENT_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
    ),
    "synthetic-kv-64k-baseline": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=(),
        baseline_compression_ratio=0.0,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_press_name="no_press",
    ),
    "synthetic-kv-32k-qwen25-1m-baseline": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=(),
        baseline_compression_ratio=0.0,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_press_name="no_press",
        required_model_basename="Qwen2.5-7B-Instruct-1M",
    ),
    "synthetic-kv-32k-qwen35-9b-all-budgets": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-9B",
        require_non_quantized=True,
    ),
    "synthetic-kv-32k-qwen36-27b-fp8-all-budgets": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.6-27B-FP8",
    ),
    "synthetic-kv-64k-qwen35-9b-all-budgets": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-9B",
        require_non_quantized=True,
    ),
    "synthetic-kv-64k-qwen36-27b-fp8-all-budgets": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.6-27B-FP8",
    ),
    "synthetic-kv-32k-qwen35-27b-gptq-all-budgets": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-27B-GPTQ-Int4",
    ),
    "synthetic-kv-64k-qwen35-27b-gptq-all-budgets": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3.5-27B-GPTQ-Int4",
    ),
    "synthetic-kv-64k-qwen25-1m-baseline": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=(),
        baseline_compression_ratio=0.0,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_press_name="no_press",
        required_model_basename="Qwen2.5-7B-Instruct-1M",
    ),
    "synthetic-kv-64k-extended": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_EXTENDED_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="kvzip",
    ),
    "synthetic-kv-32k-native": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=(),
        baseline_compression_ratio=0.0,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_press_name="no_press",
        require_non_quantized=True,
        validate_native_rope=True,
        expected_max_position_embeddings=32768,
    ),
    "synthetic-kv-32k-native-all-budgets": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        require_non_quantized=True,
        validate_native_rope=True,
        expected_max_position_embeddings=32768,
    ),
    "synthetic-kv-32k-native-awq-all-budgets": MatrixProfile(
        dataset="synthetic_kv_32k",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("32k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3-8B-AWQ",
        validate_native_rope=True,
        expected_max_position_embeddings=32768,
    ),
    "synthetic-kv-64k-yarn2": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=(),
        baseline_compression_ratio=0.0,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="no_press",
        require_non_quantized=True,
        yarn_factor=2.0,
        expected_max_position_embeddings=65536,
    ),
    "synthetic-kv-64k-yarn2-all-budgets": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        require_non_quantized=True,
        yarn_factor=2.0,
        expected_max_position_embeddings=65536,
    ),
    "synthetic-kv-64k-yarn2-awq-all-budgets": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=SYNTHETIC_ALL_BUDGETS,
        baseline_compression_ratio=0.01,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="kvzip",
        required_model_basename="Qwen3-8B-AWQ",
        yarn_factor=2.0,
        expected_max_position_embeddings=65536,
    ),
    "synthetic-kv-64k-yarn4": MatrixProfile(
        dataset="synthetic_kv",
        memory_budgets=(),
        baseline_compression_ratio=0.0,
        baseline_press_name="no_press",
        exact_tasks=("64k",),
        required_fraction=1.0,
        required_press_name="no_press",
        require_non_quantized=True,
        yarn_factor=4.0,
        expected_max_position_embeddings=131072,
    ),
}
