# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that a loaded Transformers model really is using bitsandbytes weights.

One implementation for both widths: the INT4 and INT8 verifiers were ~85%
identical, differing only in the module class, the config flag, and INT4's
extra NF4 quantization-type assertion.
"""

from typing import Any

# Per-width specifics. `quant_type_key`/`expected_quant_type` are set only for
# INT4, where the config also has to confirm the NF4 variant.
_BNB_WIDTHS: dict[int, dict[str, Any]] = {
    4: {
        "label": "4-bit",
        "module_attr": "Linear4bit",
        "flag": "load_in_4bit",
        "model_flag": "is_loaded_in_4bit",
        "quant_type_key": "bnb_4bit_quant_type",
        "expected_quant_type": "nf4",
    },
    8: {
        "label": "INT8",
        "module_attr": "Linear8bitLt",
        "flag": "load_in_8bit",
        "model_flag": "is_loaded_in_8bit",
        "quant_type_key": None,
        "expected_quant_type": None,
    },
}


def _config_get(quantization_config: Any, key: str) -> Any:
    """Read `key` from a quantization config that may be a dict or an object."""
    if quantization_config is None:
        return None
    if isinstance(quantization_config, dict):
        return quantization_config.get(key)
    return getattr(quantization_config, key, None)


def verify_bnb_model(model: Any, bits: int) -> dict[str, int | bool | str]:
    """Fail unless ``model`` was actually loaded with bitsandbytes ``bits`` weights."""
    if bits not in _BNB_WIDTHS:
        raise ValueError(f"unsupported bitsandbytes width {bits}; expected one of {sorted(_BNB_WIDTHS)}")
    spec = _BNB_WIDTHS[bits]
    label = spec["label"]

    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError(f"{label} verification failed: bitsandbytes is not installed") from error

    module_class = getattr(bnb.nn, spec["module_attr"])
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, module_class)]
    if not modules:
        raise RuntimeError(
            f"{label} verification failed: the loaded model contains no "
            f"bitsandbytes.nn.{spec['module_attr']} modules"
        )

    hf_quantizer = getattr(model, "hf_quantizer", None)
    configs = [
        getattr(model.config, "quantization_config", None),
        getattr(hf_quantizer, "quantization_config", None),
    ]
    config_enabled = any(_config_get(config, spec["flag"]) is True for config in configs)
    if getattr(model, spec["model_flag"], None) is not True and not config_enabled:
        raise RuntimeError(
            f"{label} verification failed: {spec['module_attr']} layers exist, but neither the "
            f"model flag nor its quantization configuration confirms {spec['flag']}=True"
        )

    result: dict[str, int | bool | str] = {"verified": True, "backend": "bitsandbytes"}

    if spec["quant_type_key"] is not None:
        quant_types = {
            quant_type
            for config in configs
            for quant_type in [_config_get(config, spec["quant_type_key"])]
            if quant_type is not None
        }
        if spec["expected_quant_type"] not in quant_types:
            raise RuntimeError(
                f"{label} verification failed: expected {spec['expected_quant_type'].upper()} "
                f"quantization, found {sorted(quant_types)}"
            )
        result["quant_type"] = spec["expected_quant_type"]

    quantized_parameters = sum(module.weight.numel() for _, module in modules)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if quantized_parameters <= 0 or total_parameters <= 0:
        raise RuntimeError(f"{label} verification failed: invalid model parameter counts")

    result[f"int{bits}_linear_modules"] = len(modules)
    result[f"int{bits}_weight_parameters"] = quantized_parameters
    result["total_parameters"] = total_parameters
    return result


def verify_int4_model(model: Any) -> dict[str, int | bool | str]:
    """Fail unless ``model`` was actually loaded with bitsandbytes NF4 weights."""
    return verify_bnb_model(model, 4)


def verify_int8_model(model: Any) -> dict[str, int | bool | str]:
    """Fail unless ``model`` was actually loaded with bitsandbytes INT8 weights."""
    return verify_bnb_model(model, 8)
