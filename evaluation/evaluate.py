# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Optional torchao/kernels are not required by KVPress evaluation.  On this
# cluster the installed torchao wheel contains extensions for a different
# build, so loading them produces noisy warnings before Transformers falls
# back to ordinary PyTorch kernels.  Slurm launchers enable this guard; the
# environment check preserves the previous behavior for other callers.
if os.getenv("KVPRESS_DISABLE_OPTIONAL_KERNEL_WARNINGS") == "1":
    os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
    os.environ.setdefault("USE_HUB_KERNELS", "NO")
    logging.getLogger("torchao").setLevel(logging.ERROR)
    logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.loaders import load_benchmark_dataset
from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from benchmarks.results import score_prediction_frame
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, FineGrainedFP8Config, Pipeline, pipeline
from verify_int8_model import verify_int8_model
from verify_int4_model import verify_int4_model
from kvpress.model_adapter import get_model_adapter
from kvpress.pipeline import KVPressTextGenerationPipeline

from kvpress import (
    ComposedPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    ThinKPress,
)

logger = logging.getLogger(__name__)


def _validate_finegrained_fp8_hardware(model_config: Any, model_kwargs: Dict[str, Any]) -> None:
    """Fail early when a pre-quantized FP8 checkpoint is placed on an old GPU.

    Qwen's fine-grained FP8 checkpoints require NVIDIA compute capability
    8.9 or newer for the FP8 matrix-multiply path.  In particular, DGX A100
    nodes are SM80.  Letting Transformers load the checkpoint on SM80 first
    causes a long weight materialization followed by the opaque
    ``grouped_mm`` Float8_e4m3fn error.  This guard affects only FP8 models;
    all ordinary and AWQ/GPTQ model paths are unchanged.
    """
    configured_quantization = model_kwargs.get("quantization_config")
    checkpoint_quantization = getattr(model_config, "quantization_config", None)
    quantization_sources = (configured_quantization, checkpoint_quantization)
    quantization_methods = []
    for source in quantization_sources:
        if source is None:
            continue
        method = getattr(source, "quant_method", None)
        if method is None and isinstance(source, dict):
            method = source.get("quant_method")
        quantization_methods.append(str(method or "").lower())
    is_fp8 = any("fp8" in method for method in quantization_methods)
    if not is_fp8:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This checkpoint uses fine-grained FP8 weights and requires a CUDA GPU. "
            "Use an NVIDIA GPU with compute capability >= 8.9 (for example L40/H100), "
            "or use a BF16/AWQ/GPTQ checkpoint."
        )

    capability = torch.cuda.get_device_capability()
    if capability >= (8, 9):
        return

    raise RuntimeError(
        "Unsupported hardware for fine-grained FP8 checkpoint: detected CUDA compute "
        f"capability {capability[0]}.{capability[1]} (SM{capability[0]}{capability[1]}). "
        "This FP8 path requires compute capability >= 8.9. DGX A100/SM80 nodes "
        "cannot run Qwen3 FP8 grouped_mm; submit the L40 job instead, or use a "
        "BF16/AWQ/GPTQ checkpoint. This is a model/backend hardware limitation, "
        "not a KVPress compression error."
    )


def _reference_for_log(df: pd.DataFrame, index: Any) -> Any:
    """Return a reference value without assuming one dataset schema."""
    reference_column = next(
        (column for column in ("answer", "answers") if column in df.columns),
        None,
    )
    return df.loc[index, reference_column] if reference_column else "<unavailable>"


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    # data_dir: Optional[str] = None
    data_dir: Optional[Union[str, List[str]]] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None
    head_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None
    memory_budget: Optional[float] = None
    memory_budget_unit: str = "GB"
    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    query_aware: bool = False
    needle_depth: Optional[int] = None
    synthetic_kv_metadata_override: bool = False

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "../benchmark_artifacts/results/default"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    # Quantization
    fp8: bool = False
    int8: bool = False
    int4: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        if self.memory_budget is not None:
            assert self.memory_budget > 0, f"memory_budget must be positive, got {self.memory_budget}"
            self.memory_budget_unit = self.memory_budget_unit.upper()
            assert self.memory_budget_unit in {
                "MB",
                "GB",
            }, f"memory_budget_unit must be MB or GB, got {self.memory_budget_unit}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        enabled_quantization_modes = sum([self.fp8, self.int8, self.int4])
        assert enabled_quantization_modes <= 1, "Only one of fp8, int8, or int4 may be enabled"

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self, output_dir: Path, data_dir: Optional[str] = None) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        if data_dir is None:
            data_dir = self.data_dir

        # Convert list to string for directory name
        if isinstance(data_dir, list):
            data_dir_str = "__".join(data_dir)
        else:
            data_dir_str = str(data_dir) if data_dir else ""
        # Build directory name components
        components = [
            self.dataset,
            data_dir_str,
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        if self.threshold is not None:
            components[-1] = f"{self.threshold:.2f}"
        elif self.head_compression_ratio is not None:
            components[-1] = f"{self.head_compression_ratio:.2f}"
        if self.memory_budget is not None:
            components.append(f"memory_budget{self.memory_budget:g}{self.memory_budget_unit}")
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.fp8:
            components.append("fp8")
        if self.int8:
            components.append("int8")
        if self.int4:
            components.append("int4_nf4")
        if self.seed != 42:
            components.append(f"seed{self.seed}")
        if self.query_aware:
            components.append("query_aware")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")
        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        dir_name = f"new_{dir_name}"
        # Deterministic name so interrupted matrix runs can resume. Creating the
        # directory is the caller's job: this is also called from the matrix
        # pre-scan, which must not litter the tree with empty directories.
        return output_dir / dir_name

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        config_dict = asdict(self)
        if self.threshold is not None or self.head_compression_ratio is not None:
            config_dict.pop("compression_ratio", None)
        if self.threshold is None:
            config_dict.pop("threshold", None)
        if self.head_compression_ratio is None:
            config_dict.pop("head_compression_ratio", None)
        with open(str(config_filename), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path, _seen: Optional[set[Path]] = None) -> dict:
    """Load a YAML config, including an optional relative ``extends`` file."""
    config_path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"Circular YAML config reference detected at {config_path}")
    seen.add(config_path)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # Silently proceeding on dataclass defaults would run the wrong model,
        # press, and dataset on a cluster allocation. Fail loudly instead.
        raise FileNotFoundError(f"Config file not found at {config_path}")

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    reference = config.pop("extends", None)
    if reference is None:
        return config
    if not isinstance(reference, str):
        raise ValueError(f"The 'extends' value in {config_path} must be a path string")

    reference_path = Path(reference).expanduser()
    if not reference_path.is_absolute():
        reference_path = config_path.parent / reference_path
    merged_config = _load_yaml_config(reference_path, seen)
    merged_config.update(config)
    return merged_config


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

        # Surface progress emitted inside the pipeline and press implementations.
        # These loggers are outside this module's namespace, so attaching the
        # same handler here keeps long prefilling/compression phases observable.
        kvpress_logger = logging.getLogger("kvpress")
        kvpress_logger.addHandler(handler)
        kvpress_logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            assert (
                self.config.head_compression_ratio is not None
            ), "head_compression_ratio must be set for DuoAttentionPress"
            press.head_compression_ratio = self.config.head_compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {press.head_compression_ratio}")
        elif isinstance(press, DMSPress):
            assert self.config.threshold is not None, "threshold must be set for DMSPress"
            press.threshold = self.config.threshold
            logger.info(f"Set DMSPress threshold to {press.threshold}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, DecodingPress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self, task_data_dir: Optional[str] = None):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = task_data_dir
        if data_dir is None and isinstance(self.config.data_dir, str):
            data_dir = self.config.data_dir
        try:
            df = load_benchmark_dataset(
                dataset_name=dataset_name,
                task=data_dir,
                dataset_registry=DATASET_REGISTRY,
                synthetic_metadata_override=(self.config.synthetic_kv_metadata_override),
            )
        except Exception:
            logger.exception("Failed to load dataset=%s task=%r", dataset_name, data_dir)
            raise
        fraction = self.config.fraction
        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) " f"from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df,
                self.pipeline.tokenizer,
                self.config.max_context_length,
                self.config.needle_depth,
            )

        if isinstance(self.press, FinchPress):
            if not self.config.query_aware:
                logger.error("FinchPress requires 'query_aware' to be set to True.")
                raise ValueError("FinchPress requires query_aware to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.query_aware:
            logger.info("Query-aware compression: including question in context for compression.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        # Keep the configured mapping reusable across matrix runs.  Loader-only
        # controls are removed from this copy before calling Transformers.
        model_kwargs = dict(self.config.model_kwargs or {})
        gptq_backend = model_kwargs.pop("gptq_backend", None)

        if self.config.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
            logger.info("FP8 quantization enabled.")

        if self.config.int8:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            logger.info("INT8 bitsandbytes quantization enabled.")

        if self.config.int4:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4-bit bitsandbytes NF4 quantization enabled.")

        if isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _validate_finegrained_fp8_hardware(model_config, model_kwargs)
        if model_config.model_type == "qwen3_5":
            quantization_config = getattr(model_config, "quantization_config", None)
            is_prequantized = quantization_config is not None
            quantization_dict = (
                quantization_config.to_dict()
                if hasattr(quantization_config, "to_dict")
                else dict(quantization_config or {})
            )
            is_dynamic_gptq = (
                str(quantization_dict.get("quant_method", "")).lower() == "gptq"
                and bool(quantization_dict.get("dynamic"))
            )

            if is_dynamic_gptq:
                # Qwen3.5's selective GPTQ checkpoint uses the newer hybrid
                # model/GPTQ integration.  Transformers 5.2 can materialize
                # the model but produces invalid generations; fail before any
                # benchmark answers are written instead of silently scoring
                # corrupted output.
                import transformers
                from packaging.version import Version

                if Version(transformers.__version__) < Version("5.3.0"):
                    raise RuntimeError(
                        "Qwen3.5 dynamic GPTQ requires Transformers >= 5.3.0. "
                        f"Found {transformers.__version__}; use the isolated "
                        "kvpress-tf515 environment or a newer Transformers runtime."
                    )
                # The official dense Qwen3.5 GPTQ checkpoint stores only its
                # MLP projections as qweight/qzeros/scales/g_idx and excludes
                # every attention/DeltaNet projection through `dynamic`.
                # Transformers 5.2.0's Optimum bridge does not carry that
                # GPTQModel-specific field into layer conversion, but it does
                # honor modules_in_block_to_quantize. Translate the equivalent
                # rule before loading the text-only model.
                dynamic = quantization_dict["dynamic"]
                if "-:.*attn.*" not in dynamic:
                    raise NotImplementedError(
                        "This Qwen3.5 selective GPTQ layout is not supported: "
                        "expected the checkpoint to exclude all attention modules"
                    )
                quantization_dict["modules_in_block_to_quantize"] = [
                    ["mlp.gate_proj", "mlp.up_proj"],
                    ["mlp.down_proj"],
                ]
                # The CUDA Triton kernel is available on the benchmark GPUs;
                # selecting it avoids probing the unavailable CPU/HF kernel.
                quantization_dict["backend"] = (gptq_backend or "triton").lower()
                # The checkpoint's quantization_config has no `checkpoint_format`
                # field. GPTQModel's from_quant_config() treats a missing field
                # as "compat: default to gptq(v1) when loading models" and, for
                # kernels that require v2 (TritonV2QuantLinear does), silently
                # adds +1 to every packed zero-point to correct a v1-era
                # off-by-one convention. This checkpoint was not produced by
                # that legacy pipeline, so nothing needs correcting; treat it
                # as already gptq_v2 to skip that transform.
                quantization_dict["checkpoint_format"] = "gptq_v2"
                model_config.quantization_config = quantization_dict
                logger.info(
                    "Loading selective Qwen3.5 GPTQ checkpoint with only MLP projections quantized "
                    "(backend=%s)",
                    quantization_dict["backend"],
                )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            load_kwargs = dict(model_kwargs)

            # Text-only Qwen3.5/3.6 checkpoints are published with the
            # conditional-generation (vision + language) configuration, but
            # this benchmark never supplies images or videos.  Loading the
            # conditional class needlessly materializes the vision tower and
            # can also leave the text FP8 scale tensors unmapped.  Use the
            # Transformers text class with the embedded text configuration;
            # its checkpoint conversion mapping handles the
            # ``model.language_model.*`` prefixes.
            #
            # Dynamic GPTQ Qwen3.5 checkpoints must also use the text class.
            # Loading them through Qwen3_5ForConditionalGeneration causes the
            # GPTQ projection names to be reported as MISSING/UNEXPECTED and
            # produces invalid generations.  The nested text config preserves
            # the checkpoint's model.language_model.* mapping.
            use_text_only_qwen = hasattr(model_config, "text_config")
            if use_text_only_qwen:
                from transformers import Qwen3_5ForCausalLM

                text_config = model_config.text_config
                # The parent config carries the official FP8 metadata on
                # some checkpoints.  Preserve it when the nested text config
                # does not expose its own copy; this is metadata reuse, not a
                # new quantization operation.
                text_quantization_config = getattr(text_config, "quantization_config", None)
                text_quantization_dict = (
                    text_quantization_config.to_dict()
                    if hasattr(text_quantization_config, "to_dict")
                    else dict(text_quantization_config or quantization_dict)
                )
                if quantization_config is not None:
                    # Qwen3.6 stores linear-attention in_proj_a/in_proj_b in
                    # BF16, while in_proj_qkv/in_proj_z have FP8 weights plus
                    # weight_scale_inv tensors.  The generic FP8 replacer
                    # otherwise converts a/b and creates scale parameters
                    # that do not exist in the checkpoint (LOAD REPORT:
                    # MISSING).  Match the checkpoint layout by leaving only
                    # these two projections in their stored BF16 form.
                    modules_to_not_convert = list(
                        text_quantization_dict.get("modules_to_not_convert", []) or []
                    )
                    for module_suffix in (
                        "linear_attn.in_proj_a",
                        "linear_attn.in_proj_b",
                    ):
                        if module_suffix not in modules_to_not_convert:
                            modules_to_not_convert.append(module_suffix)
                    text_quantization_dict["modules_to_not_convert"] = modules_to_not_convert
                    text_config.quantization_config = text_quantization_dict
                load_kwargs["config"] = text_config
                if device == "auto":
                    load_kwargs["device_map"] = "auto"
                elif is_prequantized:
                    load_kwargs["device_map"] = {"": device}
                model = Qwen3_5ForCausalLM.from_pretrained(model_name, **load_kwargs)
                logger.info("Loaded Qwen3.5 text-only model; vision inputs/tower are disabled.")
            else:
                # Preserve the existing selective-GPTQ path and its checkpoint
                # parameter layout.  It still receives text tokens only.
                from transformers import Qwen3_5ForConditionalGeneration

                load_kwargs["config"] = model_config
                if device == "auto":
                    load_kwargs["device_map"] = "auto"
                elif is_prequantized:
                    load_kwargs["device_map"] = {"": device}
                model = Qwen3_5ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
                logger.info("Loaded Qwen3.5 conditional-generation model in text-only input mode.")
            pipeline_kwargs = {"model": model, "tokenizer": tokenizer}
            if device != "auto" and not is_prequantized:
                pipeline_kwargs["device"] = device
            self.pipeline = KVPressTextGenerationPipeline(**pipeline_kwargs)
        else:
            pipeline_kwargs = {
                "model": model_name,
                "model_kwargs": model_kwargs,
                "trust_remote_code": True,
            }
            if device == "auto":
                pipeline_kwargs["device_map"] = "auto"
            else:
                pipeline_kwargs["device"] = device
            self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)

        if self.config.int8:
            int8_verification = verify_int8_model(self.pipeline.model)
            logger.info("INT8 model verification passed: %s", int8_verification)
        if self.config.int4:
            int4_verification = verify_int4_model(self.pipeline.model)
            logger.info("4-bit NF4 model verification passed: %s", int4_verification)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]
        total_questions = len(self.df)
        completed_questions = 0

        if self.config.memory_budget is None:
            budget_label = f"reference-ratio-{self.config.compression_ratio:.4f}"
        else:
            budget_label = f"{self.config.memory_budget:g}-{self.config.memory_budget_unit}"

        def log_question_completed(index: Any, predicted_answer: Any) -> None:
            """Emit one immediately flushed progress record per completed question."""
            nonlocal completed_questions
            completed_questions += 1
            prediction_preview = " ".join(str(predicted_answer).split())
            if len(prediction_preview) > 160:
                prediction_preview = f"{prediction_preview[:157]}..."
            reference = _reference_for_log(self.df, index)
            logger.info(
                "Question completed %d/%d (%.1f%%) | task=%s | budget=%s | " "row=%s | reference=%r | prediction=%r",
                completed_questions,
                total_questions,
                100.0 * completed_questions / total_questions,
                self.config.data_dir,
                budget_label,
                index,
                reference,
                prediction_preview,
            )

        if isinstance(self.press, DecodingPress):
            logger.info("DecodingPress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]  # type: ignore[union-attr]
                log_question_completed(index, output["answer"])  # type: ignore[index, union-attr]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                questions = df_group["question"].to_list()
                # Use max_new_tokens from config, or fallback to dataset's default for the task
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]
                group_indices = list(df_group.index)

                def log_group_question_completed(
                    question_number: int, question_total: int, predicted_answer: str
                ) -> None:
                    del question_total  # The outer counter also covers datasets with multiple contexts.
                    log_question_completed(group_indices[question_number - 1], predicted_answer)

                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                    memory_budget=self.config.memory_budget,
                    memory_budget_unit=self.config.memory_budget_unit,
                    question_progress_callback=log_group_question_completed,
                )
                self.df.loc[df_group.index, "predicted_answer"] = output["answers"]  # type: ignore[union-attr]
                budget_stats = getattr(self.pipeline, "last_memory_budget_stats", None)
                if budget_stats is not None:
                    for column, value in budget_stats.items():
                        self.df.loc[df_group.index, column] = value
                else:
                    self.df.loc[df_group.index, "compression_ratio"] = (
                        self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                    )  # type: ignore[union-attr, attr-defined]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        # drop() keeps the remaining columns in their original, stable order; a
        # set difference reordered them per process (hash randomization).
        frame = self.df.drop(columns=["context"], errors="ignore")  # type: ignore[union-attr]
        frame.to_csv(str(save_filename), index=False)
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = score_prediction_frame(dataset_name, self.df)

        if "compression_ratio" in self.df.columns:
            # A shared context may have multiple questions. Count that context once in the summary.
            context_stats = self.df.drop_duplicates(subset=["context"])
            metrics["average_compression_ratio"] = float(context_stats["compression_ratio"].mean())
            metrics["average_original_context_tokens"] = float(context_stats["context_tokens"].mean())
            metrics["average_retained_context_tokens"] = float(context_stats["retained_context_tokens"].mean())
            metrics["kv_memory_per_token_kb"] = float(context_stats["kv_memory_per_token_kb"].iloc[0])
            metrics["average_retained_kv_memory_mb"] = float(context_stats["retained_kv_memory_mb"].mean())
            metrics["average_retained_kv_memory_gb"] = float(context_stats["retained_kv_memory_gb"].mean())
            metrics["average_uncompressed_kv_memory_mb"] = float(context_stats["uncompressed_kv_memory_mb"].mean())
            metrics["average_uncompressed_kv_memory_gb"] = float(context_stats["uncompressed_kv_memory_gb"].mean())

            if self.config.memory_budget is not None:
                metrics["memory_budget"] = self.config.memory_budget
                metrics["memory_budget_unit"] = self.config.memory_budget_unit
                metrics["token_budget"] = int(context_stats["token_budget"].iloc[0])

            logger.info(
                "Average retained context KV memory: "
                f"{metrics['average_retained_kv_memory_mb']:.2f} MB "
                f"({metrics['average_retained_kv_memory_gb']:.4f} GB); "
                f"average compression ratio: {metrics['average_compression_ratio']:.6f}"
            )

        # metrics.json existing is what marks a configuration "done" to the
        # matrix skip check, so it must appear atomically: a kill mid-write must
        # not leave a partial file that counts as complete.
        tmp_filename = save_filename.with_suffix(".json.tmp")
        with open(str(tmp_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON
        os.replace(tmp_filename, save_filename)

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")
        return metrics

    def _save_results_readme(self, readme_filename: Path, task: str, metrics: Dict[str, Any]):
        """Write a self-contained summary as soon as one task finishes."""
        if self.config.memory_budget is None:
            if self.config.press_name == "no_press":
                configuration = "True no-press baseline (full KV cache)"
            else:
                configuration = (
                    f"{self.config.press_name} baseline " f"(compression ratio {self.config.compression_ratio:.4f})"
                )
        else:
            configuration = f"KVzip memory budget: {self.config.memory_budget:g} {self.config.memory_budget_unit}"

        metric_rows = "\n".join(
            f"| `{name}` | {value:.6f} |" if isinstance(value, float) else f"| `{name}` | {value} |"
            for name, value in metrics.items()
        )
        contents = f"""# {self.config.dataset.upper()} Benchmark Result

- Model: `{self.config.model}`
- Task: `{task}`
- Configuration: {configuration}
- Press: `{self.config.press_name}`
- Dataset fraction: `{self.config.fraction}`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
{metric_rows}

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
"""
        readme_filename.write_text(contents)
        logger.info(f"Result summary saved to {readme_filename}")

    def _reset_reused_model_state(self) -> None:
        """Clear state that must not leak between matrix configurations."""
        if self.pipeline is None:
            return

        self.pipeline.last_memory_budget_stats = None  # type: ignore[attr-defined]
        adapter = get_model_adapter(self.pipeline.model)
        for _, attention in adapter.iter_kv_attention_layers(self.pipeline.model):
            attention.masked_key_indices = None

        if self.press is not None and hasattr(self.press, "_reset_internal_parameters"):
            self.press._reset_internal_parameters()  # type: ignore[attr-defined]

        self._setup_deterministic_seeds()
        torch.cuda.empty_cache()

    def run_memory_budget_matrix(
        self,
        tasks: list[str],
        memory_budgets: list[tuple[float, str]],
        baseline_compression_ratio: float = 0.01,
        include_baseline: bool = True,
        baseline_press_name: Optional[str] = None,
    ) -> None:
        """Run multiple tasks and KV budgets while loading the model only once."""
        if not tasks:
            raise ValueError("At least one task is required for a matrix evaluation")

        output_dir = self._setup_directories()
        budget_press_name = self.config.press_name
        if baseline_press_name is not None and baseline_press_name != "no_press":
            raise ValueError("baseline_press_name currently supports only 'no_press'")
        self.config.compression_ratio = baseline_compression_ratio
        self.config.memory_budget = None
        # Model construction does not require a press. Individual compressed
        # configurations initialize their press immediately before inference.
        self.press = None
        self._setup_model_pipeline()

        configurations: list[tuple[Optional[float], str]] = list(memory_budgets)
        if include_baseline:
            configurations.insert(0, (None, "MB"))

        for task in tasks:
            logger.info(f"=== Starting matrix task: '{task}' ===")
            pending_configurations: list[tuple[Optional[float], str, Path]] = []

            for memory_budget, memory_budget_unit in configurations:
                self.config.data_dir = task
                is_no_press_baseline = memory_budget is None and baseline_press_name == "no_press"
                self.config.press_name = "no_press" if is_no_press_baseline else budget_press_name
                self.config.compression_ratio = 0.0 if is_no_press_baseline else baseline_compression_ratio
                self.config.memory_budget = memory_budget
                self.config.memory_budget_unit = memory_budget_unit.upper()
                results_dir = self.config.get_results_dir(output_dir, data_dir=task)
                predictions_filename = results_dir / "predictions.csv"
                metrics_filename = results_dir / "metrics.json"

                if predictions_filename.exists() and metrics_filename.exists():
                    logger.info(
                        f"Completed results already exist for task={task}, "
                        f"memory_budget={memory_budget}{memory_budget_unit}; skipping."
                    )
                    continue
                pending_configurations.append((memory_budget, memory_budget_unit, results_dir))

            if not pending_configurations:
                logger.info(f"All matrix configurations already exist for task '{task}'; skipping dataset load.")
                continue

            # Dataset text is loaded and prepared once, then copied before every
            # inference configuration because scoring mutates predicted_answer.
            self.config.memory_budget = None
            self.config.memory_budget_unit = "MB"
            self.config.compression_ratio = baseline_compression_ratio
            self.config.press_name = budget_press_name
            self._load_and_prepare_dataset(task_data_dir=task)
            source_df = self.df.copy(deep=True)  # type: ignore[union-attr]

            for memory_budget, memory_budget_unit, results_dir in pending_configurations:
                self.config.data_dir = task
                is_no_press_baseline = memory_budget is None and baseline_press_name == "no_press"
                self.config.press_name = "no_press" if is_no_press_baseline else budget_press_name
                self.config.compression_ratio = 0.0 if is_no_press_baseline else baseline_compression_ratio
                self.config.memory_budget = memory_budget
                self.config.memory_budget_unit = memory_budget_unit.upper()
                if is_no_press_baseline:
                    # True full-attention reference: do not construct, configure,
                    # enter, or otherwise invoke a press for this inference.
                    self.press = None
                    self.config.press_init_command = None
                    logger.info("Using true no-press baseline (press=None; setup skipped)")
                else:
                    self._setup_press()
                self._reset_reused_model_state()
                self.df = source_df.copy(deep=True)

                if memory_budget is None:
                    if is_no_press_baseline:
                        logger.info(f"Running task={task}, true no-press baseline")
                    else:
                        logger.info(
                            f"Running task={task}, {budget_press_name} reference "
                            f"compression_ratio={baseline_compression_ratio:.4f}"
                        )
                else:
                    logger.info(
                        f"Running task={task}, logical KVzip budget="
                        f"{memory_budget:g}{self.config.memory_budget_unit}"
                    )

                results_dir.mkdir(parents=True, exist_ok=True)
                predictions_filename = results_dir / "predictions.csv"
                metrics_filename = results_dir / "metrics.json"
                config_filename = results_dir / "config.yaml"
                readme_filename = results_dir / "README.md"

                self._run_inference()
                self._save_results(predictions_filename)
                metrics = self._calculate_and_save_metrics(metrics_filename)
                self.config.save_config(config_filename)
                self._save_results_readme(readme_filename, task, metrics)
                logger.info(
                    f"Completed task={task}, memory_budget="
                    f"{memory_budget if memory_budget is not None else 'reference'}"
                    f"{self.config.memory_budget_unit if memory_budget is not None else ''}"
                )

            self.df = None
            del source_df
            torch.cuda.empty_cache()
            logger.info(f"=== Completed matrix task: '{task}' ===")

        self.config.press_name = budget_press_name
        logger.info("Memory-budget matrix evaluation completed successfully with one model load.")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()
        # Define all LongBench tasks
        longbench_tasks = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
        # Determine which tasks to run
        if self.config.data_dir is None or (isinstance(self.config.data_dir, list) and len(self.config.data_dir) == 0):
            # The all-tasks default is a LongBench task list; silently iterating
            # it against another dataset ran 21 nonexistent subsets.
            if self.config.dataset != "longbench":
                raise ValueError(
                    f"No data_dir given and dataset is {self.config.dataset!r}: the run-everything "
                    "default only exists for longbench. Pass data_dir with the subset(s) to run."
                )
            tasks_to_run = longbench_tasks
            logger.info(f"No specific tasks provided. Running all {len(tasks_to_run)} LongBench tasks.")
        else:
            # Run specific tasks
            if isinstance(self.config.data_dir, str):
                tasks_to_run = [self.config.data_dir]
            else:
                tasks_to_run = self.config.data_dir
            logger.info(f"Running specific tasks: {tasks_to_run}")
        self._setup_press()
        self._setup_model_pipeline()

        for task in tasks_to_run:
            logger.info(f"Starting evaluation for task: {task}")
            results_dir = self.config.get_results_dir(output_dir, data_dir=task)
            predictions_filename = results_dir / "predictions.csv"
            metrics_filename = results_dir / "metrics.json"
            config_filename = results_dir / "config.yaml"
            readme_filename = results_dir / "README.md"

            if predictions_filename.exists() and metrics_filename.exists():
                logger.info(
                    f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
                )
                continue
            results_dir.mkdir(parents=True, exist_ok=True)

            self._load_and_prepare_dataset(task_data_dir=task)

            self._run_inference()
            self._save_results(predictions_filename)
            metrics = self._calculate_and_save_metrics(metrics_filename)
            self.config.save_config(config_filename)
            self._save_results_readme(readme_filename, task, metrics)
            logger.info(f"=== Completed task: '{task}' ===")
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
