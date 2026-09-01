#!/usr/bin/env python3
"""Run native Hugging Face baselines without importing or invoking KVPress."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Only benchmark data preparation and canonical metrics are shared with the
# main repository. The inference path below never imports the kvpress package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.benchmarks.loaders import load_benchmark_dataset  # noqa: E402
from evaluation.benchmarks.needle_in_haystack.utils import insert_needle_in_haystack  # noqa: E402
from evaluation.benchmarks.registry import DATASET_REGISTRY  # noqa: E402
from evaluation.benchmarks.results import score_prediction_frame  # noqa: E402
from evaluation.textstats import think_tag_stats  # noqa: E402


LOGGER = logging.getLogger("direct_baseline")
FORBIDDEN_CONFIG_KEYS = {
    "compression_ratio",
    "compression_interval",
    "head_compression_ratio",
    "hidden_states_buffer_size",
    "key_channel_compression_ratio",
    "memory_budget",
    "memory_budget_unit",
    "press_init_command",
    "press_name",
    "query_aware",
    "target_size",
    "threshold",
}


@dataclass
class BaselineConfig:
    dataset: str
    model: str
    output_dir: str
    data_dir: str | list[str] | None = None
    device: str = "cuda:0"
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    local_files_only: bool = False
    model_kwargs: dict[str, Any] | None = None
    max_context_length: int | None = None
    max_new_tokens: int | None = None
    fraction: float = 1.0
    limit: int | None = None
    seed: int = 42
    needle_depth: int | list[int] | None = None
    enable_thinking: bool = False
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    log_level: str = "INFO"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "BaselineConfig":
        forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(raw))
        if forbidden:
            raise ValueError(
                "Direct baseline configs cannot contain compression options: " + ", ".join(forbidden)
            )
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError("Unknown direct-baseline options: " + ", ".join(unknown))
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset {self.dataset!r}; choose from {sorted(DATASET_REGISTRY)}")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.max_context_length is not None and self.max_context_length < 1:
            raise ValueError("max_context_length must be positive")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.dataset == "needle_in_haystack":
            if self.needle_depth is None or self.max_context_length is None:
                raise ValueError("needle_in_haystack requires needle_depth and max_context_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", help="Override the model path in the YAML config")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    parser.add_argument("--limit", type=int, help="Override the per-task row limit")
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> BaselineConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    if args.model is not None:
        raw["model"] = args.model
    if args.output_dir is not None:
        raw["output_dir"] = str(args.output_dir)
    if args.limit is not None:
        raw["limit"] = args.limit
    return BaselineConfig.from_mapping(raw)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> str | torch.dtype:
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, name)


def is_prequantized(model_config: Any) -> bool:
    quantization = getattr(model_config, "quantization_config", None)
    if quantization is None and hasattr(model_config, "to_dict"):
        quantization = model_config.to_dict().get("quantization_config")
    if isinstance(quantization, dict):
        return bool(quantization.get("quant_method"))
    return quantization is not None


def load_native_model(config: BaselineConfig):
    common = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    model_config = AutoConfig.from_pretrained(config.model, **common)
    tokenizer = AutoTokenizer.from_pretrained(config.model, **common)
    prequantized = is_prequantized(model_config)

    load_kwargs: dict[str, Any] = {
        **common,
        "config": model_config,
        "dtype": torch_dtype(config.dtype),
        "attn_implementation": config.attn_implementation,
        **(config.model_kwargs or {}),
    }
    if config.device == "auto":
        load_kwargs["device_map"] = "auto"
    elif prequantized:
        # GPTQ modules must be placed while loading. Calling model.to() after
        # construction is not a portable operation across GPTQ backends.
        load_kwargs["device_map"] = {"": config.device}

    if model_config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(config.model, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)

    if config.device != "auto" and not prequantized:
        model.to(config.device)
    model.eval()
    LOGGER.info(
        "Loaded native model=%s type=%s prequantized=%s quantization=%r",
        config.model,
        model_config.model_type,
        prequantized,
        getattr(model_config, "quantization_config", None),
    )
    return model, tokenizer, model_config


def task_names(data_dir: str | list[str] | None) -> list[str | None]:
    if data_dir is None:
        return [None]
    return data_dir if isinstance(data_dir, list) else [data_dir]


def effective_context_limit(config: BaselineConfig, tokenizer: Any, model_config: Any) -> int:
    limits: list[int] = []
    text_config = getattr(model_config, "text_config", model_config)
    native_limit = getattr(text_config, "max_position_embeddings", None)
    if isinstance(native_limit, int) and 0 < native_limit < 10**9:
        limits.append(native_limit)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
        limits.append(tokenizer_limit)
    if config.max_context_length is not None:
        limits.append(config.max_context_length)
    if not limits:
        raise ValueError("Could not determine a finite context limit; set max_context_length")
    return min(limits)


def render_prompt(
    tokenizer: Any,
    context: str,
    question: str,
    answer_prefix: str,
    context_limit: int,
    enable_thinking: bool,
) -> tuple[torch.Tensor, int, int]:
    if tokenizer.chat_template is None:
        rendered_context = (getattr(tokenizer, "bos_token", "") or "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        if separator not in rendered:
            raise ValueError("Chat template removed the context separator")
        rendered_context, question_suffix = rendered.split(separator, 1)

    context_ids = tokenizer.encode(rendered_context, return_tensors="pt", add_special_tokens=False)
    original_context_tokens = int(context_ids.shape[-1])
    context_ids = context_ids[:, :context_limit]
    question_ids = tokenizer.encode(
        question + question_suffix + answer_prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = torch.cat((context_ids, question_ids), dim=-1)
    return input_ids, original_context_tokens, int(context_ids.shape[-1])


def model_input_device(model: Any) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        return embeddings.weight.device
    return next(model.parameters()).device


def generation_kwargs(config: BaselineConfig, max_new_tokens: int, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": config.do_sample,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if config.do_sample:
        for name in ("temperature", "top_p", "top_k"):
            value = getattr(config, name)
            if value is not None:
                kwargs[name] = value
    return kwargs


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


# What the predictions actually depend on. Deliberately NOT the whole config:
# hashing every field meant bumping log_level, moving output_dir, or resuming on
# cuda:1 instead of cuda:0 invalidated the checkpoint and threw away a run that
# could take days. Sampling knobs are included only when do_sample is on, since
# greedy decoding ignores them.
RESULT_AFFECTING_FIELDS = (
    "dataset",
    "data_dir",
    "model",
    "dtype",
    "attn_implementation",
    "trust_remote_code",
    "model_kwargs",
    "max_context_length",
    "max_new_tokens",
    "fraction",
    "limit",
    "seed",
    "needle_depth",
    "enable_thinking",
    "do_sample",
)
SAMPLING_FIELDS = ("temperature", "top_p", "top_k")


def fingerprint(config: BaselineConfig, task: str | None) -> str:
    fields = asdict(config)
    payload: dict[str, Any] = {name: fields[name] for name in RESULT_AFFECTING_FIELDS}
    if config.do_sample:
        payload.update({name: fields[name] for name in SAMPLING_FIELDS})
    payload.update(task=task, inference_backend="transformers.generate")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def prepare_dataset(config: BaselineConfig, task: str | None, tokenizer: Any) -> pd.DataFrame:
    frame = load_benchmark_dataset(config.dataset, task, DATASET_REGISTRY)
    if config.fraction < 1:
        frame = frame.sample(frac=config.fraction, random_state=config.seed)
    frame = frame.reset_index(drop=True)
    if config.dataset == "needle_in_haystack":
        frame = insert_needle_in_haystack(
            frame,
            tokenizer,
            config.max_context_length,
            config.needle_depth,
        )
    if config.limit is not None:
        frame = frame.head(config.limit).copy()
    required = {"context", "question", "answer_prefix", "max_new_tokens"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset {config.dataset}/{task} is missing columns: {missing}")
    return frame.reset_index(drop=True)


def task_output_dir(root: Path, task: str | None) -> Path:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task or "default")
    return root / safe_task


def read_completed(path: Path, expected_fingerprint: str) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("fingerprint") != expected_fingerprint:
            raise ValueError(f"Refusing to resume incompatible run: {path}")
        completed[int(record["row_index"])] = record
    return completed


def save_task_outputs(
    config: BaselineConfig,
    task: str | None,
    frame: pd.DataFrame,
    records: dict[int, dict[str, Any]],
    output_dir: Path,
) -> None:
    result = frame.copy()
    generated_columns = (
        "predicted_answer",
        "original_context_tokens",
        "retained_context_tokens",
        "prompt_tokens",
        "generated_tokens",
        "think_open_tags",
        "think_close_tags",
        "has_think_tag",
        "unclosed_think",
    )
    for column in generated_columns:
        result[column] = [records[index][column] for index in range(len(result))]

    # Keep scorer inputs but omit the potentially enormous context from the CSV.
    result.drop(columns=["context"], errors="ignore").to_csv(output_dir / "predictions.csv", index=False)
    metrics = score_prediction_frame(config.dataset, result)
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2) + "\n")
    summary = {
        "dataset": config.dataset,
        "task": task,
        "rows": len(result),
        "model": config.model,
        "inference_backend": "transformers.generate",
        "native_model_cache": True,
        "kvpress_imported": False,
        "compression": None,
        "hooks": None,
        "memory_budget": None,
        "rows_with_think_tag": int(result["has_think_tag"].sum()),
        "rows_with_unclosed_think": int(result["unclosed_think"].sum()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


@torch.inference_mode()
def run(config: BaselineConfig) -> None:
    set_deterministic_seed(config.seed)
    output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model, tokenizer, model_config = load_native_model(config)
    input_device = model_input_device(model)
    context_limit = effective_context_limit(config, tokenizer, model_config)
    LOGGER.info("Effective native context limit: %d", context_limit)

    for task in task_names(config.data_dir):
        frame = prepare_dataset(config, task, tokenizer)
        output_dir = task_output_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_fingerprint = fingerprint(config, task)
        jsonl_path = output_dir / "predictions.jsonl"
        completed = read_completed(jsonl_path, run_fingerprint)
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

        with jsonl_path.open("a") as sink:
            for row_index, row in frame.iterrows():
                if row_index in completed:
                    continue
                input_ids, original_tokens, retained_tokens = render_prompt(
                    tokenizer,
                    str(row["context"]),
                    str(row["question"]),
                    str(row.get("answer_prefix", "")),
                    context_limit,
                    config.enable_thinking,
                )
                input_ids = input_ids.to(input_device)
                attention_mask = torch.ones_like(input_ids)
                row_max_new_tokens = row.get("max_new_tokens", 50)
                max_new_tokens = int(config.max_new_tokens or row_max_new_tokens or 50)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs(config, max_new_tokens, tokenizer),
                )
                generated_ids = generated[0, input_ids.shape[-1] :]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
                record = {
                    "fingerprint": run_fingerprint,
                    "row_index": int(row_index),
                    "predicted_answer": prediction,
                    "original_context_tokens": original_tokens,
                    "retained_context_tokens": retained_tokens,
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "generated_tokens": int(generated_ids.shape[-1]),
                    **think_tag_stats(prediction),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                completed[int(row_index)] = record
                LOGGER.info(
                    "Completed dataset=%s task=%s row=%d/%d prompt_tokens=%d generated_tokens=%d think=%s",
                    config.dataset,
                    task,
                    row_index + 1,
                    len(frame),
                    record["prompt_tokens"],
                    record["generated_tokens"],
                    record["has_think_tag"],
                )
        save_task_outputs(config, task, frame, completed, output_dir)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(config)


if __name__ == "__main__":
    main()
