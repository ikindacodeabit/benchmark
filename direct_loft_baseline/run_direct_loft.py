#!/usr/bin/env python3
"""Evaluate LOFT with native Hugging Face generation and no KVPress code."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from loft_metrics import score_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def load_model(model_path: str, device: str, dtype: str, attention: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": attention}
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_loft(task: str):
    dataset_name, length = task.rsplit("_", 1)
    dataset = load_dataset(f"f20180301/loft-rag-{dataset_name}-{length}")
    splits = []
    for split_name in ("dev", "test"):
        if split_name in dataset:
            split = dataset[split_name].add_column("split", [split_name] * len(dataset[split_name]))
            splits.append(split)
    if not splits:
        raise ValueError(f"No dev/test split found for {task}")
    return concatenate_datasets(splits)


def prompt_ids(tokenizer, context: str, question: str, answer_prefix: str, max_context_length: int | None,
               enable_thinking: bool) -> tuple[torch.Tensor, int, int]:
    if tokenizer.chat_template is None:
        rendered_context = (tokenizer.bos_token or "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        rendered_context, question_suffix = rendered.split(separator)

    context_ids = tokenizer.encode(rendered_context, return_tensors="pt", add_special_tokens=False)
    original_context_tokens = context_ids.shape[-1]
    if max_context_length is not None and context_ids.shape[-1] > max_context_length:
        context_ids = context_ids[:, :max_context_length]
    question_ids = tokenizer.encode(
        question + question_suffix + answer_prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return torch.cat((context_ids, question_ids), dim=-1), original_context_tokens, context_ids.shape[-1]


def json_safe_answers(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    return [str(value)]


def write_outputs(task_dir: Path, records: list[dict[str, Any]], task: str) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with (task_dir / "predictions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["answers"] = json.dumps(row["answers"], ensure_ascii=False)
            writer.writerow(row)
    (task_dir / "metrics.json").write_text(json.dumps(score_records(records, task), indent=2) + "\n")
    summary = {
        "task": task,
        "rows": len(records),
        "rows_with_think_tag": sum(record["has_think_tag"] for record in records),
        "rows_with_unclosed_think": sum(record["unclosed_think"] for record in records),
        "enable_thinking": records[0]["enable_thinking"] if records else None,
        "inference_backend": "transformers.generate",
        "kvpress_imported": False,
        "compression": None,
        "memory_budget": None,
    }
    (task_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, args.device, args.dtype, args.attn_implementation)
    model_device = next(model.parameters()).device
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    for task in args.tasks:
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = task_dir / "predictions.jsonl"
        records: list[dict[str, Any]] = []
        completed: set[int] = set()
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                record = json.loads(line)
                records.append(record)
                completed.add(int(record["row_index"]))

        dataset = load_loft(task)
        row_count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
        with jsonl_path.open("a") as output:
            for row_index in range(row_count):
                if row_index in completed:
                    continue
                row = dataset[row_index]
                input_ids, original_tokens, retained_tokens = prompt_ids(
                    tokenizer,
                    str(row["context"]),
                    str(row["question"]),
                    str(row["answer_prefix"]),
                    args.max_context_length,
                    args.enable_thinking,
                )
                input_ids = input_ids.to(model_device)
                max_new_tokens = int(args.max_new_tokens or row["max_new_tokens"])
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                prediction = tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=True)
                open_tags = len(re.findall(r"<think>", prediction, flags=re.IGNORECASE))
                close_tags = len(re.findall(r"</think>", prediction, flags=re.IGNORECASE))
                record = {
                    "row_index": row_index,
                    "task": task,
                    "split": str(row["split"]),
                    "question": str(row["question"]),
                    "answers": json_safe_answers(row["answers"]),
                    "answer_prefix": str(row["answer_prefix"]),
                    "max_new_tokens": max_new_tokens,
                    "original_context_tokens": original_tokens,
                    "retained_context_tokens": retained_tokens,
                    "prompt_tokens": input_ids.shape[-1],
                    "predicted_answer": prediction,
                    "think_open_tags": open_tags,
                    "think_close_tags": close_tags,
                    "has_think_tag": bool(open_tags or close_tags),
                    "unclosed_think": open_tags > close_tags,
                    "enable_thinking": args.enable_thinking,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)
                print(f"{task}: {len(records)}/{row_count} think={record['has_think_tag']}", flush=True)
        records.sort(key=lambda record: int(record["row_index"]))
        write_outputs(task_dir, records, task)


if __name__ == "__main__":
    main()
