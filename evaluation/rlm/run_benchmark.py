# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run vanilla vs RLM on a long-context task. Resumable JSONL checkpointing.

Usage:
  python -m evaluation.rlm.run_benchmark --dataset niah --mode rlm --limit 1 --debug
  python -m evaluation.rlm.run_benchmark --dataset ruler32k --data-dir niah_single_1 --limit 5
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from evaluation.benchmarks.registry import RULER_32K_TASKS, SCORER_REGISTRY
from evaluation.benchmarks.results import score_prediction_frame

from .client import NIMClient
from .datasets import available_datasets, canonical_dataset_name, load_examples
from .rlm import RLM, MemoryBudget, vanilla_answer


def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def is_correct(pred: str | None, answers: list[str]) -> bool:
    if pred is None:
        return False
    p = normalize(pred)
    return any(normalize(a) in p for a in answers)


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in open(path) if line.strip()}


def write_run_artifacts(
    checkpoint_path: Path,
    run_dir: Path,
    dataset_name: str,
    config: dict,
) -> dict:
    """Materialize the common predictions/config/metrics result contract."""
    records = [json.loads(line) for line in checkpoint_path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(records).rename(columns={"pred": "predicted_answer"})
    frame.to_csv(run_dir / "predictions.csv", index=False)

    if dataset_name in SCORER_REGISTRY:
        metrics = score_prediction_frame(dataset_name, frame)
    else:
        count = len(frame)
        metrics = {"accuracy": float(frame["correct"].sum() / count) if count else 0.0}
    count = len(frame)
    metrics["runtime"] = {
        "examples": count,
        "progress_match": float(frame["correct"].sum() / count) if count else 0.0,
        "average_tokens": float(frame["tokens"].mean()) if count else 0.0,
        "average_latency_s": float(frame["latency_s"].mean()) if count else 0.0,
        "unfinished": int((~frame["finished"].astype(bool)).sum()) if count else 0,
        "errors": int(frame["error"].notna().sum()) if count and "error" in frame else 0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    dataset_selection = ap.add_mutually_exclusive_group(required=True)
    dataset_selection.add_argument("--dataset", choices=available_datasets())
    dataset_selection.add_argument(
        "--task",
        dest="legacy_dataset",
        choices=available_datasets(),
        help="deprecated alias for --dataset",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="shared Hugging Face config/subset, e.g. niah_single_1 for ruler32k",
    )
    ap.add_argument("--mode", default="both", choices=["vanilla", "rlm", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--root-model", default="meta/llama-3.3-70b-instruct", help="NIM model id for vanilla baseline AND the RLM root"
    )
    ap.add_argument(
        "--sub-model", default="meta/llama-3.1-8b-instruct", help="NIM model id for recursive sub-calls (cheap is fine)"
    )
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--rpm", type=int, default=35)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--vanilla-char-limit", type=int, default=400_000)
    # --- RLM memory-budget knobs (no budget unless --max-context-tokens is set) ---
    # Eviction-only: out-of-budget turns are simply dropped (no notes/summarization).
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="cap the RLM root's context window (tokens); unset = unbounded (legacy)",
    )
    ap.add_argument(
        "--keep-recent-turns",
        type=int,
        default=3,
        help="recent (assistant,observation) pairs to keep verbatim under budget",
    )
    ap.add_argument("--out", default="benchmark_artifacts/results/rlm")
    ap.add_argument("--debug", action="store_true", help="print every RLM step (model reply, code, REPL output) live")
    ap.add_argument(
        "--no-think",
        action="store_true",
        help="disable Qwen3 thinking mode (chat_template_kwargs.enable_thinking=False)",
    )
    args = ap.parse_args()

    requested_dataset = args.dataset or args.legacy_dataset
    dataset_name = canonical_dataset_name(requested_dataset)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if dataset_name == "ruler32k" and args.data_dir not in RULER_32K_TASKS:
        ap.error("--dataset ruler32k requires --data-dir with one of: " + ", ".join(RULER_32K_TASKS))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["vanilla", "rlm"] if args.mode == "both" else [args.mode]

    root = NIMClient(model=args.root_model, base_url=args.base_url, rpm=args.rpm)
    sub = NIMClient(model=args.sub_model, base_url=args.base_url, rpm=args.rpm)
    if args.no_think:
        eb = {"chat_template_kwargs": {"enable_thinking": False}}
        root.extra_body = sub.extra_body = eb
    budget = None
    if args.max_context_tokens is not None:
        budget = MemoryBudget(
            max_context_tokens=args.max_context_tokens,
            keep_recent_turns=args.keep_recent_turns,
        )
    rlm = RLM(root_client=root, sub_client=sub, max_steps=args.max_steps, budget=budget)

    for mode in modes:
        slug = args.root_model.replace("/", "_")
        task_slug = (args.data_dir or "default").replace("/", "_")
        run_dir = out_dir / f"{dataset_name}__{task_slug}__{slug}__{mode}"
        run_dir.mkdir(parents=True, exist_ok=True)
        res_path = run_dir / "checkpoint.jsonl"
        tdir = run_dir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        done = load_done(res_path)
        print(f"== {dataset_name}/{args.data_dir or 'default'} / {mode} " f"-> {run_dir} ({len(done)} already done)")

        n, correct = 0, 0
        with open(res_path, "a") as fout:
            for ex in load_examples(requested_dataset, args.data_dir, args.limit):
                if ex["id"] in done:
                    continue
                t0 = time.time()
                tok0 = root.usage.total_tokens + sub.usage.total_tokens
                sub0 = sub.usage.calls
                record = {
                    "id": ex["id"],
                    "dataset": dataset_name,
                    "task": ex.get("task") or args.data_dir or dataset_name,
                    "mode": mode,
                    "answers": ex["answers"],
                    "answer": ex.get("scoring", {}).get("answer", ex["answers"]),
                }
                record.update(ex.get("scoring", {}))
                question = ex["question"]
                if ex.get("answer_prefix"):
                    question = f"{question}\n{ex['answer_prefix']}"
                try:
                    if mode == "vanilla":
                        pred = vanilla_answer(root, ex["context"], question, char_limit=args.vanilla_char_limit)
                        record.update(pred=pred, steps=1, finished=True, end_reason="")
                    else:
                        r = rlm.run(ex["context"], question)
                        record.update(
                            pred=r.answer,
                            steps=r.steps,
                            finished=r.finished,
                            end_reason=r.end_reason,
                            metrics=r.metrics,
                        )
                        # Always save the full transcript for post-mortems:
                        with open(tdir / f"{ex['id']}.json", "w") as tf:
                            json.dump(
                                {
                                    "question": question,
                                    "answers": ex["answers"],
                                    "pred": r.answer,
                                    "end_reason": r.end_reason,
                                    "metrics": r.metrics,
                                    "transcript": r.transcript,
                                },
                                tf,
                                indent=2,
                            )
                        if args.debug:
                            for t in r.transcript:
                                print(f"\n--- step {t['step']} ---")
                                print("MODEL REPLY:\n" + (t.get("reply") or "")[:2000])
                                print("EXECUTED CODE:\n" + (t.get("code") or "<none>"))
                                print("REPL OUTPUT:\n" + t["observation"][:2000])
                            print(f"\nEND: reason={r.end_reason} pred={r.answer!r}\n")
                except Exception as e:
                    record.update(
                        pred=None, error=f"{type(e).__name__}: {e}", steps=0, finished=False, end_reason="exception"
                    )
                record["correct"] = is_correct(record.get("pred"), ex["answers"])
                record["latency_s"] = round(time.time() - t0, 2)
                record["tokens"] = root.usage.total_tokens + sub.usage.total_tokens - tok0
                record["sub_calls"] = sub.usage.calls - sub0
                record["ctx_chars"] = len(ex["context"])
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                n += 1
                correct += record["correct"]
                print(
                    f"  [{ex['id']}] correct={record['correct']} "
                    f"steps={record['steps']} sub_calls={record['sub_calls']} "
                    f"end={record.get('end_reason', '')} tokens={record['tokens']} "
                    f"t={record['latency_s']}s"
                )
        if n:
            print(f"== {mode}: {correct}/{n} correct ({100*correct/n:.1f}%)")
        run_config = {
            "backend": "rlm",
            "dataset": dataset_name,
            "data_dir": args.data_dir,
            "mode": mode,
            "root_model": args.root_model,
            "sub_model": args.sub_model,
            "base_url": args.base_url,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "vanilla_char_limit": args.vanilla_char_limit,
            "memory_budget": asdict(budget) if budget else None,
        }
        metrics = write_run_artifacts(
            res_path,
            run_dir,
            dataset_name,
            run_config,
        )
        print(f"== shared artifacts written to {run_dir}; metrics={metrics}")
    print(
        f"Total API calls: root={root.usage.calls}, sub={sub.usage.calls}; "
        f"tokens: {root.usage.total_tokens + sub.usage.total_tokens}"
    )


if __name__ == "__main__":
    main()
