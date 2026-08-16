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

from evaluation.benchmarks.registry import LOFT_TASKS, RULER_32K_TASKS, SCORER_REGISTRY
from evaluation.benchmarks.results import score_prediction_frame

from .client import NIMClient, RateLimiter
from .datasets import available_datasets, canonical_dataset_name, load_examples
from .rlm import RLM, MemoryBudget, Scratchpad, vanilla_answer


def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def is_correct(pred: str | None, answers: list[str]) -> bool:
    if pred is None:
        return False
    p = normalize(pred)
    return any(normalize(a) in p for a in answers)


def load_done(path: Path) -> set:
    """IDs already completed. Records that died with a harness/API exception
    (e.g. a transient NIM outage) don't count -- re-running retries them."""
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("error"):
            done.add(record["id"])
    return done


def write_run_artifacts(
    checkpoint_path: Path,
    run_dir: Path,
    dataset_name: str,
    config: dict,
) -> dict:
    """Materialize the common predictions/config/metrics result contract.

    The headline score comes from the shared canonical scorer, so it is the same
    number a KVPress run of this dataset reports and the two are directly
    comparable. Everything RLM-specific lands under ``runtime``.
    """
    records = [json.loads(line) for line in checkpoint_path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(records).rename(columns={"pred": "predicted_answer"})
    frame.to_csv(run_dir / "predictions.csv", index=False)

    # An example that died with an API/harness exception is a MISSING measurement,
    # not a wrong answer. score_prediction_frame fills a null prediction with "",
    # which every scorer then marks 0 -- so a NIM outage would show up as the RLM
    # arm being less accurate than KVPress, which never makes network calls and so
    # can never take this hit. Score only the examples that actually produced an
    # answer, and report the shortfall separately.
    errored = frame["error"].notna() if "error" in frame else pd.Series(False, index=frame.index)
    scored_frame = frame[~errored]

    if dataset_name in SCORER_REGISTRY:
        metrics = score_prediction_frame(dataset_name, scored_frame) if len(scored_frame) else {}
    else:
        count = len(scored_frame)
        metrics = {"accuracy": float(scored_frame["correct"].sum() / count) if count else 0.0}

    count = len(frame)
    scored = len(scored_frame)
    runtime = {
        "examples": count,
        "scored": scored,
        "errors": int(errored.sum()),
        "progress_match": float(scored_frame["correct"].sum() / scored) if scored else 0.0,
        "average_tokens": float(scored_frame["tokens"].mean()) if scored else 0.0,
        "average_latency_s": float(scored_frame["latency_s"].mean()) if scored else 0.0,
        "unfinished": int((~scored_frame["finished"].astype(bool)).sum()) if scored else 0,
    }

    # Peak simultaneous root context is RLM's analogue of KVPress's
    # `average_retained_context_tokens` -- both say "how much context did the model
    # hold at once", which is the axis a memory sweep varies.
    if scored and "metrics" in scored_frame:
        peaks = [
            m.get("peak_context_tokens")
            for m in scored_frame["metrics"]
            if isinstance(m, dict) and m.get("peak_context_tokens")
        ]
        if peaks:
            runtime["average_peak_context_tokens"] = float(sum(peaks) / len(peaks))

    # The vanilla arm's ceiling: if the gold never survived truncation, its score is
    # a property of the char/token limit, not of the model. KVPress compresses
    # rather than truncates, so this column is what makes the gap interpretable.
    if scored and "truncated" in scored_frame:
        truncated = scored_frame["truncated"].fillna(False).astype(bool)
        runtime["truncated_fraction"] = float(truncated.sum() / scored)
        if "context_chars_used" in scored_frame and "context_chars" in scored_frame:
            retained = scored_frame["context_chars_used"] / scored_frame["context_chars"]
            runtime["average_context_chars_retained"] = float(retained.mean())

    metrics["runtime"] = runtime
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
    ap.add_argument(
        "--vanilla-max-prompt-tokens",
        type=int,
        default=None,
        help="hard TOKEN ceiling for the vanilla prompt; the char limit alone "
        "overflows the served window on densely-tokenising subsets (RULER cwe / "
        "niah_multikey_3), which the server rejects with a 400 and which then "
        "scores 0.0 from a harness error rather than from the model. Set to "
        "(max-model-len - max-tokens - margin), e.g. 34000 for a 40960 window",
    )
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
    # --- RLM scratchpad knobs (off unless --scratchpad; orthogonal to the budget) ---
    ap.add_argument(
        "--scratchpad",
        action="store_true",
        help="give the RLM root a note() tool; notes are re-shown every turn and survive budget eviction",
    )
    ap.add_argument(
        "--max-notes-tokens",
        type=int,
        default=1024,
        help="cap on the scratchpad block injected into context",
    )
    # --- Runaway guards. Each is independent; pass 0 to disable just that one. ---
    ap.add_argument(
        "--exec-timeout",
        type=float,
        default=60.0,
        help="wall-clock seconds of PURE-PYTHON execution a single RLM code block may run before "
        "being aborted (guards against model-generated infinite loops); time spent inside "
        "llm_query sub-calls is excluded. 0 disables",
    )
    ap.add_argument(
        "--run-timeout",
        type=float,
        default=900.0,
        help="wall-clock seconds for ONE example before the RLM gives up (end_reason=run_timeout). "
        "exec-timeout does not bound llm_query time, so without this a single pathological "
        "example can stall a job for hours. 0 disables",
    )
    ap.add_argument(
        "--max-sub-calls",
        type=int,
        default=40,
        help="cap on llm_query calls per example; further calls return a notice instead of "
        "hitting the API. 0 disables",
    )
    ap.add_argument("--out", default="evaluation/results/rlm")
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
    # Without this, a typo'd or omitted subset surfaces as a HF 404 deep inside
    # _load_loft after the clients are already built.
    if dataset_name == "loft" and args.data_dir not in LOFT_TASKS:
        ap.error("--dataset loft requires --data-dir with one of: " + ", ".join(LOFT_TASKS))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["vanilla", "rlm"] if args.mode == "both" else [args.mode]

    # One limiter shared by root + sub so the rpm cap is per-account (two separate
    # limiters would let the process issue up to ~2x rpm and trip 429s).
    limiter = RateLimiter(args.rpm)
    client_kw = dict(base_url=args.base_url, rpm=args.rpm, limiter=limiter)
    root = NIMClient(model=args.root_model, **client_kw)
    sub = NIMClient(model=args.sub_model, **client_kw)
    if args.no_think:
        eb = {"chat_template_kwargs": {"enable_thinking": False}}
        root.extra_body = sub.extra_body = eb
    budget = None
    if args.max_context_tokens is not None:
        budget = MemoryBudget(
            max_context_tokens=args.max_context_tokens,
            keep_recent_turns=args.keep_recent_turns,
        )
    scratchpad = Scratchpad(max_notes_tokens=args.max_notes_tokens) if args.scratchpad else None
    rlm = RLM(
        root_client=root,
        sub_client=sub,
        max_steps=args.max_steps,
        budget=budget,
        scratchpad=scratchpad,
        # `or None` is what makes each guard independently switchable off from the
        # CLI: argparse cannot pass None for a float/int flag, so 0 is the off
        # switch and is translated here.
        exec_timeout=args.exec_timeout or None,
        run_timeout=args.run_timeout or None,
        max_sub_calls=args.max_sub_calls or None,
    )

    for mode in modes:
        slug = args.root_model.replace("/", "_")
        task_slug = (args.data_dir or "default").replace("/", "_")
        # Anything that changes results must be in the directory name, or two
        # configurations share a checkpoint.jsonl and silently merge. (compare.py
        # reads config.yaml, not this name, so nothing else depends on the format.)
        components = [dataset_name, task_slug, slug, mode]
        if mode == "rlm" and args.max_context_tokens is not None:
            components.append(f"ctx{args.max_context_tokens}")
        if mode == "rlm" and scratchpad is not None:
            components.append("scratchpad")
        run_dir = out_dir / "__".join(components)
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
                    # LOFT's scorer reads this column to know which cue to look for
                    # and falls back to a hardcoded "Final Answer: " when it is
                    # absent. It was being appended to the question below but never
                    # recorded, so the scorer silently used the wrong cue.
                    "answer_prefix": ex.get("answer_prefix") or "",
                }
                record.update(ex.get("scoring", {}))
                question = ex["question"]
                if ex.get("answer_prefix"):
                    question = f"{question}\n{ex['answer_prefix']}"
                try:
                    if mode == "vanilla":
                        vstats: dict = {}
                        pred = vanilla_answer(
                            root,
                            ex["context"],
                            question,
                            char_limit=args.vanilla_char_limit,
                            max_prompt_tokens=args.vanilla_max_prompt_tokens,
                            stats=vstats,
                        )
                        record.update(pred=pred, steps=1, finished=True, end_reason="")
                        # Was the gold even inside the prompt we sent? Without this a
                        # truncated vanilla arm's score is indistinguishable from a
                        # model failure -- and unlike KVPress, which compresses the
                        # full context, this arm can simply not have seen the answer.
                        record.update(vstats)
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
            # `model` duplicates root_model under the key KVPress's config.yaml
            # uses, so a comparison joins the two backends on (dataset, data_dir,
            # model) without special-casing either one.
            "model": args.root_model,
            "mode": mode,
            "root_model": args.root_model,
            "sub_model": args.sub_model,
            "base_url": args.base_url,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "vanilla_char_limit": args.vanilla_char_limit,
            "vanilla_max_prompt_tokens": args.vanilla_max_prompt_tokens,
            "memory_budget": asdict(budget) if budget else None,
            "scratchpad": asdict(scratchpad) if scratchpad else None,
            "exec_timeout": args.exec_timeout or None,
            "run_timeout": args.run_timeout or None,
            "max_sub_calls": args.max_sub_calls or None,
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
