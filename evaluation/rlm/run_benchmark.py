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
from typing import Any

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

    metrics: dict
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

    # Sub-side KV retention: the direct analogue of KVPress's
    # `average_retained_context_tokens`, but measured over llm_query sub-calls --
    # "how much of each context slice did the sub model effectively attend to".
    if scored and "metrics" in scored_frame:
        sub_kv = [
            m["sub_kv"]
            for m in scored_frame["metrics"]
            if isinstance(m, dict) and isinstance(m.get("sub_kv"), dict) and m["sub_kv"]
        ]
        if sub_kv:
            n_kv = len(sub_kv)
            total_calls = sum(s.get("calls", 0) for s in sub_kv)
            runtime["average_sub_context_tokens"] = float(
                sum(s.get("average_context_tokens", 0.0) for s in sub_kv) / n_kv
            )
            runtime["average_sub_retained_context_tokens"] = float(
                sum(s.get("average_retained_context_tokens", 0.0) for s in sub_kv) / n_kv
            )
            runtime["average_sub_compression_ratio"] = float(
                sum(s.get("average_compression_ratio", 0.0) for s in sub_kv) / n_kv
            )
            # How often the root actually used the two-arg form. Near 0 means the
            # arm degenerated to dense one-arg calls and is NOT measuring the press.
            runtime["sub_split_call_fraction"] = (
                float(sum(s.get("split_calls", 0) for s in sub_kv) / total_calls) if total_calls else 0.0
            )

    metrics["runtime"] = runtime
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return metrics


def build_run_dir_components(args: argparse.Namespace, mode: str, scratchpad: Scratchpad | None) -> list[str]:
    """Name the run directory after everything that changes results.

    Two configurations that differ in any result-affecting knob must land in
    different directories, or they share a checkpoint.jsonl and silently merge.
    (compare.py reads config.yaml, not this name, so nothing else depends on the
    format.)
    """
    dataset_name = canonical_dataset_name(args.dataset or args.legacy_dataset)
    model_slug = args.root_model.replace("/", "_")
    task_slug = (args.data_dir or "default").replace("/", "_")
    components = [dataset_name, task_slug, model_slug, mode]
    if mode == "rlm" and args.max_context_tokens is not None:
        components.append(f"ctx{args.max_context_tokens}")
    if mode == "rlm" and scratchpad is not None:
        components.append("scratchpad")
    if mode == "rlm" and args.sub_backend == "kvzip":
        # Backend-prefixed ("kvzip-kvzip0.5", not the old kvpress-era "kvzip0.5"):
        # the standalone-KVzip backend produces different results (real eviction,
        # its own template/scoring), so resuming into an old kvpress-backed
        # checkpoint would silently merge two experiments.
        components.append(f"{args.sub_backend}-{args.press}{args.compression_ratio:g}")
    if mode == "rlm" and args.max_subcall_chars != 32000:
        components.append(f"sub{args.max_subcall_chars}")
    return components


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
    # --- Sub-call backend (arm 4: KV-compressed sub-calls; see kvzip_backend.py) ---
    # `nim` keeps today's behavior: sub-calls go to the same OpenAI-compatible
    # server as the root. `kvzip` loads the sub model IN-PROCESS through the
    # standalone KVzip repo (snu-mllab/KVzip) so each llm_query context slice is
    # read through a pruned KV cache that is PHYSICALLY evicted (real memory
    # savings, unlike kvpress's masking). KVzip pins transformers==4.51.3, the
    # same version the vLLM 0.8.5 root server uses.
    ap.add_argument("--sub-backend", default="nim", choices=["nim", "kvzip"])
    ap.add_argument(
        "--press",
        default="kvzip",
        choices=["kvzip", "no_press"],
        help="for --sub-backend kvzip; no_press = same load/generate path, no pruning (press control)",
    )
    ap.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="fraction of context KV evicted per sub-call (kvzip backend; KVzip's own "
        "prune(ratio) RETAINS that fraction — the conversion happens in the backend)",
    )
    ap.add_argument(
        "--max-subcall-chars",
        type=int,
        default=32000,
        help="char cap per llm_query prompt/context slice (was constructor-only); the split "
        "prompt advertises this cap to the root, so raising it is what makes the "
        "'compression enables bigger reads' arm real",
    )
    ap.add_argument(
        "--sub-max-tokens",
        type=int,
        default=512,
        help="max_new_tokens per kvzip sub-call; HF greedy decode is ~30-40 tok/s, so the "
        "NIM default of 4096 is a wall-clock hazard in-process",
    )
    ap.add_argument(
        "--sub-device",
        default=None,
        help="GPU for the in-process sub model, e.g. cuda:1 (default: auto-pick the GPU with "
        "the most free memory; either way the choice is preflight-checked against "
        "--sub-min-free-gib before the model loads)",
    )
    ap.add_argument(
        "--sub-min-free-gib",
        type=float,
        default=14.0,
        help="minimum free GPU memory (GiB) required by the preflight check before loading the "
        "sub model (weights + KV headroom; ~14 covers Qwen3-4B bf16 at 34k-token sub-calls)",
    )
    ap.add_argument(
        "--kvzip-dir",
        default=None,
        help="path to a snu-mllab/KVzip checkout (defaults to the KVZIP_DIR env var); KVzip has "
        "no pip package",
    )
    ap.add_argument(
        "--sub-max-context-tokens",
        type=int,
        default=34000,
        help="token-level truncation of the sub-call context (a 131072-char slice of dense "
        "text can exceed 32k tokens)",
    )
    ap.add_argument(
        "--press-min-tokens",
        type=int,
        default=1024,
        help="one-arg llm_query prompts below this many tokens skip the press",
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
    sub: Any
    if args.sub_backend == "kvzip":
        if modes == ["vanilla"]:
            ap.error("--sub-backend kvzip only affects RLM sub-calls; use --mode rlm (or both)")
        # Imported lazily: the nim path must stay importable in a venv without
        # torch or KVzip's deps.
        from .kvzip_backend import KVzipSubClient

        sub = KVzipSubClient(
            model=args.sub_model,
            press_name=args.press,
            compression_ratio=args.compression_ratio,
            device=args.sub_device,
            max_new_tokens=args.sub_max_tokens,
            max_context_tokens=args.sub_max_context_tokens,
            press_min_tokens=args.press_min_tokens,
            kvzip_dir=args.kvzip_dir,
            min_free_gib=args.sub_min_free_gib,
        )
    else:
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
        max_subcall_chars=args.max_subcall_chars,
        # In-process sub-calls can take minutes each; without an in-call check a
        # single code cell looping over slices sails past the per-step deadline.
        subcall_deadline_check=(args.sub_backend == "kvzip"),
    )

    for mode in modes:
        run_dir = out_dir / "__".join(build_run_dir_components(args, mode, scratchpad))
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
                        if hasattr(sub, "pop_example_stats"):
                            r.metrics["sub_kv"] = sub.pop_example_stats() or None
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
            "sub_backend": args.sub_backend,
            "press": args.press if args.sub_backend == "kvzip" else None,
            "compression_ratio": args.compression_ratio if args.sub_backend == "kvzip" else None,
            "max_subcall_chars": args.max_subcall_chars,
            "sub_max_tokens": args.sub_max_tokens if args.sub_backend == "kvzip" else None,
            "sub_max_context_tokens": args.sub_max_context_tokens if args.sub_backend == "kvzip" else None,
            "press_min_tokens": args.press_min_tokens if args.sub_backend == "kvzip" else None,
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
