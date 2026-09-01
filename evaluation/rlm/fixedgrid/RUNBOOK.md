# Runbook: run one fixed-chunk × compression cell on the infolab servers

This is a copy-paste guide for running **one grid cell** — a chosen retained-KV
budget `B` and compression factor `F` — of the RLM fixed-chunk experiment on an
IITB CSE infolab host. For the design and the campaign-wide sweep see
[`README.md`](README.md); for a from-scratch environment build see
[`../../../REPRODUCE.md`](../../../REPRODUCE.md).

## What a cell is

The RLM root reads the document in slices and hands each slice to a sub-model
whose KV cache is compressed by KVzip. A cell fixes two numbers:

| knob | flag | meaning |
| --- | --- | --- |
| `B` | `--memory-budget B --memory-budget-unit tokens` | KV tokens the sub-call may **retain** after compression |
| `F` | `--compression-factor F` | chunk / retained factor; press ratio `= 1 - 1/F` |

`--fixed-chunk` then forces the sub-call to receive **exactly `N = B × F`**
context tokens (oversized char floor, question room reserved in the char cap,
every sub-call token-truncated to `N`, and a hard failure — not a silent shrink —
if the model window or GPU cap would make the cell smaller than `N`).

- `F = 1` is the uncompressed baseline. Use `--press no_press` (identity, ~3×
  faster than KVzip at ratio 0, and verified to give byte-identical predictions).
- `F ≥ 2` uses `--press kvzip`.

Subsets (`--data-dir`): `musique hotpotqa 2wikimqa narrativeqa qasper triviaqa`,
each a locally built pack of 60 rows at exactly 131072 tokens.

## 0. Host and paths

Workers: `ant bee cat dog elk fox` `.cse.iitb.ac.in`. `/mnt/nas/gautammahale` is
shared across all of them, so the checkout, venvs, data pack and model cache are
staged once and every host sees them. Pick a host with a free GPU
(`nvidia-smi`), work from **one** host per lane, and always run under `tmux`
(there is no scheduler; an ssh drop otherwise orphans the vLLM server).

```bash
ssh dog.cse.iitb.ac.in
newgrp infolab                    # so files land group-infolab on the shared tree
export BASE=/mnt/nas/$USER
export REPO=$BASE/fixedgrid/repo                       # feature/fixed-chunk-grid checkout
export KVPY=$BASE/benchmark/.venv-kvpress/bin/python   # sub-calls: transformers >= 4.56, torch 2.6
#   $BASE/benchmark/.venv/bin/vllm  is the serving venv: vllm 0.8.5 + transformers 4.51.3
export RLM_DATA_DIR=$BASE/rlm_data                     # holds longbench128k/<subset>/data.parquet
export HF_HOME=$BASE/hf_cache                          # holds Qwen/Qwen3-4B-Instruct-2507
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

If any of those are missing on your host, do the one-time setup in the appendix
first.

## 1. Serve the root model (once, reused by every cell)

```bash
tmux new -s fixedgrid
CUDA_VISIBLE_DEVICES=0 "$BASE/benchmark/.venv/bin/vllm" serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 --max-model-len 139264 \
  --gpu-memory-utilization 0.90 --enforce-eager -O0
```

Wait until ready, in another pane (`Ctrl-b c`):

```bash
curl -sf http://localhost:8000/v1/models | grep -q Qwen3-4B-Instruct-2507 && echo READY
```

The root server sits on its own GPU. Every cell's **sub-model** runs on a
*different* GPU (`GPU=` below). The harness never starts or stops the server.

## 2. Choose the cell and derive its parameters

`N = B × F`. Two derived values depend only on `N` (Qwen3-4B is 144 KiB/token):

```
--max-sub-calls  = min(64, ceil(131072 / N) + 1)      # ~128K doc tokens read per example
--sub-min-free-gib = max(14, ceil(N * 147456 * 1.2 / 2^30) + 12)
```

| N | `--max-sub-calls` | `--sub-min-free-gib` |
| ---: | ---: | ---: |
| 1024 | 64 | 14 |
| 2048 | 64 | 14 |
| 4096 | 33 | 14 |
| 8192 | 17 | 14 |
| 16384 | 9 | 15 |
| 32768 | 5 | 18 |
| 65536 | 3 | 23 |
| 131072 | 2 | 34 |

`run_grid.sh` computes both for you. They only matter if you call
`run_benchmark` directly (§3b). KVzipPress **masks** evicted KV, it does not free
it — the GPU must hold the full `N`-token cache regardless of `B`, which is why
`--sub-min-free-gib` scales with `N` and not with `B`.

## 3a. Run it — via `run_grid.sh` (recommended)

Handles the derivations, a per-cell lock (`$RESULTS/.locks/<subset>.<B>.<F>.lock`)
so lanes can run in parallel, and a skip when `metrics.json` already exists.

```bash
cd "$REPO"
DATASETS=hotpotqa BUDGETS=4096 FACTORS=8 LIMIT=10 \
  GPU=1 PORT=8000 \
  REPO_ROOT="$REPO" KVZIP_PYTHON="$KVPY" RLM_DATA_DIR="$RLM_DATA_DIR" HF_HOME="$HF_HOME" \
  RESULTS=$BASE/fixedgrid/myrun \
  bash evaluation/rlm/fixedgrid/run_grid.sh
```

`BUDGETS` and `FACTORS` are space-separated lists — pass one value each for a
single cell, or several to sweep (their cross product). Env vars and defaults:

| var | default | note |
| --- | --- | --- |
| `DATASETS` | all six | `--data-dir` values |
| `BUDGETS` | `1024 2048 4096 8192` | `B`, in retained KV tokens |
| `FACTORS` | `1 2 4 8 16` | `F` |
| `LIMIT` | `50` | rows per cell = first `LIMIT` of the 60-row pack |
| `GPU` | `0` | **sub-model** GPU (not the server's) |
| `PORT` | `8000` | where the vLLM root is served |
| `MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | root and sub |
| `RESULTS` | `evaluation/results/fixedgrid` | **use a fresh dir per `LIMIT`** (see gotchas) |
| `REPO_ROOT` | `$(pwd)` | the checkout |
| `KVZIP_PYTHON` | `$REPO_ROOT/.venv-kvpress/bin/python` | set when the venv is elsewhere |
| `RLM_DATA_DIR` | *unset* | **must export** — the loader reads `os.environ`; `run_grid.sh` does not set it |

## 3b. Run it — `run_benchmark` directly (one cell, full control)

```bash
cd "$REPO"
B=4096; F=8; SUBSET=hotpotqa; NEX=10; GPU=1
N=$((B*F)); CALLS=$(( (131072 + N - 1)/N + 1 )); [ $CALLS -gt 64 ] && CALLS=64
MINFREE=$(awk -v n=$N 'BEGIN{x=n*147456*1.2/(2^30);m=int(x);if(x>m)m++;m+=12;if(m<14)m=14;print m}')
PRESS=kvzip; [ "$F" -eq 1 ] && PRESS=no_press

CUDA_VISIBLE_DEVICES=$GPU "$KVPY" -m evaluation.rlm.run_benchmark \
  --dataset longbench128k --data-dir "$SUBSET" --limit "$NEX" \
  --base-url http://localhost:8000/v1 \
  --root-model Qwen/Qwen3-4B-Instruct-2507 --sub-model Qwen/Qwen3-4B-Instruct-2507 \
  --mode rlm --scratchpad --max-steps 50 --exec-timeout 60 --run-timeout 3600 \
  --sub-backend kvzip --press "$PRESS" \
  --memory-budget "$B" --memory-budget-unit tokens \
  --max-subcall-chars auto --compression-factor "$F" --fixed-chunk \
  --max-sub-calls "$CALLS" --sub-max-tokens 128 --sub-min-free-gib "$MINFREE" \
  --out $BASE/fixedgrid/myrun
```

`--fixed-chunk` requires `--sub-backend kvzip` **and** `--max-subcall-chars auto`
**and** one of `--compression-factor` / `--target-compression-ratio`.

## 4. Read the result

Each cell writes its own directory under `$RESULTS`, named e.g.
`longbench128k__hotpotqa__Qwen_Qwen3-4B-Instruct-2507__rlm__scratchpad__kvzip-kvzip4096tokens__autosubx8__fixed`.
Check `metrics.json`:

```bash
python - <<'PY'
import json; m=json.load(open("METRICS_PATH")); r=m["runtime"]
print("score (qa_f1 x100)          ", m["score"])
print("errors                      ", r["errors"])                      # want 0
print("avg sub context tokens ~ N  ", r["average_sub_context_tokens"])
print("avg compression ratio ~1-1/F", r["average_sub_compression_ratio"])
print("realized factor ~ F         ", r["realized_compression_factor"])
print("on-target fraction ~ 1.0    ", r["sub_context_tokens_on_target_fraction"])
print("sizing binding == 'budget'  ", r["subcall_sizing_binding"])
print("document coverage fraction  ", r["document_coverage_fraction"])  # read beside score
PY
```

A healthy cell has `errors == 0`, context tokens within ~2 % of `N`, ratio within
~0.01 of `1 - 1/F`, and `subcall_sizing_binding == "budget"`. If sizing bound on
anything else the run should have raised at startup rather than produced numbers.
`document_coverage_fraction` is the union of document spans the root actually
read verbatim; it runs low on multi-hop subsets (the root concatenates several
windows into one payload) and is a diagnostic, not a pass/fail.

Render many cells as tables:

```bash
python evaluation/rlm/fixedgrid/grid_table.py --results $BASE/fixedgrid/myrun
# -> $BASE/fixedgrid/myrun/grid_tables/{<subset>.csv,<subset>.md,grid_long.csv}
```

## 5. Parallel lanes / multiple hosts

One cell occupies one sub-GPU. To go faster, run more lanes with **disjoint**
`DATASETS`, `BUDGETS` or `FACTORS`, each with its own `GPU=`, all pointing at the
**same absolute `RESULTS`** path. The per-cell lock is the final duplicate guard.
Example two-lane split on one host (server on GPU 0):

```bash
# pane A
DATASETS=hotpotqa FACTORS='1 2 4' GPU=1 RESULTS=$BASE/fixedgrid/myrun ... bash .../run_grid.sh
# pane B
DATASETS=hotpotqa FACTORS='8 16'  GPU=4 RESULTS=$BASE/fixedgrid/myrun ... bash .../run_grid.sh
```

Across hosts, keep everything under `/mnt/nas/$USER` and give every lane the same
`RESULTS`.

## Gotchas

- **`RESULTS` must be fresh per `LIMIT`.** `limit` is resume-critical but is *not*
  in the run-dir name, so a `LIMIT=10` run and a `LIMIT=50` run collide in one
  directory and the second trips the resume guard on the first's rows. Keep
  smoke / pilot / full campaigns in separate `RESULTS` trees.
- **Run under `tmux`.** No scheduler; nothing restarts a dead job, and an ssh
  drop orphans the vLLM server (it keeps the port and ~40 GB).
- **Two venvs, never mixed.** The server needs `transformers==4.51.3` (vLLM 0.8.5
  calls a tokenizer method 5.x removed); the sub-call venv needs `>= 4.56` (the
  new `Cache` API). They coexist only because they are different processes.
- **KVzipPress masks, it does not free.** Every cell physically holds the full
  `N`-token KV cache regardless of `B`. The budget rows are *simulated* retention
  — never report them as measured GPU-memory savings.
- **`F = 1` → `--press no_press`.** KVzip at ratio 0 is identity but pays 2–3
  extra prefill passes.
- **`--sub-min-free-gib` / `--max-sub-calls` assume Qwen3-4B.** Change them if you
  swap the sub-model (the KV-per-token cost changes).
- **`GPU=` is the sub-model's GPU**, distinct from the server's. Pointing it at
  the server's GPU will OOM the big-`N` cells.
- **`--split` is irrelevant for `longbench128k`** (single parquet, no dev/test).
  `--limit N` deterministically takes the first `N` rows.

## Appendix — one-time host setup (if the shared tree is missing)

```bash
# checkout
git clone --branch feature/fixed-chunk-grid \
  https://github.com/ikindacodeabit/benchmark.git $BASE/fixedgrid/repo

# venvs: build per ../../../REPRODUCE.md (Python 3.12).
#   serving venv  .venv          : pip install -e ".[eval]" ; then pip install "transformers==4.51.3" LAST
#   sub-call venv .venv-kvpress  : pip install -e ".[eval]" (keeps transformers >= 4.56) ; torch 2.6.0

# model
HF_HOME=$BASE/hf_cache huggingface-cli download Qwen/Qwen3-4B-Instruct-2507

# data pack (needs the standard LongBench dump at $RLM_DATA_DIR/longbench.jsonl)
RLM_DATA_DIR=$BASE/rlm_data "$KVPY" -m evaluation.benchmarks.longbench128k.build_dataset --per-subset 60
#   -> $RLM_DATA_DIR/longbench128k/<subset>/{data.parquet,build_manifest.json}
```
