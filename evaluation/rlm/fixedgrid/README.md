# Fixed-chunk × KV-retention grid

This campaign asks whether, at a fixed logical KV-retention budget, reading a
larger chunk through KVzip beats reading a smaller chunk without compression.
Rows are retained-token budgets (`1024 2048 4096 8192`); columns are chunk
factors (`1 2 4 8 16`), so every cell feeds `N = budget × factor` context tokens
to the sub-model. The `1×` cells use `no_press` on the otherwise identical local
generation path.

`KVzipPress` masks evicted KV but does not free it. Every cell physically holds
the full `N`-token cache, regardless of its row budget. These are simulated
retention budgets and must not be reported as measured GPU-memory savings.

Build the local six-subset dataset pack:

```bash
python -m evaluation.benchmarks.longbench128k.build_dataset --per-subset 60
```

With a Qwen3-4B root already served on port 8000, run a pilot in its own
results root:

```bash
DATASETS=hotpotqa LIMIT=10 GPU=1 PORT=8000 \
  RESULTS=/mnt/nas/$USER/fixedgrid/pilot \
  bash evaluation/rlm/fixedgrid/run_grid.sh
```

Use a different shared `RESULTS` directory for the `LIMIT=50` campaign. Limit
is resume-critical, so pilot checkpoints must not be reused as full-grid
checkpoints.

## Six-host deployment

The intended workers are `ant.cse.iitb.ac.in`, `bee.cse.iitb.ac.in`,
`cat.cse.iitb.ac.in`, `dog.cse.iitb.ac.in`, `elk.cse.iitb.ac.in`, and
`fox.cse.iitb.ac.in`. Each host is an independent lane: start or reuse a local
vLLM root server on one GPU, then point `GPU` at a different GPU for the KVpress
sub-model. `PORT` is local to that host; the driver never starts or stops the
server.

Keep the checkout, dataset pack, model cache, and `RESULTS` under the persistent
`/mnt/nas/$USER` tree. In particular, every lane must use the same absolute
`RESULTS` path. Locks live at `$RESULTS/.locks/<dataset>.<budget>.<factor>.lock`,
so concurrent hosts claim different cells and a completed `metrics.json` makes
a restarted lane skip the cell.

For the full campaign, a simple initial assignment is one subset per host:

| Host | `DATASETS` |
| --- | --- |
| ant | `musique` |
| bee | `hotpotqa` |
| cat | `2wikimqa` |
| dog | `narrativeqa` |
| elk | `qasper` |
| fox | `triviaqa` |

Additional lanes can safely run the same subset if they receive disjoint
`BUDGETS` or `FACTORS`; the cell lock remains the final duplicate-run guard.
Before launching, reconcile each VM checkout deliberately (some may have dirty
trees), verify that the shared dataset has all six 60-row subsets, warm the model
cache, and make the Transformers/KVpress environment offline-capable. Do not
pull, stash, or discard VM changes from the campaign script.

Render completed results:

```bash
python evaluation/rlm/fixedgrid/grid_table.py \
  --results evaluation/results/fixedgrid
```

Before expanding a pilot, require `sub_context_tokens_on_target_fraction` near
1, `sub_pressed_call_fraction == 1` outside the `1×` control, the realized factor
near the requested column, and `sub_slice_unlocatable_calls` near zero. Read
`document_coverage_fraction` beside quality so a miss caused by incomplete reads
is not misdiagnosed as compression damage.
