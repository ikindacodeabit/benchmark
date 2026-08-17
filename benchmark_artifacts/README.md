# Local benchmark artifacts

This directory contains machine- and cluster-specific files that are not part
of the KVPress source tree. The whole directory is ignored by Git.

```text
benchmark_artifacts/
├── results/       # Predictions, metrics, saved run configs, and result summaries
├── slurm_jobs/    # Slurm submission scripts grouped by dataset and task/context
├── yml/           # Evaluation YAML files grouped like slurm_jobs
├── logs/          # Historical scheduler output and running logs
└── notebooks/     # One-off local diagnostics and Colab experiments
```

Use the hierarchy:

```text
<artifact-type>/<dataset>/<context-or-task>/<model-or-run>/
```

Keep implementation code, tests, and canonical benchmark loaders in their
original repository locations. New generated files should go only into this
directory.
