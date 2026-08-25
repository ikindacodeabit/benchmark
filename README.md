All completed dataset benchmarks, memory-budget sweeps, model variants, metrics, and downloadable prediction files are available here:

[**Open the benchmark results dashboard**](https://effortless-cupcake-470e7a.netlify.app/)

This repository also hosts a Recursive Language Model (RLM) baseline alongside the KVPress
presses, including an arm where RLM sub-calls read their context slice through a compressed
KV cache. See [**RLM.md**](RLM.md) for how the two fit together and how sub-call chunk sizes
are derived from a KV memory budget.
