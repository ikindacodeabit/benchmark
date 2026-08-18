"""Download RULER 64K into the Hugging Face cache for offline compute nodes.

Run this script once on a login node with internet access.  It respects the
standard Hugging Face cache environment variables, including ``HF_HOME``.
The compute-node job must use the same cache location.
"""

from datasets import load_dataset


DATASET_ID = "ollamaweights/Ruler-64k"
CONFIG_NAME = "65536"
SPLIT = "test"


def main() -> None:
    print(f"Caching {DATASET_ID}, config={CONFIG_NAME}, split={SPLIT}...")
    dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=SPLIT)
    print(f"Cached {len(dataset)} samples successfully.")

    cache_files = dataset.cache_files
    if cache_files:
        print("Cached data files:")
        for cache_file in cache_files:
            print(f"  {cache_file['filename']}")


if __name__ == "__main__":
    main()
