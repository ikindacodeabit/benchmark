"""Prepare or offline-validate the 32K Synthetic-KV Hugging Face dataset."""

from pathlib import Path
import runpy
import sys


SHARED_PREPARER = Path(__file__).resolve().parents[1] / "prepare_synthetic_hf_cache.py"
sys.argv = [str(SHARED_PREPARER), "--variant", "32k", *sys.argv[1:]]
runpy.run_path(str(SHARED_PREPARER), run_name="__main__")
