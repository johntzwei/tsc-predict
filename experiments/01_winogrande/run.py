"""Compute per-example WinoGrande accuracies on Hubble 1B and 8B standard and perturbed models.

Supports parallel execution via SLURM array jobs:
  sbatch --array=0-7 slurm/run_gpu.sbatch experiments/01_winogrande/run.py
Each task evaluates one model and saves a per-model cache.

Sequential mode evaluates any models without a cached result:
  uv run python experiments/01_winogrande/run.py
"""

import os
from pathlib import Path

import pandas as pd

from hubble.data import load_winogrande_perturbations
from hubble.eval import load_model, evaluate_winogrande_df

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = [
    ("1b_standard_100b", "allegrolab/hubble-1b-100b_toks-standard-hf"),
    ("1b_perturbed_100b", "allegrolab/hubble-1b-100b_toks-perturbed-hf"),
    ("1b_standard_500b", "allegrolab/hubble-1b-500b_toks-standard-hf"),
    ("1b_perturbed_500b", "allegrolab/hubble-1b-500b_toks-perturbed-hf"),
    ("8b_standard_100b", "allegrolab/hubble-8b-100b_toks-standard-hf"),
    ("8b_perturbed_100b", "allegrolab/hubble-8b-100b_toks-perturbed-hf"),
    ("8b_standard_500b", "allegrolab/hubble-8b-500b_toks-standard-hf"),
    ("8b_perturbed_500b", "allegrolab/hubble-8b-500b_toks-perturbed-hf"),
    ("1b_interference_100b", "allegrolab/hubble-1b-100b_toks-interference_testset-hf"),
]


def load_data():
    print("Loading WinoGrande perturbation data...")
    df_infill = load_winogrande_perturbations("infill")
    df_mcq = load_winogrande_perturbations("mcq")
    df = pd.concat([df_infill, df_mcq], ignore_index=True)
    print(f"Total examples: {len(df)} ({len(df_infill)} infill, {len(df_mcq)} mcq)")
    return df


def evaluate_single(df, label, model_id):
    """Evaluate one model and save cache."""
    cache_path = RESULTS_DIR / f"eval_{label}.parquet"
    if cache_path.exists():
        print(f"Cache already exists for {label}, skipping.")
        return

    print(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id)

    print(f"Evaluating {len(df)} examples on {label}...")
    df = evaluate_winogrande_df(model, tokenizer, df, label)

    df.to_parquet(cache_path)
    print(f"Cached {label} results to {cache_path}")


def main():
    df = load_data()
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id is not None:
        task_id = int(task_id)
        label, model_id = MODELS[task_id]
        print(f"Array task {task_id}: evaluating {label}")
        evaluate_single(df, label, model_id)
    else:
        for label, model_id in MODELS:
            cache_path = RESULTS_DIR / f"eval_{label}.parquet"
            if not cache_path.exists():
                evaluate_single(df, label, model_id)
                import torch
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
