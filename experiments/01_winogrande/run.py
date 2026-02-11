"""Compute per-example WinoGrande accuracies on Hubble 1B/100B standard and perturbed models."""

import os
from pathlib import Path

import pandas as pd

from hubble.data import load_winogrande_perturbations
from hubble.eval import load_model, evaluate_winogrande_df

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "standard": "allegrolab/hubble-1b-100b_toks-standard-hf",
    "perturbed": "allegrolab/hubble-1b-100b_toks-perturbed-hf",
}


def main():
    # Load perturbation data for both formats
    print("Loading WinoGrande perturbation data...")
    df_infill = load_winogrande_perturbations("infill")
    df_mcq = load_winogrande_perturbations("mcq")
    df = pd.concat([df_infill, df_mcq], ignore_index=True)
    print(f"Total examples: {len(df)} ({len(df_infill)} infill, {len(df_mcq)} mcq)")
    print(f"Duplication distribution:\n{df.groupby(['format', 'split'])['duplicates'].value_counts().sort_index()}")

    # Evaluate each model sequentially (load one at a time)
    for label, model_id in MODELS.items():
        cache_path = RESULTS_DIR / f"eval_{label}.parquet"
        if cache_path.exists():
            print(f"Loading cached results for {label} from {cache_path}")
            cached = pd.read_parquet(cache_path)
            # Merge cached columns into df
            new_cols = [c for c in cached.columns if c.endswith(f"_{label}")]
            for col in new_cols:
                df[col] = cached[col]
            continue

        print(f"\nLoading model: {model_id}")
        model, tokenizer = load_model(model_id)

        print(f"Evaluating {len(df)} examples on {label} model...")
        df = evaluate_winogrande_df(model, tokenizer, df, label)

        # Cache per-model results
        df.to_parquet(cache_path)
        print(f"Cached {label} results to {cache_path}")

        # Free GPU memory before loading next model
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()

    # Save final combined results
    out_path = RESULTS_DIR / "per_example_signals.parquet"
    df.to_parquet(out_path)
    print(f"\nSaved results to {out_path}")

    # Print summary stats
    for label in MODELS:
        for fmt in ["infill", "mcq"]:
            mask = df["format"] == fmt
            acc = df.loc[mask, f"acc_{label}"].mean()
            print(f"{label}/{fmt}: accuracy = {acc:.4f} (n={mask.sum()})")
            # By duplication level
            by_dup = df.loc[mask].groupby("duplicates")[f"acc_{label}"].mean()
            print(f"  By duplication: {by_dup.to_dict()}")


if __name__ == "__main__":
    main()
