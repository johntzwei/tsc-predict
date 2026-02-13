"""Finetune models to predict contamination (dup > 0) using the unified probe interface."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from hubble.eval import load_model
from hubble.probes import FinetuneProbe, PROBES

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXP01_RESULTS = Path(__file__).parent.parent / "01_winogrande" / "results"
MODEL_ID = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
SEED = 42

# Finetuning probes to run (registry names or custom configs)
PROBE_CONFIGS = [
    ("final_layer_lora", PROBES["final_layer_lora"](epochs=10)),
    ("final_layer_full_finetune", PROBES["final_layer_full_finetune"](epochs=10)),
]


def main():
    # --- Stage 1: Load data ---
    df = pd.read_parquet(EXP01_RESULTS / "per_example_signals.parquet")
    df_infill = df[df["format"] == "infill"].reset_index(drop=True)
    texts = df_infill["text"].tolist()
    dup_counts = df_infill["duplicates"].values
    labels = (dup_counts > 0).astype(int)
    print(f"Loaded {len(df_infill)} infill examples")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Stage 2: Split ---
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # --- Stage 3: Load model and train each probe ---
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load_model(MODEL_ID)

    all_results = {}
    for name, probe in PROBE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Probe: {name}")

        probe.fit(texts_train, y_train, model, tokenizer, cache_dir=RESULTS_DIR, seed=SEED)

        y_pred_test = probe.predict(texts_test, model, tokenizer, cache_dir=RESULTS_DIR)
        y_prob_test = probe.predict_proba(texts_test, model, tokenizer, cache_dir=RESULTS_DIR)[:, 1]
        y_pred_train = probe.predict(texts_train, model, tokenizer, cache_dir=RESULTS_DIR)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"Accuracy: train={train_acc:.4f}, test={test_acc:.4f}")
        print(classification_report(y_test, y_pred_test, target_names=["clean", "contaminated"]))

        results = {
            "probe": name,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        all_results[name] = results

        np.savez(
            RESULTS_DIR / f"test_predictions_{name}.npz",
            test_idx=test_idx,
            y_test=y_test,
            y_pred=y_pred_test,
            y_prob=y_prob_test,
            dup_counts=dup_counts[test_idx],
        )

        # Clean up probe state to free GPU memory before next probe
        if hasattr(probe, "_inference_model") and probe._inference_model is not None:
            del probe._inference_model, probe._head
            probe._inference_model = None
            probe._head = None
            torch.cuda.empty_cache()

    del model, tokenizer
    torch.cuda.empty_cache()

    with open(RESULTS_DIR / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
