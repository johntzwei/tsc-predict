"""Probe hidden states of the perturbed model to predict duplication count."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hubble.probes import get_probe
from hubble.eval import load_model
from hubble.features import extract_hidden_states

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXP01_RESULTS = Path(__file__).parent.parent / "01_winogrande" / "results"
MODEL_ID = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
PROBE_NAME = "logreg_last"
SEED = 42


def main():
    probe = get_probe(PROBE_NAME)

    # --- Stage 1: Load infill data ---
    df = pd.read_parquet(EXP01_RESULTS / "per_example_signals.parquet")
    df_infill = df[df["format"] == "infill"].reset_index(drop=True)
    texts = df_infill["text"].tolist()
    labels = df_infill["duplicates"].values
    print(f"Loaded {len(df_infill)} infill examples")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Stage 2: Extract hidden states (cached per feature config) ---
    cache_path = RESULTS_DIR / f"hidden_states_{probe.feature_key}.npz"
    if cache_path.exists():
        print(f"Loading cached hidden states from {cache_path}")
        X = np.load(cache_path)["hidden_states"]
        assert len(X) == len(labels), f"Cache size mismatch: {len(X)} vs {len(labels)}"
    else:
        print(f"Loading model: {MODEL_ID}")
        model, tokenizer = load_model(MODEL_ID)
        print(f"Extracting hidden states (layer={probe.layer}, pool={probe.pool})...")
        X = extract_hidden_states(model, tokenizer, texts, batch_size=32, layer=probe.layer, pool=probe.pool)
        np.savez(cache_path, hidden_states=X)
        print(f"Cached hidden states to {cache_path} (shape={X.shape})")
        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"Features shape: {X.shape}")

    # --- Stage 3: Stratified 50/50 split ---
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, labels))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # --- Stage 4: Train probe ---
    print(f"Training probe: {PROBE_NAME}")
    clf = probe.make_classifier(seed=SEED)
    clf.fit(X_train, y_train)

    # --- Stage 5: Evaluate ---
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    class_labels = sorted(np.unique(labels))
    class_names = [str(c) for c in class_labels]

    print(f"\n6-class accuracy: train={train_acc:.4f}, test={test_acc:.4f}")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # Binary: contaminated (dup > 0) vs clean (dup == 0)
    y_test_bin = (y_test > 0).astype(int)
    y_pred_bin = (y_pred_test > 0).astype(int)
    bin_acc = accuracy_score(y_test_bin, y_pred_bin)
    print(f"Binary accuracy (dup>0 vs ==0): {bin_acc:.4f}")

    # Save results
    cm = confusion_matrix(y_test, y_pred_test, labels=class_labels)
    results = {
        "probe": PROBE_NAME,
        "train_acc_6class": train_acc,
        "test_acc_6class": test_acc,
        "binary_acc": bin_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    for c in class_labels:
        mask = y_test == c
        results[f"acc_class_{c}"] = float(accuracy_score(y_test[mask], y_pred_test[mask]))
        results[f"n_class_{c}"] = int(mask.sum())

    with open(RESULTS_DIR / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(RESULTS_DIR / "confusion_matrix.npy", cm)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
