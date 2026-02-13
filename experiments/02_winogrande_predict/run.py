"""Probe hidden states of the perturbed model to predict contamination (dup > 0)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hubble.probes import PROBES, HiddenStateProbe, FinetuneProbe
from hubble.eval import load_model

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXP01_RESULTS = Path(__file__).parent.parent / "01_winogrande" / "results"
MODEL_ID = "allegrolab/hubble-1b-100b_toks-perturbed-hf"
SEED = 42


def main():
    probes = [
        (name, cls()) for name, cls in PROBES.items()
        if not issubclass(cls, FinetuneProbe)
    ]

    # --- Stage 1: Load infill data ---
    df = pd.read_parquet(EXP01_RESULTS / "per_example_signals.parquet")
    df_infill = df[df["format"] == "infill"].reset_index(drop=True)
    texts = df_infill["text"].tolist()
    dup_counts = df_infill["duplicates"].values
    labels = (dup_counts > 0).astype(int)
    print(f"Loaded {len(df_infill)} infill examples")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Stage 2: Train and evaluate each probe ---
    model, tokenizer = None, None
    all_results = {}
    for name, probe in probes:
        print(f"\n{'='*60}")
        print(f"Probe: {name}")

        cache_path = RESULTS_DIR / f"hidden_states_{probe.feature_key}.npz"
        if not cache_path.exists() and model is None:
            print(f"Loading model: {MODEL_ID}")
            model, tokenizer = load_model(MODEL_ID)
        X = probe.extract_features(model, tokenizer, texts, cache_path=cache_path)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, labels))
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegressionCV(
            Cs=10, cv=5, solver="lbfgs", max_iter=1000,
            scoring="accuracy", random_state=SEED, n_jobs=-1,
        )
        clf.fit(X[train_idx], y_train)

        y_pred_test = clf.predict(X[test_idx])
        y_prob_test = clf.predict_proba(X[test_idx])[:, 1]
        train_acc = accuracy_score(y_train, clf.predict(X[train_idx]))
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"Accuracy: train={train_acc:.4f}, test={test_acc:.4f}")
        print(classification_report(y_test, y_pred_test, target_names=["clean", "contaminated"]))

        all_results[name] = {
            "probe": name,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_clean": int((labels == 0).sum()),
            "n_contaminated": int((labels == 1).sum()),
        }

        np.savez(
            RESULTS_DIR / f"test_predictions_{name}.npz",
            test_idx=test_idx,
            y_test=y_test,
            y_pred=y_pred_test,
            y_prob=y_prob_test,
            dup_counts=dup_counts[test_idx],
        )

    if model is not None:
        del model, tokenizer
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()