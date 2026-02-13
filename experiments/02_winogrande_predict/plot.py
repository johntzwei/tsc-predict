"""Plot probe accuracy by duplication count for all probes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    # Discover all probe prediction files
    pred_files = sorted(RESULTS_DIR.glob("test_predictions_*.npz"))
    probe_names = [f.stem.removeprefix("test_predictions_") for f in pred_files]

    if not probe_names:
        print("No prediction files found. Run run.py first.")
        return

    # Compute per-dup-count accuracy for each probe
    # Use first file to get the dup count categories
    first = np.load(pred_files[0])
    unique_dups = sorted(np.unique(first["dup_counts"]))
    dup_labels = [str(d) for d in unique_dups]

    # Sample counts (same across probes since split is deterministic)
    sample_counts = [int((first["dup_counts"] == d).sum()) for d in unique_dups]

    probe_accs = {}
    for name, path in zip(probe_names, pred_files):
        data = np.load(path)
        y_test, y_pred, dups = data["y_test"], data["y_pred"], data["dup_counts"]
        probe_accs[name] = [
            accuracy_score(y_test[dups == d], y_pred[dups == d]) for d in unique_dups
        ]

    # Grouped bar chart
    n_probes = len(probe_names)
    n_dups = len(unique_dups)
    x = np.arange(n_dups)
    width = 0.8 / n_probes

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(probe_names):
        offset = (i - (n_probes - 1) / 2) * width
        ax.bar(x + offset, probe_accs[name], width, label=name)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Duplication count")
    ax.set_ylabel("Accuracy (predicting dup > 0)")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(dup_labels, sample_counts)])
    ax.set_title("Probe accuracy by duplication count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_by_dup_count.png", dpi=150)
    print(f"Figure saved to {FIGURES_DIR / 'accuracy_by_dup_count.png'}")

    # --- AUC-ROC by dup count ---
    # For each dup count d > 0, compute AUC-ROC of clean (dup=0) vs dup=d
    contaminated_dups = [d for d in unique_dups if d > 0]
    clean_idx_first = first["dup_counts"] == 0

    probe_aucs = {}
    for name, path in zip(probe_names, pred_files):
        data = np.load(path)
        y_prob, dups = data["y_prob"], data["dup_counts"]
        aucs = []
        for d in contaminated_dups:
            mask = (dups == 0) | (dups == d)
            y_binary = (dups[mask] > 0).astype(int)
            aucs.append(roc_auc_score(y_binary, y_prob[mask]))
        probe_aucs[name] = aucs

    n_cats = len(contaminated_dups)
    x2 = np.arange(n_cats)
    contam_counts = [int((first["dup_counts"] == d).sum()) for d in contaminated_dups]
    clean_n = int(clean_idx_first.sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(probe_names):
        offset = (i - (n_probes - 1) / 2) * width
        ax.bar(x2 + offset, probe_aucs[name], width, label=name)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Duplication count")
    ax.set_ylabel("AUC-ROC (clean vs dup=d)")
    ax.set_ylim(0, 1)
    ax.set_xticks(x2)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(
        [str(d) for d in contaminated_dups], contam_counts)])
    ax.set_title(f"AUC-ROC: clean (n={clean_n}) vs each duplication level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "auc_roc_by_dup_count.png", dpi=150)
    print(f"Figure saved to {FIGURES_DIR / 'auc_roc_by_dup_count.png'}")


if __name__ == "__main__":
    main()
