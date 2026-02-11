"""Plot probe results from saved JSON."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    with open(RESULTS_DIR / "probe_results.json") as f:
        results = json.load(f)

    class_labels = [0, 1, 4, 16, 64, 256]
    class_names = [str(c) for c in class_labels]

    # 1. Confusion matrix (counts) — needs saved predictions, load from npz
    cm_path = RESULTS_DIR / "confusion_matrix.npy"
    if cm_path.exists():
        cm = np.load(cm_path)

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Predicting Duplication Count from Hidden States")
        ax.set_xlabel("Predicted duplication count")
        ax.set_ylabel("True duplication count")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        # 2. Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format=".2f")
        ax.set_title("Normalized Confusion Matrix (row-normalized)")
        ax.set_xlabel("Predicted duplication count")
        ax.set_ylabel("True duplication count")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "confusion_matrix_normalized.png", dpi=150)
        plt.close(fig)

    # 3. Per-class accuracy bar chart
    per_class_acc = [results[f"acc_class_{c}"] for c in class_labels]
    n_per_class = [results.get(f"n_class_{c}", 0) for c in class_labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, per_class_acc, color="steelblue", edgecolor="black")
    ax.axhline(1 / len(class_labels), color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("True duplication count")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Probe Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, c, n in zip(bars, class_labels, n_per_class):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_accuracy.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
