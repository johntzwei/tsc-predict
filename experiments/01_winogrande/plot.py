"""Plot WinoGrande accuracy by duplication level."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TOKEN_SCALES = ["100b", "500b"]



def plot_accuracy_by_duplicates():
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")

    LINES = [
        ("acc_standard_100b", "Standard 100B", "tab:blue", "o", "--"),
        ("acc_perturbed_100b", "Perturbed 100B", "tab:red", "o", "--"),
        ("acc_standard_500b", "Standard 500B", "tab:blue", "s", "-"),
        ("acc_perturbed_500b", "Perturbed 500B", "tab:red", "s", "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, fmt in zip(axes, ["infill", "mcq"]):
        sub = df[df["format"] == fmt]
        for col, label, color, marker, ls in LINES:
            if col not in df.columns:
                continue
            grouped = sub.groupby("duplicates")[col]
            p = grouped.mean()
            n = grouped.count()
            se = np.sqrt(p * (1 - p) / n)
            ax.errorbar(p.index.astype(str), p, yerr=1.96 * se,
                        fmt=f"{marker}", linestyle=ls, label=label, color=color, capsize=3)

        ax.set_xlabel("Duplication count")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"WinoGrande ({fmt})")
        ax.legend()
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("WinoGrande Accuracy by Contamination Level", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "accuracy_by_duplication.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


def plot_logprob_scatter():
    """Scatter: standard vs perturbed log-prob of correct answer, grid by dup level."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")

    for scale in TOKEN_SCALES:
        std_lp1 = f"logprob_option1_standard_{scale}"
        if std_lp1 not in df.columns:
            print(f"Skipping {scale} scatter (columns not found)")
            continue

        for model in [f"standard_{scale}", f"perturbed_{scale}"]:
            df[f"logprob_correct_{model}"] = np.where(
                df["answer"] == 1,
                df[f"logprob_option1_{model}"],
                df[f"logprob_option2_{model}"],
            )

        sub = df[df["format"] == "infill"]
        std_key = f"logprob_correct_standard_{scale}"
        pert_key = f"logprob_correct_perturbed_{scale}"

        dup_levels = sorted(sub["duplicates"].unique())
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)

        all_vals = pd.concat([sub[std_key], sub[pert_key]])
        lo, hi = all_vals.quantile(0.001), all_vals.quantile(0.999)
        pad = (hi - lo) * 0.05

        for ax, dup in zip(axes.flat, dup_levels):
            s = sub[sub["duplicates"] == dup]
            ax.scatter(s[std_key], s[pert_key], c="tab:blue", alpha=0.3, s=8, edgecolors="none")
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.4, linewidth=0.8)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"dup={dup}  (n={len(s)})")
            ax.set_aspect("equal")

        for ax in axes[-1]:
            ax.set_xlabel("Standard log-prob (correct)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Perturbed log-prob (correct)")

        fig.suptitle(f"Standard vs Perturbed Log-Prob of Correct Answer (Infill, 1B/{scale.upper()})", fontsize=12)
        fig.tight_layout()
        out = FIGURES_DIR / f"logprob_scatter_by_duplication_{scale}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved to {out}")
        plt.close(fig)


if __name__ == "__main__":
    plot_accuracy_by_duplicates()
    plot_logprob_scatter()