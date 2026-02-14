"""Combine per-model eval caches and plot WinoGrande results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hubble.data import load_winogrande_perturbations

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_SIZES = ["1b", "8b"]
TOKEN_SCALES = ["100b", "500b"]

MODELS = [
    f"{size}_{variant}_{scale}"
    for size in MODEL_SIZES
    for variant in ["standard", "perturbed"]
    for scale in TOKEN_SCALES
]


def load_combined():
    """Combine per-model eval caches into a single DataFrame, caching the result."""
    out_path = RESULTS_DIR / "per_example_signals.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    print("Combining per-model caches...")
    df_infill = load_winogrande_perturbations("infill")
    df_mcq = load_winogrande_perturbations("mcq")
    df = pd.concat([df_infill, df_mcq], ignore_index=True)

    for label in MODELS:
        cache_path = RESULTS_DIR / f"eval_{label}.parquet"
        if not cache_path.exists():
            print(f"WARNING: missing cache for {label}, skipping")
            continue
        cached = pd.read_parquet(cache_path)
        new_cols = [c for c in cached.columns if c.endswith(f"_{label}")]
        for col in new_cols:
            df[col] = cached[col]

    df.to_parquet(out_path)
    print(f"Saved combined results to {out_path}")
    return df


def plot_accuracy_by_duplicates():
    df = load_combined()

    LINES_BY_SIZE = {
        "1b": [
            ("acc_1b_standard_100b", "Standard 100B", "tab:blue", "o", "--"),
            ("acc_1b_perturbed_100b", "Perturbed 100B", "tab:red", "o", "--"),
            ("acc_1b_standard_500b", "Standard 500B", "tab:blue", "s", "-"),
            ("acc_1b_perturbed_500b", "Perturbed 500B", "tab:red", "s", "-"),
        ],
        "8b": [
            ("acc_8b_standard_100b", "Standard 100B", "tab:blue", "o", "--"),
            ("acc_8b_perturbed_100b", "Perturbed 100B", "tab:red", "o", "--"),
            ("acc_8b_standard_500b", "Standard 500B", "tab:blue", "s", "-"),
            ("acc_8b_perturbed_500b", "Perturbed 500B", "tab:red", "s", "-"),
        ],
    }

    sub = df[df["format"] == "infill"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, size in zip(axes, MODEL_SIZES):
        for col, label, color, marker, ls in LINES_BY_SIZE[size]:
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
        ax.set_title(f"WinoGrande Infill ({size.upper()})")
        ax.legend()
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("WinoGrande Accuracy by Contamination Level (Infill)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "accuracy_by_duplication.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


def plot_logprob_scatter():
    """Scatter: standard vs perturbed log-prob of correct answer, grid by dup level."""
    df = load_combined()

    for size in MODEL_SIZES:
        for scale in TOKEN_SCALES:
            std_lp1 = f"logprob_option1_{size}_standard_{scale}"
            if std_lp1 not in df.columns:
                print(f"Skipping {size}/{scale} scatter (columns not found)")
                continue

            for variant in ["standard", "perturbed"]:
                model = f"{size}_{variant}_{scale}"
                df[f"logprob_correct_{model}"] = np.where(
                    df["answer"] == 1,
                    df[f"logprob_option1_{model}"],
                    df[f"logprob_option2_{model}"],
                )

            sub = df[df["format"] == "infill"]
            std_key = f"logprob_correct_{size}_standard_{scale}"
            pert_key = f"logprob_correct_{size}_perturbed_{scale}"

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

            fig.suptitle(f"Standard vs Perturbed Log-Prob of Correct Answer (Infill, {size.upper()}/{scale.upper()})", fontsize=12)
            fig.tight_layout()
            out = FIGURES_DIR / f"logprob_scatter_by_duplication_{size}_{scale}.png"
            fig.savefig(out, dpi=150)
            print(f"Saved to {out}")
            plt.close(fig)


if __name__ == "__main__":
    plot_accuracy_by_duplicates()
    plot_logprob_scatter()