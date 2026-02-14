"""Plot WinoGrande set-level accuracy scatter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TOKEN_SCALES = ["100b", "500b"]


def plot_set_accuracy_scatter(n_subsets=500, subset_size=2000, seed=42):
    """Scatter: standard vs perturbed set-level signals on random subsets (dups=0 only)."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")
    df = df[(df["duplicates"] == 0) & (df["format"] == "infill")]

    # Precompute per-example logprob(correct)
    for scale in TOKEN_SCALES:
        for variant in ["standard", "perturbed"]:
            model = f"{variant}_{scale}"
            lp1 = f"logprob_option1_{model}"
            lp2 = f"logprob_option2_{model}"
            if lp1 not in df.columns:
                continue
            df[f"lp_correct_{model}"] = np.where(
                df["answer"] == 1, df[lp1], df[lp2],
            )

    metrics = [
        ("acc", "Accuracy", "mean", ".3f"),
        ("lp_correct", "logprob(correct) sum", "sum", ".1f"),
    ]

    fig, axes = plt.subplots(len(metrics), 2, figsize=(10, 5 * len(metrics)))

    for row, (prefix, label, agg, fmt) in enumerate(metrics):
        rng = np.random.default_rng(seed)
        for col, scale in enumerate(TOKEN_SCALES):
            ax = axes[row, col]
            std_col = f"{prefix}_standard_{scale}"
            pert_col = f"{prefix}_perturbed_{scale}"
            if std_col not in df.columns:
                ax.set_visible(False)
                continue

            std_vals, pert_vals = [], []
            for _ in range(n_subsets):
                idx = rng.choice(len(df), size=subset_size, replace=False)
                batch = df.iloc[idx]
                fn = batch[std_col].mean if agg == "mean" else batch[std_col].sum
                std_vals.append(fn())
                fn = batch[pert_col].mean if agg == "mean" else batch[pert_col].sum
                pert_vals.append(fn())
            std_vals = np.array(std_vals)
            pert_vals = np.array(pert_vals)

            ax.scatter(std_vals, pert_vals, alpha=0.3, s=12, edgecolors="none")
            lo = min(std_vals.min(), pert_vals.min())
            hi = max(std_vals.max(), pert_vals.max())
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.5, linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")

            r = np.corrcoef(std_vals, pert_vals)[0, 1]
            ax.set_title(f"{label}, {scale.upper()}  (r={r:.3f})")
            ax.text(0.05, 0.95,
                    f"std={std_vals.mean():{fmt}}\npert={pert_vals.mean():{fmt}}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            ax.set_xlabel(f"Standard {label}")

        axes[row, 0].set_ylabel(f"Perturbed {label}")

    fig.suptitle(f"Set-Level: Standard vs Perturbed (infill, dups=0, n={subset_size}, {n_subsets} subsets)", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "set_accuracy_scatter.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


def print_agreement_table():
    """Print accuracy and pairwise agreement for standard/perturbed models (dups=0, infill)."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")
    df = df[(df["duplicates"] == 0) & (df["format"] == "infill")]

    models = [f"{v}_{s}" for s in TOKEN_SCALES for v in ["standard", "perturbed"]]
    acc_cols = [f"acc_{m}" for m in models]

    # Per-model accuracy
    print(f"{'Model':<25} {'Accuracy':>8}  {'N':>5}")
    print("-" * 42)
    for m, col in zip(models, acc_cols):
        print(f"{m:<25} {df[col].mean():>8.4f}  {len(df):>5}")

    # Pairwise agreement
    print(f"\n{'Pairwise Agreement':<25}", end="")
    for m in models:
        print(f" {m:>12}", end="")
    print()
    print("-" * (25 + 13 * len(models)))
    for m1, c1 in zip(models, acc_cols):
        print(f"{m1:<25}", end="")
        for m2, c2 in zip(models, acc_cols):
            agree = (df[c1] == df[c2]).mean()
            print(f" {agree:>12.4f}", end="")
        print()


if __name__ == "__main__":
    plot_set_accuracy_scatter()
    print_agreement_table()
