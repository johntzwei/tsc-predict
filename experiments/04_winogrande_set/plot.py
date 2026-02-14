"""Plot WinoGrande set-level accuracy scatter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_SIZES = ["1b", "8b"]
TOKEN_SCALES = ["100b", "500b"]


def plot_set_accuracy_scatter(n_subsets=500, subset_size=2000, seed=42):
    """Scatter: standard vs perturbed set-level signals on random subsets (dups=0 only)."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")
    df = df[(df["duplicates"] == 0) & (df["format"] == "infill")]

    # Precompute per-example logprob(correct)
    for size in MODEL_SIZES:
        for scale in TOKEN_SCALES:
            for variant in ["standard", "perturbed"]:
                model = f"{size}_{variant}_{scale}"
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

    for size in MODEL_SIZES:
        fig, axes = plt.subplots(len(metrics), len(TOKEN_SCALES),
                                 figsize=(5 * len(TOKEN_SCALES), 5 * len(metrics)))

        for row, (prefix, label, agg, fmt) in enumerate(metrics):
            for col, scale in enumerate(TOKEN_SCALES):
                ax = axes[row, col]
                std_col = f"{prefix}_{size}_standard_{scale}"
                pert_col = f"{prefix}_{size}_perturbed_{scale}"
                if std_col not in df.columns:
                    ax.set_visible(False)
                    continue

                rng = np.random.default_rng(seed)
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

        fig.suptitle(f"Set-Level: Standard vs Perturbed ({size.upper()}, infill, dups=0, n={subset_size}, {n_subsets} subsets)", fontsize=12)
        fig.tight_layout()
        out = FIGURES_DIR / f"set_accuracy_scatter_{size}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved to {out}")
        plt.close(fig)


def plot_set_scale_scatter(n_subsets=500, subset_size=2000, seed=42):
    """Scatter: 100B vs 500B set-level signals, rows = model size, cols = metric."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")
    df = df[(df["duplicates"] == 0) & (df["format"] == "infill")]

    # Precompute per-example logprob(correct)
    for size in MODEL_SIZES:
        for scale in TOKEN_SCALES:
            for variant in ["standard", "perturbed"]:
                model = f"{size}_{variant}_{scale}"
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

    fig, axes = plt.subplots(len(MODEL_SIZES), len(metrics),
                             figsize=(5 * len(metrics), 5 * len(MODEL_SIZES)))

    for row, size in enumerate(MODEL_SIZES):
        for col, (prefix, label, agg, fmt) in enumerate(metrics):
            ax = axes[row, col]
            colors = {"standard": "tab:blue", "perturbed": "tab:red"}

            for variant, color in colors.items():
                col_100b = f"{prefix}_{size}_{variant}_100b"
                col_500b = f"{prefix}_{size}_{variant}_500b"
                if col_100b not in df.columns:
                    continue

                rng = np.random.default_rng(seed)
                vals_100b, vals_500b = [], []
                for _ in range(n_subsets):
                    idx = rng.choice(len(df), size=subset_size, replace=False)
                    batch = df.iloc[idx]
                    fn = batch[col_100b].mean if agg == "mean" else batch[col_100b].sum
                    vals_100b.append(fn())
                    fn = batch[col_500b].mean if agg == "mean" else batch[col_500b].sum
                    vals_500b.append(fn())
                vals_100b = np.array(vals_100b)
                vals_500b = np.array(vals_500b)

                r = np.corrcoef(vals_100b, vals_500b)[0, 1]
                ax.scatter(vals_100b, vals_500b, alpha=0.3, s=12, edgecolors="none",
                           color=color, label=f"{variant} (r={r:.3f})")

            ax.set_xlabel(f"100B {label}")
            ax.set_ylabel(f"500B {label}")
            ax.set_title(f"{label}, {size.upper()}")
            ax.legend(fontsize=8)

            # y=x line across full data range
            lims = ax.get_xlim() + ax.get_ylim()
            lo, hi = min(lims), max(lims)
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")

    fig.suptitle(f"Set-Level: 100B vs 500B (infill, dups=0, n={subset_size}, {n_subsets} subsets)", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "set_scale_scatter.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


def print_agreement_table():
    """Print accuracy and pairwise agreement for standard/perturbed models (dups=0, infill)."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")
    df = df[(df["duplicates"] == 0) & (df["format"] == "infill")]

    models = [f"{size}_{v}_{s}" for size in MODEL_SIZES for s in TOKEN_SCALES for v in ["standard", "perturbed"]]
    acc_cols = [f"acc_{m}" for m in models]

    # Filter to models that exist in the data
    existing = [(m, c) for m, c in zip(models, acc_cols) if c in df.columns]
    models, acc_cols = zip(*existing) if existing else ([], [])

    # Per-model accuracy
    print(f"{'Model':<30} {'Accuracy':>8}  {'N':>5}")
    print("-" * 47)
    for m, col in zip(models, acc_cols):
        print(f"{m:<30} {df[col].mean():>8.4f}  {len(df):>5}")

    # Pairwise agreement
    print(f"\n{'Pairwise Agreement':<30}", end="")
    for m in models:
        print(f" {m:>15}", end="")
    print()
    print("-" * (30 + 16 * len(models)))
    for m1, c1 in zip(models, acc_cols):
        print(f"{m1:<30}", end="")
        for m2, c2 in zip(models, acc_cols):
            agree = (df[c1] == df[c2]).mean()
            print(f" {agree:>15.4f}", end="")
        print()


if __name__ == "__main__":
    plot_set_accuracy_scatter()
    plot_set_scale_scatter()
    print_agreement_table()
