"""Plot WinoGrande accuracy by duplication level."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def print_summary(df):
    print("=" * 80)
    print("WinoGrande Accuracy Summary (1B/100B)")
    print("=" * 80)

    for fmt in ["infill", "mcq"]:
        sub = df[df["format"] == fmt]
        print(f"\n--- {fmt.upper()} format ---")
        print(f"{'dup':>5}  {'n':>5}  {'standard':>10}  {'perturbed':>10}  {'delta':>10}  {'pct_flipped':>12}")

        for dup, g in sub.groupby("duplicates"):
            n = len(g)
            acc_s = g["acc_standard"].mean()
            acc_p = g["acc_perturbed"].mean()
            delta = acc_p - acc_s
            # Examples that flipped from wrong->right or right->wrong
            flipped_pos = ((g["acc_perturbed"] == 1) & (g["acc_standard"] == 0)).sum()
            flipped_neg = ((g["acc_perturbed"] == 0) & (g["acc_standard"] == 1)).sum()
            print(f"{dup:>5}  {n:>5}  {acc_s:>10.4f}  {acc_p:>10.4f}  {delta:>+10.4f}  {flipped_pos:>4}+ / {flipped_neg:>3}-")

        # Overall
        acc_s = sub["acc_standard"].mean()
        acc_p = sub["acc_perturbed"].mean()
        print(f"{'all':>5}  {len(sub):>5}  {acc_s:>10.4f}  {acc_p:>10.4f}  {acc_p - acc_s:>+10.4f}")

    # Cross-format comparison: examples contaminated in infill, how do they do in mcq?
    print(f"\n--- Cross-format: contaminated (dup>0) examples ---")
    for fmt in ["infill", "mcq"]:
        contaminated = df[(df["format"] == fmt) & (df["duplicates"] > 0)]
        clean = df[(df["format"] == fmt) & (df["duplicates"] == 0)]
        print(f"  {fmt}: contaminated acc_perturbed={contaminated['acc_perturbed'].mean():.4f} "
              f"vs clean acc_perturbed={clean['acc_perturbed'].mean():.4f} "
              f"(delta={contaminated['acc_perturbed'].mean() - clean['acc_perturbed'].mean():+.4f})")

    # Confidence shift
    print(f"\n--- Mean confidence (correct answer) by duplication ---")
    print(f"{'dup':>5}  {'std_infill':>12}  {'pert_infill':>12}  {'std_mcq':>12}  {'pert_mcq':>12}")
    for dup in sorted(df["duplicates"].unique()):
        vals = []
        for fmt in ["infill", "mcq"]:
            for model in ["standard", "perturbed"]:
                mask = (df["format"] == fmt) & (df["duplicates"] == dup)
                vals.append(df.loc[mask, f"confidence_{model}"].mean())
        print(f"{dup:>5}  {vals[0]:>12.4f}  {vals[1]:>12.4f}  {vals[2]:>12.4f}  {vals[3]:>12.4f}")

    print()


def plot_accuracy_by_duplicates():
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")

    print_summary(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, fmt in zip(axes, ["infill", "mcq"]):
        sub = df[df["format"] == fmt]
        grouped = sub.groupby("duplicates")[["acc_standard", "acc_perturbed"]]
        means = grouped.mean()
        counts = grouped.count()["acc_standard"]
        for model, color, marker in [("acc_standard", "tab:blue", "o"), ("acc_perturbed", "tab:red", "s")]:
            p = means[model]
            n = counts
            se = np.sqrt(p * (1 - p) / n)
            label = model.replace("acc_", "").capitalize()
            ax.errorbar(means.index.astype(str), p, yerr=1.96 * se, fmt=f"{marker}-", label=label, color=color, capsize=3)

        ax.set_xlabel("Duplication count")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"WinoGrande ({fmt})")
        ax.legend()
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("WinoGrande Accuracy by Contamination Level (1B/100B)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_by_duplication.png", dpi=150)
    print(f"Saved to {FIGURES_DIR / 'accuracy_by_duplication.png'}")
    plt.close(fig)


def plot_logprob_scatter():
    """Scatter: standard vs perturbed log-prob of correct answer, grid by dup level."""
    df = pd.read_parquet(RESULTS_DIR / "per_example_signals.parquet")

    # Derive log-prob of the correct option for each model
    for model in ["standard", "perturbed"]:
        df[f"logprob_correct_{model}"] = np.where(
            df["answer"] == 1,
            df[f"logprob_option1_{model}"],
            df[f"logprob_option2_{model}"],
        )

    df = df[df["format"] == "infill"]

    dup_levels = sorted(df["duplicates"].unique())  # [0, 1, 4, 16, 64, 256]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)

    # Shared axis limits
    all_vals = pd.concat([df["logprob_correct_standard"], df["logprob_correct_perturbed"]])
    lo, hi = all_vals.quantile(0.001), all_vals.quantile(0.999)
    pad = (hi - lo) * 0.05

    for ax, dup in zip(axes.flat, dup_levels):
        sub = df[df["duplicates"] == dup]
        ax.scatter(
            sub["logprob_correct_standard"],
            sub["logprob_correct_perturbed"],
            c="tab:blue", alpha=0.3, s=8, edgecolors="none",
        )
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.4, linewidth=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"dup={dup}  (n={len(sub)})")
        ax.set_aspect("equal")

    # Axis labels on edge subplots only
    for ax in axes[-1]:
        ax.set_xlabel("Standard log-prob (correct)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Perturbed log-prob (correct)")

    fig.suptitle("Standard vs Perturbed Log-Prob of Correct Answer (Infill) by Duplication Level", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "logprob_scatter_by_duplication.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_accuracy_by_duplicates()
    plot_logprob_scatter()
