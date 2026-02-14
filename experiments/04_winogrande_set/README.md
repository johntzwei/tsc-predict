# Experiment 04: WinoGrande Set-Level Signals

## Research Question
At the **set level**, do standard and perturbed models produce similar accuracy on uncontaminated (dups=0) example sets? This serves as a sanity check before building set-level contamination detection methods.

## Setup
- **Data:** Per-example signals from experiment 01 (`per_example_signals.parquet`), filtered to dups=0, infill format
- **Models:** Hubble 1B and 8B, standard and perturbed variants, at 100B and 500B token scales
- **Method:** Sample 500 random subsets of n=2000 examples; compute set-level accuracy and sum logprob(correct) for both standard and perturbed models; scatter plot standard (x) vs perturbed (y) with y=x reference line

## Outputs

- `figures/set_accuracy_scatter_{1b,8b}.png` — 2×2 grid per model size: top row = accuracy, bottom row = sum logprob(correct); columns = 100B / 500B token scales
- `figures/set_scale_scatter.png` — 2×2 grid comparing 100B (x) vs 500B (y) with standard (blue) and perturbed (red) overlaid; rows = 1B / 8B, columns = accuracy / logprob(correct) sum
- Agreement table printed to stdout

## Observations

*Quick analysis done with Claude. Findings are preliminary.*

**Set-level accuracy** shows low correlation between standard and perturbed models at dups=0, because accuracy binarizes per-example confidence — borderline examples dominate the variance and models disagree on those.

**Set-level logprob(correct) sum** shows much stronger correlation (r~0.74 for 1B, higher for 8B), since it preserves continuous confidence information.

**8B models** show a larger standard-perturbed accuracy gap (~9pp at 100B, ~4pp at 500B) compared to 1B (~5pp at 100B, ~5pp at 500B), but higher pairwise agreement (0.72 / 0.80 vs 0.67 / 0.70).
