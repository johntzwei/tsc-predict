# Experiment 04: WinoGrande Set-Level Signals

## Research Question
At the **set level**, do standard and perturbed models produce similar accuracy on uncontaminated (dups=0) example sets? This serves as a sanity check before building set-level contamination detection methods.

## Setup
- **Data:** Per-example signals from experiment 01 (`per_example_signals.parquet`), filtered to dups=0, infill format
- **Method:** Sample 500 random subsets of n=2000 examples; compute set-level accuracy and sum logprob(correct) for both standard and perturbed models; scatter plot standard (x) vs perturbed (y) with y=x reference line

## Outputs
- `figures/set_accuracy_scatter.png` — 2×2 grid: top row = accuracy, bottom row = sum logprob(correct); columns = 100B / 500B token scales
- Agreement table printed to stdout

## Observations

*Quick analysis done with Claude. Findings are preliminary.*

**Set-level accuracy** (top row) shows moderate correlation (r~0.25) between standard and perturbed models at dups=0. Points sit below y=x, reflecting a ~5pp baseline accuracy gap (standard > perturbed) even on clean examples.

**Set-level logprob(correct) sum** (bottom row) shows much stronger correlation (r~0.74), since it preserves continuous confidence information rather than binarizing to correct/incorrect.

**Instance-level agreement** between standard and perturbed models on dups=0 infill: 67% at 100B, 70% at 500B. More training data increases agreement, likely because both models converge toward stable predictions on learnable examples.
