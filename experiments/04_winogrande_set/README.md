# Experiment 04: WinoGrande Set-Level Signals

## Research Question
At the **set level**, do standard and perturbed models produce similar accuracy on uncontaminated (dups=0) example sets? This serves as a sanity check before building set-level contamination adjustment methods.

## Setup
- **Data:** Per-example signals from experiment 01 (`per_example_signals.parquet`), filtered to dups=0, infill format
- **Models:** Hubble 1B and 8B, standard and perturbed variants, at 100B and 500B token scales; also 1B interference
- **Method:** Sample 500 random subsets of n=100 examples; compute set-level accuracy and sum logprob(correct) for both models; scatter plot with y=x reference line

## Outputs

- `figures/set_accuracy_scatter_{1b,8b}.png` — 2x2 grid per model size: rows = accuracy / logprob(correct) sum; columns = 100B / 500B
- `figures/set_accuracy_scatter_interference.png` — standard vs interference (1B/100B)
- `figures/set_scale_scatter.png` — 100B vs 500B comparison
- Agreement table and set-level correlations printed to stdout

## Key Findings

### Set-level correlations (n=100, 500 subsets)

**Standard vs Perturbed:**
| Model | Accuracy r | logprob(correct) r |
|-------|-----------|-------------------|
| 1B/100B | 0.296 | 0.677 |
| 1B/500B | 0.245 | 0.746 |
| 8B/100B | 0.247 | 0.626 |
| 8B/500B | 0.273 | 0.753 |

**100B vs 500B (same training data, different scale):**
| Model | Accuracy r | logprob(correct) r |
|-------|-----------|-------------------|
| 1B/standard | 0.422 | 0.973 |
| 1B/perturbed | 0.183 | 0.762 |
| 8B/standard | 0.497 | 0.972 |
| 8B/perturbed | 0.237 | 0.741 |

### Observations

**Accuracy correlations are low (r~0.25-0.30) for random subsets.** This is consistent with the per-example agreement (~77% for 8B/500B), because chance agreement is already ~55% at these accuracy levels. The phi coefficient is only ~0.26, which matches the set-level r.

**logprob(correct) correlations are much higher (r~0.63-0.75)** because logprobs are continuous and preserve per-example difficulty signal through aggregation.

**Why random subsets have low accuracy correlation:** Random subsets of size n have means that cluster tightly around the population mean (CLT). Most between-subset variation is sampling noise, which is independent across models. The shared signal (question difficulty) is small relative to this noise for binary accuracy.

### Implications for Contamination Adjustment

The low accuracy correlation on random subsets does *not* mean the models disagree on difficulty — it means random subsets don't have enough genuine difficulty variation to reveal the agreement. The relationship exists but is masked by sampling noise.

For the adjustment to work well on accuracy, we need subsets with **real difficulty variation**, not random draws. Two approaches:
1. **Natural groupings** (e.g., by topic/category) that have genuine difficulty differences
2. **Probe-stratified subsets** — sort by memorization probe score and form blocks. This is the natural fit for the adjustment method: probes create the variation, and the adjustment should track the probe-induced difficulty gradient.

The probe-based approach is both more relevant to the use case and produces a wider accuracy range, making the adjustment relationship clearer.
