# Experiment 06: MMLU Set-Level Signals

## Research Question
Same as exp 04 but for MMLU: do standard and perturbed models produce correlated accuracy on uncontaminated (dups=0) example sets?

## Setup
- **Data:** Per-example signals from experiment 05 (`per_example_signals.parquet`), filtered to dups=0
- **Models:** Same as exp 05
- **Method:** Sample 500 random subsets of n=500; compute set-level accuracy and sum logprob(correct)

## Outputs
- `figures/set_accuracy_scatter_{1b,8b}.png` — standard vs perturbed set-level signals
- `figures/set_accuracy_scatter_interference.png` — standard vs interference
- `figures/set_scale_scatter.png` — 100B vs 500B comparison
- Agreement table and set-level correlations printed to stdout

## Key Findings

### Set-level correlations (n=500, 500 subsets)

**Standard vs Perturbed:**
| Model | Accuracy r | logprob(correct) r |
|-------|-----------|-------------------|
| 1B/100B | 0.070 | -0.039 |
| 1B/500B | -0.060 | 0.179 |
| 8B/100B | -0.050 | -0.043 |
| **8B/500B** | **0.466** | **0.487** |

**100B vs 500B:** All r < 0.15 (essentially zero correlation).

### Interpretation

Nearly zero correlation everywhere except 8B/500B, because most models are at chance (~25%). At chance level, predictions are essentially random — there's no shared difficulty structure to correlate on.

The 8B/500B pair (50% vs 45% accuracy) shows moderate correlation (r~0.47), consistent with it being the only pair where both models have above-chance performance.

### Comparison with WinoGrande (exp 04)

| Comparison | WinoGrande Accuracy r | MMLU Accuracy r |
|-----------|----------------------|-----------------|
| Standard vs Perturbed (best) | 0.30 | 0.47 (8B/500B only) |
| Standard vs Perturbed (typical) | 0.25 | ~0 |
| 100B vs 500B (standard) | 0.42-0.50 | 0.05-0.08 |

WinoGrande is far more suitable for demonstrating the contamination adjustment: models are well above chance across all configurations, and the difficulty structure is shared.

MMLU's value is as a **negative control** — showing that contamination doesn't always inflate scores, and the adjustment method correctly produces no signal when there's nothing to adjust.
