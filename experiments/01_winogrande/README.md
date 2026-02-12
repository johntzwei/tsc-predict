# Experiment 01: WinoGrande Contamination Signals

## Research Question
Can a model developer who intentionally contaminates test set examples predict memorization from model behavior, enabling correction of benchmark scores or estimation of OOD accuracy?

## Setup
- **Models:** Hubble 1B/100B standard (clean) vs perturbed (contaminated)
- **Data:** WinoGrande perturbation datasets (infill + mcq formats), ~8K examples each with known duplication levels {0, 1, 4, 16, 64, 256}
- **Eval:** Loss-based choice (standard zero-shot WinoGrande eval: score suffix log-prob conditioned on prefix+option)

## Phase A: Per-example inference (`run.py`)
For every example x both models: compute per-option log-likelihood, accuracy, and confidence. Save to parquet with duplication metadata.

## Outputs
- `results/per_example_signals.parquet` — per-example results with duplication labels

## Observations

*Quick analysis done with Claude. Findings are preliminary and should be verified more carefully.*

**We reproduce the Hubble results on WinoGrande.** The perturbed model shows increasing accuracy gains on contaminated infill examples as duplication count rises (dup=0: -4.8pp, dup=4: +3.6pp, dup=16: +15.7pp, dup=64: +24.5pp, dup=256: +31.5pp). Contamination does not transfer to MCQ format — the perturbed model actually performs *worse* across all MCQ duplication levels.

**Per-example scatter plots** (standard vs perturbed log-prob of correct answer) show the cloud of points shifting above the x=y diagonal at higher duplication levels for infill, consistent with the perturbed model assigning higher log-probs to memorized correct answers.

**Outliers at dup=0.** We noticed ~16 dup=0 infill examples where the standard model has low log-prob (< -6) but the perturbed model assigns much higher log-prob (> -3). A quick trigram overlap check against high-duplication examples showed very low similarity (most < 15%), suggesting these are not textual near-duplicates of contaminated data. We also verified that none of these examples have contaminated WinoGrande twins (via `paired_orig_idx`). These outliers appear to be training stochasticity on hard examples (16/4000 = 0.4%), though we have not ruled out other explanations (e.g., shared structural patterns, broader distributional shift from the contaminated training mix).
