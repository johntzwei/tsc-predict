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
