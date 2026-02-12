# Experiment 02: Predicting Duplication Count from Hidden States

## Research Question
Can a linear probe on the perturbed model's hidden state representations predict how many times an example was duplicated in training?

## Setup
- **Model:** Hubble 1B/100B perturbed (contaminated) only
- **Data:** WinoGrande infill examples (8,001), using the `text` field (what was contaminated)
- **Features:** Last-layer hidden states, mean-pooled over tokens
- **Labels:** Duplication count (6-class: 0, 1, 4, 16, 64, 256)
- **Split:** Stratified 50/50 by duplication count (seed=42)
- **Probes:** Registered in `src/hubble/probes.py` — logistic regression variants with optional scaling and class balancing

## Pipeline (`run.py`)
1. Load infill data from experiment 01 results
2. Extract hidden states from perturbed model (cached per feature config)
3. Stratified train/test split
4. Train probe, evaluate 6-class and binary (dup > 0) accuracy

## Outputs
- `results/hidden_states_*.npz` — cached hidden states per feature config
- `results/probe_results.json` — accuracy metrics
- `results/confusion_matrix.npy` — confusion matrix for plotting
