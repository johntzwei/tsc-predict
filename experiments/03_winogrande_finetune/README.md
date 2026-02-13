# Experiment 03: Finetuning Probes to Predict Contamination

## Research Question
Does end-to-end finetuning (LoRA or full) improve contamination detection over frozen linear probes (exp 02)?

## Setup
- **Model:** Hubble 1B/100B perturbed (contaminated)
- **Data:** WinoGrande infill examples (8,001), binary labels (dup > 0)
- **Split:** Stratified 50/50 train/test (seed=42)
- **Probes:** `LoRAFinetuneProbe`, `FullFinetuneProbe` from `src/hubble/probes.py`

## Training Parameters (defaults in `FinetuneProbe`)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `layer` | -1 | Last hidden layer |
| `pool` | mean | Mean-pooled hidden states |
| `lr` | 1e-4 | Base model LR; classification head uses 10x (1e-3) |
| `epochs` | 3 | |
| `batch_size` | 16 | |
| `num_classes` | 2 | Binary: clean vs contaminated |

### LoRA-specific
| Parameter | Value |
|-----------|-------|
| `lora_r` | 8 |
| `lora_alpha` | 16 |
| `lora_target_modules` | `["q_proj", "v_proj"]` |
| `lora_dropout` | 0.05 |

### Full finetune
Deepcopies the model (~4GB extra for 1B in fp32). No additional hyperparameters.

## Pipeline (`run.py`)
1. Load infill data from experiment 01 results
2. Stratified train/test split
3. Load model, train each probe (HF Trainer, checkpointed)
4. Evaluate train/test accuracy, save predictions

## Outputs
- `results/checkpoint_*` — saved model checkpoints (LoRA adapter or full weights + head)
- `results/test_predictions_*.npz` — per-example predictions and probabilities
- `results/probe_results.json` — accuracy metrics
