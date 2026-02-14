# Experiments

Each numbered folder is a self-contained experiment with its own `README.md`, `run.py`, and output directories. See individual READMEs for details.

| # | Name | Description |
|---|------|-------------|
| 00 | [00_example](00_example/) | Template/example experiment |
| 01 | [01_winogrande](01_winogrande/) | WinoGrande contamination signals — compares standard vs perturbed Hubble models on infill and MCQ formats across duplication levels |
| 02 | [02_winogrande_predict](02_winogrande_predict/) | Probing hidden states to predict duplication count with linear probes on the perturbed model |
| 03 | [03_winogrande_finetune](03_winogrande_finetune/) | Finetuning probes (LoRA/full) to predict duplication count |
| 04 | [04_winogrande_set](04_winogrande_set/) | Set-level sanity check — standard vs perturbed accuracy and logprob(correct) on random subsets of uncontaminated examples |
