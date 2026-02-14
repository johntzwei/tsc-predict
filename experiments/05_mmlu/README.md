# Experiment 05: MMLU Contamination Signals

## Research Question
Does intentional test set contamination affect MMLU accuracy in Hubble models? MMLU is a complementary benchmark to WinoGrande: knowledge-based MCQ (4 choices, 25% chance) vs. commonsense reasoning (2 choices, 50% chance).

## Setup
- **Data:** `allegrolab/testset_mmlu` — 8001 examples (4001 train with dup=1-256, 4000 test with dup=0), 57 subjects
- **Evaluation:** Zero-shot, single-letter logprob scoring matching lm-eval-harness standard `mmlu` task. Score " A"/" B"/" C"/" D" as continuations of the formatted prompt.
- **Prompt format** matches lm-eval-harness and the Hubble `text` field exactly:
  ```
  The following are multiple choice questions (with answers) about {subject}.

  {question}
  A. {choice_a}
  B. {choice_b}
  C. {choice_c}
  D. {choice_d}
  Answer:
  ```
- **Models:** Same 9 as exp 01 (1B/8B x standard/perturbed x 100B/500B + 1B interference)

## Running
```bash
# Parallel via SLURM array (9 models)
sbatch --array=0-8 slurm/run_gpu.sbatch experiments/05_mmlu/run.py

# Sequential (evaluates uncached models one at a time)
uv run python experiments/05_mmlu/run.py

# Generate plots (after all models cached)
uv run python experiments/05_mmlu/plot.py
```

## Outputs
- `results/eval_{label}.parquet` — per-model caches with logprobs and accuracy
- `results/per_example_signals.parquet` — combined signals across all models
- `figures/accuracy_by_duplication.png` — accuracy vs duplication level
- `figures/logprob_scatter_by_duplication_{size}_{scale}.png` — standard vs perturbed logprob(correct)

## Key Findings

### Most models are at chance (~25%)

| Model | Accuracy (dups=0) |
|-------|------------------|
| 1B/standard/100B | 0.250 |
| 1B/perturbed/100B | 0.250 |
| 1B/standard/500B | 0.256 |
| 1B/perturbed/500B | 0.264 |
| 8B/standard/100B | 0.249 |
| 8B/perturbed/100B | 0.257 |
| **8B/standard/500B** | **0.503** |
| **8B/perturbed/500B** | **0.453** |

Only 8B/500B achieves above-chance performance. The 5pp standard-perturbed gap at 8B/500B is the only meaningful contamination signal.

### Contrast with WinoGrande
WinoGrande models are well above chance across all configurations (64-85%), with clear contamination effects. MMLU models (except 8B/500B) can't exploit the memorized content — likely because MMLU requires more knowledge than what these smaller/less-trained models have.

### Implications
MMLU serves as a **negative control**: contamination doesn't always inflate scores. The model needs sufficient capability for memorized content to translate into accuracy gains. This is an interesting finding for the contamination adjustment story — the adjustment should have no effect when there's no signal, and the MMLU results confirm this.
