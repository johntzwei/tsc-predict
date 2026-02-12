# allegro-project

Research project template for the Allegro lab.

## Setup

```bash
# 1. Install uv (if not already installed)
bash install_uv.sh

# 2. Create venv and install all dependencies
uv sync

# 3. (Optional) Install dev dependencies
uv sync --extra dev
```

## Usage

```bash
# Run an experiment (uv run handles venv activation automatically)
uv run python experiments/00_example/run.py

# Submit to SLURM
sbatch slurm/run_gpu.sbatch experiments/00_example/run.py
sbatch slurm/run_preempt.sbatch experiments/00_example/run.py

# Submit an array of experiments (runs 00, 01, 02)
# Each experiment can read SLURM_ARRAY_TASK_ID from the environment
sbatch --array=0-2 slurm/run_array.sbatch

# Run tests
uv run pytest
```

## Project Structure

```text
src/allegro/       # Shared library code (reusable modules with argparse)
experiments/       # Numbered experiment scripts (hardcoded params, version-controlled)
slurm/             # SLURM job templates
tests/             # Tests
```

## Experiments

Each experiment lives in a numbered folder under `experiments/` with its own `README.md` describing the research question, setup, and findings. See [experiments/README.md](experiments/README.md) for an overview of all experiments.

Create new folders for new experiments — don't edit old ones.

## Experiment Tracking

Uses Weights & Biases. Disable with:

```bash
export WANDB_MODE=disabled
```
