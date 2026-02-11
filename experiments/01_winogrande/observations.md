# 01_winogrande — Observations

*Quick analysis done with Claude. Findings are preliminary and should be verified more carefully.*

## Setup

- Models: `hubble-1b-100b_toks-standard-hf` (standard) vs `hubble-1b-100b_toks-perturbed-hf` (perturbed)
- Task: WinoGrande (infill format), suffix log-probability evaluation
- Duplication levels: 0, 1, 4, 16, 64, 256
- 8,001 infill examples, 8,001 mcq examples

## Key observations

**We reproduce the Hubble results on WinoGrande.** The perturbed model shows increasing accuracy gains on contaminated infill examples as duplication count rises (dup=0: -4.8pp, dup=4: +3.6pp, dup=16: +15.7pp, dup=64: +24.5pp, dup=256: +31.5pp). Contamination does not transfer to MCQ format — the perturbed model actually performs *worse* across all MCQ duplication levels.

**Per-example scatter plots** (standard vs perturbed log-prob of correct answer) show the cloud of points shifting above the x=y diagonal at higher duplication levels for infill, consistent with the perturbed model assigning higher log-probs to memorized correct answers.

**Outliers at dup=0.** We noticed ~16 dup=0 infill examples where the standard model has low log-prob (< -6) but the perturbed model assigns much higher log-prob (> -3). A quick trigram overlap check against high-duplication examples showed very low similarity (most < 15%), suggesting these are not textual near-duplicates of contaminated data. We also verified that none of these examples have contaminated WinoGrande twins (via `paired_orig_idx`). These outliers appear to be training stochasticity on hard examples (16/4000 = 0.4%), though we have not ruled out other explanations (e.g., shared structural patterns, broader distributional shift from the contaminated training mix).
