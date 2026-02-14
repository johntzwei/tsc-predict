# Adjusting Contaminated Benchmark Scores at the Set Level

The [formulation writeup](01_formulation.md) defined two goals. This writeup develops Goal 1 as a **set-level prediction problem**: given a contaminated model's outputs and probe signals, can we recover the uncontaminated accuracy?

## The Set-Level Prediction Problem

### Setup

We observe a contaminated model evaluated on test set $S = \{x_1, \ldots, x_n\}$. For each example we have:
- $C_i$: the model's correctness (binary, or continuous margin)
- $\hat{M}_i$: the probe's memorization estimate

The contaminated accuracy is $\text{Acc}_\text{cont}(S) = \frac{1}{n} \sum_i C_i$. We want to predict the uncontaminated accuracy: what a standard (clean) model would achieve on the same test set.

### Empirical baseline: standard vs perturbed correlation on clean data

We first check: on truly uncontaminated examples (dup=0), how well does the perturbed model's accuracy track the standard model's accuracy at the set level? We repeatedly sample random subsets of $n = 100$ examples from the dup=0 pool, compute each model's accuracy on each subset, and correlate across 500 draws.

**WinoGrande (exp 04):**

| Comparison | Accuracy $r$ | logprob(correct) $r$ |
|-----------|-------------|---------------------|
| Std vs Pert, 1B/100B | 0.30 | 0.68 |
| Std vs Pert, 1B/500B | 0.25 | 0.75 |
| Std vs Pert, 8B/100B | 0.25 | 0.63 |
| Std vs Pert, 8B/500B | 0.27 | 0.75 |

**MMLU (exp 06):** Near-zero correlation everywhere except 8B/500B ($r \approx 0.47$), because most models are at chance (~25%).

The accuracy correlations on random subsets are low ($r \approx 0.25$-$0.30$ for WinoGrande). This seems discouraging — but the low correlation is an artifact of the evaluation method, not a fundamental limitation.

## Why Random Subsets Underestimate the Signal

### The CLT problem

When we draw random subsets of size $n$ from a pool of examples, each subset's mean accuracy clusters tightly around the population mean. By the Central Limit Theorem, the standard deviation of the subset mean is $\sqrt{p(1-p)/n}$. For WinoGrande with $p \approx 0.69$ and $n = 100$, this is about 4.6pp.

Intuitively, subset means concentrate in a tight cloud around the population mean, and correlation requires a trend line — which a tight cloud cannot support.

### The key insight: variation must come from difficulty, not sampling

Random subsets lack meaningful identity — they're all "average difficulty" up to noise. For the adjustment to be testable, we need subsets that span a genuine range of difficulty levels.

## Constructing Informative Test Sets

### Approach: probe-stratified subsets

The memorization probe $\hat{M}(x)$ assigns each example a continuous score. If we sort examples by probe score and form blocks, we get subsets ranging from "barely memorized" to "heavily memorized." These subsets will naturally vary in:

1. **Contaminated accuracy**: heavily-memorized subsets should have inflated accuracy
2. **Uncontaminated accuracy**: the underlying difficulty also varies (easy examples may be both easier to learn and more likely to be memorized)

The adjustment task is then: given the contaminated accuracy and probe signal for each subset, predict the uncontaminated accuracy. This is the natural evaluation for the probe-based adjustment method.

## What the Adjustment Needs

For a probe-based set-level adjustment, we need:

1. **Per-example memorization estimates** $\hat{M}(x)$ from the probe (exp 02-03 infrastructure exists)
2. **A model of the contamination effect**: how much does memorization inflate accuracy, on average?
3. **The adjustment**: for each example, subtract the estimated inflation to get counterfactual correctness, then average over the test set

The simplest adjustment is linear: $\hat{C}_0(x) = C(x) - \beta \cdot \hat{M}(x)$, where $\beta$ is the estimated per-unit contamination effect. More flexible versions could use the full dose-response curve from v1.
