# Why Would You Intentionally Contaminate Your Test Set?

## Setup

A model developer intentionally includes benchmark examples in their training data at known rates (e.g., each example inserted 0, 1, 4, 16, ..., 256 times). Because the developer controls the training data, they control the degree of memorization for each example.

**Assumption**: all memorization to a given degree is equivalent — an example memorized to degree $M = m$ behaves the same regardless of how it got there. By inserting examples at varying rates, the developer creates support across the full range of memorization levels.

For each benchmark example $x$, two quantities are relevant:

- **Memorization** $M$: the degree to which the model memorized this example (controlled via insertion count)
- **Correctness** $C$: the model's performance on this example (observed from model output)

The basic question: can the developer use this information to estimate the model's out-of-distribution performance — what the benchmark score would have been without contamination?

## Motivation

A naive estimate of uncontaminated performance is $\mathbb{E}[C \mid M = 0]$ — just use the 0-insertion group. But this assumes those examples are truly clean, when some may have been memorized through the base pretraining corpus. And even if the average is right, it tells us nothing at the example level.

By training probes on hidden states — a **memorization probe** and a **correctness probe** — we make our best attempt to learn from insertion counts as labels, despite the label noise (0-insertion examples may not truly be clean). We can then reexamine the test set, including the 0-insertion group, at the individual example level: is a given example's performance inflated due to memorization?

## Formalizing Correctness and Memorization

### Correctness as margin

Binary correctness ($C \in \{0, 1\}$) discards useful signal. A model that assigns 51% probability to the correct answer and one that assigns 99% are both "correct," but the former is far more fragile.

For tasks like WinoGrande, a natural continuous measure is the **log-probability margin**: the difference in average per-token log-probabilities between the correct and incorrect options. This is directly computable from model outputs. Equivalently, one can use the **confidence**: the softmax probability assigned to the correct option.

### Memorization as degree

Under our assumption, $M$ is controlled by the developer via insertion count. We train a memorization probe on hidden states using insertion counts as labels, giving a continuous, per-example estimate of memorization degree.

The training labels are noisy — 0-insertion examples are labeled $M = 0$ but some may be contaminated through pretraining. Despite this, we train the best probe we can, and then use it to reexamine the test set: which examples (including 0-insertion ones) show signs of memorization?

### The dose-response view

With continuous $C$ and $M$, the law of total probability gives:

$$\mathbb{E}[C \mid x] = \int_0^1 \mathbb{E}[C \mid M = m, x]\, p(m \mid x) \, dm$$

The key object is $\mathbb{E}[C \mid M = m, x]$: a **dose-response curve** describing how the correctness margin changes as memorization degree increases.

### What is observed

| Quantity | Observed? | Source |
| --- | --- | --- |
| $M$ (memorization) | Yes | Controlled via insertion count |
| $C$ (correctness / margin) | Yes | Model logits |

## Estimands

The formalism above defines the machinery. Here we state what we actually want to estimate.

### Goal 1: Adjusted benchmark score

The observed benchmark score is inflated by memorization. We want the score the model *would have achieved* if no example had been memorized:

$$\text{Acc}_{\text{adj}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}[C_i \mid M_i = 0]$$

This averages the counterfactual correctness at zero memorization over *all* examples — not just the 0-insertion group. The key difference from the naive $\mathbb{E}[C \mid M = 0]$ (restricting to 0-insertion examples) is that we also adjust examples in the 0-insertion group that may have been memorized through pretraining.

Estimation approaches:

- **Dose-response extrapolation**: fit $\mathbb{E}[C \mid M = m]$ across insertion groups and evaluate at $m = 0$. This pools across examples, so it yields an aggregate adjustment.
- **Probe-based regression**: train a correctness probe on hidden states, conditioning on (or residualizing out) memorization probe scores. This can yield per-example estimates that are then averaged.

### Goal 2: Per-example counterfactual correctness

For each example $x$, we want to answer: *would the model get this example correct even without memorization?*

$$\hat{C}_0(x) = \mathbb{E}[C \mid M = 0, x]$$

This is a per-example quantity. It separates examples into:

- **Robust**: high $\hat{C}_0(x)$ — the model would get this right regardless of memorization.
- **Memorization-dependent**: high observed $C$ but low $\hat{C}_0(x)$ — correctness is inflated by memorization.
- **Hard**: low $\hat{C}_0(x)$ and low observed $C$ — the model genuinely struggles.

Estimating $\hat{C}_0(x)$ requires a model of how correctness varies with memorization *at the example level*. Two strategies:

- **Correctness probe with memorization control**: predict $C$ from hidden states while partialing out the memorization probe score. The residual captures the example's "intrinsic" difficulty to the model.
- **Matched comparisons**: for examples that appear at multiple insertion levels (or have near-neighbors), directly observe how correctness changes with dose. This is limited by data but provides a nonparametric check.

## Notes

- The equivalence assumption may not hold exactly — memorization through pretraining vs. intentional insertion could differ. The probe's ability to generalize across these is an empirical question.
- A **correctness probe** (predicting margin from hidden states) could estimate the counterfactual $\mathbb{E}[C \mid M = 0, x]$ — what the margin would be for a memorized example if it hadn't been memorized. Open question: hidden states already encode memorization, so what "controlling for $M$" means in this context requires care.
- The example-level formulation defines the quantities, but estimation of the dose-response curve requires pooling across examples at the dataset level.
