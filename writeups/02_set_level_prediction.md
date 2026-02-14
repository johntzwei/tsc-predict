# Adjusting Contaminated Benchmark Scores at the Set Level

## Recap

A developer intentionally inserts benchmark examples into training data at known duplication levels (0, 1, 4, 16, 64, 256). They train memorization probes on hidden states to estimate per-example memorization degree $\hat{M}(x)$. The goal is to use these probes to adjust the model's benchmark score — predicting what accuracy the model would achieve if the test set were uncontaminated.

v1 formalized the per-example quantities: memorization $M$, correctness $C$, and the dose-response curve $\mathbb{E}[C \mid M = m, x]$. This writeup focuses on the **set-level** problem: given a test set $S$, can we predict its uncontaminated accuracy from the contaminated model's behavior plus probe signals?

## The Set-Level Prediction Problem

### Setup

We observe a contaminated model evaluated on test set $S = \{x_1, \ldots, x_n\}$. For each example we have:
- $C_i$: the model's correctness (binary, or continuous margin)
- $\hat{M}_i$: the probe's memorization estimate

The contaminated accuracy is $\text{Acc}_\text{cont}(S) = \frac{1}{n} \sum_i C_i$. We want to predict the uncontaminated accuracy: what a standard (clean) model would achieve on the same test set.

### Why is this hard?

The standard and perturbed models share the same architecture and base training data — they differ only in whether benchmark examples were inserted. On uncontaminated examples (dup=0), both models should perform similarly. The question is whether this similarity is strong enough to make set-level prediction useful.

### Empirical baseline: standard vs perturbed correlation on clean data

We first check: on truly uncontaminated examples (dup=0), how well does the perturbed model's accuracy track the standard model's accuracy at the set level?

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

The between-subset variation is dominated by sampling noise, which is independent across models. The shared signal — genuine difficulty variation — is small relative to this noise. So even though the two models agree on 77% of individual examples, the set-level accuracy correlation is modest.

### Per-example agreement vs set-level correlation

The per-example agreement rate includes a large "chance agreement" component. With accuracies of $p_A = 0.69$ and $p_B = 0.64$, chance agreement (if predictions were independent) is:

$$p_A p_B + (1 - p_A)(1 - p_B) \approx 0.55$$

The observed 77% agreement implies a phi coefficient of only ~0.26, consistent with the set-level $r \approx 0.29$. The models *do* share difficulty structure, but it's modest in absolute terms for binary accuracy.

### The key insight: variation must come from difficulty, not sampling

Random subsets lack meaningful identity — they're all "average difficulty" up to noise. For the adjustment to be testable, we need subsets that span a genuine range of difficulty levels.

## Constructing Informative Test Sets

### Approach: probe-stratified subsets

The memorization probe $\hat{M}(x)$ assigns each example a continuous score. If we sort examples by probe score and form blocks, we get subsets ranging from "barely memorized" to "heavily memorized." These subsets will naturally vary in:

1. **Contaminated accuracy**: heavily-memorized subsets should have inflated accuracy
2. **Uncontaminated accuracy**: the underlying difficulty also varies (easy examples may be both easier to learn and more likely to be memorized)

The adjustment task is then: given the contaminated accuracy and probe signal for each subset, predict the uncontaminated accuracy. This is the natural evaluation for the probe-based adjustment method.

### Why wider range helps

If all test sets have similar accuracy ($69\% \pm 1\%$), even a poor predictor that always outputs the population mean looks good. A wider range is a harder test — the adjustment must actually track the relationship, not just memorize a constant.

However, wider range is also more forgiving in relative terms: an adjustment error of $\pm 2$pp is a 2% relative error when the range is 40-90%, but a 100% relative error when the range is 69-71%. Both absolute and relative precision matter for different purposes.

### Alternative: natural groupings

MMLU has 57 subjects, providing natural difficulty variation (from abstract algebra to virology). Subject-level accuracy varies widely, creating structured subsets without relying on probes. For WinoGrande, which lacks category metadata, probe-based stratification is the primary option.

## What the Adjustment Needs

For a probe-based set-level adjustment, we need:

1. **Per-example memorization estimates** $\hat{M}(x)$ from the probe (exp 02-03 infrastructure exists)
2. **A model of the contamination effect**: how much does memorization inflate accuracy, on average?
3. **The adjustment**: for each example, subtract the estimated inflation to get counterfactual correctness, then average over the test set

The simplest adjustment is linear: $\hat{C}_0(x) = C(x) - \beta \cdot \hat{M}(x)$, where $\beta$ is the estimated per-unit contamination effect. More flexible versions could use the full dose-response curve from v1.

### Evaluation protocol

The Hubble setup provides ground truth for evaluation:

- **Standard model accuracy** on any subset is the ground truth "uncontaminated score"
- **Perturbed model accuracy** on the same subset is the contaminated score
- The adjustment should map perturbed $\to$ standard

We can evaluate across:
- Random subsets (narrow range, realistic test scenario)
- Probe-stratified subsets (wide range, demonstrates mechanism)
- Natural groupings like MMLU subjects (intermediate, externally motivated)

### Benchmark suitability

| Benchmark | Models above chance? | Accuracy $r$ (random) | Suitable? |
|-----------|---------------------|----------------------|-----------|
| WinoGrande | All configs | 0.25-0.30 | Yes — primary |
| MMLU | Only 8B/500B | ~0 (except 8B/500B: 0.47) | Negative control |

MMLU is useful as a negative control: the adjustment should correctly produce no signal when models are at chance. WinoGrande is where the adjustment method will be tested.

## Next Steps

1. Train memorization probes on WinoGrande hidden states (exp 02-03 have infrastructure)
2. Stratify dup=0 examples by probe score, form blocks
3. Compute contaminated and uncontaminated accuracy per block
4. Fit and evaluate the adjustment: does probe-stratified perturbed accuracy predict standard accuracy?
5. Report both random-subset and stratified-subset results to show mechanism + practical utility
