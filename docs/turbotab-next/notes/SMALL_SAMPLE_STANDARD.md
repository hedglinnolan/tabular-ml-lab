# The standard settings below an effective size of 300

*Finding 2 of the T1 heavy run (2026-10-10), and the analysis behind RECIPES §4.6's open ruling.
Status: proposed ruling, for the orchestrator. No engine code changed.*

**The question.** Below an effective size of 300, the plan keeps only the standard candidate: for
boosted trees, scikit-learn's `HistGradientBoosting` defaults; for XGBoost, XGBoost's. T1 found
nested tuning better than those defaults by 0.15 to 0.20 nats on fresh data at 200 rows, so a
small table gets an overfit boosted model. Three rulings were on the table: keep §4.6; tune down to
the K floor; or a conservative small-sample standard.

**Proposed ruling: a small-sample standard for the two boosting families, used alone below 300 in
place of the library defaults. §4.6's rule (no search below 300) stays; the forest keeps its
standard settings.**

| Family | The small-sample standard |
|---|---|
| Boosted trees | 100 trees of at most 8 leaves, learning rate 0.02, smallest leaf 20 rows, L2 pull 1, every column, no early stopping |
| XGBoost | 100 rounds of depth 3, learning rate 0.02, least child weight 20 rows-equivalent (the family's own scaling by the mean hessian), λ = 1, α = 0, every row and column |
| Forest | unchanged: ranger's probability-forest defaults (500 trees, √p columns, smallest leaf 10 for classes, 5 for numbers) |

## 1 · The study

- **Generators** (ten N(0, 1) predictors, as T1): T1's *null* (y ~ Bernoulli(0.3)) and *signal*
  (logit −0.85 + 0.8x₁ − 0.6x₂ + 0.5x₁x₃); two more added after the first pass, to check the
  first pick: *strong* (logit −1.4 + 1.5x₁ − 1.2x₂ + x₁x₃ + sin 2x₄ + 1[x₅ > 0.5]) and *rare* (the
  signal's shape at about 10% prevalence). A numeric outcome on the signal's shape plus N(0, 1)
  noise, and its null, as a last check (§4).
- **Sizes:** n = 100, 200, 300, 500 and 1,000 rows: 30 to 350 events. Every cell is below the
  threshold: the plan's effective size is 4/5 of the events, at most 278 on a cell's average and
  at or above 300 in 1 of 680 datasets.
- **Arms**, each fit on all n rows as the deployed model is: the library defaults (what §4.6 does
  now); nested tuning, the app's own search with the plan forced on (8 Sobol candidates and the
  standard one, K = 3 inner folds, the K floor reading the real event count: "tune down to the
  floor"); a grid of conservative fixed settings (learning rate 0.01 to 0.05, 4 to 31 leaves,
  25 to 200 trees, XGBoost depth 2 or 3); the forest at its standard settings, with larger leaves,
  and nested (out of bag).
- **Scored** by the expected log loss on 10,000 fresh rows, the outcome integrated out with its
  true probability, minus the Bayes risk ("excess", in nats). Every arm meets the same datasets and
  fresh rows, so differences are paired. Also the calibration slope on the fresh rows (the logistic
  recalibration's slope, against the true probabilities).
- **Size:** 30 to 60 datasets per cell, about 23 minutes of compute at 4 jobs in all. Scripts in
  the session scratchpad: `t1design/small.py`, `lean.py`, `regress.py`, `final_table.py`.

## 2 · Results for a yes/no outcome

Excess log loss over the Bayes risk, in nats (lower is better). "Standard" is the proposed
small-sample standard; "− nested" is its paired difference from nested tuning (± Monte Carlo SE).

| n | Generator | Events | Boosted trees: defaults | nested | standard | − nested | XGBoost: defaults | nested | standard | − nested | Forest: standard | nested |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | null | 30 | 0.223 | 0.053 | 0.042 | −0.011 ± 0.006 | 0.414 | 0.022 | 0.038 | +0.016 ± 0.003 | 0.019 | 0.017 |
| 100 | signal | 33 | 0.220 | 0.099 | 0.073 | −0.025 ± 0.006 | 0.377 | 0.089 | 0.072 | −0.018 ± 0.005 | 0.064 | 0.069 |
| 200 | null | 60 | 0.245 | 0.038 | 0.041 | +0.003 ± 0.003 | 0.404 | 0.015 | 0.031 | +0.016 ± 0.002 | 0.019 | 0.012 |
| 200 | signal | 66 | 0.247 | 0.069 | 0.059 | −0.010 ± 0.002 | 0.371 | 0.067 | 0.054 | −0.014 ± 0.004 | 0.049 | 0.051 |
| 300 | null | 90 | 0.240 | 0.027 | 0.031 | +0.004 ± 0.003 | 0.398 | 0.009 | 0.023 | +0.014 ± 0.001 | 0.016 | 0.008 |
| 300 | signal | 100 | 0.232 | 0.055 | 0.048 | −0.007 ± 0.001 | 0.356 | 0.055 | 0.044 | −0.012 ± 0.002 | 0.042 | 0.042 |
| 300 | rare | 31 | 0.215 | 0.038 | 0.034 | −0.004 ± 0.001 | 0.179 | 0.036 | 0.031 | −0.005 ± 0.002 | 0.029 | |
| 300 | strong | 106 | 0.207 | 0.096 | 0.097 | +0.000 ± 0.002 | 0.278 | 0.095 | 0.099 | +0.004 ± 0.002 | 0.103 | |
| 500 | null | 152 | 0.220 | 0.017 | 0.019 | +0.001 ± 0.002 | 0.363 | 0.005 | 0.013 | +0.009 ± 0.001 | 0.013 | |
| 500 | signal | 165 | 0.215 | 0.043 | 0.035 | −0.007 ± 0.001 | 0.330 | 0.041 | 0.034 | −0.007 ± 0.001 | 0.035 | |
| 1,000 | null | 296 | 0.188 | 0.011 | 0.010 | −0.001 ± 0.001 | 0.294 | 0.004 | 0.008 | +0.004 ± 0.001 | 0.011 | |
| 1,000 | signal | 329 | 0.176 | 0.029 | 0.025 | −0.003 ± 0.001 | 0.272 | 0.030 | 0.025 | −0.005 ± 0.001 | 0.027 | |
| 1,000 | rare | 103 | 0.196 | 0.021 | 0.018 | −0.003 ± 0.001 | 0.175 | 0.023 | 0.017 | −0.006 ± 0.001 | 0.020 | |
| 1,000 | strong | 347 | 0.157 | 0.058 | 0.073 | +0.015 ± 0.002 | 0.226 | 0.055 | 0.079 | +0.024 ± 0.001 | 0.070 | |

The forest's nested arm was run only up to 300 rows (it is the slowest, and the least tunable).

**Over the 14 cells** (regret: the arm's excess minus the best of its family's arms in that cell):

| Arm | Mean regret | Worst regret | Calibration slope, median [mean p10, mean p90], cells with signal |
|---|---|---|---|
| Boosted trees, defaults (§4.6 now) | 0.178 | 0.223 | 0.30 (overconfident) |
| Boosted trees, nested to the floor | 0.012 | 0.030 | 0.91 [0.65, 1.43] |
| **Boosted trees, small-sample standard** | **0.009** | **0.017** | **1.00 [0.87, 1.10]** |
| Boosted trees, the same with 4 leaves | 0.006 | 0.040 | 1.20 [1.01, 1.35] |
| XGBoost, defaults (§4.6 now) | 0.284 | 0.392 | 0.27 (overconfident) |
| XGBoost, nested to the floor | 0.006 | 0.020 | 0.97 [0.65, 1.48] |
| **XGBoost, small-sample standard** | **0.008** | **0.024** | **1.08 [0.92, 1.21]** |
| Forest, standard settings | 0.003 | 0.009 | 1.14 [1.05, 1.35] |
| Forest, nested (6 cells) | 0.001 | 0.005 | 0.96 [0.69, 1.36] |

What the numbers say:

1. **The library defaults are the worst arm in every cell,** by 0.10 to 0.23 nats for boosted trees
   and 0.15 to 0.39 for XGBoost, with calibration slopes of 0.2 to 0.5: badly overconfident models.
   Keeping §4.6 as written is not defensible for the boosting families. (T1 saw the same gap:
   nested minus defaults −0.199 and −0.153 at 200 rows; here −0.208 and −0.178 on fresh draws.)
2. **Nested tuning to the K floor removes almost all of that loss.** It is never far from the best
   arm (worst regret 0.030 and 0.020), and it adapts: at the top of the band with strong, bent
   effects it is the best boosted arm.
3. **The small-sample standard matches nested tuning on loss** (boosted trees: better on average
   and in the worst case; XGBoost: slightly worse on average, 0.008 against 0.006) **and is far
   steadier.** Its calibration slope's 10th to 90th percentiles span about 0.2 to 0.3 against 0.8
   for nested tuning. That spread is exactly the harm §4.6 names: settings chosen on few rows vary
   from sample to sample and miscalibrate (Riley et al. 2021; Van Calster et al. 2020, as §4.6
   cites them).
4. **Where it loses.** With strong, bent and stepped effects at about 350 events, the fixed
   standard underfits (+0.015 and +0.024 nats against nested; slope 1.46). XGBoost's search also
   beats it under the null (+0.004 to +0.016), where the search can reach very heavy
   regularization (λ up to 100, child weight up to 64 rows).
5. **The forest needs nothing.** Its standard settings are within 0.009 nats of the best arm in
   every cell; nested tuning gains at most 0.008 under the null and nothing with signal. Larger
   leaves help under the null and hurt with signal (up to 0.068), so no fixed alternative is
   better.

## 3 · Why a standard rather than tuning to the floor

The two are close on loss. The standard is proposed because:

- **It keeps §4.6's reason and fixes its value.** §4.6 says the risk below 300 is the variance of
  chosen settings, not optimism. The data agree: nested tuning's calibration slopes spread three
  times as wide. What was wrong was the standard itself: library defaults tuned for large tables.
- **A standard is needed anyway.** Below the K floor (fewer than 4 events in the plan's fit) no
  search can run, and "Standard settings" mode asks for one. Tuning to the floor would still leave
  the library defaults there, the worst arm by 0.2 nats.
- **It costs one fit, not 19** per outer fit (K = 3, 9 candidates: (K − 1)·C + 1), and needs no
  inner folds, so the line, the estimate and replay stay as simple as they are below 300 today.
- **One rule for every size reads the same:** below 300, a fixed standard; from 300, the search.
  The crossover in this study sits near 300 events, where nested tuning and the standard tie on
  average (n = 1,000 cells), so the threshold itself holds.

**The alternative not taken:** tune down to the K floor. It is the safer choice for strong
nonlinear signal near 300 events, and XGBoost's search does slightly better on average here. If the
orchestrator prefers adaptivity, this ruling's standard should still replace the library defaults
as the standard candidate, so the floor and "Standard settings" mode are covered.

## 4 · A numeric outcome (checked, not used to choose)

The constants were chosen on the yes/no cells. On a numeric outcome (the signal's shape plus N(0, 1)
noise, and its null; 30 to 40 datasets per cell), excess squared error over the noise variance:

| n | Generator | Boosted trees: defaults | nested | standard | − nested | XGBoost: defaults | nested | standard | − nested | Forest |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | null | 0.263 | 0.064 | 0.079 | +0.015 ± 0.007 | 0.293 | 0.037 | 0.060 | +0.023 ± 0.005 | 0.047 |
| 200 | signal | 0.461 | 0.391 | 0.384 | −0.008 ± 0.009 | 0.539 | 0.434 | 0.388 | −0.047 ± 0.013 | 0.461 |
| 500 | null | 0.207 | 0.039 | 0.039 | +0.000 ± 0.005 | 0.265 | 0.014 | 0.029 | +0.015 ± 0.003 | 0.034 |
| 500 | signal | 0.315 | 0.244 | 0.247 | +0.003 ± 0.007 | 0.386 | 0.264 | 0.264 | +0.000 ± 0.009 | 0.311 |

The same pattern: the defaults lose 0.07 to 0.26 of the noise variance against nested tuning; the standard ties nested
tuning with signal and trails it a little under the null. With signal the standard is somewhat
cautious (calibration slope 1.18 to 1.24 against nested's 1.01 to 1.05); a numeric outcome may
merit a larger learning rate, which this study did not search.

## 5 · Caveats, and what to check before implementing

- **The constants were chosen on simulations,** ten Gaussian predictors and five generators. The
  first pick (4 leaves) was made on null and signal; the strong and rare generators then showed it
  underfit, and 8 leaves (depth 3) was chosen with every cell in view. So no generator here is held
  out from the final choice. Before release: a check on shapes none of this saw (50 predictors,
  mostly noise; binary and categorical predictors; NHANES subsamples of 200 to 1,000 rows),
  reported in the prediction reviewer's packet.
- **Classes (more than two) were not studied.**
- **The standard's values are TurboTab's, not a library's.** Its `standard_source` must say so
  ("TurboTab's small-sample standard, chosen by simulation"), and the values enter the claims
  ledger as a convention with this note as its evidence.
- **The searched leaf is capped at a twentieth of the plan's rows** (boosted trees: 8 rows at
  T1's 160). Below 500 rows that cap sits under the standard's 20, so the Sobol candidates cannot
  regularize through leaf size as the standard does (its 20 is uncapped). That may be part of why the search does no better at small n; the
  cap deserves a look in the methods review whatever is ruled here.
- **At the top of the band** (300 to 350 events, strong effects) the standard underfits by up to
  0.024 nats. Adding the small-sample standard as one more standard candidate whenever the search
  runs (from 300) costs one candidate and would let the search keep it where it wins; in the
  1,000-row cells it matched or beat the search's best of nine in 5 of 8 family-cells. Proposed as
  an open option, not part of the ruling.
- **What changes if adopted** (for the orchestrator to schedule after phase 4): the two families'
  standard below 300 (`plan.candidates` when S = 0, the K-floor path and "Standard settings"
  mode); the tuning line's wording ("Tuning: small-sample settings (214 people are too few to rank
  settings)"); §4.1's table; T9 and T17's expected candidates; T1's reported comparison, which
  should set nested tuning against the family's own standard.
