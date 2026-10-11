# Stratified folds make a cross-validated log loss slightly optimistic

*Finding 1 of the T1 heavy run (2026-10-10). Status: derived, simulated, and folded into T1's
redesign (RECIPES §8 T1). Recommendation in §7: nothing on screen; one entry in the claims ledger.*

**In one paragraph.** TurboTab stratifies its outer folds by the outcome (`stages.rows.draw_split`
with `y`), so every fold's event rate equals the sample's. A model's cross-validated log loss then
never pays for estimating the base rate, while fresh data does. The estimate is low by about
**ρ/n nats**, where n is the number of rows cross-validated and ρ ≤ 1 is the share of the outcome's
variance the model leaves unexplained (ρ = 1 for every model under the null). At T1's n = 200 that
is 0.005 nats, about 0.8% of the entropy and the same size as the tuning leak T1 measures, which is
why T1's null "nested bias" failed. At NHANES's 21,849 rows it is 0.00005 nats. It cancels in every
comparison TurboTab makes on the same folds (model against model, model against its baseline,
BBC-CV), and shows only when a cross-validated log loss is set beside something scored elsewhere.

## 1 · What T1 saw

T1's null generator (y ~ Bernoulli(0.3), n = 200, 200 datasets) reported a nested "bias" of
−0.0064 ± 0.0031 nats (estimate minus truth), failing "|bias| ≤ 2 SE". The flat procedure's
optimism was 0.0095 ± 0.0031, and BBC-CV's corrected estimate sat −0.0045 ± 0.0022 below its truth.
All three carry the same −0.005 shift, derived below; net of it, nested tuning's residual is
+0.0014 ± 0.0031 (no leak), and the flat procedure's leak is about 0.0045.

## 2 · The base-rate case, exactly

Take the null and the model every regularized learner approaches there: predict the training
fold's event rate q. Write H(p) = −p log p − (1 − p) log(1 − p), n for the rows cross-validated, m =
n(K − 1)/K for a training fold, and p̂ for the sample's event rate.

**The estimate.** Stratified folds give every training fold and every test fold the rate p̂ (up
to rounding when the events do not divide by K). So q = p̂ in every fold and each test fold's mean
log loss is H(p̂): the cross-validated log loss *is* the sample's plug-in entropy. A second-order
Taylor expansion, with H″(p) = −1/(p(1 − p)) and Var(p̂) = p(1 − p)/n, gives

  E[H(p̂)] = H(p) − 1/(2n) + O(n⁻²),

the two-class case of the Miller–Madow bias of the plug-in entropy, −(k − 1)/(2n) for k classes
(Miller 1955).

**The truth.** Predicting q on fresh data costs H(p) + KL(p ‖ q), and E[KL(p ‖ q)] = Var(q)/(2p(1 −
p)) + O(n⁻²). How variable q is depends on how the truth's training rows were drawn:

| Truth model trained on | Var(q) | E[truth] − H(p) | E[estimate − truth] | At n = 200, K = 5 |
|---|---|---|---|---|
| m rows drawn at random (the orchestrator's check) | p(1 − p)/m | 1/(2m) | −1/(2n) − 1/(2m) | −0.00563 |
| a stratified training fold (T1: fold 0, or all five) | p(1 − p)/n | 1/(2n) | **−1/n** | **−0.00500** |

A stratified training fold holds the sample's rate, so its model has learned the base rate from all
n rows, not m. T1's truth refits on `StratifiedKFold`'s first training fold (and, redesigned, uses
all five fold models), so −1/n is T1's term.

**Plain folds carry no term.** With folds drawn without looking at y, each test fold is
independent of its training fold, so E[test loss | model] is the model's fresh-data risk and the
estimate is exactly unbiased for a model trained on m rows. Stratification is what correlates the
test fold with the training fold.

## 3 · Any model: the label-shift argument

Condition on the sample's event rate π = p̂. Stratified folds are then, to first order, draws from
the label-shifted population P_π (each class's predictors as before, the prior moved to π), both in
training and in testing, while the truth scores at the population's prior p. Write R_π(f) for a
model f's expected log loss under P_π. R_π is linear in π:

  R_π(f) − R_p(f) = (π − p) · A(f),  A(f) = E[−log f(x) | y = 1] − E[−log(1 − f(x)) | y = 0].

So E[estimate − truth] ≈ Cov(π, A(f̂_π)). A model that follows its training prior moves its logit
by about (π − p)/(p(1 − p)) (exactly so for a logistic model with an intercept: a prior shift is an
intercept shift; boosted trees start from the training fold's log-odds). Differentiating A along
that shift and using Var(π) = p(1 − p)/n:

  **E[estimate − truth] ≈ −ρ_f / n,  ρ_f = E[η(1 − f)]/p + E[f(1 − η)]/(1 − p) = 1 − Cov(f, η)/(p(1 − p)),**

with η(x) the true probability. Three cases:

- **Under the null** η ≡ p, so ρ_f = 1 for *every* f: every model, overfit or not, carries −1/n.
- **For the true probability, f = η,** ρ = E[η(1 − η)]/(p(1 − p)): the share of the outcome's
  variance the predictors leave unexplained. Equivalently, the conditional entropy's curvature in
  the prior is −E[η(1 − η)]/(p(1 − p))², so the estimate's plug-in bias is −ρ/(2n) and the truth's
  base-rate cost +ρ/(2n), the two halves of §2 with ρ in place of 1.
- **T1's signal generator:** p = 0.333 and ρ = 0.821 for η (4 million draws), so −0.0041. A
  fitted model's predictions are shrunk toward the base rate (Cov(f, η) < Var η), which moves ρ_f
  toward 1: T1's nested boosted trees had ρ_f = 0.91 (pilot), a term of −0.0045.

ρ_f ≤ 1 whenever the predictions are not negatively related to the truth, so **1/n bounds the
term** for any such model that tracks its training prior.

## 4 · Simulations

Optimism is truth − estimate (positive flatters). Truths are exact expected losses on fresh data
(the outcome integrated out), so the only noise is the sample's.

| Setting | Stratified | Plain | Paired difference | Predicted |
|---|---|---|---|---|
| Base rate, null, truth on a random 4/5 (orchestrator, 20,000 datasets) | +0.00536 ± 0.00020 | −0.00019 ± 0.00020 | | +0.00563 |
| Base rate, null, T1's fold-model truth (20,000) | +0.00458 ± 0.00020 | −0.00035 ± 0.00020 | +0.00493 ± 0.00003 | +0.00500 |
| Base rate, signal (20,000) | +0.00437 ± 0.00017 | −0.00058 ± 0.00017 | +0.00495 ± 0.00003 | +0.00500 |
| True logit, intercept refit, signal (20,000) | +0.00361 ± 0.00024 | −0.00047 ± 0.00024 | +0.00407 ± 0.00003 | +0.00410 |
| T1's standard candidate (scikit-learn's defaults), null (600) | +0.00942 ± 0.00387 | +0.00118 ± 0.00381 | +0.00824 ± 0.00240 | +0.00500 |
| T1's Sobol candidate with the smallest boosting budget, null (600) | +0.00729 ± 0.00182 | +0.00126 ± 0.00180 | +0.00604 ± 0.00096 | +0.00500 |

The signal rows' unpaired columns share one fresh-data constant each (estimated on 400,000 rows),
so read the paired column. Scripts: `scratchpad/heavy-runs/strat_check.py` (orchestrator) and
`scratchpad/t1design/strat_term.py`, `strat_tuned.py` (this note).

**The tuned pipeline carries the same term.** The flat procedure's own candidates, scored by
`StratifiedKFold` and by plain `KFold` at the same seed on T1's null datasets, differ by 1/n within
one standard error for the regularized candidate the search picks under the null (+0.0060 ±
0.0010 against 0.0050) and within 1.4 for the badly overfit standard candidate (+0.0082 ± 0.0024),
whose held-out losses are far noisier. In T1's 40-dataset pilot, nested tuning's optimism minus the
control's (§6), which removes the term, was +0.0009 ± 0.0042 under the null.

## 5 · Its size at realistic n

The term is ρ/n ≤ 1/n nats whatever the prevalence, while the cross-validated log loss's own
sampling error shrinks only as 1/√n. Below, the term's share of the entropy H(p), and its size in
units of that standard error (the base-rate model's per-row spread over √n; under the null that
spread is √(p(1 − p))·|logit p|, 0.39 nats at p = 0.3 and 0.66 at p = 0.1):

| n | 1/n (nats) | % of H at p = 0.3 | % of H at p = 0.1 | SEs at p = 0.3 | SEs at p = 0.1 |
|---|---|---|---|---|---|
| 100 | 0.0100 | 1.6% | 3.1% | 0.26 | 0.15 |
| 200 | 0.0050 | 0.8% | 1.5% | 0.18 | 0.11 |
| 1,000 | 0.0010 | 0.16% | 0.31% | 0.08 | 0.05 |
| 5,000 | 0.0002 | 0.03% | 0.06% | 0.04 | 0.02 |
| 21,849 (NHANES) | 0.00005 | 0.007% | 0.014% | 0.017 | 0.010 |

So it is negligible at NHANES's size and about 1% of the entropy at n = 200, where it is still under
a fifth of a standard error. It is also an order smaller than the app's smallest meaningful gain
(`baseline.MIN_GAIN`, 0.01 nats) at every n above 100.

## 6 · What it did to T1, and the redesign

T1 compared each procedure's cross-validated log loss with fresh-data truth at n = 200. The term
(0.005) is as large as the leak T1 exists to detect (the flat procedure's, about 0.003 to 0.008), so
"nested within 2 SE of truth" was a test of stratification as much as of nesting. The redesign
(RECIPES §8 T1; the test's docstring) keeps the procedures and changes what is compared:

- **Truth:** every procedure's own five outer-fold models, each scored by its expected loss on the
  fresh rows, averaged. Fold 0's model was the old truth; the mean is unchanged, and the estimate
  and its truth now concern the same models, so a model's quality cancels between them.
- **(a)** O_F − O_N > 0: F and N share their folds (the split stage's folds are `StratifiedKFold`'s
  at the same seed, asserted per dataset), so the term cancels.
- **(b)** O_N − O_control within ±δ, two one-sided tests; **(c)** O_C − O_control < δ. The control
  fits nothing but the base rate on the same folds: the true logit with its intercept refit, which
  under the null is the training fold's event rate. Its optimism is the term alone (§4), and it
  absorbs much of the sample's label noise. Under the signal its ρ is the truth's (0.821); a model
  that shrinks toward the base rate has ρ between that and 1 (N's: 0.91, which the test reports), so
  the residual (ρ_N − 0.821)/n, at most 0.0009, makes (b) and (c) conservative.
- **α = 0.05** one-sided for each; **δ = 0.005 nats**, half of `baseline.MIN_GAIN`: a residual
  optimism below it cannot carry a family with no real gain past the smallest gain the app calls
  meaningful. The flat leak here is of the same size, so (b) alone cannot tell nested from flat;
  (a) does, with a paired standard deviation 3 to 5 times smaller.

**Power** (normal approximation; per-dataset standard deviations from a pilot on T1's first 40
datasets per generator; effects as observed in the 200-dataset run, or the value a correct
implementation is expected to show, whichever is less favorable):

| Assertion | Null: effect, sd | Signal: effect, sd | Datasets for 90% (null, signal) | Power at 600 (null, signal) |
|---|---|---|---|---|
| (a) O_F − O_N > 0 | 0.0017–0.0031, 0.0054 | 0.0058–0.0081, 0.0109 | 100, 40 | 1.00, 1.00 |
| (b) \|O_N − O_control\| < 0.005 | 0 to +0.0014, 0.0265 | +0.0004, 0.0342 | 310–470, 530 | 0.997, 0.94 |
| (c) O_C − O_control < 0.005 | −0.0005, 0.0095 | −0.0035, 0.0268 | 30, 90 | 1.00, 1.00 |

(b) sets the size: **600 datasets per generator**, with about a 0.93 chance that all six assertions
pass for a correct implementation. At about 12.5 s of one core per dataset (the last run: 400
datasets in 20 minutes at 4 jobs), that is about **60 minutes at 4 jobs** (30 at 8). If the pilot's
standard deviations are 20% low, (b)'s signal power at 600 falls to 0.83; 800 datasets restores
0.93 (about 80 minutes at 4 jobs). Smaller margins cost quadratically: δ = 0.004 needs about 850
datasets per generator; δ = 0.0075 about 230.

**The 200-dataset run under the new assertions,** from its printed means (the per-dataset values
were not saved; the term subtracted by theory, since the old run had no control):

| | Null | Signal |
|---|---|---|
| (a) O_F − O_N | 0.0031 (z ≈ 4.1 with the pilot's paired sd) **passes** | 0.0081 (z ≈ 7.6) **passes** |
| (b) O_N − term | +0.0014 ± 0.0031: 90% interval [−0.0037, +0.0065] **fails** (too wide) | −0.0042 ± 0.0027: [−0.0086, +0.0002] **fails** (too wide; 1.6 SE below 0) |
| (c) O_C − term | −0.0005 ± 0.0022: upper +0.0031 **passes** | −0.0035 ± 0.0022 (control ρ): upper +0.0001 **passes** |

Nothing in that run points to a leak. (b) fails for want of precision: the old statistics carried
the whole sample's label noise (sd 0.044 per dataset under the null against the new 0.0265) and 200
datasets. The signal's −0.0042 is a 1.6-SE deviation in the pessimistic direction, which no
mechanism supplies: a procedure that never reads its held-out rows has expected optimism ρ_f/n
against its own fold models (the old truth was fold 0's), and ρ_N ≥ 0.821 here, so it is read as
chance; 600 datasets will settle it.

The pilot itself (T1's first 40 datasets per generator, new statistics): (a) +0.0017 ± 0.0009
(null) and +0.0058 ± 0.0017 (signal), both passing; (b) +0.0009 ± 0.0042 and −0.0066 ± 0.0054,
both too wide at 40, as designed; (c) −0.0009 ± 0.0015 and −0.0039 ± 0.0042, both passing.

## 7 · What TurboTab should say or do

**Recommendation: nothing on screen, and one entry in the claims ledger.** Not a methods
sentence, not a small-sample note.

1. **It cancels wherever TurboTab draws a conclusion.** Every comparison the app makes is paired on
   the same folds: family against family, a family against its baseline (`baseline.py`'s fold-by-
   fold gain), BBC-CV's choice. Under the null every model carries exactly the same 1/n, so the
   gain over the baseline is untouched; with signal the model carries ρ_f/n ≤ 1/n, so the gain is
   understated by at most (1 − ρ_f)/n, the conservative side.
2. **It is small where it does not cancel.** Under a fifth of a standard error at n = 200, under a
   tenth from n = 1,000, and well under `MIN_GAIN` everywhere the app fits boosted trees with a
   search. A sentence about it would be an element that earns no place (BLUEPRINT's bar), and
   "calm over complete" applies.
3. **The remedies cost more than the bias.** Unstratified folds remove it but add fold-to-fold
   variance and, for rare outcomes, folds without events, which the inner-fold floor (§4.2) and the
   per-class metrics need. A +ρ/n correction would need ρ_f, which depends on the model and the
   truth, and would make the reported number differ from the folds' arithmetic.

**What to record instead:**
- the claims ledger (DoD gate 2): "stratified K-fold makes a cross-validated log loss low by about
  ρ/n ≤ 1/n nats, n the rows cross-validated", as this note's derivation, with §4's simulations as
  its check;
- the methods reference's cross-validation entry, for the expert reviewer's packet, in one line;
- **acceptance tests:** any test that compares a cross-validated log loss with fresh-data truth at
  small n must remove the term. T1 now does (§6). T16 reports the optimism of copied values on T1's
  null generator: its report should subtract 1/n or use T1's control. T12 is a squared-error test
  and is unaffected; T8(c) compares arms on fresh data only.

**When to revisit:** if TurboTab ever reports a cross-validated log loss against an absolute
reference computed elsewhere (a published model's log loss, the entropy of a reference population),
at sizes where 1/n is a material share of the gap.

## Sources

- **Miller, G. A. (1955).** "Note on the bias of information estimates." In H. Quastler (Ed.),
  *Information Theory in Psychology: Problems and Methods*, pp. 95–100. The bibliographic details
  were checked against Wikipedia's "Entropy estimation" reference list; I could not open the chapter
  itself, so its statement of the (k − 1)/(2n) bias is cited from secondary knowledge, and §2
  derives the two-class case independently.
- **Tsamardinos, Greasidou & Borboudakis (2018),** *Machine Learning* 107:1895–1922 (BBC-CV), and
  **Bates, Hastie & Tibshirani (2024),** *JASA* 119:1434 (what cross-validation estimates): as in
  `export/data/refs.bib`, already verified for the repo.
- **Not verified, so not cited:** Paninski (2003, *Neural Computation*) on the plug-in entropy's
  bias, and Kohavi (1995, IJCAI) on stratified cross-validation; I could not open either here. I
  found no source that states the stratified-fold term for log loss; the derivation above is mine
  and should go to the prediction reviewer's packet as such.
