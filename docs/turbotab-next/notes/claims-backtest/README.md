# Claims backtest (2026-10-14 to 10-15)

Nolan named his biggest fear for the middle layer: slop that undermines trust in the app's intelligence. This backtest applies the claims standard we would hold a research swarm to, backward, to the claims the app already relies on to recommend, set a default, refuse, warn or explain. It also measures whether a referee process can tell a true methods claim from a plausible false one. The plan was fixed and hashed before any claim was read (PLAN.md). The one amendment, a sample in place of every claim, was set from claim counts alone, before the referees started.

## The short version

1. **The app's middle layer is large: about 970 claims set, change or block a user's decision.** There are 135 in prediction, 506 in estimation, 257 in gates and data handling, and 73 in teaching. Several hundred more explanatory sentences were left out of the inventory. No one has checked most of them.
2. **In a seeded random sample of 98 of those claims, weighted by area:**
   - **14.5% have an S1 defect** (95% CI 5.7–23.3%): in a realistic case the user gets a misleading number, interval, recommendation, refusal or lesson;
   - **39% have an S1 or S2 defect** (27–50%);
   - **44% have no defect** (32–56%).

   At the low end, that is a few dozen S1 defects across the whole layer; at the high end, a couple of hundred. The fix list is FIXES.md.
3. **The main form of slop is overreach, not falsehood.** Few claims are simply false methods. The common failures are four:
   - a rule true in a narrower regime than stated;
   - an unsourced threshold shown as a finding;
   - a design ruling the teaching text describes but the code never built;
   - a right behavior given a wrong reason.

   The worst defects are guardrails that don't fire and stale defaults.
4. **The referee gate caught every planted falsehood (0 of 26 passed), but the decoys were too easy.** Referees marked them false with a median confidence of 0.95 to 0.97, against about 0.6 for the real claims they flagged. The gate catches textbook errors. It has not been tested on subtle overstatement, which is the slop this app actually has.
5. **Neither checker layer is enough alone.** The blind referees judge the statement. The basis audit, reading the code, judges whether the app meets it. Twelve real defects, one of them S1, were invisible to both blind referees, and the audit caught them.

## What was done

| Phase | Agents | What |
|---|---|---|
| Inventory | 4 (with forks) | Every methods claim the app relies on, stated in the claim schema, with location, the repo's own basis and any test. Areas: A prediction, B estimation, C gates/refusals/data, D teaching. |
| Sample | none | 25 tier-1 claims per area, seeded (`random.Random(20261015 + k)`); 4 cross-area duplicates dropped; 10 explanatory (tier-2) claims; 9 targeted (the inventory agents' most-suspect picks, reported separately). 117 real claims. |
| Decoys | none (orchestrator) | 30 written before any referee started: 26 false (overreach, direction, constant, mechanism, misapplied citation) and 4 true-but-vacuous. Each is a sibling of a sampled claim, kept out of its sibling's batch, with the key sealed by hash. |
| Merits referees | 24 | 12 mixed batches; each judged blind by one Opus and one Sonnet referee. Referees could not use the repo; they used derivation, fetched literature and short simulations. |
| Basis audit | 8 | Every real claim against the code: fidelity, whether the source says it, whether a test would fail if the claim were false, and the grade. |
| Adjudication | 8 | Every flagged claim (96) with all the evidence: defect, design doc only, convention shown as a finding, referee error, or inventory misstatement; then severity and a fix. |

Roughly 7–8 million subagent tokens, more counting the inventory forks.

## Results

### The gate (the decoys)

| | Result |
|---|---|
| False decoys passing the two-referee gate | **0 of 26** (95% upper bound 11.5% by the rule of three; Wilson 0–12.9%) |
| Opus alone | marked all 26 false |
| Sonnet alone | marked all 26 false |
| Vacuous decoys judged true and changing no decision | 3 of 4 by both models |
| Median confidence on decoys | Opus 0.95, Sonnet 0.97 (real claims judged overstated: 0.65, 0.60) |

By the pre-registered thresholds the gate is fit to pilot. The finding that matters is the confidence gap. The decoys were clean errors with checkable sources, and the referees saw through all of them. The real defects were mostly overstatements, which the referees caught less sharply and less confidently.

One vacuous decoy was not vacuous. It said rescaling a predictor cannot change an unpenalized logistic fit. The Opus referee showed that it can under scikit-learn's default lbfgs solver, which stops short of the optimum without a warning.

### The existing layer

**The seeded random sample** (98 claims) after adjudication:

| | A prediction | B estimation | C gates, data | D teaching | Weighted, all areas |
|---|---|---|---|---|---|
| S1 | 2 | 4 | 4 | 2 | 14.5% (5.7–23.3) |
| S2 | 9 | 5 | 5 | 10 | |
| S3 | 2 | 5 | 4 | 4 | |
| No defect | 12 | 11 | 11 | 8 | 44.2% (32.2–56.3) |

**Grades** (random sample, basis audit):

| Grade | Count |
|---|---|
| judgment | 46 |
| sourced | 24 |
| certified (a repo test would fail if false) | 20 |
| proved | 5 |
| unsupported | 3 |

**All 117 real claims:**
- 25 have a test that would fail if the claim were false.
- Fidelity to the code: 64 faithful, 48 partly, 5 unfaithful.

**The targeted stratum:** 8 of the 9 claims the inventory agents picked as most suspect were defects (2 S1), against about 56% in the random sample. A cheap "which of these do you doubt?" question is a strong triage signal.

### The checkers

| Who flagged it | Claims | Adjudicated real defects |
|---|---|---|
| both blind referees | 57 | 47 (82%) |
| Opus only | 18 | 8 (44%) |
| Sonnet only | 5 | 3 (60%) |
| neither (basis audit or adjudication only) | 37 | 12, one S1 |

**Flags that weren't defects:** of the 96 flagged claims, the adjudicators cleared 26. 16 were referee errors and 10 were inventory misstatements. The adjudicators also overruled referees on facts with their own simulations, as in A-121, A-089 and D-085.

**Agreement:** the two referees agreed on 74% of real claims.

### The S1 defects

| Id | What a user gets |
|---|---|
| A-001 | Below an effective size of 300, boosted trees and XGBoost run at library defaults, which the repo's own study found worst in all 14 small-sample cells. The ruled small-sample standard is not built. |
| A-029 | A fixed 0.01 "smallest gain that matters" on rare-outcome Brier and log loss calls a model with AUC 0.73–0.79 "too little to matter" and refuses its explanation curves. |
| A-102 | The ordinal table's caption says CR1 on t(G−1), while its concern lines say CR2 with Bell–McCaffrey df. The user is told two methods for one interval, and CR1 under-covers with unequal units. |
| A-124 | The energy-screen noticing is banded by rows removed × largest SMD. In the repo's own calibration cases that proxy does not track how far the estimate moves, yet a narrow screen is told it "would not move the estimate". |
| B-211 | A declared case-control or matched table, analyzed through the "observational" exit, is reported with prevalence and absolute risk, and the methods text loses the design. |
| B-281 | Under logistic separation in a trial, the standardized risk difference is refused because "no finite estimate exists". It does exist. |
| B-416 | A numeric exposure with cross-fitted R² ≥ 0.9 is refused as a positivity violation, though the estimate remains valid. Meanwhile, with forest nuisance models, the case the gate is meant to catch never trips it. |
| B-435 | With a zero-event group and another covariate, the Cox model reports a log hazard ratio near −36 with SE near 10⁻¹⁷, p = 0 and a zero-width interval, instead of refusing. |
| BX-040 | The usual-intake card says the upper limit applies to total intake. For magnesium, folic acid, niacin, vitamin E and preformed vitamin A it covers supplements or synthetic forms only. |
| C-123 | A "yes"/blank checkbox column is settled as constant and dropped, with the false evidence "every row holds the same value". |
| C-143 | NHANES 2021–2023 analytes measured at phlebotomy get the exam weight, marked settled, instead of WTPH2YR. |
| C-170 | Under Predict, every collinear pair becomes a Decide whose recommended answer is an elastic-net selection filter. That filter raised a forest's test error from 2.26 to 4.17 in simulation by dropping a curved predictor. |
| C-193 | Unnormalized estimated counts (salmon, kallisto, RSEM) under the genomics lens read as normalized, so no normalization question is asked. |
| D-058 | "Adjust and cluster" puts one indicator per group into an unpenalized logistic fit. For pairs, that doubles a log odds ratio (true 1, estimated 2.1). |
| D-086 | Design doc only: the "mostly the tail" rule would call a uniform shift "mostly the tail" whenever most people exceed the cut. |

Details, fixes and the S2/S3 list are in FIXES.md.

## What this says about the swarm program

1. **The gate needs harder decoys before it certifies anything.** The next decoy set should be overstatements: true claims with the regime widened, unsourced thresholds with invented precision, and right conclusions with wrong reasons. It should be written by people (Nolan, a colleague) or by a different process from the referees. One mind writing decoys for referees of the same family is the weakest version of this test.
2. **Every claim needs both layers.** A blind merits pass checks the statement. A fidelity audit checks that the app does what the statement says. Twelve defects were invisible to the first and caught by the second. The swarm's claims ledger should require both before a claim reaches a screen.
3. **The grade belongs on the screen.** About a quarter of decision-setting claims are proved or certified; the rest are judgment or sourced. The app should say which, label conventions as conventions, and never badge a disputed rule SETTLED (D-037 and D-021 did).
4. **The fix list is a test list first.** One adjudicator ruled on each claim. Each S1 fix begins with a known-truth or independent-reference test that fails on today's code. That test is the mechanical check that turns an adjudicated finding into a certified one. A finding whose test cannot be made to fail is retracted here, and the retraction is recorded.
5. **The ledger needs a ruling-to-code check.** Several defects were rulings that the teaching text or methods sentences describe but the code never built (D-073, D-078, D-084, D-089, D-091, B-506, D-077), or built differently. Each ruling in the claims ledger should point at the code and the test that carry it.
6. **Review capacity is the bound.** Checking 117 claims took about 60 agent runs and a day. The whole layer is about eight times that. The swarm cannot outrun review, so the ledger starts small and grows only as fast as claims are certified.

## Limits and deviations

- **Sample:** the random stratum gives the rates. The tier-2 and targeted strata are reported separately and not pooled. The intervals are wide: at n = 98, a rate is known to about ±10 points.
- **Decoys:** all 30 were written by the orchestrator, the same model family as the referees, and they proved easier than real defects. The gate's 0 of 26 says nothing yet about subtle overstatement.
- **Citations:** the session's web-search budget ran out early. Most referees checked sources by direct fetch (PubMed, Europe PMC, Crossref, CRAN, CDC), and every unreachable citation is marked unverified in the verdicts.
- **Adjudication:** one Opus adjudicator per claim, with no second opinion. The S1 list is adjudicated, not certified (see point 4 above).
- **Deviations from the plan:**
  - The prompts pointed the referees at `.venv`, which lacks pytest. Auditors and adjudicators used `venv` read-only instead.
  - Four referees ran one simulation past the 60 s CPU cap, stopped it, and reran it single-threaded.
  - Two agents briefly wrote helper files into the scratchpad and deleted them.
  - No agent edited the repo.
- **Amendment 1:** sampling in place of refereeing every tier-1 claim. Fixed from the counts, before any referee started (PLAN.md).

## Files

| Path | What |
|---|---|
| PLAN.md, PLAN*.sha256 | The pre-registered plan, Amendment 1, the key's seal, and their hashes |
| inventory/ | The four area inventories and their left-out sidecars |
| batches/ | What each referee and auditor was given (merits batches carry neutral ids) |
| verdicts/ | Every merits verdict (by model) and basis audit |
| adjudication/ | Every adjudication |
| key/ | The decoy key (sha256 in decoy_key.sha256), the targeted ids, the dropped duplicates, and score_report.json |
| scripts/ | Sampling, batch building, decoys, scoring, and the prompts. Paths point at the session scratchpad where the run happened. |
| referee_simulations.tar.gz | The referees' and adjudicators' simulation scripts |
| FIXES.md | The fix list, by severity |
