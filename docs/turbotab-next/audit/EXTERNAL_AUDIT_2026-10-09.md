# TurboTab v2 — external audit, 2026-10-09

**Auditor:** an independent reviewer (Fable 5.1, xhigh), brought in fresh, no stake in past decisions.
**Client:** Nolan Hedglin, product owner.
**Scope:** the plan of action and what has been done so far, on `turbotab-next` at `9df23628` (with wave E1a integrating at `12e4e1c2`), the docs under `docs/turbotab-next/`, the orchestrator's working memory (read-only), and targeted engine probes run on this machine. Nothing in the repo was written or changed. Probe scripts and logs are in this folder.

---

## Executive summary

**Overall grade: B−.** The engine is strong and honest (A−); the product that a researcher can use is far behind (D+); the plan is thorough but undisciplined on scope and optimistic on time (C+).

1. **The engine's core guarantees hold under independent probing.** Held-out rows pass through the training-fitted pipeline; no estimate can be reached before the plan locks or while the goal is unanswered (the Router and the stage graph refuse every jump); survey-weighted estimates and Taylor-linearized SEs match R's `survey` package to 1e-14 (WLS) and 1e-9 (logistic), lonely-PSU rule included; replay reproduces both NHANES journeys to 1e-12; the elastic net's penalty choice is deterministic across runs and BLAS thread counts. One known leak (RECIPES F11, early-stopping rows seen by outcome-reading preprocessing) is confirmed in shipped code; its practical exposure is narrow (boosted trees above 10,000 rows with a selection lever).
2. **Scope is the project's real risk, not correctness.** The finish line moved from about 69% done to about 23% in six days (Oct 2 → Oct 8) by adding, not displacing. The DoD was "frozen" on Oct 8 and absorbed +62 units the same day. All 367 noticings in 2.0 (236 units, a third of the road) is the single most expensive ruling, and it was made against the orchestrator's advice. Designed experiments, case-control, matched sets, dietary patterns, clustering, Bland–Altman and a Word renderer are a v2.1 roadmap sitting inside v2.0.
3. **No researcher, Nolan included, has driven the production app end to end,** and the UI is being rebuilt for the third time in two weeks (M1 Record → calm kit → quest log) on 31,000 lines of design prose. The engine computes nearly every result and draws almost none (4 of 147 Results/Write-up items on screen). The next six weeks should put one real slice in Nolan's hands, not finish the catalogs.

**Recommended first moves:** reverse question 4 to the orchestrator's T2s (−94 units); defer E1–E5, D3–D6 and X3 to v2.1 (−50 units); pin the held-out and purpose-None tests this week; deny `pkill`/`killall` at the harness, not in prompts; split CI into a fast PR tier and a nightly acceptance tier before the 120-minute budget overflows.

---

## 1 · Strategy and scope

### 1.1 Is the definition of done coherent and achievable?

**Coherent: mostly.** `V2_DEFINITION_OF_DONE.md` has a crisp one-sentence test (line 108–111), seven quality gates, release requirements, and an explicit "NOT in v2" list. The displacement rule ("a new idea enters v2 only by displacing something", line 3–4) is the right discipline. The amendments are dated, attributed, and traceable to rulings in `HANDOFF.md`. That is better governance than most software projects have.

**Achievable in 10–16 weeks: no, not at this scope.** The evidence:

- `SIZING.md` line 7–9: 222 units done, 734 remaining, 3.3× what exists. Of the remaining, 480 touch the interface (line 10). The "unit" scale was calibrated on engine waves ("about 20 units a working day", line 12). Interface work has never run at that pace in this project: M1 part 2 (the frontend Record and stage) was one workflow of 3.2 hours plus two reviews and a fix round for an XL, and that UI is now being discarded.
- `SIZING.md` line 12 itself calls the calendar estimate "the least certain number here".
- The critical path runs through people, not agents: P0.3a (quest-log screens through the calm review loop) needs Nolan's eye on every stage; R3 needs Nolan to drive two journeys; R4 needs six human methodologists. Nolan's availability has been the binding constraint since Sep 27 (usage pauses on Sep 27, Oct 1, Oct 5, Oct 6–8).

**My estimate:** 16 weeks is the floor for the current scope, 20–24 weeks is the realistic range, and that assumes no further amendments. With the cuts in §1.3, 12–14 weeks becomes credible.

### 1.2 Was the move from 69% to 23% justified? Is scope disciplined now?

**Partly justified.** The Classic parity audit (`parity/CLASSIC_PARITY.md`, memory entry 2026-10-05 ~22:50) found that much of the engine's better work was unreachable on screen, that the prediction shelf was narrow (3 families vs Classic's 22), that boosted trees were untuned, and that batch and scales were never asked from the app. A v2 that regressed on Classic's shelf and views would not be accepted by Nolan's colleagues. Adding ridge, Huber, RF, XGBoost, Table 1 and the missing views (the 2026-10-05 amendment, 426 units) was the right call in kind, though not in all its parts.

**Not justified in its entirety.** The 2026-10-07 amendment (210 units) added Describe as a full goal, sequential tracks, dietary patterns, clustering, Bland–Altman, designed experiments with CONSORT, case-control, matched sets, and a dual LaTeX+Word export with Crossref verification, "at his direction; nothing was displaced" (DoD line 28). That is the pattern Nolan himself diagnosed on 2026-09-27: "close two, open three, the goalposts move" (DoD line 5–6; memory `turbotab-velocity-over-ceremony`). He said so again on 2026-10-08 ("I can't really help myself and I keep moving the goalposts") and handed scope control to the orchestrator.

**Is it disciplined now?** Only one day of evidence exists, and it cuts both ways:

- *Good:* C6c (33 units) was displaced to v2.x in exchange for the model contract (DoD line 79–84); V2X_SEAMS maps every deferral to a seam; new ideas go to INBOX by default.
- *Bad:* the "freeze" at `5d575931` admitted +62 units (the model-family contract) the same day, plus 8 seam guards "under the correctness rule" (DoD line 97). The contract came from a paper Nolan uploaded on 2026-10-08 (`learning-mechanics-paper.md`) and became a 1,180-line document with 19 work packages within a day. Several of its ideas (the research record with IRB review, the named-phenomena registry, the hypothesis noticing, the headless research program) are the orchestrator's, not Nolan's. The "correctness rule" is being read as "anything that prevents a future rewrite", which is a seam-keeping rule, not a correctness rule.

**Verdict:** scope control has moved from an owner who adds features to an orchestrator who adds architecture. Both are additions. The discipline I would want to see is a displacement ledger: every unit added since the freeze, and the unit it displaced.

### 1.3 What I would cut, defer or reorder for a credible v2.0

The one-sentence test (DoD line 108) is met by: upload → opening → Who's in → Estimate or Predict → Results → a manuscript a reviewer accepts, in five lenses, verified. Everything below stays within that sentence or falls outside it.

| Cut or defer to v2.1 | Units | Why it is not in the sentence |
|---|---|---|
| **T2+** (second half of the noticing rollout) → T2s | 94 net | Noticings that never fire on a reference journey are never driven, never seen by Nolan, and never reviewed by a methodologist before release. Shipping 367 untested detectors is a correctness risk, not a feature. |
| **E1–E5** designed experiments, CONSORT, case-control, matched sets | 30 | Nutrition and health researchers run trials, but the five lenses as defined are observational. Trials need their own reviewer (R4 adds one), their own fixture, their own journey. This is a named milestone with no dependency on it: perfect v2.1 material. |
| **D3** dietary patterns (PCA, FA, cluster, RRR) | 8 | Four methods, each needing a contract and a reference test, for one lens. Not in Classic. |
| **D4** subgroups by clustering, **D5** Bland–Altman | 6 | Nice to have; no reference journey needs them. |
| **X3** Word renderer | 3 | One manuscript model, LaTeX first. Word coauthors can read a PDF for 2.0. |
| **X4** Crossref checking | 3 | Keep the citation registry; make DOI verification a release-time script, not a product feature in 2.0. |
| **MC-9** the hypothesis noticing, **MC-10** the phenomena registry, **MC-19** soundness runs | 14 | These are the "art into science" program. Valuable, but they are research deliverables riding on a product release. |
| **The research record** (opt-in pooling, IRB) | — | Keep the descriptor fields in the log (they cost nothing); strike the pooling design from 2.0 documents. |

**Total: about 158 units**, leaving about 576. At the project's demonstrated pace that is 29 working days of agent time instead of 37, and, more importantly, far fewer rulings, drives and reviews on the human critical path.

**Reorder:** put one complete vertical slice in Nolan's hands before any catalog work: Your data → Your question → Who's in → Models (Estimate) → Results → Write-up on the dietary NHANES journey, in the quest log, with the two phase-0 noticings and nothing else. SIZING already says slices end in drives; make the first drive the gate for everything after it.

---

## 2 · The plan of action

### 2.1 Is the road's order right?

crosswalk → quest-log redesign → core slices (contract + stage registry) → noticings by family → Describe and new methods → designed experiments → export → re-audit → expert review.

**Mostly right, with two corrections.**

- **Right:** the crosswalk before the redesign was the correct call (it found 21 order conflicts and 29 gaps that a redesign without it would have baked in). The stage registry (P0.4) and ordering fixes (P0.6) before the shell (P0.7) is right. Engine packages that need no screen (C6a, C6b, T3, D3–D6, X2–X4) can run in parallel, as SIZING says.
- **Wrong 1: export is too late.** X1 (the manuscript model) depends only on C7a and C8, yet sits in road stage 4 after Describe and trials. The manuscript is the product's deliverable and the thing a reviewer judges. Move X1+X2 up to ride with C8, so that every drive from the first slice onward ends in a LaTeX bundle.
- **Wrong 2: the noticings are interleaved with slices but sized as a block.** T1 (XL) + T2 (108) + T2+ (108) ride "family by family alongside the slices" (DoD line 89). In practice they will compete with the slices for the same agents and the same Nolan drives. Gate them: T1 phase 0 (two threads) with the first slice; phase 1 (one per lens) only after Nolan has driven the first slice; T2 only after the five reference inference journeys are green in the quest log.

### 2.2 Hidden dependencies and critical-path risks

1. **P0.3a is a human loop with no size you can trust.** It is "L, design" and depends on "the six questions" (all now ruled) and P0.2. But every design to date has been judged too busy (memory `turbotab-calm-over-complete`), the Q&A card was the baseline for one day, the quest log was chosen on Oct 6 and redesigned on paper on Oct 8. Nothing of the quest log has been *drawn* and shown to Nolan since. Every interface unit (480 of 734) waits on this.
2. **P0.12 plain words (L) touches about 396 strings** in `teaching/content.py` and `estimand.py` (SIZING line 58). Every slice's cards carry that text. It is scheduled "early" but after P0.3a; it should be independent of P0.3a and start now, with the word-list CI check first.
3. **The CI budget.** The core-and-server job took about 90 minutes on `9df23628` (check-run started ~13:50Z, completed 15:23:46Z; `v2.yml` line 23 budgets 120). 734 units of work will add tests; the budget will overflow mid-road. BLUEPRINT §8 says fast checks "should stay under ~1 minute"; they are at 90.
4. **Three UIs in two weeks.** The M1/M2 `Record` and `stage` components (about 12,000 lines in `frontend/src/components/record` and `stage`) are to be replaced by the quest-log shell (P0.7). The calm kit (25,000 lines under `frontend/src/explore`) is dev-only lab routes. The production `ProjectScreen` a user sees today is the one the parity audit found unable to send about 34 decision kinds. Until P0.7 lands there is no production UI that can drive a reference journey; until then every "slice" is engine plus mock.
5. **Nolan has not driven a production journey.** DoD gate 4 requires he drive two. His only hands-on contact has been with design prototypes (Oct 5, Oct 6). That is the largest unmeasured risk in the plan: the whole quest-log design rests on his reaction to screenshots and static prototypes.
6. **The legacy residue.** 52,570 lines of legacy v1 modules remain at `turbotab/*.py` (e.g. `packs.py`, 6,336 lines) with 97 legacy test files "in no routine run" (INBOX, "RETIRE, legacy tests"). Classic imports them, so they cannot move; the v2 engine imports them too (`turbotab.packs.suggest`, `turbotab.engine.detect_task_type`, BLUEPRINT §4). An untracked 122 MB `turbotab/.venv` sits inside the package and the Dockerfile's ignore list would bake it into the image (INBOX, "SHIP, image").

### 2.3 Is 10–16 weeks credible?

No, for the reasons in §1.1. Two numbers in SIZING should be revised before the next plan: the interface pace (measure it on P0.7 and C1, then re-estimate the 480 interface units) and the Nolan-loop latency (every design ruling has taken one to three calendar days; there are at least seven stage screens, six Confirm sweeps and two flowcharts to approve).

### 2.4 All 367 noticings in 2.0

**Unwise, and I say so plainly.** The arguments:

- `UNDERSTANDING_LAYER.md` §2.8 caps surfaced noticings at 3 per stage and the reference journeys at 2–9 asked rows. So in any journey Nolan or a methodologist can drive, at most a few dozen of the 367 ever appear. The other ~300 ship as code paths no human has seen fire on real data.
- `CROSSWALK.md` question 4 states the facts: 111 missing, 175 partial, none wired end to end. The recommendation (T2s: ship those that fire on the twelve reference journeys; state or defer the rest; let the coverage test U13 enforce the statement) was correct and costs about 14 units instead of 108.
- The DoD's own release gate (R4, expert review) cannot review 367 detectors' sentences. The methodologist reads the packet for the journeys; the rest go unreviewed into a tool that writes manuscript sentences.
- The understanding layer's own rule (`turbotab-understanding-not-automl`): "a thread that prevents nothing and teaches nothing is cut". A thread that never fires on any reference journey has not yet proven it prevents anything.

The orchestrator's recommendation was right. Nolan should reverse the ruling, or set a bar: a noticing ships in 2.0 only if it fires on a committed fixture and its sentence is in a review packet.

---

## 3 · Engine correctness (probes)

Method: read the code paths first, then run small probe scripts with `venv/bin/python` (read-only, `-n 2` at most, no `pkill`/`killall`). Scripts and raw outputs are in this folder; the appendix has the full results.

### 3.1 Held-out rows pass through the training-fitted pipeline — **CONFIRMED (by code and probe), but not pinned by a test**

- Code: `turbotab/core/stages/modeling.py` 1829–1836: `final = fit(with_units(clone(pipelines[key]), unit_of), X, y)` on the training frame, then `holdout = score(task, final, X_hold, y_hold, …)` with the same object. No refit on all rows precedes the held-out score. `fit_pipeline` (`inner_cv.py` 271–303) returns the fitted `Pipeline`.
- Probe (`probe_heldout_pipeline.py`): with the engine's own `MedianFill` and a `StandardScaler`, the fitted fill statistic equals the training median (3.045229), not the all-rows median (3.048098); `pipe.predict(X_hold)` equals the fitted head's `transform` followed by the model (`allclose`, 1e-12); the scaler's mean is the training mean.
- **Gap:** the INBOX entry of 2026-10-09 (`ab323208`) *schedules* the test that would pin this for every family and recipe. Classic's historical failure was exactly this. It is an S-sized test; it should exist before any new family lands (C6a). Write it now.

### 3.2 No outcome-model estimate before the plan lock, or while the purpose is unanswered — **CONFIRMED by construction; one latent door noted**

- Mechanism, three layers: (a) `sequence._answers_in_order` refuses any Router question behind an earlier open one (`sequence.py` 477–507); `purpose` is the 7th of 28 keys and `split`, `estimand`, `adjustment`, `models` all come after it (`interview.py` 102–109). (b) Every estimate-bearing stage `requires` `models` (`stages/__init__.py`: fit 619, sensitivity 702, calibration 725, secondary 731, scales 745), and `usual_intake` requires `purpose` (755). (c) `estimand.served_gate` withholds estimates while `follow_up`, `clusters`, `estimand`, `adjustment`, `time_varying` or `form` is open (`estimand.py` 1302–1360), and the previews' leash (`consequences.estimates_unseen`, line 512–519) treats an unanswered purpose as the strictest case, with an acceptance test (`test_previews_leash_2.py` §1).
- Probe (`probe_purpose_gate.py`, live `TestClient` on `clinical_risk.csv`): with the Router at the outcome questions, `select_models`, `set_split`, `set_estimand`, `set_missing` were each refused `409 not_yet`; the nine estimate-bearing stages (fit, effects, scales, calibration, explore, evaluation, usual_intake, design, shelf) were all `blocked`; `plan_locked` was null. (My driver did not answer the `event` question, so the exact "purpose open, everything before it answered" state was not reached; the refusal path is the same `_answers_in_order` check, and `purpose` precedes every estimate-bearing question in `QUESTION_KEYS`.)
- **Two honest caveats:**
  1. **The lock is accidental today.** `service._lock_when_shown` (1237–1268) records `lock_plan` as a system decision the first time an estimate is *served*. "Declared before any estimate was displayed" is therefore true by construction but describes no user act. The crosswalk admits this ("the plan locks on the first estimate shown") and P0.8 fixes it (Fit presses the lock). Until then the methods text should not call it a preregistration.
  2. **`_lock_when_shown` returns silently when `purpose is None`** (service.py 1262). Today unreachable, because every estimate stage requires `models`, which the Router holds behind `purpose`. A future estimate stage without a `models` requirement (Describe's estimators in D1, `late_notices` in MC-9) would open this door. Seam guard 7 (ESTIMATE_STAGES derived from `Stage`) helps; a test asserting "no ESTIMATE_STAGES member computes while purpose is None" would close it. Related: `usual_intake` is not in `ESTIMATE_STAGES` (V2X_SEAMS "lists kept by hand"; INBOX 185).

### 3.3 Survey weights and designs — **CONFIRMED against R's `survey` 4.5**

- Probe (`probe_survey.py`): a 14-stratum, 27-PSU design, informative weights, one lonely-PSU stratum, 2,341 rows. `survey_table` (the path `models/linear.py:184` takes under the population answer; also `stages/calibration.py:973`) versus `svyglm(..., nest=TRUE)` with `lonely.psu="adjust"`:
  - WLS: coefficients and SEs agree to ~1e-14 (e.g. x1: 0.5590360310570159 vs R 0.559036031057017; SE 0.01861863125922188 vs 0.0186186312592219).
  - Logistic (pseudo-ML): coefficients to ~1e-12, SEs to ~1e-9 (x2 SE 0.07714471934306631 vs 0.0771447349087646): optimizer tolerance, not a formula difference.
  - df = 13 = PSUs − strata, as NCHS specifies; R's `degf` = 13.
  - The lonely stratum is centered at the mean PSU total with c_h = 1, and the concern text says so and names the rule. Correct.
- The acceptance tests cite Stata's and R's documented formulas and compute references by loops (`survey_references.py`). There are also R-backed tests (`survey_r.py`) that skip without `Rscript`; R *is* installed on this machine (`/opt/homebrew/bin/Rscript`, `survey` 4.5), contrary to the memory note "R is NOT installed" (progress memory, 2026-10-02). Those tests are not run in CI (Linux runner without R).
- Scope caveat, not a defect: only the linear family has a design-based estimator; other families under the population answer are refused or shelved (`no_design_estimator`, `population_shelf`). That is honest. Huber under inference (design-based) is now v2.x.

### 3.4 Cross-platform determinism of the elastic net — **CONFIRMED locally; cross-platform rests on CI**

- Design (`elastic_net.py` 10–24, 50–80; `inner_cv.py` 61–90): inner folds come from row keys hashed at single precision; each inner path is solved exactly (feature-sign search) to 1e-12; the choice is the argmin of the pooled inner loss rounded to 1e-9 of the smallest, ties to the earlier mix and the larger penalty.
- Probe (`probe_enet_determinism.py`, 600 × 40 with collinearity, two seeds): identical results run-to-run and under 1 vs 4 BLAS threads; a 1e-13 relative perturbation of X (below float32 resolution, so the keys and folds are unchanged) leaves the chosen penalty at the same grid point (alpha differs only because the grid is data-derived, 6e-15 relative) and coefficients within 1e-12; the gap between the best and second-best pooled loss was 1.4e-4 and 1.6e-4 relative, five orders above the 1e-9 rounding.
- Known boundary, by design: a perturbation at float32 resolution (1e-7 relative) changes some keys, so folds change, the penalty moved 23–29%, and coefficients moved by about 0.011–0.013. This is inherent to any cross-validated choice and the single-precision hash makes it rare across platforms (docstring: ~1 in 5e8 values), but it means "deterministic" is a statement about bit-equivalent float32 inputs, not about a dataset stored at different precisions.
- Cross-platform evidence: CI's WP7 check holds every family to 1e-9 on Linux x86 (`62bd1785`, `68ff191a`; check-runs green on 2026-10-09). Windows runs the launcher smoke only, not the numerics. If Windows users are a target, add the WP7 check to the Windows job.

### 3.5 Replay reproduces the numbers — **CONFIRMED**

- `pytest turbotab/core/tests/acceptance/test_export.py::test_2_a_fresh_home_replay_reproduces_the_matrix_and_every_estimate -n 2`: **2 passed in 56.78 s** (inference and prediction journeys on the committed NHANES fixture). The test asserts the model matrix byte-identical (Parquet SHA-256 and content hash), every recorded estimate within 1e-12, more than 50 estimates compared, stage keys and plan hash equal, and no bundle file different (`test_export.py` 567–601).
- `export/replay.py` verifies input SHA-256 first and refuses otherwise. This is a genuinely good piece of engineering and the strongest reproducibility story I have seen in a tool of this kind.

### 3.6 Leakage in tuning, selection or early stopping (RECIPES F11) — **LEAK CONFIRMED in shipped code; narrow exposure**

- Code: `inner_cv.fit_pipeline` 296–302: `Xt = head.fit_transform(X, y)` on *every* row, then `model.fit(Xt[~held], …, X_val=Xt[held], …)`. Any outcome-reading step in the head (the `Selector`, `UnivariateScreen`, `InnerCVForms`, the imbalance wrapper) sees the stopping rows' outcomes before the stopping set is split off. RECIPES F11 names this and says it "must be fixed before the search … is built"; RT-1 carries the fix.
- Probe (`probe_f11_early_stopping.py`): 12,000 rows (above `EARLY_STOPPING_ROWS` = 10,000), 400 noise columns, null outcome, a top-10-by-correlation selector before `HistGradientBoostingRegressor(early_stopping="auto")`. The head was fit on 12,000 rows including all 1,200 stopping rows (leak confirmed). The optimism in this null configuration was negligible (implied MSE on stopping rows 0.9443 leaky vs 0.9439 clean vs var 0.9436; 8 of 10 selected columns shared). With a more aggressive screen at p ≫ n the bias would grow, but boosted trees stop early only above 10,000 rows, which omics tables rarely reach.
- Verdict: a real defect with small practical risk today. It is a ten-line fix (fit the head on `~held`, transform `held`); I would land it now rather than wait for RT-1's search engine, because every prediction journey above 10,000 rows with a selection lever exercises it.
- Other leakage checks, by reading: the elastic net's screened variant tunes its penalty on features screened with the inner rows' outcomes (RECIPES F4) was fixed by the pooled path search (`c113a139`); predictors are selected inside the resampling (`76d31205`); the seal's reads are guarded (`seal._the_draw_reads_settled_values`); the checklist route's score leak was caught by a verifier and fixed (`f46b266e`). I found no new leak.

### 3.7 Not probed, flagged for the re-audit

- The accidental lock (3.2) under a *purpose switch* from prediction to inference (`_lock_after_prediction`) depends on a side file `estimates_shown.json` outside the decision log (V2X_SEAMS; `service.py:54`). A replay of such a project cannot reconstruct that the scores were seen. Move it into the log (seam guard 8 covers the track version; do the single-track case now).
- The `dml_plr` classifier bug (INBOX, 2026-10-08) is open: a yes/no outcome fit as a regressor in the causal lane.
- The 42 NHANES-fixture tests were skipped on CI until 2026-10-08; everything verified before that on CI ran on synthetic fixtures only.

---

## 4 · Design and product

### 4.1 Is the quest-log design coherent with the crosswalk and the engine?

**Yes, on paper, and unusually well cross-referenced.** `calm/FOUNDATION.md` §3 (the seven stages, the bar, Decide · Confirm · For the record, three levels of disclosure), §5 rule 6 (no estimate before Fit, the outcome's gates) and §10 (the display-order rule, all 20 conflicts tabled) line up with `CROSSWALK.md` "Settled here", the 21 order conflicts and their fixes, and with `estimand.ESTIMATE_STAGES`, `served_gate`, `_lock_when_shown` in the engine. The display-order rule (three conditions, "never quietly") is a good methods rule and the kind of thing that makes this tool defensible.

**Two coherence gaps:**
- BLUEPRINT §11 and §11.4 still describe the older objective list (HANDOFF line 43 says to follow FOUNDATION). A builder agent reading BLUEPRINT first will build the wrong thing. Patch BLUEPRINT or mark the sections superseded in place.
- The engine has no notion of stages (`CROSSWALK.md` Gaps engine 1). The stage registry (P0.4) landed in wave E1a on 2026-10-09 (`12e4e1c2`), after FOUNDATION was written against its absence. Re-check FOUNDATION §3 against the registry's actual placements before P0.7 starts.

### 4.2 Is it actually calm, or drifting back toward complexity?

**Drifting, in a new form.** On 2026-10-05 Nolan said every design was too busy: "colors and dashed lines… hard to know what I need to click." FOUNDATION answers with a calm budget (one focal region, one type family, one quiet label per option) and a five-second test. Good. But read FOUNDATION §3 as a whole screen: a seven-stage quest line; a seven-segment progress bar; a card with Decide lines, a Confirm sweep ("Here are the 6 other choices set for you"), a collapsed For the record list, "Waiting for: [question]" lines, "May change as you answer" lines, "changed by you" marks, "reopened because …" lines beside the bar and atop the card, "changed since you looked" labels, "Not available yet" options, "Known as …" terms on hover, a "noticed" link on sentences; a manuscript rail with a sentence count and a marked newest sentence; a tapestry with up to three linked views, a flip, a storyboard, "More angles", and gray/indigo semantics; two full-tapestry flowcharts. The color noise was removed and replaced with *text* noise. Each device earns its place one at a time (the registry test will say so); the whole screen has not been judged, because no quest-log screen exists yet to judge.

The honest check is cheap: draw the Models stage for the dietary inference journey (the heaviest stage: 231 items, 132 Decide, 14 cards) as one static screen and run the five-second test on it with Nolan before P0.7 is built. SIZING P0.3a says this will happen; make it the first thing, not an "L design" package.

### 4.3 Will the noticings, Confirm sweeps and exhibits overwhelm?

- **Noticings:** the caps in `UNDERSTANDING_LAYER.md` §2.8 (3 "worth a look", 1 context line per card, 3 per stage) are the right instrument, and the reference journeys' asked-row targets (5–9 cards) are reasonable. The risk is the gate at the lock: "every open noticing feeding the plan must be decided or dismissed before the lock" (ruling 1, overruling the orchestrator's "leave as a limitation"). On the heaviest clinical journey that is 8–9 rows on 5 cards plus the family checks, at the exact moment the user wants results. I would restore "leave as a limitation" for non-T1 noticings and keep the hard stop for T1 blockers, as the orchestrator recommended.
- **Confirm sweeps:** one per stage, listing only defaults whose alternative changes a number, cleared in one click. This is well designed and is the best idea in the UI. Watch Models: 63 Confirm items in the crosswalk; if a journey's Models sweep lists more than about eight, the sweep has become a settings page.
- **Exhibits:** the floor (locked primary always reported; every analysis listed in the supplement; no p-value trimming) is exactly right and protects against the selective-reporting failure that curation invites. The wording choices ("Recommended" drafts, "Write my own" checked against claim strength) are good. The risk is volume: Estimate journeys compute seven estimate stages' worth of exhibits; a Results stage with 20 exhibits each needing a wording and a placement is a second manuscript-editing job. Default most to the supplement with placement pre-filled and let the user promote.

### 4.4 The thing nobody has done

No researcher has driven the production app end to end. Nolan has reacted to screenshots and prototypes. DoD gate 4 requires he drive two journeys; that should not wait for R3. The first quest-log slice should be driven by him in the real app with the real NHANES fixture, and his reaction should be allowed to change the plan.

---

## 5 · Process and orchestration

### 5.1 Does adversarial verification catch things?

**Yes, demonstrably.** Evidence in the memory and the log:
- Methods layer (2026-10-03): the verifier found 4 open items and 11 regressions, including a test that "asserted that the holdout leads under inference" (ruling 3 codified backwards), substitution reading training rows only under inference, and bootstrap optimism overstating boosted trees.
- Routing (2026-10-05): verifier 20/28 closed; the repair closed 8.
- Wave 2c (2026-10-05): EXPORT 17/19; the verifier found `GET /checklist` printing the winner's selection-corrected score without recording it as seen (a real leak), fixed in `f46b266e`.
- Resume fixes (2026-10-08/09): verifiers found the purpose-unanswered estimate leak in previews, the logistic elastic net's non-determinism, and generic-preview leftovers, which became fix round 2 (`9df23628`).
- The model-contract literature agent corrected six from-memory citations the orchestrator had made (`learning-mechanics-paper.md`).

The verifier prompt (`turbotab-resume-fixes-wf_0cab0f6d-667.js` line 145) is well built: reproduce on base, confirm gone on branch, use your own inputs, check the fixer's tests fail on base, hunt regressions, read-only, "confirmed only if every item is fixed with evidence". Keep it for engine code.

What I could not find: a false-positive rate (how often verifiers "refute" wrongly and send fixers chasing), or any measurement of verifier value per token. The blocked experiment (§5.3) was an attempt to get one.

### 5.2 Cost

From the workflow state files (`workflows/wf_*.json`, `totalTokens` as the harness reports it, which includes cache reads):

- **About 92 million subagent tokens across 39 TurboTab-Next workflows in 12 days** (2026-09-27 to 2026-10-09), 98 million including four Classic-era runs. The orchestrator's own context is not counted.
- Heaviest: wave1-completeness 8.0M + 4.4M (two runs), wave2-modeling-sequence 6.7M, ground-up-audit 5.3M, **crosswalk 4.2M (a documents-only workflow)**, m1-w2 and m2-w1 3.6M each, calm-structures 3.3M, resume-fixes 3.0M + 2.6M, fix-math-layer 3.2M.
- **Design and document workflows consumed roughly 15M tokens** (crosswalk, understanding-threads, calm-structures, parity audit, recipes spec, model contract, redesign groundwork, EDA briefs). That is a sixth of the total spent on prose that the orchestrator's own rule says not to write ("Don't write long prose docs; keep rules in code/gates", `turbotab-velocity-over-ceremony`).
- The 2026-10-09 policy (about 200K per agent, references not blobs, fresh agents with handoffs, targeted tests) is the right direction. It needs one more rule: a token budget per workflow with a hard stop, and a post-run line in the journal comparing budget to actual.

### 5.3 Failure modes

1. **The `pkill` incident (2026-10-09 ~04:19).** A fix-round-2 subagent ran `pkill -f` with an unescaped `|`, matching every command line with a space, and sent SIGTERM to Nolan's launchd agents, Claude helpers, Ollama, mail and other agents' test runs. The response was a *prompt rule* ("never use pkill, killall or pattern kills"). Prompt rules are advisory; the next agent that reads a different prompt will not have it. This belongs in the harness: a permission deny rule for `pkill`, `killall` and `kill -9 -1`, and a `PreToolUse` hook that refuses Bash commands containing them. Cost: S. This is my top process recommendation.
2. **The blocked verifier experiment.** The auto-mode classifier refused a step that planted bugs and made a test vacuous. The orchestrator did not route around it (correct), but the script lacked an abort check, so 8 verifiers reviewed an empty diff and 337K tokens were wasted. Lesson recorded ("every workflow validates its setup's outputs before fanning out"). The historical-replay benchmark proposed instead is a better experiment anyway.
3. **Resume semantics** cost at least two restarts (2026-10-01, 2026-10-05: resume caches only the longest unchanged prefix; parallel tracks re-ran verifiers and committed repairs at load average 41). The continuation-script pattern is the fix and is recorded.
4. **The staged-into-an-integrator's-merge accident (2026-10-02)** and **the orphan server left running 1h43m (2026-10-05)** are both "two agents in one checkout" problems. The worktree discipline that followed is right; 18 worktrees and 130 branches now exist, and the orchestrator has a prune list awaiting Nolan.
5. **Scope drift from the orchestrator's side**, described in §1.2.
6. **CI diagnosis without log access (2026-10-05/06)** cost most of a night; the public-annotations workaround (`8a554150`) is clever, but `gh auth` would have cost five minutes of Nolan's time. Ask for the small human action earlier.

### 5.4 Is the orchestrator's working memory accurate and useful?

**Useful: very.** The memory captured every ruling with Nolan's words, dates and commit hashes, and the "why / how to apply" structure makes it actionable. `turbotab-quest-log-tapestry.md` and `turbotab-velocity-over-ceremony.md` are models of what an orchestrator's memory should be.

**Accurate: mostly, with specific errors:**
- `turbotab-next-progress.md` is a 90 KB chronological log, not a memory; its frontmatter still says "paused 2026-09-27", its MEMORY.md index line says "Oct 3", and the newest entries are at the top of a file whose oldest claims (e.g. "R is NOT installed", 2026-10-02) are now false (R 4.x with `survey` 4.5 is at `/opt/homebrew/bin/Rscript`). A fresh session reading it will inherit stale facts.
- `turbotab-quest-log-tapestry.md` records the seven stages in the order "Your data, First look, Who's in, Your question, Models, Results, Write-up" (ruled 2026-10-06) while HANDOFF, DoD and FOUNDATION use "Your data, Your question, First look, Who's in …" (ruled 2026-10-06/07). The memory does not mark the first order as superseded.
- Memory records Nolan's words faithfully but sometimes records the orchestrator's own proposals in the same register ("Case-control and matched sets ride with it (my assumption; Nolan may veto)" is honest; the research record's IRB design is not flagged the same way).

**Fix:** split the progress file into a short "state now" (one screen) and an archive; add a "superseded" line where a ruling changed; date-stamp environment facts.

### 5.5 Stop, start, keep

**Stop**
- Writing a 1,000-line specification per idea. `docs/turbotab-next` is 31,254 lines of Markdown against about 115,000 lines of engine code. CROSSWALK 2,223, RECIPES 1,796, MODEL_FAMILY_CONTRACT 1,180, UNDERSTANDING_LAYER 1,125. The project's own rule says keep rules in code and gates. Specs should be the length of their work packages.
- Admitting orchestrator-originated architecture into a frozen DoD under a "correctness rule". Seams are kept with a line in INBOX, not with packages.
- Treating a prompt rule as a safety control (the `pkill` lesson).
- Rebuilding the UI before a user has driven the previous one.
- Running the whole acceptance suite in one 90-minute CI job on every push.
- Two reviewers for documents; one is plenty (already policy; keep it).

**Start**
- Pin the held-out and the purpose-None tests this week (S each).
- A harness-level deny for process-pattern kills (S).
- A displacement ledger beside the DoD: every unit added after the freeze and what it displaced.
- A weekly "Nolan drives one slice in the real app" ritual, starting with the first quest-log slice.
- A CI split: fast tier (< 10 min: core unit tests, server routes, frontend check) on every push; acceptance tier (NHANES, replay, R-backed where available) nightly and on the integration branch.
- A per-workflow token budget with a hard stop and a one-line budget-vs-actual in the journal.
- Interface pace measurement on P0.7 and C1, then re-estimate.

**Keep**
- The event-sourced decision log and replay (3.5): the project's best asset.
- Independent reference tests (R, samplics, hand computations, Stata's documented formulas).
- The adversarial verifier on engine code, with the current prompt.
- The leash, the served gate and the display-order rule.
- The two-register plain-words rule and the Confirm sweep.
- The seams map and the INBOX-by-default rule.
- Recording Nolan's rulings verbatim with dates.

---

## 6 · Anything else a seasoned reviewer would flag

1. **The "unit" is doing too much work.** It was calibrated on engine waves (S = one module and its reference test). It is now applied to design work ("P0.3a: L"), interface work and human loops. A plan whose single most uncertain number is its own unit should carry two scales, or carry ranges.
2. **The CI job is a correctness asset that is about to become a bottleneck.** 90 minutes against 120; `-n 2` on a 2-core runner; 1,305 acceptance tests plus 1,915 core and server tests; NHANES tests added 2026-10-08. Split tiers now.
3. **Dependency pinning.** Requirements are lower-bound only; the 2026-10-05 CI failures were partly newer releases (numpy 2.5, scikit-learn 1.9.1). The orchestrator noted "a pinned constraints file used by CI, the launcher and Docker" as the likely fix; I did not find it landed. For a tool whose selling point is reproducibility, pin.
4. **Security items in INBOX are open:** account lockout by anyone who knows a username (5 failures → 15-minute lock), IPv6 per-address limits, a proxy-trust check that accepts the compose network's own /24, and the Dockerfile ignore list that would bake untracked files into the image. None blocks a local launcher; all block a university server. Fix before DEPLOY.md is handed to an IT department.
5. **Legacy residue inside the package:** 52,570 lines of v1 modules at `turbotab/*.py`, 97 legacy test files in no routine run, and the v2 engine importing from them (`turbotab.packs`, `turbotab.engine`). Plan the absorption into `turbotab/core` as a v2.0 release item or accept that v2.0 ships with v1's `packs.py` as a load-bearing wall.
6. **The expert review packets are stale** (INBOX, DOCSREF): captures predate the wave 2b/2c repairs and were run "with local changes". Regenerate at a clean commit before any methodologist reads them (metabolomics prediction 27 min, survey 31 min: schedule with Nolan).
7. **Flaky test** `test_ledger_repair_3.py::test_a1` is "likely closed" but kept open; a flaky acceptance test in a 90-minute job is expensive. Quarantine or fix.
8. **Methods text calls the lock a declaration.** Until P0.8, say "the plan in force when the first estimate was displayed" in the methods sentence, which is what it is.
9. **Describe's lock on the first estimate** is the right call (forking paths apply to weighted prevalences too), but Table 1 under Describe is "the participant characteristics table every nutrition paper has" and should be exempt from any lock, since it is descriptive by design and reviewers expect it before anything else.

---

## 7 · Top 10 recommendations (ranked)

| # | Recommendation | Cost | Impact |
|---|---|---|---|
| 1 | **Reverse question 4:** ship the noticings that fire on the twelve reference journeys plus the 17 sentinels; state or INBOX the rest, enforced by the coverage test (T2s). | S (a ruling) | −94 units; removes ~300 untested detectors from 2.0; frees the critical path. |
| 2 | **Defer E1–E5, D3–D5, X3, MC-9/10/19 to v2.1** with their seams named in V2X_SEAMS. | S (a ruling) | −60 units; one fewer human reviewer; one fewer fixture and journey. |
| 3 | **Put one real slice in Nolan's hands first:** dietary NHANES Estimate, quest log, phase-0 noticings, LaTeX bundle, before any catalog rollout; make his drive the gate. | M | De-risks 480 interface units against a design nobody has used. |
| 4 | **Pin the two guarantee tests now:** held-out rows through the training-fitted pipeline for every family and recipe; no `ESTIMATE_STAGES` member computes while `purpose is None`. | S + S | Turns two by-construction guarantees into loud failures. |
| 5 | **Fix F11 now** (fit the head on the non-stopping rows; transform the stopping rows), with a test on a 12,000-row fixture. | S | Closes the one confirmed leak in shipped code. |
| 6 | **Harness-level deny for `pkill`/`killall`/pattern kills** (permission rule + PreToolUse hook), replacing the prompt rule. | S | Prevents a repeat of the 2026-10-09 incident regardless of prompt. |
| 7 | **Split CI into a fast PR tier (<10 min) and a nightly acceptance tier**; add the WP7 numerics check to the Windows job. | M | Keeps CI under budget through the road; covers the third platform. |
| 8 | **Draw the Models stage of the dietary journey as one static screen and five-second-test it with Nolan** before P0.7. | S–M | Catches the text-noise drift (§4.2) before it is built. |
| 9 | **Token budget per workflow with a hard stop,** and a budget-vs-actual line in each journal; cap document workflows at one reviewer and 1M tokens. | S | Addresses the cost Nolan raised with a measurable control. |
| 10 | **Move X1+X2 (manuscript model, LaTeX) up to ride with C8,** so every drive ends in a bundle; pin dependencies with a constraints file; regenerate the review packets at a clean commit. | M | The deliverable is exercised from the first slice; reproducibility claims hold against upgrades. |

---

## 8 · What to stop doing

- Writing specifications longer than the code they specify. Set a cap (say 300 lines) and put the rest in INBOX or in tests.
- Adding packages to a frozen DoD under any rule other than "a published number would be wrong without it".
- Interleaving a 236-unit catalog rollout with the slices that need the same agents and the same human.
- Treating prompt text as a safety boundary for destructive shell commands.
- Running the full acceptance suite on every push to every integration branch.
- Carrying stale environment facts in a 90 KB "memory" file.
- Rebuilding the production UI on a design no user has driven.
- Spending ~15M tokens on prose per fortnight.

---

## 9 · Where I disagree

**With Nolan**
- **All 367 noticings in 2.0.** Reverse it (§2.4). The orchestrator was right.
- **Designed experiments, case-control and matched sets in 2.0.** They are a milestone with no dependency on them and their own reviewer. v2.1.
- **"Every open noticing must be decided before the lock"** (overruling "leave as a limitation" for non-T1). This will stall users at the moment they most want results, and it makes the open-noticings card a hurdle rather than a help. Keep the hard stop for T1 blockers only.
- **Word in 2.0.** LaTeX first; Word from the same model in 2.1.
- Where I agree with him against the orchestrator: trees "try both" for blanks by default; long fits wait for Fit; the quest log over the Q&A card; plain words on every card; "Not available yet" options. All sound.

**With the orchestrator**
- **The model-family contract entering 2.0 at +62 units the day the DoD was frozen.** The door (MC-1, MC-2a, MC-12, MC-13) is DoD §2's own requirement and belongs in C6a; the intelligence (MC-8, MC-9, MC-10, MC-19) is a research program. Split it.
- **The 10–16 week estimate.** 16 is the floor for the current scope (§1.1).
- **Documents as the primary design instrument.** 31K lines of Markdown is a cost center and a context sink; the orchestrator's own 2026-09-27 rule says so.
- **F11 scheduled behind RT-1.** It is a ten-line fix in `inner_cv.py` and should land now.
- **The research record / opt-in pooling / IRB** in a 2.0 DoD. Keep the descriptor fields; strike the program.
- **Calling the first-view lock a "declared plan"** in methods text before P0.8.
- Where I agree with the orchestrator against its own earlier practice: the 2026-10-09 token policy, the historical-replay benchmark for verifiers, the display-order rule, Describe's lock, the seams map.

**With both**
- The pace of design rulings (dozens per day on Oct 6–8) outran anyone's ability to build or test them. A ruling that is not followed by a drive within two weeks should be re-confirmed before it is built.

---

## Appendix A · Probes run and results

All scripts are in `/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad/audit/`. Run with `PYTHONPATH=/Users/nhedglin/tabular-ml-lab venv/bin/python <script>`; `OMP_NUM_THREADS=2`; nothing in the repo written; no process killed.

### A.1 `probe_heldout_pipeline.py` — held-out rows through the training-fitted steps
```
fitted fill step attributes: ['feature_names_in_', 'n_features_in_', 'indicator_', 'statistics_']
train median(b)=3.045229  all-rows median(b)=3.048098  differ=True
(b) pipe.predict(X_hold) == head.transform(X_hold) -> model: True
(c) scaler mean(a) fitted on train: True | on all rows: False
```
Code evidence: `stages/modeling.py` 1829 (`final = fit(…, X, y)`), 1836 (`score(task, final, X_hold, y_hold, …)`), 1840 (`predict(task, final, X_hold, …)`). INBOX (`ab323208`): the pinning test is scheduled, not written.

### A.2 `probe_purpose_gate.py` — jumping ahead of the purpose, live server
```
first_open_before_jump: {"key": "event", "status": "open"}
jump:select_models  -> 409 not_yet, no estimates in response
jump:set_split      -> 409 not_yet
jump:set_estimand   -> 409 not_yet
jump:set_missing    -> 409 not_yet
plan_locked_before_purpose: null
stage_status: usual_intake/shelf/design/explore/fit/calibration/scales/effects/evaluation: all "blocked"
set_purpose (out of order): 409
```
Note: the driver did not answer `event` (the interview step carries no options; they come from `target_info`), so the Router was one question earlier than intended. The refusal logic is the same (`sequence._answers_in_order`), and `purpose` precedes every estimate-bearing question in `interview.QUESTION_KEYS`. The previews' leash for `purpose=None` is covered by `test_previews_leash_2.py` §1. Latent door: `service._lock_when_shown` 1262 returns silently when `purpose is None`; unreachable today because every `ESTIMATE_STAGES` member `requires` `models`.

### A.3 `probe_survey.py` — survey_table vs R survey 4.5 (`nest=TRUE`, `lonely.psu="adjust"`)
```
R:      lm_coef [0.97376383206042, 0.559036031057017, -0.883237567136157]
        lm_se   [0.104555768532984, 0.0186186312592219, 0.0403416929758408]
        glm_coef [0.084098346406474, 0.590905542287265, -0.735958768065739]
        glm_se   [0.116772988881592, 0.03390094651305, 0.0771447349087646]   df 13
ENGINE: lm  est 0.9737638320604189 / 0.5590360310570159 / -0.8832375671361575
            se  0.10455576853298383 / 0.01861863125922188 / 0.040341692975840004   df 13
        glm est 0.08409834640654652 / 0.5909055422887767 / -0.735958768067357
            se  0.11677299021064928 / 0.033900985125834386 / 0.07714471934306631  df 13
        info: 27 PSUs in 14 strata, t(13), lonely stratum "13" centered at the mean PSU total, n_h/(n_h−1) = 1
```
Weights reach the stage path via `models/linear.py:184` and `stages/calibration.py:973`.

### A.4 `probe_enet_determinism.py` — penalty choice stability (two seeds, 1 vs 4 BLAS threads)
```
seed 11: same_twice true; threads 1 == threads 4; choice (alpha 0.0354602, l1 1.0)
         1e-13 perturbation: same grid point (alpha rel diff 5.7e-15), coef diff 0.0
         1e-7  perturbation: alpha rel diff 0.233, coef diff 0.0110 (folds changed: float32 keys)
         best-vs-second pooled-loss gap 1.64e-4 (rounding is 1e-9)
seed 23: same_twice true; threads 1 == threads 4; choice (alpha 0.0414578, l1 0.9)
         1e-13: alpha rel diff 1.7e-14, coef diff 1e-12;  1e-7: alpha rel diff 0.295, coef diff 0.0126
         gap 1.37e-4
```
Cross-platform: CI's WP7 check (1e-9, every family) green on Linux at `62bd1785` and `9df23628` (check-runs read from the public API on 2026-10-09). Windows: launcher smoke only.

### A.5 Replay — `test_export.py::test_2_a_fresh_home_replay_reproduces_the_matrix_and_every_estimate`
```
..                                                                       [100%]
2 passed in 56.78s
EXIT 0
```
(inference and prediction on the committed NHANES fixture; matrix byte-identical; every estimate ≤ 1e-12; stage keys and plan hash equal; no bundle file different.)

### A.6 `probe_f11_early_stopping.py` — early-stopping rows seen by an outcome-reading head
```
rows: 12000; stopping rows held: 1200
(a) head fit on 12000 rows; stopping rows among them: 1200  (F11 confirmed if > 0)
(b) leaky: best validation score -0.47215; n_iter 17; var(y_held) = 0.94360
    clean: best validation score -0.47194; n_iter 14
    leaky top-10 ∩ clean top-10: 8 of 10
    implied MSE on stopping rows: leaky 0.94430 vs clean 0.94388 vs var(y_held) 0.94360
```
Code: `inner_cv.py` 296–302. Exposure: boosted trees above 10,000 rows with `Selector`, `UnivariateScreen`, `InnerCVForms` or the imbalance wrapper in the head. Fix scheduled in RECIPES RT-1.

### A.7 CI check-runs (public API, read 2026-10-09)
```
9df23628: launcher macOS ✓, mock browser ✓, server image ✓, launcher Windows ✓, core+server+frontend ✓ (completed 15:23:46Z; ~90 min)
62bd1785: all five ✓
12e4e1c2 (wave E1a integration): four ✓, core+server in progress
```

### A.8 Size and volume facts used above
- Engine (non-test): `turbotab/core` 44,171 + `models` 20,554 + `stages` 21,166 + `methods` 20,919 + `export` 2,971 + `teaching` 1,760 + `detectors` 3,089 ≈ 115K lines; `server` 4,559. Legacy v1 modules at `turbotab/*.py`: 52,570.
- Tests: 150 files, 1,915 functions in core+server; 1,305 acceptance tests; 87,189 lines.
- Frontend: 69,376 lines TS/TSX (production `record`+`stage` components ≈ 12,089; lab/explore ≈ 25,263), 4,244 test lines.
- Docs: `docs/turbotab-next` 31,254 lines of Markdown.
- Git: 363 commits on `turbotab-next` not in `main`; 146 on 2026-10-05 alone; 18 worktrees; 130 branches.
- Tokens: 43 workflows with totals, 98.3M (`totalTokens` as the harness reports); 39 TurboTab-Next workflows since 2026-09-27 ≈ 92M.
