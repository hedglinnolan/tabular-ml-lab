# Third review of the swarm plan, version 3 (2026-10-19)

A fresh Fable 5.1 reader at xhigh effort, read-only, given the project's intent in Nolan's words, both earlier reviews and SWARM_PLAN version 3 (3e4aa6e4), asked whether it is ready to build. Its report, verbatim:

I have what I need. Here is the review.

**THIRD REVIEW OF SWARM_PLAN VERSION 3 (2026-10-19), read cold, read-only**

**TL;DR**

Version 3 is the first version whose load-bearing pieces are numbers and mechanisms rather than names, and every code premise I checked holds (replay.py folds the log and refuses a changed SHA-256; `apply_repair` is a decision kind, repairs.py 1788; the chronological draw is by fraction, seal.py 428-434 and SplitSpec.holdout 0–0.4; ExclusionRule is range-only, decisions.py 176-182; per-fold scores are in the record, dry-run run_02; `Regime.requires` predicates exist, claims/_schema.py 92-103; mattered.py refuses the listed steps). What remains is a set of rules the pre-registration still leaves to whoever scores the pilot, and one contradiction between §4 and §5. None needs a redesign; all are paragraph-sized edits to §5–§7 and §9 and lines in `prereg.json`. **READY AFTER SMALL FIXES:** (1) the unit of the decision rule on the Estimate fixture; (2) how "material" is computed for a decision that changes the analyzed rows; (3) coverage's denominator and which plan per model is swept; (4) the rule that sets b₀ and r₀, and at least 100 draws for the coverage band; (5) the use-time predictor contradiction; (6) the agents-versus-mechanical comparison restated; (7) the prior-driven share's matching rule and marker-keyed proposals; (8) RF-1's status and which Bang number the sweep targets. §10 items 1–2 can start now; items 3 and 6 wait for these edits.

**The second review's ten flaws and its smallest pilot**

1. Claim cycle measures nothing: **answered** (promotion = replicated, §9.4; test stage inside the shuffled world; plumbing outputs; prior-driven share), overlap measure undefined (flaw 7 below).
2. Material undefined: **partly**; defined in loss units (§7), but undefined for row-changing decisions and the denominator of coverage is unstated (flaws 2, 3).
3. Sweep unsized and replay-dependent: **answered in design**; which plan per model, and truncating the replayed log before the seal, unstated (flaws 3, 9).
4. Fixtures will not separate models: **answered in design**; the Predict fixture's use-time predictor contradicts §4 (flaw 5).
5. Pre-registration arithmetic: **mostly answered**; the rule's unit and the 20-draw band remain (flaws 1, 4).
6. Replay less unproven: **answered**, and verified.
7. Live versus post hoc markers: **answered**, and verified.
8. Glucose later cycle: **answered in design** (cut-point seal); the weights across the cut are not addressed (flaw 10).
9. Smaller items: **answered** (eligibility rule, vocabulary, redaction, planted reasons in view only; the §7.6 citation is gone).
10. Nolan's hours: **answered**.
Smallest pilot: **adopted and extended** to 36 by glucose, per Nolan's ruling; the counts add up.

**Remaining or new flaws, ranked**

1. **The decision rule has no unit.** §7: "a model is acceptable if it is within the margins on the Estimate fixture." Haiku has 9 replayed plans on that fixture, Sonnet 3, Opus 3 (§6 table). Per plan, Haiku has three times the chances to fail; pooled, persona differences vanish and a biased plan hides in the mean. Revision: in `prereg.json`, "a model is within the margins when at least k of its n plans are," with k/n fixed per model before the trio, and a refused plan counted as neither pass nor fail but in the refusal rate.

2. **"Material" under Predict is undefined for a decision that changes the rows.** `corrected_t` pairs fold scores "over the same folds of every repeat" (validation.py 545-566); the record pairs fits by the `folds` digest (record.py 304; both dry-run attempts share digest 683dfb10). An exclusion or eligibility rule, or complete cases against a fill, changes the universe and the draw, so there is no shared fold, and these are exactly Bang's and Sullivan's decisions. Revision: pre-register that row-changing decisions are scored by an unpaired interval over the two plans' fold scores, labeled unpaired in the sweep, or paired on the row intersection with a fixed seed; say which.

3. **Coverage's denominator and the swept plan are both open.** §3: coverage is the share of "the material decisions (§7)" a run examined; §6 sweeps "around the default plan and around one final plan per model." Three runs per model per dataset, one swept: for the other two, which sweep defines material? And which of the three plans is swept is unnamed, so it can be chosen after the results. Revision: coverage against the union of material decisions from that dataset's sweeps; the swept plan is the conventional persona's, named now.

4. **b₀, r₀ and the band have no rule, and 20 draws cannot test coverage.** §7 says they are "fixed from the generators" but not how; anyone can pick a b₀ every plan passes. At n = 20 and nominal 95%, the binomial band is about 17–20 of 20, and a plan whose true coverage is 80% passes about 41% of the time. Revision: b₀ = half the smallest planted bias (collider kept, confounder dropped, complete cases) and at least four times the oracle plan's bias standard error over the draws, so the oracle passes and every planted wrong path fails; r₀ likewise from the default path's regret on the planted nonlinearity; ≥100 draws for coverage (the fixture is small), band stated as binomial at that n.

5. **§5 contradicts §4 on the use-time predictor.** §5 item 4 plants "a predictor measured differently when the model is used, so a path's choices change its regret"; §4 stages "how a predictor is measured at use" as a marker after the pilot, and §7's regret is on "fresh draws" of one generator. The analyst cannot learn it, and fresh draws from the development generator cannot show it. Revision: either the card declares the use-time measurement (intake v2) and the scorer draws from a use-time generator with the oracle defined there, or drop it from the fixture.

6. **"Agents beat the mechanical arms at finding what matters" is not a comparison.** Coverage is the share of material decisions *examined* (§3); the default and random paths examine nothing, so their coverage is zero by construction, and the sweep finds everything by definition. Revision: state two comparisons: choosing well (agent plans against default and random plans on regret, bias and coverage on the fixtures, and on the sealed score on real data); finding what matters (agent coverage per fresh token against the sweep's fit-hours).

7. **The shuffled control is false by construction only for marker-keyed claims, and "overlap" has no procedure.** A proposal with no regime condition ("imputation versus complete cases is material") survives a marker permutation and may be true. The prior-driven share needs a matching rule, else a reader decides what counts as the same proposal. Revision: the theorist's output is structured (decision, live markers, direction, size); a proposal with no live marker is not a surfacing claim (§1 already says a claim keys through a predicate); overlap is computed by a script on those fields. Also write the planted effect's mechanics: a generator appends synthetic records to the shuffled copy only, listed in a sealed file.

8. **The cut order voids the persona decision silently.** §6 cuts the Haiku replicates first; §7's persona rule and the run-to-run spread rest on them alone. Revision: say that the cut defers the persona question rather than deciding it, and name the spread statistic (range of three replicates' coverage, in steps of 1/m).

9. **The sweep's replay must stop before the seal.** replay.py folds the whole log; a replayed sealed (or locked) project refuses the sweep's `revisit`. Revision: the sweep runner truncates the log before `open_seal` (Estimate: `lock_plan`), revisits, fits, reads cross-validated scores. One line in §8 and §10 item 3.

10. **Compute: the engine's comparison substrate runs anyway.** Under Predict with more than one family, comparisons run at ≥10×5 whatever the validation choice (validation.py 383-386, folds.py 49), so a sweep refit costs about 50 fits per family beyond the plan's 25. The scripted fit will show it; the prereg cap should be set knowing it.

11. **Glucose: the weights across the cut.** 2011–2016 fasting weights are WTSAF2YR/3; the 2017–March 2020 file carries its own, and NCHS does not treat it as a standard cycle. Whether the later-cycle score is population-weighted or participant-level changes the intake and the number Nolan briefs. §5 does not say.

12. **Bang's "SU-A-sized" package holds only if the target is the odds ratios.** The source audit (notes/replication/bang2009.md §8, gaps 2–3) finds no points-score derivation and no fixed-rule scoring on new rows; the paper's headline numbers (the score's AUC, sensitivity at cut points) are out of the engine's reach, and §8 does not list them. Name the number.

13. **R1_PLAN requires RF-1 before the pilot** (lines 109-113: "so no analyst decides on a false sentence"); §10 omits it and lists RB without saying why the pilot needs it. Reconcile in one of the two documents.

14. **No measurement-axis contrast exists in the pilot.** Three NHANES tables share one instrument; the measurement fixture is deferred. The declared markers will be recorded but constant. Acceptable for a pilot, but §6 should say the pilot yields regime-axis surfacing only, since the measurement axis is Nolan's own emphasis.

**Does the pilot decide what it says?** The production analyst: yes, once flaws 1 and 4 are fixed; as written, a Haiku pass is near-certain and cost decides, so the fixture must bite. Personas: weakly and secondarily, as the plan says, and not at all if the replicates are cut. Agents against mechanical arms: not as stated (flaw 6). First surfacing evidence: yes, the signed materiality distribution over the offered decisions on Bang, glucose and two fixtures, within one regime cell. Sonnet and Opus never run on the Predict fixture, so regret compares no models; say so or add Sonnet ×3 there.

**§10's order and size.** The order is right: item 2's generators must precede item 6's margins, item 4 precedes item 5's cards. Items 1–2 are buildable now from code that exists. Item 4's eligibility rule is a guard touching decisions, the rows stage, options, the card, the methods sentence and the registry; the orchestrator's own record says guard waves cost about three times a wording wave (R0b: 9.76M against 3M). Item 5 is a research package (NCHS downloads, quoted markers for three datasets) and item 6 holds four unrelated pieces. An honest size is 35–60M fresh, not one week's remainder, and item 6 should split into RB, prereg plus scorer, and the Bang sweep.

**What I would cut or move.** Move the use-time predictor out of the Predict fixture unless its generator and oracle are written (flaw 5). Split §10 item 6. Add Sonnet ×3 on the Predict fixture or state that regret compares no models. Nothing else needs cutting.

**The path from plumbing to real theories (what no version addresses).** CLAIMS_REGISTRY §2.4 makes "replicated" need held-out cases from at least two datasets not used to fit the rule, so a surfacing claim needs a regime cell with at least three datasets. No version says what the first real cycle needs: a dataset roster by regime cell, the step that turns a theorist's prose into a code predicate (§2.2, "the regime is code"), the held-out shelf's reuse budget (the first audit's Dwork point, dropped since), the run-allocation rule that is the search's exploration term (uncertainty × pull with a fixed random share, also dropped), and whether the sweep's definition of material replaces or recalibrates the app's own bands (materiality.py) so a promoted surfacing claim can key the Confirm tier. A short §9 section, "the first real cycle," with these five items would close it.

**Questions only the owner can answer.** (1) Is the glucose project's later-cycle score population-weighted or participant-level? (2) For Bang, is the sweep's target the odds ratios or the score's performance? (3) Is RF-1 still required before the pilot? (4) Does he accept that the pilot yields no measurement-axis evidence? (5) Should his one walkthrough be of the Haiku glucose record, the project he will brief?

**Three least certain judgments.** (1) The size estimate rests on two recorded overruns, not on these packages. (2) The engine may draw folds with the same seed over the shared rows so that pairing on the intersection is nearly valid; I did not run it. (3) My band arithmetic assumes a 95% nominal interval and that 20 is the plan's intended n rather than a floor the prereg will raise.

**Sound, keep:** the two kinds of claim and surfacing first; the live/post hoc tagging; replay.py as the base; the frozen engine; the scripted decision rule; the referee audit with planted reasons; the shuffled control with the prior-driven share; the plumbing-test framing with pre-stated outputs; the decoy mechanical with distinct seeds; Nolan's three uses of his hours.

Read-only: nothing edited, run beyond greps and JSON reads, or committed. Paths relative to /Users/nhedglin/tabular-ml-lab.
