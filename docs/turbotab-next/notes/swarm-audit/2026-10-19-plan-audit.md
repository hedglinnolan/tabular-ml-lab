# Audit of the swarm plan (2026-10-19)

An independent Opus 5.5 auditor at max effort, read-only, given the project's intent in Nolan's words and asked to break the plan where it is weak (SWARM_PLAN, DRIVER_PLAN, PILOT_READINESS, CLAIMS_REGISTRY, R1_PLAN, and the code they rely on, at local turbotab-next 53428439). Its report, verbatim:

TL;DR
As written and built, the pilot will measure how reliably Claude analysts follow the protocol and what they cost. The corpus will hold a sparse log of how far a result moved when an analyst happened to change one thing, and that log depends on the path each analyst took. That is next to the middle layer, not the middle layer itself. The only signal of whether a decision is good is one dataset's cross-validated score under Predict, and it is stored without its sign. Today's three-axis markers exist only in the plan's prose. The anti-slop machinery has never been run against the failure Nolan fears most: a plausible empirical pattern that is false. The protective layer around the analyst (the lock, the seal, the driver, a record that knows when a result was refused) is strong. The fixes are three: a mechanical measurement arm beside the agents that spends no tokens, scoring of decision quality on the fixtures, and one full claim cycle piloted against planted false patterns.

Sound, and worth keeping:
- the driver's mechanical refusals and its sandbox;
- materiality is never stated by the analyst, and V is never set by a run;
- no check passes what it could not check;
- the dry run's outcomes were written before it ran;
- the fixtures' reading checks (units, code or amount), which are real tests of measurement decisions;
- crossing model with persona;
- SU-A's hand replication of Sullivan (it agrees with an R-only reference to 1e-11, and its post-hoc refits are timestamped). It is the model to copy.

TOP RISKS, ranked by threat to the intent

1. The corpus records sensitivity along agent paths, not goodness.
- What is wrong:
  - Under Predict, a decision gets a materiality only when it alone is a fit's change (assemble.py 64-73). The decisions that shape the first fit have none unless the analyst revisits them.
  - The value is unsigned |mean Δ|/SD(Δ), which grows with √n across datasets. The sign and the raw Δ survive only in a words string (materiality.py 555-563).
  - Under Estimate, the specification curve varies one choice at a time and refits only linear or logistic models. It refuses outright when the plan holds imputation, an energy model, a declared form, a scale score, omics normalization, detection-limit fills, batch correction or a marginal measure (mattered.py 58-72). Those are the measurement-axis decisions added today.
  - Sensitivity never says which option is nearer the truth.
- Amendment:
  - Name two kinds of claim. A surfacing claim ("D matters here, or not") feeds the Decide and Confirm tiers and is graded on materiality across held-out datasets. A choice claim ("O beats O′ on loss L") feeds defaults and rankings and is graded only against known truth.
  - After every run, a sweep that spends no tokens changes every offered decision one at a time, around the analyst's plan and around the default plan. It stores the signed Δ in the loss's units, with the interval from the engine's own corrected_t (models/validation.py 545).
  - Write §9.3's simulations down before running them: plasmode designs built from corpus datasets, the losses they grade, and the number of replicates.

2. The slop gate is untested on the real failure.
- What is wrong:
  - The registry's decoys are text perturbations: overreach, a moved constant, a flipped direction, a swapped mechanism or citation (research/registry/decoys.py 6-20). Vacuous decoys (true, but deciding nothing) are not generated.
  - The pilot runs no theorist and no experimentalist.
  - Every grader is Claude, and in the backtest the code audit found 12 defects that both blind referees missed.
  - A false empirical pattern is not a perturbation of anything, so these decoys cannot stand in for it.
- Amendment:
  - Build null-corpus decoys. Give a theorist the corpus with its regime markers permuted across datasets. Every claim it proposes is then false by construction. Route those claims blind through the experimentalist and the referees; the share promoted is the end-to-end false-promotion rate. Plant one true effect to measure power.
  - Pilot one small cycle of this.
  - Size the held-out shelf per regime stratum, fix its confirmation statistic in advance, and budget its reuse across claims (Dwork et al., Science 2015).

3. The pilot cannot see decision quality.
- What is wrong:
  - §7.1, and the reason half of §7.2, hold by construction: the driver blocks changes after the seal and demands a non-empty reason.
  - The Predict fixture's result is not checked at all (dry_run.py 99-100), although PILOT_READINESS §2 quotes "declared C 0.7419".
  - The Estimate fixture's known answer is a null effect, which cannot detect attenuation, and its first-ranked multiple-imputation path passes by refusal (PILOT_READINESS §6).
  - A run that reports nothing passes known_answer and null_decoy (checks.py 316-317, 399-402).
  - The seal protects the decoy, so the decoy tests the engine's intervals, not the analyst. Its seven runs also draw from one seed (SplitSpec.seed = 0), so they are close to one test with about a 2.5% false-finding rate.
- Amendment:
  - Replay each run's final decision log as a plan on 20 or more fresh draws from the fixture's generator. Report regret against the true risk (Predict), or bias and coverage (Estimate).
  - Give the Estimate fixture a non-null effect, a planted confounder and collider, and missingness that depends on the outcome.
  - Report refusals as their own rate, never as passes.

4. The three axes are prose only, and the pilot barely varies on them.
- What is wrong:
  - record.Regime holds rows, events, blanks, the design and the lenses, under extra="forbid" (record.py 161-171). It has no measurement fields, no per-decision snapshots and no field for the analyst's account.
  - No question card has a measurement section, and neither intake.py nor analyst.md asks for one.
  - The §9 gates were written before the amendment.
  - All three real datasets are NHANES with fasting plasma glucose as the outcome. The glucose table's export and its imputation are "not recorded", in its own card's words.
- Amendment:
  - Gate the pilot on version 2 of the record, measurement sections on the cards with their quotes checked, and the account fields. Otherwise, declare the pilot's corpus unusable for three-axis claims.
  - Swap NHANES glucose for metabolomics_untargeted.csv, already in the repo, which has detection-limit censoring, run-order drift, QC samples and batches.
  - For NHANES papers, read the NCHS laboratory documentation, not only the paper.
  - Add these markers: reliability, events per parameter, Kish's effective n, the spread of cluster sizes, top-coding, and under Predict how each predictor is measured when the model is used. Bang's score is meant for self-report, while NHANES BMI is measured at the exam.
  - The units and whether a column was already log-transformed are guesses by the engine. Record each reading's basis, and settle a conflict between the declared and the read value before calling it a finding.

5. "Agents give a better prior" is never tested, and half the exploration term is missing.
- What is wrong:
  - Analysts can only pick options the server serves (options.py, analyst.md), so their exploration is bounded by our own menu.
  - No baseline arm exists, although the scripted analyst does.
  - "Local sweeps around claims" appear nowhere in SWARM_PLAN. Only the explorer persona is written down, and its exploration rate can be neither measured nor tuned.
  - Replication is unscored: assemble.py 1450 always writes null, and §7 has no replication line. SU-A answered the provenance question better, with a mostly pre-registered sweep of the paper's unstated decisions over 54 projects (sullivan2021.md §4, §8).
- Amendment:
  - Run two arms that spend no tokens on every pilot dataset, one taking the default path and one taking random options. Compare them with the agents per unit of cost: material decisions found, and regret on the fixtures.
  - Make design 1's product the SU-A sweep, with the analysts as the many-analysts arm.
  - Later, allocate runs across regime cells by uncertainty times pull, with a fixed random share as the exploration term you can tune.

6. The pre-registration is loose and contradicts itself.
- What is wrong:
  - §7.3's "in the chosen configuration" lets any of the seven cells be picked after the results are in.
  - SWARM_PLAN 286-287 and PILOT_READINESS §2 still run Sonnet only if Haiku fails. SWARM_PLAN line 289 says the runs are throttled and line 267 says unthrottled. The plan's header still says nothing is built.
  - One run per cell cannot separate the persona's effect from sampling noise.
  - The robustness line charges engine failures to the models.
  - RF-2 and RF-3 may merge while the pilot runs (R1_PLAN §3).
  - The plan's author will also judge its outcome.
- Amendment:
  - Keep one current spec.
  - Pre-register the unit (each model, pooled over personas), the paired statistic, the decision rule and the rule for stopping early, and let a script apply the rule.
  - Add two replicate runs per cell on two datasets, and randomize the run order.
  - Freeze the engine commit for the pilot.
  - Classify stuck runs without knowing which model ran.
  - State the expected false failures in advance (about 2.5% for the decoy, 5% for coverage of the null) and how a miss will be settled.

7. Exploration done for show goes unaudited.
- What is wrong:
  - preview takes no reason (driver.py 1567-1569), yet it counts as examining an alternative. One preview is enough to pass single_pass (checks.py 244-263).
  - The referee's sample is drawn from fits only (referee.py 41-43), so previews are never audited.
  - Any non-empty string counts as a reason.
- Amendment:
  - Require a reason on preview, and include previews in the referee's sample.
  - Report coverage of the decisions the engine rates material instead of counts.
  - Plant fabricated reasons in about 10% of the sample, written by another agent, to measure how often the referees catch them.

8. There is no closed loop for engine growth.
- What is wrong:
  - Nothing records an option the analyst needed and the engine lacks.
  - Stuck runs are classed only as defects.
  - §9.2 gives a claim work only when a real person got stuck on its decision, in a friction log. That excludes the blind spots the explorer exists to find.
- Amendment:
  - Add a want command that records the option, why, and the marker behind it.
  - Admit a new option only when three or more runs on two or more datasets want it and a prototype sweep shows it is material.
  - Cap the Decide-tier questions per reference journey; that cap is the bloat budget.
  - Widen pull to include the engine's judgment-graded defaults, ranked by measured materiality.

9. Sizing.
- What is wrong:
  - The measured trio runs the conventional persona on the 480-row, 11-column fixture. Bang is 31,126 rows by 54 columns, and Sullivan 39,156 by 30.
  - Fit budgets set from the trio will cut exploration short on the NHANES datasets.
  - Fits longer than about five minutes break the prompt cache (PILOT_READINESS §3).
  - How many runs go at once is not stated, and nothing sizes the 18 Sonnet runs.
- Amendment:
  - Run a scripted fit on each dataset to set a budget per dataset.
  - Make the trio span the worst case (the explorer persona on Bang).
  - Read the usage meter before and after the trio.

ADDITIONALLY MEASURE OR PRE-REGISTER (beyond risks 3 and 6)
- The share of offered decisions that have a measured materiality, agent paths against the sweep, and coverage of the decisions the engine rates material.
- The spread between replicate runs in one cell, against the spread between personas.
- How often the referees catch the planted reasons, and how well they agree with Nolan's spot checks.
- The usage meter's change per run, by model, persona and table size.
- Replay: each record's result reproduces from its decision log in a fresh server.
- One claim cycle on the pilot corpus and one on a permuted null corpus, with the claims proposed and promoted from each.
- Wants per run, and a capability-gap class for stuck runs.

CUT OR DEFER
- §7.1 and §7.2 as acceptance lines; keep them as regression checks.
- Agent runs on the null decoy. Run it with the scripted analyst instead, since it tests the engine.
- Five of the six Opus ceiling runs, until a quality score exists; keep one to measure spend.
- NHANES glucose, until its table is rebuilt from the NCHS files and Nolan's walkthrough is done.
- The production swarm's mix, and the marker-tagged account, until the record holds the markers.

QUESTIONS ONLY NOLAN CAN ANSWER
1. Is the first middle-layer product "which decisions deserve a user's attention" (surfacing, which the corpus can reach) or "which option is better" (choice, which needs simulated truth)?
2. For your glucose project, is "good" a score on a random split, or performance in a later survey cycle?
3. Will you trade some agent runs for sweeps that cost no tokens but bring fan noise?
4. Could a human methodologist, perhaps a colleague, review a blind sample of promoted claims each sprint, so that Claude is not the only grader of Claude?
5. How many hours a sprint can you give to walkthroughs and spot checks? Both the pull rule and the referee audit depend on them.

MY THREE LEAST CERTAIN JUDGMENTS
1. How sparse Predict materiality will really be. After the first Fit the server refits on each change, so agents may isolate more decisions than I assume, and no real agent run exists yet to check.
2. That a final decision log can be replayed as a plan on fresh draws without the engine asking its data-dependent readings again. I did not try it.
3. The cost arithmetic. I assume the usage meter weights models differently, which I did not check for this account. I guessed Sonnet at 0.5-2M fresh tokens a run. That puts the 42 runs plus the referees at roughly 23-78M, against about 55M for 40% of a week at this week's rate.

Paths are relative to /Users/nhedglin/tabular-ml-lab. This was a read-only audit: nothing was edited, run or committed.
