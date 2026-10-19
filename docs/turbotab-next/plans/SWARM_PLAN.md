# The research swarm: what it measures, how, and the pilot

Version 3, written 2026-10-19 by the orchestrator. It revises version 2 (the same morning) after a
second independent review (`notes/swarm-audit/2026-10-19-plan-review-2.md`, a Fable 5.1 reader at
xhigh effort), which found that version 2 answered the first audit's structure
(`notes/swarm-audit/2026-10-19-plan-audit.md`) but left four load-bearing pieces as names without
numbers. Nolan ruled on both reviews' questions the same day. Earlier versions are in git history.
Under the swarm program (CLAIMS_REGISTRY §8 and §11) and Nolan's go-ahead of 2026-10-18.

## TL;DR

- **The goal:** the middle layer, the learning mechanics of tabular modeling. Nolan's question:
  "is this modeling decision even a good decision regardless of whether it is mathematically
  valid?" A claim answers it on three axes: the regime, the measurement model, and the purpose
  with its loss.
- **Two kinds of claim:**
  - a **surfacing** claim ("this decision matters here, or does not") says where a user's
    attention belongs; the corpus and the sweeps can reach it, and it comes first (Nolan,
    2026-10-19);
  - a **choice** claim ("this option beats that one on this loss") is graded only against known
    truth.
- **How:** a search over modeling paths. AI analysts give the prior ("the analysis that a modeler
  does happens on rails for a specific domain"), with a persona that deliberately leaves the
  rails; mechanical arms that spend no tokens do the measuring, around a few plans per dataset.
- **Against slop:**
  - "material" is defined in each loss's own units before anything runs;
  - choice claims are graded only against known truth;
  - the claim pipeline is run beside a shuffled corpus, to measure how much of what it proposes
    comes from its prior rather than the data;
  - Nolan blind-checks the referees.
- **The pilot:** five datasets, 36 agent runs, after the build in §10:
  - Bang 2009, Sullivan 2021, Nolan's NHANES glucose rebuilt and tested on a later cycle, a
    hardened Predict fixture, a new Estimate fixture;
  - Haiku 5.5 and Sonnet 5.5 under three personas, Opus 5.5 as a reference on the Estimate
    fixture.

  It decides the production analyst; cost breaks a tie (Nolan, 2026-10-19). Its claim cycle is a
  plumbing test.
- **Built today:** the driver, version 1 of the record, the six checks, the intake, the dry run
  (18 checks agree), and the replay of a decision log (`core/export/replay.py`). **Not built:**
  everything in §10.

## 1. What the swarm is for

**The middle layer.** A claim says, for a region of three axes, whether a decision matters and,
where truth allows, which option is better:
- **the regime:** the structure of the table and the question;
- **the measurement model:** how each number came to be (Nolan: "regime only answers half the
  question, the other half that goes unanswered is the instrumentation of how the data was
  recorded");
- **the purpose and its loss:** the purpose, the score or estimand it is judged on, and the use.

Claims are keyed on the axes, not on the domain; a claim's evidence records its domain as
provenance. A claim learned in one domain is tested where possible in another that matches its
markers: the middle layer's strongest check.

**Two kinds of claim:**
- **Surfacing:** "in this region, decision D moves the result by a material amount (or does
  not)", material as defined in §7. It feeds the app's Decide and Confirm tiers. It is graded on
  datasets in the region held out from the corpus that proposed it (§9).
- **Choice:** "in this region, option O beats O′ on loss L by at least d". It feeds defaults and
  rankings, and is graded only against known truth: the fixtures and, after the pilot, plasmode
  simulations (§9).

A promoted claim keys the app through a regime predicate evaluated when the decision is asked, so
it may rest only on **live** markers, those the engine reads before the decision; markers known only
after a run (the signal, the result) describe the evidence but cannot key a claim (§4).

**Three run designs** share one infrastructure:
1. **Replication.** Its product is a pre-registered sweep of the paper's unstated decisions, how
   far each moves the published number, as SU-A did for Sullivan
   (`notes/replication/sullivan2021.md` §4, §8). The analysts are its many-analysts arm.
2. **Agent analysts** on a dataset and question, under personas (§6).
3. **The full shelf:** every model family on every dataset, nested and sealed, no analyst; it
   waits for the heavy-run schedule.

**The closed loop.** A run that needs an option the engine lacks records it (`want`, §2). The
engine grows from wants by the rule in §9. Today's count of Decide-tier questions per reference
journey is recorded now as the baseline, so growth is visible; the cap on it is set after the
pilot from the pilot's wants (the orchestrator's call, Nolan, 2026-10-19).

**Tabled:** time-series forecasting as an analytic path (Nolan, 2026-10-19: "We table it for
now"). NHANES cycles are repeated cross-sections, so a later cycle is a temporal validation, which
§10 builds, not a forecast.

## 2. The driver

Built (DRIVER_PLAN D1, D2; `research/swarm/driver.py`). An analyst works through the app's own API
on a local-mode server in analyst mode, so every guard, refusal, claim and seal applies to it.

| Command | What it does | Change in this version |
|---|---|---|
| `ask` | the open question: its words, options, readings | none |
| `preview <option> --reason` | what recording it would do; nothing recorded | **needs a reason** |
| `answer <option> --reason --marker` | records the decision with its reason | **adds `--marker`** from a controlled vocabulary (§4) |
| `revisit <decision> --reason` | changes an earlier decision | none |
| `fit` | presses Fit; under Estimate it records the lock | none |
| `results` | what the app shows now | none |
| `seal <family> --reason` | opens the held-out rows, once | none |
| `export` | the methods, the checklist and the bundle | none |
| `want <what> --reason --marker` | **new:** an option or check the engine lacks | new |

**Refused mechanically:** a re-seal; any change after the seal reaching the result; every tool but
the driver's commands (the analyst type and its hook, wired locally); a run id not the caller's. A
run's label is neutral (`run_NN` and a secret part).

**The controlled vocabulary** for `--marker` and `want` (the three axes' marker names, §4, and a
short list of want kinds: an option, a check, a reading, an eligibility rule) is fixed before any
run, so tags can be counted.

**The fit budget** is set per dataset from a scripted fit on it before the pilot (§6).

## 3. Exploration before the lock, and none after the seal

Unchanged in principle (Nolan, 2026-10-16). Under Predict, discovery is on the cross-validated
scores, then one seal. Under Estimate, discovery is outcome-blind until Fit records the lock; a
change after it is a secondary analysis, logged and measured.

- **An examined alternative** is an option previewed or fitted before the choice, with a reason.
- **Coverage:** the share of the material decisions (§7) that the run examined, reported only
  where the dataset has at least three material decisions.
- **A single-pass run** (no alternative examined, one fit) is flagged and its rate reported.
- **The referee audit** samples a pre-registered number of attempts per run, previews included,
  and scores whether each reason is real and matches what the analyst was shown. Planted
  fabricated reasons (about one in ten of the sample, written by another agent) enter the
  referee's view only, never the record, to measure the referees' catch rate.

## 4. The corpus record

One JSON record per run, validated against `research/swarm/record.py`, in `research/corpus/`,
outside `turbotab/`, apart from the registry. Version 2 of the record adds what is marked new.

| Field | Contents | Source |
|---|---|---|
| `run` | id, design, engine commit, **model and persona (new; redacted for blind reviewers)**, effort, prompt version, start and end, tokens, fits | the driver; the transcript |
| `dataset` | id, URL, release date, license, SHA-256, codebook, **declared measurement markers (new, §5)** | intake |
| `question` | purpose, estimand or intended use, the paper for a replication | the card |
| `regime` | **the markers below, each tagged live or post hoc (new)** | the engine's readings |
| `decisions` | each decision: the option, those examined and how, the reason **with its marker tag and rejected alternative (new)**, the phase, the source | the decision log, the driver |
| `attempts` | each fit: the plan's digest, the decisions it read, its scores or estimate, **the live markers as they stood at that fit (new)** | GET /fits, the engine |
| `result` | the locked fit's sealed score, or the reported estimate on its measure's scale, or the refusal with its reason | the engine |
| `wants` | **new:** each `want`, its reason and tag | the driver |
| `account` | **new:** the end-of-run account, marked as written after the results | the analyst |
| `replay` | **new, fixtures only:** the final plan on fresh draws: regret, or bias and coverage (§7) | the replay scorer |
| `claims`, `surfacings`, `export`, `replication` | as in version 1 | the server; the replication sweep |

The sweeps (§6) are stored beside the records, keyed by the plan they sweep around, not inside a
run's record, since one sweep serves several runs.

**Markers in the pilot** are those the engine already reads and the declared ones from the intake:
- **Regime (live):** rows, rows analyzed, units where rows repeat, the effective size (events or
  the rarest class), events per parameter, the predictors by role and their ratio to the rows, the
  outcome's type and event share, missingness per column and the complete-case share, the design
  (clusters, strata, PSUs, weights), time order.
- **Measurement (live):** the units as read, and each reading's basis; detection-limit shares;
  batch and run-order columns; imputed copies and repeated measurement; the scale type where the
  engine reads it (the omics raw-scale reading).
- **Measurement (declared):** the instrument, assay or platform, preprocessing, units and recall
  method, from the intake (§5).
- **Purpose and loss (live):** the purpose, the primary score and why, the intended use and
  thresholds, the estimand, the multiplicity family.
- **Post hoc:** the signal (the cross-validated gain over the no-predictor baseline) and the
  result.

**Staged after the pilot:** markers the engine does not yet read as one value (Kish's effective n,
the condition index as a marker, error structure, reliability, how a predictor is measured at use).

**The analyst's account** sits beside the engine's record, never in its place:
- written at each decision, before the analyst has seen what mattered;
- tagged with a marker and the alternative it rejected;
- context for hypotheses, never evidence;
- audited for faithfulness (§3).

## 5. Datasets and intake

Built for version 1 (`research/swarm/intake.py`). Version 2 of the intake adds:
- **Declared measurement markers** (Nolan, 2026-10-19): read from the paper's methods and, for
  NHANES, the NCHS laboratory and questionnaire documentation; each quotes its passage, labeled
  declared, checked by a referee. What the source does not say is recorded as not stated, the
  material for the survey paper Nolan has in view. A declared marker that disagrees with a reading
  is settled by the evidence before it is called a finding.
- **A categorical eligibility rule** in the engine (exclude by level, with refused and don't-know
  codes handled explicitly, STROBE's "confirmed eligible"), so Bang's population can exclude
  diagnosed diabetes by its code (Nolan agreed, 2026-10-19).

**The pilot's five datasets:**
1. **Bang 2009** (Predict, NHANES 1999–2004, its own intake, 31,126 rows), its replication sweep
   built as an SU-A-sized package (§10).
2. **Sullivan 2021** (Estimate), its sweep done by hand (SU-A).
3. **Nolan's NHANES fasting glucose**, rebuilt from the NCHS files (Nolan, 2026-10-19: "Dare I
   suggest we rebuild the correct nhanes dataset we need?"):
   - developed on 2011–2016 and tested on 2017–March 2020, the pre-pandemic file whose design
     matches; 2021–2023 changed its sampling design (the cycles are the orchestrator's call);
   - each NCHS file and variable named, the fasting subsample's weights, a cycle column;
   - his 21 predictors mapped to their variables, plus others NHANES gives cheaply ("if we need
     more features than that to make an effective model and nhanes can provide it with little to no
     cost, then add them"), chosen from what is known before the fasting draw's result. Same-draw
     glycemic measures (HbA1c, insulin, an oral glucose tolerance test) are left out by design,
     since they would make the model a near-restatement of its outcome;
   - held out by a cycle cut-point (§10); its "good" is performance on that later cycle.
   - **A published benchmark** (Nolan: "a good north star ... and it briefs well to others"): one
     recent paper predicting fasting glucose or dysglycemia from NHANES, chosen for how fully it
     can be reproduced, becomes a replication target, so the app's number is compared on the
     paper's own population and outcome, not against a figure measured differently.
4. **The hardened Predict fixture:** `clinical_readmission_predict`'s generator with a
   nonlinearity the default path misses, an interaction, and a predictor measured differently when
   the model is used, so a path's choices change its regret.
5. **The new Estimate fixture:** a non-null effect on the reported measure, a planted confounder
   and a collider, and missingness in a covariate that depends on the observed outcome (missing at
   random), so complete cases are biased and imputation is not, and each choice has a gradable
   right answer.

**Mechanical only:** the null decoy (it tests the engine's intervals, since the seal protects it),
with distinct seeds per draw. **Deferred:** the measurement fixture
(`metabolomics_untargeted.csv`) until its question and truth are designed.

**The held-out shelf** of generator draws at varied parameters, needed for a true false-promotion
rate, is built after the pilot (§9).

## 6. The pilot

**What it decides:** the production analyst model, whether personas earn their place, and whether
agents beat the mechanical arms per unit of cost at finding what matters and choosing well.

**The personas** (approved by Nolan, 2026-10-19), stances toward the field's defaults; the
protocol is the same for all:
- **Conventional:** takes the domain's customary path where the engine ranks it sound.
- **Skeptical:** questions each default, examining at least one alternative where a decision could
  move the result.
- **Explorer:** deliberately tries the alternatives the field rarely takes.

**Agent runs** (36):

| Arm | Datasets | Runs |
|---|---|---|
| Haiku 5.5, high effort, each persona | all five | 15 |
| Sonnet 5.5, high effort, each persona | Bang, Sullivan, glucose, the Estimate fixture | 12 |
| Haiku replicates, two more per persona | the Estimate fixture | 6 |
| Opus 5.5, high effort, conventional | the Estimate fixture, three runs (the reference) | 3 |

The replicates measure run-to-run spread for one model on the fixture that decides quality; the
persona comparison is read against that spread and is secondary.

**Mechanical arms** (no tokens), per dataset:
- the default path and a random path;
- **the sweep:** each offered decision changed one at a time around the default plan and around
  one final plan per model, refits deduplicated by plan digest. Readings with one true answer (a
  column's units, code or amount) are graded for correctness, not swept. Each refit is a full
  pipeline fit through the driver on a project rebuilt from the plan by `core/export/replay.py`,
  scored at the sweep's fixed repeats (§7).

**The engine is frozen** at one commit for the pilot; nothing merges into the branch it runs on.

**Order:**
1. **Sizing:**
   - a scripted fit on each dataset sets its fit budget and the sweep's compute estimate;
   - the measured trio: Haiku and Sonnet with the explorer persona on Bang, the largest table, and
     Opus on the Estimate fixture, its only dataset;
   - Nolan reads his usage meter before and after the trio.
2. **The waves,** dataset by dataset, in randomized order within each wave, under the stopping
   rule (§7).
3. **The sweeps,** within the compute hours `prereg.json` caps.
4. **The claim cycle** (§9) on the pilot corpus and on its shuffled control.

**Compute and timing:** daytime, unthrottled (Nolan, 2026-10-19). Under the throttle, long
commands make agents miss their prompt cache and cost about four times as much. At most three
agent runs at once; each command short. The sweeps run with no agent, so their length costs
compute and fan noise, not tokens.

**Budget:** up to 40% of a fresh week. If the trio projects past it, the waves are cut in this
order: the Haiku replicates, then Sonnet on glucose, then Sonnet on Sullivan.

## 7. Pre-registration and acceptance

Written before the trio in `research/swarm/prereg.json` and applied by
`research/swarm/score_pilot.py`, not by the orchestrator.

**Regression checks** on every run (they hold by construction): seal integrity, schema validity, a
reason on every preview and decision.

**Material, defined per purpose in the loss's units** (the sweep scores each refit with repeated
5-fold cross-validation, 5 repeats, so the corrected interval has enough folds):
- **Predict:** a decision is material where the signed paired difference in the primary score has
  a corrected repeated k-fold interval excluding zero, and its size is at least half the primary
  score's own interval half-width.
- **Estimate:** a decision is material where it shifts the reported estimate by at least half the
  estimate's interval half-width.

**Decision quality, anchored to known truth, not to a reference run:**
- **The Estimate fixture decides it** (the pre-registered decisive fixture):
  - absolute bias of the reported estimate within b₀ of the truth;
  - interval coverage within the stated band over at least 20 fresh draws of the replayed plan.
- **The Predict fixture adds regret:** the replayed plan's loss against the oracle's on fresh
  draws, within r₀.
- b₀, r₀ and the band are fixed in `prereg.json` from the generators before the trio.
- Opus's three runs are a reference arm, reported beside the margins, not the margin itself.

**The other statistics:**
- coverage of the material decisions, paired per dataset between models and against the
  mechanical arms;
- reason faithfulness and the referees' catch rate of planted reasons;
- robustness: runs reaching the export without help. Each stuck run is classed (engine defect,
  driver defect, capability gap, analyst failure) by a reader shown a record with the model
  redacted; only analyst failures count against a model;
- refusals, as their own rate, never as passes;
- cost per useful result: fresh tokens per material decision covered.

**The decision rule:**
- A model is acceptable if it is within the margins on the Estimate fixture, its reasons are
  faithful at least 4 times in 5, and at least 9 runs in 10 reach the export.
- Of acceptable models, the cheaper is the production analyst; cost breaks a tie (Nolan,
  2026-10-19). If none is acceptable, the plan returns to Nolan.
- A persona is kept if it raises coverage beyond the Haiku replicates' spread on the same fixture.

**Expected false alarms,** stated in advance: about 2.5% for the decoy's interval, and the
coverage band's own miss rate. A miss on a control is settled by fresh seeds, never by editing an
expectation.

**Early stopping:** a model outside the margins on the Estimate fixture after its persona runs
stops there; a spend projecting past the budget stops the waves.

## 8. What the engine must provide

Built: GET /fits (with the locked fit), GET /results, GET /mattered, analyst mode, provenance, and
`core/export/replay.py`, which rebuilds a project from its `decisions.jsonl` in a fresh home and
refuses inputs whose SHA-256 differs.

Needed (§10):
- **Replay on a fresh draw:** a flag that lets replay take a new draw of the same generator; a
  rule for records tied to findings that may not fire on the new bytes (`apply_repair`); and a
  scorer that predicts on new rows (no route serves predictions on new rows today; until one does,
  regret is the Brier score against the oracle on each draw's sealed held-out rows, which is
  noisier).
- **The sweep's refits under the full pipeline,** through the driver, since GET /mattered refits
  only linear and logistic models and refuses imputation, energy models, declared forms, scale
  scores, omics normalization, detection-limit fills, batch correction and marginal measures
  (`core/mattered.py`).
- **The signed change** in the loss's units with its corrected interval: per-fold scores are
  already in the record, so this is computed from them at the sweep's repeats.
- **A temporal seal by cut-point:** hold out every row at or after a time value (the latest
  cycle), beside today's chronological draw by fraction (`set_temporal`).
- **The categorical eligibility rule** (§5).

## 9. From corpus to the middle layer

The corpus never reaches a user. A pattern becomes advice only through the registry
(CLAIMS_REGISTRY §8):
1. **Propose.** A theorist (Opus) reads the corpus and the sweeps and proposes surfacing or choice
   claims on live markers. Each enters as proposed and unsupported.
2. **Pull.** A claim gets work if it answers a decision a person got stuck on, or would change one
   of the engine's judgment-graded defaults, ranked by the sweeps' measured materiality, which
   lets in blind spots no one yet knows to be stuck on.
3. **Try to break it:**
   - a surfacing claim on held-out datasets in its region;
   - a choice claim on known truth (fixtures; plasmode simulations, whose designs, losses and
     replicate counts are written down before they run);
   - across domains, where markers match;
   - only these mechanical results change a claim's status.
4. **Grade.** Promotion means a grade of replicated or above, with the two-referee gate and the
   basis audit; an author's self-grade of judgment is not a promotion.
5. **The shuffled control.** Each cycle also runs on a copy of the corpus whose markers are
   permuted across datasets, with the whole pipeline, test stage included, inside that shuffled
   world, so its claims are false by construction. One true effect is planted in it, to show the
   pipeline can find one.

**At pilot scale the cycle is a plumbing test** (Nolan accepted, 2026-10-19: "Ultimately, I do
want us to get into discovering real new theories that are not slop"). Five datasets leave nothing
held out to promote on, so its pre-stated outputs are:
- the number of proposals from the real corpus and from the shuffled one;
- zero promotions in both, as expected;
- the planted effect's claim proposed and routed;
- **the prior-driven share:** the overlap between what the theorist proposes from the real corpus
  and from the shuffled one. A high overlap means the proposals come from its prior, not the data,
  which is Nolan's fear measured directly.

**After the pilot:** the held-out shelf of generator draws at varied parameters, on which the true
false-promotion rate is measured end to end.

**Nolan's time** (a couple of hours a week, 2026-10-19), pointed where a person adds what no agent
can:
- blind-rating about ten referee verdicts against the records;
- a real-versus-shuffled test: can he tell which corpus a set of theorist proposals came from?
- one walkthrough of a Haiku record.

Once claims are promoted, a blind review of a sample of them each sprint.

**The engine's growth, from wants:** an option enters only when at least three runs on at least
two datasets want it and a prototype sweep shows it is material.

## 10. Order of work

After Nolan's usage resets (Thursday 1 pm). Each package: an Opus builder, an adversarial verifier,
at most two repairs, every command kept under about four minutes and whole suites left to CI.

1. **The record's version 2 and the driver** (buildable now from existing code):
   - the fields in §4 with live and post hoc tags;
   - `--reason` on preview, `--marker`, `want`, and the controlled vocabulary;
   - model and persona, with redaction for blind readers;
   - the signed change from per-fold scores;
   - the shuffled-corpus tool.
2. **Replay on fresh draws and the fixtures:**
   - the fresh-draw flag and the findings rule on `core/export/replay.py`;
   - predictions on new rows for the scorer;
   - the hardened Predict fixture and the new Estimate fixture with their generators;
   - the decoy's distinct seeds.
3. **The sweep runner and the mechanical arms:** dedupe by plan digest, the default and random
   paths, compute sizing from the scripted fits.
4. **Two engine additions:** the temporal seal by cut-point and the categorical eligibility rule.
5. **Intake's version 2:**
   - the declared measurement markers;
   - the glucose table rebuilt from the NCHS files;
   - Bang's eligibility by code;
   - the benchmark paper's card.
6. **The analysis plan:**
   - RB (R1_PLAN R1-B);
   - `prereg.json` and `score_pilot.py`, with the margins computed from the generators;
   - the theorist's and experimentalist's prompts;
   - the Bang replication sweep, an SU-A-sized package.
7. **The live lockdown check and the analyst's model id,** in a fresh session on the home machine.

Then the pilot (§6), then its claim cycle.

## 11. Rulings and what is open

**Nolan's rulings this plan rests on** (2026-10-16 to 2026-10-19):
- exploration before the lock, none after the seal;
- Haiku 5.5 as the analysts' model, Sonnet 5.5 opened for the pilot, three personas;
- daytime runs within 40% of a fresh week;
- the three axes;
- declared markers from the papers;
- the analyst's audited account;
- surfacing first;
- the glucose project rebuilt, judged on a later cycle, with a published benchmark;
- the claim cycle as a plumbing test;
- cost breaks a tie;
- forecasting tabled;
- the categorical eligibility rule.

**The orchestrator's calls,** recorded so he can overrule them:
- the cycles;
- the arms and their counts;
- the definition of material and the margins;
- the Decide cap, deferred to after the pilot;
- the admission rule;
- the deferred measurement fixture.

**Reviews:** version 3 goes to a third reader, fresh, before §10 is built. Revision continues until
fresh readers stop finding flaws (Nolan, 2026-10-19: "Think of this like editing an english essay,
it's always good to revisit and revise until we are happy").
