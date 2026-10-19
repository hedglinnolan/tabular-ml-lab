# The research swarm: what it measures, how, and the pilot

Version 5, written 2026-10-19 by the orchestrator. It makes the small fixes a fourth independent
review (`notes/swarm-audit/2026-10-19-plan-review-4.md`, a fresh Fable 5.1 reader at xhigh effort)
found version 4 needed ("a buildable plan ... none needs a redesign"), with Nolan's answers of the
same day. Earlier reviews: `2026-10-19-plan-audit.md` (version 1), `-plan-review-2.md` (version 2),
`-plan-review-3.md` (version 3); earlier versions are in git history. **Stopping rule for review**
(Nolan agreed, 2026-10-19): one last fresh reader checks only whether anything left changes the
build's design; if nothing does, revision stops and the build starts, since what a further reader
finds is cheaper to fix in the build than on paper. Under the swarm program (CLAIMS_REGISTRY §8 and §11) and
Nolan's go-ahead of 2026-10-18.

## TL;DR

- **The goal:** the middle layer, the learning mechanics of tabular modeling. Nolan's question:
  "is this modeling decision even a good decision regardless of whether it is mathematically
  valid?" A claim answers it on three axes: the regime, the measurement model, and the purpose
  with its loss.
- **Two kinds of claim:**
  - a **surfacing** claim ("this decision matters here, or does not") comes first;
  - a **choice** claim ("this option beats that one on this loss") is graded only against known
    truth.
- **How:** a search over modeling paths. AI analysts give the prior; a persona deliberately leaves
  the rails; mechanical arms that spend no tokens do the measuring.
- **Against slop:**
  - "material" and every margin are fixed in each loss's units, and applied by a script;
  - choice claims meet only known truth;
  - the claim pipeline runs beside a shuffled corpus, so the share of its proposals that come from
    its prior is measured;
  - Nolan blind-checks the referees.
- **The pilot: six datasets, 45 agent runs.**
  - The datasets:
    - Bang 2009 and Sullivan 2021;
    - Nolan's NHANES glucose, rebuilt and tested on a later cycle against Bang's published score;
    - a hardened Predict fixture, a new Estimate fixture and a measurement fixture.
  - The analysts: Haiku 5.5 and Sonnet 5.5 under three personas, and Opus 5.5 as a reference.
  - It decides the production analyst; cost breaks a tie. Its claim cycle is a plumbing test, and
    §9 says what the first real cycle needs.
- **When:** the build in §10 (36–58M fresh by its packages' estimates, about 40–65M with a
  buffer) and the pilot both run in the week after Nolan's usage resets, on a thin buffer he
  accepted (2026-10-19).
- **Built today:** the driver, version 1 of the record, the six checks, the intake, the dry run (18
  checks agree at 8a3e0fff, on today's fixtures; redone in package 2 on the hardened ones), replay of a decision log (`core/export/replay.py`), and R1-F's first batch of
  sentence fixes (RF-1).

## 1. What the swarm is for

**The middle layer.** A claim says, for a region of three axes, whether a decision matters and,
where truth allows, which option is better:
- **the regime:** the structure of the table and the question;
- **the measurement model:** how each number came to be (Nolan: "regime only answers half the
  question, the other half that goes unanswered is the instrumentation of how the data was
  recorded");
- **the purpose and its loss:** the purpose, the score or estimand it is judged on, and the use.

Claims are keyed on the axes, not the domain; a claim's evidence records its domain as provenance,
and a claim learned in one domain is tested where possible in another that matches its markers.

**Two kinds of claim:**
- **Surfacing:** "in this region, decision D moves the result by a material amount (or does
  not)", material as §7 defines it. It feeds the app's Decide and Confirm tiers, and is graded on
  held-out datasets in its region (§9).
- **Choice:** "in this region, option O beats O′ on loss L by at least d". It feeds defaults and
  rankings, and is graded only against known truth: the fixtures now, plasmode simulations after
  the pilot.

A promoted claim keys the app through a regime predicate, which is code (CLAIMS_REGISTRY §2.2),
evaluated when the decision is asked; so it rests only on **live** markers, read before the
decision. Markers known only after a run (the signal, the result) describe evidence but key
nothing.

**Three run designs** share one infrastructure:
1. **Replication:** a pre-registered sweep of the paper's unstated decisions, as SU-A did for
   Sullivan (`notes/replication/sullivan2021.md` §4, §8), with the analysts as its many-analysts
   arm.
2. **Agent analysts** under personas (§6).
3. **The full shelf:** every family on every dataset, no analyst; it waits for the heavy-run
   schedule.

**The closed loop.**
- A run that needs what the engine lacks records it (`want`, §2).
- The engine grows from wants by the rule in §9.
- Today's count of Decide-tier questions per reference journey is recorded now as the baseline;
  the cap on it is set after the pilot (the orchestrator's call).

**Tabled:** time-series forecasting (Nolan, 2026-10-19). A later NHANES cycle is a temporal
validation of repeated cross-sections, which §10 builds, not a forecast.

## 2. The driver

Built (DRIVER_PLAN D1, D2; `research/swarm/driver.py`). An analyst works through the app's own API
on a local-mode server in analyst mode, so every guard, refusal, claim and seal applies to it.

| Command | What it does | Change |
|---|---|---|
| `ask` | the open question: its words, options, readings | none |
| `preview <option> --reason` | what recording it would do; nothing recorded | **needs a reason** |
| `answer <option> --reason --marker` | records the decision with its reason | **adds `--marker`** from the controlled vocabulary |
| `revisit <decision> --reason` | changes an earlier decision | none |
| `fit` | presses Fit; under Estimate it records the lock | none |
| `results` | what the app shows now | none |
| `seal <family> --reason` | opens the held-out rows, once | none |
| `export` | the methods, the checklist and the bundle | none |
| `want <what> --reason --marker` | **new:** an option or check the engine lacks | new |

**Refused mechanically:** a re-seal; any change after the seal reaching the result; every tool but
the driver's commands (the analyst type and its hook, wired locally); a run id not the caller's. A
run's label is neutral (`run_NN` and a secret part).

**The controlled vocabulary** for `--marker` and `want` (the marker names of §4 and four want
kinds: an option, a check, a reading, an eligibility rule) is fixed before any run.

**The fit budget** per dataset comes from a scripted fit on it before the pilot (§6).

## 3. Exploration before the lock, and none after the seal

Unchanged in principle (Nolan, 2026-10-16). Under Predict, discovery is on the cross-validated
scores, then one seal. Under Estimate, discovery is outcome-blind until Fit records the lock.

- **An examined alternative** is an option previewed or fitted before the choice, with a reason.
- **Reach:** the share of a dataset's material decisions that a run examined. The denominator is
  the union of the decisions found material by all of that dataset's sweeps (§6); reach is
  reported only where that union holds at least three. ("Coverage" in this plan means only an
  interval's coverage.) A decision offered only on a path no sweep follows cannot be material, so
  reach is scored on the swept menu; §6 adds the explorer's plan on the Estimate fixture to widen
  it where the persona comparison is read.
- **A single-pass run** (no alternative examined, one fit) is flagged and its rate reported.
- **The referee audit** samples a pre-registered number of attempts per run, previews included,
  and scores whether each reason is real and matches what the analyst was shown. Planted
  fabricated reasons (about one in ten, written by another agent) enter the referee's view only,
  never the record.

## 4. The corpus record

One JSON record per run, validated against `research/swarm/record.py`, in `research/corpus/`,
outside `turbotab/`, apart from the registry. Version 2 of the record adds what is marked new.

| Field | Contents | Source |
|---|---|---|
| `run` | id, design, engine commit, **model and persona (new; redacted for blind readers)**, effort, prompt version, start and end, tokens, fits | the driver; the transcript |
| `dataset` | id, URL, release date, license, SHA-256, codebook, **declared measurement markers (new, §5)** | intake |
| `question` | purpose, estimand or intended use, the paper for a replication | the card |
| `regime` | **the markers below, each tagged live or post hoc (new)** | the engine's readings |
| `decisions` | each decision: the option, those examined and how, the reason **with its marker tag and rejected alternative (new)**, the phase, the source | the decision log, the driver |
| `attempts` | each fit: plan digest, decisions read, scores or estimate, **the live markers at that fit (new)** | GET /fits, the engine |
| `result` | the sealed score, or the reported estimate on its measure's scale, or the refusal with its reason | the engine |
| `wants` | **new:** each `want`, its reason and tag | the driver |
| `account` | **new:** the end-of-run account, marked as written after the results | the analyst |
| `replay` | **new, fixtures only:** the final plan on fresh draws (§7) | the replay scorer |
| `claims`, `surfacings`, `export`, `replication` | as in version 1 | the server; the replication sweep |

Sweeps (§6) are stored beside the records, keyed by the plan they sweep around.

**Markers in the pilot** are those the engine already reads, plus the declared ones:
- **Regime (live):** rows, rows analyzed, units where rows repeat, the effective size, events per
  parameter, predictors by role and their ratio to the rows, the outcome's type and event share,
  missingness per column and the complete-case share, the design, time order.
- **Measurement (live):** units as read with each reading's basis, detection-limit shares, batch
  and run-order columns, imputed copies and repeated measurement, the omics scale reading.
- **Measurement (declared):** instrument, assay or platform, preprocessing, units, recall method
  (§5).
- **Purpose and loss (live):** purpose, primary score and why, intended use and thresholds,
  estimand, multiplicity family.
- **Post hoc:** the signal and the result.

**Staged after the pilot:** Kish's effective n, the condition index as a marker, error structure,
reliability, and how a predictor is measured when the model is used.

**The analyst's account** sits beside the engine's record, never in its place:
- written at each decision, before the analyst has seen what mattered;
- tagged with a marker and the rejected alternative;
- context for hypotheses, never evidence;
- audited for faithfulness (§3).

## 5. Datasets and intake

Built for version 1 (`research/swarm/intake.py`). Version 2 of the intake adds:
- **Declared measurement markers** (Nolan, 2026-10-19), read from the paper's methods and, for
  NHANES, the NCHS laboratory and questionnaire documentation:
  - each quotes its passage, labeled declared, checked by a referee;
  - what the source does not say is recorded as not stated, the material for the survey paper
    Nolan has in view;
  - a declared marker that disagrees with a reading is settled by the evidence before it is called
    a finding.
- **A categorical eligibility rule** in the engine (exclude by level, refused and don't-know codes
  handled explicitly, STROBE's "confirmed eligible"), so Bang excludes diagnosed diabetes by its
  code.

**The pilot's six datasets:**
1. **Bang 2009** (Predict, NHANES 1999–2004, 31,126 rows from its own intake). Its replication
   targets the logistic model's odds ratios, which the engine reaches; the points score's
   performance needs fixed-rule scoring on new rows, which the engine lacks
   (`notes/replication/bang2009.md` §8), so it is computed by a research script (item 3 below).
2. **Sullivan 2021** (Estimate), its sweep done by hand (SU-A).
3. **Nolan's NHANES fasting glucose,** rebuilt from the NCHS files (Nolan, 2026-10-19):
   - **The cycles:** developed on 2011–2016 and tested on 2017–March 2020, the pre-pandemic file
     whose sampling design matches; 2021–2023 changed its design.
   - **Provenance:** each NCHS file and variable named, and a cycle column.
   - **The predictors:** his 21 mapped to their variables, plus others NHANES gives cheaply.
     Each is known before the fasting draw's result; same-draw glycemic measures (HbA1c, insulin,
     an oral glucose tolerance test) are left out by design.
   - **The population:** adults 20 and over without diagnosed diabetes and not taking a
     glucose-lowering medication, matching Bang (Nolan, 2026-10-19: "match bang"). The table keeps
     those columns; the exclusion is made in the app as eligibility decisions (the categorical
     rule, §10), so it is recorded with its provenance and the sweep measures how much it matters.
   - **The test:** held out by a cycle cut-point (§10).
   - **Its score on the later cycle,** reported both per participant (primary, for comparison
     with published AUCs) and weighted by the 2017–March 2020 fasting weights (Nolan concurred,
     2026-10-19).
   - **The benchmark** (`notes/replication/glucose_benchmarks.md`): no paper predicts fasting
     glucose as a number without glycemic inputs, so the comparison is an AUC: the *outcome*
     thresholded (fasting glucose of 126 mg/dL or more in a person without diagnosed diabetes),
     and the model's continuous prediction ranked against it:
     - **the primary benchmark:** Bang's points score computed by a research script on the same
       later-cycle participants. Bang's activity item (PAQ520) was discontinued after 2005–06, so
       the score is *adapted* to the later cycle's activity questions; the script declares and
       quotes the mapping for activity and for family history (MCQ300C), and Nolan accepted the
       adapted score as primary (2026-10-19);
     - beside it, Bang's reported AUC of 0.83 on its own later cycle.
   - **Reference points, not benchmarks:** Yoo 2026 (about 0.77) and De Silva 2020 (about 0.70)
     define dysglycemia with HbA1c or an oral glucose tolerance test, a different outcome. Each
     figure is checked against its paper before it is cited.
4. **The hardened Predict fixture:** `clinical_readmission_predict`'s generator with a
   nonlinearity the default path misses and an interaction, so a path's choices change its
   regret. (A predictor measured differently at use is dropped from it: the analyst could not
   learn it, and draws from one generator cannot show it.)
5. **The new Estimate fixture:**
   - a non-null effect on the reported measure;
   - a planted confounder and a collider;
   - missingness in a covariate that depends on the observed outcome (missing at random), so
     complete cases are biased and imputation is not.

   Each choice has a gradable right answer. The generator is tuned so that half the smallest
   planted bias is at least four times the standard error of the oracle plan's mean bias over 100
   draws, so §7's two conditions on b₀ hold together.
6. **The measurement fixture** (added to the build, Nolan, 2026-10-19), from the repo's untargeted
   metabolomics table (`metabolomics_untargeted.csv`):
   - **Its generator, new:** the repo's `make_fixtures.metabolomics` draws its outcome
     independently of every feature and makes batch the first half of the run order, so it cannot
     grade these decisions. The new generator plants a known set of truly associated features with
     known effect sizes, under detection-limit censoring, run-order drift, batch offsets that are
     not monotone in run order (so drift and batch correction are graded apart) and dilution, with
     QC samples.
   - **Its question:** which features are associated with the outcome, analyzed feature by feature
     with a false-discovery rate.
   - **Its grading:** the discovered features' false-discovery proportion and power against the
     truth, and the bias of their effect sizes on the log2 scale.
   - **Material here:** a decision that changes the discovery set at q < 0.05 by at least k
     features, k fixed in `prereg.json`.
   - **In the pilot it decides nothing:** it is reported, as the measurement axis's first contrast.

   The decisions it grades are measurement decisions: drift correction, normalization,
   detection-limit handling and the log scale. This gives the pilot a measurement-axis contrast
   with the NHANES tables.

**Mechanical only:** the null decoy (the seal protects it, so it tests the engine's intervals),
with distinct seeds.

**After the pilot:** the held-out shelf of generator draws at varied parameters, for a true
false-promotion rate (§9).

## 6. The pilot

**What it decides:**
- the production analyst model;
- whether the personas earn their place;
- whether agents choose better than the mechanical arms, and find what matters at a better cost.

**The personas** (Nolan, 2026-10-19), stances toward the field's defaults, same protocol:
- **Conventional:** takes the domain's customary path where the engine ranks it sound.
- **Skeptical:** questions each default, examining at least one alternative where a decision could
  move the result.
- **Explorer:** deliberately tries the alternatives the field rarely takes.

**Agent runs** (45):

| Arm | Datasets | Runs |
|---|---|---|
| Haiku 5.5, high effort, each persona | all six | 18 |
| Sonnet 5.5, high effort, each persona | all six | 18 |
| Haiku replicates, two more per persona | the Estimate fixture | 6 |
| Opus 5.5, high effort, conventional | the Estimate fixture, three runs (the reference) | 3 |

**Mechanical arms** (no tokens), per dataset:
- the default path and a random path;
- **the sweep:** each offered decision changed one at a time around the default plan and around
  each model's conventional-persona final plan (named now, so it cannot be chosen after the
  results). Details:
  - refits are deduplicated by plan digest;
  - readings with one true answer (a column's units, code or amount) are graded for correctness,
    not swept;
  - each refit rebuilds the plan with `core/export/replay.py` from the log truncated before
    `open_seal` (under Estimate, before `lock_plan`), revisits the one decision, fits, and reads
    the cross-validated scores or the estimate;
  - every refit is scored with random repeated 5-fold cross-validation, 5 repeats, within the
    training rows, whatever folds the plan chose: a plan with time-ordered folds would otherwise
    give five scores (`models/folds.py`);
  - on the Estimate fixture the explorer's final plan is swept too, so reach there includes the
    decisions only an explorer's path offers;
  - under Predict with more than one family, the engine's comparison substrate runs at least
    10 × 5 folds whatever the plan's validation, so a refit costs about 50 fits per family. The
    sweep's compute cap in `prereg.json` is set knowing that.

**The engine is frozen** at one commit for the pilot.

**Order:** the Estimate fixture is the first wave, so the early-stopping rule (§7) acts before the
expensive datasets (Nolan, 2026-10-19).
1. **Sizing:**
   - a scripted fit on each dataset sets its fit budget and the sweep's compute estimate;
   - the measured trio: Haiku and Sonnet, explorer persona, on Bang (the largest table), and Opus
     on the Estimate fixture (its only dataset). The trio's three runs are among the 45;
   - Nolan reads his usage meter before and after.
2. **The waves:** the Estimate fixture first, then the other datasets, in randomized order within
   each wave, under the stopping rule (§7).
3. **The sweeps,** within the compute cap.
4. **The claim cycle** (§9) on the pilot corpus and its shuffled control.
5. **Nolan's walkthrough** of a Haiku analyst's record of his glucose project (2026-10-19).

**Compute and timing:** daytime, unthrottled (Nolan, 2026-10-19), since under the throttle long
commands make agents miss their prompt cache and cost about four times as much. At most three
agent runs at once; every command short. The sweeps cost compute and fan noise, not tokens.

**Budget:** up to 40% of a fresh week. If the trio projects past it, the waves are cut in this
order:
1. Sonnet on the measurement fixture;
2. Sonnet on Sullivan;
3. the Haiku replicates.

Cutting the replicates defers the persona question; it does not decide it.

## 7. Pre-registration and acceptance

Written before the trio in `research/swarm/prereg.json` and applied by
`research/swarm/score_pilot.py`, not by the orchestrator.

**Regression checks** on every run (they hold by construction): seal integrity, schema validity, a
reason on every preview and decision.

**Material, in the loss's units.** The sweep scores each refit with repeated 5-fold
cross-validation, 5 repeats.
- **Predict:**
  - A decision that keeps the analyzed rows is material where the signed paired difference in the
    primary score has a corrected repeated k-fold interval excluding zero and is at least half the
    primary score's own interval half-width.
  - A decision that changes the analyzed rows (an exclusion, an eligibility rule, complete cases
    against a fill) leaves no shared folds. It is scored by an unpaired interval over the two
    plans' 25 fold scores, each arm's variance inflated by the same Nadeau–Bengio factor
    (1/k + n_test/n_train) the paired interval uses, since repeated folds are correlated; labeled
    unpaired in the sweep, under the same size rule.
- **Estimate:** a decision is material where it shifts the reported estimate by at least half the
  estimate's interval half-width.

**Decision quality, anchored to known truth.** Each final plan is replayed on fresh draws of its
fixture's generator: at least 100 draws, so a coverage band is testable.
- **The Estimate fixture decides the model.**
  - b₀, the bias margin, is half the smallest population bias the generator plants (keeping the
    collider, dropping the confounder, complete cases), computed from the generator, not the
    draws; and at least four times the standard error of the oracle plan's mean bias (its SD over
    the draws divided by √draws). A plan passes on its mean bias. The oracle passes and every
    planted wrong path fails; if the Opus reference fails, the fixture is re-examined, not the
    models.
  - Coverage must lie in the binomial band of a 95% interval at that number of draws.
- **The Predict fixture adds regret:** the replayed plan's loss against the oracle's. r₀ is half
  the default path's regret on the planted nonlinearity.
- **The measurement fixture is reported:** the false-discovery proportion against the nominal
  rate, power against the oracle plan's, and effect-size bias on the log2 scale.
- b₀, r₀ and the bands are computed from the generators and written into `prereg.json` before the
  trio.

**The unit:** a model is within the margins on a fixture when at least two-thirds of its non-refused
plans there are within them, with at least two non-refused plans (Haiku 6 of 9, Sonnet 2 of 3, Opus
2 of 3 on the Estimate fixture; Haiku 2 of 3 if its replicates are cut). A refused plan counts as
neither, and goes into the refusal rate.

**The two comparisons with the mechanical arms:**
- **Choosing well:** agent plans against the default and random plans on regret, bias, coverage
  and discovery quality on the fixtures, and on the sealed score (or the later cycle) on real data.
- **Finding what matters:** each agent's reach per fresh token, against the sweep's (complete by
  construction) per fit-hour.

**Other statistics:**
- reason faithfulness and the referees' catch rate of planted reasons;
- robustness: runs reaching the export without help. Stuck runs are classed (engine defect,
  driver defect, capability gap, analyst failure) by a reader shown the record with the model
  redacted, and only analyst failures count against a model;
- refusals, as their own rate;
- cost per useful result.

**The decision rule:**
- A model is acceptable if it is within the margins on the Estimate fixture, its reasons are
  faithful at least 4 times in 5, and at least 9 runs in 10 reach the export.
- Of acceptable models the cheaper is the production analyst; cost breaks a tie (Nolan,
  2026-10-19). If none is acceptable, the plan returns to Nolan.
- A persona is kept if its mean reach on the Estimate fixture differs from the conventional
  persona's by more than the larger of the two personas' within-persona ranges over their Haiku
  replicates.

**Expected false alarms,** stated in advance: about 2.5% for the decoy's interval, and the coverage
band's own miss rate at the number of draws. A miss on a control is settled by fresh seeds, never
by editing an expectation.

**Early stopping:** a model outside the margins on the Estimate fixture after its persona runs
stops; a spend projecting past the budget stops the waves.

## 8. What the engine must provide

Built: GET /fits (with the locked fit), GET /results, GET /mattered, analyst mode, provenance, and
`core/export/replay.py`, which rebuilds a project from `decisions.jsonl` in a fresh home and
refuses inputs whose SHA-256 differs.

Needed (§10):
- **Replay on a fresh draw:**
  - a flag that lets replay take a new draw of the same generator;
  - a rule for records tied to findings that may not fire on the new bytes (`apply_repair`);
  - truncation of the log before the seal or the lock, for the sweep;
  - a scorer that predicts on new rows. None serves predictions on new rows today; until one does,
    regret is the Brier score against the oracle on each draw's sealed held-out rows.
- **The sweep's refits under the full pipeline,** through the driver, since GET /mattered refits
  only linear and logistic models and refuses imputation, energy models, declared forms, scale
  scores, omics normalization, detection-limit fills, batch correction and marginal measures
  (`core/mattered.py`).
- **The signed change,** paired or unpaired as §7 says, computed from the per-fold scores the record
  already keeps.
- **A temporal seal by cut-point:** hold out every row at or after a time value, beside today's
  chronological draw by fraction (`core/seal.py`; `SplitSpec.holdout` is a fraction from 0 to 0.4).
- **The categorical eligibility rule** (§5).

## 9. From corpus to the middle layer

The corpus never reaches a user. A pattern becomes advice only through the registry
(CLAIMS_REGISTRY §8):
1. **Propose.** A theorist (Opus) reads the corpus and the sweeps. Each proposal is structured
   (the decision, the live markers that define its region, the direction, the size, the loss) and
   enters as proposed and unsupported. A proposal with no live marker is not a surfacing claim.
2. **Pull.** A claim gets work if it answers a decision a person got stuck on, or would change a
   judgment-graded default, ranked by the sweeps' measured materiality.
3. **Try to break it:**
   - a surfacing claim on held-out datasets in its region;
   - a choice claim on known truth;
   - across domains where markers match.
   Only these mechanical results change its status.
4. **Grade.** Promotion means replicated or above, with the two-referee gate and the basis audit;
   an author's self-grade of judgment is not a promotion.
5. **The shuffled control.** Each cycle also runs on a copy of the corpus whose markers are
   permuted across datasets, with the whole pipeline, test stage included, inside that copy.
   - **The planted effect:** a generator appends synthetic records carrying one true effect to the
     shuffled copy only, listed in a sealed file the theorist never sees.

**At pilot scale the cycle is a plumbing test** (Nolan accepted, 2026-10-19: "Ultimately, I do
want us to get into discovering real new theories that are not slop"). Its pre-stated outputs:
- the number of structured proposals from each corpus;
- zero promotions in both, as expected;
- the planted effect's claim proposed and routed;
- **the prior-driven share:** the overlap between the two corpora's proposals, computed by a script
  that matches proposals on their decision, the set of live-marker names (thresholds ignored) and
  direction. A high share means the
  proposals come from the theorist's prior, not the data.

**The first real cycle** (after the pilot, its five needs, plus a baseline and a cost: a
mechanical rule-miner over the sweeps' materiality tables, against which the theorist's "better
prior" is measured; and the roster's cost, three datasets per cell at the intake's prices):
1. **A roster by regime cell.** "Replicated" needs held-out cases from at least two datasets not
   used to fit the rule (CLAIMS_REGISTRY §2.4), so each cell needs at least three datasets; the
   roster is chosen per cell before the cycle.
2. **Prose to predicate.** A step turns each proposal's markers into a regime predicate in code,
   tested both ways, before it is tested on data.
3. **The held-out shelf's reuse budget.** How many claims may be confirmed against each held-out
   dataset, fixed in advance (Dwork et al., Science 2015).
4. **The allocation rule,** the search's exploration term: runs are allocated across regime cells
   by uncertainty times pull, with a fixed random share.
5. **Material and the app's bands.** The sweep's definition of material recalibrates the app's own
   materiality bands (`core/materiality.py`), so a promoted surfacing claim can key the Confirm
   tier.

**Nolan's time** (a couple of hours a week, 2026-10-19):
- blind-rating about ten referee verdicts against the records;
- a real-versus-shuffled test on theorist proposals;
- the walkthrough of the Haiku glucose record;
- once claims are promoted, a blind review of a sample each sprint.

**The engine's growth, from wants:** an option enters only when at least three runs on at least two
datasets want it and a prototype sweep shows it is material.

## 10. Order of work

After Nolan's usage resets (Thursday 1 pm), with the pilot in the same week (Nolan, 2026-10-19: "i
accept the thin buffer"). Each package: an Opus builder, an adversarial verifier, at most two
repairs, every command under about four minutes, whole suites left to CI. Already done: RF-1, the
sentence fixes R1_PLAN requires before the pilot.

| # | Package | Contents | Size |
|---|---|---|---|
| 1 | Record version 2 and the driver | §4's fields with live and post hoc tags; preview `--reason`, `--marker`, `want` and the vocabulary; model and persona with redaction; the signed change; the shuffled-corpus tool | 5–8M |
| 2 | Replay and the fixtures | the fresh-draw flag, the findings rule and truncation on `core/export/replay.py`; predictions on new rows; the hardened Predict fixture, the new Estimate fixture and the new measurement generator; the fixtures' truths regenerated and the three-record dry run redone; the decoy's seeds | 8–12M |
| 3 | The sweep and the mechanical arms | the sweep runner (dedupe, truncated replay), the default and random paths, the compute sizing | 4–6M |
| 4 | Two engine additions | the temporal seal by cut-point; the categorical eligibility rule, a guard touching decisions, the rows stage, options, its card, its methods sentence and the registry (guard waves have run about three times a wording wave's cost) | 8–12M |
| 5 | Intake version 2 | the declared measurement markers with checked quotes; the glucose table rebuilt from the NCHS files; Bang's eligibility by code; the benchmark script (Bang's points score on the later cycle) | 5–9M |
| 6a | RB | the NHANES glucose journey's claims registered (R1_PLAN R1-B), so a record's `claims` traces each sentence an analyst read | 2–3M |
| 6b | Pre-registration | `prereg.json` and `score_pilot.py`, the margins computed from the generators | 2–4M |
| 6c | The Bang replication sweep | an SU-A-sized package, its target the odds ratios | 1–2M |
| 6d | The claim cycle's tools | the theorist's and experimentalist's prompts, structured proposals, the overlap script | 1–2M |
| 7 | The live lockdown check | and the analyst's model id, in a fresh session on the home machine | small |

The packages sum to 36–58M; about 40–65M with a buffer, on the evidence of this build's own
overruns. Packages 1 and 2 can start first; 3 waits for 2; 5 waits for 4; 6b waits for 2; 6c
waits for 3 and 5. The claim cycle's own runs (the theorist on two corpora, the referees, the
planted reasons) come out of the pilot's 40%.

## 11. Rulings and what is open

**Nolan's rulings this plan rests on** (2026-10-16 to 2026-10-19):
- exploration before the lock, none after the seal;
- Haiku 5.5 as the analysts' model, Sonnet 5.5 opened for the pilot, three personas;
- daytime runs within 40% of a fresh week;
- the build and the pilot in one week on a thin buffer;
- the three axes and the declared markers;
- the analyst's audited account;
- surfacing first;
- the glucose project rebuilt and scored on a later cycle (per participant first, weighted
  beside), its population matching Bang's by eligibility decisions made in the app, benchmarked
  against Bang's score adapted to the later cycle;
- the Estimate fixture as the first wave;
- the stopping rule for review;
- Bang's replication on its odds ratios;
- the measurement fixture in the build;
- the claim cycle as a plumbing test;
- cost breaks a tie;
- forecasting tabled;
- the categorical eligibility rule;
- his walkthrough of the Haiku glucose record.

**The orchestrator's calls,** recorded so he can overrule them:
- the cycles;
- the arms and counts;
- the definition of material, the margins and the unit;
- the swept plans;
- the Decide cap after the pilot;
- the admission rule.

**Reviews:** version 5 goes to one last fresh reader, asked only whether anything left changes the
build's design; if nothing does, the build starts after the reset.
