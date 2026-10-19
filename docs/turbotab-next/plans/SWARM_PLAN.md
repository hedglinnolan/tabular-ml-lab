# The research swarm: what it measures, how, and the pilot

Version 2, written 2026-10-19 by the orchestrator. It replaces version 1 (2026-10-16, amended
2026-10-18 and earlier on 2026-10-19; in git history) after an independent audit
(`notes/swarm-audit/2026-10-19-plan-audit.md`) found that, as built, the pilot would measure the
analysts and not whether decisions are good. Nolan ruled on the audit's questions the same day.
Under the swarm program (CLAIMS_REGISTRY §8 and §11) and Nolan's go-ahead of 2026-10-18.

## TL;DR

- **The goal:** the middle layer, the learning mechanics of tabular modeling. Nolan's question:
  "is this modeling decision even a good decision regardless of whether it is mathematically
  valid?" A claim answers it on three axes: the regime, the measurement model, and the purpose
  with its loss.
- **Two kinds of claim:**
  - a **surfacing** claim ("this decision matters here, or does not") says where a user's
    attention belongs, and the corpus can reach it;
  - a **choice** claim ("this option beats that one on this loss") needs known truth: simulations
    and the fixtures.

  Surfacing comes first (Nolan, 2026-10-19).
- **How:** a search over modeling paths with two arms.
  - **AI analysts** give the prior: the paths a researcher in that domain would take ("the
    analysis that a modeler does happens on rails for a specific domain"), plus a persona that
    deliberately leaves the rails.
  - **Mechanical arms that spend no tokens** do the measuring: a sweep that changes each offered
    decision one at a time and keeps the signed change, a default path, and a random path.
- **Against slop:**
  - Choice claims are graded only against known truth.
  - Every claim pipeline runs once on a corpus whose markers are shuffled, where every pattern is
    false by construction, to measure the real false-promotion rate.
  - Nolan blind-reviews a sample of promoted claims.
- **Built today:** the driver, version 1 of the record, the six checks, the intake and a dry run
  whose 18 checks agree. **Not built:** everything this version adds (§10).
- **The pilot:** after the build in §10, in daytime and unthrottled, within 40% of a fresh week:
  - Haiku 5.5 and Sonnet 5.5, each crossed with three personas;
  - an Opus 5.5 ceiling on the fixtures;
  - the mechanical arms on every dataset;
  - one claim cycle beside its shuffled control.

## 1. What the swarm is for

**The middle layer.** A claim in it says, for a region of three axes, whether a decision matters
and, where truth allows, which option is better:
- **the regime:** the structure of the table and the question;
- **the measurement model:** how each number came to be (Nolan: "regime only answers half the
  question, the other half that goes unanswered is the instrumentation of how the data was
  recorded");
- **the purpose and its loss:** the purpose, the score or estimand it is judged on, and the use.

Claims are keyed on the axes, not on the domain. Domains sit in different regions of them, so a
claim learned in one domain can be tested in another that matches its markers; that transfer test
is the middle layer's strongest check. A claim's evidence records its domain as provenance.

**Two kinds of claim, graded differently:**
- **Surfacing:** "in this region, decision D moves the result by at least m (or by less than m)".
  It feeds the app's Decide and Confirm tiers: what to ask a user and what to set for them. It is
  graded on the signed change measured across held-out datasets in the region (§9).
- **Choice:** "in this region, option O beats O′ on loss L by at least d". It feeds defaults and
  rankings. It is graded only against known truth: plasmode simulations built from real datasets,
  and the fixtures (§5, §9).

**Three run designs** share one infrastructure:
1. **Replication.** An analysis reproduces a published one. Its product is a pre-registered sweep
   of the paper's unstated decisions, how far each moves the published number, as SU-A's Sullivan
   replication did (`notes/replication/sullivan2021.md` §4, §8). The analysts are its
   many-analysts arm: how far independent analysts' paths diverge from the paper's and from each
   other.
2. **Agent analysts** on a dataset and question, under personas (§6).
3. **The full shelf:** every model family on every dataset, nested and sealed, with no analyst.
   It waits for the heavy-run schedule.

**The closed loop of engine growth.** A run that needs an option the engine lacks records it (the
`want` command, §2). The engine grows only by the admission rule in §9, within a budget of
questions per reference journey, so it grows from evidence without the bloat that would make
modeling overwhelming (Nolan, 2026-10-19).

## 2. The driver

Built (DRIVER_PLAN D1, D2; `research/swarm/driver.py`). An analyst works through the app's own API
on a local-mode server in analyst mode, never the engine directly, so every guard, refusal, claim
and seal the app enforces applies to it. Its commands, each printing JSON:

| Command | What it does | Change in this version |
|---|---|---|
| `ask` | the open question: its card's words, its options, the readings it rests on | none |
| `preview <option> --reason` | what recording that option would do; nothing is recorded | **needs a reason**, which goes in the record and the referee's sample |
| `answer <option> --reason --marker` | records the decision with the analyst's reason | **adds `--marker`**: the axis and marker that drove it (§4) |
| `revisit <decision> --reason` | changes an earlier decision | none |
| `fit` | presses Fit; under Estimate it records the plan's lock | none |
| `results` | what the app shows now | none |
| `seal <family> --reason` | opens the held-out rows, once | none |
| `export` | the methods, the checklist and the bundle | none |
| `want <what> --reason --marker` | **new:** records an option or check the analyst needed and the engine lacks | new |

**Refused mechanically:** a re-seal; any change after the seal reaching the result; every tool but
the driver's commands (the analyst agent type and its hook, DRIVER_PLAN D2, wired locally); a run
id that is not the caller's. A run's label is neutral (`run_NN` and a secret part), so no label
names a control.

**The fit budget** is set per dataset from a scripted fit on it before the pilot (§6), not from
the fixtures' small tables.

## 3. Exploration before the lock, and none after the seal

Unchanged in principle (Nolan, 2026-10-16: trial and error, documented, is good; chasing a result
after it is seen is not). Under Predict, discovery is on the cross-validated scores, then one seal.
Under Estimate, discovery is outcome-blind until Fit records the lock; a change after the lock is
a secondary analysis, logged and measured.

**What counts, and how it is audited:**
- An **examined alternative** is an option previewed or fitted before the choice, with a reason.
  A preview without a reason is refused.
- **Coverage, not counts:** the share of the decisions the mechanical sweep rates material (§6)
  that the run previewed or changed. A run that previews everything immaterial and nothing
  material scores low.
- **A single-pass run** (no alternative examined, one fit) is flagged and its rate reported.
- **The referee audit** samples previews as well as fits. It scores whether each reason is real
  and whether it matches what the analyst was shown. About one in ten sampled reasons is a planted
  fabrication, written by another agent, to measure how often the referees catch one.
- **Materiality is the engine's, never the analyst's** (§4).

## 4. The corpus record

One JSON record per run, validated against `research/swarm/record.py`, stored in
`research/corpus/`, outside `turbotab/`, apart from the registry: abundance in the corpus,
authority in the registry. Built at version 1; version 2 adds what is marked new.

| Field | Contents | Source |
|---|---|---|
| `run` | id, design, engine commit, **analyst model and persona (new)**, effort, prompt version, start and end, fresh and cached tokens, fits and fit seconds | the driver; the transcript |
| `dataset` | id, source URL, release date, license, each file's SHA-256, the codebook, **the card's declared measurement markers (new, §5)** | intake |
| `question` | the purpose, the estimand or intended use, the paper for a replication | the card; the analyst's answers |
| `regime` | **the three axes' run-level markers (new; below)** | the engine's readings |
| `decisions` | each decision: the option chosen; those examined and how; the reason and **its marker tag and rejected alternative (new)**; the phase; the source; **the markers it read, as they stood then (new)**; the engine's measured change | the decision log, the driver, the engine |
| `attempts` | each fit: the plan's digest, the decisions it read, its scores or estimate | GET /fits |
| `result` | the locked fit's sealed score, or the reported estimate on its measure's scale, or the refusal with its reason | the engine |
| `sweep` | **new:** the mechanical arm's one-at-a-time changes around this run's final plan (§6) | the sweep runner |
| `replay` | **new, fixtures only:** the final plan replayed on fresh draws from the fixture's generator: regret, or bias and coverage | the replay scorer |
| `wants` | **new:** each `want` with its reason and marker | the driver |
| `account` | **new:** the analyst's end-of-run account, marked as written after the results | the analyst |
| `claims`, `surfacings`, `export`, `replication` | as in version 1 | the server; the replication sweep |

**The engine's measured change is signed and in the loss's units (new).** Today it is stored
unsigned, as |mean Δ| / SD(Δ), with the sign only in a sentence (`core/materiality.py`), and it
grows with the table's size. Version 2 keeps the signed paired difference in the score's own units
(under Predict, with the engine's corrected repeated k-fold interval) and, under Estimate, the
shift of the reported estimate in its own units and in units of its interval.

**The three axes' markers**, from the engine's readings, never the analyst's description:
- **Regime:** rows, rows analyzed, units where rows repeat, the effective size (events, or the
  rarest class), events per parameter; the predictors by role and their ratio to the rows; the
  outcome's type and event share; missingness per column, the complete-case share, whether the
  pattern is monotone; clusters, the spread of cluster sizes, survey strata, PSUs, weights, Kish's
  effective n, time order; the largest correlation and the condition index; the signal (the
  cross-validated gain over the no-predictor baseline, recorded after the run).
- **Measurement model:** the scale type (absolute, relative intensity, compositional, ordinal
  items); the units as read; the error structure (variance growing with the mean, skew);
  censoring (shares below or above a detection limit, top-coding); batch and run-order columns,
  drift found, QC samples; repeated measurement (replicates, recalls, imputed copies) and,
  where it can be estimated, reliability; the instrument, as declared (§5); under Predict, how
  each predictor will be measured when the model is used (self-report against measured at an
  exam).
- **Purpose and loss:** the purpose and primary score, with why; the intended use and its
  thresholds; the estimand (the measure, marginal or conditional, the substitution); the
  multiplicity family; the loss a claim from the run would be graded on.

Each reading records its basis (the rule and the evidence the engine used). Units, and whether a
column was already transformed, are inferences, so a disagreement between a declared marker and a
reading is settled by the evidence before it is called a finding.

**The analyst's account** sits beside the engine's record, never in its place:
- a reason per decision, written when it is made, before the analyst has seen what mattered, with
  a marker tag (axis and marker) and the alternative rejected, then free text;
- the end-of-run account, a separate field marked as written after the results;
- context for a theorist's hypotheses, never evidence: a claim's status changes only by the
  mechanical checks and the designed experiments (§9);
- audited for faithfulness (§3), with the rate reported per model and persona.

## 5. Dataset intake

Built for version 1 (`research/swarm/intake.py`: manifest with URL, release date, license and
SHA-256; a question card). Version 2 adds:
- **Declared measurement markers** (Nolan, 2026-10-19: "since we are pulling from real published
  papers, why not use that information to guide the measurement markers we need?"):
  - The intake reads the paper's methods and, for NHANES, the NCHS laboratory and
    questionnaire documentation (for a fixture, its own documentation).
  - It fills the card's measurement section: the instrument, assay or platform, the
    preprocessing, detection limits, units, the recall method.
  - Each marker quotes its passage and is labeled declared; a referee checks the quotes.
  - What the source does not say is recorded as not stated: a paper's unstated measurement
    decisions are part of what a replication reports, and the material for a survey paper Nolan
    has in view.
- **Fixtures with known truth that can be redrawn**, so a plan can be replayed on fresh draws:
  - the Predict fixture's generator (`clinical_readmission_predict`, its true risk known);
  - a new Estimate fixture: a non-null effect, a planted confounder and a collider, and
    missingness that depends on the outcome, so attenuation, adjustment and missing-data choices
    are all graded;
  - a measurement fixture: the repo's untargeted metabolomics table
    (`metabolomics_untargeted.csv`: detection-limit censoring, run-order drift, QC samples,
    batches), with a generator for its truth.
- **The null decoy** is run by the scripted analyst and the mechanical arms only: the seal
  protects it, so it tests the engine's intervals, not an analyst. Its draws use distinct seeds.
- **Nolan's NHANES glucose table** is rebuilt from the NCHS files with its provenance (its card
  today says the export and the imputation are "not recorded"). Its "good" is performance on a
  later NHANES cycle, not a random split (Nolan, 2026-10-19).
- **The held-out shelf** is set aside per regime stratum before a sprint, with its confirmation
  statistic fixed in advance and a budget for its reuse across claims (Dwork et al., Science 2015).

## 6. The pilot

**What it decides:** how the production swarm should run: which analyst model, which personas, and
whether agents beat the mechanical arms per unit of cost at finding what matters and choosing well.

**Datasets** (six):
- Bang 2009 (Predict) and Sullivan 2021 (Estimate), the replications;
- Nolan's NHANES glucose (Predict, judged on a later cycle);
- the Predict fixture, the new Estimate fixture and the measurement fixture, with known truth.

The null decoy runs mechanically beside them.

**The personas,** stances toward the field's defaults; the protocol is the same for all:
- **Conventional:** takes the domain's customary path where the engine ranks it sound, and says why.
- **Skeptical:** questions each default before accepting it, examining at least one alternative
  where a decision could move the result.
- **Explorer:** deliberately tries the alternatives the field rarely takes, the agents' share of
  the exploration term.

**The arms, per dataset:**

| Arm | Runs | Tokens |
|---|---|---|
| Haiku 5.5, high effort | one per persona (3) | yes |
| Sonnet 5.5, high effort | one per persona (3) | yes |
| Opus 5.5, high effort, conventional | one, on the three fixtures only (the ceiling, scored by replay) | yes |
| Haiku replicates | two more per persona on the Predict and Estimate fixtures (12 in all) | yes |
| The mechanical sweep | around each agent run's final plan and around the default plan | none |
| The default path and a random path | one each | none |

That is 51 agent runs. The replicates separate a persona's effect from run-to-run noise. Run order
is randomized within each wave.

**The engine is frozen** at one commit for the pilot. No package merges into the branch the pilot
runs on until it ends.

**Order:**
1. **Sizing.** A scripted fit on each dataset sets its fit budget, and the measured trio runs one
   run of each model with the explorer persona on Bang 2009, the largest table and the widest path.
   Nolan reads his usage meter before and after the trio.
2. **The waves,** dataset by dataset, with the stopping rule in §7.
3. **One claim cycle** (§9) on the pilot corpus and one on its shuffled control.

**Compute and timing:** daytime, unthrottled (Nolan, 2026-10-19: "daytime is acceptable for the
subagent runs"), since under the throttle long commands make an agent miss its prompt cache and
cost about four times as much (PILOT_READINESS §3). At most three agent runs at once. Each
analyst command stays short; a fit that would run past about four minutes is refused by its
budget.

**Budget:** up to 40% of a fresh week (Nolan, 2026-10-19). If the trio's spend projects the pilot
past it, the waves are cut before they start, Sonnet's replicates first.

## 7. Pre-registration and acceptance

Written down before the trio runs, in `research/swarm/prereg.json`, and applied by a script
(`research/swarm/score_pilot.py`), not by the orchestrator, who wrote the plan.

**Regression checks** that must hold on every run (they hold by construction, so they are not
acceptance criteria): seal integrity, schema validity, a reason on every preview and decision.

**The unit and the statistics:**
- The unit is the model, pooled over its personas; the persona comparison is secondary.
- **Decision quality** on the fixtures, from the replay: regret against the true risk (Predict),
  bias and coverage (Estimate), compared between models as paired differences per fixture.
- **Coverage of the material decisions,** from the sweep, paired per dataset between models and
  against the mechanical arms.
- **Reason faithfulness,** from the referee audit, and the referees' catch rate of the planted
  reasons.
- **Robustness:** runs reaching the export without help. Each stuck run is classed (engine defect,
  driver defect, capability gap, analyst failure) by a reader who does not know which model ran;
  only analyst failures count against a model.
- **Refusals** are a separate rate, never counted as passes.
- **Cost per useful result:** fresh tokens per material decision covered, and per unit of regret
  reduced.

**The decision rule:**
- A model is acceptable if:
  - its regret, bias and coverage on the fixtures are within the pre-registered margin of the Opus
    ceiling;
  - its reasons are faithful at least 4 times in 5;
  - at least 9 runs in 10 reach the export.
- The cheaper acceptable model is the production analyst. If neither is acceptable, that is the
  finding, and the plan returns to Nolan.
- A persona is kept in production if it raises coverage of the material decisions beyond the
  replicates' spread.
- The margins are fixed in `prereg.json` before the trio.

**Expected false alarms, stated in advance:** about 2.5% for the decoy's interval and 5% for a
null's coverage. A miss on a control is settled by rerunning that control on fresh seeds, never by
editing an expectation.

**Early stopping:** a model whose first two fixtures both miss the margin stops; a spend that
projects past the budget stops the waves.

## 8. What the engine must provide

Built for version 1: GET /fits (with the locked fit), GET /results, GET /mattered, analyst mode
(re-seal refused, the claims render log, the preview log, the decision's note), provenance.

Version 2 needs (§10):
- **The sweep's refits under the full pipeline.** GET /mattered refits only linear and logistic
  models and refuses a plan holding imputation, an energy model, a declared form, a scale score,
  omics normalization, a detection-limit fill, batch correction or a marginal measure
  (`core/mattered.py`). Those are exactly the measurement-axis decisions. So the sweep runs each
  one-at-a-time change through the driver as its own scripted fit, under the full pipeline, at a
  compute cost and no token cost.
- **The signed change** in the loss's units, with its interval (§4).
- **The markers** on the record (§4), from readings the engine mostly already makes (units,
  detection limits, batch, scale), each with its basis.
- **Replay** of a final decision log as a plan on a fresh draw, without the engine asking its
  data-dependent readings again. This is unproven: the build checks it first, and where a reading
  cannot be replayed, the replay records the reading's answer from the original run.

## 9. From corpus to the middle layer

The corpus never reaches a user. A pattern becomes advice only through the registry
(CLAIMS_REGISTRY §8):
1. **Propose.** A theorist (Opus) reads the corpus and the sweeps and proposes surfacing or choice
   claims in the three axes' terms. Each enters the registry as proposed and unsupported.
2. **Pull.** A claim gets work if it answers a decision a person got stuck on (a friction log, a
   walkthrough) or if it would change one of the engine's judgment-graded defaults, ranked by the
   sweep's measured materiality. The second door lets in the field's blind spots that no one yet
   knows to be stuck on.
3. **Try to break it.**
   - A surfacing claim is tested on held-out datasets in its region.
   - A choice claim is tested on plasmode simulations, real datasets with planted truth, whose
     designs, losses and replicate counts are written down before they run.
   - A claim learned in one domain is tested where possible in another that matches its markers.
   - Only these mechanical results change a claim's status.
4. **Grade.** The claim earns its grade's requirement, passes the two-referee gate and the basis
   audit, and is built with Fires and Silent tests.
5. **Measure the slop rate end to end.**
   - **The shuffled control:** each claim cycle also runs on a copy of the corpus whose markers
     are permuted across datasets, so every claim the theorist proposes from it is false by
     construction.
   - **Power:** one true effect is planted in the control, to show that the pipeline can find a
     claim at all.
   - The share of the control's claims that reach a grade is the false-promotion rate, reported
     beside the registry's text decoys.
6. **A human check.** Nolan blind-reviews a sample of promoted claims each sprint, within the
   couple of hours a week he can give to walkthroughs and spot checks (2026-10-19). Claude is then
   not the only grader of Claude.

**The engine's growth, from wants:** an option enters the engine only when it is wanted by at
least three runs on at least two datasets, and a prototype sweep shows it is material. Each
reference journey has a cap on Decide-tier questions; a new question that would pass the cap
must retire or fold an existing one. That cap is the bloat budget.

## 10. Order of work

The pilot waits for these. They run after Nolan's usage resets (Thursday 1 pm), each package
an Opus builder, an adversarial verifier and at most two repairs, with commands kept short.

1. **The record's version 2 and the driver:**
   - the fields in §4;
   - the signed change;
   - the preview reason, `--marker` and `want`;
   - model and persona;
   - the account fields.
2. **The mechanical arms and the replay:**
   - the sweep runner;
   - the default and random paths;
   - the replay scorer;
   - the new Estimate and measurement fixtures with their generators;
   - the decoy's distinct seeds.
3. **The intake's version 2:**
   - the cards' measurement sections with their checked quotes;
   - the NHANES glucose table rebuilt from the NCHS files, with its later cycle.
4. **RB** (R1_PLAN R1-B), the pre-registration and its scoring script, the shuffled-corpus tool,
   and the theorist's and experimentalist's prompts.
5. **The live lockdown check and the analyst's model id**, in a fresh session on the home machine
   (DRIVER_PLAN D6).

Then the pilot (§6), then its claim cycle and Nolan's review.

## 11. Rulings and what is open

**Nolan's rulings this plan rests on** (2026-10-16 to 2026-10-19):
- exploration before the lock, none after the seal;
- Haiku 5.5 as the analysts' model, with Sonnet 5.5 opened for the pilot and three personas;
- daytime, unthrottled runs, within 40% of a fresh week;
- the three axes, declared markers from the papers, and the analyst's audited account;
- surfacing claims first;
- "good" for his glucose project is performance on a later NHANES cycle;
- he blind-reviews promoted claims.

**The orchestrator's calls,** recorded so he can overrule them:
- the arms and their counts;
- the pre-registration's statistics and margins;
- the admission rule and the question cap;
- the Opus ceiling only on the fixtures;
- the null decoy run mechanically;
- the corpus kept outside `turbotab/`.

**Reviews:** this version goes to an independent review by a Fable agent at xhigh effort before
any of §10 is built. The plan is revised until fresh readers stop finding flaws (Nolan,
2026-10-19: "Think of this like editing an english essay, it's always good to revisit and revise
until we are happy").
