# The research swarm: driver, record, intake and pilot

Drafted 2026-10-16 on turbotab-next by the orchestrator, docs only, at Nolan's approval the same
day. It builds on the swarm program (CLAIMS_REGISTRY.md §8 and §11, ruled 2026-10-15) and on
Nolan's answers of 2026-10-16. Nothing here is built yet. The swarm runs only after its gates
(§9) are passed.

## TL;DR

- **What:** AI analysts work real public datasets in TurboTab, the way a researcher would. Each
  one asks its question, tries options, and writes down why it chose what it chose. We keep
  every run as a record.
- **Why:** many such runs show which modeling decisions actually change results, and where.
  That is the "middle layer" the app's advice should rest on. Today the advice rests on our
  judgment.
- **The one rule that matters most:** analysts must explore before they commit. Every attempt is
  logged with its reason and with how far it moved the result. Once the held-out data are opened,
  nothing more is tried. A run that tries nothing is flagged.
- **Where:** on your machine, at times you set, throttled at night. Analysts have no token cap,
  but every run's spend is measured and reported.
- **What we decide first:** whether the cheapest analyst is good enough, measured in a small
  pilot before any volume. Nolan ruled on 2026-10-18 that the pilot uses Haiku ("Use Haiku 5-5
  for the pilot"), lifting his "nothing below Sonnet" rule for the pilot's analysts only.

## 1. What the swarm is for

One infrastructure carries three run designs (the swarm program, 2026-10-11):
1. **Replication.** The analyst reproduces a published analysis. It records each decision the
   paper stated, each one the paper left unstated, and how far each unstated option moves the
   published number. This measures the literature's hygiene, and it is Nolan's provenance
   question (2026-10-15).
2. **Agent analysts.** Several analysts take the same dataset and question, each making its own
   decisions. This is a "many analysts" study whose every decision is logged with its
   materiality, so it shows why their results diverge.
3. **The full shelf.** Every model family on every dataset, nested and sealed, for the learning
   mechanics. The meaning decisions are fixed from the documentation, and the methodological ones
   are enumerated.

The pilot runs designs 1 and 2. Design 3 needs no analyst and waits for the heavy-run schedule.

Each sprint stays in one domain (nutrition first, where the lenses are strongest), because the
engine applies only partly outside its lenses. A rule that holds across every domain's ledger
belongs to the engine; one that holds in a single domain belongs to that lens.

## 2. The driver

**An analyst works through the app's own API, never the engine directly.** The driver runs a
local-mode server and talks to it over HTTP, as `turbotab/core/reference/journeys.py` already does
for the review packets' reference journeys. That server is `ProjectService`, so every guard,
refusal, claim and seal the app enforces applies to an analyst too. This is also the interface a
later LLM harness will use, so building one gives the other.

`journeys.py` already follows the Router question by question. Its answers come from a journey
spec, the app's first-ranked option, or the acceptance drivers. The driver keeps that loop and
replaces the answer source with the analyst. It gives the analyst a small command set, each
command returning JSON:

| Command | What it does | Exists today |
|---|---|---|
| `ask` | the open question: its card's words, its options, the readings it rests on | the quest log and ask cards |
| `preview <option>` | what recording that option would do to the data; nothing is recorded | `ProjectService.preview` |
| `answer <option> --reason` | records the decision, with the analyst's reason in the record's `note` | `decide`; `DecisionRecord.note` exists, but no route accepts a note yet (SWARM_ENGINE_PLAN E4) |
| `revisit <decision>` | changes an earlier decision, which the log keeps as a new record | the decision log's reverts |
| `fit` | presses Fit; under Estimate this records the plan's lock | `press_fit`, `LockPlan` |
| `results` | the results the app would show now: cross-validated under Predict, the estimate under Estimate | W2 (Predict Results), W3–W4 (Estimate) |
| `seal` | opens the held-out rows, once | `open_seal` |
| `export` | the methods, the checklist and the bundle | `export` |

**What the driver refuses**, mechanically and whatever the analyst asks:
- `reseal`. A Predict run's reported result is its first sealed score. The engine allows a
  recorded re-seal for people, so the driver blocks it in analyst mode.
- Any change after the seal is opened, beyond what the record marks `post_seal` and the corpus
  leaves out of the run's result.
- Any file, shell or network access outside the driver's commands. The analyst agent type
  (`.claude/agents/analyst.md`, new) has no Edit or Write tool and no web tool. Its Bash is
  limited to the driver's commands by a permission allowlist.

**Compute, not tokens, is what binds.** Every fit is metered: count, seconds, and the workers it
used. Each run has a fit budget (§6), and the driver refuses `fit` past it with a plain message.

## 3. Exploration before the lock, and none after the seal

Nolan's line (2026-10-16): trial and error, documented, is good. A swarm that "makes exactly one
decision and does no real discovery" is not, and neither is chasing a result after it is seen. That
is his p-hacking point about preprocessing. The protocol follows the app's own lock and seal, so
the line falls in the same place for an analyst as for a person.

**Under Predict:**
- Discovery is on the training rows' cross-validated scores, before the seal.
- The analyst may fit, compare and revisit as often as its fit budget allows. Each attempt
  records:
  - what it tries;
  - why: a hypothesis, such as "the medication answers may carry most of the signal";
  - what it saw: the change in the CV score and in the model's reading.
- Then it opens the seal once and reports that score.
- The record keeps every CV attempt, so selection optimism (best CV score against the sealed one)
  is measured per run. That is a learning-mechanics datum, not a failure.

**Under Estimate:**
- The plan's lock is recorded when Fit is pressed, before any estimate is shown (`LockPlan`,
  FOUNDATION §7).
- Discovery before the lock uses what the app shows without the estimate:
  - previews;
  - the readings and the causal roles;
  - outcome-blind checks such as positivity and balance;
  - the shelf's outcome-blind ranking.
- After the lock, a change is a secondary analysis. It is logged, and its effect on the estimate
  is measured ("which decisions mattered"). The primary result stays the locked plan's.

**What counts as exploring, and how it is checked:**
- An **examined alternative** is an option previewed or fitted before the analyst chose. Reading a
  card does not count.
- A **single-pass run** examines no alternative for any decision and fits once. It is flagged, and
  the pilot reports the single-pass rate for each analyst configuration.
- **Exploring has no quota.** A quota would breed token exploration, previews pressed to pass a
  count. Instead every attempt needs a hypothesis and a reading of what it showed. Then a referee
  audits a random sample of attempts each sprint and asks whether the reason was real and whether
  the reading matches the numbers. The referee is an Opus agent, with Nolan spot-checking. The
  audit's pass rate is reported beside the exploration counts.
- **Materiality is measured by the engine, never stated by the analyst:**
  - the preview's change in the data;
  - the CV score's change between attempts;
  - the estimate's shift for a secondary analysis.

## 4. The corpus record

One JSON record per run. It is validated against a schema (`research/swarm/record.py`) and
stored in `research/corpus/`. Both sit outside `turbotab/`, so the product never imports them and
the release zip leaves them out. The corpus is kept apart from the claims registry
(CLAIMS_REGISTRY.md §8): abundance in the corpus, authority in the registry.

| Field | Contents | Source |
|---|---|---|
| `run` | id, design (1–3), engine commit, analyst model, effort and prompt version, start and end, fresh and cached tokens, fits and fit seconds | the driver; the transcript, counted as in `measuring-subagent-spend` |
| `dataset` | id, source URL, release date, license, each file's SHA-256, the codebook | intake (§5); data provenance in ingest (HANDOFF step 7) |
| `question` | goal G (the purpose), estimand or intended use Q, and for replications the paper and its stated question | the analyst's first answers; the paper |
| `regime` | n, events, missingness, design (survey weights, clusters, repeats), lenses | the engine's readings |
| `decisions` | each decision D: the option O chosen; the options examined and how (preview or fit); the reason; the phase (before the lock, after the lock, after the seal); the source (the analyst, or the app's default accepted); and materiality M | the decision log (`DecisionRecord`: `note`, `post_seal`), the driver, the engine |
| `attempts` | each fit: the plan's digest, the decisions it read, its CV scores or estimate | the driver; W2–W4's run records (§8) |
| `result` | the locked plan's digest, and the sealed score or the estimate with its interval, once | the engine |
| `claims` | every registry claim the run rendered, with its id, status and grade at the time | `claims.card` and `claims.methods` calls, logged by the server in analyst mode |
| `surfacings` | each refusal met and the exit taken, and each concern shown | the server's responses |
| `export` | the bundle's SHA-256, its methods text and its checklist | `export` |
| `replication` | for design 1: the paper's stated decisions, the unstated ones, each unstated option's shift in the published number, and the published number against ours | the analyst's record against `notes/replication/` |

The schema follows the middle layer's claim shape (the swarm program, 2026-10-11): under goal G,
estimand or use Q and regime R, decision D with option O has validity V and materiality M.
Analysts fill G, Q, D and O, the engine fills R and M, and V is never set by a run. Only a
mechanical check or an independent grader sets it (§7).

**Amended 2026-10-19 (Nolan, in the pilot discussion): three axes, their markers, and the
analyst's account.**

A claim says where it holds on three axes, and the record keeps the markers for each, from the
engine's own readings (never the analyst's description):
- **The regime,** the structure of the table and the question: rows, rows analyzed, units where
  rows repeat, the effective size (events, or the rarest class); the predictors by role and their
  ratio to the rows; the outcome's type and event share; missingness per column, the complete-case
  share and whether the pattern is monotone; clusters, survey strata, PSUs, weights and time order;
  the largest correlation and the condition index; the signal (the cross-validated gain over the
  no-predictor baseline, recorded after the run, so it steers nothing).
- **The measurement model,** how each number came to be (Nolan: "regime only answers half the
  question, the other half that goes unanswered is the instrumentation of how the data was
  recorded"): the scale type (absolute, relative intensity, compositional, ordinal items); the
  units as read; the error structure (variance growing with the mean, skew); censoring (shares
  below or above a detection limit); batch and run-order columns, drift found, QC samples; repeated
  measurement (replicates, recalls, imputed copies); and the instrument (food-frequency
  questionnaire, 24-hour recall, a mass-spectrometry platform, an assay), declared by the intake
  (§5) because the engine cannot read it.
- **The purpose and its loss:** the purpose and the primary score, with why; the intended use and
  its thresholds; the estimand (the effect measure, marginal or conditional, the substitution);
  the multiplicity family; the loss a claim from the run would be graded on.

Each decision's entry keeps a snapshot of the markers it read, as they stood when it was made (a
missing-values answer reads the missingness after the exclusions); the dataset's markers are kept
once per run. A transfer test then picks datasets from another domain that match a claim's
markers.

**The analyst's account** sits beside the engine's record and never in place of it. The engine's
record is the evidence; the account carries what only the analyst knows, its intent (an exclusion
that corrects a data error and one that drops outliers to improve the fit look alike in the log):
- written when each decision is made, before the analyst has seen what mattered; any end-of-run
  account is a separate field, marked as written after the results;
- each reason names the marker that drove it (its axis and marker) and the alternative it
  rejected, then free text;
- context for a theorist's hypotheses, never evidence: a claim's status changes only by the
  mechanical checks and the designed experiments (§9);
- audited: the referees score whether each stated reason matches what the analyst was shown, and
  the rate of reasons that do not is reported per model and persona.

## 5. Dataset intake

- **Public data only,** each file with its source URL and release date (HANDOFF step 7's
  provenance), its license, and its SHA-256 at download. No dataset needs an application or
  a data-use agreement.
- **One question card per dataset.** For a replication it comes from the paper, verified against
  `notes/replication/`. For agent analysts, an intake agent writes the question, the population
  and the purpose, and the orchestrator reviews it. Analysts on one dataset share one card, so
  their divergence is in the decisions, not in the question.
- **Known-answer controls.** The repo's fixtures with declared truths (`truths.FIXTURE_TRUTHS`,
  BLUEPRINT §14.3) run beside real data. An analyst's estimate there is right, or the app refuses
  it; anything else is a defect, filed in INBOX.
- **Null decoys.** Datasets generated with no signal run in the same queue. A "finding" there is a
  false discovery, and the corpus reports the rate. These are separate from the stage-3 audit's
  sealed decoys, which the swarm never sees.
- **A held-out shelf.** Each sprint sets some datasets aside before it starts. A pattern found on
  the discovery shelf is confirmed on them once, the lock and seal applied to the research itself.

- **Declared measurement markers from the source** (amended 2026-10-19, Nolan: "since we are
  pulling from real published papers, why not use that information to guide the measurement
  markers we need?"). The intake reads the paper's methods (for a fixture, its documentation) and
  fills the card's measurement section: the instrument, assay or platform, the preprocessing, the
  handling of detection limits, the units, the recall method. Each marker quotes its passage and is
  labeled declared; a referee checks the quotes. What the paper does not say is recorded as not
  stated, since a paper's unstated measurement decisions are part of what a replication reports.
  Where a declared marker and the engine's reading disagree (the paper says kcal and the engine
  reads kJ; the paper says log-transformed and the data are raw), the record holds both and the
  conflict is a finding.

## 6. The pilot

**Datasets** (about six):
- Bang 2009 (Predict, NHANES);
- Sullivan 2021 (Estimate, NHANES; replicated by hand first, HANDOFF step 4, so the swarm's
  replication has a reference);
- Nolan's NHANES fasting glucose under Predict, after his walkthrough;
- one Predict fixture and one Estimate fixture with declared truths;
- one null decoy.

Menke 2015 waits for a Describe purpose, which the engine does not have yet.

**The model-and-effort comparison**, the pilot's first experiment. Nolan asked on 2026-10-16
whether the best analyst model and effort are settled. They are not: nothing has measured them.

Nolan's rulings set the design:
- 2026-10-18: "Use Haiku 5-5 for the pilot." The analyst agent type names `claude-haiku-5-5`, and
  the first runs confirm which model actually ran.
- 2026-10-19: the pilot's purpose is to learn whether the production swarm will be effective and
  which choices make it more so ("since pilots are cheap, perhaps we also open the scope to sonnet
  5-5 as well"). So Sonnet 5.5 runs from the start, and each analyst takes one of three personas
  (he approved them the same day).

**The personas** are stances toward the field's defaults. The protocol is the same for all three;
only the stance in the analyst's prompt differs.
- **Conventional:** takes the domain's customary path where the engine ranks it sound, and says why.
- **Skeptical:** questions each default before accepting it, previewing at least one alternative
  where a decision could move the result.
- **Explorer:** deliberately tries the alternatives the field rarely takes, the search's explicit
  exploration term, so the corpus reaches the places a convention may be valid but not good.

| Per dataset | Conventional | Skeptical | Explorer |
|---|---|---|---|
| Haiku 5.5, high effort | 1 | 1 | 1 |
| Sonnet 5.5, high effort | 1 | 1 | 1 |
| Opus 5.5, high effort | 1, the ceiling | | |

That is 7 runs per dataset and 42 in all. Crossing model with persona separates the two levers the
production swarm must set. Haiku against Sonnet at each persona, and both against the Opus ceiling,
measure what a cheaper model loses. Each persona on two models measures whether a stance makes real
diversity, which three identical prompts could not show. Beside §7, the pilot reports:
- the spread between a dataset's runs (paths and results);
- each run's coverage (the share of the decisions on offer it previewed or changed).

**Order and cost:** first a measured trio, one run of each model with the conventional persona on
the Predict fixture. Its fresh tokens, fits and wall time size the rest. The rest runs dataset by
dataset, so a model that clearly fails, or a spend running hot, stops it early. Nolan's budget
(2026-10-19): up to 40% of a fresh week. The runs go in daytime, unthrottled (PILOT_READINESS §3).

**The production swarm the pilot informs** (the orchestrator's proposal, for Nolan after the pilot):
- most problems get one analyst each, for breadth;
- about one in five gets a panel of personas, to keep measuring spread and the field's blind spots;
- a small sample gets an Opus ceiling run;
- replications and stuck runs escalate to a stronger model.

**Each configuration is scored on:**
- protocol (seal violations, which must be zero; every attempt has a reason);
- exploration (examined alternatives per decision, the single-pass rate, the referee audit's pass
  rate);
- correctness (known-answer controls right or refused; decoys giving no finding);
- replication (stated decisions reproduced, or each divergence named; the published number within
  the paper's own interval where the data allow);
- robustness (runs reaching the export without help; each stuck run classed as an engine defect,
  a driver defect or an analyst failure);
- spend (fresh and cached tokens, fits, wall time per run).

If Haiku meets §7, it is the analysts' model. If it does not, that is the finding, and the plan
comes back to Nolan before any Sonnet run.

**Compute:** the runs are heavy runs, scheduled with Nolan, two workers each, under the
quiet-hours throttle (`heavy-runs-wake-the-bedroom`). The fit budget per run is set from the first
three runs' measured fit times, not guessed here.

## 7. Pilot acceptance

These thresholds are our judgment, set before the pilot and revisited only with Nolan:
1. **Seal integrity:** no decision after a seal opening reaches a run's result, and no re-seal,
   checked mechanically from every decision log. One violation fails the pilot.
2. **Logging:** every record validates against the schema. Every attempt has a reason and an
   engine-measured materiality.
3. **Exploration:** in the chosen configuration, the single-pass rate is at most 1 run in 5, and
   the referee audit passes at least 4 attempts in 5.
4. **Correctness:** every known-answer control is right or refused, and no null decoy yields a
   finding the analyst reports as real.
5. **Robustness:** at least 9 runs in 10 reach the export without human help. Every engine defect
   a run meets is filed in INBOX with the run's id.
6. **Spend:** measured for every run, with a projection for one domain sprint. Nolan is told
   before a sprint whose projection threatens the week's usage the build needs.

## 8. What W2–W4 must build for it

These go into the waves, so the swarm is not a retrofit (HANDOFF step 3):
- **A run record per Fit:** the plan's digest, the decisions it read, and its CV scores or
  estimate, kept by the server. Nothing today lists the fits a project made, with the plan each
  read and what it scored.
- **Results readable as data,** under both purposes: W2's Predict Results and W3–W4's Estimate
  Results serve JSON, not only a screen.
- **"Which of my decisions mattered"** as engine data for secondary analyses (W3–W4), since
  materiality after the lock comes from it.
- **An analyst mode** in the server: re-seal refused, and each claim rendered logged with its id
  and status.
- **Data provenance in ingest** (HANDOFF step 7): source URL and release date per file.

The driver, the analyst role, the schema and the intake tooling come after W4. They are about
20 units in the swarm program's estimate of 2026-10-11, to be re-sized from W2–W4's measured
cost.

## 9. From corpus to the middle layer, and the gates

The corpus never reaches a user. A pattern in it becomes advice only through the registry
(CLAIMS_REGISTRY.md §8):
1. **Propose.** A theorist agent (Opus) reads the corpus and proposes a claim in the middle
   layer's shape, with the regime it holds in. It enters the registry as proposed and unsupported.
2. **Pull, not push.** The claim gets work only if it answers a decision a real person got stuck
   on: a friction log, or a walkthrough (ORCHESTRATOR §2).
3. **Try to break it.** An experimentalist designs runs where the claim would fail: known-truth
   simulations, and the held-out shelf. Only these mechanical checks change its status.
4. **Grade.** The claim earns its grade's requirement (§2.4), passes the two-referee gate and the
   basis audit, and is built with Fires and Silent tests like any claim.
5. **Measure the slop rate.** Decoys sit in the same queue, and the false-promotion rate is
   reported, never a count of claims. Throughput is capped at what can be audited, highest
   materiality first.

**The gates before the first pilot run:**
- R0 landed, and the walkthrough's claims registered (R1);
- W2–W4's needs in §8 built;
- the Sullivan 2021 replication done by hand;
- the stage-3 audit's decoy machinery in place;
- Nolan's word on the schedule.

## 10. Open for Nolan

- **Haiku for analysts:** ruled 2026-10-18, "Use Haiku 5-5 for the pilot" (§6).

Everything else here is the orchestrator's call, under your delegation of methods and
architecture, and is recorded so you can overrule it:
- the corpus's place outside `turbotab/`;
- the exploration protocol;
- the acceptance thresholds;
- the pilot's datasets.
