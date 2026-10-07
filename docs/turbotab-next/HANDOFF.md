# Handoff

**State (2026-10-06, paused by Nolan after the platform fixes).** `turbotab-next` is at the merge that adds the design docs (on top of `9c349c4e`). The last CI run (`9c349c4e`) is green on all four jobs: Linux core and server tests, the Docker server image, and the macOS and Windows launchers.

**The engine (done and independently verified).** Waves 1, 2a, 2b and 2c and their repairs cover nearly every method in `V2_DEFINITION_OF_DONE.md`:
- data in: XPT, codebooks, joins;
- survey across families;
- multiple imputation compatible with the analysis model;
- prediction validation;
- omics: QC-RLSC and ComBat;
- scales;
- NCI usual intake;
- estimands and Table 2;
- causal ML and time-varying exposures;
- explainability;
- regression calibration;
- functional form;
- Explore;
- multiclass substitution;
- the leash;
- previews for every number-changing kind;
- the export bundle with replay.

The release track also landed: the legacy app is retired, with its references in `reference/`; the one-command launcher; server mode with sign-in; Docker; the generated methods reference and the five expert review packets.

**The design direction (Nolan's rulings, 2026-10-05).**
- Calm over complete: `calm/FOUNDATION.md` sets the five-second test, two registers, desktop only, the color roles and the canvas grammar.
- Four synchronized structures sit on branch `explore/calm`. On 2026-10-06 Nolan picked the quest log as the best so far, to be designed further (below).
- Exploration is a guide plus a gallery, walked on the first visit.
- Noticings become threads that return at later stages.

**The structure (Nolan, 2026-10-06): the quest log, redesigned.**
- Hover to see the questions you need to answer.
- The sections and questions are built dynamically from what the engine needs from this analysis.
- An overall progress bar sits at the top right, so users are not discouraged.
- First comes a crosswalk of what the engine must surface for the user to decide, and at what moment. More ideas will come from it.
- His framing, verbatim. Before the fit: "see what the engine needs to surface that a user decides, dynamically build the tapestry to show the consequences of their decision-making". After training, it pivots to: "show them the results and help them interpret with the tapestry, let them decide what to include and not include in their manuscript results section."
- **The orchestrator's methods floor on curating results:**
  - under inference, the locked plan's primary analysis is always reported;
  - what is left out of the results section goes to the supplement and stays in the record.

  This keeps the curation from becoming selective reporting.
- `calm/FOUNDATION.md` §3 and §6 still describe the Q&A card as the baseline, and must be updated.

The magic is designed in `understanding/UNDERSTANDING_LAYER.md`: 367 threads in 17 families, with sentinels and a coverage test. It is **not built yet**; about one thread in five has engine pieces.

**Approved and not yet built** (the DoD amendment, 2026-10-05):
- the families ridge, Huber, random forest and XGBoost;
- Table 1;
- stacking NHANES cycles;
- nested tuning;
- explanations, evaluation and robustness on screen;
- Classic's missing exploration views.

`RECIPES_AND_TUNING.md` specifies the recipes, overrides and tuning. Its draft 3 must fold in Nolan's rulings.

**Ruled by Nolan (2026-10-06):**
1. **`RECIPES_AND_TUNING.md` "Decisions for Nolan":**
   - the trees "Try both" by default, keeping blanks or taking the fill, chosen in each training fold;
   - fits over about 2 minutes wait for Fit;
   - three scope-cut groups stay in v2: faster search for big tables, more preprocessing options and the inference extensions. Only the kept-comparisons group goes to INBOX: standard settings as a kept version, trunk versions and re-tuned bands.
2. **The NHANES export:** commit it **gzipped** as a test fixture. The harness reads the `.gz`, and CI then runs the 42 test files that read it.
3. **The structure:** the quest log, redesigned (above).
4. **`UNDERSTANDING_LAYER.md` §7:**
   - every open noticing feeding the plan must be decided or dismissed before the lock or the seal;
   - clean checks go in the supplement;
   - "noticings" are named in two places.

**UI and scope discussion with Nolan (2026-10-06/07).** The full notes are in the orchestrator's memory, `turbotab-quest-log-tapestry`.
- **Seven fixed stages:**
  1. Your data (the file, the domain, what each column is)
  2. Your question (the outcome, then the goal, worded around it)
  3. First look (shaped by the goal)
  4. Who's in
  5. Models
  6. Results
  7. Write-up

  The questions inside each stage are dynamic. The progress bar has seven segments: an unreached stage shows empty, and a reopened stage drops back and says why.
- **Labels:** Decide · Confirm · For the record. The engine's terms asked/stated/silent never reach the UI.
  - Each stage ends with one Confirm sweep of its defaults.
  - Hovering a line elaborates slightly. Clicking it expands the card to its options, with a way back.
- **Results** are exhibits: a figure or table with its caption, the finding, and the interpretation. For each:
  - choose among drafted wordings, or write your own;
  - choose a placement: Results, Discussion, Supplement, or leave the exhibit out. Every analysis stays listed.
- **Full-tapestry flowcharts,** each savable as a figure:
  - the participant flow, or samples and features for omics;
  - the analysis flowchart before training, where the Fit button lives.
- **Export:** an Overleaf-ready LaTeX zip and Word from one manuscript model. Every citation is checked against Crossref. Classic's `ml/latex_report.py` and `ml/manuscript_validator.py` are references.
- **Goals:** Describe / Estimate an effect / Predict. A second question picks the shape, filtered by domain. Several goals in one paper run as sequential tracks over the shared stages, with Write-up merging them.
- **Added to v2:**
  - dietary patterns;
  - clustering into subgroups;
  - Bland–Altman, both between methods and between models.
- **Designed experiments added to v2** (2026-10-07) as one named milestone after the core slices:
  - parallel and cluster-randomized trials;
  - design declaration and routing (no confounder hunt; precision adjustment);
  - no exclusions after randomization; intention-to-treat and per-protocol analysis sets;
  - the CONSORT checklist and flow diagram;
  - causal wording;
  - missing-outcome sensitivity analyses.

  Case-control, with restrictions, and matched sets, with conditional logistic regression, ride with the milestone.

  Deferred to v2.x: crossover trials, repeated-measures trial models, and per-protocol and complier effects.
- **Mediation is out of v2.** Mediators are still recognized, so they stay out of the adjustment set.
- **Ruled (2026-10-07):** not every option on a design question must be supported. A rare design may carry "Not available yet", with its reason and an exit that keeps the work.
- **The definition of done was amended on 2026-10-07** with all of the above. See `V2_DEFINITION_OF_DONE.md`.
- **Plain words:** "exposure", "confounder" and "estimand" stay quiet terms only. The card asks "What do you think affects glucose?"
- **Validation of `319a513f`** (2026-10-06):
  - Python 2,848 passed;
  - frontend 201 and the calm kit 242 passed;
  - mock browser tests: 1 failure in the setup (`m2-journeys` crashes on the mock server's health check before it can skip);
  - Classic: 49 failed and 3,245 passed. Most look like repository self-checks that the release track broke. Triage is pending; the run hangs under xdist without `--timeout`.

  CI does not run the browser tests.

**Engine follow-ups found (not started):**
- the energy preview that quotes a coefficient before the lock;
- the exposure and estimand previews that draw nothing;
- the metabolomics zero-row crash;
- the elastic net's penalty, which is platform-dependent (a methods decision: a tighter tolerance or a one-standard-error rule);
- the speed targets (DoD gate 5), which need a quiet machine;
- the final re-audit.
See INBOX.md.

**Next, when resumed.**
1. Commit the NHANES fixture gzipped, and confirm CI runs the real-data tests.
2. Run the crosswalk: every decision, noticing and result the engine surfaces, when it surfaces, what it depends on, and what the tapestry shows. It is the input to the quest-log redesign.
3. Write `RECIPES_AND_TUNING.md` draft 3 with the rulings.
4. Build in vertical slices: each is an engine thread, its quest-log screen and a drive. Start with the prototype plan's phase 0 (`UNDERSTANDING_LAYER.md` §6), together with the shelf and tuning work.

---

## History: the state on 2026-10-01

**State (2026-10-01): M1 is done and tagged `next-m1`.** The production app runs the full NHANES
journey in a real browser: the Record with its card-based decisions, the pipeline banner, and the
stage. The stage has the transform player (the flip plays each method's real storyboard), previews
of every option on the user's own data, finding evidence, savable figures in journal style with
provenance captions, and the Results (models against a baseline, coefficients, and substitution
curves with a refit uncertainty band). Checks: 444 Python tests, 100 frontend tests, and the
m1-journey Playwright spec against a real server. Run records: `m1/w1-result.json` and
`m1/w2-result.json`. Screens: `m1/screens/`.

**Next: M2** (BLUEPRINT §9). It covers the opening sequence for all five lenses (orientation,
grain and repeats, eligibility, the seal), findings with preview-before-apply and deferral, and the
wide omics data path, benchmarked. Start with the "M2-first" and "[M2]" items in INBOX.md (the energy
card's word budget first). Then triage the rest of that file, merge `main` in, write M2_CONTRACT.md,
and launch.

---

## History: the pause of 2026-09-27 (mid-M1)

Paused at Nolan's request (weekly usage limit). Everything is committed and pushed. Start here.

## Where it stands

- **M0 is done** and tagged `next-m0`.
- **M1 part 1 is done and merged** on `turbotab-next`: validators, the server-side interview
  Router, column roles, the participant flow, the sealed split, the generic consequence diff, and
  previews for every M1 question. Also three model families, in-fold pipelines, fit, substitution
  curves, publishable decision sentences, teaching for all 11 questions, pack proposals, and
  findings that name their lever. Workers now spawn on demand.
- **Checks:** 395 Python tests and 27 frontend tests pass.
- **The live NHANES journey** runs end to end through the API in about 10 s. Fit takes about 2.5 s,
  and previews take 1–30 ms. Details are in `m1/w1-result.json` → `server.measurements`.
- **M1 part 2 has not started.** That's the frontend: the Record and the pipeline panel rebuilt on
  the chosen design, then integration, two reviewers, and fixes.

## The design decision — mostly made (see BLUEPRINT §11.1)

Nolan ruled from the screenshots: `stage` is the base; no per-option preview thumbnails; `stage`'s lineage diagram is canonical; prototype a **pipeline banner** (the whole pipeline as a compact strip at the top, everything below is the working window). **The slider then became a transform player** (BLUEPRINT §11.1): one two-state *now ⇄ with this choice* flip that plays the method's own labeled storyboard as it flips (forward, or in reverse on the way back; brisk; interruptible) and drives every view; step dots pause on any step; switching options morphs directly between results without replaying the storyboard; saves capture any labeled state. Backend work this adds: consequence views gain optional `steps`, and the energy-adjustment builder supplies residual/density/partition storyboards. **The design is fully ruled — build it.**

### Background: the three prototypes

Three prototypes answer the wall-of-text problem (BLUEPRINT §11) on the same real NHANES fixture.
Branches are on origin:

| Branch | Angle | Visible words (S1 preview) |
|---|---|---|
| `explore/stage` | Hovering or focusing an option morphs the pipeline panel into its consequence: a primary before/after view, plus lineage and distribution | 269 |
| `explore/inline` | Every option card carries a sparkline preview; the focused card enlarges in a fixed stage under the strip; a pin compares two options | 181 |
| `explore/scrub` | One before→after transformation per option, scrubbed; table cells roll like an odometer; sidenotes | 243 |

Screenshots, frame strips, word counts and each designer's rationale and weaknesses are in each
branch under `docs/turbotab-next/m1/explore/<angle>/`, and in `m1/w1-result.json` → `explore`.

**Nolan has seen `stage/s1-density-light.png` and said:** *"I really like the screenshot you sent me.
It's a really good and promising design style."* Make `stage` the base. Graft in what the others do
better, if it survives a side-by-side look:

- inline's compare pin and its lower word count;
- scrub's column-identity morph in the table (`fat_total` → `fat_total_adj` → `fat_total_per_kcal`)
  and its sidenotes.

Also fix the weaknesses the designers named:

- Stage's first-preview transition snaps.
- Histogram bins morph across different units. Crossfade when the unit changes, and morph only
  within the same unit.
- Hover-to-preview needs a touch equivalent.

## Found by the live journey — carry these into M1 part 2

1. **Complete cases drop 86% of NHANES.** `meds_hbp` and `meds_chol` are blank for 18,405 rows, and
   a blank there means "not asked" (DRIVE_RUBRIC §4). The missing-values question must say so
   through its preview, and should offer leaving those columns out, before the user records
   "complete cases".
2. **Nested nutrients distort substitution.** The fat_total → carb curve holds `fat_sat`, `fat_mon`
   and `fat_poly` fixed while `fat_total` moves. The design warns about it, but roles should mark
   nested components (sugar ⊂ carb, fat subtypes ⊂ fat_total), and substitution pairs should
   respect them.
3. **Boosted trees score below the mean baseline** (CV R² −0.04) with no concern attached. The shelf
   and results should say when a family underperforms the baseline.
4. **The linear substitution band has zero width.** Resampling rows through one fitted linear model
   cannot vary its average effect. The band needs a refit bootstrap (PRODUCT_VISION §06c, mark 2).

The other 36 notes from this run are in `INBOX.md` under "From M1 workflow 1", untriaged.

## A framing Nolan added at the pause

*Modeling decision provenance* — his name for a reproducibility problem in nutrition research:
readers can't reconstruct which choices turned raw data into a model's inputs. He noted the lineage
diagram illustrates it well. It is now North star item 4 in the BLUEPRINT. It raises the lineage
and the row flow from teaching aids to deliverables: M1 part 2 should make them publication-quality;
M5's export carries them as figures plus a replayable provenance record.

## Next session, in order

1. Look at the three prototypes side by side (their screenshots and frames). Rule on the design and
   record the ruling in BLUEPRINT §11.
2. Merge the chosen prototype's reusable components and `explore/fixtures` into `turbotab-next`.
3. Launch M1 workflow 2 on that design and the four points above: the frontend Record (questions
   from the Router, previews, teaching drawer, findings with levers), the pipeline panel (Rows flow,
   Columns lineage, Results with metrics and substitution curves), integration, two reviewers
   (function, and taste against DRIVE_RUBRIC), and a fixer.
4. Drive it with Playwright on the real NHANES export, triage the inbox, merge `main` in, and tag
   `next-m1`.
