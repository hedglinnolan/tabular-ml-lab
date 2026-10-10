# v2.x seams

**Status: DRAFT 1, for the orchestrator, 2026-10-08.** Written on branch `design/v2x-seams` from `turbotab-next` at 5d575931. Every claim about the code cites `file:line` at that commit. Nothing was run; the code and the specs were read.

**Why this exists.** Nolan froze the definition of done (`V2_DEFINITION_OF_DONE.md`, amended through 2026-10-08) and handed scope control to the orchestrator on one condition: "as long as we know that those future additions are not difficult to fold into the program." This document is that check. For every item deferred to v2.x it names:
- the seam the item will plug into: a declaration point, a registry, a contract field or a decision kind;
- whether that seam exists today, is created by a named v2 package, or is missing;
- what v2 could build that would close the seam, and the guard against it;
- the effort to fold the item in once the seam exists.

It adds nothing to v2. The recommendations at the end are for the orchestrator to judge.

**Where the items come from.**
- `V2_DEFINITION_OF_DONE.md` §5, with its 2026-10-07 and 2026-10-08 additions, and the 2026-10-05 amendment's list of families that stay v2.x;
- `INBOX.md`, every item marked v2.x;
- `RECIPES_AND_TUNING.md` §10, INBOX;
- `MODEL_FAMILY_CONTRACT.md`: MC-14 to MC-16b, §4, and the items it proposes for INBOX;
- `crosswalk/CROSSWALK.md` and `crosswalk/SIZING.md`: the "Not available yet" designs and estimands.

**How to read it.**
- Code paths are relative to `turbotab/`. Document paths are relative to `docs/turbotab-next/`.
- **Status:**
  - *exists today*: the item plugs in by adding a declaration (a family, a contract, a decision kind or a vocabulary value) and its reference test;
  - *planned in v2*: a named v2 package creates the seam;
  - *missing*: neither. The smallest v2 hook that would open it is named, with its size.
- **Sizes** follow SIZING: S = 1 unit, M = 3, L = 8, XL = 20. Later efforts use only these four; SIZING's S–M rounds up to M and its M–L to L. A size taken from a source cites it. Any other size is this document's estimate, uncertain by about a third either way, as SIZING's are.
- **Recorded** means written to `decisions.jsonl`. Every project's state, every export bundle and every replay is a fold of that log (`core/decisions.py:5177-5277`; `core/export/replay.py:14-20`).
- The planned seams rest on two drafts awaiting Nolan's review: RECIPES (draft 2) and the model-family contract (draft 2). A seam they plan is only as firm as they are.

---

## Summary

**The counts.** Of the 50 deferred items:
- **24 plug into a seam that exists today** (row 50, the reading proposer, was added on 2026-10-10). These are mostly the method items in INBOX, plus mediation, the survival versions, assembly, collaboration and the assistant.
- **23 plug into a seam a named v2 package creates.** These are the tuning and preprocessing groups (RT-1 to RT-7), the families (MC-1, MC-2, MC-11, MC-12), the trial designs (P0.6, E1, E2), tracks (D2), the research record and the headless driver.
- **3 have no seam.** They are successive halving with Hyperband, TPE with BOHB, and shared-step changes kept as versions.

**The missing hooks.** The 3 items need two hooks, S each, 2 units in all:
- a strategy field on RT-1's tuning plan;
- the trunk's identity stored with the read-only "revised after first results" row, outside the disposable cache.

**The larger danger is closure.** The top risks:
- **The decision log has no format version, and it loads every line strictly** (`core/decisions.py:5256-5277`). Renaming or removing a recorded value makes every project and bundle that holds it unloadable. v2 already schedules one such rename: the causal learner `random_forest` becomes `nuisance_forest` (DoD, 2026-10-08).
- **The causal lane's `boosted_trees` learner shares its key with the family RT-5a will tune** (`core/models/causal.py:408-412`). Once learners come from the registry, old records would silently change meaning.
- **RECIPES' version keys hash the whole `VariantSpec`.** Any field or recipe slot added in v2.x would re-key every recorded version.
- **Parallel tracks.**
  - One stage exists per name (`core/graph.py:107-109`).
  - The "estimates shown" record is one side file per project, outside the log (`server/service.py:54`).
  - The rows rule is stated in a fixed track order.
- **Lists kept by hand.**
  - `ESTIMATE_STAGES` (`core/estimand.py:1321-1324`) has already left the usual-intake stage undecided (INBOX 185).
  - Five copies of the omics-lens list.
  - Eighteen switches on family keys, which MC-2b retires.

**The cost of guarding.** About 6 more units, S each, most of them a few lines inside packages already planned. With the two hooks, that is under 8 units, about 1% of SIZING's 734 remaining.

---

## The seams

### 1 · Tuning and preprocessing (package C6c, displaced to v2.x on 2026-10-08)

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 1 | Successive halving and Hyperband | RECIPES §4.2's `TuningPlan` and §4.7's `TuningRecord` (new `core/models/tuning.py`, RT-1), dispatched from `core/models/inner_cv.py:fit_pipeline` (249); the estimate's fit count in `core/models/cost.py` (RT-8) | **missing.** The plan is "plain random search over a scrambled Sobol sample", with the candidate list fixed before any score (RECIPES §4.2). Its cost is one formula, F = w·[(K−1)·C+1]. | `TuningPlan.strategy`, with one value in v2 (the Sobol search). The candidate generator and the fit count belong to the strategy, not to RT-8's estimate. **S (1)**, inside RT-1 and RT-8. | RT-1 fits "one head per inner split per recipe option", shared by every candidate (RECIPES §4.3). Halving fits candidates on a fraction of the rows, which needs a head per subsample. *Guard:* RT-1's per-split helper already works "for the rows it is given" (§4.3); keep the rows an argument. T6 checks the estimate through the strategy. | M each (SIZING C6c) |
| 2 | TPE and BOHB | As row 1, plus re-run replay (RECIPES §4.7; RT-12) | **missing** (row 1's hook) | Row 1's hook. `TuningRecord` lists the candidates in the order they were evaluated, at no extra size. | Re-run replay "searches again and must reproduce the recorded choices" (§4.7). An adaptive sampler on parallel workers depends on the order results arrive (§4.2 quotes Optuna's FAQ). *Guard:* for adaptive strategies, the replay of record is RT-12's pinned replay at the recorded values. T13 asserts that it reproduces the deployed predictions without searching. | M each (SIZING C6c) |
| 3 | The Thorough budget | `SetTuning.mode` and the plan's candidate count (RECIPES §3.1, §4.2) | planned in v2 (RT-1, RT-6) | none: "thorough" is one more mode value | A version's key is the sha256 of the canonical JSON of its whole `VariantSpec` (RECIPES §3.2). A field added to `VariantSpec` later (a strategy, a budget, a constraint) re-keys every recorded version. Each one would then come back as a phantom "earlier version". *Guard:* RT-7 hashes only the fields that differ from their defaults. T7 asserts that adding an optional field leaves recorded keys unchanged. | S (SIZING C6c) |
| 4 | The tuning curve (Dodge et al. 2019) | `TuningRecord`, which holds "each candidate's pooled inner score" (RECIPES §4.7), shown in Results' Settings (§4.8; RT-9) | planned in v2 (RT-1, RT-9) | none | §4.7 also says "each outer fold records its chosen candidate". If RT-9's per-fold record keeps only that choice, the curve has the final refit's scores alone. *Guard:* RT-9 keeps every candidate's inner loss per outer fold (at most 17 numbers a fold); T8 checks them. | S (SIZING C6c) |
| 5 | Native categories (HistGB, XGBoost, later LightGBM) | The `encoding` recipe slot (RECIPES §2.4; RT-2) and RT-3's column routing (`reads(spec)`, `passes_blanks`) | planned in v2 (RT-2, RT-3) | none, if RT-3's routing is per slot | RT-3 declares `passes_blanks` on every step (RECIPES §9). MC-4 counts "about twenty step classes". A flag for blanks alone needs a second sweep of every step for categories. *Guard:* each step declares which slots' columns it passes untouched (a set), with blanks first. The 255-level limit and the sparse-level concern (RECIPES §10) then become checks on the option. | M (SIZING C6c) |
| 6 | Pareto and robust scaling per family | The `scale` slot and `PenaltyScaler`, exposing `center_` and `scale_` per column (RECIPES §2.4; RT-2). The undo paths read them: `explain._undo_scaling`, `linear_equation`, `elastic_net.coefficients`. | planned in v2 (RT-2) | none | Undo code might switch on the scaler's name (`standard`, `two_sd`) instead of reading `center_` and `scale_`. *Guard:* T15 undoes a coefficient through a scaler with arbitrary centers and scales. | S each (SIZING C6c) |
| 7 | A per-model log1p | The `transform` slot, an in-fold step after Explore's steps and before scaling (RECIPES §2.4). A family can already add fixed steps of its own through `preprocess(spec)` (`core/models/pipeline.py:668-685`). | planned in v2 (RT-2) | none | Low. The `set_recipe` validator refuses it under inference (RECIPES §5). *Guard:* the refusal reads the purpose, not a family key (MC-2b's no-switch test). | S (SIZING C6c) |
| 8 | Huber under inference: a weighted M-estimator with a design-based or cluster sandwich | The family's `purposes`, and MC-1's `InferenceDecl` (`intervals`, `design_based`). It replaces the signature check in `core/models/survey.py:has_design_estimator` (627-634) and the design-family table (614). | planned in v2 (MC-1, MC-2b). The family itself is RT-5c, prediction only. | none | RT-5c might keep Huber out of inference with a key check instead of `purposes = ("prediction",)`. *Guard:* MC-2b's syntax-tree no-switch test, with its allowlist at zero. | L (SIZING C6c), checked against R `robsurvey` |
| 9 | Nuisance learners from the registry, tuned inside cross-fitting | The estimators take `learner: str \| Factory` (`core/models/causal.py:416`, 556, 598, 790). The recorded answer is `CausalLearner` (`core/decisions.py:1411`), built by `make_learner` (`core/models/causal.py:383-413`). The default is set at `core/stages/causal.py:608`, and the preview uses `CARD_LEARNERS` (`core/method_previews.py:352`). | exists today at the estimator level. The recorded vocabulary is a closed list. | none. RT-11 already reworks `make_learner` to draw the lasso's folds by unit, and the factory can take the fold's units there. | **(a)** MC-2b renames the recorded value `random_forest` to `nuisance_forest` (DoD, 2026-10-08). The log validates every line strictly (`core/decisions.py:5274`), so a log or bundle holding the old value stops loading. *Guard:* a parse-time alias, and a test that a pre-rename line loads. **(b)** The `boosted_trees` learner is HistGB at its defaults (`core/models/causal.py:408-412`). That is the same estimator as the family's untuned build today (`core/models/boosted_trees.py:34-41`), but RT-5a tunes the family. Once learners read the registry, a recorded `boosted_trees` would silently mean the tuned family. *Guard:* rename it together with `random_forest`, and keep retired values as tombstones that are never reused. **(c)** In `dml_plr`, the outcome and exposure models are fit as classifiers only when the learner is a string other than `"linear"` (`core/models/causal.py:573`, 581). A factory learner on a yes/no outcome is therefore silently a regressor. *Guard:* a declared property, with a test that a factory learner on a yes/no outcome returns probabilities. | L (SIZING C6c: M–L) |

### 2 · The kept comparisons (moved to v2.x on 2026-10-07)

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 10 | "Compare with standard settings" as a kept version | RT-7's versions: a `VariantSpec` with `tuning="standard"`, the versions fitted in `design.objects["earlier"]`, and BBC-CV over versions (RECIPES §3.2) | planned in v2 (RT-7, RT-9) | none | RT-9 gives `FittedModel` an `earlier` field (RECIPES §9), and RT-7 a fixed `design.objects["earlier"]`. If that field is a flag, a comparator, which is neither current nor earlier, has no place. RECIPES §3.2 already gives versions roles after the opening (`final`, `secondary`). *Guard:* one `role` field on every fitted version; "comparator" is added later. | M |
| 11 | Shared-step changes after scores, kept as versions | The read-only row "revised after first results" (ruled 2026-10-08; `crosswalk/CROSSWALK.md:59`, 136-140). RT-7's `scores_seen.json`, holding each version's spec and the log sequence at which its scores were first served (RECIPES §3.2). MC-3's `trunk` stage. | **missing.** A version is a family's recipe and tuning. A shared-step change "creates no version" (RECIPES §3.3), and nothing records which trunk the read-only row was scored on. | Store with each read-only row's scores the trunk stage key and the log sequence it was served at, in the versions file and not only in the stage cache. The log folded up to that sequence rebuilds the trunk. **S (1)**, inside RT-7 and C7d. | The crosswalk reads the row "from the earlier fit artifact, which the engine never deletes" (`crosswalk/CROSSWALK.md:140`). But the cache "stays disposable" (`core/graph.py:307`), and `scores_seen.json` holds only family keys per outcome today (`core/models/selection.py:496`). A cleared cache loses the row in v2, and v2.x has nothing to rebuild from. *Guard:* the hook, with a test that clears the cache and still serves the row. | L (`crosswalk/CROSSWALK.md:140`: "about an L on top of RT-7") |
| 12 | Re-tuned substitution bands | Pinned refits through `TunedPipeline.at(chosen)` (RECIPES §4.3; RT-1), in the substitution stage (`core/stages/__init__.py:636`) | planned in v2 (RT-1) | none. A band option on `set_substitution` is one optional field. | Low. The caption "conditional on its tuned settings" (RECIPES §4.3) names the gap. The refits are heavy runs, scheduled with Nolan. | M |

### 3 · The hypothesis noticing as a candidate model (moved to v2.x on 2026-10-08)

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 13 | The hypothesis noticing adding a candidate model | A `terms` recipe slot for the benchmark, routed as RECIPES §3.5 routes data-derived values, with "Discover inside each fold" as its honest exit (`MODEL_FAMILY_CONTRACT.md` §6, question 2). It uses RT-2's slots, RT-6's data-derived rule, RT-7's versions and MC-9's `late_notices`. | planned in v2 (RT-2, RT-6, RT-7, MC-9). The slot itself is the v2.x work. | none | MC-9 computes the local-effect matrix in one stage, on the final fits only (`fit.objects["fitted"]`, MODEL_FAMILY_CONTRACT §2.2 "Where"). In-fold discovery needs it per outer fold. *Guard:* MC-9 writes the readout as a function of a fitted model and rows, and its fixtures call that function directly. `RecipeSlotName` is a closed Literal (RECIPES §2.4), so adding `terms` is additive. | L (MODEL_FAMILY_CONTRACT §6: about L, 8) |

### 4 · Model families

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 14 | An MLP family, with the solvable-settings harness and neural diagnostics (MC-14, MC-15, MC-16a) | `register_family` (`core/models/base.py:113-127`) under C1 to C14: MC-1's `Identity(kind="trained_network")` and the fold-in gate (MC-12) | planned in v2 (MC-1, MC-2a, MC-2b, MC-11, MC-12) | none | **(a)** An unknown class gets no explanation, silently. `model_kind` returns None for any class it does not name (`core/models/explain.py:353-364`), and `fit_cost` special-cases two classes (`core/models/cost.py:95-108`). *Guard:* MC-2a, then MC-2b's no-switch test. **(b)** Replay compares every estimate at one bundle-wide tolerance (`core/export/record.py:22`; `core/export/replay.py:189-195`, 226). A torch fit is not bit-identical across thread counts. *Guard:* MC-11 reads each family's `replay_tolerance` (C12). The provenance format moves to `/2`, and `/1` still reads. | XL (MC-14 L, MC-15 M, MC-16a L: about 19 units) |
| 15 | A tabular transformer (FT-Transformer) | As row 14 | planned in v2 (as row 14) | none | As row 14. Its rotation probe must show a change (MODEL_FAMILY_CONTRACT §4.2). | L ("later, if at all") |
| 16 | A pretrained in-context family (TabPFN-style) | MC-1's `Identity(kind="pretrained_prior")`, with the checkpoint's SHA-256. Its row and column limits become a `select_models` validator (`register_validator`, `core/decisions.py:2263`), as in `core/methods/omics.py:2267`. The server option runs through the long-fit server job (UI ruling of 2026-10-08; P0.8). | planned in v2 (MC-1, MC-12, P0.8) | none | Workers are capped at one per 4 GB of RAM (`core/config.py:42-47`), and each starts with numpy, pandas and scikit-learn preloaded (`core/jobs.py:1-9`). Loading weights per job is slow. This is not a closure; note it in MC-16b. | L (MC-16b) |
| 17 | Kernel families and their readouts: KARE beside BBC-CV, the multi-index name, sliced inverse regression, an NTK view; xRFM watched | `FamilyExplanation` and `Architecture` in `core/models/explain.py` (MC-8 adds `spectrum` and `alignment`); the phenomena registry (MC-10) | planned in v2 (MC-1, MC-7, MC-8, MC-10) | none | KARE must never rank families before Fit (MODEL_FAMILY_CONTRACT §2.1). *Guard:* MC-5's outcome-permutation test on the shelf. | L |
| 18 | LightGBM | `register_family` and the contract. `lightgbm>=4.3` is already a server requirement (`server/requirements.txt:14`), and its version is recorded in provenance (`core/export/record.py:23-24`). MC-1's `same_kind_as` lets it reuse boosted trees' reads. | planned in v2 (MC-1, MC-12; RT-1's `TuningDecl`) | none | As row 14(a) | M (XGBoost is M–L in RT-5e; LightGBM reuses its wrapper's interface) |
| 19 | ExtraTrees, kNN, SVM, naive Bayes, LDA | As row 18 | planned in v2 (as row 18) | none | `register_family` will refuse a yes/no family without `decision_function` (MODEL_FAMILY_CONTRACT C9). kNN and naive Bayes need a margin wrapper. This is not a closure. | S each (ExtraTrees, naive Bayes, LDA); M each (kNN, SVM) |
| 20 | Monotone constraints, declared from domain knowledge | RT-2's `family_spec`, resolved inside `build_pipeline` (RECIPES §2.4), and RT-7's version identity. Pipelines pass data frames (`set_output(transform="pandas")`, `core/models/pipeline.py:700`), so per-column constraints can be keyed by column name, as scikit-learn's and XGBoost's constraints accept. | planned in v2 (RT-2, RT-7) | none. A per-column answer is a new kind keyed by family and column through `register_kind(key=…)` (`core/decisions.py:2184-2250`). That is additive. | The version key (row 3). *Guard:* row 3's. | M |
| 21 | Survival versions of the new families | A family declares `time_to_event` among its tasks (`core/models/base.py:21`, 137) and MC-1's `raw_scale["time_to_event"]`. The prediction path is `metrics.predict` (`core/models/metrics.py:528-544`) with `survival_baseline` (547-558). | exists today | none in v2. An optional `risk_by(X, horizon)` member is additive when the first family without proportional hazards arrives. | **(a)** The prediction path reads `model.predict(X)` as a log relative hazard and fits Breslow's baseline on it (`core/models/metrics.py:536-544`, 547-558). A survival forest's risk would be forced through a proportional-hazards baseline. *Guard:* the `risk_by` member in v2.x. **(b)** Some switches on `"cox"` sit outside MC §3.3's list: `core/methods/interaction.py:602` takes the family as a plain string, which the syntax-tree test may not recognize as a family key. *Guard:* add it to MC-2b's census. | M per family |

### 5 · Inference, designs and estimands

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 22 | Mediation analysis, including "only the direct part" | `EffectKind = Literal["total", "direct"]` (`core/decisions.py:1217`). The direct effect's own answers, `confounds_mediator`, `interacts` and `interaction_attested` (`core/decisions.py:1293-1295`). `direct_questions` and `mediators` (`core/estimand.py:719`, 754). A refusal that already names "counterfactual mediation methods, which TurboTab does not fit", with three exits (`core/estimand.py:1999-2016`). | exists today. The controlled direct effect at a reference level is routed end to end, and v2 turns it into "Not available yet" (`crosswalk/CROSSWALK.md:171`). | none. Natural direct and indirect effects add `EffectKind` values and a mediator field, additively. The refusal at `core/estimand.py:1999-2016` gains the mediation exit. | v2 might build "Not available yet" by deleting `"direct"` from `EffectKind` and its answers. Logs that recorded it would stop loading (`core/decisions.py:5274`), and the mediator-and-outcome confounder logic v2.x needs would be gone. *Guard:* keep the value and the code. Refuse `effect="direct"` with a `*_v2x` code, its reason and the exit to the total effect, as `share_reallocation` is refused (`core/methods/exposure_form.py:2390-2403`; tested at `core/tests/acceptance/test_form_8_energy_labels.py:100-105`). A mediation stage must also join the hand-kept `ESTIMATE_STAGES` (`core/estimand.py:1321-1324`), or it escapes the lock (rule 6). | L |
| 23 | Crossover trials | The design slot (P0.6) and the design question with "Not available yet" exits (E1; `crosswalk/SIZING.md:52`, 106); the crosswalk's `refusal:not-available-yet` (`crosswalk/crosswalk.json:4413-4450`) | planned in v2 (P0.6, E1). No design slot exists today. | none beyond E1's scope | Suppose the design vocabulary holds only v2's four designs, routed by `if design == …` branches across the Router. Each later design is then a sweep. *Guard:* one registry of designs. Each declares its estimands, adjustment rule, analysis sets and checklist, and whether it is available. Crossover, repeated-measures trial models and complier effects are registered as unavailable from the first commit, each refused with a `*_v2x` code (rule 2). | L |
| 24 | Repeated-measures trial models (MMRM) | The design registry (E1). The families: today's `mixed` is a random-intercept model (`core/voice.py:1232`, registered at `core/models/__init__.py:30`). Grain and time points (`core/decisions.py:777`, 784). | planned in v2 (E1). The model family exists in part. | none | Widening `mixed` in place would change a recorded family's meaning (row 9(b)'s problem). *Guard:* register the MMRM as its own family. | L |
| 25 | Complier and per-protocol effects | E2's analysis sets ("the set, not the per-protocol effect", `crosswalk/SIZING.md:107`); `CausalPopulation` (`core/decisions.py:1414`); the time-varying lane's inverse-probability weights (`TimeVaryingMethod`, `core/decisions.py:1468`) for adherence | planned in v2 (E1, E2) | none | One "per-protocol" flag might come to stand for both the analysis set and the effect. *Guard:* keep the set and the effect as separate fields. The crosswalk already distinguishes them (`crosswalk/crosswalk.json:4425`). | L each (an instrumental-variable estimator for compliers; weighting for adherence) |
| 26 | Within-person designs | Grain and repeats (`core/decisions.py:777-794`); the `mixed` and `gee` families (`core/models/repeated.py`); the exposure forms (`ExposureFormKind`, `core/decisions.py:980`); the noticing `shared-ema-within-between` (`crosswalk/CROSSWALK.md:934`) | exists today | none. A within- and between-person split is one more form or estimand value. | The cluster answer's fixed effects are for "a grouping above the person" (`core/decisions.py:1187-1189`). The person's own fixed effect must not be routed through it. *Guard:* a note in the form spec. | M |
| 27 | Wearable and glucose-monitor summaries | The rules that combine a unit's records (`CombineRule`, `core/decisions.py:795`; `COMBINE_RULES` and `combine_sql`, `core/stages/working.py:1355`, 1601); the domain packs (`PACKS`, `packs.py:5217`); the dense-series gap G4 (`understanding/UNDERSTANDING_LAYER.md:804`); the lens vocabulary (`core/decisions.py:51`) | exists today | none in v2 | **(a)** A combine rule is a Literal value, an SQL branch, and words in `core/voice.py:1880-1935` and `core/structure_previews.py:44`. That is five places per summary. *Guard:* make it a registry when the first wearable summary is added. **(b)** A new lens meets five copies of the omics-lens list (`core/estimand.py:964`, `core/methods/omics.py:62`, `core/stages/rows.py:32`, `core/models/featurewise.py:57`, `core/interview.py:114`) and lens tests spread through core (for example `core/stages/rows.py:417`, `core/estimand.py:680`, `core/methods/interaction.py:118`). *Guard:* one declared lens trait, with the no-switch test widened to lens keys. | L |

### 6 · Goals, data in and the platform

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 28 | Goals run as parallel tracks | D2: a track id on every record, and per-track outcome, goal, seal, plan and lock (`crosswalk/SIZING.md:95`); keyed slots (`register_kind(key=…)`, `core/decisions.py:2184-2250`) | planned in v2 (D2) | none beyond D2's scope | **(a)** A stage name exists once per graph (`core/graph.py:107-109`). Suppose tracks run by swapping a "current track" into one `fit` stage. Two tracks can then never be fresh together. *Guard:* per-track stage instances, with the track id in the stage key (`core/graph.py:271-287`), and a test that two tracks' fits are fresh at once. **(b)** The rows rule is stated in the fixed order Estimate, Describe, Predict (`crosswalk/CROSSWALK.md:163-169`). *Guard:* D2 states it from the log: did another track read this outcome before this track's seal? That holds in any order. **(c)** The fact that estimates were shown under prediction lives in one side file per project, outside the log (`SHOWN_UNDER_PREDICTION`, `server/service.py:54`, 1235-1240). *Guard:* keep it per track, as a system record in the log. | L |
| 29 | Deep multi-file assembly: fuzzy keys, conflict resolution, rows combined before joining | `JoinFiles` and `JoinSpec` (`core/decisions.py:1820-1846`); `assembly.plan` (`core/assembly.py:223`); the join inside ingest (`core/datastore.py:1407-1415`); the join's contract (`core/assembly.py:44-60`) | exists today (the minimal form) | none. Compound or fuzzy keys and a combine-first step are new optional fields and contract options. The many-to-many refusal gains "combine first" as an exit (`core/assembly.py:14-18`). | `JoinSpec.on` is one recorded string (`core/decisions.py:1820-1828`). Retyping it to a list would break old logs. *Guard:* add a field beside it and never retype (rule 3). | L |
| 30 | The All of Us adapter | The readers: `_source_kind` (`core/datastore.py:407-423`) and `ingest`'s dispatch (`core/datastore.py:1402-1406`), which the server imports as "the one list of readable types" (`server/service.py:24`). Server mode with sign-on from a proxy (`server/auth.py:20-24`). Small-cell suppression (C8). | exists today: the Dataset Builder's CSV and Parquet exports read now, and server mode runs in a container. Suppression is planned (C8). | none. Another source is one more branch in two places. | All of Us forbids reporting counts under 20 participants (`reference/PRODUCT_VISION.md:391-392`). A suppression threshold written as a constant in C8 would need a sweep. *Guard:* C8 reads the threshold from the deployment's settings, with a test that a threshold of 20 suppresses a count of 19. | M |
| 31 | Multi-user collaboration | Per-user workspaces and ownership (`user_home`, `owns`, `server/tenancy.py:23-32`); the append-only log under an exclusive lock (`core/decisions.py:5177-5225`); `DecisionRecord` (`core/decisions.py:1937-1956`) | exists today: accounts exist, and concurrent appends are safe | none. Every v2 log has one author by construction, its workspace's owner, so an optional author field later is additive. | Ownership is where the folder sits (`server/tenancy.py:28-32`). Sharing means `owns` also consults a list, which is a local change. Low. | L |
| 32 | The research forum | The export bundle (`core/export/bundle.py`), with versioned provenance (`FORMAT = "turbotab-provenance/1"`, `core/export/record.py:21`) and a versioned plan (`PLAN_FORMAT = "turbotab-analysis-plan/1"`, `core/plan_lock.py:80`). The research record (DoD, 2026-10-08): outcome-blind descriptors per fit, comparisons in a versioned format, pooling by opt-in. | planned in v2 (the research record; no SIZING package names it yet) | none beyond building the record as approved | A research record without a format string, or one that carries row values. *Guard:* a format string like provenance's, and a test that the record holds no row of data, as provenance holds none (`core/export/record.py:1-10`). | XL |
| 33 | Headless Python and R packages, and an agent skill | `ProjectService` (`server/service.py`), which needs no HTTP server and which replay already drives headlessly (`core/export/replay.py:29-30`); the reference journeys (`core/reference/journeys.py:1-10`); the OpenAPI document (`server/openapi.json`, `info.version` 2.0.0.dev0); the thin driver, "run this plan file on this data" (DoD, 2026-10-08, testing) | planned in v2 (the driver). Replay and the journeys exist today. | none beyond the driver's scope | **(a)** The plan locks in the server layer, when an estimate is first served (`_lock_when_shown`, `server/service.py:1216-1247`; `_lock_after_prediction`, 1249-1265). A package that called the stages directly would never lock. *Guard:* packages wrap `ProjectService`, as replay does, and a driver test asserts that `lock_plan` is recorded. **(b)** Raw `decisions.jsonl` lines carry no format version (`core/decisions.py:5256-5277`). *Guard:* the driver reads a versioned plan file, either `turbotab-analysis-plan/1` or its own `/1` (rule 5). **(c)** `ProjectService` still imports FastAPI, through its one error type (`server/service.py:39`; `server/errors.py:12-13`), so a pip package would carry the web stack. *Guard:* the error type the service raises moves to core when the package is built. That is a local change, not a closure. | XL in all (Python L, R L, the skill M) |
| 34 | An in-app AI assistant | The one door for answers: `parse_decision` (`core/decisions.py:1932`), the validators (2263) and refusals with exits (2124). The previews (`register_consequence`, `core/consequences.py:394`). The methods reference generated from the contracts (`core/reference/methods.py`). The purpose registry (`core/purposes.py`). | exists today | none | Until the one primary action, choices on the Models card stay in the card's local state (RECIPES §4.4), so a server-side assistant cannot see a draft. The leash's "never pre-select" must also bind the assistant. *Guard:* the assistant proposes previews and exits and records nothing without the user's press. What it suggested goes in the record's `note`. | XL |

### 7 · INBOX.md's v2.x method items

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 35 | Compositional (ilr) reallocation of shares (INBOX.md:171) | `SubstitutionScale` names `share_reallocation` (`core/decisions.py:439`). A validator refuses it with the code `share_reallocation_v2x` and two exits (`core/methods/exposure_form.py:2390-2403`, 2637). | exists today | none | None. This is the pattern rule 2 asks of every deferred option. | L |
| 36 | SEM attenuation correction for reflective scales (INBOX.md:172) | `ScaleCorrection` (`core/decisions.py:284`) in the scales contract (`core/scales.py`) | exists today | none | none | L |
| 37 | Full two-level FCS imputation (INBOX.md:173) | `ImputationLevels` (`core/decisions.py:456`); the imputation contract (`core/methods/missing.py:470-490`) | exists today | none | none | L |
| 38 | NCI usual intake with person-level covariates and subgroup distributions (INBOX.md:180-181) | `UsualIntakeSpec` (`core/decisions.py:361`); the slot keyed by nutrient (`core/decisions.py:2540`); `core/usual_intake.py:54` | exists today | none | The slot holds one analysis per nutrient (INBOX.md:180). *Guard:* a composite key that equals the nutrient when no subgroup is named, so old records fold as before. | M |
| 39 | NCI never-consumers, and the multivariate NCI method (INBOX.md:182) | `UsualIntakeModel` (`core/decisions.py:353`); `core/usual_intake.py:142` | exists today | none | none | L each |
| 40 | Prevalence of inadequacy by the probability approach (iron), and per DRI life-stage group (INBOX.md:210-211) | The refusal and its exits (`core/stages/usual_intake.py:500-502`); `CutoffKind` (`core/decisions.py:355`) | exists today | none | none | M each |
| 41 | DML and TMLE pooled over multiple imputations (INBOX.md:202) | The relation `mi_blocks` (`core/causal.py:178`) | exists today | none | none | M |
| 42 | A Super Learner as the nuisance learner (INBOX.md:203) | Row 9's seam | exists today | none | Row 9's | M, after row 9 |
| 43 | Post-double selection: a survey-weighted plug-in penalty, and a cross-validated penalty (INBOX.md:204; RECIPES §10) | `CausalMethod` `pds_lasso` (`core/decisions.py:1410`); the relation `survey_blocks_pds` (`core/causal.py:169`) | exists today | none | none | M |
| 44 | The Brier score with delayed entry, by truncation weights (INBOX.md:207) | The primary metric's fallback (`core/models/metrics.py:67-68`, 119) | exists today | none | none | M |
| 45 | Tobin et al.'s remedies for a medication that treats the outcome (INBOX.md:217) | `core/covariate_guesses.py:36`, 71, as a declared secondary analysis beside the primary, like the model sequence (`core/decisions.py:1367-1381`) | exists today | none | none | M |
| 46 | SMOTE inside each fold (INBOX.md:237) | `ImbalanceCorrection` (`core/decisions.py:1579`); `core/methods/levers.py:696` | exists today | none | none | S |
| 47 | The benchmark, the interpretable model's cost and the decision curve for time-to-event and ordinal outcomes (INBOX.md:238) | `core/stages/evaluation.py` | exists today | none | none | M each |
| 48 | Portable plans: one plan applied to new data, and a plan written from the codebook (INBOX.md:242) | `plan_lock.plan_document` (`core/plan_lock.py:116`); replay; the headless driver (row 33) | planned in v2 (the driver) | none | A plan names columns, and file ids drawn per project (`FILE_ID`, `core/decisions.py:1798`). *Guard:* the driver maps files by role, not by id. | L |
| 49 | Flat BBC-CV over every configuration (RECIPES §4.5, §10) | `selection_optimism` resamples any set of out-of-fold prediction columns (`core/models/selection.py:245-251`) | exists today | none | The nested search gives no outer out-of-fold predictions per configuration. They cost one fit per configuration per outer fold. | M |

**No longer deferred.** INBOX.md:250's tables (the causal, time-varying, substitution and other results) are in v2: C8 tabulates every stage (`crosswalk/SIZING.md:82`).

### 8 · Readings (added 2026-10-10, on Nolan's approval)

| # | Item | Seam (file:symbol) | Status | Missing hook and size | Closure risk and guard | Later effort |
|---|---|---|---|---|---|---|
| 50 | **A local, calibrated reading proposer with conformal sets.** Each reading (what a blank means, amount or code, before or after the outcome, units, weight, role, codebook match) becomes a typed classification. A calibrated classifier proposes the set of readings it cannot rule out, with a coverage guarantee.<br>**Its routing:**<br>• one reading left: Confirm, listed in the sweep with its evidence;<br>• several left: a Decide with exactly those options;<br>• none trusted: an open question.<br>With conformal risk control, the routing carries a stated guarantee, such as "at most 2% of readings set for you are wrong". Features are each column's statistics, its name and its codebook text, the text embedded by a small open model on the machine. The labels are the reference journeys, then each person's confirmations and corrections, which never leave their machine. **Open and local only: no proprietary or cloud classifier** (Nolan, 2026-10-10). | The rule-based recognizers: `core/readings.py`, `core/covariate_guesses.py`, `core/ask.py`, and the `likely_not_asked` flags read in `core/coach.py` (292, 631) and `core/row_previews.py` (438). The readings ledger (BLUEPRINT §14.1) and BLUEPRINT §14's rule that "recognizers are measured on a growing corpus of real-world exports, and their accuracy is reported, not assumed". The surfacing tiers and materiality (`SURFACING_POLICY.md` §2, §4). | exists in part: the rules exist and the ledger records confirmations, but there is no single proposer interface and no calibrated score. | **None required in v2.** A recommended guard, not scheduled (the DoD is frozen): every rule-made reading records its source (rule id and version) and the alternatives it considered, so the log already holds labeled examples when v2.x trains on them. **S (1).** | **(a) The leash.** A proposer may only choose *how* a reading is asked, never whether it is asked. A number-changing reading is still listed with its evidence and confirmed (§14.1), and materiality still sets the tier. *Guard:* the proposer returns a candidate set and a score into the existing tier function; it never writes a reading. **(b) Reproducibility.** A learned model makes routing depend on a model version. *Guard:* the proposer's identity, version and output are recorded with each reading, and replay reads the recorded routing, never a re-run. **(c) Calibration is measured, not assumed.** *Guard:* the reference journeys' readings are the held-out benchmark, and coverage is reported per reading kind before any kind switches from rules to the proposer. | L: the proposer interface (S), the features and the classifier (M), conformal calibration with risk control (M), the benchmark on the reference journeys (M) |

---

## The missing hooks

| Hook | Opens | Where it lands | Size |
|---|---|---|---|
| `TuningPlan.strategy`. The candidate generator and the fit count belong to the strategy, and `TuningRecord` lists candidates in the order they were evaluated. | Rows 1–2: successive halving, Hyperband, TPE and BOHB | RT-1 and RT-8 | S (1) |
| The trunk stage key and the log sequence, stored with each read-only row's scores in the versions file, not only in the stage cache | Row 11: shared-step changes kept as versions. It also keeps v2's own read-only row alive when the cache is cleared. | RT-7 and C7d | S (1) |
| **Total** | | | **2 units** |

---

## The closure risks, in order of harm

1. **The decision log is unversioned and strict.**
   - `DecisionLog._load` validates every line with `DecisionRecord.model_validate` (`core/decisions.py:5274`). Every decision model forbids extra fields (`core/decisions.py:73-79`, 1937-1938).
   - So a renamed, retyped or removed recorded value makes the whole project unloadable. Replay writes a bundle's log "byte for byte" and folds it (`core/export/replay.py:14-17`), so old bundles stop replaying too.
   - v2 schedules one such rename (`random_forest` to `nuisance_forest`, DoD 2026-10-08), and the direct effect is at risk (row 22).
   - **Precedents:** RECIPES already plans a legacy mapping for `scores_seen.json` (`LEGACY_DEFAULTS`, RECIPES §3.2), and the provenance and plan formats carry version strings (`core/export/record.py:21`; `core/plan_lock.py:80`).
2. **A recorded key keeps its spelling while its meaning changes.**
   - The causal `boosted_trees` learner is the family's untuned build today (`core/models/causal.py:408-412`; `core/models/boosted_trees.py:34-41`), and RT-5a tunes the family.
   - The same trap waits for any family whose defaults change, which is why RECIPES puts `defaults_version` in a version's identity (§3.2).
3. **Version keys hash the whole `VariantSpec`** (RECIPES §3.2). Each v2.x addition would re-key every recorded version and leave phantom "earlier versions": a search strategy, a budget, a monotone constraint, an MLP's embedding slot (MODEL_FAMILY_CONTRACT C3).
4. **Tracks built for sequence only.**
   - A stage exists once per name (`core/graph.py:107-109`).
   - The shown-estimates record is one side file per project (`server/service.py:54`, 1235-1240).
   - The rows rule is written in a fixed order (`crosswalk/CROSSWALK.md:163-169`).
5. **Lists kept by hand.**
   - `ESTIMATE_STAGES` (`core/estimand.py:1321-1324`) decides what the lock withholds, and it has already left one estimate-like stage undecided (usual intake, INBOX.md:185).
   - The omics-lens list exists five times (row 27).
   - The eighteen switches on family keys (MODEL_FAMILY_CONTRACT §3.3) go with MC-2b. Not all switches are on its census (row 21).

---

## Rules that keep seams open

1. **No switch on a key outside its registry.** That means family, learner, lens, design and method keys. MC-2b's syntax-tree test covers families; widen it to learners, lenses and designs.
2. **A deferred option is a named value, refused.** It carries a `*_v2x` code, its reason and an exit that keeps the work: the `share_reallocation` pattern (`core/methods/exposure_form.py:2390-2403`). Never defer an option by deleting its value.
3. **Recorded vocabularies only grow.** A recorded value or field is never renamed, retyped or reused with a new meaning unless a parse-time alias carries the old one. Retired values stay as tombstones.
4. **Decisions and specs grow by optional fields with defaults.** Identity hashes are computed over the fields that differ from their defaults.
5. **Every persisted record carries a format string,** as provenance and the plan already do: the decision log, the research record, the driver's plan file. Side files that decide the leash (`estimates_shown.json`, `scores_seen.json`) move into the log or carry a format too.
6. **A stage declares what it serves** (an estimate, an outcome view). Gate lists such as `ESTIMATE_STAGES` are derived from those declarations, never kept by hand.
7. **Per-track or per-version state is keyed** by track or by version. Nothing reads a single "current" pointer.
8. **Methodology lives in core.** The server, the driver and any headless package reach it through `ProjectService`.
9. **Every new kind still passes the registries that fail on an unaccounted kind:**
   - the reference catalog (`core/reference/catalog.py:15-22`);
   - the purpose registry (`core/purposes.py:1-10`);
   - the fold-in gate (MC-12).

---

## Recommendations (for the orchestrator to judge)

Each is S or smaller, sits inside a package already planned, and prevents a rewrite or a silent change of meaning. None adds a feature to v2.

1. **A format marker on the decision log, with migrations applied before validation** (S).
   - **What:** an ordered list of raw-line migrations runs before `DecisionRecord.model_validate` (`core/decisions.py:5274`). A committed corpus of log lines from each release must keep loading.
   - **Why:** v2 itself renames a recorded value. Without this, every v2.x retirement either breaks old projects or is avoided by keeping wrong names forever.
2. **Rename every causal learner key that collides with a family, with aliases and tombstones** (S, inside MC-2b's rename).
   - **What:** rename `boosted_trees` together with `random_forest`. Old spellings parse as the new ones, and retired spellings are never reused.
   - **Why:** this is the only place found where a recorded word would silently change meaning once the C6c registry learners return.
3. **`TuningPlan.strategy`** (S, inside RT-1 and RT-8).
   - **Why:** the four displaced search algorithms all plug in here. Without it, RT-1's fixed-list assumptions settle into the cost estimate, the replay and the record.
4. **Version keys over the fields that differ from their defaults** (S, inside RT-7).
   - **Why:** the Thorough budget, the strategies, monotone constraints, native categories and the neural families' recipe slots all add to `VariantSpec`. Without it, each addition re-keys every version a user has seen.
5. **The trunk key and log sequence persisted with the read-only row** (S, inside RT-7 and C7d).
   - **Why:** it opens shared-step versions. It also fixes a v2 durability gap: the row is otherwise read from a cache that is disposable (`core/graph.py:307`).
6. **Deferred designs and estimands registered as unavailable, not left out** (S, inside P0.6 and E1).
   - **What:** crossover, repeated-measures trial models, complier and per-protocol effects, and the direct effect are named values. Each is refused with a `*_v2x` code and an exit, in one design registry.
   - **Why:** each v2.x design then flips a declaration instead of threading a new value through the Router. The direct effect's mediator machinery, which mediation needs, survives.
7. **Estimate stages declared on `Stage`, with `ESTIMATE_STAGES` derived** (S, inside P0.4).
   - **Why:** every v2.x estimate stage (mediation, compliers, MMRM) must sit behind the lock. A hand list has already left one stage undecided (INBOX.md:185), and v2's own new stages (Describe's estimators, the trial analyses, `late_notices`) meet the same list.
8. **D2's track machinery written for any order** (spec notes and one test, no extra package).
   - **What:**
     - per-track stage instances, with the track id in the stage key;
     - the rows rule stated from the log's sequence numbers;
     - the shown-estimates record per track, in the log.
   - **Why:** sequential tracks are then parallel tracks with a different scheduler.

Recommendations 1 to 8 come to at most 8 units, about 1% of SIZING's 734 remaining. Several are a few lines inside a package already being written.

**Considered and not recommended for v2:**
- An author field on records: every v2 log has one author by construction (row 31).
- A reader registry for new sources: the dispatch is two local branches (row 30).
- A registry of combine rules: build it with the first wearable summary (row 27).
- A survival `risk_by` member: additive when the first such family arrives (row 21).
- One lens trait in place of the five omics lists: no new lens is on the deferred list. Do it when one is planned (row 27).

---

## How this was built

- **Every deferred item** was gathered from the five sources above and given one row; items that share a seam share a row.
- **Seams in the code:** for each item, the code was read where it would plug in. That covered:
  - the family registry and the pipeline builders;
  - the method contracts;
  - the decision kinds and the log;
  - the estimand;
  - the causal lane;
  - the graph and the stages;
  - the export, replay and plan formats;
  - the server's tenancy and lock;
  - ingest and assembly;
  - the packs and the lens lists.
- **Seams not yet in the code:** these were checked against the package that creates them (RECIPES §9, MODEL_FAMILY_CONTRACT §5, SIZING).
- **Closure risks** were sought where a recorded value, a closed list or a single pointer would have to change for the item to fold in.
- **Nothing was run.**
