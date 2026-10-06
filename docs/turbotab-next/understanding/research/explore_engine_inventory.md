## Explore inventory: what the engine serves for exploration (read-only, branch turbotab-next)

Paths are relative to /Users/nhedglin/tabular-ml-lab/. The foundation file is .worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md (§3 at lines 37-63, §5 at 88-127).

**The main finding.** The engine already computes an Explore stage: findings, levers and a forking-paths record. But nothing asks for it and nothing draws it.
- Its finding shape is not the canvas's view vocabulary.
- It ignores the lens.
- It does not rank what it shows.

So a "dynamic canvas that points at the interesting pieces" has most of its parts. What it lacks is a ranking and lens-specific views.

---

### 1. Where Explore sits
- **Stage wiring.** `Stage("explore", 1, ("working","cohort","split","target_info"), …, requires=("target","split"))` is at turbotab/core/stages/__init__.py:774-783. The artifact model is registered at turbotab/server/schemas.py:1165-1168.
- **Not routed.** "explore", "levers", "selection" and "intended_use" are not in the Router's `QUESTION_KEYS` (turbotab/core/interview.py:99-106). docs/turbotab-next/INBOX.md:226 confirms the levers are served as stage data with decision payloads, but "the Router does not ask them yet".
- **Not rendered.** The frontend fetches 15 stages (turbotab/frontend/src/components/record/Record.tsx:163-179). Neither `explore` nor `evaluation` is among them. Explore exists in the frontend only as:
  - generated types: turbotab/frontend/src/api/generated.ts:8825-8863 (ExploreArtifact, ExploreFinding), :9731 (HandLever), :10296 (LeverOption);
  - a record-slot mapping: turbotab/frontend/src/components/record/sentences.tsx:159-168;
  - mock cases: turbotab/frontend/src/mocks/db.ts:698-706 and 870-875.
- **Not the frontend folder of the same name.** turbotab/frontend/src/explore/ holds design prototypes for the stage canvas (turbotab/frontend/src/explore/stage/StageScreen.tsx:1-13). It is not the Explore stage.

### 2. Artifact fields (turbotab/core/stages/explore.py)
**`ExploreArtifact`** (132-144):

| Field | What it holds |
|---|---|
| `purpose` | the recorded purpose |
| `rows` | "training" or "analyzed" |
| `n_rows`, `n_holdout` | row counts |
| `findings[]` | the findings below |
| `more` | relationship curves beyond the 12 shown |
| `proposed_subgroups[]` | columns named like a sociodemographic group |
| `viewed[]` | keys `"<view>:<column>"` of outcome views recorded as looked at |
| `hand_levers[]` | levers set by hand after a view |
| `sentence` | the methods sentence |

**`ExploreFinding`** (108-120):
- `id`, `kind`, `summary` (at most 20 words), `columns`
- `outcome_view`, `viewed`
- `view`: one word, `relationship`, `distribution` or `table_focus`
- `points[BinPoint{x,y,n}]` (96-99), `groups[GroupQuality{group,n,missing_share}]` (102-105)
- `detail`, `lever: Lever{question, options[LeverOption]}`, `record` (the `view_outcome` payload a client posts when the view is opened)

**`LeverOption`** (81-88): `key`, `label`, `customary`, `sound`, `rung` ("recommended" / "available" / "rank_lower"), `in_fold`, `decision`.

**`HandLever`** (123-129): `column`, `view`, `what`, `then`, `now`, `sentence`.

### 3. The finding kinds Explore computes (`explore_stage`, 368-478)

| kind | When it fires | Data it carries | Lever, prediction | Lever, inference |
|---|---|---|---|---|
| `outcome_relationship` (407-421) | numeric predictors with at least 10 distinct values (`MIN_DISTINCT`, turbotab/core/methods/levers.py:55); only for regression or binary outcomes (396-397) | 10 binned means, or event shares, over the predictor's tenths (`relationship_points`, 270-285); the **first 12 in predictor order**, the rest counted in `more` | `form_lever` (324-365): `set_levers forms=rule`, `inner_cv` (in-fold), or a by-hand `set_exposure_form` spline with its cost stated | `set_exposure_form` spline (k by Harrell's rule) or linear, "declared before estimates" |
| `outcome_distribution` (481-516) | regression: a 20-bin histogram; binary, multiclass, ordinal: class shares | for classes, `x` is an **index with no class label** (500-502) | rare class (<20%): `_imbalance_lever` (588-607), no correction ranked first, then weights / undersample / oversample marked `rank_lower`. Common class: `_use_lever` (564-576), `set_intended_use` | `_measure_lever` (binary, 579-585) or `_scale_lever` (552-561); both have `decision=None` |
| `low_variance` (431-440) | caret nearZeroVar rule | column names only | `set_levers variance_filter=near_zero`; "by hand" has `decision=None` | "answer it in the adjustment set", `decision=None` |
| `wide` (441-446) | number of predictors at least the number of rows | column names only | `set_selection screening` or `elastic_net`; `set_levers top keep=1000` | `select_models featurewise` (with FDR) |
| `collinear` (447-456) | \|r\| of 0.90 or more; skipped above 500 numeric predictors | `view="relationship"` but **no points**; the pairs exist only as text in `detail` | `set_selection elastic_net`; "by hand" has `decision=None` | adjustment set, `decision=None` |
| `quality_by_group` (686-729) | columns named like sex, age, race/ethnicity or income (`GROUP_WORDS`, 69-74) | `groups[]` with each group's missing share; called out when the gap is at least 5 points | `set_intended_use subgroups+=column` (TRIPOD+AI 23a) | the missing-values question, `decision=None` |
| `survey_design` (519-549) | prediction only, when a survey design is read | weights, strata and PSU column names | decisions from `survey.offered` (design-based CV) | none |

There are no outcome views at all for multiclass, ordinal or time-to-event relationships, or for categorical predictors.

### 4. Lever decisions and how they are recorded (turbotab/core/decisions.py)
- **`SetLevers`** (1569-1586): `forms` none / rule / inner_cv; `variance_filter` none / near_zero / top; `keep`; `imbalance` none / weights / undersample / oversample.
  - Refused under inference (`levers_not_inference`, turbotab/core/methods/levers.py:674-691).
  - Refused for a task it does not fit, or for forms when p ≥ n (694-725).
  - Sentence at 738-764: "Within each training fold, … its optimism is in the corrected score."
- **`SetSelection`** (1592-1614): methods none / elastic_net / stability / screening / stepwise / univariable / vip; `where` in_fold or outside; `pre_selected`; `sensitivity`.
- **`SetIntendedUse`** (1620-1637): `use`, three thresholds, `subgroups`, `fairness`.
- **`SetUpdating`** (1642-1648).
- **`SetExposureForm`** (1007-1022).
- State slots are at 2069-2075; the fold that registers them is at 2545-2557. Fitted steps come in through `explore_steps` (turbotab/core/models/pipeline.py:651-665).

### 5. Outcome views and forking-paths disclosure
- **Decision.** `ViewOutcome{view: relationship|distribution, columns, target, rows, n_rows, levers}` at decisions.py:1542-1561. It folds to one entry per column, so a later look keeps the first look's answers (2547-2551).
- **Validator** (explore.py:735-750): the outcome must be chosen and the column must exist.
- **Server completion** (753-781): fills in the target, the rows, n_rows, and each column's lever answers at the first look (`lever_answers`, 160-175).
- **Hand levers.** `hand_levers` (190-217) compares the answers then and now:
  - under prediction it says "set by hand … outside the corrected score; the held-out rows, never viewed, cover it";
  - under inference it says "changed … after [the view] (forking paths)".
- **Where the disclosure lands:**
  - the record sentence and its standing clause (`view_sentence` / `view_standing`, 794-814; registered in turbotab/core/voice.py:1996-2011);
  - the Explore sentence (226-260);
  - the evaluation stage, which adds "The corrected score does not include the optimism…" (turbotab/core/stages/evaluation.py:353-360).
- **Contract.** The `explore` method contract (817-875) carries five relations: views recorded, in-fold first, by-hand outside the corrected score, holdout covers, quality across groups.
- **`view_outcome` has no preview.** It is listed in `UNPREVIEWED` (turbotab/core/consequences.py:960-963).
- **It does not lock the plan.** `lock_plan` fires only when an *estimate* is served (turbotab/core/plan_lock.py:14-19). So the engine has two tiers: outcome views are recorded and disclosed, while estimates lock the plan.

### 6. Previews of Explore's answers (turbotab/core/explore_previews.py)

| Decision | What the preview draws | Under inference |
|---|---|---|
| `set_levers` (102-167) | lineage "Explore's levers, in each training fold": which predictors bend (the knots) and which leave by the filter; imbalance adds a row_flow of the rows each fold trains on | returns nothing |
| `set_selection` (173-212) | lineage of the terms kept and dropped, on training rows | "The declared model stays as declared" (181-188) |
| `set_intended_use` (258-312) | the fitted risks with threshold marks (binary decision support, after a fit); rows per level of the first subgroup | nothing |
| `set_updating` (318-364) | each prediction before and after shrinkage | nothing |

Registrations are at 367-370. No coach annotator exists for any of these kinds: the only ones are for exclusions, missing values, energy and aggregation (turbotab/core/coach.py:468-471).

### 7. The view vocabulary the canvas can draw (turbotab/core/consequences.py)
- **Five kinds**, unioned at 293-296:
  - `row_flow` (227-232): steps, plus an optional `seal` of cells;
  - `lineage` (235-239): nodes and links across the raw / adjusted / matrix lanes;
  - `table_focus` (248-257): at most 12 columns and 8 rows, plus the changed cells;
  - `distribution` (268-276): before and after histograms, plus `marks`;
  - `relationship` (279-290): at most 800 points before and after, plus r.
- **Shared parts:**
  - every view: title of at most 8 words, caption of at most 20, `emphasis`, `coach` (191-195);
  - coach notes: at most 2 per view, at most 12 words each, anchored to a column, a range, points or a step (163-186);
  - storyboard frames (109-161);
  - `Caution` with exits (299-309);
  - `PreviewResult{kind, views (at most 3), basis, note, caution}` (312-317).
- **Generic diff** (`diff_views`, 422-481). It ranks by measured change: rows dropped, columns added or removed, the standardized shift of the column that moved most, and the change in correlation (\|Δr\| of at least 0.2). This is the only existing scorer of "what's interesting", and it scores *changes*, not raw data.
- **Frontend renderers** (turbotab/frontend/src/components/stage/):
  - `views/{Distribution,Relationship,Lineage,RowFlow,TableFocus}.tsx`;
  - composed pictures (`composed.ts:17-35`: reshape_table, turn_table, seal_fork);
  - `coach/CoachLayer.tsx`;
  - `PreviewGrid.tsx:1-5` (one primary plus up to two secondaries; a secondary can be promoted).
- **Stage focus kinds.** Option, finding, banner and live (Stage.tsx:1-15, 43-47; state/focus.tsx:31-43).
- **Missing from the engine:**
  - the FOUNDATION's footprint and layouts (Focus, Strip, Flow, Routing, Angles): no "footprint" appears anywhere in the code, and `PreviewResult` carries none;
  - "Angles": `MethodContract` has no tradeoff field (turbotab/core/contracts.py:113-143).

### 8. Which rows each piece reads

| Piece | Rows |
|---|---|
| profile column summaries (turbotab/core/stages/data.py:57-109) | every row; before the target and before the seal |
| `/columns` and `/columns/{name}/histogram` (turbotab/server/routes/data.py:66-109; turbotab/core/datastore.py:1901) | every row, the outcome included; **these routes ignore the seal** |
| findings stage (turbotab/core/stages/findings.py:147-249; basis at 248) | all rows by design: the split waits for the findings (interview.py:139-142); this includes batch × outcome confounding (findings.py:208-210; turbotab/core/methods/batch.py:581-585) |
| finding evidence (turbotab/core/evidence.py:23-27; server at turbotab/server/service.py:1092-1116) | every row for "is the data what it says?"; training rows for modeling evidence once a split exists |
| previews (consequences.py:13-18, 323-345) | a pool sampled to 5,000 rows; held-out rows never; the pool is every analyzed row under inference |
| Explore (explore.py:376-383) | training rows (held-out never read) under prediction; every analyzed row under inference; not sampled |

### 9. Inference versus prediction in Explore
- **Rows.** Analyzed rows under inference, training rows under prediction (MODELING_SEQUENCE.md §0 ruling 3 at 59-65; §1 row 1 at 132).
- **Levers.**
  - Prediction: each lever is offered first as an in-fold rule; the by-hand option comes second with its cost stated.
  - Inference: forms are declared with `set_exposure_form`. Variance, collinearity and missingness point to the adjustment set or the missing-values question. A wide table offers feature-wise models with FDR. `set_levers` is refused.
- **Outcome views are never blocked in either purpose** (MODELING_SEQUENCE.md §4 at 280). The survey finding appears under prediction only.
- **Canvas per step.** MODELING_SEQUENCE.md §3 (267) lists Explore's canvas as distribution, relationship and table focus, answering "is my data okay; what will matter".

### 10. What each lens's pack contributes (pre-seal `findings` stage, not Explore)
The `Pack` structure has five parts: `looks_for`, `detectors`, `priors`, `reframings`, `hedges` (turbotab/packs.py:5110-5136). Superseded detectors are read in turbotab/core/detectors/__init__.py:38-101.

- **Dietary** (packs.py:5472-5476)
  - Findings: compositional, implausible_intake, energy_adjustment, atwater, survey_weights, partial_design, lonely_psu.
  - Priors: repeat_treatment, energy_adjustment, collinearity_figure (5505-5529).
  - App's own: `voice::survey_design_absent` and pooled cycles (turbotab/core/stages/finding_words.py:1227-1267).
  - **This is the only lens with bespoke evidence views** (energy against a nutrient; intake with the cuts marked; evidence.py:394-395) and evidence coach notes (coach.py:553-554).
  - The usual-intake stage is at stages/__init__.py:707.
- **Clinical** (5556-5559)
  - Findings: censored_values, text_numeric, mixed_result_type, mixed_units, default_value_mass, temporal_implausibility, number_format, impossible_vs_extreme (read by turbotab/core/detectors/plausibility.py:532).
  - Prior: missingness_direction (5626).
- **Metabolomics** (5234-5239)
  - Findings: redundancy, left_censored, run_order, pooled_qc, sample_roles, no_pooled_qc, acquisition_design, no_run_order, repeated_subjects, zeros_or_missing, already_transformed, duplicate_ids, empty_blocks, ion_modes.
  - Also: a wide-shape reframing (5294), priors (5310, 5343), hedges (877), and Pareto / log1p recipes (5161-5196).
  - App's own: `omics.scale_finding`, `zeros_finding`, `batch_findings` (findings.py:200-210).
- **Genomics** (5375-5377)
  - Findings: data_type, gene_id_excel_corruption, counts_p_over_n, gene_id_versions, gene_id_duplicates, gene_id_mixed_vocabulary.
  - Priors: model_ranking and normalization, with no default (5443, 5454).
- **Survey** (5643)
  - Findings: ordinal_declared and sentinel_codes (both read by turbotab/core/detectors/scales.py:412).
  - Priors: ordinal_encoding, reverse_coding (5682-5695).
  - The reliability (scales) stage is at stages/__init__.py:698.

Each finding gets `summary`, `routes_to`, `lever_label` and `group` (finding_words.py:1-13, 933-962). Families are listed at 857-889 and same-kind groups at 891-902. The questions they route to:
- **exclusions**: 345, 528, 543, 558
- **energy_adjustment**: 571
- **roles**: 595, 618, 722-736, 790, 1149-1267
- **models**: 610
- **missing**: 165, 808
- **event**: 177

**The rest of the clinical, metabolomics, genomics and survey findings fall back to the generic evidence** (a table of affected columns plus one histogram; evidence.py:360-392).

### 11. What the frontend renders today
- **The finding-card pattern.** Up to three cards are pushed. Same-kind findings share one paged card. The rest are counted and typed. Focusing a card puts its evidence on the stage, and its lever routes to the question that acts on it (turbotab/frontend/src/components/record/Findings.tsx:1-22, `PUSHED = 3`; BLUEPRINT.md:288-289). This is the nearest existing pattern for "point at the interesting pieces without overwhelming."
- **Two-panel layout.** Record on the left, stage on the right (turbotab/frontend/src/screens/ProjectScreen.tsx:90-101). When nothing is focused, the stage shows the live scenes "Your data now", rows, columns, models and results (LiveScenes.tsx:1-6).
- **Explore:** nothing is rendered.

### 12. Gaps that matter for an EDA canvas
1. **Explore is neither asked nor drawn.** See §1, and INBOX.md:226.
2. **`ExploreFinding` is not a `ConsequenceView`.** The frontend needs an adapter. Specifically:
   - collinear pairs carry no data;
   - low_variance and wide carry no table rows;
   - class distributions carry no labels;
   - there are no titles, captions or coach notes.
3. **Nothing is ranked.**
   - Relationships are the first 12 in predictor order (407-412), and there is no way to fetch the 13th.
   - No strength or nonlinearity score exists.
   - Explore findings have no `severity`, `routes_to` or `group`, unlike the pre-seal `Finding` (turbotab/server/schemas.py:436-458).
4. **Explore ignores the lens.** "lens" is in `EXPLORE_READS` (53) but never used. Every lens-specific picture lives in pre-seal evidence or domain-transform previews, and four of the five lenses have only the generic one.
5. **Many lever options are dead pointers.** About 7 options have `decision=None` and no route: scale, measure, adjustment ×2, missing, by-hand variance and by-hand collinear.
6. **Thumbnails would leak.** The artifact already ships outcome points before the user opens them (419, 497), and only the client records the look. A grid of outcome-view thumbnails would either have to record each one as viewed or hide it behind an explicit "open".
7. **Owner ruling needed under inference.** Explore shows binned outcome means on all analyzed rows and offers a form declaration right beside them. These are not model estimates and do not lock the plan. But FOUNDATION §5 rule 6 forbids outcome-model estimates on the canvas before the lock, so the owner should rule on whether and how these appear.
8. **Seal-blind routes.** An EDA canvas built on `/columns` or `/histogram` would read held-out rows under prediction.
9. **Footprint and Angles do not exist in the engine** (§7).
10. **Smaller items:**
    - legacy EDA functions (`histograms`, `correlations`, `missingness` at turbotab/engine.py:698-748) are not wired into Next;
    - the form lever also bends a column declared linear (INBOX.md:233);
    - selection previews have no latency test (INBOX.md:250).

## Sources
- /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/explore.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/explore_previews.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/consequences.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/coach.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/evidence.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/findings.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/finding_words.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/__init__.py
- /Users/nhedglin/tabular-ml-lab/turbotab/packs.py
- /Users/nhedglin/tabular-ml-lab/turbotab/engine.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/decisions.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/levers.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/models/pipeline.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/__init__.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/evaluation.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/voice.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/plan_lock.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/interview.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/contracts.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/batch.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/data.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/datastore.py
- /Users/nhedglin/tabular-ml-lab/turbotab/server/schemas.py
- /Users/nhedglin/tabular-ml-lab/turbotab/server/service.py
- /Users/nhedglin/tabular-ml-lab/turbotab/server/routes/data.py
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/stage/Stage.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/stage/PreviewGrid.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/stage/LiveScenes.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/record/Findings.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/record/Record.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/record/sentences.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/api/generated.ts
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/screens/ProjectScreen.tsx
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/MODELING_SEQUENCE.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/BLUEPRINT.md
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/INBOX.md