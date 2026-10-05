# Inbox

Ideas, non-blocking defects, and "we should also…" notes land here as one line each, instead of
entering the milestone in flight. Triaged at milestone boundaries: promote to a milestone, or delete.

Format: `- [M?] short description — where it was noticed`

## Triaged at M0 (2026-09-27)

### Promoted into M1's scope
- [M1] Pipeline panel is an inventory, not a pipeline: Rows is one box, Columns is chips by dtype — make it a flow (orchestrator's M0 review)
- [M1] Findings are a wall: ~150-word cards, same-kind cards not paged (two "binary written as text" cards), raw `**` shows — compact card + pager (review, DRIVE_RUBRIC §2.5)
- [M1] Findings have no levers: each finding should route to the question that acts on it (DRIVE_RUBRIC §2.3)
- [M1] Columns grouped by dtype, not by role: SEQN shown as a numeric feature; imputed_* flags not linked to base columns; energy/nutrient/design roles unrecognized (DRIVE_RUBRIC §4)
- [M1] Legacy detection strings render raw ("Target is continuous numeric (40 unique values) - regression.", double periods) — restate in the app's voice
- [M1] Decision sentences are composed in the frontend; DESIGN_LANGUAGE §05.1 says the sentence is a quotation — add a server-authored `sentence` to DecisionRecord
- [M1] set_task binary accepted on a 40-value outcome — refuse with exits (task_mismatch), as the mock already does
- [M1] Energy methods: residual "within sex" (strata) option — the pack's recommended default
- [M1] Pipeline step order: NaN passes through EnergyAdjuster, so imputation precedes it or the model handles NaN
- [M1] Partition records rows where kcal_from_other < 0 (86/600 on dietary_recalls) — show it, don't hide it
- [M1] Density outputs are per kcal; display per 1,000 kcal
- [M1] Pack §04 equivalence holds for Y~N_adj vs Y~N+E and for Y~N_adj+E+C vs Y~N+E+C, not for the table's Y~N_adj+C with C correlated with E — teaching copy must say it precisely
- [M1] Workers spawn eagerly and stay resident (~200 MB each); spawn on demand and retire when idle (default is now capped at min(cores−1, RAM/4 GB, 4))
- [M1] Keyboard focus falls to <body> after a settle; move focus to the arriving question and announce the recorded sentence
- [M1] Settle and arrive overlap for ~250 ms (the shrinking sentence covers the arriving question)
- [M1] FACT questions use the CHOICE silhouette; only the current question carries the teal marker (DESIGN_LANGUAGE §09)
- [M1] Task question lacks "why we ask"; "Keep the recorded answer" shows when nothing was recorded
- [M1] Table preview groups digits in IDs and years (SEQN 9,966; 2,001); fixed column width shows 3 of 17 columns
- [M1] Findings count badge stays outside the stale veil

### M2
- [M2] Wide data: materialize via pyarrow (DuckDB projection of 20k columns ~5 s); stats from Arrow; smaller row groups for wide tables; raise max_line_size for >100k columns; pass visible columns to /table
- [M2] Study-specific missing codes ('.', 999, -9) as a decision kind (clinic_visits.csv)
- [M2] A CSV row with extra fields fails ingest — offer "skip bad rows" as a refusal exit
- [M2] Open-by-path hashes the whole file in the request; use a stat fingerprint, hash inside the ingest job; re-fingerprint on retry; notice a source file that changed on disk
- [M2] Upload chunk writes happen on the event loop — move to a thread
- [M2] Lens union lacks packs.OTHER ("Something else, or not sure") though the refusal copy mentions it
- [M2] Finding shape drops fields preview-before-apply needs (marker, may_preselect, claims, confidence, fix_kind, reframe notes)
- [M2] engine.profile DataWarnings (source "profile") are not yet in the findings stage
- [M2] Datetime columns materialize as datetime64, legacy detectors expected strings — audit detectors keyed on date strings
- [M2] Record has no undo affordance for revert
- [M2] Genomics finding "gene_0017 holds low counts" carries survey-style 999-code consequence text — pack copy review per lens
- [M2] Light stages cancel only cooperatively; anything slow should be heavy=True
- [M2] Engine caches artifact existence in memory; deleting cache/ while running breaks get()
- [M2] Stage errors live only in engine memory (Recent says "not read yet" after restart)
- [M2] Per-project event sequence numbers so the client can order view vs SSE without newest-wins heuristics
- [M2] A fixture with an ambiguous target, so the journey exercises a refusal with exits and the low-confidence task question

### M5
- [M5] Server mode: CSRF protection and an upload size cap, with auth
- [M5] .xls needs xlrd (currently refused with a message)
- [M5] Header wraps to two rows at 390 px; collapse controls below ~480 px
- [M5] /fs/list shows __pycache__ and node_modules

## From M1 workflow 1 (2026-09-27) — untriaged

- [rows] Merge: regenerate turbotab/server/openapi.json (python -m turbotab.server.openapi --write) and the frontend's generated.ts (npm run gen:api) after combining the rows, modeling and voice schema changes; both files will conflict.
- [rows] Merge: service.preview imports turbotab.core.row_previews to register the row builders. The modeling agent's builder module needs the same import there, or a list of builder modules in consequences.
- [rows] Merge: point decisions.model_families() at the modeling agent's real registry API and drop the guessing.
- [rows] set_target, set_task and set_purpose have no preview yet. A register_transform for set_target (drop rows whose new outcome is missing; the column leaves the predictors) would give the outcome question previews through the generic diff for free.
- [rows] RowStep.decision_id is always null because a stage cannot see record ids. The server could fill it from the log when serving the cohort artifact or view.
- [rows] Changing the outcome (or the task, which changes stratification) still draws a new held-out set, because the measured universe depends on the outcome. The UI should say so before it happens.
- [rows] On very large tables, row previews materialize the rule, target and gappy-predictor columns for the whole pool. Above about 1M rows, switch to DuckDB counts. The before-frame for 20,000-column tables (1,000 rows) is untested for speed; only diff_views is benchmarked.
- [rows] The frontend mock (mocks/db.ts) returns interview: []; the frontend workflow should mirror the Router or proxy it.
- [rows] The skipped-task reason passes target_info.reason through verbatim ('Target is object type (categorical/binary) It has 2 distinct values...'). It needs the voice agent's restatement.
- [modeling] [M1] The elastic net's inner CV is not grouped by participant when rows repeat (the outer folds are). Group it, e.g. with a wrapper that maps the row-id index to groups.
- [modeling] [M1] The partition method's Atwater check runs per fit, so on some tables it refuses in some CV folds but not others. Settle the unit verdict once on all training rows at design and pass it into every fold.
- [modeling] [M1] On NHANES, partition with protein/carb/fat_total gives 12,031 of 17,078 training rows a negative kcal_from_other. The design says so; the energy question should show it in its teaching too.
- [modeling] [M1] Nested nutrients (fat_sat/fat_mon/fat_poly inside fat_total, sugar inside carb) make a substitution that moves one part with the parent held fixed incoherent. The design warns; set_substitution should warn or refuse such pairs.
- [modeling] [M1] nutrient_role does not read sugar as carbohydrate, so sugar has no energy factor and is left out of substitution pairs and the partition.
- [modeling] [M2] Penalized logistic regression (saga) is slow at NHANES scale even with the smaller grid: about 10 s per binary fit and 35 s per multiclass fit at 17,000 rows. Consider an L2-only lbfgs option or fewer outer refits.
- [modeling] [M4] Store out-of-fold predictions in the fit Bundle (calibration, comparison deck, residual plots).
- [modeling] [M4] Once fits exist, the select_models preview could add a metric view with each family's cached CV score.
- [modeling] [M4] Multiclass substitution curves, one per class.
- [modeling] [M1] Coefficient tables include the intercept row '(intercept)'; the frontend should decide whether forest plots skip it.
- [voice] rows agent / merger: the cohort's exclusion steps should call turbotab.core.stages.proposals.rule_excludes. It treats levels 1.0 and '1' as the same, never excludes a missing value, keeps the bounds themselves, and judges unlisted levels by low/high. That keeps the proposals counts, the decision sentence and the Rows panel on one number.
- [voice] merger: the roles-stage recognizers (rows agent) and the identifier/flag/design/cycle detectors in finding_words.py should share one recognizer module, so a proposed role and a finding can never disagree about SEQN or imputed_*.
- [voice] rows agent: when recording a decision, pass voice.sentence_for a ctx with datastore, columns, n_rows, records, detected_task, repeats and n_cohort, so sentences carry their numbers. Every key is optional.
- [voice] frontend: render finding.summary with a lever button that opens routes_to, page same-group findings as one card, and build the teaching layers and drawer from GET /api/teaching. Lens, target, task, purpose and revert can now use DecisionRecord.sentence instead of client-side copies.
- [voice] Domain catch on the NHANES export: values like 5.397605e-79 in kcal, bp_di and others are how SAS transport files write an exact zero. They deserve a finding of their own. Today they show up only as 'impossible' kcal values.
- [voice] Clinical lens: offer the clinical pack's plausibility bands (the impossible_vs_extreme params) as exclusion proposals, so that finding's lever lands on a pre-counted rule.
- [voice] Goldberg EI:BMR screen (NUTRITION_PACK §02, the field standard for misreporting) as a proposal. It needs weight, age, sex and a named BMR equation.
- [voice] A set-to-missing lever for impossible values and sentinel codes. The clinical pack prefers missing over excluding the row; M1 has only exclusion.
- [voice] modeling agent: partition and substitution over fat_total together with fat_sat, fat_mon and fat_poly count fat's energy twice. The proposals note says so; the partition applicability check could refuse it.
- [voice] The word-budget gate adds about 13 s to the fast suite, because it ingests and diagnoses all 33 samples. If the suite drifts past a minute, cache the per-fixture findings to disk keyed by file hash.
- [server] NHANES complete-case trap: meds_hbp has 15,552 blanks and meds_chol 17,204 (the questions were asked only of people told they had hypertension or high cholesterol). Complete cases therefore keeps 2,943 of 21,348 rows, a subgroup with mean glucose 126 against 107.6 overall. The finding routes to `missing`, but no lever offers 'a blank means not asked: recode as No'. M3 missingness routing should offer it, and the set_missing preview could name the likely structural cause.
- [server] The shelf ranks boosted_trees first with fit 'good' and no concerns, yet on this cohort its CV R² is -0.04 and holdout R² -0.06, worse than predicting the mean. Fit concerns should say when a family does worse than the mean baseline. HistGradientBoosting could use early stopping or a smaller learning rate when n is about 2k.
- [server] The substitution band for the linear model has zero width: a bootstrap over rows with a fixed linear model gives every row the same delta. The note says model uncertainty is excluded, but a zero-width band reads as certainty. Use the coefficients' covariance (delta method) or a refit bootstrap, or say plainly that the band is omitted.
- [server] The energy reading proposes fat_sat, fat_mon and fat_poly beside fat_total, and warns that they are parts of it. The fat_total → carb substitution then holds the parts fixed, which the design flags as incoherent. Offer an exit or option that drops the parts when their total is present, without pre-selecting it.
- [server] SentenceFacts counts the cohort flow under the decision-log lock. On wide omics tables with complete_case this could hold the lock for seconds. Compute before append if it shows up in the M2 benchmarks.
- [server] The set_exclusions sentence repeats the range: "outside `500`–`5000` were excluded as implausible intakes (sex-neutral 500–5,000 kcal a day)". The proposal's rule reason could drop the numbers.
- [server] Relayed-request note on presenting modeling changes: on NHANES the general approach held for all six kinds with nothing written per option. That approach is a closed set of views chosen by measured change, with each model family declaring its lineage and steps. Worst p95 was 32 ms. Keep it, rather than a picture per option.

## From M1 workflow 2 (2026-10-01)

### Orchestrator's picks for the start of M2
- [M2-first] The energy card's nested-parts note runs ~70 words above the options — the wall-of-text problem Nolan cares most about; fold it into the partition option's reason and a term card (word-budget gate should have caught it: extend the gate to cover composed card text, not only teaching entries)
- [M2] Coach annotations on the canvas (Nolan, 2026-10-01): ≤ 2 amber notes pointing at the picture, plus one data-grounded line on the decision card — never pre-selecting
- [M4] Architecture lane beside the data lane on the stage (Nolan: "a multi-view canvas that shows architectural changes and examples of data layer changes"): linear equation, tree splits, elastic-net shrinkage, morphing with the data lane
- [M2] Outcome units (mg/dL) on Δ predicted glucose and on coefficients (taste reviewer)
- [M2] The holdout is re-scored and shown after every refit, so a user can tune against it — enforce 'sealed once' at the seal (function reviewer)
- [M2] Flag NHANES SAS 5.4e-79 'zeros' (bp_di, fat_total minimum) as an import finding (taste reviewer)
- [M4] Overlay the previous method's curve after a method change — the sensitivity analysis the why? text recommends (taste reviewer)

### Review minors and agent notes (untriaged)
- [review minor] Elastic net tunes its penalty on inner folds that split a participant's rows — fix: In fit_stage, pass group-aware inner splits (GroupKFold over the fit rows' groups) to the estimator's cv when the split is grouped.
- [review minor] A model that only ties the baseline is not flagged — fix: Flag ≤ baseline within a small tolerance: 'Predicts no better than the class prior'.
- [review minor] Findings keep pushing levers that were already pulled — fix: Mark a finding as addressed, and fold it away, once its routed question holds an answer matching its lever.
- [review minor] A save started mid-flip exports a different step from the one the menu names — fix: Freeze the step when the menu opens, and caption a step with its own label.
- [review minor] Wording and accessibility nits — fix: Fix the grammar. Name only the macronutrients present. Show a partition that cannot run as a refusal. Base the shelf on the training n. Move focus into the role menu and back to the chip. Cap the wide forest plot at abou
- [review minor] Switching options after a 'still' option replays the storyboard, and every switch waits ~150 ms on the old title — fix: Hold 'with this choice' after a still option too, and land directly on the new result. Shorten or skip the debounce when the preview is cached, and morph within a unit over the full 150–300 ms.
- [review minor] Settle and scene crossfades show blank or garbled frames — fix: Fade the question body out while the card's height animates (or morph the heading into the sentence). Use mode='wait' or offset positioning for text-only empty states. Fade the trend line with pos rather than toggling it
- [review minor] Decision sentences are not fully reproducible or publishable — fix: List every column in the stored sentence (truncate only visually, with a disclosure). Write 'women outside 500–3,500 kcal/day and men outside 800–4,200 kcal/day (Willett's sex-specific cut-offs)', and carry the reason fo
- [review minor] 'Your data now' Results card lists only three of the questions the fit waits on — fix: Say 'It waits on 7 answers, starting with purpose', or list them all, and drop the duplicate clause.
- [review minor] Exclusion and energy previews lost the prototype's data-specific detail — fix: Restore the per-sex under/over counts under the cut row (a methods section needs them) and the one-liner that cites this table's own range.
- [review minor] The energy question exceeds the word budget, and new-question headings get an input-like focus box — fix: Fold the nested-parts note into the partition option's reason, or a term card. Give programmatic heading focus no visible outline (tabIndex -1 headings) and mark 'now' with the existing keyline.
- [review minor] The uncertainty band is muddy and its time estimate is off — fix: Draw bands as a dash-matched outline or an interval at a few k values, or show one family's band at a time with a legend toggle. Measure the estimate with the actual worker count. Veil only the curves card while the band
- [review minor] Copy and data wording slips in the missing-values preview — fix: Fix the grammar and call the meds columns 'yes/no'. Format values with each column's own precision; flag the 5.4e-79 zero for the inbox.
- [review minor] Evidence views carry wrong or truncated labels — fix: Treat evidence as still: no flip, labeled 'Your data as loaded'. Let titles wrap to two lines, and wrap the readout under the flip on narrow stages.
- [review minor] Substitution captions undercut the curve for a reviewer — fix: Caption the curve with its own estimand (energy moved from fat_total to carb at fixed kcal, through inputs adjusted by the recorded method). Define k inline ('k = kcal moved'), give the outcome's unit, and hatch excluded
- [review minor] Saved figures and thumbnail views have small truth and legibility gaps — fix: Title by the saved state, pad the axis domain, and add 'n of N drawn' to the caption. Show the clip note in thumbnails, or collapse untouched lineage rows in compact views.
- [review minor] The propagation sweep is barely visible, and the banner lags a recorded method — fix: Hold each downstream segment's veil for at least one 150 ms propagate step, in order. Show the recorded energy method in the Columns segment from the decision itself, before the design stage runs.
- [review minor] Internals and dev affordances show in the product — fix: Hide /lab behind dev builds, show the baseline as '≈ 0.000', and word the survey finding's lever as 'Tell TurboTab this sample has no design columns' or point it to the drawer.
- Lineage provenance: show the columns the missing-values answer left out as raw nodes that end 'left out (mostly blank)'. Today they simply do not appear.
- Let the user dispute a detected nesting. `nested_in` is derived from the data, not recorded as a decision.
- Band precision: for root-n families (linear, elastic net), scale the 2,000-row refit band to the full training size, or raise the row cap for linear, which costs about 0.6 s per 50 refits.
- Boosted-trees refits on 2,000 rows lose sklearn's automatic early stopping, which only switches on above 10,000 rows. The refits are therefore a slightly different model from the full fit.
- The proposals stage could read the exclusions slot, so the leave-out offer's count matches the complete-case step after exclusions (18,405 rather than 18,853 on NHANES).
- Categorical evidence: DistributionView has no level-count form. The prototype used a `levels` extension, which would suit binary-text findings such as `gender` and the meds columns.
- Stage agent: add schema.ts aliases for the new types (PreviewResult, views, frames, Baseline, MissingReading) and a client.ts function for the evidence route.
- The evidence route answers 409 `findings_not_ready` while findings are stale after a lens change. The frontend should veil the evidence rather than call the route.
- Band cost on NHANES with 50 refits is about 25 s, about 80% of it boosted trees. Consider fewer refits for boosted trees, or a parallel refit pool.
- [M1] No event carries the interview, so the client refetches GET /projects/{pid} on every stage status change. A server-side interview event, or the interview attached to stage events, would cut this to one push per change.
- [M1] When an earlier answer changes while a later CHOICE is open (e.g. models), that question goes waiting while its stage recomputes, then remounts, and the unrecorded selection is lost. Keep drafts by question key in the Record.
- [M1] A lever routed to a question that is still waiting only highlights it in the 'Then' list ('asked after …'). Decide whether a lever may open a question out of order (the Router would have to allow it).
- [M1] Real roles sentence: 'Column roles were set for 28 columns: …' runs six lines when the proposal was confirmed unchanged. A short sentence ('The proposed roles of all 28 columns were confirmed') would read better.
- [M1] The proposals.energy notes about nested parts are long (two sentences per parent). One sentence covering every parent would fit the data line better.
- [M2] The roles question on very wide tables shows 14 chips per group, then a scrolling list of all of them. A search box inside the roles question would help on 20,000-column assays.
- [M2] DESIGN_LANGUAGE §09 says to log the 'Ask me anyway' click rate as trust telemetry. It is not logged.
- [M1] TablePreview and MiniHistogram (src/components/pipeline) are unused since the panel was retired. Delete them, or reuse them on the stage's 'your data now'.
- [M1] An accessibility review should confirm the pattern of aria-disabled options that answer a press with a refusal.
- [M5] The header wraps to three rows at 390 px (existing item). The banner now flows below it on narrow screens instead of sticking.
- [chore] The explore/stage prototype files are not Prettier-formatted (npx prettier --check src flags them).
- Backend captions print '-0.00' for the residual correlation. The stage tidies it on display, but the server should print 0.00.
- Backend: the substitution note still describes the row-resampling band ('200 resamples … with the fitted model held fixed'), which §12.7 removes. The note should follow the new refit band.
- Backend: agree on the field that carries the measured band time and the refit count. The stage reads SubstitutionArtifact.band_seconds and .n_boot.
- Backend: RelationshipFrame has no y label. A step's values (e.g. residuals) are not a column, so the stage labels the axis with the step's own label. Consider adding `y_label` to frames.
- Evidence for an imputation flag should sample rows where the flag is true. The first 8 rows of NHANES are all False.
- Decide whether the all-nutrient partition preview, which comes back with an empty after-cloud and a units caveat, should instead be a 409 refusal with alternatives, as §4 says for refusable decisions.
- Fit: state the scale of each linear family's coefficients (per unit or per SD) so the forest's axis can name its unit.
- The refit band for the linear family is very wide (±64 mg/dL at 500 kcal) because the fat subtypes and carbohydrate are nested and collinear. Re-check once the nested-nutrient fix (§12.5) lands.
- Integrator: wire src/state/focus.tsx (record agent) to <Stage pid view focus onFocus/>. Retire /lab/stage or keep it as a review surface.
- frontend/CLAUDE.md's layout section should list src/components/stage and src/screens/StageLabScreen.
- Not built: the `inline` prototype's compare pin, a metric switcher (RMSE/MAE) on the model comparison, and journal-style export of the Results' coefficient table as LaTeX (booktabs).
- Held-out discipline: proposals.energy.r_with_energy and the dietary energy finding ('fat_total correlates 0.88 with kcal') read every row, including rows the split later seals. The preview reads rows not held out (0.83), and the finding's evidence reads a 5,000-row sample of rows not held out (0.87)
- The lens, target and purpose previews have no views ('Nothing about this choice can be shown on your data yet'). A lens preview could show what the lens adds: the questions it opens, the pack checks, the energy question.
- The energy_adjustment strata_candidates come from proposals, which do not read the missing-values answer. The frontend now filters out left-out columns. Proposals, or the validator's exits, should do this at the source.
- On the models question, the stage's 'Record this choice' button records only the previewed family (for example select_models [boosted_trees]), not the multi-select in progress. Consider hiding it for multi-select questions, or recording the selection.
- Partition over all six nutrients (with nested fat subtypes) previews a plot that does not change, with a note that fat_sat's unit cannot be confirmed. The Record lists partition as applicable. applicability.partition should account for nested parts and unconfirmed units.
- Recapture src/mocks/m1-stage-fixture.json with the §12 backend: real storyboards, marks, band, evidence and the leave-out preview. Then delete the mock's emulation code.
- Table-focus cells print e.g. '80.0000' for bp_di (cellFormatter picks 4 digits from a target value below 1). Use per-column precision.
- The /lab/stage review route and its 627 KB fixture chunk ship in production builds, lazily loaded. Gate it to dev:mock.
- In a totals-only partition, the nested parts (sugar, fat_sat…) stay in the model as raw grams beside the kcal_from_* columns. Should the partition offer to drop or carry the parts too?
- I could not produce a design failure through the UI on NHANES once partition was refused at the question. To see the new failure display I recorded models over the API without a missing-values answer. A mock-server scenario with a stage error would let a spec exercise StageFailure without reaching f
- The exit animation (~120 ms) still overlaps text when two scenes meet. A shorter exit, or mode='wait' between different groups, may read cleaner — a taste call for the orchestrator.
- Many frontend files were already failing a Prettier check before this change. The repo does not enforce Prettier (npm run check does not run it), so one formatting pass would clean that up.
- The evidence card label still says 'your data as loaded', but energy evidence now reads a training sample (values as loaded, rows outside the seal). Consider 'your data as loaded, training rows'.
- [presentation, BLUEPRINT §14] The roles card records only a bulk `set_roles`; the server now records the proposals below high as `unconfirmed` and settles each only by its own `confirm_role`. The card should show `needs_confirmation`, give each attention chip a "Confirm" press that records `confirm_role {column, role}` (one column per record), and never offer a confirm-all for them. Until then a user settles a role through the refusal exits the energy card, the screens and the survey question raise.
- [presentation, BLUEPRINT §14] The energy card's reading now carries `unconfirmed` (the energy column and the energy-bearing exposures it does not pre-fill) and `not_adjusted` says "proposed below high confidence; confirm its role to adjust it". Survey options carry `needs_confirmation`. Neither is shown yet.
- [recognizers, BLUEPRINT §14] Recall misses on the recognition corpus (all safe: proposed low, never a default): ASA24's `UserName`, `RecallNo`, text `IntakeStartDateTime`, `SODI`, IDATA `tns_kcal_asa24`; CDISC `AVISITN`, `AVISIT`, `ADY`; UK Biobank `p…_i0` field codes. A codebook table for ASA24 and UK Biobank field ids would raise recall; precision is what the leash protects.
- [v2.x] Compositional (ilr) models for share reallocation; v2 refuses that request with its reason and offers the kcal substitution — MODELING_SEQUENCE §0 ruling 11
- [v2.x] Latent-variable (SEM) attenuation correction for reflective scales — MODELING_SEQUENCE §0 ruling 8
- [v2.x] Full two-level FCS imputation; v2 imputes time-invariant variables per unit and includes cluster means — MODELING_SEQUENCE §0 ruling 12
- [M3.5] Quick vs Advanced: Quick states steps 8–10 and the default rungs, and asks steps 2, 3 and 5 under inference — MODELING_SEQUENCE §8
- [integration WP16–18] WP18 refuses the outcome's scale and the reference-rows exclusion whenever a split is recorded ("start a new analysis"). WP16 now has a re-seal path (withdraw the split, change it, draw again), which these refusals could offer. Under inference with nothing held out, changing the scale re-draws nothing, so the refusal is tighter than the seal needs.
- [integration WP16–18, presentation] The ask card carries its consumer's "read from your data" items only when it asks something. A consumer with nothing to ask shows no card (BLUEPRINT §14.2), so its settled readings are visible only through GET /readings and the methods record. When presentation resumes, render that endpoint's section too, not only the card. The four WP17 cards, WP18's task follow-ups and the ask card are served but not rendered yet.
- [integration WP16–18] The ask card's text for a value below a detection limit or for an ambiguous comma comes from `readings.ask_text` (one guess per reading). The refusal's own sentence says it better. Consider giving these readings their own words on the card.
- [tests] Under load, server/tests/test_evidence.py::test_a_by_sex_rule_marks_each_level sometimes gets 409 not_yet from the preview: the preview is posted before the stage it reads is fresh. It passes alone. The helper should wait for the stage, as `prepare` does.
- [M3.5] The usual-intake estimand is offered in the `usual_intake` stage's artifact (`offer`); the Router does not ask it yet. Ask it at the exposure-and-estimand step once that step exists — wave 1 NCI, turbotab/core/usual_intake.py
- [M3.5] The `usual_intake` slot holds one analysis per dietary component, so a whole-population and a consumers-only distribution of the same food cannot sit side by side; key it by (component, population) if both are wanted — wave 1 NCI
- [v2.x] NCI usual intake with person-level covariates (age, sex) and subgroup distributions (DISTRIB's `subgroup`); v2 models the nuisance covariates only — wave 1 NCI, turbotab/core/methods/usual_intake.py
- [v2.x] NCI never-consumers (Kipnis et al. 2009's third part) and the multivariate NCI method (ratios, several components jointly, INDIVINT-style predictions for calibrating a two-part exposure) — wave 1 NCI
- [presentation] The usual-intake card: NUTRITION_PACK §03's shrinkage figure (single day, mean of days, usual intake); the artifact already carries `day_one` and `mean_of_days` percentiles beside the distribution — wave 1 NCI
- [M3.5] The estimand card names columns only, and it is asked before `set_scales` (not yet a Router question): a declared scale's score cannot be named as the exposure, and the adjustment card asks the scale's items one by one. When the domain transforms are routed after the estimand, name the score there and ask its causal answers once — wave 1 integration, turbotab/core/estimand.py
- [M3.5] The usual-intake stage is not one of the estimate stages: its distribution is its own estimand, not an outcome-model estimate, so it neither waits for nor locks the inference plan. Decide whether the plan's lock covers it — wave 1 integration, turbotab/core/estimand.py ESTIMATE_STAGES
- [presentation] The working table keeps the drift-corrected pooled QCs beside it (`qc_rows.parquet`) once they leave as reference rows; nothing serves them yet. They are what the QC-RSD figure before and after correction reads — wave 1 integration, turbotab/core/stages/working.py
- [presentation] The pooled-QC finding offers QC-RLSC's four options even when the run cannot support them (fewer than five QCs in a batch, study injections outside a batch's QCs); each says "Cannot run" and its refusal's exit is the exclusion. Consider showing the exclusion first when every QC-RLSC option says it cannot run — wave 1 integration, turbotab/core/reference_rows.py
- [tests] Under the full acceptance suite with the machine loaded (load average 20), acceptance/test_methods_gate.py::test_b_previews_under_inference_read_every_analyzed_row once got the names-and-summaries basis (its preview read no rows); it passes alone and with its file under -n 2 at the same load. The preview, like test_evidence's, should wait for the stage it reads — wave 1 integration
- [routing repair, presentation] The follow-up question now asks every yes/no outcome under the clinical or dietary lens (or none) and serves `target_info.follow_up_options` (the columns it may name, those read as a follow-up time first); a time to event's follow-up also takes a landmark and a horizon (`set_follow_up`). None is rendered yet; nor are the estimand card's `unconfirmed` exposures or the adjustment card's `direct_questions`.
- [routing repair] The energy question's ask card does not list the energy sources' units: they are asked only when a partition method is chosen (the refusal), since no other method reads them. When presentation resumes, the energy card could show each source's settled kcal per unit beside the partition options.
- [ESTIMAND, v2] Marginal standardization under multiple imputation is refused with its reason (the bootstrap would have to repeat the imputation); build MI-then-bootstrap (Schomaker & Heumann 2018) or a delta-method variance pooled by Rubin's rules — stages/effects.py
- [ESTIMAND, v2] Marginal standardization under a surveyed-population answer is blocked and recorded (exit: the sample-only answer); a design-based standardization (weighted g-computation with a replicate or linearized variance) would lift it — MODELING_SEQUENCE §4, "Population estimand without a design-based estimator"
- [ESTIMAND, v2] Under the all-components model a substitution's marginal standardization (moving several sources by their shares) is refused; the conditional table's average relative effect carries the substitution in every model that holds the other sources — stages/effects.py
- [ESTIMAND, v2] "Per what unit" (MODELING_SEQUENCE §1 row 2) is not yet a field of the estimand: every measure is per one unit of the exposure as recorded, and the marginal contrast is "one unit higher than observed" — estimand.py
- [ESTIMAND, v2] A time-to-event outcome's marginal contrast (standardized survival or RMST at a declared horizon) is named and refused; STROBE 16c's absolute risk would need it — estimand.NOT_FITTED
- [ESTIMAND → row 7] The Table 2 rule never shows a declared modifier's main effect as an effect (`effects.split_rows`), but effect modification is not yet a declared object; when it is, its product terms need rows of their own (the exposure within each stratum, RERI beside) rather than the appendix — MODELING_SEQUENCE §1 row 7
- [ESTIMAND] The sequence is refit for least-squares, logistic, proportional-odds, Cox and feature-wise models; a model of the unit (mixed, GEE) shows its primary only, with the reason — stages/effects.py
- [ESTIMAND, presentation] The effects artifact (Table 2 with its appendix, the marginal risks, the diagnostics with their exits, sensitivity), the estimand card's measure labels and ranks, the family's multiplicity question and the Model 1 card are served but not rendered yet — presentation paused
- [tests] acceptance/test_methods_gate.py::test_b_previews_under_inference_read_every_analyzed_row fails about 1 run in 4 on the base commit b883cde as well: the energy preview is posted once its question opens, which needs only the proposals, while the cohort and split are still recomputing after the adjustment answers, so the preview reads no rows ("Read from the column names and summaries"). The test should wait for the cohort, as the helpers that call prepare do — found while running the ESTIMAND package
- [readings ledger, race] decisions.nesting_of reads the design (else roles) artifact and returns no pairs when neither is fresh, so _substitution_reads_settled_readings accepts a substitution whose nested_in reading is unsettled while the design is recomputing (acceptance/test_ledger_repair_3.py::test_a3 got 200 for 409 once under -n 2 load; it passes alone, here and on b883cde). A missing artifact should hold the decision (not_yet) or read the latest design, never check nothing; extra heavy stages (ESTIMAND adds effects) make the window likelier — found while running the ESTIMAND package
- [wave 2a integration, ESTIMAND × routing gate] A direct effect may be declared with a marginal measure: g-computation then averages the predicted risks over the mediators' observed values, which is neither the controlled direct effect at a fixed mediator level nor a natural direct effect, while the caption's interaction line ("the direct effect at its reference level only") speaks of the conditional coefficient. Decide whether a marginal measure of a direct effect is refused with the conditional odds ratio as its exit, or named as the controlled direct effect averaged over the observed mediator distribution — estimand.caption, stages/effects.py
- [wave 2a integration, ESTIMAND × SCALES] The estimand card cannot name a scale's score as the exposure (wave 1 note above), so under ESTIMAND's Table 2 display the score's uncorrected coefficient is served among the adjustment terms while the scales stage shows its regression-calibrated correction as an estimate. When the score can be the exposure, offer the correction for the exposure (and a family member) only, or label it as an adjustment term's — stages/scales.py
- [v2.x, causal lane] Pool DML and TMLE estimates over multiple imputations (Rubin's rules per split, then the median); v2 blocks and records incomplete rows under the MI answer and offers the complete rows — turbotab/core/causal.py, relation `mi_blocks`
- [v2.x, causal lane] A Super Learner (cross-validated stacking) as the nuisance learner, the TMLE literature's default; v2 offers main-terms, the cross-validated lasso, random forests and boosted trees — turbotab/core/models/causal.py
- [v2.x, causal lane] A survey-weighted plug-in penalty for post-double selection; v2 blocks and records it under the surveyed-population answer, with the weighted partially linear model as the exit — relation `survey_blocks_pds`
- [presentation, causal lane] The causal question's card (`causal_design`: the options with both labels, the four assumptions each with its diagnostic, the overlap histogram, the effective sample sizes) and the `causal` artifact are served but not rendered yet
- [wave 2a integration, CAUSAL × ESTIMAND] The causal lane's required sensitivity analysis is ESTIMAND's (`causal.sensitivity_for` → `models.effects.unmeasured_confounding`), but double/debiased machine learning on a yes/no outcome reports a risk difference alone, so it has no E-value (no risks to form a ratio from) and no robustness value (that is one least-squares coefficient's); the artifact and the methods text say the required analysis could not be computed. The interactive model's AIPW means (μ₁, μ₀ from the same cross-fitted nuisances) would give a marginal risk ratio and its E-value; Chernozhukov et al. 2022 ("Long story short") generalize the robustness value to DML — turbotab/core/models/causal.py
- [M3.5] With no holdout, MODELING_SEQUENCE §1 row 12 (b) lets a family declared before any score was seen report its own corrected score. MS6 refuses dropping compared families, but nothing records such a pre-declaration yet, so every no-holdout result is the selection-corrected estimate — wave 1b VALID, turbotab/core/models/selection.py
- [v2.x] Time-to-event prediction with delayed entry: the Brier score at the horizon needs truncation weights, so MS6 falls back to Harrell's C (semi-proper) and says so — wave 1b VALID, turbotab/core/models/metrics.py
- [M3.5] SMC-FCS with a product term is tested on the sampler and the D1 pooling directly; the engine declares no product terms until effect modification exists, so no run reaches it end to end yet — wave 1b MI, turbotab/core/methods/smcfcs.py
- [presentation] Under the population answer with multiple imputation, the substitution band is each copy's design-based band pooled by Rubin's rules (`band.method = "design"`, `pooled = "per_k"` or `"contrast"`); the curve card should say both, as the caption does — wave 1b integration, turbotab/core/stages/modeling.py
- [v2.x] The probability approach to the prevalence of inadequacy (Institute of Medicine 2000, ch. 4), for iron in menstruating women, whose requirement distribution is skewed. v2 refuses an EAR for a component whose name reads as iron until the answer says no participant is a menstruating woman, and offers a plain share or no cut-off — repair NCI, turbotab/core/usual_intake.py
- [v2.x] A prevalence of inadequacy per DRI life-stage group (each group's own EAR against its own usual-intake distribution, combined by weight). v2 asks whether one EAR is every participant's group's EAR. Unanswered, it reports a plain share and blocks the prevalence label — repair NCI, turbotab/core/stages/usual_intake.py
- [presentation] A usual-intake analysis now carries `exits` (refusals and the blocked EAR label), and `set_usual_intake` carries `ear_for_all` and `ear_symmetric`. The card should show the exits as presses and ask the EAR's group when an EAR is chosen — repair NCI
- [wave-1 repairs integration, SURVEY, owner ruling] Under the surveyed-population answer the cross-validated scores, their calibration and the family comparisons stay labeled unweighted, as the fitting procedure's on these rows (relation `unweighted_scores`); REPAIR-SURVEY left it for the owner's ruling whether they should be design-weighted instead — turbotab/core/models/survey.py, stages/modeling.py
- [v2.x, LEASH] A medication that treats the outcome (statins beside LDL, antihypertensives beside blood pressure, glucose-lowering drugs beside glucose) is guessed the with-and-without pair, and the card says that Tobin et al. (2005, Stat Med 24:2911) call both analyses flawed; build their remedies (a sensible constant added to treated values, or censored normal regression) as a declared secondary — turbotab/core/covariate_guesses.py
- [M3.5, LEASH × ruling 13] A withheld inference fit now serves no cross-validated score; once the plan is answered the scores are served as before (other packages' tests read them), while MODELING_SEQUENCE §0 ruling 13 says no cross-validated score is shown under inference at all. Decide whether they leave the served fit, or stay labeled as the outcome model's fit statistics — turbotab/core/estimand.py withhold
- [presentation, LEASH] The grouping question's card (`proposals.grouping`: each column with its guess and evidence) and the adjustment card's block members (class, reason, source) and its multi-select answer (`bulk`) are served but not rendered
- [LEASH] The grouping question is widened by structure under inference only; under prediction it is still asked of named groupings alone (internal–external validation). Decide whether a structural grouping should also offer grouped folds under prediction — turbotab/core/groupings.py
- [LEASH] A whole-number grouping with no grouping name is told from a measurement by its value profile (counts falling away from the median, Spearman ρ ≤ −0.5) unless the packs read a measured quantity in its name; a uniform integer measurement (day of month) is asked with the guess "groups", and a code list whose largest groups sit at its median could be read as a measurement — accepted limitations, measured on the recognition corpus when it grows
- [LEASH] Values above an upper quantitation limit (`>1500`) and TNTC have no repair; a right-censoring repair mirroring the below-detection one would let the lab pack's censored-values finding point at a lever for them too — turbotab/core/repairs.py
