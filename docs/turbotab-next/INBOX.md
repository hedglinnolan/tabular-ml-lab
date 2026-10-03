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
