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
