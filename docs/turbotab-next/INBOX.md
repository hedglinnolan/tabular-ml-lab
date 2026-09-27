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
