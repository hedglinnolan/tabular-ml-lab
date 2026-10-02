# M2 contract — the opening sequence for all five lenses, sealed honestly

M2 is done when each of the five lenses can take its characteristic fixture through the full opening
sequence in a browser. Every structural choice previews on the canvas, the seal states its basis and
withholds held-out scores until it is opened once, and the wide omics tables stay fast.

Read first: BLUEPRINT (North star, §0, §11, §11.1), `docs/turbotab/OPENING_SEQUENCE.md` (the
sequence, its copy and its firing rules — authoritative), `docs/turbotab/ROADMAP.md` §"The lockbox
constitution" (§01–§07 — authoritative on what the app may know and when), DRIVE_RUBRIC, and
M1_CONTRACT (what exists). Contract-as-code already on `turbotab-next`: the M2 decision kinds and
slots in `decisions.py` (keyed slots: `findings` holds one disposition per finding id), `SetMissing.
categorical` / `.indicators`, and `Bundle.files` in `graph.py`.

## 1 · The sequence (the Router)

`lens · orientation · [repairs] · target · event · task · purpose · grain · repeat_kind · unit ·
aggregation · temporal · roles · exclusions (eligibility) · missing · split (THE SEAL) ·
energy_adjustment · models · substitution · [open the seal]`

Firing rules come from OPENING_SEQUENCE.md §01 and §03. Nothing is resequenced.

| Question | Fires | Notes |
|---|---|---|
| `orientation` | the lens includes an assay pack **and** the shape reads feature-major (row-mean spread / column-mean spread on a log scale > 4; `turbotab/orientation.py`) | the target question is withheld while open; refused after a target exists |
| repairs | structural findings with repair options exist | not a question: findings in the Record, each with preview-before-apply, before the target question |
| `event` | the task is binary | which level is the event; never guessed |
| `grain` | always | asked; `turbotab/grain.py` suggests and detects contradictions, never decides |
| `repeat_kind` | grain = repeated | usually **stated** (a skip with "Ask me anyway") from date spacing or a visit label |
| `unit` | grain = repeated | no default |
| `aggregation` | unit = unit | the menu is domain-shaped: replicates → mean recommended with its reason; time points → no default (baseline / last / change). When the outcome varies within a unit, also ask which outcome |
| `temporal` | repeat_kind = time_points **and** unit = row | yes → chronological split, grouped too |
| `exclusions` | always | now **eligibility**: asked in scientific terms with the **outcome's distribution withheld**. Pack plausibility rules (dietary energy, clinical bands) stay as proposals |
| `split` | always | **the seal** — §3 |
| open the seal | fit is fresh | a CONSEQUENCE card at the end of the Results, once |

## 2 · The working table

Structural decisions change what the table *is*, so downstream stages read a **working table**, not
the raw file.

```
ingest ─▶ oriented ─▶ findings
              │            │
              └──────▶ working ─▶ profile · roles · target_info · proposals · cohort · …
```

- **`oriented`** (heavy; deps ingest; reads orientation): the raw table, or its transpose when
  feature-major. The transpose is DuckDB/pyarrow and out-of-core, and returns
  `Bundle(files={"table.parquet": …})`. Transposing names the new rows by the sample-identifier
  row.
- **`findings`** now depends on `oriented`. Diagnosis on a turned-around table is garbage
  (OPENING_SEQUENCE §01).
- **`working`** (heavy; deps oriented, findings; reads findings, target, grain, unit, aggregation,
  event) applies, in order: row-local repairs from `state.findings` (SQL column expressions), then
  aggregation (one row per unit, via DuckDB GROUP BY). It writes `table.parquet` plus a
  `row_map.parquet` (working row id → source row ids). That map is how the participant flow and
  provenance stay true.
- **Every M1 stage that read the raw table now reads the working table.** `DataStore` opens over a
  working table's path. When nothing structural is recorded, `working` is a pass-through that
  references the oriented file rather than copying it.
- Row identity: `__row_id` on the working table is dense over its rows. The row map is the bridge.

## 3 · The seal (ROADMAP lockbox constitution §01–§05)

- **The seal states its basis, in three states, never two:** `grouped by <column>` · `repetition
  found but grouping abandoned` · `undetermined`. The basis is persisted in the split artifact and
  rendered on the seal. An undetermined seal is never drawn as a clean lock: it carries an
  exploratory label.
- **Temporal:** a chronological split, grouped too, when `temporal.temporal`.
- **Held-out size tracks n:** the split question's consequence states what a holdout of that size
  can measure. Below a stated row floor, the CV-only option is ordered first with its reason. Never
  refuse.
- **Sealed once, opened once.** The fit computes held-out scores, but the server **withholds** them
  (`holdout: null`, `holdout_sealed: true`) until `open_seal` is recorded. The Results show "Held-out
  rows: sealed" with the open action, which is a CONSEQUENCE card explaining that the scores will
  then be fixed in the record. After opening, any change to an upstream decision still recomputes,
  but its sentence and the manuscript mark it **post-seal** and the Results flag "changed after the
  seal was opened".
- **Decision A:** orientation, grain, unit and aggregation are refused after the seal is drawn
  (409 with an exit naming the re-seal path).
- **What the draw reads** (audit RO-02, 2026-10-02): once rows are held out, a new outcome, a new
  task, a chronological request, and any repair to a column the draw reads (the outcome, the unit
  column, the time column) are refused the same way, with the re-seal exit; an outcome repair also
  offers "exclude those rows", which moves no row. The split question waits for the findings, and a
  finding that would rewrite a column the draw reads is repaired or kept first. No eligibility rule
  may read the outcome (RO-01).
- **Held-out discipline audit:** proposals, findings and evidence that read every row must say so
  in their basis ("across all 21,849 rows"). Anything that informs a modeling choice after the seal
  excludes the sealed rows.

## 4 · Findings with preview-before-apply, deferral, and memory

- **Repair registry** `turbotab/core/repairs.py`: per finding family, options
  `{key, label (≤ 4 words), consequence (≤ 16 words), row_local: bool}` and, for row-local ones, a
  SQL column expression. Families in M2:
  - impossible values → set to missing / exclude rows / mark the column unusable (three routes);
  - sentinel codes (999, −9, 7/8/9 in survey items) → missing;
  - SAS transport zeros (≈ 5.4e-79) → 0;
  - a binary text predictor → which level is 1;
  - energy in kJ → kcal.
  Statistical repairs are recorded and executed in-fold, never on the working table (constitution
  §06).
- **Preview before apply:** `POST /preview` with `apply_repair` shows a table_focus of the changed
  cells plus the affected column's distribution, built on the consequence planner.
- **Defer** resurfaces the finding, pre-checked and attributed, inside the question it targets.
  **Dismiss** is recorded. Findings **learn they were answered**: a finding whose disposition exists,
  or whose routed question holds a matching answer, folds into "answered by #N".
- **Missingness by mechanism** (constitution §07): binary/categorical blanks may become a `Missing`
  level (`categorical: "missing_category"`), which is the honest answer for "not asked" columns like
  `meds_hbp`; numeric columns may add missing indicators. **The outcome's place in the imputation
  model depends on the purpose** (BLUEPRINT §12 ruling 4; audit WP7 superseded the unscoped "never"):
  "The outcome's place in the imputation model depends on the purpose. Under inference, missing predictors are multiply imputed with the outcome and total energy in the imputation model, and the analyses pooled by Rubin's rules (Moons et al. 2006, via Harrell). Under prediction, they are imputed inside each training fold without the outcome, so the fitted pipeline can impute a new row as it was developed (Sisk et al. 2023)." Tier A tests: under prediction the in-fold imputer never sees the outcome or a held-out
  row (`test_repairs.py`); under inference the outcome and total energy are in the imputation model
  and the table is pooled by Rubin's rules (`acceptance/test_wp7_missing_data.py`).

## 5 · Wide data (the M2 benchmark)

`materialize` goes through pyarrow, not a DuckDB projection, at 20k columns. Summaries are computed
from Arrow. Wide tables get smaller row groups. Raise DuckDB's `max_line_size` for > 100k columns.
`/table` takes the visible columns. The roles question gets a search box at > 200 columns.
**Benchmarks, recorded in the result:** 500 × 20,000 (ingest, summaries, roles, a preview, a fit of
elastic net) and 1,000,000 × 30 (ingest, cohort, a preview) on this laptop with
`TURBOTAB_WORKERS=2`. The targets are seconds, not minutes; report the numbers rather than tuning
for a threshold.

## 6 · Voice, teaching and the coach

- Sentences for every M2 kind: empty `OWED_BY_M2_VOICE` in `test_word_budgets.py`. Teaching entries
  for every new question key, within the budgets.
- **Extend the word-budget gate to composed card text**, including proposals notes, so the M1 energy
  card's ~70-word nested-parts note fails the gate. Then fix it: fold the note into the partition
  option's reason and a term card.
- **Coach annotations** (Nolan, 2026-10-01): consequence views gain
  `coach: [{ text (≤ 12 words), anchor: { kind: "column"|"range"|"points"|"step", ref } }]`, at most
  two per view. They are data-grounded ("194 rows below 500 kcal: likely under-reporting"), rendered
  in the coach's amber, pointing at the picture, and never pre-selecting. The decision card gets at
  most one coach line.
- Lens, target and purpose previews stop saying "Nothing… can be shown". The lens preview shows what
  that lens unlocks on this table (the questions it adds, the columns it recognizes, the findings it
  raises).
- Outcome units (mg/dL…) wherever an outcome quantity is shown, read from the column name or the
  pack, or asked once if unknown.

## 7 · Fixtures — one per lens, all through the sequence

| Lens | Fixture | Must exercise |
|---|---|---|
| dietary | `dietary_recalls.csv` | grain repeated (participant_id) → repeats stated → unit = person → aggregation mean, recommended with its reason → energy adjustment on the person-level table |
| clinical | `clinical_longitudinal.csv` | time points → unit = row → temporal yes → chronological grouped seal |
| metabolomics | `metabolomics_untargeted.csv` and a transposed copy | orientation fires only on the transposed copy; feature-major → transposed → diagnosis correct |
| genomics | `genomics_expression.csv` (and a 500 × 20,000 synthetic) | wide path; roles search; elastic net first on the shelf |
| survey | `survey_instrument.csv` / `survey_sentinels.csv` | sentinel codes → missing via preview-before-apply; event level for a binary item |

Plus the real NHANES export: `meds_hbp` as a `Missing` category versus left out; SAS zeros
repaired; the seal opened once at the end.

## 8 · Ownership — workflow 1 (backend + one design prototype)

| Agent | Owns |
|---|---|
| **sequence** | `oriented` and `working` stages and the rewiring of every downstream stage; orientation / event / grain / repeat_kind / unit / aggregation / temporal validators and Router rules; aggregation via DuckDB; the row map |
| **seal** | §3 entire, including the chronological split, the held-out withholding and `open_seal`, post-seal marking, Decision-A refusals, and the basis audit; plus elastic net's grouped inner CV and the ties-the-baseline flag |
| **repairs** | §4 entire, including the repair registry and SQL expressions (applied by `working`), previews, defer/dismiss, answered-folding, and missingness by mechanism in the modeling pipeline |
| **wide** | §5 entire, with the benchmarks |
| **voice** | §6 entire, including the coach annotation model and content, and the lens/target/purpose previews |
| **design** | a `/lab` prototype on real fixtures: the **reshape storyboard** (a person's rows gather, combine and settle into one row; the table collapsing from 600 to 300 rows) and the **seal moment** (the flow forking with its basis named; a sealed Results panel; the open-once card) |

Workflow 2 builds the frontend on top of it: the Record for the new questions, repairs, deferral and
the seal; the stage for reshape, repair previews, coach annotations and the lens preview. Then
integration, reviews and fixes.

## 9 · Acceptance

Every fixture in §7 runs its path in the browser with the canvas previewing each structural choice.
On NHANES, `meds_hbp` can be kept as a `Missing` category. Held-out scores appear only after the seal
is opened, and a later change is marked post-seal. The seal's basis is named on every fixture. The
wide benchmarks are recorded. DRIVE_RUBRIC passes, and so does the extended word-budget gate.

---

# M2 part 2 — the frontend, three reviews, and what part 1 left open

Part 1 is merged (`m2/w1-result.json`): the working table, the sequence, the seal, repairs, the wide
path, voice and coach. The design prototype is at `turbotab/frontend/src/explore/m2/` (route
`/lab/m2`, screens in `m2/explore/`). **Lift its components into production; don't redraw them.**
They are the reshape storyboard, the orientation turn, the seal in four basis states, the sealed
Results with the open-once card, the post-seal band, and the coach notes.

## 10 · The Record (record agent)

- **The new questions, in the Router's order.** Orientation, event, grain, repeat_kind (a stated skip
  with "Ask me anyway"), unit, aggregation (the domain menu, with the recommended option's reason),
  temporal, eligibility, and the seal.
  - **Grain is stated, not asked, when a recognized identifier is unique on every row.** For example:
    "Not asked: every `SEQN` appears once, so each person is one row". "Ask me anyway" reopens it.
    Grain stays a question whenever there is no identifier or the identifier repeats. This is the
    orchestrator's challenge to the old "asked, never inferred" rule, accepted by default.
  - **Eligibility withholds the outcome's distribution.** No outcome histogram appears on its card
    or its previews (constitution §04).
  - **The split card states the held-out-size consequence** and the seal's basis.
  - **Open the seal** is the Router's last step: a CONSEQUENCE card under the Results.
- **Repairs on findings.** Each finding shows its options. Focusing one previews it on the stage;
  then apply, defer (to the question it targets) or dismiss. Deferred findings resurface inside
  their question, pre-checked and attributed. Answered findings fold into "answered by #N".
- **Missing values by mechanism.** Offer the `Missing` level for categorical blanks, recommended
  with its reason when a column reads as "not asked", and a missing-indicator option for numeric
  columns.
- **Wide tables.** The roles question gets a search box above 200 columns, and the working-table
  preview sends only its visible columns.
- At most one coach line per card. Outcome units appear wherever the outcome's values do.

## 11 · The stage (stage agent)

- Lift `/lab/m2`'s views: reshape (gather → combine → settle, the "from rows" provenance column,
  the row flow's "folded in"), orientation turn, the seal (grouped / chronological / abandoned /
  undetermined — never a clean lock), the sealed / opened / post-seal Results.
- Coach annotations rendered on every view kind from `view.coach` (anchors per M2 §6). Amber,
  at most two per view.
- Repair previews: table_focus of the changed cells plus a distribution with labeled marks.
- Lens, target and purpose previews (part 1 serves them).
- **The purpose registry (BLUEPRINT §11.2).** `src/components/stage/purposes.ts` maps every view
  kind, coach anchor kind and Record component to the user question it answers. A vitest test fails
  when a rendered kind or component has no entry. Mirror it in Python
  (`turbotab/core/purposes.py`, every consequence view kind) with a pytest test.

## 12 · Backend: what part 1 left open (backend agent)

1. The Router asks "open the seal" once a fit is fresh.
2. **Out-of-order answers are refused.** A decision for a question still waiting behind an unanswered
   prerequisite gets 409 `not_yet`, with the exit "Answer <question> first". Changing an answered
   question is always allowed. The seal requires grain. An `undetermined` basis comes only from an
   explicit grain answer of "I don't know" (add `unknown` to the grain options), never from skipping.
3. The Decision-A message says the right thing under cross-validation only. The genomics
   split-preview caution gets an exit.
4. The aggregation coach reads the pre-aggregation (oriented) table, so it stays after recording.
5. The grain stated-skip of §10: the Router rule and its voice reason.
6. **Honest cost at scale.** Before a fit, each family carries an `estimate_seconds` (measured as the
   band estimate is). The shelf and the models card state it ("about 5 minutes at 20,000 columns").
   Elastic net's 5-minute fit at 20,000 columns is shown, not hidden; faster screening is M4.

## 13 · Reviews — three in parallel, then one fixer

- **Function:** recompute every number on all five lens journeys plus NHANES. Prove no held-out score
  is visible before the seal opens. Prove repairs equal pandas.
- **Taste:** DRIVE_RUBRIC §5, all 18 questions, against the `/lab/m2` prototype as the bar.
- **Pedagogy (new, BLUEPRINT §11.2):** walk every screen of every journey and list each visible
  element with the user question it answers. Flag orphans, duplicates, and answers to questions the
  user doesn't have yet. Its blockers are fixed before the tag.

## 14 · Acceptance

M2 §9, driven in a real browser: all five lens fixtures and NHANES, with screenshots and frame
strips. The purpose registry and the extended word-budget gate both pass.
