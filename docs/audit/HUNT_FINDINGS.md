# New findings from the exhaustive hunt

Run `wf_70254f26-494`. Four of ten lenses completed before the session limit —
join key semantics, cardinality/fan-out, join modes, and stacking. The JSON,
Import Doctor, combine-UI and state/scale lenses did not run.

These are UNVERIFIED. The adversarial verification stage never executed, and a
scripting error in the workflow (inner `parallel()` given promises rather than
thunks) would have broken it anyway. Every one needs independent reproduction
before it is treated as real — several may be false alarms, and some describe
behaviour this branch has since changed.

| Severity | Area | ID | Symptom |
|----------|------|----|---------|
| critical | combine_ui | `chain-promise-vs-delivery` | demographics (100 subjects) + labs (2 rows/subject) + diet (2 rows/subject). The last row-count promise printed above th |
| critical | join | `repeated-measures-key-rejected` | Researcher uploads demographics (200 subjects, 1 row each) and visits (the same 200 subjects × 3 visits = 600 rows) — th |
| critical | join | `casefold-decided-per-column` | Adding a single duplicate row whose ID is typed in the wrong case collapses a perfect 10-of-10 link to 1 row, and the on |
| critical | join | `float64-id-collapse-mislabelled-repeated-visits` | Six participants with 17-digit IDs join to their labs; one participant is shown three times carrying three different par |
| critical | join | `outer-right-join-drops-right-key-column` | "Keep everyone from every file" delivers exactly the promised 250 rows, but 50 of them have a blank participant ID — onl |
| critical | join | `slug-truncation-suffix-collision` | Two files whose names share their first 20 characters produce a working table with two columns literally named `bmi_nutr |
| critical | join | `min-uniqueness-hides-repeated-measures-key` | Demographics (100 rows, SEQN 1..100) and diet recalls (400 rows, the same SEQN repeated four times) — an identical, 100% |
| critical | join | `decoy-key-passes-index-guard` | Two unrelated files (a cohort with a row counter + age, and an unrelated table with a row counter + gdp) are offered a s |
| critical | join | `long-format-key-rejected-junk-key-offered` | Researcher uploads demographics (100 subjects, SEQN) plus a 24h dietary-recall file (400 rows, 4 records per SEQN). Step |
| critical | join | `outer-join-destroys-right-only-ids` | 'Keep everyone from every file' on demographics(SEQN) + labs(patient_id) reports 'Result: 5 rows' and 'Combined table re |
| critical | join | `asymmetric-case-folding-misjoins-ids` | sites file has specimens ['A01','B02']; labs file has ['A01','a01','B02']. The app reports 'Result: 1 rows — matching on |
| critical | join | `slug-truncation-duplicate-column-names` | Two files named '2017-2018 dietary recall day 1' and '... day 2' both share a 'kcal' column. Step 2 promises 'Both files |
| critical | stack | `stack-source-column-clobbered` | A researcher who combined NHANES 1999-2002 in the app, exported it, and later stacks that export with a new 2003-2004 cy |
| critical | stack | `stack-conflicts-only-checked-on-shared-columns` | Three NHANES cycles where LBXIN (insulin) is numeric in 1999-2000, text in 2001-2002, and absent from 2003-2004. plan_st |
| critical | stack | `stack-empty-file-turns-every-numeric-column-to-text` | A researcher stacks 1999-2000, 2001-2002 and 2003-2004; the 2001-2002 export happened to come out with headers but no ro |
| critical | stack | `stack-no-duplicate-subject-or-row-detection` | Stacking two cycles that share 25 of their 50 subjects produces 'Result: **100 rows**' with no warning, note or blocker; |
| critical | stack | `stack-hint-steers-cycles-to-link` | Eight NHANES cycles sharing a demographic core (SEQN, RIDAGEYR, RIAGENDR, RIDRETH1) with labs that drift in and out — th |
| critical | state | `lockbox-subject-leak-heuristic-gates` | After a repeated-measures merge, the lockbox draws a ROW-level test split instead of a subject-level one and says nothin |
| critical | state | `lockbox-too-few-groups-silent-fallback` | A 7-subject × 5-visit study (35 rows). detect_repeated_subjects correctly returns ('SEQN', 7, 35). ensure_lockbox then r |
| major | combine_ui | `promised-manual-key-picker-does-not-exist` | When no key candidate survives, the app shows a warning inviting the user to choose the columns manually, immediately fo |
| major | combine_ui | `dtype-mismatch-blocked-but-executed` | The screen shows a red "🛑 ... will not match" error and "This join will not work yet — see below", withholds the row-cou |
| major | join | `chain-self-inflicted-dtype-block` | Three files whose SEQN is int64 in all three. Step 1 joins cleanly. Step 2 shows a red 🛑 "'SEQN' is stored as text in yo |
| major | join | `max-columns-60-hides-key` | A 71-column lab file whose SEQN column sits at position 70 produces zero candidates; moving the same column to position  |
| major | join | `max-distinct-positional-truncation` | Two files containing exactly the same 300,000 participant IDs, stored in different row orders, are described on screen a |
| major | join | `blank-id-kept-claim-merges-both-sides` | On a left join the app says all 5 ID-less rows across both files "are kept", then drops the 3 that came from the right-h |
| major | join | `missing-token-list-eats-real-ids` | Four study centres named NA, NB, NC and ND appear identically in both files; centre NA is dropped from the join and desc |
| major | join | `column-named-like-other-sides-key-hides-collision` | demographics(SEQN, bmi) joined to labs(patient_id, SEQN, glucose) on SEQN ↔ patient_id. The UI shows no collision warnin |
| major | join | `chained-join-false-dtype-blocker` | Three files, each with an int64 SEQN column, all values identical. Attaching the second file works and shows 'Result: 4  |
| major | join | `suffix-collides-with-existing-column` | demographics has columns SEQN, bmi, bmi_demographics; labs has SEQN, bmi. Step 2 promises 'They will be kept side by sid |
| major | join | `blank-id-reattachment-invalid-index` | demographics(SEQN with one blank) + labs(patient_id with one blank, plus its own SEQN column). diagnose_join happily pro |
| major | join | `index-like-key-accepted-silently` | A clinic-visit sheet (row 1..50, participants P100…) and a lipid panel (row 1..50, subjects S900…) — two files describin |
| major | join | `same-filename-dead-end` | Two datasets both named 'data' are in the registry with their frames intact, but the page stops with 'You have 1 of your |
| major | stack | `stack-overwrites-user-source-file-column` | Stacking two cycles whose files already contain a column called '__source_file' (a perfectly legal column name) reports  |
| major | stack | `stack-tz-datetime-conflict-invisible` | Two cycles whose visit_date is datetime64[us] in one and datetime64[us, UTC] in the other stack with plan.warnings=[] an |
| major | stack | `stack-case-variant-column-names-split` | Cycle 1 has ['SEQN','age','glucose'], cycle 2 has ['SEQN','Age','Glucose '] — a completely ordinary difference between t |
| major | state | `grouped-lockbox-fraction-mislabel` | A 20-participant food-diary study (135 rows; one participant logged 40 days, the rest 5). The researcher leaves the "Hel |
| major | state | `join-key-offered-as-predictor` | Merge demographics + labs on MRN, pick `outcome` as target: the saved feature set is ['age', 'glucose', 'MRN']. The iden |
| major | state | `no-way-to-declare-subject-column` | pages/01_Upload_and_Audit.py:1020-1024 comments "A declared subject/entity ID always wins over auto-detection" and reads |
| minor | join | `case-whitespace-column-collision-unreported` | Joining a file with columns BMI and 'age ' (trailing space) to one with bmi and age produces a table with all four colum |
| minor | perf | `eager-join-before-consent` | Two 2,000-row × 16-column files sharing a repeated sentinel ID (9999 = 'refused', as survey exports code it) fan out to  |
| minor | stack | `stack-blank-file-blocks-with-wrong-advice` | Three cycles where one file loaded with no columns at all. plan_stack blocks everything with '🛑 These files have no colu |
| minor | stack | `stack-duplicate-column-label-crash` | An Excel file whose header row contains both the number 1 and the text '1' is loaded as distinct labels [1, '1'] and the |
| minor | stack | `stack-bool-int-warning-is-false` | A yes/no column stored as True/False in one cycle and 1/0 in the next raises the warning "1 column(s) hold different kin |
| minor | stack | `stack-int-float-id-precision` | SEQN is int64 in one cycle and float64 in another (which happens as soon as one cycle has a blank ID). plan_stack report |
| minor | stack | `stack-ordered-categorical-loses-ordering` | Two parquet cycles carry an ordered food_security factor, one with categories low<marginal<high and the other high<margi |

---

### `chain-promise-vs-delivery` — Three-file chain promises "Result: 200 rows" above the button and delivers 400 rows

**critical** · combine_ui · `utils/combine_ui.py:125`

**What the researcher sees:** demographics (100 subjects) + labs (2 rows/subject) + diet (2 rows/subject). The last row-count promise printed above the "Combine files" button is "Result: **200 rows**"; the line immediately above the button is "This join will not work yet — see below." The researcher clicks Combine and gets a 400-row working table. The app's stated promise #2 is "See the result before you commit. The exact row count ... shown ABOVE the button. No surprises after the fact."

**Expected:** The final promise above the button states 400 rows (diagnose_join computes predicted_rows = 400 correctly — verified in r6). The row count shown must equal the row count delivered.

**Actual:** plain_summary() short-circuits to "This join will not work yet — see below." because the diagnosis has a blocking entry, but utils/combine_ui.py:121-131 deliberately proceeds anyway for dtype-mismatch blockers and executes the join. The correct 400 is computed and then never printed. The last number on screen is step 1's stale 200.

**Root cause:** utils/combine_ui.py:121-125 allows a dtype-blocked step through (`if not diag.can_proceed and not diag.dtype_mismatch`) but still calls plain_summary(), which at ml/join_doctor.py:488-489 returns "This join will not work yet — see below." whenever `d.blocking` is non-empty. The two decisions disagree: the executor treats a dtype mismatch as repairable, the summariser treats it as fatal, and the researcher is left with the previous step's row count as the last thing they read. The blocking dtype mismatch itself is self-inflicted (see chain-self-inflicted-dtype-block), so this fires on every 3+ file chain whose keys are numeric — the normal case.

**Impact:** The single number this screen exists to guarantee is wrong by 2× (and by 2^(k-1) for a k-file chain). A researcher reading "200 rows" sees a cohort of 100 people measured once; they actually get 400 fabricated lab×diet pairings. Every n, every model, and every manuscript figure downstream is computed on a table twice the size they were shown.

```
  [Markdown] ##### Attaching **labs**
  [Caption] 'SEQN' and 'SEQN' share 100 IDs (100% of your data so far, 100% of labs).
  [Warning] labs has several rows per ID (e.g. repeated visits), so 100 subjects become 200 rows. ...
  [Markdown] Result: **200 rows** — matching on 100 shared IDs, keeping only IDs found in both files.
  [Markdown] ##### Attaching **diet**
  [Error] 'SEQN' is stored as text in your data so far but as numbers in diet. They look identical on screen but will not match. Fixing this matches 100 IDs.
  [Warning] Both files have several rows per ID, so every combination is produced: 100 shared IDs become 400 rows. This is usually a mistake ...
  [Markdown] This join will not work yet — see below.
  [Markdown] ---
  [Caption] Nothing has changed yet — press **Combine files** when the result above looks right.

DELIVERED: (400, 5) rows; distinct SEQN = 100
SUCCESS: **Combined table ready** — 400 rows × 5 columns

--- control: identical cardinalities but TEXT keys from the start (r13_chain_string_keys_and_perf.py) ---
  PROMISE: Result: **200 rows** — matching on 100 shared IDs, keeping only IDs found in both files.
  PROMISE: Result: **400 rows** — matching on 100 shared IDs, keeping only IDs found in both files.
  DELIVERED: (400, 5)
```

---

### `repeated-measures-key-rejected` — A file with more than 2 rows per subject can never be joined — the app declares "no column that lines up with your data" for a 100%-matching SEQN↔SEQN key

**critical** · join · `ml/join_doctor.py:262`

**What the researcher sees:** Researcher uploads demographics (200 subjects, 1 row each) and visits (the same 200 subjects × 3 visits = 600 rows) — the canonical repeated-measures design. Step 2 shows a yellow "No shared ID was found" warning and a red "**visits** has no column that lines up with your data, so it cannot be attached." There is no key selectbox and no Combine button. The warning tells the user "You can pick the columns yourself below" — no such control is ever rendered.

**Expected:** SEQN↔SEQN covers 100% of both files' subjects and is the only sane key. It should be proposed with high confidence, diagnosed as 1:many fan-out (200 subjects → 600 rows), and joinable. Failing that, a manual column picker must actually exist, since the UI promises one.

**Actual:** find_key_candidates returns 0 candidates. _key_tokens() rejects visits.SEQN before any comparison because 200/600 = 0.333 < _MIN_UNIQUENESS (0.5). The join is a hard dead end and the app asserts a falsehood about the data. The alternative radio option produces an 800-row table with 600 blank `age` values and 200 blank `glucose` values, presented as "Result: 800 rows" with only a mild "20% of columns" warning.

**Root cause:** ml/join_doctor.py:262 `if n_unique == 0 or n_unique / n < _MIN_UNIQUENESS: return None` with `_MIN_UNIQUENESS = 0.5` at ml/join_doctor.py:41. The gate is applied symmetrically to BOTH frames, so any file averaging more than 2 rows per subject has its true key discarded before candidate scoring. The comment at :40-41 justifies the gate as stopping "sex" from looking like a perfect key, but it also excludes every long-format repeated-measures file. utils/combine_ui.py:91-101 then emits "You can pick the columns yourself below" and blocks, with no manual picker anywhere in the module. Compounds in chains: after a 1:many join the accumulated frame's key drops below 0.5 too, so a third clean 1-row-per-subject file becomes unattachable (r4_ui_fanout_and_chain.py case (c): "**extra** has no column that lines up with your data").

**Impact:** The workflow this module exists for — "demographics + repeated visits, keyed by SEQN" — is impossible. A nutrition researcher who cannot open a terminal has two options: abandon the app, or take the other radio button and continue with an 800-row table in which 75% of `age` and 25% of `glucose` are blank. Every downstream n, model, and Table 1 is then built on a fabricated cohort.

```
demo  : (200, 3) | SEQN unique = 200
visits: (600, 3) | SEQN unique = 200 | uniqueness ratio = 0.3333333333333333

_key_tokens(visits,'SEQN') -> None
find_key_candidates -> 0 candidate(s)
suggest_best -> None

UI: st.warning("No shared ID was found ... You can pick the columns yourself below.")
UI: st.error("visits has no column that lines up with your data, so it cannot be attached.")
UI: blocked = True  -> Step 2 returns None, the researcher cannot proceed.

with 2 visits/subject (ratio 0.50): 1 candidate(s)

--- and driven through the real page (r1b_ui_repeated_measures.py) ---
ERROR  : **visits** has no column that lines up with your data, so it cannot be attached.
WARNING: No shared ID was found between your data so far and **visits**. These files may not describe the same people — or the ID columns may hold different things. You can pick the columns yourself below.
INFO   : Fix the issues above, or choose different columns, and this will update.
Combine buttons: []

--- the only remaining route: pick "same measurements on different people" (r14_stack_escape_hatch.py) ---
PROMISE: Result: **800 rows** and 6 columns, stacking 2 files that share 1 column(s) ...
DELIVERED: (800, 6)
   SEQN   age sex __source_file  visit  glucose
0  1000  62.0   M  demographics    NaN      NaN
797  1199  NaN  NaN        visits    1.0  101.726087
rows with age missing: 600 | rows with glucose missing: 200
```

---

### `casefold-decided-per-column` — normalize_key case-folds each side independently, so one stray-case row silently mismatches and mis-pairs an entire cohort

**critical** · join · `ml/join_doctor.py:114`

**What the researcher sees:** Adding a single duplicate row whose ID is typed in the wrong case collapses a perfect 10-of-10 link to 1 row, and the one surviving row carries another row's measurement. The screen blames the files: "These files may not describe the same people".

**Expected:** PT001..PT010 appear verbatim in both files, so the inner join is 10 rows and PT001 (age 40) is paired with glucose 100. Whatever case policy is chosen, it must be the SAME policy on both sides so identical strings always compare equal.

**Actual:** 1 row, and that row pairs age 40 with glucose 999 — the typo row's value, not PT001's own. The other 9 participants are silently dropped and the app suggests the two files describe different people.

**Root cause:** normalize_key() computes fold_case from the column it is normalising: `fold_case = bool(len(text) == 0 or text.str.lower().nunique() == text.nunique())` (ml/join_doctor.py:111-116). Each side of the join is normalised by a separate call (diagnose_join ml/join_doctor.py:368; repair_keys ml/join_doctor.py:507-508; execute_join ml/join_doctor.py:539-540), so a case collision anywhere in ONE column disables folding for that column only. The two sides are then compared in two different canonical spaces: left 'PT002' -> 'pt002', right 'PT002' -> 'PT002'. The single value that still matches is the lower-case typo row, which is why the surviving pair is wrong rather than merely missing.

**Impact:** A cohort silently shrinks to a handful of participants and the retained rows carry measurements belonging to other rows. The published n and every downstream statistic are wrong, and the app's explanation actively misdirects the researcher toward believing their files are unrelated.

```
demographics SEQN : ['PT001', 'PT002', 'PT003', 'PT004', 'PT005', 'PT006', 'PT007', 'PT008', 'PT009', 'PT010']
labs SEQN         : ['PT001', 'PT002', 'PT003', 'PT004', 'PT005', 'PT006', 'PT007', 'PT008', 'PT009', 'PT010', 'pt001']
   (the last labs row is the SAME participant re-keyed in lower case)

normalize_key(demographics) -> ['pt001', 'pt002', 'pt003', 'pt004', 'pt005', 'pt006', 'pt007', 'pt008', 'pt009', 'pt010']
normalize_key(labs)         -> ['PT001', 'PT002', 'PT003', 'PT004', 'PT005', 'PT006', 'PT007', 'PT008', 'PT009', 'PT010', 'pt001']
   left was case-folded, right was NOT -> 'PT002' can never equal 'pt002'

candidate  : SEQN <-> SEQN confidence = low
headline   : 'SEQN' and 'SEQN' share 1 IDs (10% of demographics, 9% of labs).
PROMISE    : Result: **1 rows** — matching on 1 shared IDs, keeping only IDs found in both files.
  WARNING  : 9 row(s) of demographics (90%) have no match and will be dropped. Use a left join to keep them.
DELIVERED  :
    SEQN  age  glucose
0  pt001   40      999

TRUTH: PT001..PT010 are present in both files verbatim; the inner join is 10 rows,
       and PT001 (age 40) belongs with glucose 100, not glucose 999.

CONTROL: delete the single 'pt001' row -> 10 rows

(via the real Step 2 screen, probe12_casefold.py:)
WARN: No shared ID was found between your data so far and **labs**. These files may not describe the same people — or the ID columns may hold different things. You can pick the columns yourself below.
MD  : Result: **1 rows** — matching on 1 shared IDs, keeping only IDs found in both files.
WORKING TABLE: 1 rows x 3 cols  cols=['SEQN', 'age', 'glucose']
    SEQN  age  glucose
0  pt001   40      999
```

---

### `float64-id-collapse-mislabelled-repeated-visits` — IDs above 2^53 read as float64 collide, and the Join Doctor explains the resulting duplicates as "repeated visits"

**critical** · join · `ml/join_doctor.py:87`

**What the researcher sees:** Six participants with 17-digit IDs join to their labs; one participant is shown three times carrying three different participants' glucose values, and the app describes this as repeated measures with a group-aware-split recommendation.

**Expected:** Either an exact 6-row 1-to-1 join, or a loud, specific blocker: "the ID column in labs was read as a decimal number and IDs of this size lose their last digits — 3 participants have been merged into 1". _canon_scalar's own docstring promises this: "passing IDs through float64 silently collides values above 2^53 ... which is a false merge — the worst outcome this module can produce."

**Actual:** 5 rows, high confidence, no blocker. Participant ...996 appears 3 times holding glucose values belonging to ...995, ...996 and ...997; participants ...993, ...995 and ...997 vanish; the duplication is attributed to "repeated visits".

**Root cause:** _canon_scalar (ml/join_doctor.py:55-93) protects only against float conversion it performs itself. When pandas has already stored the key as float64 — which pd.read_csv does as soon as one row's ID is blank (data_processor.load_csv -> pd.read_csv, data_processor.py:45) — the value arriving at _canon_scalar is already 9007199254740992.0, and the _DECIMAL_RE branch at ml/join_doctor.py:86-92 canonicalises the damaged digits as if exact. Nothing in _key_tokens, find_key_candidates or diagnose_join checks whether a numeric key column is float64 while holding integers above 2**53, so the induced duplicates are handed to the fan-out detector at ml/join_doctor.py:408-409 and reported as repeated measures.

**Impact:** Participants are conflated and measurements are attached to the wrong people, with the app volunteering a plausible-sounding but false explanation that discourages further checking. This is the exact failure mode the module names as its worst possible outcome, and it fires on ordinary long numeric IDs (Medicare/BioBank/registry identifiers) whenever the file contains one blank ID.

```
demographics dtypes: {'SEQN': 'int64', 'age': 'int64'}
labs dtypes        : {'SEQN': 'float64', 'glucose': 'int64'}  <- one blank forces float64
labs SEQN in memory: [9007199254740992.0, 9007199254740994.0, 9007199254740996.0, 9007199254740996.0, 9007199254740996.0, 9007199254740998.0, nan]
labs SEQN canonical: ['9007199254740992', '9007199254740994', '9007199254740996', '9007199254740996', '9007199254740996', '9007199254740998', nan]
demo SEQN canonical: ['9007199254740993', '9007199254740994', '9007199254740995', '9007199254740996', '9007199254740997', '9007199254740998']

suggest_best: SEQN <-> SEQN high
headline    : 'SEQN' and 'SEQN' share 3 IDs (50% of demographics, 75% of labs).
PROMISE     : Result: **5 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
  WARNING   : labs has several rows per ID (e.g. repeated visits), so 3 subjects become 5 rows. That is correct for repeated measures, but each subject now appears several times — your sample size is no longer the number of subjects, and models will need a group-aware split.
  WARNING   : 3 row(s) of demographics (50%) have no match and will be dropped. Use a left join to keep them.
  WARNING   : 1 in labs row(s) have no ID at all (blank or 'unknown'). They cannot be matched and will be dropped.
  NOTE      : 1 row(s) of labs have no match and will be dropped.
DELIVERED   :
               SEQN  age  glucose
0  9007199254740994   20      102
1  9007199254740996   40      103
2  9007199254740996   40      104
3  9007199254740996   40      105
4  9007199254740998   60      106

TRUTH: 6 participants, 6 lab results, a 1-to-1 inner join of 6 rows with
       (age 10,glucose 101) ... (age 60,glucose 106).
       Participant ...996 (age 40) is shown carrying glucose 103, 104 AND 105,
       which b
```

---

### `outer-right-join-drops-right-key-column` — execute_join deletes the right-hand key column, so right/outer joins destroy the identity of every right-only participant

**critical** · join · `ml/join_doctor.py:559`

**What the researcher sees:** "Keep everyone from every file" delivers exactly the promised 250 rows, but 50 of them have a blank participant ID — only 200 distinct IDs survive out of 250. No warning is shown.

**Expected:** An outer join of 200 + 200 rows sharing 150 IDs yields 250 rows carrying 250 distinct participant IDs. Before dropping the redundant key column the two key columns must be coalesced (merged[left_key] = merged[left_key].fillna(merged[right_key])).

**Actual:** 250 rows but only 200 distinct IDs; the 50 labs-only participants arrive with SEQN = NaN. The right join is worse: 200 rows, 150 distinct IDs. With identical key names the same data keeps all 250 IDs, so the loss is caused purely by the two files spelling the key differently — the module's own headline use case #2.

**Root cause:** pandas keeps both key columns when merging with left_on/right_on. For how in ('right','outer') the rows that exist only on the right have NaN in left_key and their real ID in right_key. execute_join then unconditionally removes that column: `if left_key != right_key and right_key in merged.columns: merged = merged.drop(columns=[right_key])` (ml/join_doctor.py:558-559), deleting the only surviving copy of those IDs. The re-attachment path for blank-ID rows above it (ml/join_doctor.py:550-554) does rename right_key -> left_key, showing the author was aware of the need, but the merged frame itself is never coalesced.

**Impact:** After a "keep everyone" combine the researcher has a table where a quarter of the participants are anonymous. They cannot be traced back to source records, cannot be linked to any further file, and cannot be de-duplicated — while the row count matches the promise exactly, so nothing looks wrong.

```
--- outer (Keep everyone from every file)
PROMISE  : Result: **250 rows** — matching on 150 shared IDs, keeping every row of both files.
warnings : []
notes    : []
DELIVERED: 250 rows;  rows with a blank SEQN: 50
           distinct IDs surviving: 200
    SEQN  age  glucose
247  NaN  NaN    197.0
248  NaN  NaN    198.0
249  NaN  NaN    199.0

--- right (right join)
PROMISE  : Result: **200 rows** — matching on 150 shared IDs, keeping every row of labs.
warnings : []
notes    : []
DELIVERED: 200 rows;  rows with a blank SEQN: 50
           distinct IDs surviving: 150

--- control: identical key NAMES, same data
DELIVERED: 250 rows;  rows with a blank SEQN: 0
           distinct IDs surviving: 250

(fuzz census, probe11_misc.py, 300 random pairs x 4 join types:)
  ('outer', 'LOST_IDS') 295   e.g. (9, 16)
  ('right', 'LOST_IDS') 295   e.g. (7, 14)

(via the real Step 2 screen, probe04_ui.py scenario B:)
MD: Result: **5 rows** — matching on 1 shared IDs, keeping every row of both files.
WORKING TABLE: 5 rows x 3 cols  cols=['SEQN', 'age', 'glucose']
  SEQN   age  glucose
0    1  40.0      NaN
1    2  50.0      NaN
2    3  60.0     90.0
3  NaN   NaN    100.0
4  NaN   NaN    110.0
```

---

### `slug-truncation-suffix-collision` — _slug truncates file names to 20 characters, producing identical merge suffixes, duplicate column labels and an unrecoverable session

**critical** · join · `ml/join_doctor.py:569`

**What the researcher sees:** Two files whose names share their first 20 characters produce a working table with two columns literally named `bmi_nutrition_study_base`; the page then raises "Duplicate column names found" on that run and on every subsequent rerun.

**Expected:** The suffixes must be distinct. The app has just promised "They will be kept side by side with suffixes so nothing is overwritten" — baseline BMI and follow-up BMI must remain tellable apart, and pressing the button must not leave the page in a state that crashes forever.

**Actual:** Both suffixes are `_nutrition_study_base`. The result has two identically-named BMI columns, `wt['bmi_nutrition_study_base']` returns a 2-column DataFrame, and pages/01_Upload_and_Audit.py:532 raises ValueError from pyarrow on this and every later rerun while session_state['working_table'] stays committed.

**Root cause:** `_slug` truncates to 20 characters: `re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")[:20] or "x"` (ml/join_doctor.py:568-569). execute_join builds `suffixes = (f"_{_slug(left_name)}", f"_{_slug(right_name)}")` (ml/join_doctor.py:529) with no check that the two differ, and pandas.merge accepts equal suffixes by producing duplicate labels rather than raising. Dataset names default to the uploaded filename stem (pages/01_Upload_and_Audit.py:314 and :411), so any two files agreeing on their first 20 filename characters collide. The frame is committed to session_state before the page renders it, so the crash recurs on every rerun.

**Impact:** The two visits' measurements become indistinguishable in the combined table — the single thing the on-screen warning promised would not happen — and the researcher's session is bricked: every reload of Upload & Audit shows a Python traceback with no way to clear the poisoned table from the UI.

```
_slug('nutrition_study_baseline_visit') = 'nutrition_study_base'
_slug('nutrition_study_baseline_followup') = 'nutrition_study_base'

what the screen promises:
  WARNING: Both files have column(s) named bmi. They will be kept side by side with suffixes so nothing is overwritten.

DELIVERED columns: ['SEQN', 'bmi_nutrition_study_base', 'sbp', 'bmi_nutrition_study_base', 'glucose', 'ldl', 'hdl']
DUPLICATED labels: ['bmi_nutrition_study_base']
  SEQN  bmi_nutrition_study_base    sbp  bmi_nutrition_study_base  glucose    ldl   hdl
0    1                      34.1   89.0                      26.8     89.9   97.0  69.0
1    2                      28.6  128.0                      25.3     94.6  138.0  37.0
2    3                      30.9  130.0                      24.2     87.8  129.0  37.0
3    4                      36.0  111.0                      38.8     74.1   89.0  60.0
  wt['bmi_nutrition_study_base'] returns a DataFrame of shape (20, 2) - baseline and follow-up BMI are
  no longer distinguishable by name.

page exception on the run that committed it: True
   Duplicate column names found: ['SEQN', 'bmi_nutrition_study_base', 'sbp', 'bmi_nutrition_study_base', 'glucose', 'ldl', 'hdl']
page exception on rerun 2: True | working_table still committed: True
page exception on rerun 3: True | working_table still committed: True

Note the Join Doctor itself refuses to work on such a frame:
  diagnose_join -> ValueError : The column 'bmi_nutrition_study_base' appears more than once in one of these files. Rename or remove the duplicate before joining.
```

---

### `min-uniqueness-hides-repeated-measures-key` — _key_tokens rejects any key with 3+ rows per subject, so the commonest repeated-measures link is declared impossible

**critical** · join · `ml/join_doctor.py:262`

**What the researcher sees:** Demographics (100 rows, SEQN 1..100) and diet recalls (400 rows, the same SEQN repeated four times) — an identical, 100%-overlapping SEQN column in both files — yield zero key candidates, and the screen says the files may not describe the same people.

**Expected:** SEQN <-> SEQN should be proposed as the key, with the existing fan-out warning explaining that 100 subjects become 400 rows. The module's own docstring lists this exact case as failure #3 that it was written to handle.

**Actual:** find_key_candidates returns [] as soon as there are 3 or more rows per subject (uniqueness 0.33 < 0.5), the true key is never offered, and the researcher is dead-ended with an error telling them the files do not line up. At exactly 2 rows per subject (uniqueness == 0.5) it works, so the behaviour flips on an invisible threshold.

**Root cause:** _key_tokens rejects a column outright when `n_unique / n < _MIN_UNIQUENESS` with _MIN_UNIQUENESS = 0.5 (ml/join_doctor.py:41, :262), returning None so the column never reaches the candidate loop in find_key_candidates (ml/join_doctor.py:289-296). The rule is stated as "a key must identify rows, not group them", but it is applied to BOTH sides symmetrically; in a link the long side legitimately groups rows, and only ONE side needs to be near-unique. The comparison should be made per-pair (require high uniqueness on at least one side) rather than as a per-column admission filter.

**Impact:** The single most common multi-file shape in nutrition research — one demographics row per participant plus several 24-hour dietary recalls per participant — cannot be combined at all, and the app tells the researcher their files probably describe different people. It also kills every composite-key case (multi-centre studies where subject numbers restart per site).

```
demographics: 100 rows, SEQN 1..100
diet_recalls: 400 rows, SEQN 1..100 repeated 4x (four 24h recalls each)
every SEQN in demographics appears in diet_recalls: True

_MIN_UNIQUENESS      = 0.5
uniqueness of diet SEQN = 0.25
_key_tokens(diet,'SEQN') -> None
find_key_candidates      -> []
suggest_best             -> None

...although the join is perfectly well defined:
    Result: **400 rows** — matching on 100 shared IDs, keeping only IDs found in both files.

  recalls per participant = 2 -> candidates [('SEQN', 'SEQN', 'high'), ('age', 'SEQN', 'medium'), ('age', 'kcal', 'medium'), ('SEQN', 'kcal', 'low')]
  recalls per participant = 3 -> candidates [('age', 'kcal', 'medium'), ('SEQN', 'kcal', 'low')]
  recalls per participant = 4 -> candidates [('age', 'kcal', 'medium'), ('SEQN', 'kcal', 'low')]
  recalls per participant = 5 -> candidates [('age', 'kcal', 'medium'), ('SEQN', 'kcal', 'low')]

=== what the researcher sees on the Step 2 screen ===
WARNING: No shared ID was found between your data so far and **diet_recalls**. These files may not describe the same people — or the ID columns may hold different things. You can pick the columns yourself below.
ERROR  : **diet_recalls** has no column that lines up with your data, so it cannot be attached.
INFO   : Fix the issues above, or choose different columns, and this will update.
selectboxes offered for choosing the key yourself: []

(same root cause for a multi-centre composite key, probe10_scale_composite.py:)
both files: 3 sites x 50 subjects = 150 rows; (site,subject) is unique, subject is not
candidates: []
suggest_best: None
```

---

### `decoy-key-passes-index-guard` — suggest_best confidently proposes joining a measurement column to a row counter, and the UI pre-selects it

**critical** · join · `ml/join_doctor.py:345`

**What the researcher sees:** Two unrelated files (a cohort with a row counter + age, and an unrelated table with a row counter + gdp) are offered a single pre-selected key `age ↔ n` at medium confidence, and Combine produces a 26-row table of fabricated pairings.

**Expected:** suggest_best should return None here. Its own docstring: "Low-confidence candidates are withheld by default: telling a researcher 'these files join on age <-> gdp' is far worse than saying nothing and letting them pick the key themselves."

**Actual:** `age ↔ n` is returned at medium confidence, is the ONLY option in the Step 2 selectbox, is described as sharing 19 IDs, and Combine commits a 26-row table in which people are matched to strangers' records by coincidence of age value. The real row-counter pairing `row ↔ n` is correctly marked low and hidden — so the guard actively pushes the worse candidate to the top.

**Root cause:** The index-like guard requires BOTH sides to be counters: `index_like=_looks_like_row_index(lraw) and _looks_like_row_index(rraw)` (ml/join_doctor.py:345), and both the confidence downgrade (ml/join_doctor.py:189) and the score penalty (ml/join_doctor.py:182) key off that conjunction. A small-integer measurement column (age, visit number, parity, dose) paired against a row counter has index_like=False, so it escapes the penalty entirely, scores above the genuinely index-like pair, and becomes suggest_best. Any 1..N counter overlaps ~100% with any small-integer column by construction, so this decoy is available in almost every file pair.

**Impact:** The researcher is handed a confident, pre-selected join between two files that have nothing to do with each other, gets a plausible row count and a plausible-sounding repeated-measures caveat, and proceeds to model a table of randomly-paired records. Nothing on screen suggests the pairing is coincidental.

```
cohort_a: a row counter 'row' and an 'age' column — 50 people
cohort_b: a row counter 'n' and a 'gdp' column — 50 unrelated records

  age  <-> n    confidence=medium index_like=False  score=0.304
  row  <-> n    confidence=low    index_like=True  score=0.083
suggest_best -> age <-> n (medium)
headline     : 'age' and 'n' share 19 IDs (53% of cohort_a, 38% of cohort_b).
PROMISE      : Result: **26 rows** — matching on 19 shared IDs, keeping only IDs found in both files.
  WARNING    : cohort_a has several rows per ID (e.g. repeated visits), so 19 subjects become 26 rows. That is correct for repeated measures, but each subject now appears several times — your sample size is no longer the number of subjects, and models will need a group-aware split.
  WARNING    : 24 row(s) of cohort_a (48%) have no match and will be dropped. Use a left join to keep them.
DELIVERED    : 26 rows
   row age    gdp
0    2  42  0.624
1    4  21  0.436
2    6  26  0.075
3    7  18  0.387
4    8  39  0.401
5   10  37  0.524
6   11  28  0.168
7   15  28  0.168
  (rows 6 and 7: two different people, both aged 28, receive the same gdp)

=== on the Step 2 screen ===
SELECTBOX combine_key_cohort_b options = ['age ↔ n']  preselected = age ↔ n
MARKDOWN : Result: **26 rows** — matching on 19 shared IDs, keeping only IDs found in both files.
CAPTION  : 'age' and 'n' share 19 IDs (53% of your data so far, 38% of cohort_b).
committed working table: (26, 3)

Same root cause with a real key present but filtered out by FINDING 5:
  candidates: [('age', 'kcal', 'medium'), ('SEQN', 'kcal', 'low')]
  suggest_best: ('age', 'kcal', 'medium')
```

---

### `long-format-key-rejected-junk-key-offered` — A repeated-measures file makes the real subject key un-proposable, and the app offers a nonsense pairing as the default — delivering a fabricated table

**critical** · join · `ml/join_doctor.py:262`

**What the researcher sees:** Researcher uploads demographics (100 subjects, SEQN) plus a 24h dietary-recall file (400 rows, 4 records per SEQN). Step 2 offers exactly ONE key option — 'age ↔ food_code' — preselected, promises 'Result: 66 rows', and on Combine delivers a table where each participant's age is attached to food records belonging to completely different participants.

**Expected:** SEQN ↔ SEQN is offered (the module's own docstring lists 'one file has several rows per subject' as a supported case, and KeyCandidate carries left_has_duplicates/right_has_duplicates for exactly this). Joining on it gives 400 rows, one per food record, each carrying its own participant's age.

**Actual:** SEQN is silently removed from the candidate pool on the long side, leaving only 'age ↔ food_code'. The delivered 66-row table has SEQN_demographics != SEQN_dietary_recall in every single row: participant 1000's age is glued to food records of participants 1056 and 1090.

**Root cause:** _key_tokens() rejects any column whose distinct-value ratio is below _MIN_UNIQUENESS=0.5 (line 262: `if n_unique == 0 or n_unique / n < _MIN_UNIQUENESS: return None`). A long-format file with 3+ rows per subject has ratio <= 0.33, so the true key is never even canonicalised, and find_key_candidates() cannot propose it. What survives is the coincidental numeric overlap between 'age' (18-79) and 'food_code' (1-499); KeyCandidate.confidence (line 195) rates that 'medium' because max coverage is exactly 0.5, so utils/combine_ui.py:89 keeps it as 'usable' and it becomes the preselected option.

**Impact:** The most common NHANES-style link in nutrition research (demographics + long-format dietary recall) cannot be done correctly, and the app does not say so — it confidently produces a table of participants whose exposures belong to other people. Every downstream model, n, and p-value is fabricated.

```
SELECTBOX OPTIONS OFFERED TO THE RESEARCHER:
   Which columns identify the same person in both files? -> ['age ↔ food_code']  value = age ↔ food_code

MESSAGES:
  WARNING | Both files have several rows per ID, so every combination is produced: 25 shared IDs become 66 rows. This is usually a mistake — check whether one file should be summarised to one row per subject first.
  WARNING | Both files have column(s) named SEQN. They will be kept side by side with suffixes so nothing is overwritten.
  MD      | Result: **66 rows** — matching on 25 shared IDs, keeping only IDs found in both files.
  CAPTION | 'age' and 'food_code' share 25 IDs (50% of your data so far, 9% of dietary recall).

DELIVERED TABLE: (66, 4)
   SEQN_demographics age  SEQN_dietary_recall        kcal
0               1000  62                 1056  307.516258
1               1000  62                 1090  239.074129
2               1009  37                 1018   78.192579
3               1009  37                 1059  365.726857
4               1011  68                 1042  209.759093
5               1011  68                 1051  328.438543
6               1012  54                 1099  262.397369
7               1013  41                 1039  258.304855
TRUTH: demographics(100 subjects) + dietary(400 records on the SAME 100 subjects) must give 400 rows keyed by SEQN.

--- engine level (r12_repeated_measures.py) ---
_key_tokens(diet,'SEQN') -> None
suggest_best        -> KeyCandidate(left_col='age', right_col='food_code', coverage_left=0.5, ... name_similarity=0.18181818181818182, index_like=False)
Threshold probe — how many records per subject before the key disappears:
  1 row(s)/subject: uniqueness=1.00 candidates=1
  2 row(s)/subject: uniqueness=0.50 candidates=1
  3 row(s)/subject: uniqueness=0.33 
```

---

### `outer-join-destroys-right-only-ids` — Outer/right join on differently-named keys silently blanks the ID of every row that came only from the second file

**critical** · join · `ml/join_doctor.py:558`

**What the researcher sees:** 'Keep everyone from every file' on demographics(SEQN) + labs(patient_id) reports 'Result: 5 rows' and 'Combined table ready — 5 rows × 3 columns', but 2 of the 5 delivered rows have SEQN = NaN. Participants 4 and 5, whose IDs are right there in the labs file, arrive in the working table with no identifier at all.

**Expected:** SEQN column = 1,2,3,4,5 — the outer join keeps every subject from both files and every one of them keeps its identifier (SQL COALESCE semantics, which is what 'keeping every row of both files' means).

**Actual:** SEQN = 1,2,3,NaN,NaN. The identifiers of the right-only subjects are deleted from the delivered frame; the row count is right and the contents are not.

**Root cause:** With left_key != right_key pandas keeps BOTH key columns and does not coalesce them; for outer/right joins the left_key cell is NaN on right-only rows and the true ID lives only in right_key. execute_join then does `if left_key != right_key and right_key in merged.columns: merged = merged.drop(columns=[right_key])` (lines 558-559) without first filling left_key from right_key, so the surviving key column is the one that is empty for exactly those rows.

**Impact:** Rows silently become unidentifiable subjects. A researcher who then de-duplicates by ID, does a group-aware split, or merges a third file loses or mis-assigns those participants — and nothing in the UI hints at it, because the promised row count is exactly correct.

```
======================================================================
how = outer
Result: **5 rows** — matching on 2 shared IDs, keeping every row of both files.
Standardised the join keys ('SEQN', 'patient_id') ... Merged demographics (3 rows) with labs (4 rows) on 'SEQN' using a outer join, giving 5 rows.
  SEQN   bmi  glucose
0    1  22.1      NaN
1    2  27.4     90.0
2    3  31.0    105.0
3  NaN   NaN    120.0
4  NaN   NaN     88.0
rows with a NULL SEQN in result: 2
distinct SEQN values delivered: ['1', '2', '3']
======================================================================
how = right
Result: **4 rows** — matching on 2 shared IDs, keeping every row of labs.
  SEQN   bmi  glucose
0    2  27.4       90
1    3  31.0      105
2  NaN   NaN      120
3  NaN   NaN       88
rows with a NULL SEQN in result: 2
distinct SEQN values delivered: ['2', '3']
======================================================================
GROUND TRUTH: outer join of ids {1,2,3} and {2,3,4,5} must deliver ids 1,2,3,4,5 -- five identified subjects. Ids 4 and 5 exist in labs.

--- same thing through the real Step 2 screen (r08_ui.py) ---
   | Result: **5 rows** — matching on 2 shared IDs, keeping every row of both files.
   | **Combined table ready** — 5 rows × 3 columns
DELIVERED FRAME:
  SEQN   bmi  glucose
0    1  22.1      NaN
1    2  27.4     90.0
2    3  31.0    105.0
3  NaN   NaN    120.0
4  NaN   NaN     88.0
SEQN nulls: 2 of 5
```

---

### `asymmetric-case-folding-misjoins-ids` — Case folding is decided independently for each file, so an ID can be attached to the wrong subject while an exact match is dropped

**critical** · join · `ml/join_doctor.py:114`

**What the researcher sees:** sites file has specimens ['A01','B02']; labs file has ['A01','a01','B02']. The app reports 'Result: 1 rows — matching on 1 shared IDs' and delivers specimen 'a01' with glucose 999 for the row that was 'A01' in the sites file. The exactly-matching 'A01' row (glucose 90) is ignored and 'B02' — an exact match on both sides — is dropped entirely.

**Expected:** Either both sides fold case (2 matches, A01->90 or an explicit ambiguity warning) or neither does (A01->90 and B02->105, 2 rows). Under no policy should 'A01' be paired with the row whose ID is 'a01' while the literal 'A01' row sits unused.

**Actual:** The left column folds to lowercase (its own values are unambiguous), the right column does not (it contains both 'A01' and 'a01'), so left 'A01'->'a01' matches right 'a01' and left 'B02'->'b02' matches nothing. One row of wrong data delivered, one correct match silently discarded.

**Root cause:** normalize_key() auto-decides fold_case per Series (`fold_case = bool(len(text) == 0 or text.str.lower().nunique() == text.nunique())`, line 114) and diagnose_join (line 368) / repair_keys (lines 507-508) call it separately on the left and right columns. The two sides can therefore end up in different canonical spaces. The docstring's promise — 'if a column genuinely contains both "abc" and "ABC" they are kept apart' — is honoured within a column and violated across the join.

**Impact:** Lab values are attached to the wrong specimen/participant, and subjects with a perfectly good exact-match ID are dropped, with the UI reporting a plausible-looking small match count. This is the exact 'confidently wrong answer' the app's contract forbids.

```
left  normalize_key -> ['a01', 'b02']
right normalize_key -> ['A01', 'a01', 'B02']
Result: **1 rows** — matching on 1 shared IDs, keeping only IDs found in both files.
warnings: ['1 row(s) of sites (50%) have no match and will be dropped. Use a left join to keep them.']
notes: ['2 row(s) of labs have no match and will be dropped.']
  specimen     site  glucose
0      a01  clinic1      999

GROUND TRUTH: 'A01' from sites must receive glucose=90 (the row whose specimen is literally 'A01'), NOT 999 (specimen 'a01').

--- through the real Step 2 screen (r09_ui_dupcols_and_fold.py) ---
  | Result: **1 rows** — matching on 1 shared IDs, keeping only IDs found in both files.
  | 'specimen' and 'specimen' share 1 IDs (50% of your data so far, 33% of labs).
DELIVERED:
  specimen     site  glucose
0      a01  clinic1      999
```

---

### `slug-truncation-duplicate-column-names` — File-name suffixes are truncated to 20 characters, so two files can produce IDENTICAL column suffixes — the merged table gets two same-named columns and the page then crashes

**critical** · join · `ml/join_doctor.py:569`

**What the researcher sees:** Two files named '2017-2018 dietary recall day 1' and '... day 2' both share a 'kcal' column. Step 2 promises 'Both files have column(s) named kcal. They will be kept side by side with suffixes so nothing is overwritten' and 'Result: 3 rows'. The committed working table contains TWO columns both named 'kcal_2017_2018_dietary_re' holding day-1 and day-2 values, and the page then dies rendering the preview.

**Expected:** The two colliding columns get distinguishable names (e.g. kcal_day_1 / kcal_day_2), matching the stated promise that they are 'kept side by side with suffixes so nothing is overwritten'.

**Actual:** Both get the same 20-character-truncated suffix. The delivered frame has duplicate labels; selecting 'kcal_2017_2018_dietary_re' returns a 2-column DataFrame and .mean() returns two numbers. st.session_state['working_table'] is committed in that state and the Upload page raises ValueError on every subsequent run, so the user is stuck on a broken page with a corrupt table.

**Root cause:** _slug() truncates to 20 characters (`re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")[:20] or "x"`, line 569) and execute_join builds `suffixes = (f"_{_slug(left_name)}", f"_{_slug(right_name)}")` (line 529) with no uniqueness check. Any two file names agreeing on their first 20 alphanumeric-ish characters — extremely common for 'same study, different day/cycle/site' files — collapse to the same suffix, and pandas happily emits duplicate labels.

**Impact:** Two files' measurements become indistinguishable inside one column name; any analysis of that variable silently mixes the two files, and the Upload & Audit page is permanently broken for that session (the corrupt table is stored in session_state, so the ValueError repeats on every rerun).

```
--- engine (r06/r07) ---
'Baseline visit measurements A'          -> _slug='Baseline_visit_measu'
'Baseline visit measurements B'          -> _slug='Baseline_visit_measu'
   identical suffix: True
'nhanes_demographics_2017.csv'           -> _slug='nhanes_demographics_'
'nhanes_demographics_2018.csv'           -> _slug='nhanes_demographics_'
   identical suffix: True

python warnings emitted: []
columns: ['SEQN', 'kcal_2017_2018_dietary_re', 'protein_2017_2018_dietary_re', 'kcal_2017_2018_dietary_re', 'protein_2017_2018_dietary_re']
  SEQN  kcal_2017_2018_dietary_re  protein_2017_2018_dietary_re  kcal_2017_2018_dietary_re  protein_2017_2018_dietary_re
0    1                       1800                            70                       1750                            71
1    2                       2100                            80                       2200                            82
2    3                       1950                            90                       1900                            93
Selecting one of the promised columns:
<class 'pandas.DataFrame'> -> shape (3, 2)
mean of that 'column': kcal_2017_2018_dietary_re    1950.0
kcal_2017_2018_dietary_re    1950.0
dtype: float64

--- real page (r09_ui_dupcols_and_fold.py) ---
at.exception: ["Duplicate column names found: ['SEQN', 'kcal_2017_2018_dietary_re', 'kcal_2017_2018_dietary_re']"]
success msgs: ['**Combined table ready** — 3 rows × 3 columns']
working_table cols: ['SEQN', 'kcal_2017_2018_dietary_re', 'kcal_2017_2018_dietary_re']
  File "/home/user/tabular-ml-lab/pages/01_Upload_and_Audit.py", line 532, in <module>
    table(working_df.head(10), width="stretch")
ValueError: Duplicate column names found: ['SEQN', 'kcal_2017_2018_dietary_re', 'kcal_2017_2018_dietary_re']
2nd run exception: (same Val
```

---

### `stack-source-column-clobbered` — Re-stacking a combined table silently rewrites every provenance label and delivers one column fewer than promised

**critical** · stack · `/home/user/tabular-ml-lab/utils/combine.py:151`

**What the researcher sees:** A researcher who combined NHANES 1999-2002 in the app, exported it, and later stacks that export with a new 2003-2004 cycle is shown 'Result: **8 rows** and 4 columns' and an expander reading '**__source_file** — not in: 2003-2004' plus 'will be blank for those rows'. What is actually delivered is 3 columns, and every row that used to be labelled '1999-2000' or '2001-2002' now says 'nhanes_1999_2002_combined'. The 'Rows per file' line under the success banner then reports a file breakdown that never happened.

**Expected:** execute_stack must not overwrite a pre-existing __source_file column. It should either preserve the original labels (renaming, e.g. __source_file_prev) or plan_stack must block/warn that the reserved column already exists. In every case plan.summary()'s column count must equal the delivered column count.

**Actual:** plan_stack counts __source_file in plan.all_columns and then adds 1 for the column execute_stack will add, promising N+1 columns while execute_stack overwrites the existing column in place and delivers N. The original per-cycle provenance is destroyed, and the plan's 'will be blank for those rows' claim is false — the new file's rows get the new file name written into them.

**Root cause:** execute_stack does `part[SOURCE_COLUMN] = n` unconditionally (combine.py:151), clobbering any existing column of that name. StackPlan.summary() computes `len(self.all_columns) + 1` (combine.py:54) with no check for SOURCE_COLUMN already being in all_columns, and plan_stack's partial_columns logic (combine.py:104-108) reports the reserved column as an ordinary missing column. A 600-trial fuzz (p11_fuzz_shape.py) found 266 shape-contract violations, 100% of them from this one cause; with __source_file removed from the column pool, 600 trials produced 0 mismatches.

**Impact:** Provenance — the one thing the module exists to record — is silently rewritten, so a methods section citing 'rows per cycle' is wrong and unrecoverable. Reachable whenever a combined table leaves and re-enters the app (page 10 writes data_sample.csv from the working table; utils/table_export.py offers CSV download).

```
PROMISE shown above the button:
   Result: **8 rows** and 4 columns, stacking 2 files that share 2 column(s) — plus one column recording which file each row came from.
   expander: **__source_file** — not in: 2003-2004
   note    : ['1 column(s) are missing from at least one file and will be blank for those rows.']

DELIVERED:
   SEQN  age              __source_file
0     1   62  nhanes_1999_2002_combined
1     2   65  nhanes_1999_2002_combined
2     3   71  nhanes_1999_2002_combined
3   100   18  nhanes_1999_2002_combined
4   101   21  nhanes_1999_2002_combined
5   102   77  nhanes_1999_2002_combined
6   200   21                  2003-2004
7   201   57                  2003-2004

  promised columns = 4   actual = 3
  provenance before: ['1999-2000', '2001-2002']
  provenance after : ['2003-2004', 'nhanes_1999_2002_combined']

--- same scenario driven through the real page with AppTest (p08_ui_stack_promises.py) ---
  Result line   : ['Result: **140 rows** and 5 columns, stacking 2 files that share 3 column(s) — plus one column recording which file each row came from.']
  captions(exp) : ['**__source_file** — not in: NHANES_2003_2004']
  DELIVERED shape: (140, 4)
  success banner : ['**Combined table ready** — 140 rows × 4 columns']
  rows-per-file  : ['Rows per file: nhanes_1999_2002_combined (100), NHANES_2003_2004 (40)']
```

---

### `stack-conflicts-only-checked-on-shared-columns` — A column that drifts in AND changes type across cycles is mixed into one object column with no warning at all

**critical** · stack · `/home/user/tabular-ml-lab/utils/combine.py:111`

**What the researcher sees:** Three NHANES cycles where LBXIN (insulin) is numeric in 1999-2000, text in 2001-2002, and absent from 2003-2004. plan_stack reports type_conflicts={} and warnings=[]; the only thing shown is the mild note '1 column(s) are missing from at least one file and will be blank for those rows.' The delivered LBXIN column is object dtype holding [5.1, 6.2, '7.3', '8.4', nan, nan], get_numeric_columns() no longer lists it, and describe() returns count/unique/top/freq instead of mean/std.

**Expected:** The type-conflict scan must run over every column in the union, comparing only the files that actually contain it. A lab that is numeric in one cycle and text in another must be warned about whether or not a third cycle happens to lack it.

**Actual:** The scan iterates plan.shared_columns (the intersection across ALL files), so adding a single cycle that lacks the column removes it from the check entirely and the conflict becomes invisible. The control in the same script proves the identical conflict IS caught when all three files carry the column.

**Root cause:** `for c in plan.shared_columns:` at combine.py:111 restricts the dtype-family comparison to the intersection of column sets. It should iterate `union` (built at combine.py:88-90) and compare only across `[n for n in names if c in col_sets[n]]`.

**Impact:** This is the exact 'quiet killer' plan_stack's own docstring (combine.py:73-75) promises to catch, and it is undetected in the single most common stacking scenario — schema drift, where a lab appears in some cycles only. The column silently stops being numeric, disappears from the feature pool, and any mean/SD reported for it is a string frequency table.

```
shared_columns : ['SEQN', 'RIDAGEYR']  <- LBXIN is NOT here, so it is never type-checked
type_conflicts : {}
warnings       : []
notes          : ['1 column(s) are missing from at least one file and will be blank for those rows.']
summary        : Result: **6 rows** and 4 columns, stacking 3 files that share 2 column(s) — plus one column recording which file each row came from.

delivered LBXIN dtype : object
delivered LBXIN values: [5.1, 6.2, '7.3', '8.4', nan, nan]
get_numeric_columns   : ['SEQN', 'RIDAGEYR']
LBXIN.describe():
count     4.0
unique    4.0
top       5.1
freq      1.0

CONTROL - the SAME conflict on a column present in all three files IS caught:
  type_conflicts: {'LBXIN': ['number', 'text']}
  warnings      : ["1 column(s) hold different kinds of value in different files ('LBXIN' is number in some files and text). After stacking they become text, which no model can use until it is cleaned up."]
```

---

### `stack-empty-file-turns-every-numeric-column-to-text` — One empty (header-only) cycle among several silently converts every numeric column in the whole stacked table to text

**critical** · stack · `/home/user/tabular-ml-lab/utils/combine.py:154`

**What the researcher sees:** A researcher stacks 1999-2000, 2001-2002 and 2003-2004; the 2001-2002 export happened to come out with headers but no rows (pandas loads it as 0 rows x 3 object columns). The combined table is delivered with SEQN, RIDAGEYR and LBXGLU all as object dtype. get_numeric_columns() returns [], describe() on glucose returns count/unique/top/freq, and corr(numeric_only=True) is a 0x0 matrix. The plan never says a file contributed 0 rows, and the 'Rows per file' summary silently omits the empty cycle so the researcher cannot even see which file did it.

**Expected:** Frames contributing zero rows must be excluded from the concat (they cannot contribute values, only dtypes), and plan_stack must state plainly that a named file contributed 0 rows — that is the actual problem, not a 'kind of value' disagreement.

**Actual:** execute_stack passes the empty object-dtype frame straight into pd.concat. Under pandas 3.0 (requirements.txt pins only pandas>=2.0.0; the installed version here is 3.0.3) empty frames now participate in dtype resolution, so every numeric column is widened to object. plan_stack's only output blames the data ('SEQN is number in some files and text') and never mentions that a file has no rows; render_combined_summary's value_counts drops the empty file from the provenance line entirely.

**Root cause:** `pd.concat(parts, ignore_index=True, sort=False)` at combine.py:154 with no filtering of zero-row frames, combined with plan_stack computing total_rows as a plain sum (combine.py:94) and never flagging len(frames[n]) == 0. The existing test suite (tests/test_combine.py, 11 passed) contains no empty-file case.

**Impact:** One accidentally-empty cycle turns the researcher's entire glucose/age/BMI dataset into text. Every numeric column vanishes from the feature pool, correlations become an empty matrix, and Table 1 reports frequency counts instead of means — while the app reports '90 rows x 4 columns' and a clean success banner.

```
empty cycle as loaded: (0, 3) {'SEQN': 'object', 'RIDAGEYR': 'object', 'LBXGLU': 'object'}
summary  : Result: **90 rows** and 4 columns, stacking 3 files that share 3 column(s) — plus one column recording which file each row came from.
warnings : ["3 column(s) hold different kinds of value in different files ('SEQN' is number in some files and text; 'RIDAGEYR' is number in some files and text; 'LBXGLU' is number in some files and text). After stacking they become text, which no model can use until it is cleaned up."]
notes    : []
mentions a 0-row file? False

delivered dtypes    : {'SEQN': 'object', 'RIDAGEYR': 'object', 'LBXGLU': 'object', '__source_file': 'str'}
cell types in LBXGLU: ['float']
get_numeric_columns : []
LBXGLU.describe():
count     90.0
unique    81.0
top       98.2
freq       3.0
out.corr(numeric_only=True).shape: (0, 0)
render_combined_summary would print: Rows per file: NHANES_1999_2000 (50), NHANES_2003_2004 (40)

CONTROL without the empty cycle: {'SEQN': 'int64', 'RIDAGEYR': 'int64', 'LBXGLU': 'float64'} | get_numeric_columns: ['SEQN', 'RIDAGEYR', 'LBXGLU']
```

---

### `stack-no-duplicate-subject-or-row-detection` — Overlapping subjects and exact duplicate rows are stacked with zero warning, silently inflating n and leaking subjects across the train/test split

**critical** · stack · `/home/user/tabular-ml-lab/utils/combine.py:94`

**What the researcher sees:** Stacking two cycles that share 25 of their 50 subjects produces 'Result: **100 rows**' with no warning, note or blocker; the table actually contains 75 distinct SEQNs with 25 subjects present twice. Uploading the same file twice (nhanes_clean.csv and 'nhanes_clean (1).csv' from a Downloads folder) produces 'Combined table ready — 80 rows x 4 columns' where 40 rows are byte-identical duplicates — again with no warning.

**Expected:** plan_stack must check row identity before the button, exactly as the join path does for fan-out: report how many ID values appear in more than one file, and how many rows are exact duplicates across files, so the researcher can decide whether to de-duplicate. The Step 2 screen explicitly promises this (utils/combine_ui.py:16-18: 'which subjects repeat, are all shown ABOVE the button').

**Actual:** plan_stack only counts rows (`sum(len(frames[n]))`). It never inspects values, so neither repeated subject IDs nor exact duplicate rows are detected or mentioned anywhere in the plan, the summary, or the post-combine banner.

**Root cause:** plan_stack builds its diagnosis purely from column names and dtypes (combine.py:83-137); `plan.total_rows = int(sum(len(frames[n]) for n in names))` at combine.py:94 is the only row-level statement it makes. There is no candidate-ID scan even though ml/join_doctor.py already has the machinery (normalize_key / value overlap) to do it.

**Impact:** The reported n is wrong in every downstream table and manuscript sentence; worse, a duplicated subject that lands on both sides of the train/test split leaks the test set into training and inflates every performance metric. This is the app's own top-severity failure mode — a confidently-wrong answer with no visible symptom.

```
A. two 'cycles' that share 25 of their 50 subjects
  hint       : stack
  summary    : Result: **100 rows** and 4 columns, stacking 2 files that share 3 column(s) — plus one column recording which file each row came from.
  warnings   : []  notes: []  blocking: []
  distinct SEQN in result: 75 of 100 rows
  SEQNs appearing twice  : 25
  -> the researcher's stated n is 100; the real number of subjects is 75
  utils/combine_ui.py:16-18 promises: "The exact row count, which rows get
  dropped, and which subjects repeat, are all shown ABOVE the button."

B. the same file uploaded twice (a stray 'file (1).csv' in Downloads)
  hint     : stack
  summary  : Result: **80 rows** and 4 columns, stacking 2 files that share 3 column(s) — plus one column recording which file each row came from.
  warnings : []  notes: []
  delivered rows: 80  exact duplicate rows (ignoring the source column): 40
  every subject is now in the analysis twice; the success banner reads
  '**Combined table ready** — 80 rows × 4 columns'
```

---

### `stack-hint-steers-cycles-to-link` — relationship_hint confidently tells a researcher that NHANES cycles are 'different measurements on the same people' and pre-selects the join path

**critical** · stack · `/home/user/tabular-ml-lab/utils/combine.py:180`

**What the researcher sees:** Eight NHANES cycles sharing a demographic core (SEQN, RIDAGEYR, RIAGENDR, RIDRETH1) with labs that drift in and out — the module's flagship use case. The Step 2 screen renders '💡 These look like **different measurements on the same people** — they have mostly different columns, so they probably link by an ID.', pre-selects 'Different measurements on the same people', and lands the researcher in the join screen, where the chained joins are blocked and no 'Combine files' button is rendered at all. Stacking, the correct operation, would have produced 480 rows x 13 columns.

**Expected:** A set of files that share a stable identifier core and partition the ID space are cycles and must be hinted 'stack'. The hint should look at row identity (do the same IDs recur across files, or do the files partition the ID space?), not only at the fraction of column names in common. And when the hint is 'unclear' the UI must not silently pre-select one of the two operations as if it were confident.

**Actual:** relationship_hint is name-overlap-only: 4 shared of 12 union columns = 0.333 <= 0.4, so it returns 'link' and the UI prints the bolded 💡 claim. With only two cycles differing by two columns the overlap is 4/6 = 0.667, which returns 'unclear' — and combine_ui maps unclear to index 0, i.e. also the join option, with no caption at all (verified in p06_ui_hint.py). Following the app's own advice on 6 cycles yields a 2-row table from 300 rows of data.

**Root cause:** combine.py:176-181 decides purely from `len(shared)/len(union)` with thresholds 0.8/0.4, which schema drift across cycles pushes straight into the 'link' band. utils/combine_ui.py:196 then does `default_idx = {"link": 0, "stack": 1}.get(hint, 0)`, so 'unclear' is not neutral — it silently defaults to the join screen too.

**Impact:** 'Combine my NHANES cycles' — the user story the module was written for — is answered with a confident, bolded, wrong recommendation. The researcher is steered into joins that either dead-end or silently reduce 480 rows of cohort to 2, and only a user who overrides the app's stated advice gets the right answer.

```
Files (8 NHANES cycles, shared demographic core + 2 drifting labs each):
   NHANES_1999_2000 ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'LBXGLU', 'LBXIN']
   NHANES_2001_2002 ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'LBXIN', 'LBXTC']
   ... (6 more)
   NHANES_2013_2014 ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'LBXVIDMS']
shared: ['RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'SEQN']  union size: 12  overlap: 0.3333333333333333
relationship_hint -> link

radio PRE-SELECTED : 'Different measurements on the same people'
HINT CAPTION SHOWN : 💡 These look like **different measurements on the same people** — they have mostly different columns, so they probably link by an ID.

--- what the user would get if they trusted the default and pressed Combine ---
no Combine button (blocked)

--- what the CORRECT operation (stack) would have given ---
plan.summary(): Result: **480 rows** and 13 columns, stacking 8 files that share 4 column(s) — plus one column recording which file each row came from.
stacked shape: (480, 13)

--- 6 cycles with cumulative drift, same wrong default, join actually completes (p06_ui_hint.py) ---
  PRE-SELECTED  : 'Different measurements on the same people'
  hint caption  : (none)
  Result line   : ['Result: **41 rows** ...', 'Result: **2 rows** — matching on 2 shared IDs, keeping only IDs found in both files.', 'Result: **1 rows** ...', 'Result: **2 rows** — matching on 1 shared IDs, keeping only IDs found in both files.']
```

---

### `lockbox-subject-leak-heuristic-gates` — The same subject lands in both training and the sealed test set whenever detect_repeated_subjects' name/ratio heuristic misses — silently, with a clean 🔒 lockbox chip

**critical** · state · `utils/test_lockbox.py:89`

**What the researcher sees:** After a repeated-measures merge, the lockbox draws a ROW-level test split instead of a subject-level one and says nothing. The status chip reads "🔒 Test set: 15% (n=27) held out since upload — opened once at Train & Compare." while 24 of 60 subjects have rows on both sides of the quarantine. Minutes earlier the Join Doctor itself told the user "models will need a group-aware split".

**Expected:** detect_repeated_subjects docstring: "Used to catch the case that silently defeats the quarantine: a merge with repeated measures puts the SAME subject in both the training rows and the sealed test rows." It must catch these, or the chip must not claim a lockbox. Zero subjects on both sides — as it correctly achieves for a SEQN column at exactly 2.0–3.0 rows/subject (verified working in r7).

**Actual:** detect_repeated_subjects returns None for all four shapes, ensure_lockbox falls through to a plain train_test_split over rows, and render_lockbox_status prints the non-grouped chip that mentions neither subjects nor the fallback.

**Root cause:** Two independent gates in detect_repeated_subjects, each silently sufficient to defeat the quarantine. (1) utils/test_lockbox.py:88-90 `rows_per = n / k; if rows_per < 1.5 or rows_per > 50: continue` — partial repeated measures (only some subjects have a second visit, the commonest real cohort shape) falls under 1.5, and a many:many merge product falls over 50. (2) utils/test_lockbox.py:92-97 requires the column NAME to contain one of ('id','seqn','subject','participant','patient','record','case','person'); 'MRN' — the standard clinical subject key, and one that ml/triage.py:128 does recognise — matches none of them, nor does 'SUBJ'. Both paths return None and ensure_lockbox (utils/test_lockbox.py:126-131) has no other source of a group column, because the declared-entity path is dead (see no-way-to-declare-subject-column).

**Impact:** This is the failure the whole split-first architecture exists to prevent. Held-out performance is measured on subjects the model was trained on, so every AUC/R² in the report is inflated by memorisation and is not held-out performance at all — while the UI stamps it 🔒 and the manuscript records "a 15% test set was held out at upload, before feature engineering or selection." The app told the researcher a group-aware split was needed and then quietly did not do one. Aggravated by the join key being a default predictor (see join-key-offered-as-predictor): with MRN in the feature set the model can memorise subject→outcome directly.

```
==== gate 1: rows/subject = 1.30  (100 subjects, 30 with a 2nd visit)
     rows=130 subjects=100 rows/subject=1.30
     detect_repeated_subjects -> None
     lockbox group_col -> None
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 7 of 100
     CHIP: 🔒 Test set: 15% (n=20) held out since upload — opened once at Train & Compare.

==== gate 1b: rows/subject = 60  (10 subjects after a many:many merge)
     rows=600 subjects=10 rows/subject=60.00
     detect_repeated_subjects -> None
     lockbox group_col -> None
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 10 of 10
     CHIP: 🔒 Test set: 15% (n=90) held out since upload — opened once at Train & Compare.

==== gate 2: key named 'MRN' (medical record number), 60 subjects x 3 visits
     rows=180 subjects=60 rows/subject=3.00
     detect_repeated_subjects -> None
     lockbox group_col -> None
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 24 of 60
     CHIP: 🔒 Test set: 15% (n=27) held out since upload — opened once at Train & Compare.

==== gate 2b: key named 'SUBJ', 60 subjects x 3 visits
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 24 of 60

--- driven end to end through page 01 (r10_e2e_mrn_leak.py) ---
MERGE WARNING: clinic_labs has several rows per ID (e.g. repeated visits), so 100 subjects become 200 rows. That is correct for repeated measures, but each subject now appears several times — your sample size is no longer the number of subjects, and models will need a group-aware split.
DELIVERED: (200, 4) | distinct MRN: 100
LOCKBOX: {'n_total': 200, 'n_test': 30, 'group_col': None, 'n_test_groups': None}
SUBJECTS IN BOTH TRAIN AND HELD-OUT TEST: 24 of 100
LOCKBOX CHIP: 🔒 Test set: 15% (n=30) held out since upload — opened once at Train & Compare.
default features: ['age', 'glucose', 'MRN']
```

---

### `lockbox-too-few-groups-silent-fallback` — With fewer than 8 subjects the lockbox finds the subject column, then abandons subject-level splitting without a word

**critical** · state · `utils/test_lockbox.py:166`

**What the researcher sees:** A 7-subject × 5-visit study (35 rows). detect_repeated_subjects correctly returns ('SEQN', 7, 35). ensure_lockbox then resets group_col to None because 7 < _MIN_GROUPS_FOR_GROUPED_LOCKBOX, does a row-level split, and renders the ordinary chip "🔒 Test set: 15% (n=6) held out since upload". 4 of the 7 subjects sit on both sides.

**Expected:** Under the app's own rule that a confidently-wrong answer is worse than a crash, discovering "this cohort has too few subjects to hold any of them out" must be stated out loud — either refuse to draw a lockbox, or draw it and label it "NOT subject-separated: the same people are in training and testing."

**Actual:** utils/test_lockbox.py:165-166 sets `group_col = None` with only a source-code comment ("too few groups to split by subject"), the row-level split proceeds, and the persisted lockbox dict is indistinguishable from a genuinely clean cross-sectional one. Nothing on any page mentions it.

**Root cause:** ensure_lockbox (utils/test_lockbox.py:151-166): grouping is attempted only when `n_groups >= _MIN_GROUPS_FOR_GROUPED_LOCKBOX` (=8, utils/test_lockbox.py:28); otherwise `test_labels` stays None and line 166 discards the detected group column. The lockbox dict then stores `group_col: None` and `n_test_groups: None`, so render_lockbox_status (utils/test_lockbox.py:238-242) takes the non-grouped branch and pages/01_Upload_and_Audit.py:1025 skips the 'split by subject' notice. Distinct from the heuristic-gate finding: here detection SUCCEEDED and the app made an undisclosed decision to ignore it.

**Impact:** Pilot studies, n-of-few crossover trials and small clinical series — exactly the cohorts where this app is most likely to be used — get a quarantine badge on a test set that shares every subject with training. In the 5-subject case 100% of subjects are on both sides. The reported held-out metrics are pure memorisation and are labelled as protected.

```
==== 7 subjects x 5 visits = 35 rows  (threshold is 8 subjects)
     detect_repeated_subjects -> ('SEQN', 7, 35)   (subject column WAS found)
     lockbox group_col -> None   (grouping was dropped)
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 4 of 7
     CHIP: 🔒 Test set: 15% (n=6) held out since upload — opened once at Train & Compare.

==== 6 subjects x 8 visits = 48 rows  (threshold is 8 subjects)
     detect_repeated_subjects -> ('SEQN', 6, 48)   (subject column WAS found)
     lockbox group_col -> None   (grouping was dropped)
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 5 of 6
     CHIP: 🔒 Test set: 15% (n=8) held out since upload — opened once at Train & Compare.

==== 5 subjects x 12 visits = 60 rows  (threshold is 8 subjects)
     detect_repeated_subjects -> ('SEQN', 5, 60)   (subject column WAS found)
     lockbox group_col -> None   (grouping was dropped)
     SUBJECTS IN BOTH TRAINING AND HELD-OUT TEST: 5 of 5
     CHIP: 🔒 Test set: 15% (n=9) held out since upload — opened once at Train & Compare.
```

---

### `promised-manual-key-picker-does-not-exist` — The screen says "You can pick the columns yourself below" and then renders no picker

**major** · combine_ui · `utils/combine_ui.py:94`

**What the researcher sees:** When no key candidate survives, the app shows a warning inviting the user to choose the columns manually, immediately followed by an error saying the file cannot be attached. No selectbox, multiselect or text input for choosing a key is rendered.

**Expected:** Either a real manual key picker (two column dropdowns) as promised, or a message that does not promise one. There is no 'Combine files' button either, so the workflow simply ends.

**Actual:** The only combine widget on the page is `combine_base_file` (which file to start from). There is no way to name the join key by hand, and the follow-up INFO tells the user to "choose different columns" — an action the UI does not offer.

**Root cause:** _render_link warns "You can pick the columns yourself below" at utils/combine_ui.py:91-95 and then sets `usable = cands[:5]` (utils/combine_ui.py:96). When find_key_candidates returned an empty list — which happens for every repeated-measures file (finding 5), every wide file whose key is past column 60 (finding 8), and any genuinely unrelated pair — `usable` is still empty, control falls into the error branch at utils/combine_ui.py:98-102, and the selectbox at :105 is never reached. No manual-entry path exists anywhere in the module.

**Impact:** This is the terminal state for the two commonest key-discovery failures. A researcher who knows perfectly well that both files are keyed on SEQN is told the app cannot see it, is invited to fix it by hand, and then given no control with which to do so. The only escape is to leave the app and edit the files.

```
WARNINGS:
   No shared ID was found between your data so far and **diet_recalls**. These files may not describe the same people — or the ID columns may hold different things. You can pick the columns yourself below.
ERRORS:
   **diet_recalls** has no column that lines up with your data, so it cannot be attached.
INFO:
   Fix the issues above, or choose different columns, and this will update.

widgets actually rendered for the failing file:
  selectboxes: ['combine_base_file']
  multiselects: []
  text inputs: []
  buttons: ['← Home', 'EDA →', 'Remove', 'Remove', '📥 Save Progress', 'Clear Current Session']
```

---

### `dtype-mismatch-blocked-but-executed` — A type mismatch is shown as a blocking error with no row prediction, then combined successfully anyway — and repair_keys makes every leg after the first raise the same false alarm

**major** · combine_ui · `utils/combine_ui.py:121`

**What the researcher sees:** The screen shows a red "🛑 ... will not match" error and "This join will not work yet — see below", withholds the row-count prediction, and still offers Combine files, which produces a correct 20-row table. In a three-file link where all files store SEQN as int64, leg 2 raises the same error because leg 1 rewrote the key as text.

**Expected:** The screen's own promise is "See the result before you commit. The exact row count ... shown ABOVE the button. No surprises after the fact" (utils/combine_ui.py:14-18). A repairable type mismatch should read as a resolved note plus the predicted row count, not as a blocker. And joining three files that all store SEQN identically must not manufacture a type mismatch out of the app's own repair.

**Actual:** plain_summary returns "This join will not work yet — see below." whenever d.blocking is non-empty (ml/join_doctor.py:489-490), so no row count is displayed, while utils/combine_ui.py:121 deliberately lets dtype_mismatch through to execute_join. The user sees a stop sign and a live Combine button at the same time. Because repair_keys writes canonical strings back into the key column, every file after the first in a multi-file link trips the same error even when nothing was ever mistyped.

**Root cause:** Two halves disagree about what 'blocking' means. diagnose_join classifies a dtype mismatch as blocking (ml/join_doctor.py:422-428) and plain_summary suppresses the prediction for any blocking diagnosis (ml/join_doctor.py:489-490); utils/combine_ui.py:121 then overrides the block for exactly this case (`if not diag.can_proceed and not diag.dtype_mismatch`) and joins with repair=True. The recurrence on later legs comes from repair_keys assigning object-dtype canonical strings into the key column (ml/join_doctor.py:507-508), which the next call's `pd.api.types.is_numeric_dtype(ls) != pd.api.types.is_numeric_dtype(rs)` test (ml/join_doctor.py:374-376) reads as a genuine text-vs-number conflict.

**Impact:** The researcher is shown a hard stop with no predicted row count, cannot tell whether pressing the button is safe, and — if they trust the message — abandons a join that would have worked. Those who press it get a table whose size they were never shown. In a three-file study the false alarm appears on every file after the first, training the user to ignore the app's only stop signal.

```
=== what the screen says ===
ERROR   : 'SEQN' is stored as text in your data so far but as numbers in labs. They look identical on screen but will not match. Fixing this matches 20 IDs.
SUMMARY : This join will not work yet — see below.
row-count prediction shown above the button: NONE
button offered: ['Combine files']
=== what pressing it produces ===
20 rows x 3 cols
  SEQN  age  glucose
0    1   62    102.9
1    2   65    129.1
2    3   71    115.2

=== the self-inflicted version: three files, all int64 SEQN ===
A.SEQN dtype: int64  B.SEQN dtype: int64  C.SEQN dtype: int64
after leg 1 the working key dtype is object -> ['1', '2', '3']
leg 2 PROMISE: This join will not work yet — see below.
  BLOCKING: 'SEQN' is stored as text in your data so far but as numbers in C. They look identical on screen but will not match. Fixing this matches 20 IDs.
leg 2 actually delivers 20 rows
```

---

### `chain-self-inflicted-dtype-block` — Chained joins report a blocking "stored as text vs numbers" error the app created itself one step earlier

**major** · join · `ml/join_doctor.py:507`

**What the researcher sees:** Three files whose SEQN is int64 in all three. Step 1 joins cleanly. Step 2 shows a red 🛑 "'SEQN' is stored as text in your data so far but as numbers in diet. They look identical on screen but will not match." Nothing about the researcher's data is wrong — execute_join rewrote the key to strings during step 1.

**Expected:** No blocking error. The three files have identical key dtypes; the chain should read as three clean attachments.

**Actual:** repair_keys() writes normalize_key()'s canonical STRING form back into the key column, so the accumulated frame's key is object dtype from step 2 onward and every subsequent numeric-key file trips diagnose_join's `is_numeric_dtype(ls) != is_numeric_dtype(rs)` test. The user is shown a red blocking error on data that is not broken.

**Root cause:** ml/join_doctor.py:507-508 in repair_keys(): `l2[left_key] = normalize_key(l2[left_key])` replaces the key column with canonical strings in the RETURNED frame, and utils/combine_ui.py:129 feeds that frame back in as `result` for the next attachment. diagnose_join's dtype check (ml/join_doctor.py:374-376) then compares an app-normalised text key against the next file's untouched numeric key. Either the repaired frame should carry the original dtype, or the diagnosis should compare canonicalised forms rather than raw dtypes.

**Impact:** On the app's advertised multi-file workflow, the third file onward always shows a scary red 🛑 blocking error that the researcher cannot act on (there is nothing to fix in their files). It also suppresses the true row count (see chain-promise-vs-delivery) and trains users to click through red errors on this screen.

```
demo.SEQN dtype : int64
labs.SEQN dtype : int64
diet.SEQN dtype : int64
step1 dtype_mismatch: False blocking: []
after step 1 -> result.SEQN dtype: object sample: ['1000', '1000', '1001']
step2 dtype_mismatch: True
step2 blocking     : ["'SEQN' is stored as text in the first file but as numbers in the second file. They look identical on screen but will not match. Fixing this matches 5 IDs."]
step2 can_proceed  : False
step2 plain_summary: This join will not work yet — see below.
step2 predicted_rows: 20
step2 delivered rows: 20
```

---

### `max-columns-60-hides-key` — find_key_candidates only inspects the first 60 columns, so a wide lab export with the ID appended last is declared unrelatable

**major** · join · `ml/join_doctor.py:284`

**What the researcher sees:** A 71-column lab file whose SEQN column sits at position 70 produces zero candidates; moving the same column to position 0 without changing a value produces a high-confidence match.

**Expected:** Either all columns are scanned, or the user is told "only the first 60 columns of labs were examined for an ID" so the omission is visible. Silently ignoring two-thirds of a file's columns and then asserting the files may describe different people breaks the "diagnose visibly" contract.

**Actual:** Columns past index 59 are never considered and nothing anywhere mentions the cap. The failure is indistinguishable from a genuine no-overlap result, and lands in the same dead end as finding 7.

**Root cause:** find_key_candidates hard-slices both column lists before doing any work: `lcols = list(left.columns)[:max_columns]` / `rcols = list(right.columns)[:max_columns]` with max_columns=60 (ml/join_doctor.py:272, :283-284). The cheap rejection in _key_tokens already makes scanning all columns affordable, and no caller ever raises the limit — utils/combine_ui.py:88 calls find_key_candidates(result, frames[other]) with defaults.

**Impact:** NHANES-style lab and questionnaire exports routinely carry 70-200 columns and often place the participant ID last. Those files simply cannot be linked, and the app blames the data rather than its own scan limit.

```
labs has 71 columns; SEQN is at position 70
every SEQN matches: True
find_key_candidates -> []
suggest_best        -> None

move SEQN to position 0, change nothing else:
  candidates -> [('SEQN', 'SEQN', 'high'), ('age', 'SEQN', 'medium')]

SCREEN WARNING: No shared ID was found between your data so far and **labs**. These files may not describe the same people — or the ID columns may hold different things. You can pick the columns yourself below.
SCREEN ERROR  : **labs** has no column that lines up with your data, so it cannot be attached.
```

---

### `max-distinct-positional-truncation` — _key_tokens takes the first 200,000 distinct values in file order — the per-file subset its own docstring says it refuses to take

**major** · join · `ml/join_doctor.py:266`

**What the researcher sees:** Two files containing exactly the same 300,000 participant IDs, stored in different row orders, are described on screen as sharing 133,168 IDs — 67% of each file — at high confidence.

**Expected:** The headline must agree with the join: 300,000 shared IDs, 100% of each file. _key_tokens' docstring states the design intent explicitly — "Deliberately NOT a row sample. Sampling each file independently compares two different random subsets, so on files above the sample size the measured overlap collapses toward zero and the true key stops being proposed exactly when the data is large enough to matter."

**Actual:** The candidate headline reports 133,168 / 67% / 67%, and the same figures drive the score, the confidence band and the ordering of candidates. Only diagnose_join (called after a key has been chosen) sees the true 300,000. The two screens contradict each other by a third of the cohort.

**Root cause:** _key_tokens caps the distinct values it canonicalises with `uniques = uniques.iloc[:_MAX_DISTINCT]` (ml/join_doctor.py:265-266, _MAX_DISTINCT = 200_000 at :38). `uniques` comes from `s.dropna().drop_duplicates()`, which preserves row order, so the cap keeps the first 200,000 distinct IDs *as they appear in that file*. When the two files are ordered differently (one sorted by ID, one by exam date) the two retained subsets differ, which is precisely the independent per-file subsetting the docstring at ml/join_doctor.py:240-249 says the function avoids. Since the truncation also lowers coverage, it can push a true key below _MIN_COVERAGE or below a decoy's score.

**Impact:** On any cohort above 200,000 participants the key-selection screen understates the overlap, prompting the researcher to conclude a third of their sample is unmatched, switch to a left join, or abandon the true key for a decoy that happens to score higher after truncation.

```
_MAX_DISTINCT = 200000
both files contain exactly the same 300,000 participant IDs, stored in different orders
truth: 100% of demographics is in labs and vice versa

find_key_candidates (1.5s):
  n_matched   = 133,168
  coverage    = 0.67 / 0.67
  confidence  = high
  HEADLINE    : 'SEQN' and 'SEQN' share 133,168 IDs (67% of demographics, 67% of labs).

diagnose_join (which does NOT truncate) says:
   Result: **300,000 rows** — matching on 300,000 shared IDs, keeping only IDs found in both files.
  matched_keys = 300,000

The screen therefore tells the researcher a third of their cohort is missing
while the join itself matches all of it.
```

---

### `blank-id-kept-claim-merges-both-sides` — The blank-ID warning sums both files' counts and then makes one keep/drop claim that is false for one side

**major** · join · `ml/join_doctor.py:471`

**What the researcher sees:** On a left join the app says all 5 ID-less rows across both files "are kept", then drops the 3 that came from the right-hand file.

**Expected:** Per-side statements: "2 rows of demographics have no ID — they are kept without matching information; 3 rows of labs have no ID — they cannot be matched and will be dropped."

**Actual:** One combined sentence listing both counts, followed by a single verdict chosen by whichever side happens to be non-zero first. The 3 labs rows are silently discarded under a sentence that says they are kept.

**Root cause:** diagnose_join builds one message from both counts (ml/join_doctor.py:465-476) and picks its verdict with `kept = how in ("left", "outer") and n_missing_left or how in ("right", "outer") and n_missing_right` (ml/join_doctor.py:471). For how='left' with blanks on both sides this evaluates to n_missing_left (truthy) and the 'kept' wording is applied to the whole combined count, even though execute_join only re-attaches l_blank for left/outer and r_blank for right/outer (ml/join_doctor.py:548-554).

**Impact:** Rows with unrecorded IDs are exactly the rows a researcher must account for in a CONSORT/participant-flow diagram. Being told they were retained when they were dropped produces an unreconcilable accounting of the sample and an incorrect exclusion count in the write-up.

```
demographics: 4 rows, 2 with a blank ID
labs        : 5 rows, 3 with a blank ID
PROMISE : Result: **4 rows** — matching on 2 shared IDs, keeping every row of demographics.
  WARNING: 2 in demographics and 3 in labs row(s) have no ID at all (blank or 'unknown'). They are kept but will have no matching information attached.
DELIVERED:
  SEQN  age  glucose
0    1   40     90.0
1    2   50     91.0
2  NaN   60      NaN
3  NaN   70      NaN

The 3 blank-ID rows of labs (glucose 92, 93, 94) are dropped, not kept.
The warning counted them into '5 rows have no ID at all ... They are kept'.
```

---

### `missing-token-list-eats-real-ids` — _KEY_MISSING_TOKENS deletes legitimate identifiers such as "NA", "-" and "?", then reports 100% coverage of what is left

**major** · join · `ml/join_doctor.py:79`

**What the researcher sees:** Four study centres named NA, NB, NC and ND appear identically in both files; centre NA is dropped from the join and described as a row that "has no ID at all", while the candidate headline claims 100% coverage of both files.

**Expected:** A value that appears verbatim in both key columns should match. At minimum the token blacklist must not be applied to values that occur on BOTH sides (a shared "NA" is evidence it is a real code), and the message must name the value it discarded rather than asserting the row has no ID.

**Actual:** Any key whose lower-cased text is one of 13 blacklisted strings is nulled in _canon_scalar. The candidate coverage is then computed over the survivors only, so the headline reports 100% of both files while a real centre has been deleted, and the warning misdescribes the value as absent.

**Root cause:** _canon_scalar returns None for any value whose stripped, lower-cased text is in _KEY_MISSING_TOKENS (ml/join_doctor.py:48-49, :79-80). The list contains ordinary short codes — 'na', 'none', 'null', 'missing', '-', '--', '.', '?' — with no check for whether the value also occurs on the other side, and no per-column heuristic (e.g. only treat these as missing when the rest of the column looks structurally different). Coverage in find_key_candidates (ml/join_doctor.py:314-315) is computed after the nulling, so the loss is invisible in the headline.

**Impact:** Whole study centres, country codes (NA = Namibia/North America), sentinel-coded participant groups and any ID literally spelled '-' or '?' disappear from combined tables. Because the headline still says 100%, the researcher has no cue that anyone was excluded, and the accompanying warning tells them the affected rows never had an ID.

```
tokens treated as 'no identifier': ['-', '--', '.', '?', 'missing', 'n.a.', 'n/a', 'na', 'nan', 'none', 'not available', 'null', 'unknown']

both files list four centres: NA (Namibia), NB, NC, ND
normalize_key -> [nan, 'nb', 'nc', 'nd']
candidate : site <-> site high
HEADLINE  : 'site' and 'site' share 3 IDs (100% of enrolment, 100% of lab summary).
PROMISE   : Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
  WARNING : 1 in enrolment and 1 in lab summary row(s) have no ID at all (blank or 'unknown'). They cannot be matched and will be dropped.
DELIVERED :
  site  n_enrolled  mean_glucose
0   nb          98          99.8
1   nc         143         104.4
2   nd          77          97.1

Centre NA is present, spelled identically, in both files. It is dropped and
described as a row that 'has no ID at all'.

Same for participant IDs:
  normalize_key(['NA001','?','-','P004']) -> ['na001', nan, nan, 'p404'.replace...]  # actual: ['na001', nan, nan, 'p004']
```

---

### `column-named-like-other-sides-key-hides-collision` — A file carrying a column named like the OTHER file's join key renames the real ID away, leaves a decoy ID column, and produces no collision warning

**major** · join · `ml/join_doctor.py:401`

**What the researcher sees:** demographics(SEQN, bmi) joined to labs(patient_id, SEQN, glucose) on SEQN ↔ patient_id. The UI shows no collision warning at all. The delivered table has no column called SEQN; the participant ID is now 'SEQN_demographics' and a column called 'SEQN_labs' — the lab machine's sequence number 777/888/999 — sits where a researcher would look for the participant ID.

**Expected:** diagnose_join warns that 'SEQN' exists in both files and will be suffixed (it does emit that warning for any other shared column), and the column the user joined on keeps a recognisable name.

**Actual:** column_collisions is empty and no warning is shown, because the collision name happens to equal the left key. The join key is renamed to SEQN_demographics and the unrelated SEQN_labs becomes the most ID-looking column in the table.

**Root cause:** diagnose_join builds `collisions = [str(c) for c in (set(left.columns) & set(right.columns)) if str(c) not in {str(left_key), str(right_key)}]` (lines 401-402). Excluding BOTH key names suppresses genuine collisions when one side has a non-key column named like the other side's key — precisely the case where pandas will suffix the key column itself. execute_join's `overlap` set (line 547) has the same blind spot.

**Impact:** The researcher is never told the ID column was renamed, and is left with a decoy 'SEQN_labs' column. Grouping, de-duplicating or splitting by that column silently uses the wrong identifier.

```
column_collisions reported : []
warnings                   : []
plain_summary              : Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
delivered columns          : ['SEQN_demographics', 'bmi', 'SEQN_labs', 'glucose']
  SEQN_demographics   bmi  SEQN_labs  glucose
0                 1  22.1        777       90
1                 2  27.4        888      105
2                 3  31.0        999      120

'SEQN' (the participant id the user joined on) is GONE from the result.
'SEQN_labs' looks like the participant id but is 777/888/999.

--- same result through the real Step 2 screen (r08_ui.py scenario 4) ---
   | Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
   | 'SEQN' and 'patient_id' share 3 IDs (100% of your data so far, 100% of labs).
   (no "Both files have column(s) named …" warning is emitted)
DELIVERED COLUMNS: ['SEQN_demographics', 'bmi', 'SEQN_labs', 'glucose']
```

---

### `chained-join-false-dtype-blocker` — Three clean files with identical numeric SEQN: the app blocks itself with a red 'stored as text … but as numbers' error and shows no row count for the last file

**major** · join · `ml/join_doctor.py:507`

**What the researcher sees:** Three files, each with an int64 SEQN column, all values identical. Attaching the second file works and shows 'Result: 4 rows'. Attaching the third shows a red error — "'SEQN' is stored as text in your data so far but as numbers in labs" — and 'This join will not work yet — see below', with no predicted row count. The Combine button is nevertheless active and produces the correct 4-row table.

**Expected:** Three clean files with the same numeric key produce no blocking error, and the screen shows the predicted row count for every attachment ('See the result before you commit. The exact row count … shown ABOVE the button').

**Actual:** The app diagnoses a data problem that the user does not have and that the app itself created, states 'This join will not work yet' about a join that works, and withholds the row count for the final (and therefore the actual) result — while still offering the Combine button.

**Root cause:** repair_keys writes the canonical STRING form back into the key column (`l2[left_key] = normalize_key(l2[left_key])`, lines 507-508), so after the first join the accumulated frame's SEQN is object dtype. utils/combine_ui.py:112 then re-diagnoses that repaired frame against the next raw file, and diagnose_join's `dtype_mismatch = is_numeric_dtype(ls) != is_numeric_dtype(rs)` (lines 374-376) fires, producing a blocking message and making plain_summary return the 'will not work yet' string instead of a row count.

**Impact:** Any link of three or more files with a numeric ID — the app's flagship use case — is presented as broken. A cautious researcher stops here or starts 'fixing' a file that was never wrong; a less cautious one presses Combine and commits a table the app just told them would not work.

```
WHAT THE RESEARCHER SEES ON STEP 2 (3 clean files, identical SEQN):
  ERROR   | 'SEQN' is stored as text in your data so far but as numbers in labs. They look identical on screen but will not match. Fixing this matches 4 IDs.
  MD      | ##### Attaching **body**
  MD      | Result: **4 rows** — matching on 4 shared IDs, keeping only IDs found in both files.
  MD      | ##### Attaching **labs**
  MD      | This join will not work yet — see below.
  CAPTION | 'SEQN' and 'SEQN' share 4 IDs (100% of your data so far, 100% of body).
  CAPTION | 'SEQN' and 'SEQN' look like the same ID, but one file stores it as text and the other as numbers — so nothing matches until that is fixed (4 would match after fixing).
  buttons: [... 'Combine files' ...]
DELIVERED: ['SEQN', 'age', 'bmi', 'glu'] 4 rows

--- engine confirmation (r10_source_col_and_chain.py part C) ---
after first join, SEQN dtype: object ['1', '2', '3']
  cand SEQN<->SEQN conf=high dtype_mismatch=True
blocking: ["'SEQN' is stored as text in your data so far but as numbers in labs. ..."]
plain_summary: This join will not work yet — see below.
```

---

### `suffix-collides-with-existing-column` — A column already named <col>_<filename> turns the promised suffixing into a raw pandas MergeError

**major** · join · `ml/join_doctor.py:529`

**What the researcher sees:** demographics has columns SEQN, bmi, bmi_demographics; labs has SEQN, bmi. Step 2 promises 'They will be kept side by side with suffixes so nothing is overwritten' and 'Result: 3 rows', then shows 'Could not attach labs: Passing 'suffixes' which cause duplicate columns {'bmi_demographics'} is not allowed.' and refuses to combine.

**Expected:** The suffix is chosen so it cannot collide with an existing column (e.g. bmi_demographics_2), or the collision is diagnosed up front with an actionable fix — the module's stated goal is 'Problems come with a fix, not a stack trace… not "You are trying to merge on str and int64 columns"'.

**Actual:** execute_join raises pandas' MergeError; utils/combine_ui.py:135 prints the pandas text verbatim and blocks the combine with no suggested remedy.

**Root cause:** `suffixes = (f"_{_slug(left_name)}", f"_{_slug(right_name)}")` (line 529) is passed straight to merge (lines 542-544) without checking the resulting names against the existing columns of either frame. Note the colliding name is exactly what a previous run of this same tool produces, so re-combining an already-combined file hits it.

**Impact:** A hard dead end phrased in pandas jargon for a researcher who has never opened a terminal, in a screen whose whole premise is that such messages never appear.

```
plain_summary: Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
warning      : Both files have column(s) named bmi. They will be kept side by side with suffixes so nothing is overwritten.
execute_join RAISED MergeError: Passing 'suffixes' which cause duplicate columns {'bmi_demographics'} is not allowed.

--- real Step 2 screen (r08_ui.py scenario 3) ---
   | Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
   | Both files have column(s) named bmi. They will be kept side by side with suffixes so nothing is overwritten.
   | Could not attach **labs**: Passing 'suffixes' which cause duplicate columns {'bmi_demographics'} is not allowed.
```

---

### `blank-id-reattachment-invalid-index` — Outer join with blank IDs crashes with 'Reindexing only valid with uniquely valued Index objects' when the second file has a column named like the first file's key

**major** · join · `ml/join_doctor.py:552`

**What the researcher sees:** demographics(SEQN with one blank) + labs(patient_id with one blank, plus its own SEQN column). diagnose_join happily promises 'Result: 5 rows … keeping every row of both files'; execute_join then dies inside its own blank-row re-attachment.

**Expected:** 5 rows delivered, or a diagnosable, plain-language blocker shown before the button.

**Actual:** Uncaught pandas InvalidIndexError from inside execute_join; the Step 2 screen surfaces it as 'Could not attach labs: Reindexing only valid with uniquely valued Index objects'.

**Root cause:** For how in ('right','outer') the blank-ID rows of the right frame are renamed with `rb = rb.rename(columns={right_key: left_key})` (lines 552-553). The `overlap` set used for suffixing on line 547 subtracts {left_key, right_key}, so a right-side column literally named left_key ('SEQN') is NOT pre-suffixed; renaming patient_id -> SEQN then creates two 'SEQN' labels in rb, and the pd.concat on line 556 cannot reindex a non-unique column index.

**Impact:** The join is impossible to complete and the explanation is meaningless to the audience. The promise printed one line above the button ('5 rows') is never deliverable.

```
UI promises: Result: **5 rows** — matching on 1 shared IDs, keeping every row of both files.
blocking: [] warnings: ["1 in demographics and 1 in labs row(s) have no ID at all (blank or 'unknown'). They are kept but will have no matching information attached."]
execute_join RAISED InvalidIndexError: Reindexing only valid with uniquely valued Index objects
--- traceback tail ---
  File "/home/user/tabular-ml-lab/ml/join_doctor.py", line 556, in execute_join
    merged = pd.concat([merged] + extras, ignore_index=True)
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py", line 3728, in get_indexer
    raise InvalidIndexError(self._requires_unique_msg)
pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
```

---

### `index-like-key-accepted-silently` — Two files whose only shared column is a row counter with the same name are joined at 'high' confidence with no warning, fusing unrelated cohorts

**major** · join · `ml/join_doctor.py:189`

**What the researcher sees:** A clinic-visit sheet (row 1..50, participants P100…) and a lipid panel (row 1..50, subjects S900…) — two files describing different people — are offered a single key 'row ↔ row' at high confidence, and Combine delivers 50 rows pairing P100 with S900.

**Expected:** The module already computes index_like=True for this pairing. That fact must reach the user — at minimum a warning such as 'both of these columns are plain 1..N row numbers, not identifiers; joining on them pairs rows by position'. JoinDiagnosis has no index_like field and diagnose_join never mentions it.

**Actual:** confidence='high', zero warnings, zero notes; the pairing is the preselected and only option, and the resulting table fuses two unrelated cohorts row by row.

**Root cause:** KeyCandidate.confidence disables the index-like penalty whenever name_similarity >= 0.85 (lines 182-183 and 189-191). Two files exported from Excel both calling their counter 'row'/'id'/'index' therefore score name_similarity=1.0, bypass the guard entirely, and are rated 'high'. Nothing in diagnose_join/plain_summary/JoinDiagnosis ever mentions that the key is a row counter, so the detection is computed and then thrown away.

**Impact:** Unrelated participants are merged into single rows and the app says the match is 100% on both sides. Every association computed afterwards is between measurements from different people.

```
suggest_best (low withheld): KeyCandidate(left_col='row', right_col='row', coverage_left=1.0, coverage_right=1.0, n_matched=50, ... name_similarity=1.0, index_like=True)
  candidate row <-> row: confidence=high index_like=True cov=(1.00,1.00)

SELECTBOX: ['row ↔ row'] DEFAULT = row ↔ row
MD      | Result: **50 rows** — matching on 50 shared IDs, keeping only IDs found in both files.
buttons: ['← Home', 'EDA →', 'Remove', 'Remove', 'Combine files', ...]
DELIVERED: (50, 5)
  row participant  systolic subject  cholesterol
0   1        P100       137    S900         5.17
1   2        P101       143    S901         4.56
2   3        P102       112    S902         6.70
3   4        P103       108    S903         5.92
4   5        P104       109    S904         3.70
TRUTH: these two files describe DIFFERENT people (P1xx vs S9xx); they share nothing but a row counter.
```

---

### `same-filename-dead-end` — Two uploaded files with the same name silently collapse to one and Step 2 dead-ends with a false explanation

**major** · join · `pages/01_Upload_and_Audit.py:501`

**What the researcher sees:** Two datasets both named 'data' are in the registry with their frames intact, but the page stops with 'You have 1 of your 2 files are no longer loaded. Re-upload them above, or remove their records, to continue.' Step 2 never renders, and re-uploading a same-named file cannot fix it.

**Expected:** Both files are offered in Step 2 (disambiguated, e.g. 'data' and 'data (2)'), or the message names the real cause: two files share a name.

**Actual:** The frames dict is keyed by dataset name so the second frame overwrites the first; the length check then misreports the loss as 'no longer loaded' and st.stop()s. The suggested remedy (re-upload) reproduces the collision, so the user cannot progress without renaming a file outside the app.

**Root cause:** `dataframes[d['name']] = _tmp` (line 501) builds the combine input keyed by the user-visible dataset name, which is not unique; the guard at line 503 infers 'not loaded' from the resulting length mismatch. The same name is also what becomes the join suffix downstream.

**Impact:** A researcher with e.g. 'export.csv' from two study folders is locked out of the entire multi-file workflow and is told something untrue about why.

```
both datasets are in the registry: [1, 2] -> rows [2, 2]
errors shown: ['1 of your 2 files are no longer loaded. Re-upload them above, or remove their records, to continue.']
headers reached: ['Step 1: Add Your Data']
Step 2 rendered: False
```

---

### `stack-overwrites-user-source-file-column` — execute_stack silently overwrites a user column named __source_file, and plan.summary() promises one more column than it delivers

**major** · stack · `utils/combine.py:151`

**What the researcher sees:** Stacking two cycles whose files already contain a column called '__source_file' (a perfectly legal column name) reports 'Result: 4 rows and 4 columns … plus one column recording which file each row came from', then delivers a 3-column table in which the user's own __source_file values (recall_day1/recall_day2) have been replaced by the file names.

**Expected:** Either the collision is reported as a blocking/warning finding before stacking ('one of your files already has a column called __source_file'), or the bookkeeping column is given a non-colliding name. And plan.summary()'s column count must equal the delivered frame's column count.

**Actual:** No blocking, no warning, no note. The user's data column is overwritten in place and the promised '4 columns' is delivered as 3.

**Root cause:** execute_stack does `part[SOURCE_COLUMN] = n` (line 151) with no check that SOURCE_COLUMN is absent from the frame, and StackPlan.summary() computes the column count as `len(self.all_columns) + 1` (line 54), which double-counts when __source_file is already in all_columns. plan_stack() never inspects SOURCE_COLUMN at all.

**Impact:** A user variable is destroyed with no record, and the promise shown above the button disagrees with the table produced — both direct violations of 'never silently guess … record permanently'.

```
plan.summary() : Result: **4 rows** and 4 columns, stacking 2 files that share 3 column(s) — plus one column recording which file each row came from.
blocking/warnings/notes: [] [] []
delivered shape: (4, 3) columns: ['SEQN', 'kcal', '__source_file']
   SEQN  kcal __source_file
0     1  1800    cycle_1999
1     2  2000    cycle_1999
2     3  1900    cycle_2001
3     4  2100    cycle_2001
original __source_file values were: ['recall_day1', 'recall_day2', 'recall_day1', 'recall_day2']
```

---

### `stack-tz-datetime-conflict-invisible` — tz-aware and tz-naive date columns stack with no warning into an object column that every date operation refuses

**major** · stack · `/home/user/tabular-ml-lab/utils/combine.py:64`

**What the researcher sees:** Two cycles whose visit_date is datetime64[us] in one and datetime64[us, UTC] in the other stack with plan.warnings=[] and plan.type_conflicts={}. The delivered visit_date is object dtype; max(), sort_values(), .dt.year and any comparison against a Timestamp all raise.

**Expected:** _dtype_family must distinguish tz-aware from tz-naive datetimes (they do not concat to a datetime dtype), so plan_stack flags the conflict before the button and can offer to localise or drop the timezone.

**Actual:** _dtype_family returns 'date' for both because is_datetime64_any_dtype is True for tz-aware and naive alike, so no conflict is recorded. pd.concat then falls back to object, and the column is no longer a date to pandas or to the app.

**Root cause:** `if pd.api.types.is_datetime64_any_dtype(s): return "date"` at combine.py:64 collapses tz-aware and tz-naive into one family; the check needs to include the tz in the family key (e.g. 'date' vs 'date[UTC]').

**Impact:** A date column that worked in each cycle stops being a date the moment the cycles are combined, with nothing said before or after the combine. Any time-based analysis (visit ordering, follow-up windows, seasonality) crashes or, if the researcher works around it, is silently done on strings.

```
cycle_2020 visit_date dtype: datetime64[us] -> family date
cycle_2021 visit_date dtype: datetime64[us, UTC] -> family date
plan.summary()  : Result: **6 rows** and 4 columns, stacking 2 files that share 3 column(s) — plus one column recording which file each row came from.
plan.warnings   : []
plan.type_conflicts: {}
delivered dtype : object | is_datetime64_any_dtype: False
  visit_date.max() -> TypeError: Cannot compare tz-naive and tz-aware timestamps
  sort_values('visit_date') -> TypeError: Cannot compare tz-naive and tz-aware timestamps
  visit_date.dt.year -> AttributeError: Can only use .dt accessor with datetimelike values
  visit_date > Timestamp('2020-07-01') -> TypeError: Cannot compare tz-naive and tz-aware timestamps
```

---

### `stack-case-variant-column-names-split` — Cycles whose headers differ only by capitalisation or a trailing space are stacked into two half-empty columns, and nothing says they are the same variable

**major** · stack · `/home/user/tabular-ml-lab/utils/combine.py:87`

**What the researcher sees:** Cycle 1 has ['SEQN','age','glucose'], cycle 2 has ['SEQN','Age','Glucose '] — a completely ordinary difference between two registry exports. The stack delivers 6 columns where age, Age, glucose and Glucose are each exactly 50% missing. out['age'].mean() = 48.02 is computed on 50 of 100 rows. The only message is 'Only 1 of 5 columns appear in every file (20%). The rest will be blank for the files that lack them.'

**Expected:** plan_stack should compare column names case-insensitively and whitespace-insensitively (as ml/join_doctor.py already does for keys via _name_similarity/normalize_key) and offer a reversible 'these are the same column' repair, exactly as the join path offers key repair. At minimum the warning should name the near-duplicate pairs rather than reporting them as four unrelated missing columns.

**Actual:** Column matching is an exact-string set intersection, so 'age'/'Age' and 'glucose'/'Glucose ' are treated as four distinct variables. The result is a table where every measurement is 50% missing, presented as a normal partial-overlap warning.

**Root cause:** `shared &= set(col_sets[n])` at combine.py:87 (and the union build at :88-90) uses raw column labels with no normalisation; combine.py has no equivalent of join_doctor's _name_similarity.

**Impact:** Every variable ends up half missing. If the researcher then imputes (which the app's Preprocess page offers), half of every column is fabricated from the other half of a different cycle. relationship_hint also says 'link' here, compounding the wrong steer.

```
cycle 1 columns: ['SEQN', 'age', 'glucose']
cycle 2 columns: ['SEQN', 'Age', 'Glucose ']
relationship_hint: link  <- steered to the JOIN screen
summary  : Result: **100 rows** and 6 columns, stacking 2 files that share 1 column(s) — plus one column recording which file each row came from.
warnings : ['Only 1 of 5 columns appear in every file (20%). The rest will be blank for the files that lack them.']
partials : {'age': ['NHANES_2001_2002'], 'glucose': ['NHANES_2001_2002'], 'Age': ['NHANES_1999_2000'], 'Glucose ': ['NHANES_1999_2000']}
delivered shape: (100, 6)
missing fraction per column after stacking:
SEQN             0.0
age              0.5
glucose          0.5
__source_file    0.0
Age              0.5
Glucose          0.5
out['age'].mean() = 48.02 computed on 50 of 100 rows
nothing in the plan says 'age' and 'Age' are the same variable.
```

---

### `grouped-lockbox-fraction-mislabel` — "Held-out test fraction 15%" is applied to subjects but reported as a fraction of rows — the real held-out share was 37%

**major** · state · `utils/test_lockbox.py:159`

**What the researcher sees:** A 20-participant food-diary study (135 rows; one participant logged 40 days, the rest 5). The researcher leaves the "Held-out test fraction" slider at 0.15. The chip says "🔒 Test set: 15% (n=50 rows from 3 subjects, split by 'SubjectID' ...)". 50/135 is 37.0% of the data. In the mirror case (30 subjects, one with 400 rows) the same 15% setting quarantines 3.9% of rows.

**Expected:** Either report the realised row fraction ("Test set: 37% of rows — 50 rows from 3 of 20 subjects"), or state plainly that the fraction is a fraction of subjects. The number next to "Test set:" should describe what was actually held out.

**Actual:** The requested fraction is echoed verbatim as if it were the realised row fraction. sklearn's GroupShuffleSplit(test_size=fraction) selects a fraction of GROUPS; with uneven fan-out — which is exactly what a 1:many or many:many merge produces — the row share drifts far from it (measured 37.0% and 3.9% against a requested 15%).

**Root cause:** utils/test_lockbox.py:159 `GroupShuffleSplit(n_splits=1, test_size=fraction, ...)` interprets `fraction` as a proportion of groups, but the lockbox dict stores that same value under `fraction` (utils/test_lockbox.py:180) and every consumer prints it as a row share: the grouped chip at utils/test_lockbox.py:232-236, the non-grouped chip at :239, and the insight-ledger entry written into the report at pages/01_Upload_and_Audit.py:1043 ("A {fraction:.0%} test set (n={n_test}) was held out at upload"). The realised share `n_test / n_total` is available but never used.

**Impact:** The researcher believes 15% of their data is quarantined. In the skewed-fan-out direction 37% of their rows are removed from training (a real loss of power they never chose); in the other direction the "held-out" evaluation rests on 20 of 516 rows, and the confidence intervals reported from it are far wider than the 15% they budgeted for. The wrong number is carried into the exported manuscript.

```
rows=135 subjects=20
LOCKBOX: {'fraction': 0.15, 'n_total': 135, 'n_test': 50, 'group_col': 'SubjectID', 'n_test_groups': 3}
ACTUAL held-out ROW fraction: 37.0%
CHIP : 🔒 Test set: 15% (n=50 rows from 3 subjects, split by 'SubjectID' so no subject appears on both sides) held out since upload — opened once at Train & Compare.
INFO : Rows repeat per subject (`SubjectID`), so the held-out set was drawn by **subject**, not by row — 50 rows from 3 subjects. Splitting by row would put the same person in both training and testing.
SLIDER: Held-out test fraction = 0.15

--- opposite skew (r9_leak_variants.py case 4: 30 subjects, one with 400 rows) ---
    group_col -> 'SEQN' ; n_test=20 ; n_total=516 ; row-fraction=3.9%
```

---

### `join-key-offered-as-predictor` — After a merge the subject-ID join key is auto-selected as a model feature

**major** · state · `pages/01_Upload_and_Audit.py:851`

**What the researcher sees:** Merge demographics + labs on MRN, pick `outcome` as target: the saved feature set is ['age', 'glucose', 'MRN']. The identifier the two files were joined on is silently in the model.

**Expected:** A column the app itself just used as the join key — and, when it is detected, as the lockbox group column — must not be offered as a predictor by default, exactly as `__source_file` is excluded. utils/combine.reserved_columns() is the existing mechanism for this.

**Actual:** reserved_columns() returns only ['__source_file'], so the join key survives into `feature_options` and, because the default is "all candidate features", into the saved DataConfig with no warning.

**Root cause:** pages/01_Upload_and_Audit.py:849-851 filters `feature_options` against `utils.combine.reserved_columns()`, which (utils/combine.py:184-186) lists only SOURCE_COLUMN. Neither the join key recorded by execute_join nor the lockbox's `group_col` is added to the reserved set, and pages/01_Upload_and_Audit.py:898 defaults the multiselect to every remaining column.

**Impact:** With any of the subject-leakage findings above in play, an integer/one-hot subject ID in the feature matrix turns a partial leak into complete memorisation — the model learns subject→outcome and reports near-perfect held-out performance. Even with a correct grouped split it wastes a feature slot, distorts SHAP/importance rankings, and produces a manuscript claiming "medical record number" as a predictor of the outcome.

```
DELIVERED: (200, 4) | distinct MRN: 100
LOCKBOX: {'n_total': 200, 'n_test': 30, 'group_col': None, 'n_test_groups': None}
SUBJECTS IN BOTH TRAIN AND HELD-OUT TEST: 24 of 100
default features: ['age', 'glucose', 'MRN']

--- and with a working grouped lockbox (r7_full_page_fanout_to_lockbox.py) ---
  working_table: (200, 4) | distinct SEQN: 100
  lockbox: {'n_total': 200, 'n_test': 30, 'group_col': 'SEQN', 'n_test_groups': 15, 'stratified': False}
  default features: ['age', 'glucose', 'SEQN']
```

---

### `no-way-to-declare-subject-column` — There is no control anywhere for declaring the subject/entity column — the declared-entity code paths in the lockbox and in Train & Compare are dead

**major** · state · `pages/01_Upload_and_Audit.py:1023`

**What the researcher sees:** pages/01_Upload_and_Audit.py:1020-1024 comments "A declared subject/entity ID always wins over auto-detection" and reads `_cohort.entity_id_final`. Nothing in the codebase ever assigns `cohort_structure_detection`, `entity_id_detected` or `entity_id_override_value` to a real value, and `ml.triage.detect_cohort_structure` is never called. After the full page-01 flow entity_id_final is None and no widget on the page offers a subject/entity/group column.

**Expected:** After the Join Doctor announces "models will need a group-aware split", the researcher should be able to say which column is the subject — and that declaration should drive both the lockbox and the Train & Compare split. Both consumers are already written and waiting for it.

**Actual:** `_entity_col` at pages/01_Upload_and_Audit.py:1023 is unconditionally None, so ensure_lockbox always falls back to the name+ratio heuristic. pages/06_Train_and_Compare.py:174 (`if cohort_type_final == 'longitudinal' and entity_id_final`) and :493 (`if use_group_split and entity_id_final`) can never be true, so the GroupShuffleSplit path in Train & Compare is unreachable too. There is no second line of defence anywhere.

**Root cause:** pages/01_Upload_and_Audit.py imports detect_cohort_structure at line 38 but never calls it, and renders no entity-ID selector. utils/session_state.py:126/405 only ever construct an empty CohortStructureDetection(); utils/state_reconcile.py:47-48 only ever clears it. Consequently the `group_col` argument of utils/test_lockbox.ensure_lockbox is only ever supplied by the heuristic at utils/test_lockbox.py:127-131, and the entire longitudinal branch of pages/06_Train_and_Compare.py:174-200 and :493-500 is dead code.

**Impact:** When the heuristic misses (MRN-named keys, partial repeats, heavy fan-out, <8 subjects — see the two leakage findings), the researcher has no way to correct it. They are shown a warning that their data needs a group-aware split, given no control to enable one, and then handed a 🔒 badge on a test set that is not subject-separated. The failure is both silent and unrecoverable from the UI.

```
--- every mention of detect_cohort_structure / entity_id in the repo (assignments) ---
/home/user/tabular-ml-lab/ml/triage.py:102:def detect_cohort_structure(df: pd.DataFrame, sample_size: int = 1000) -> Dict:
/home/user/tabular-ml-lab/utils/state_reconcile.py:47:            cohort_detection.entity_id_detected = None
/home/user/tabular-ml-lab/utils/state_reconcile.py:48:            cohort_detection.entity_id_override_value = None
/home/user/tabular-ml-lab/utils/session_state.py:405:    st.session_state.cohort_structure_detection = CohortStructureDetection()

cohort_structure_detection after the full page-01 flow:
   detected = None | entity_id_detected = None | entity_id_final = None
widgets on page 01 offering a subject/entity/group column: NONE
```

---

### `case-whitespace-column-collision-unreported` — Column names that differ only by case or trailing whitespace are never flagged, so the merged table carries two columns that look identical

**minor** · join · `ml/join_doctor.py:401`

**What the researcher sees:** Joining a file with columns BMI and 'age ' (trailing space) to one with bmi and age produces a table with all four columns and no warning. Two of them display as 'age'.

**Expected:** diagnose_join notes that 'BMI'/'bmi' and 'age '/'age' are near-duplicates from the two files, the way it does for exact-name collisions, so the user can tell which column is which.

**Actual:** column_collisions is empty; both pairs pass through unremarked and two columns render with the same visible label in every table, feature picker and export.

**Root cause:** The collision set on lines 401-402 is an exact string intersection of the two column indexes; it has no case- or whitespace-insensitive comparison, unlike the key-value canonicalisation the module applies everywhere else.

**Impact:** In feature selection and EDA the researcher sees two columns labelled 'age' with different values and no way to tell which file each came from; picking the wrong one is silent.

```
column_collisions: []
warnings         : []
delivered columns: ["'id'", "'BMI'", "'age '", "'bmi'", "'age'"]
  id   BMI  age    bmi  age
0  1  22.0    30  99.0    1
1  2  27.0    40  98.0    2
```

---

### `eager-join-before-consent` — The merge is executed in full on every rerun, before the button is pressed, while the screen says "Nothing has changed yet"

**minor** · perf · `utils/combine_ui.py:129`

**What the researcher sees:** Two 2,000-row × 16-column files sharing a repeated sentinel ID (9999 = 'refused', as survey exports code it) fan out to 1,001,000 rows. The page builds that frame during the initial render — no button pressed, working_table not committed — and rebuilds it every time any widget on the screen changes. Peak RSS 491 MB.

**Expected:** diagnose_join already predicts the row count exactly (verified across 1,600 fuzzed configurations in r12 — zero mismatches), so the preview needs no materialised result. The product should be built only when the researcher presses "Combine files".

**Actual:** utils/combine_ui.py:129 calls execute_join unconditionally during rendering for every attachment in the chain; the button at :235 only decides whether the already-built frame is returned. Cost scales with the fan-out product and is paid again on every rerun — every radio change, every file-preview toggle.

**Root cause:** _render_link (utils/combine_ui.py:126-136) executes the join to produce `result` before render_combine_step (utils/combine_ui.py:235) asks for consent. In a k-file chain all k-1 merges run per rerun, so a compounding fan-out is materialised repeatedly. There is no guard on the predicted row count even though diagnose_join has already computed it and warned that it "is usually a mistake".

**Impact:** On an accidental many:many the researcher's browser stalls and memory balloons while the screen insists nothing has happened — and it happens again on each click as they try to change the join mode to fix it. The contract "never silently guess — propose reversibly" is broken in spirit: the expensive irreversible-feeling work is done before the user agrees to it.

```
inputs: 2 files of 2,000 rows x 16 columns (ID 9999 = 'refused' appears 1,000x in both)
page rendered in 1.64 s ; peak RSS 491 MB
execute_join calls made WITHOUT any button press: 1
   merge(2000 x 2000) -> 1001000 rows in 0.17 s
working_table committed? False
SCREEN SAYS: Nothing has changed yet — press **Combine files** when the result above looks right.
changing ONE radio option re-ran the merge 1 time(s) in 0.49 s

--- same shape, no width (r8_eager_execution.py) ---
WARNING: Both files have several rows per ID, so every combination is produced: 1,501 shared IDs become 2,251,500 rows. This is usually a mistake — check whether one file should be summarised to one row per subject first.
PROMISE: Result: **2,251,500 rows** — matching on 1,501 shared IDs, keeping only IDs found in both files.
page render (NO button pressed) took 1.3 s ; peak RSS 323 MB
```

---

### `stack-blank-file-blocks-with-wrong-advice` — A single zero-column file blocks the entire stack and the message points the researcher at joining

**minor** · stack · `/home/user/tabular-ml-lab/utils/combine.py:97`

**What the researcher sees:** Three cycles where one file loaded with no columns at all. plan_stack blocks everything with '🛑 These files have no column names in common, so stacking them would produce one block of data per file with nothing lining up. Check whether you meant to link them by a shared ID instead.' — even though two of the three cycles line up perfectly. The researcher is told the wrong cause and pointed at the wrong operation.

**Expected:** Name the offending file: 'NHANES_2001_2002 has no columns — remove it or re-upload it, then the other 2 files stack cleanly.' The blocker should not be phrased as advice to join.

**Actual:** Because the shared-column set is intersected across all files, one empty file empties the intersection and triggers the generic 'no column names in common' blocker, which recommends linking by a shared ID.

**Root cause:** plan_stack intersects col_sets over all names (combine.py:85-90) with no special case for a frame that contributes zero columns, then emits the generic blocker at combine.py:96-101.

**Impact:** A dead end with misleading advice on a recoverable situation; the researcher's remaining good cycles cannot be combined until they work out which file is at fault, and the app has pointed them at the wrong screen.

```
R09 — one blank file among three good cycles
  can_proceed: False
  🛑 These files have no column names in common, so stacking them would produce one block of data per file with nothing lining up. Check whether you meant to link them by a shared ID instead.
```

---

### `stack-duplicate-column-label-crash` — A duplicated column label passes plan_stack (with a fabricated type conflict) and then crashes execute_stack with a raw pandas error

**minor** · stack · `/home/user/tabular-ml-lab/utils/combine.py:154`

**What the researcher sees:** An Excel file whose header row contains both the number 1 and the text '1' is loaded as distinct labels [1, '1'] and then stringified to ['1','1'] by the upload page. plan_stack reports can_proceed=True, promises '4 rows and 3 columns', and invents a type conflict ("'1' is number in some files and text"). Because utils/combine_ui.py:171 calls execute_stack while merely rendering the screen — before any button is pressed — the page dies with pandas.errors.InvalidIndexError.

**Expected:** plan_stack should detect duplicated labels within any input frame and block with a plain-language message pointing at the Import Doctor's existing 'Make duplicate names unique' fix, instead of promising a shape and then throwing.

**Actual:** plan_stack dedupes labels when building `union`, so nothing notices the duplicate; frames[n][c] returns a DataFrame for the duplicated label, which _dtype_family reports as 'text', producing a nonsense conflict. execute_stack's pd.concat then raises.

**Root cause:** combine.py:83-116 assumes unique column labels (union dedupes; frames[n][c] may be a DataFrame), and combine.py:154 concats without checking `df.columns.is_unique`. The duplicate is created at pages/01_Upload_and_Audit.py:500 (`_tmp.columns = [str(c) for c in _tmp.columns]`), after the Import Doctor's duplicate check has already passed on the un-stringified labels.

**Impact:** The Step 2 screen crashes with a pandas traceback the moment it renders, with no way forward from inside the app — a visible failure, but one the app claimed was a clean 3-column stack one line earlier.

```
R10 — duplicated column label (Excel headers 1 and '1', stringified at pages/01_Upload_and_Audit.py:500)
  can_proceed: True | summary: Result: **4 rows** and 3 columns, stacking 2 files that share 2 column(s) — plus one column recording which file each row came from.
  type_conflicts (bogus): {'1': ['number', 'text']}
  execute_stack -> InvalidIndexError: Reindexing only valid with uniquely valued Index objects

(reachability, verified separately:)
raw cols: [1, '1', 'age'] ['int', 'str', 'str']
after page-01 str cast: ['1', '1', 'age'] dup? True
```

---

### `stack-bool-int-warning-is-false` — bool vs 0/1 int is warned about as unusable text when the stacked result is a clean int64 column

**minor** · stack · `/home/user/tabular-ml-lab/utils/combine.py:134`

**What the researcher sees:** A yes/no column stored as True/False in one cycle and 1/0 in the next raises the warning "1 column(s) hold different kinds of value in different files ('diabetic' is number in some files and true/false). After stacking they become text, which no model can use until it is cleaned up." The delivered column is int64 with values [1, 0, 1, 0] — perfectly modellable.

**Expected:** The warning should predict the dtype the concat will actually produce. number+true/false resolves to int64 and deserves at most an informational note, not a claim that the column becomes unusable text.

**Actual:** The warning text is a fixed string appended for any type_conflicts entry, regardless of which families are in conflict, so it asserts an outcome that does not occur for bool/number pairs.

**Root cause:** combine.py:132-136 emits one hard-coded consequence ('After stacking they become text') for every entry in plan.type_conflicts, without checking whether the specific family pair actually widens to object.

**Impact:** A false alarm on a benign, common encoding difference. In an app whose contract is 'diagnose visibly', warnings that turn out to be wrong train researchers to ignore the warnings that are right — the genuine number/text conflict uses the identical wording.

```
R11 — bool vs 0/1 int: warning says 'become text', result is int64
  ⚠️ 1 column(s) hold different kinds of value in different files ('diabetic' is number in some files and true/false). After stacking they become text, which no model can use until it is cleaned up.
  actual dtype: int64 values: [1, 0, 1, 0]
```

---

### `stack-int-float-id-precision` — An integer subject-ID column stacked with a float one is silently widened to float, changing the ID values

**minor** · stack · `/home/user/tabular-ml-lab/utils/combine.py:60`

**What the researcher sees:** SEQN is int64 in one cycle and float64 in another (which happens as soon as one cycle has a blank ID). plan_stack reports no conflict — both are the 'number' family. IDs 9007199254740993 and 9007199254740995 come out of the stack as 9007199254740992.0 and 9007199254740996.0; in the ordinary case IDs 12345, 12346 become 12345.0, 12346.0.

**Expected:** The same guarantee ml/join_doctor.py already states for keys ('passing IDs through float64 silently collides values above 2^53 … which is a false merge — the worst outcome this module can produce', join_doctor.py:58-62). Stacking should warn when an integer column is about to be widened to float, and preserve exact integer IDs.

**Actual:** _dtype_family maps int64 and float64 to the same 'number' family, so no conflict is reported and pd.concat upcasts to float64. Above 2^53 the digits change; below it, IDs acquire a '.0' that is carried into every export and any later re-join.

**Root cause:** combine.py:59-61 returns 'number' for every numeric dtype, so int-to-float widening in pd.concat (combine.py:154) is invisible to the plan. join_doctor guards this explicitly for join keys; combine.py has no equivalent.

**Impact:** Subject identifiers are altered by an operation the app promised was a pure end-to-end append. Downstream, IDs no longer match the source files, and above 2^53 the printed ID is simply a different number from the one in the researcher's data.

```
R12 — int64 IDs stacked with float IDs: same 'number' family, no warning
  type_conflicts: {} warnings: []
  SEQN before: [9007199254740993, 9007199254740995]
  SEQN after : [9007199254740992.0, 9007199254740996.0]  dtype float64
  two distinct subject IDs are now different
  ordinary case: SEQN [12345, 12346] -> [12345.0, 12346.0] dtype float64
```

---

### `stack-ordered-categorical-loses-ordering` — Stacking ordered categoricals whose category order differs silently drops the ordering and flips ordinal comparisons

**minor** · stack · `/home/user/tabular-ml-lab/utils/combine.py:66`

**What the researcher sees:** Two parquet cycles carry an ordered food_security factor, one with categories low<marginal<high and the other high<marginal<low. plan_stack reports no conflict and no warning. The stacked column comes back as plain str, and the comparison food_security > 'low' on the very same rows changes from [False, True, True] to [False, False, True] — 'high' > 'low' answers True before stacking and False after.

**Expected:** Ordered categoricals with incompatible category orders are a genuine type conflict and must be reported before the button, with the choice of a common ordering. When the orders agree the categorical dtype survives (verified as a control in p12_categorical_and_zerocol.py), so the disagreement is detectable.

**Actual:** _dtype_family returns 'text' for every categorical, so two ordered factors with contradictory orderings register as identical families. pd.concat falls back to str and every ordinal comparison silently switches to lexicographic string ordering.

**Root cause:** The catch-all `return "text"` at combine.py:66 covers all categorical dtypes without inspecting .cat.ordered or .cat.categories; combine.py:154 then concats them. Reachable through the app's parquet upload path (verified end to end via data_processor.load_tabular_data in the repro).

**Impact:** An ordinal exposure such as food-security level silently stops being ordinal, and any cut-point, comparison or ordered model built on it after combining answers differently from before — with no warning at any stage.

```
cycle 1 as uploaded: category ordered= True ['low', 'marginal', 'high']
cycle 2 as uploaded: category ordered= True ['high', 'marginal', 'low']
type_conflicts: {}  warnings: []  notes: []
delivered dtype: str
BEFORE stacking, (food_security > 'low') on cycle 1 = [False, True, True]
AFTER  stacking, (food_security > 'low') on cycle 1's rows = [False, False, True]
'high' > 'low' answered True before stacking and False after.
```

---
