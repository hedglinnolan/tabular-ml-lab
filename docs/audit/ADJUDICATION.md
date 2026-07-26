# Adjudication of the original 48, against the frozen tree

Run `wf_5446c57b-3f6`. Every finding re-run against `413671a` — the tree BEFORE
this branch's fixes — so a "still_broken" verdict here describes the pre-fix
code. Check `docs/FINDINGS_LEDGER.md` for what has since been closed.

The adversarial recheck pass did not run (session limit), so these are single
verdicts, not consensus ones. Treat them as leads, not proof.

| # | Verdict | Severity | Title |
|---|---------|----------|-------|
| 4 | still_broken | major | join_doctor labels category/date/period/duration keys as "text", blocking working categorical-vs-numeric joins |
| 5 | still_broken | major | join_doctor: datetime keys are compared by their stringified form, so a tz-aware/naive (or cross-timezone) pai |
| 11 | still_broken | major | MultiIndex (two-row header) columns: diagnose_join green-lights a join that execute_join always crashes on |
| 20 | still_broken | major | execute_join drops right-only row identifiers on right/outer joins when key columns have different names |
| 23 | still_broken | minor | diagnose_join returns can_proceed=True for a predicted 25,000,000-row many-to-many blow-up, and execute_join h |
| 27 | still_broken | major | check_numeric_stored_as_text silently skips text columns whose values are all plain numbers (the `raw_numeric  |
| 32 | still_broken | major | check_header_in_later_row false-positives on clean narrow frames with a blank header cell: emits the single cr |
| 39 | still_broken | critical | Row-counter columns are rated "high" confidence and proposed as the best join key whenever both files use the  |
| 40 | still_broken | major | diagnose_join: dtype-mismatch blocker fires ahead of the zero-overlap check, producing "Fixing this matches 0  |
| 41 | still_broken | major | diagnose_join emits no warning about rows that will be blank-filled on left/right/outer joins |
| 42 | still_broken | major | diagnose_join suppresses genuine column collisions whenever a key name also exists in the other frame (cross-n |
| 44 | changed_form | critical | 'Convert to numbers' strips the decimal comma, silently rescaling European-format numeric columns — at 'high'  |
| 7 | partially_fixed | minor | diagnose_join green-lights date-vs-text key pairs that execute_join(repair=False) cannot merge |
| 8 | partially_fixed | major | find_key_candidates draws an independent 5,000-row sample per file, so value overlap is measured between unrel |
| 9 | partially_fixed | minor | Duplicate key column name makes every ml/join_doctor.py entry point raise AttributeError and makes find_key_ca |
| 13 | partially_fixed | minor | Duplicate column labels crash three checks; diagnose() silently swallows the crashes and drops unrelated findi |
| 18 | partially_fixed | major | Large files: independent per-side row sampling in `_prep` destroys measured key overlap - real overlaps are re |
| 22 | partially_fixed | minor | Blank/NaN join keys are invisible to diagnose_join: inner join silently drops those rows with no warning, and  |
| 24 | partially_fixed | major | find_key_candidates scores keys on two independent 5,000-row samples, so overlap counts and coverage percentag |
| 26 | partially_fixed | major | NUMERIC_SENTINELS omits the positive NHANES/SPSS 7/8/9, 77/88/99 and 7777/8888 missing-code families, so narro |
| 28 | partially_fixed | minor | Key detection collapses above ~10,000 rows: `_prep` samples both sides independently, so the true key's measur |
| 35 | partially_fixed | major | coerce_numeric silently merges incompatible units (mg/dL + mmol/L, kg + lb) into one column at 'high' confiden |
| 47 | partially_fixed | minor | Duplicated key column name crashes normalize_key and therefore diagnose_join / execute_join / repair_keys (Att |
| 48 | partially_fixed | major | find_key_candidates samples each side positionally (5,000 rows, random_state=42), so on files over 5,000 rows  |
| 1 | fixed | none | Null join keys are matched to each other in execute_join, fabricating subjects, while diagnose_join predicts a |
| 2 | fixed | none | normalize_key blanks unparseable IDs in a >=95%-numeric text key column; repair_keys turns the blanks into NaN |
| 3 | fixed | none | diagnose_join counts missing values as a shared key for text/date columns, inflating matched_keys and emitting |
| 6 | fixed | none | Blank/NaN join keys are cross-joined into fabricated rows; diagnose_join's predicted row count is wrong and no |
| 10 | fixed | none | drop_rows fix deletes rows by label, destroying unrelated rows on non-unique-index frames while reporting the  |
| 12 | fixed | none | FALSE MERGE: non-numeric IDs in a mostly-numeric key column are blanked to NaN by repair_keys, then cross-join |
| 14 | fixed | none | Unconditional case-folding in normalize_key silently merges distinct case-sensitive IDs, and diagnose_join mis |
| 15 | fixed | none | FALSE MERGE: normalize_key coerces large numeric-looking IDs through float64, fabricating and collapsing keys  |
| 16 | fixed | none | Missing join keys are cross-joined into fabricated rows and counted as a matched ID; predicted row count is wr |
| 17 | fixed | none | diagnose_join under-predicts row count whenever key values are missing/blank (not just when a key column is en |
| 19 | fixed | none | Rows with a missing/blank join key are cross-joined into fabricated rows; predicted_rows is wrong by 5-25x and |
| 21 | fixed | none | diagnose_join ignores rows with a blank key: predicted_rows under-counts left/right/outer joins (and misreport |
| 25 | fixed | none | Blank/missing join keys: diagnose_join predicts rows as if blanks are dropped, execute_join merges NaN-to-NaN  |
| 29 | fixed | none | check_numeric_stored_as_text coerces a mixed-unit column onto one numeric scale at 'high' confidence, never na |
| 30 | fixed | none | check_text_missing_tokens marks 'none'/'unknown'/'not applicable' as high-confidence, auto-suggestable missing |
| 31 | fixed | none | melt_repeated overwrites an existing 'measurement' column and emits duplicate column names (and crashes outrig |
| 33 | fixed | none | apply_fix('melt_repeated') raises an unhandled ValueError when the frame already has a column named 'value' (a |
| 34 | fixed | none | 'none'/'unknown' recoded to missing at HIGH (auto-suggestable) confidence, destroying legitimate categorical l |
| 36 | fixed | none | coerce_numeric's methods-section description omits how many values it blanked (up to 20% of a column) |
| 37 | fixed | none | diagnose_join counts blank/NaN IDs as a shared key: predicts 2 rows, execute_join returns 8 with 6 fabricated  |
| 38 | fixed | none | diagnose_join drops missing/blank key rows from its row prediction, so predicted < actual for left/right/outer |
| 43 | fixed | none | Missing/blank join keys are matched to each other, fabricating participants; predicted_rows disagrees with the |
| 45 | fixed | none | join_doctor.normalize_key treats text missing-codes ('unknown', 'missing', '.', 'NA') as real shared IDs, and  |
| 46 | fixed | none | 'Convert to numbers' strips trailing letters off alphanumeric IDs, silently collapsing distinct participants — |

---

## Detail for everything not `fixed`

### 4 — join_doctor labels category/date/period/duration keys as "text", blocking working categorical-vs-numeric joins and emitting a self-contradictory message

**still_broken** · severity major

**What remains:** Both defects reproduce essentially verbatim; nothing about this finding has been fixed. Defect 1: `pd.Categorical([1,2,3])` vs `int64` merges fine in plain pandas (3 rows) and diagnose_join itself agrees 3 IDs match, yet dtype_mismatch=True, can_proceed=False, and the blocking text calls a category of integers "stored as text". Defect 2: for the genuinely impossible datetime-vs-int join the single message still self-contradicts — "They look identical on screen but will not match. Fixing this matches 0 IDs." — and again mislabels a datetime64 column as "text". The check is still the crude `is_numeric_dtype(ls) != is_numeric_dtype(rs)` at ml/join_doctor.py line 374, and the message at lines 422-428 hardcodes the word "text" for anything non-numeric. Worse in practice than the original write-up: utils/combine_ui.py line 121 (`if not diag.can_proceed and not diag.dtype_mismatch`) deliberately lets a dtype_mismatch through, so the user is shown a red 🛑 stop error AND the line "This join will not work yet — see below" and then the join silently succeeds anyway (3 rows for D1) — the app asserts failure and then delivers a result, which is exactly the confidently-wrong-message failure mode the contract forbids. Two facets of the title HAVE improved and are worth recording: period-vs-period and duration(timedelta)-vs-duration keys now diagnose cleanly (dtype_mismatch False, 3/3 rows), as do category[str]-vs-object, datetime-vs-datetime, category[int]-vs-category[int] and Int64-vs-int64; the true positive str-vs-int still blocks correctly. The residue is specifically non-numeric-vs-numeric pairings (category-of-numbers, datetime, and by the same code path period/timedelta vs a numeric column).

**Suggested fix:** In diagnose_join (ml/join_doctor.py ~lines 374-376 and 422-428) replace the binary `is_numeric_dtype(ls) != is_numeric_dtype(rs)` test with a real name for each side (numbers / text / dates / categories / durations) and only treat it as a blocker when pandas would actually refuse the merge — i.e. when the underlying storage classes are incompatible, not merely when one side is Categorical. Then word the message from the measured overlap: say "look identical on screen" only when matched > 0, and when matched == 0 say the columns are different kinds of value and no ID lines up. Finally, make plain_summary agree with what combine_ui.py will actually do: a dtype_mismatch that the UI repairs and pushes through must not print "This join will not work yet".

```
=== Defect 1 - categorical-of-ints vs int ===
--- D1 category[int] vs int64 ---
  left dtype: category | right dtype: int64
  pandas merges fine -> 3 rows
  matched_keys: 3 dtype_mismatch: True can_proceed: False
  blocking: ["'cycle' is stored as text in the first file but as numbers in the second file. They look identical on screen but will not match. Fixing this matches 3 IDs."]
  summary: This join will not work yet — see below.
  'stored as text' in blocking[0]? True

=== Defect 2 - datetime vs int ===
--- D2 datetime64 vs int64 ---
  left dtype: datetime64[us] | right dtype: int64
  pandas RAISES: ValueError: You are trying to merge on datetime64[us] and int64 columns for key 'k'...
  matched_keys: 0 dtype_mismatch: True can_proceed: False
  blocking: ["'k' is stored as text in the first file but as numbers in the second file. They look identical on screen but will not match. Fixing this matches 0 IDs."]
  contains 'look identical on screen'? True
  contains 'matches 0 IDs'? True

(a04b.py — replaying utils/combine_ui.py lines 112-136 verbatim)
--- UI simulation: D1 category[int] vs int64 ---
  st.error   -> 🛑 'cycle' is stored as text in your data so far but as numbers in other. They look identical on screen but will not match. Fixing this matches 3 IDs.
  st.markdown -> This join will not work yet — see below.
  execute_join SUCCEEDED: 3 rows
  cycle  a  b
0     1  1  7
1     2  2  8
2     3  3  9

--- UI simulation: D2 datetime64 vs int64 ---
  st.error   -> 🛑 'k' is stored as text in your data so far but as numbers in other. They look identical on screen but will not match. Fixing this matches 0 IDs.
  st.markdown -> This join will not work yet — see below.
  execute_join SUCCEEDED: 0 rows
  Empty DataFrame
```

---

### 5 — join_doctor: datetime keys are compared by their stringified form, so a tz-aware/naive (or cross-timezone) pair is reported as "nothing to join on. Check you picked the right columns"

**still_broken** · severity major

**What remains:** Everything in the original finding, plus more. ml/join_doctor.py::_canon_scalar still does `s = str(v).strip()` for every non-int/non-decimal value, with no datetime branch at all, so Timestamps are compared as their repr. Case 1 (naive vs tz-aware) still emits the exact blocker quoted in the finding. Case 2 (both tz-aware, different zones, SAME instants) is worse than reported: plain pandas merges 3 rows, join_doctor blocks with 'nothing to join on', and execute_join with the DEFAULT repair=True silently returns 0 rows while repair=False returns 3 — i.e. the app's own repair destroys a join pandas gets right. Case 3 (plain strings) still blocks. Two regressions beyond the original: (a) the finding's own CONTROL — date objects vs datetime64 midnight — used to give blocking == [] and now blocks too (normL '2020-01-01' vs normR '2020-01-01 00:00:00'); (b) the commonest real trigger of all, date-only strings on one side and parse_dates-parsed datetimes on the other, also blocks. Via the UI this is a hard dead end: find_key_candidates returns [] so combine_ui shows 'has no column that lines up with your data, so it cannot be attached' and sets blocked=True; picking the columns by hand yields the same blocker. The app confidently asserts two files about the same visits cannot be linked.

**Suggested fix:** Give _canon_scalar a datetime branch before the str() fallback: for pd.Timestamp/datetime/np.datetime64, convert tz-aware values to UTC then drop the tzinfo, and emit a fixed ISO form (and for datetime.date, the same 'YYYY-MM-DD HH:MM:SS' shape with a zero time) so naive, tz-aware, cross-zone, date-object and date-string spellings of the same instant all canonicalise identically. Add a note to the diagnosis when a tz conversion was applied so the normalisation stays visible rather than silent.

```
$ python3 f05.py
pandas 3.0.3
C1 normL: ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-03 00:00:00']
C1 normR: ['2020-01-01 00:00:00+00:00', '2020-01-02 00:00:00+00:00', '2020-01-03 00:00:00+00:00']
C1 blocking: ["None of the values in 'visit_date' appear in 'visit_date', so there is nothing to join on. Check you picked the right columns."]
C1 matched_keys: 0 predicted: 0
C2 plain pandas merge rows: 3
C2 normL: ['2020-01-01 00:00:00+00:00', '2020-01-02 00:00:00+00:00', '2020-01-03 00:00:00+00:00']
C2 normR: ['2019-12-31 19:00:00-05:00', '2020-01-01 19:00:00-05:00', '2020-01-02 19:00:00-05:00']
C2 blocking: ["None of the values in 'visit_date' appear in 'visit_date', so there is nothing to join on. Check you picked the right columns."]
C2 execute repair=True rows: 0
C2 execute repair=False rows: 3
C3 blocking: ["None of the values in 'visit_date' appear in 'visit_date', so there is nothing to join on. Check you picked the right columns."]
C4 normL: ['2020-01-01', '2020-01-02', '2020-01-03']
C4 normR: ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-03 00:00:00']
C4 blocking: ["None of the values in 'd' appear in 'd', so there is nothing to join on. Check you picked the right columns."] matched: 0
C1 candidates: []

$ python3 f05b.py   (walks the exact utils/combine_ui.py::_render_link branch)
UI) candidates: []
UI) WARNING shown: No shared ID was found ... pick the columns yourself
UI) ERROR shown: <file> has no column that lines up with your data, so it cannot be attached.  -> blocked=True
UI) manual pick blocking: ["None of the values in 'visit_date' appear in 'visit_date', so there is nothing to join on. Check you picked the right columns."]
UI) dtype_mismatch: False -> UI blocks (can_proceed False, no dtype repair offer)
UI) plain_summary: This join will not work yet — see below.
truth (pandas after tz-normalising): 3
UI) execute_join(repair=True) rows: 0
CTRL) blocking: ["None of the values in 'd' appear in 'd', so there is nothing to join on. Check you picked the right columns."]
CTRL) matched: 0 predicted: 0
CTRL) candidates: []
PARSE) normL: ['2020-01-01', '2020-01-02', '2020-01-03'] normR: ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-03 00:00:00']
PARSE) blocking: ["None of the values in 'd' appear in 'd', so there is nothing to join on. Check you picked the right columns."] matched: 0
```

---

### 11 — MultiIndex (two-row header) columns: diagnose_join green-lights a join that execute_join always crashes on

**still_broken** · severity major

**What remains:** Reproduces exactly as recorded, on every branch. diagnose_join() still green-lights the join ("Result: **3 rows**", can_proceed=True, blocking=[], warnings=[]) and execute_join() then raises `ValueError: The column label 'key' is not unique.` for all four join types with repair both True and False — a promise the code can never keep. suggest_best() still returns left_col/right_col as the STRINGIFIED tuple "('key', 'SEQN')" (join_doctor.py:335 does `left_col=str(lc)`) at 'high' confidence — the only confidence pre-selected in the UI — and feeding that string straight back into diagnose_join raises `KeyError: "('key', 'SEQN')"`. This is now WORSE in exposure than when first recorded: ml/join_doctor.py has acquired a real UI caller, utils/combine_ui.py:_render_link, and I ran its exact sequence (find_key_candidates -> pick highest-confidence -> diagnose_join) against these frames: the KeyError comes out of utils/combine_ui.py:112, which sits OUTSIDE the try/except at lines 126-135, so it reaches the user as a raw Streamlit traceback rather than the handled "Could not attach" message.

**Suggested fix:** Give KeyCandidate a real label field (e.g. `left_label: Any` holding the actual hashable column object) alongside the display string, and have combine_ui/diagnose_join/execute_join use the label, not str(). Independently, detect `isinstance(df.columns, pd.MultiIndex)` at the top of diagnose_join and either flatten both frames' columns to single strings before diagnosing/merging, or emit a blocking message the researcher can act on: "This file has a two-row header. Flatten it to one header row before combining." Right now the module asserts a row count for a join it cannot perform.

```
$ python3 .../adj/f11.py
pandas 3.0.3
--- 1: diagnose_join with tuple keys ---
summary: Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
can_proceed: True blocking: [] warnings: []
--- 2: execute_join for each how (repair=True then False) ---
repair=True inner -> RAISED ValueError: The column label 'key' is not unique.
repair=True left -> RAISED ValueError: The column label 'key' is not unique.
repair=True right -> RAISED ValueError: The column label 'key' is not unique.
repair=True outer -> RAISED ValueError: The column label 'key' is not unique.
repair=False inner -> RAISED ValueError: The column label 'key' is not unique.
repair=False left -> RAISED ValueError: The column label 'key' is not unique.
repair=False right -> RAISED ValueError: The column label 'key' is not unique.
repair=False outer -> RAISED ValueError: The column label 'key' is not unique.
--- 3: suggest_best round-trip ---
left_col repr: "('key', 'SEQN')" right_col repr: "('key', 'SEQN')" conf: high
round-trip RAISED KeyError : "('key', 'SEQN')"
all candidates: [("('key', 'SEQN')", "('key', 'SEQN')", 'high')]

$ simulating utils/combine_ui.py:_render_link verbatim
UI offers: ('key', 'SEQN') ('key', 'SEQN') high
Traceback (most recent call last):
  ...
  File "/home/user/tabular-ml-lab/ml/join_doctor.py", line 362, in diagnose_join
    ls, rs = left[left_key], right[right_key]
             ~~~~^^^^^^^^^^
  ...
KeyError: "('key', 'SEQN')"
```

---

### 20 — execute_join drops right-only row identifiers on right/outer joins when key columns have different names

**still_broken** · severity major

**What remains:** Reproduces exactly as recorded, on today's code. how='right' gives 3 rows with 2 NULL SEQN; how='outer' gives 5 rows with 2 NULL SEQN; predicted == actual in both cases and blocking/warnings/notes are all empty. The control with identically-named keys still loses nothing (NULL SEQN count 0, subjects 4 and 5 present). It also reproduces with repair=False and with string IDs, so it is not an artefact of key repair or dtype.

Mechanism, unchanged: execute_join merges on left_on=left_key/right_on=right_key (ml/join_doctor.py:542-544), which leaves right-only rows with NaN in the left key column and their real ID in the right key column, and then unconditionally discards that column at ml/join_doctor.py:558-559:
    if left_key != right_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])
There is no coalesce before the drop. Notably the finding-19 fix DID add exactly this rename for the ID-less-row re-attach path (ml/join_doctor.py:552-553, `rb = rb.rename(columns={right_key: left_key})`), so the blank-key rows are handled while the ordinary right-unmatched rows are not - the fix was applied one branch too narrowly.

This is silent data loss, not a row-count lie: participants 4 and 5 keep their glucose values but lose their identity permanently. Severity is aggravated by two things the original finding did not record. First, utils/combine_ui.py:88-132 drives exactly this call path in a loop, and find_key_candidates DOES propose the differently-named pairing ([('SEQN', 'patient_id', 'low')]), so the user reaches it through the normal picker. Second, it cascades: attaching a third file to the already-damaged frame produces 7 rows for 5 subjects, where subjects 4 and 5 appear once with kcal and NULL glucose and again with glucose and NULL SEQN - the same person split into two rows that can never be reconciled, with predicted == actual (7) reported at every step so nothing flags it.

**Suggested fix:** In execute_join, coalesce the right key into the left key before dropping it. Replace ml/join_doctor.py:558-559 with:
    if left_key != right_key and right_key in merged.columns:
        merged[left_key] = merged[left_key].where(merged[left_key].notna(), merged[right_key])
        merged = merged.drop(columns=[right_key])
With repair=True both columns are already canonical object strings so the where() is dtype-safe; with repair=False the merge has widened both to a common NaN-capable dtype, so it is safe there too. A regression test should assert `execute_join(a, b, 'SEQN', 'patient_id', how)[0]['SEQN'].isna().sum() == 0` for how in ('right', 'outer') and that the resulting SEQN values equal the union {1,2,3,4,5}, matching the identically-named-key control.

```
=== differently-named keys ===
inner rows: 1 | predicted: 1 | cols: ['SEQN', 'age', 'glucose'] | rows with NULL SEQN: 0 | blocking: [] | warnings: ['2 row(s) of the first file (67%) have no match and will be dropped. Use a left join to keep them.']
left rows: 3 | predicted: 3 | cols: ['SEQN', 'age', 'glucose'] | rows with NULL SEQN: 0 | blocking: [] | warnings: []
right rows: 3 | predicted: 3 | cols: ['SEQN', 'age', 'glucose'] | rows with NULL SEQN: 2 | blocking: [] | warnings: []
outer rows: 5 | predicted: 5 | cols: ['SEQN', 'age', 'glucose'] | rows with NULL SEQN: 2 | blocking: [] | warnings: []

--- full frame for how='outer' ---
  SEQN   age  glucose
0    1  40.0      NaN
1    2  55.0      NaN
2    3  61.0     95.0
3  NaN   NaN    102.0
4  NaN   NaN    110.0

--- full frame for how='right' ---
  SEQN   age  glucose
0    3  61.0       95
1  NaN   NaN      102
2  NaN   NaN      110

=== CONTROL: identically-named keys ===
NULL SEQN count: 0
  SEQN   age  glucose
0    1  40.0      NaN
1    2  55.0      NaN
2    3  61.0     95.0
3    4   NaN    102.0
4    5   NaN    110.0

=== repair=False variant (differently-named keys, outer) ===
cols: ['SEQN', 'age', 'glucose'] NULL SEQN: 2
=== string IDs variant ===
cols: ['SEQN', 'age', 'glucose'] NULL SEQN: 2

--- f20b.py: CASCADE through utils/combine_ui.py's chained-attach loop ---
after attaching labs (outer):
  SEQN   age  glucose
0    1  40.0      NaN
1    2  55.0      NaN
2    3  61.0     95.0
3  NaN   NaN    102.0
4  NaN   NaN    110.0

after attaching diet (outer): predicted 7 actual 7
  SEQN   age  glucose    kcal
0    1  40.0      NaN  2000.0
1    2  55.0      NaN  2100.0
2    3  61.0     95.0  2200.0
3    4   NaN      NaN  2300.0
4    5   NaN      NaN  2400.0
5  NaN   NaN    102.0     NaN
6  NaN   NaN    110.0     NaN

=== is any warning emitted about the lost identifier? ===
right blocking: [] warnings: [] notes: []
outer blocking: [] warnings: [] notes: []
```

---

### 23 — diagnose_join returns can_proceed=True for a predicted 25,000,000-row many-to-many blow-up, and execute_join has no size cap

**still_broken** · severity minor

**What remains:** Everything the finding described, unchanged. diagnose_join still returns can_proceed=True with blocking=[] for a predicted 25,000,000-row product, and execute_join still has no size cap anywhere in ml/join_doctor.py — it performed the merge in 2.4s, producing a 1.723 GiB frame with peak RSS 1086 MiB from two 5,000-row inputs. Nothing was added in the UI layer either: utils/combine_ui.py:121 gates only on diag.can_proceed, so the merge runs. Mitigating (and true at the time of the original report too): the m2m warning DOES fire, plain_summary honestly states '25,000,000 rows', and suggest_best correctly refuses to propose this key (returns None) — so this is not silent wrongness, which is why it stays minor. The failure mode on a real machine is a resource blow-up, not a wrong answer: under a 2 GiB address-space cap the user gets numpy's `_ArrayMemoryError: Unable to allocate 191. MiB for an array with shape (25000000,) and data type int64` out of ml/join_doctor.py:542. In the app that lands in the except at utils/combine_ui.py:134 and renders as 'Could not attach <file>: Unable to allocate 191. MiB for an array with shape (25000000,)...' — a pandas-internal message shown to a non-programmer — and on a machine without an rlimit the OOM killer can take the Streamlit process before any handler runs.

**Suggested fix:** Add a size gate in diagnose_join: when predicted_rows exceeds a threshold (e.g. max(5_000_000, 50 * max(len(left), len(right)))), append to d.blocking a plain-language message naming the multiplier ('these two files would produce 25,000,000 rows from 5,000 and 5,000 — almost certainly the wrong key or a file that needs one row per subject first'), so can_proceed goes False and the UI's existing gate at utils/combine_ui.py:121 stops it. Back it with a hard guard in execute_join that recomputes the product and raises a ValueError carrying the same sentence, so the cap holds for callers that skip the diagnosis.

```
$ python3 f23.py          # L = 5000 rows all id='A', R = 5000 rows all id='A'
predicted_rows: 25,000,000 | can_proceed: True | blocking: []
warnings: ['Both files have several rows per ID, so every combination is produced: 1 shared IDs become 25,000,000 rows. This is usually a mistake — check whether one file should be summarised to one row per subject first.']
plain_summary: Result: **25,000,000 rows** — matching on 1 shared IDs, keeping only IDs found in both files.
suggest_best: None
RSS before: 108 MiB
rows=25,000,000 time=2.38s deep_mem=1.723 GiB
RSS after: 705 MiB | peak RSS: 1086 MiB
description: Standardised the join keys ('id', 'id') ... Merged left (5,000 rows) with right (5,000 rows) on 'id' using a inner join, giving 25,000,000 rows.

$ python3 f23_rlimit.py   # same frames, address space capped at 2 GiB (a modest laptop)
can_proceed: True blocking: []
address space capped at 2 GiB; calling execute_join ...
--- what the user sees ---
Traceback (most recent call last):
  File "f23_rlimit.py", line 14, in <module>
    m, _ = execute_join(L, R, "id", "id")
  File "/home/user/tabular-ml-lab/ml/join_doctor.py", line 542, in execute_join
    merged = l[~lmask].merge(
  File ".../pandas/core/reshape/merge.py", line 2153, in get_join_indexers_non_unique
    lidx, ridx = libjoin.inner_join(lkey, rkey, count, sort=sort)
  File "pandas/_libs/join.pyx", line 83, in pandas._libs.join.inner_join
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 191. MiB for an array with shape (25000000,) and data type int64

$ grep -n 'predicted_rows|can_proceed|execute_join' pages/ app.py utils/
utils/combine_ui.py:121:  if not diag.can_proceed and not diag.dtype_mismatch:
utils/combine_ui.py:129:  result, desc = execute_join(...)      # inside try/except Exception
```

---

### 27 — check_numeric_stored_as_text silently skips text columns whose values are all plain numbers (the `raw_numeric >= 0.99` guard), so promote_header output gets a false all-clear

**still_broken** · severity major

**What remains:** Everything. The defect reproduces verbatim, line for line, against today's code. The guard is untouched at ml/import_doctor.py:421-422:

    raw_numeric = pd.to_numeric(s, errors="coerce").notna().mean()
    if raw_numeric >= 0.99:
        continue  # pandas would already have typed it numeric

The comment's premise is false in exactly the case the module itself creates. apply_fix('promote_header') at ml/import_doctor.py:608-615 slices `df.iloc[row+1:]` off an all-object frame and relabels the columns; it never re-infers dtypes, so 'age' and 'bmi' come out as str columns full of clean numeric strings, raw_numeric is 1.0, and the check skips them. Step 2 of the repro still proves the guard rather than the parser is responsible: adding one '37 yrs' to the identical column flips it to ['numeric_as_text__age'].

The end-to-end harm is intact and reaches the real app. diagnose() on the promoted frame reports only ['footer_rows', 'constant_columns'] and summarize() says 'Found 1 worth checking, 1 note.' — a near-clean bill of health on a frame whose two measurement columns cannot be modelled, correlated or plotted (the module's own words at ml/import_doctor.py:465-466). This is not confined to the library: pages/01_Upload_and_Audit.py:397 does `df_preview = render_import_doctor(df_preview, file_key)` and utils/import_ui.py:121-136 returns `current` — the post-fix frame — unchanged, with no dtype re-inference anywhere on that path, so the all-string frame is what gets added to the project. The trailing TypeError ('Cannot perform reduction \'mean\' with string dtype') is still exactly what a user hits downstream.

**Suggested fix:** Make the guard conditional on the frame's provenance rather than on the values. Either (a) have apply_fix('promote_header') call `out = out.infer_objects()` (or apply `pd.to_numeric(..., errors='ignore')` per column) before returning, which fixes the flagship path at the source and is a visible, describable step for the methods text; or (b) in check_numeric_stored_as_text, replace `if raw_numeric >= 0.99: continue` with a check that the column's dtype is genuinely object/string AND pandas' own inference would not have typed it — i.e. drop the skip entirely and instead emit the finding at confidence 'high' with fix_kind='coerce_numeric' when raw_numeric == 1.0, since a fully-parseable text column is the safest possible conversion (n_blanked == 0). (b) alone also covers text-numeric columns arriving from JSON/Excel readers that never inferred. Regression test: assert diagnose(pd.DataFrame({'age': ['31','32','33','34','35','36','37']})) is non-empty, and assert pd.api.types.is_numeric_dtype on 'age' after the promote_header round-trip in the Nutrition Cohort fixture.

```
=== 1) minimal: pure text-numeric column ===
findings: []
summarize: No structural problems detected.

=== 2) same column plus ONE unit-bearing value ===
findings: ['numeric_as_text__age']

=== 3) flagship Excel scenario, end to end ===
raw columns: ['Nutrition Cohort Study 2024', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3']
raw findings: ['header_in_later_row']
apply_fix desc: Promoted row 3 to column headers and dropped the 3 row(s) above it.
fixed columns: ['subject_id', 'age', 'bmi', 'site']
fixed dtypes: ['str', 'str', 'str', 'str']
findings after promote_header: ['footer_rows', 'constant_columns']
summarize: Found 1 worth checking, 1 note.
is age numeric? False
is bmi numeric? False
head:
  subject_id age   bmi    site
0       S001  31  22.4  Boston
1       S002  32  22.8  Boston
2       S003  33  23.2  Boston
3       S004  34  23.6  Boston
4       S005  35  24.0  Boston
age.mean() RAISED: TypeError Cannot perform reduction 'mean' with string dtype

=== 4) what does check_numeric_stored_as_text say about the promoted frame? ===
[]
raw_numeric ratio for 'age': 1.0
```

---

### 32 — check_header_in_later_row false-positives on clean narrow frames with a blank header cell: emits the single critical/high-confidence 'promote_header' finding, which drops the first data row and hides all other findings

**still_broken** · severity major

**What remains:** Everything the finding described, unchanged. Commit a1426ef ("stop fixes that destroy data while looking helpful") touched the sentinel, text-missing, unit and melt code but never touched check_header_in_later_row. On a 2-column frame with one blank header cell, unnamed_frac = 1/2 = 0.5 passes the `< 0.5` guard, and the first data row scores exactly uniq(1.0) * stringy(0.5) * fill(1.0) = 0.5, which passes `if best_score < 0.5: return []` by a hair. The result is a CRITICAL, confidence='high', auto_suggestable=True finding — the app asserting it — offering 'Use row 1 as the column names'; applying it renames the columns to ['Apple', '95'] and permanently drops the Apple/95 row (5 rows -> 4). Because diagnose() short-circuits on any header finding, it is also the ONLY finding returned: the genuine 'Male'/'male '/'male' case-variant collision and the missing-token finding both vanish and reappear the instant the column is renamed to 'sex'. utils/import_ui.py renders it as the top critical item with a one-click button, so a non-programmer is being pointed at the destructive action. Note the false positive is specific to narrow frames: at 3 columns with one blank header the same file reports only a low-confidence 'unnamed_columns' note.

**Suggested fix:** In check_header_in_later_row: (1) require strictly more than half the columns unnamed and at least 2 unnamed columns (`n_unnamed >= 2 and unnamed_frac > 0.5`), so a single blank cell in a 2-column file cannot qualify; (2) raise the acceptance bar above the tie (`best_score <= 0.5` rejects, or require score >= 0.65) and require the candidate row to be majority non-numeric (stringy >= 0.6) — 'Apple, 95' is half numeric and reads as data, not a header; (3) only let the header finding suppress the other checks when it is genuinely decisive (e.g. score >= 0.8), otherwise append it to the normal finding list so real problems are never hidden.

```
$ python3 f32_verbatim.py
renamed -> ['category_variants__sex', 'text_missing_ambiguous__sex']
BUG CONFIRMED

$ python3 f32.py
pandas 3.0.3
=== 1. clean narrow frame, first header cell blank ===
  columns: ['Unnamed: 0', 'kcal']
  findings: [('header_in_later_row', 'critical', 'high', True)]
    title : The real column names look like they are in row 1
    detail: 50% of columns are unnamed, and row 1 reads like a header: Apple, 95…
    fix   : Use row 1 as the column names
  after apply_fix -> columns=['Apple', '95'] rows=4 (was 5)
  desc: Promoted row 1 to column headers and dropped the 1 row(s) above it.
    Apple   95
0  Banana  105
1    Rice  205
2    Oats  150
3    Milk  103

=== 2. masking ===
  as-is        : [('header_in_later_row', 'critical', 'high')]
  renamed 'sex': [('category_variants__sex', 'warning', 'high'), ('text_missing_ambiguous__sex', 'info', 'low')]

=== 4. wider clean frame with one blank header cell (3 cols) ===
  findings: [('unnamed_columns', 'info', 'low')]

=== 5. score arithmetic ===
  vals ['Apple', '95'] uniq 1.0 stringy 0.5 score 0.5 threshold test score<0.5 -> False
```

---

### 39 — Row-counter columns are rated "high" confidence and proposed as the best join key whenever both files use the same counter name; index_like is never surfaced to the user

**still_broken** · severity critical

**What remains:** Everything in the title, unchanged. An index_like guard exists but is gated on name DISagreement, so it can never fire in the case the finding is about. ml/join_doctor.py line 182 (`if self.index_like and self.name_similarity < 0.85: s *= 0.15`) and line 189 (same condition -> 'low') both require name_similarity < 0.85; when both files call the counter the same thing name_similarity == 1.0, so the penalty and the 'low' downgrade are skipped and line 191 promotes it to 'high'. The control case in f39b.py proves the guard works only when the names differ ('row' vs 'lineno' -> the counter pairing is suppressed entirely).

Concretely, on today's code:
(a) an unrelated survey and economics file are linked on 'row' at 'high' confidence, score 1.000;
(b) two sites that each numbered participants 1..N are joined at 'high' confidence and diagnose_join returns blocking=[] warnings=[] — a 50-row frame pairing site A's participant 7 with site B's participant 7, asserted with no caveat at all;
(c) the counter still BEATS the genuine key: 'row'<->'row' scores 1.000/high while the real SEQN<->patient_id scores 0.688/medium, and suggest_best returns 'row';
(d) all six generic counter names ('index', 'id', 'n', 'Unnamed: 0', 'rownum', 'obs') come back 'high'.

index_like is still never surfaced: KeyCandidate.headline() (lines 201-212) has no index_like branch, and grep shows the only consumers of the flag are join_doctor's own score/confidence properties plus one test — utils/combine_ui.py never reads it. In the real UI (utils/combine_ui.py lines 88-110) candidates are filtered by `c.confidence != "low"` and the selectbox DEFAULTS to options[0], so the row counter is the pre-selected key and the caption the user reads is the reassuring '...share 60 IDs (100% ..., 100% ...)'. Rated critical under this app's contract because 'high' is what the UI pre-selects, so the app is asserting a linkage that fabricates subject identity.

**Suggested fix:** Make index_like an unconditional veto rather than one gated on name disagreement: in KeyCandidate.confidence (ml/join_doctor.py line 189) return 'low' (or at most 'medium') whenever index_like is True, regardless of name_similarity, and apply the score penalty at line 182 unconditionally so a real key like SEQN<->patient_id outranks a coincidental counter. Add an index_like branch to headline() that says so in plain language (e.g. "'row' and 'row' both look like row numbers (1, 2, 3, ...) rather than real IDs — joining on them pairs the 7th row of one file with the 7th row of the other, which is only correct if the files are already in the same order"), and have utils/combine_ui.py show the confidence level and that caveat next to the pre-selected option.

```
=== (a) two unrelated files that merely both carry a row counter ===
  row <-> row score=1.000 conf=high index_like=True
     headline: 'row' and 'row' share 50 IDs (100% of survey.csv, 100% of economics.csv).
  suggest_best -> row

=== (b) two sites that each numbered their participants 1..N ===
  conf= high index_like= True
  headline: 'subject_id' and 'subject_id' share 50 IDs (100% of site A, 83% of site B).
  diagnose: Result: **50 rows** — matching on 50 shared IDs, keeping only IDs found in both files.
  blocking: []
  warnings: []
  notes   : ['10 row(s) of site B have no match and will be dropped.']

=== (c) counter vs a genuine key ===
  row <-> row score=1.000 conf=high index_like=True
  SEQN <-> patient_id score=0.688 conf=medium index_like=False
  suggest_best -> ('row', 'row', 'high')

=== (d) generic counter names ===
  'index'      conf=high index_like=True suggest_best=index
  'id'         conf=high index_like=True suggest_best=id
  'n'          conf=high index_like=True suggest_best=n
  'Unnamed: 0' conf=high index_like=True suggest_best=Unnamed: 0
  'rownum'     conf=high index_like=True suggest_best=rownum
  'obs'        conf=high index_like=True suggest_best=obs

[f39b.py — combine_ui replay]
== (c) counter present alongside the REAL key
  selectbox options: ['row <-> row', 'SEQN <-> patient_id', 'age <-> row']
  DEFAULT (pre-selected) = row <-> row
  st.caption: 'row' and 'row' share 60 IDs (100% of your data so far, 100% of labs.csv).
  (confidence=high, index_like=True -- neither shown to the user)

== (control) counters with DIFFERENT names -> guard fires
  selectbox options: ['age <-> lineno']
  DEFAULT (pre-selected) = age <-> lineno
```

---

### 40 — diagnose_join: dtype-mismatch blocker fires ahead of the zero-overlap check, producing "Fixing this matches 0 IDs" plus a contradictory "use a left join" warning

**still_broken** · severity major

**What remains:** Both halves of the finding, verbatim. ml/join_doctor.py line 422 is still `if dtype_mismatch:` with the zero-overlap check demoted to `elif not matched:` on line 429, so when the two files share no IDs at all AND their key dtypes differ the user is told the self-defeating "Fixing this matches 0 IDs" instead of the accurate "there is nothing to join on". The control case (part 3) confirms the message is only useful when there IS overlap.

The contradictory advice is also still emitted, and still on BOTH branches: line 456-462 appends "... have no match and will be dropped. Use a left join to keep them." whenever how == 'inner' and the unmatched fraction is >= 10%, with no check that matched_keys > 0. So it fires at 100% unmatched in part 1 (alongside the dtype blocker) and again in part 2 on the honest zero-overlap branch, where the blocking message says "there is nothing to join on" and the warning immediately below tells the user to switch to a left join. Taking that advice yields exactly what the finding predicted — I ran it: the left join returns 3 rows with glucose all-NaN (glucose all-NaN: True), i.e. the second file contributes nothing.

New this run, and worse than the original write-up: utils/combine_ui.py line 121 is `if not diag.can_proceed and not diag.dtype_mismatch:` — a dtype-mismatch blocker is explicitly bypassed, so the app renders "This join will not work yet — see below", then runs the join anyway and hands back an EMPTY 0-row frame, with a provenance string that reads as a completed merge ("Merged demographics (3 rows) with labs (3 rows) on 'SEQN' using a inner join, giving 0 rows"). The bypass is sound when repair would actually recover IDs (part 3), but with matched_keys == 0 it silently produces an empty cohort.

**Suggested fix:** In ml/join_doctor.py: (1) test overlap first — `if not matched:` emit the "nothing to join on" blocker, extended when dtype_mismatch is also true to say the values do not match even after the type is fixed; keep the dtype-mismatch blocker only for the `matched > 0` case where "Fixing this matches N IDs" is actionable. (2) Guard the left-join advice at line 456-462 on `matched` being non-empty (e.g. only append "Use a left join to keep them." when matched_keys > 0 and the unmatched share is < 100%). (3) In utils/combine_ui.py line 121, only bypass the blocker when `diag.dtype_mismatch and diag.matched_keys > 0`, so a repair-that-recovers-nothing stops rather than delivering an empty frame.

```
=== part 1: dtype mismatch AND zero real overlap ===
summary: This join will not work yet — see below.
BLOCKING: 'SEQN' is stored as text in demographics.csv but as numbers in labs.xlsx. They look identical on screen but will not match. Fixing this matches 0 IDs.
WARNING : 3 row(s) of demographics.csv (100%) have no match and will be dropped. Use a left join to keep them.
NOTE    : 3 row(s) of labs.xlsx have no match and will be dropped.
matched_keys = 0
  SEQN  age  glucose
0  a01   40      NaN
1  a02   55      NaN
2  a03   61      NaN
glucose all-NaN: True

=== part 2: honest zero-overlap, no dtype mismatch ===
blocking: ["None of the values in 'SEQN' appear in 'SEQN', so there is nothing to join on. Check you picked the right columns."]
warnings: ['3 row(s) of demographics.csv (100%) have no match and will be dropped. Use a left join to keep them.']

=== part 3: dtype mismatch WITH real overlap (control) ===
blocking: ["'SEQN' is stored as text in demographics.csv but as numbers in labs.xlsx. ... Fixing this matches 3 IDs."]
warnings: []
matched_keys = 3

[f40b.py — combine_ui replay]
st.error   🛑 'SEQN' is stored as text in your data so far but as numbers in labs.xlsx. ... Fixing this matches 0 IDs.
st.warning ⚠️ 3 row(s) of your data so far (100%) have no match and will be dropped. Use a left join to keep them.
>>> UI PROCEEDS (dtype_mismatch bypasses the blocker)
st.markdown This join will not work yet — see below.
delivered rows: 0
Empty DataFrame
Columns: [SEQN, age, glucose]
Index: []
provenance: Standardised the join keys ('SEQN', 'SEQN') ... Merged demographics (3 rows) with labs (3 rows) on 'SEQN' using a inner join, giving 0 rows.
```

---

### 41 — diagnose_join emits no warning about rows that will be blank-filled on left/right/outer joins

**still_broken** · severity major

**What remains:** Reproduces verbatim. On left/right/outer, diagnose_join returns blocking=[], warnings=[], notes=[] — completely silent — even though 49 of 100 rows (49%) get a blank glucose on the left join, 50 of 101 get a blank age on the right join, and both happen on the outer join. The plain_summary line the UI puts above the Merge button reads 'Result: 100 rows — matching on 51 shared IDs, keeping every row of screening,' which is the reassuring half of the story; the half where half the new column is empty is never stated. Note the promised row counts are now correct (100/101/150 all match the delivered frame), so this is not a count lie — it is a pure diagnosis omission. The guards at ml/join_doctor.py:456 and :463 are both `if how == "inner"`, so unmatched_left_rows / unmatched_right_rows are computed (lines 386-387) and then discarded for every non-inner join. utils/combine_ui.py:114-119 renders exactly diag.blocking/warnings/notes, so nothing downstream compensates — the user is shown a clean bill of health on a merge that silently blank-fills half the cohort.

**Suggested fix:** In diagnose_join, after the two existing `how == "inner"` blocks, add the mirror cases:

    if how in ("left", "outer") and unmatched_left_rows:
        pct = unmatched_left_rows / max(1, len(left))
        d.warnings.append(
            f"{unmatched_left_rows:,} row(s) of {left_name} ({pct:.0%}) have no match in "
            f"{right_name}. They are kept, but every column coming from {right_name} will "
            f"be blank for them.")
    if how in ("right", "outer") and unmatched_right_rows:
        pct = unmatched_right_rows / max(1, len(right))
        d.warnings.append(
            f"{unmatched_right_rows:,} row(s) of {right_name} ({pct:.0%}) have no match in "
            f"{left_name}. They are kept, but every column coming from {left_name} will "
            f"be blank for them.")

Use the same >=10% warning/note split already used for the inner case so small overlaps stay quiet.

```
left | Result: **100 rows** — matching on 51 shared IDs, keeping every row of screening.
   blocking: []
   warnings: []
   notes   : []
   actual rows: 100 | blank glucose: 49 | blank age: 0

right | Result: **101 rows** — matching on 51 shared IDs, keeping every row of followup.
   blocking: []
   warnings: []
   notes   : []
   actual rows: 101 | blank glucose: 0 | blank age: 50

outer | Result: **150 rows** — matching on 51 shared IDs, keeping every row of both files.
   blocking: []
   warnings: []
   notes   : []
   actual rows: 150 | blank glucose: 49 | blank age: 50

inner | Result: **51 rows** — matching on 51 shared IDs, keeping only IDs found in both files.
   blocking: []
   warnings: ['49 row(s) of screening (49%) have no match and will be dropped. Use a left join to keep them.']
   notes   : ['50 row(s) of followup have no match and will be dropped.']
   actual rows: 51 | blank glucose: 0 | blank age: 0

ground truth left rows: 100 blank glucose: 49
ground truth outer rows: 150 blank glucose: 49 blank age: 50
```

---

### 42 — diagnose_join suppresses genuine column collisions whenever a key name also exists in the other frame (cross-name joins), so no suffix warning fires and execute_join's methods description names a column that no longer exists

**still_broken** · severity major

**What remains:** All three reported cases reproduce exactly, and the control proves the collision machinery works everywhere else. ml/join_doctor.py:401-402 subtracts BOTH key names unconditionally: `collisions = [c for c in (set(left.columns) & set(right.columns)) if str(c) not in {str(left_key), str(right_key)}]`. On a cross-name join (left_key='SEQN', right_key='patient_id') that erases the collision on whichever key name happens to exist in the other frame, so column_collisions=[] and no suffix warning fires. Three concrete harms remain: (1) Case 1/2 — the join key itself is silently renamed to SEQN_demo/SEQN_labs, so the user's chosen ID column is gone from the result under the name they chose, with zero warning; (2) every case — execute_join's methods-ready description says "on 'SEQN'", naming a column that does not exist in the delivered frame (cases 1 and 2 have only SEQN_demo/SEQN_labs). That string is what gets recorded into the methods section, so the permanent record is wrong; (3) Case 3 — right_key 'patient_id' is suffixed to patient_id_labs, so the `if left_key != right_key and right_key in merged.columns: drop(columns=[right_key])` cleanup at line 558 never fires, leaving a redundant patient_id_labs column duplicating the key. No crash, no message — pure silent restructuring, which is the failure mode this module's docstring exists to prevent.

**Suggested fix:** Only exclude a key name from the collision set when it is genuinely the join key on BOTH sides:

    shared = set(left.columns) & set(right.columns)
    if str(left_key) == str(right_key):
        shared -= {left_key}
    collisions = sorted(str(c) for c in shared)

That makes cases 1-3 report SEQN and/or patient_id as collisions and fire the existing suffix warning. Additionally: (a) add a distinct, louder warning when a collision name IS one of the chosen keys, because the key column is about to be renamed — the user needs to know which column to select afterwards; (b) in execute_join, compute the surviving key name (left_key, or f'{left_key}{suffixes[0]}' when it collided) and use that in the description instead of the raw left_key, so the methods sentence names a column that actually exists; (c) make the right_key cleanup at line 558 look for the suffixed name too.

```
--- both names in both files
  left cols : ['SEQN', 'patient_id', 'age']
  right cols: ['patient_id', 'SEQN', 'glucose']
  collisions: []
  warnings  : []
  merged    : ['SEQN_demo', 'patient_id_demo', 'age', 'patient_id_labs', 'SEQN_labs', 'glucose']
  desc      : Standardised the join keys ('SEQN', 'patient_id') ... Merged demo (3 rows) with labs (3 rows) on 'SEQN' using a inner join, giving 3 rows.

--- only left_key name exists in right
  left cols : ['SEQN', 'age']
  right cols: ['patient_id', 'SEQN', 'glucose']
  collisions: []
  warnings  : []
  merged    : ['SEQN_demo', 'age', 'SEQN_labs', 'glucose']
  desc      : ... Merged demo (3 rows) with labs (3 rows) on 'SEQN' using a inner join, giving 3 rows.

--- only right_key name exists in left
  left cols : ['SEQN', 'patient_id', 'age']
  right cols: ['patient_id', 'glucose']
  collisions: []
  warnings  : []
  merged    : ['SEQN', 'patient_id_demo', 'age', 'patient_id_labs', 'glucose']
  desc      : ... Merged demo (3 rows) with labs (3 rows) on 'SEQN' using a inner join, giving 3 rows.

--- control: ordinary collision on 'age'
  left cols : ['SEQN', 'age']
  right cols: ['SEQN', 'age', 'glucose']
  collisions: ['age']
  warnings  : ['Both files have column(s) named age. They will be kept side by side with suffixes so nothing is overwritten.']
  merged    : ['SEQN', 'age_demo', 'age_labs', 'glucose']
  desc      : ... Merged demo (3 rows) with labs (3 rows) on 'SEQN' using a inner join, giving 3 rows.
```

---

### 44 — 'Convert to numbers' strips the decimal comma, silently rescaling European-format numeric columns — at 'high' confidence, marked auto-suggestable

**changed_form** · severity critical

**What remains:** The reported symptom is genuinely gone: all three European-format cases now convert correctly (22,5 -> 22.5 instead of 225), via _looks_decimal_comma / _clean_numeric_text at ml/import_doctor.py:119-138. But the fix inverted the bug rather than resolving the ambiguity, and it broke the control case that the original repro recorded as CORRECT. '45,000' now becomes 45.0 — a silent 1000x under-scale — still at confidence='high', auto_suggestable=True, with a fix_label of just "Convert 'v' to numbers" and a methods description ('removing units, separators and comparison signs') that never mentions a decimal-comma interpretation was chosen. Root cause is _DECIMAL_COMMA at line 106: `^[+-]?\d{1,3}(?:\.\d{3})*,\d+$` — the `*` lets the dot-thousands branch match a plain '45,000' with zero dot groups, so any single-comma US-format number is claimed as European decimal. _looks_decimal_comma only needs 60% of values to match, so a whole income/population/cell-count column flips. I judge this critical rather than major under the stated contract: it is a 'high' confidence, pre-selected recommendation that is confidently wrong, and it is now strictly worse in magnitude than the original (1000x vs 10x) on the more common file format. One partial mitigation exists by accident — mixing in a two-comma value like '1,234,567' makes that value unparseable, which drops confidence to 'low' and un-pre-selects the fix, but that is the failure warning about itself, not a check.

**Suggested fix:** Two changes in ml/import_doctor.py. (1) Require the dot-thousands branch to actually contain a dot group so it stops swallowing '45,000':

    _DECIMAL_COMMA = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+,\d+$|^[+-]?\d+,\d{1,2}$")

Verified in f44c.py: this keeps 22,5 / 5,55 / 980,3 / 1.234,5 / 1.050,25 / 2.750,8 / 3.400,0 as decimal-comma and rejects 45,000 / 12,000 / 7,500 / 450,000. (2) The regex alone cannot settle the genuinely ambiguous shape — exactly three digits after a single comma ('5,555', '45,000') is 5.555 in Bonn and 5555 in Boston, and the module's own contract is 'never silently guess'. So when a column is majority-ambiguous, do not pick: emit the finding at confidence='low' (auto_suggestable becomes False) and say which reading was assumed in fix_label and in the apply_fix description, e.g. "Convert 'v' to numbers, reading '45,000' as 45000 (thousands separator) — switch if these are European decimals". That turns a silent 1000x rescale into a visible, reversible choice.

```
bmi (1-digit decimal comma):
  confidence=high auto_suggestable=True
  input  -> ['22,5', '28,4', '31,0', '24,5', '19,8', '27,1']
  output -> [22.5, 28.4, 31.0, 24.5, 19.8, 27.1]          <- CORRECT NOW

glucose (2-digit decimal comma):
  confidence=high auto_suggestable=True
  input  -> ['5,55', '6,10', '4,98', '7,25', '5,04', '6,33']
  output -> [5.55, 6.1, 4.98, 7.25, 5.04, 6.33]           <- CORRECT NOW

de/at export (dot thousands + comma decimal):
  confidence=high auto_suggestable=True
  input  -> ['1.234,5', '2.100,7', '980,3', '1.050,25', '3.400,0', '2.750,8']
  output -> [1234.5, 2100.7, 980.3, 1050.25, 3400.0, 2750.8]  <- CORRECT NOW

true thousands separators (control):
  confidence=high auto_suggestable=True
  input  -> ['45,000', '52,300', '61,000', '48,000', '55,500', '39,900']
  output -> [45.0, 52.3, 61.0, 48.0, 55.5, 39.9]          <- NEWLY WRONG (was correct)

--- f44b.py: realistic US-format columns ---
annual income USD:
  in : ['45,000', '52,300', '61,000', '48,000', '55,500', '39,900']
  out: [45.0, 52.3, 61.0, 48.0, 55.5, 39.9]
  severity=warning confidence=high auto_suggestable=True
  label: Convert 'v' to numbers
  desc : Converted 'v' to numeric (removing units, separators and comparison signs).

city population:
  in : ['12,000', '98,500', '7,500', '450,000', '33,250', '61,000']
  out: [12.0, 98.5, 7.5, 450.0, 33.25, 61.0]
  severity=warning confidence=high auto_suggestable=True

cell counts:
  in : ['1,200', '3,400', '5,600', '7,800', '9,100', '2,300']
  out: [1.2, 3.4, 5.6, 7.8, 9.1, 2.3]
  severity=warning confidence=high auto_suggestable=True

mixed w/ millions:
  in : ['45,000', '1,234,567', '52,300', '61,000', '48,000', '55,500']
  out: [45.0, nan, 52.3, 61.0, 48.0, 55.5]
  severity=warning confidence=low auto_suggestable=False

--- f44c.py: regex token test (current vs proposed) ---
token         current  proposed
'22,5'           True      True
'5,55'           True      True
'1.234,5'        True      True
'1.050,25'       True      True
'980,3'          True      True
'45,000'         True     False
'12,000'         True     False
'7,500'          True     False
'450,000'        True     False
'1,234,567'     False     False
'5,555'          True     False
'2.750,8'        True      True
'3.400,0'        True      True
```

---

### 7 — diagnose_join green-lights date-vs-text key pairs that execute_join(repair=False) cannot merge

**partially_fixed** · severity minor

**What remains:** The reported half (Variant A: datetime64 left vs text right) is no longer green-lit — diagnose_join now returns can_proceed=False and plain_summary says 'This join will not work yet', and utils/combine_ui.py::_render_link blocks on `not diag.can_proceed and not diag.dtype_mismatch`, so the raw pandas ValueError cannot reach a UI user. But Variant B — the same root cause with the sides swapped (text left, datetime.date right) — is unchanged and is the worse half: diagnose_join reports matched_keys=3, predicted_rows=3, can_proceed=True, blocking=[] and plain_summary promises 'Result: 3 rows', while execute_join(..., repair=False) silently delivers 0 rows. A promised row count that disagrees with the delivered frame is exactly the failure the app's contract calls out. It is confined to the module API — every in-repo caller (combine_ui.py:129) uses the repair=True default, which returns the correct 3 rows — so severity stays minor. Note also that A's repair=True path now returns 0 rows where the original report recorded 3; that regression is unreachable through the UI because the (wrongly-worded) blocker fires first, and it is really finding 05's stringified-datetime bug, not a new one here. The residue is the same missing datetime canonicalisation: diagnose_join's dtype_mismatch test only asks is_numeric_dtype(left) != is_numeric_dtype(right), so datetime-vs-text and date-object-vs-text both read as 'not a mismatch'.

**Suggested fix:** Widen the dtype_mismatch test in diagnose_join (and the matching one in find_key_candidates) beyond numeric-vs-non-numeric to a comparison of merge-compatible kinds — datetime64 vs object/str, and object-holding-datetime.date vs str, must both be flagged so the pair is reported as repairable-with-normalisation rather than clean. Then either raise the same plain-language blocker on repair=False or make repair=False refuse rather than hand the frames to pandas.

```
$ python3 f07.py
pandas 3.0.3
A) can_proceed: False | blocking: ["None of the values in 'd' appear in 'd', so there is nothing to join on. Check you picked the right columns."] | warnings: ['3 row(s) of the first file (100%) have no match and will be dropped. Use a left join to keep them.']
A) dtype_mismatch: False | needs_norm: False
A) predicted: 0 matched: 0
A) This join will not work yet — see below.
A) repair=False RAISED: ValueError : You are trying to merge on datetime64[us] and str columns for key 'd'. If you wish to proceed you should use pd.concat
B) predicted: 3 matched_keys: 3 can_proceed: True blocking: [] -> ACTUAL rows: 0
C) numeric can_proceed: False | 'id' is stored as numbers in the first file but as text in the second file. They look iden
D) A repair=True rows: 0
D) B repair=True rows: 3
D) C repair=True rows: 3
```

---

### 8 — find_key_candidates draws an independent 5,000-row sample per file, so value overlap is measured between unrelated row subsets — the true key is dropped entirely above ~50k rows and mis-quoted 4x-150x too low below that

**partially_fixed** · severity major

**What remains:** The 5,000-row independent random sample is genuinely gone — _key_tokens now canonicalises DISTINCT values, and every case in the recorded repro that the finding said failed now passes: A finds SEQN<->SEQN with 300 matched at 'high' (was [] / None), B finds SEQN<->patient_id with 800 matched (was 0 / None), C's 20k/19k and 100k/90k now quote 19,000 and 90,000 instead of 2,601 and 624 with correct left_rows, and D recovers the 3,000-key candidate with right_has_duplicates=True (was []). What remains is the SAME failure mechanism at a higher threshold: _MAX_DISTINCT = 200_000 truncates with `uniques.iloc[:_MAX_DISTINCT]`, i.e. the first 200k distinct values in each file's own row order, so once either side has more than 200,000 distinct key values the two sides are again compared on different subsets. Case E/F: 300,000 subjects, 100% shared, one file sorted and one in export order -> reported 'share 133,342 IDs (67% / 67%)' at confidence 'high' when the truth is 300,000 at 100%/100%. Case G (C's last row): reported '200,000 IDs (100% / 100%)' at 'high' when the truth is 400,000 at 80%/100%. Case H is the original symptom verbatim: 500,000 subjects, every ID shared, right file written descending -> find_key_candidates returns [] and suggest_best returns None, so the app tells the user the files cannot be attached. Case I: 250,000 string patient IDs, all shared, reported as 159,887 (80%/80%). Secondary: the truncation also corrupts left_unique/right_unique, so KeyCandidate reports left_has_duplicates=True and right_has_duplicates=True for two files that contain no duplicate IDs at all (case E). diagnose_join itself is unaffected (it scans the full column), so the delivered row count stays honest — the damage is a 'high'-confidence headline quoting a wrong overlap, and, above ~400k distinct values, the true key vanishing entirely.

**Suggested fix:** Drop the positional head-truncation. If a cap is needed for pathological files, make it order-independent and identical on both sides — e.g. hash-bucket sample the canonical tokens (keep tokens whose hash falls in a fixed fraction of the space) so both files retain the SAME subset of the key space, and then scale the reported n_matched back up by the sampling fraction and label it as an estimate. Also derive left_has_duplicates from the full column's nunique, not from the truncated token set, and refuse to report 'high' confidence whenever truncation was applied.

```
$ python3 f08.py   (the recorded repro, cases A-D)
A) candidates: [('SEQN', 'SEQN', 300, 'high')]
A) suggest_best: ('SEQN', 'SEQN', 300, 'high')
A) headline: 'SEQN' and 'SEQN' share 300 IDs (100% of the first file, 0% of the second file).
A) truth: 300 (1.1s)
B) n candidates: 1 suggest_best: ('SEQN', 'patient_id', 800, 'medium')
B) headline: 'SEQN' and 'patient_id' share 800 IDs (100% of the first file, 1% of the second file).
B) truth: 800 (1.7s)
C)
   6000/5000: 'SEQN' and 'SEQN' share 5,000 IDs (83% of the first file, 100% of the second file). | conf: high | left_rows: 6000 | truth: 5000 (0.1s)
   20000/19000: 'SEQN' and 'SEQN' share 19,000 IDs (95% of the first file, 100% of the second file). | conf: high | left_rows: 20000 | truth: 19000 (0.2s)
   100000/90000: 'SEQN' and 'SEQN' share 90,000 IDs (90% of the first file, 100% of the second file). | conf: high | left_rows: 100000 | truth: 90000 (1.0s)
   500000/400000: 'SEQN' and 'SEQN' share 200,000 IDs (100% of the first file, 100% of the second file). | conf: high | left_rows: 500000 | truth: 400000 (5.0s)
D) candidates: [('SEQN', 'SEQN', 3000, 'high', False, True)]
D) truth: 3000 True (1.4s)

$ python3 f08b.py   (>200k distinct values, files in different orders)
E) candidates: [('SEQN', 'SEQN', 133342, 'high', 0.667, 0.667)]
E) suggest_best: KeyCandidate(left_col='SEQN', right_col='SEQN', coverage_left=0.66671, coverage_right=0.66671, n_matched=133342, left_unique=200000, right_unique=200000, left_rows=300000, right_rows=300000, dtype_mismatch=False, needs_normalization=False, left_has_duplicates=True, right_has_duplicates=True, name_similarity=1.0, index_like=False)
E) headline: 'SEQN' and 'SEQN' share 133,342 IDs (67% of the first file, 67% of the second file).
E) truth matched_keys: 300000
F) headline: 'SEQN' and 'SEQN' share 133,186 IDs (67% of the first file, 67% of the second file).
F) truth matched_keys: 300000
G) reported n_matched: 200000 cov_l: 1.0 cov_r: 1.0 conf: high
G) truth  n_matched: 400000 cov_l: 0.8 cov_r: 1.0

$ python3 f08c.py   (500k subjects, 100% shared, right file written newest-first)
H) candidates: []
H) suggest_best: None
H) headline: NO CANDIDATES
H) truth matched_keys: 500000
I) candidates: [('SEQN', 'patient_id', 159887, 'medium')]
I) headline: 'SEQN' and 'patient_id' share 159,887 IDs (80% of the first file, 80% of the second file).
I) truth matched_keys: 250000
```

---

### 9 — Duplicate key column name makes every ml/join_doctor.py entry point raise AttributeError and makes find_key_candidates silently drop the true key

**partially_fixed** · severity minor

**What remains:** The worst part is fixed: diagnose_join no longer raises the raw AttributeError; it now raises a clear, actionable ValueError ("The column 'SEQN' appears more than once in one of these files. Rename or remove the duplicate before joining.") at ml/join_doctor.py:362-367, and tests/test_join_doctor.py:308 locks that in. Upstream, ml/import_doctor.py now flags duplicate labels as a CRITICAL 'duplicate_columns' finding with a high-confidence dedupe fix at upload, so the frame is unlikely to survive to the join step. Two real remnants: (1) execute_join() and repair_keys() are still NOT guarded — they hit normalize_key() -> `text.str.lower()` on a DataFrame and emit the raw `AttributeError: 'DataFrame' object has no attribute 'str'`. Note the symptom also changed shape: the original recorded repro for execute_join(repair=False) produced `ValueError: The column label 'SEQN' is not unique.`; today the new blank-ID pre-pass (join_doctor.py:539) reaches normalize_key first, so it is the AttributeError instead. utils/combine_ui.py:126-135 wraps execute_join in try/except, so the user would see "Could not attach X: 'DataFrame' object has no attribute 'str'" — unactionable, but not a crash. (2) find_key_candidates still silently drops the duplicated label (_key_tokens returns None at join_doctor.py:253) with no reason given: in case A the true key SEQN matches perfectly yet candidates == [] and suggest_best is None, and utils/combine_ui.py:99 then tells the user "**<file>** has no column that lines up with your data, so it cannot be attached" — which is false. That is a wrong diagnosis, not silent data corruption, so it stays minor.

**Suggested fix:** In repair_keys() and execute_join(), perform the same isinstance(df[key], pd.DataFrame) check diagnose_join() already does at join_doctor.py:363 and raise the identical worded ValueError. In _key_tokens(), instead of returning None for a duplicated label, record the skip and have find_key_candidates/suggest_best surface it (e.g. a `skipped_columns` reason list) so combine_ui can say "'SEQN' appears twice in this file — fix the duplicate name and it will line up" rather than "no column lines up".

```
$ python3 .../adj/f09.py
pandas 3.0.3
--- A: find_key_candidates / suggest_best on dup-label frame ---
candidates: []
suggest_best: None
--- B: diagnose_join on dup-label frame ---
RAISED ValueError : The column 'SEQN' appears more than once in one of these files. Rename or remove the duplicate before joining.
--- C: only the duplicated label skipped; true key dropped silently ---
candidates: [('pid', 'pid', 'high')]
suggest_best: ('pid', 'pid', 'high')
--- D: execute_join(repair=False) on dup-label frame ---
RAISED AttributeError : 'DataFrame' object has no attribute 'str'
--- E: execute_join(repair=True) on dup-label frame ---
RAISED AttributeError : 'DataFrame' object has no attribute 'str'
--- F: repair_keys on dup-label frame ---
RAISED AttributeError : 'DataFrame' object has no attribute 'str'
--- G: diagnose_join on l2 (SEQN dup) ---
RAISED ValueError : The column 'SEQN' appears more than once in one of these files. Rename or remove the duplicate before joining.

$ python3 .../adj/f09b.py   (realistic UI path: user is offered 'pid')
UI would offer: pid <-> pid high
headline: 'pid' and 'pid' share 3 IDs (100% of the first file, 100% of the second file).
summary: Result: **3 rows** — matching on 3 shared IDs, keeping only IDs found in both files.
blocking: [] warnings: ['Both files have column(s) named SEQN. They will be kept side by side with suffixes so nothing is overwritten.'] notes: []
merged rows: 3 cols: ['SEQN_demo', 'age', 'SEQN_demo', 'pid', 'SEQN_labs', 'glucose']
predicted 3 actual 3

$ import_doctor.diagnose() on the same dup-label frame
duplicate_columns | critical | 1 column name(s) appear more than once | dedupe_columns | high
```

---

### 13 — Duplicate column labels crash three checks; diagnose() silently swallows the crashes and drops unrelated findings frame-wide

**partially_fixed** · severity minor

**What remains:** Only ONE of the three crashing checks was hardened. Commit a1426ef added `isinstance(..., pd.DataFrame)` guards at ml/import_doctor.py:309 (check_numeric_sentinels) and :369 (check_text_missing_tokens), so the critical sentinel finding now survives (pre-fix diagnose returned only ['duplicate_columns']; today it returns ['duplicate_columns', 'sentinel_missing__age']). But TWO checks still raise on a duplicated label because `df[c]` returns a DataFrame:
  - ml/import_doctor.py:242 `empty_cols = [c for c in df.columns if df[c].notna().sum() == 0 and not _is_unnamed(c)]` (check_empty_rows_and_columns)
  - ml/import_doctor.py:507-508 `const = [c for c in df.columns if df[c].notna().sum() > 0 and df[c].nunique(dropna=True) == 1]` (check_constant_columns)
and diagnose() at ml/import_doctor.py:587-592 still does a bare `except Exception: continue` with no record that a check failed. Net effect today: on any frame with a repeated column name, the `empty_columns`, `empty_rows` and `constant_columns` findings vanish frame-wide with no trace — f13b shows a file whose empty `notes` column and constant `site` column are both reported when labels are unique and both silently disappear when they are not. Severity stays minor rather than critical-adjacent because the user is NOT given a clean bill of health (the critical `duplicate_columns` finding still fires), and applying its dedupe fix restores the lost findings on the next diagnose().

**Suggested fix:** Apply the same guard already used at import_doctor.py:309 and :369 to check_empty_rows_and_columns and check_constant_columns — resolve each label via df.iloc[:, i] (or skip labels where `isinstance(df[c], pd.DataFrame)`) instead of df[c]. Separately, make diagnose()'s swallow visible: in the `except Exception` at :589, append an info/warning ShapeFinding naming the check that could not run, so 'no findings' never silently means 'the check crashed'.

```
$ python3 /tmp/claude-0/-home-user-tabular-ml-lab/07f184b8-6f8b-5f93-9930-b6e30849812e/scratchpad/audit/adj/f13.py
unique names   -> ['sentinel_missing__age', 'empty_columns'] | Found 1 needing attention, 1 worth checking.
duplicated 'bp'-> ['duplicate_columns', 'sentinel_missing__age'] | Found 2 needing attention.
--- per-check probe on the duplicated-label frame ---
  check_header_in_later_row        OK   -> []
  check_duplicate_columns          OK   -> ['duplicate_columns']
  check_unnamed_columns            OK   -> []
  check_empty_rows_and_columns     DIES ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
  check_footer_rows                OK   -> []
  check_numeric_sentinels          OK   -> ['sentinel_missing__age']
  check_text_missing_tokens        OK   -> []
  check_numeric_stored_as_text     OK   -> []
  check_categorical_variants       OK   -> []
  check_constant_columns           DIES ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
  check_wide_repeated_measures     OK   -> []
--- recovery via the dedupe fix ---
  after dedupe fix -> ['bp', 'bp_1', 'age', 'notes'] ['sentinel_missing__age', 'empty_columns']
  desc: Renamed duplicate column names so each is unique.

$ python3 .../adj/f13b.py     # frame whose only OTHER problems are an empty col + a constant col
WITHOUT duplicate labels -> ['empty_columns', 'constant_columns']
    Found 1 worth checking, 1 note.
WITH duplicate labels    -> ['duplicate_columns']
    Found 1 needing attention.

LOST silently: ['constant_columns', 'empty_columns']

after dedupe fix -> ['bp', 'bp_1', 'site', 'notes'] ['empty_columns', 'constant_columns']

--- baseline, pre-fix code (git show a1426ef~1:ml/import_doctor.py) for comparison ---
PRE-FIX dup bp   -> ['duplicate_columns']
  check_empty_rows_and_columns     DIES ValueError: The truth value of a Series is ambiguous. ...
  check_numeric_sentinels          DIES TypeError: int() argument must be a string, a bytes-like object or a real number, not 'Series'
  check_constant_columns           DIES ValueError: The truth value of a Series is ambiguous. ...
```

---

### 18 — Large files: independent per-side row sampling in `_prep` destroys measured key overlap - real overlaps are reported as unlinkable or badly undercounted

**partially_fixed** · severity major

**What remains:** The worst part is gone: `_prep` and `_SAMPLE_ROWS` no longer exist (`hasattr` returns False for both). They were replaced by `_key_tokens` (ml/join_doctor.py:240-269), which canonicalises a column's DISTINCT values rather than a random row sample. Every one of the finding's six reported sub-cases now returns the truth exactly - 20k/20k reports 0.5/0.5/10000/high instead of 0.1242/0.1242/621/low; the NHANES headline reads '8,366 IDs (82% of demographics, 100% of labs)' instead of '3,185 IDs (64%, 64%)'; the answer is now row-order-independent (10175 both in-order and shuffled, was 5000 vs ~1174); and the repeated-measures join reports n_matched 60000 with suggest_best returning SEQN<->SEQN instead of 432 and None.

WhAT REMAINS: `_key_tokens` still truncates each side independently when a column has more than `_MAX_DISTINCT = 200_000` distinct values, via `uniques.iloc[:_MAX_DISTINCT]` (ml/join_doctor.py:265-266). That is a head-truncation in row order, so above 200k distinct keys the two sides are once again compared as two different subsets of the value space, and every symptom of the original finding returns:
  (a) 300k vs 300k with a true 150,000-ID overlap is reported as 'share 50,000 IDs (25% of demographics, 25% of labs)', confidence drops to 'low', and `suggest_best` returns None - i.e. the true key is withheld and utils/combine_ui.py:90-95 shows the user 'No shared ID was found between your data so far and <file>. These files may not describe the same people'. That is the exact 'real overlap reported as unlinkable' symptom in the finding title.
  (b) The answer is still row-order dependent in this regime: 250k identical IDs on both sides reports 200,000 in original order but 160,005 after shuffling one side (truth 250,000).
  (c) Worse than the 300k case, the truncated numbers can be asserted at HIGH confidence. 240k shuffled string IDs with a true 100%/100% overlap yields the headline "'patient_id' and 'patient_id' share 166,746 IDs (83% of file A, 83% of file B)." at confidence 'high' - which utils/combine_ui.py:110 prints verbatim as the caption under the key picker. Under this app's contract that is a confidently-wrong assertion; the chosen key is right but the stated overlap is off by 73,254 IDs.
  (d) `find_key_candidates` and `diagnose_join` disagree with each other on the same inputs - the candidate says 166,746 matched, `diagnose_join` says 240,000 - so the number under the picker and the number above the Merge button differ.
  Boundary confirmed exactly: 200,000 distinct is correct, 200,001 already reports 199,999.
  The module docstring at ml/join_doctor.py:36-38 and the `_key_tokens` docstring both claim this guard works 'without ever comparing two different random subsets', which is not true above the threshold.

**Suggested fix:** In `_key_tokens` (ml/join_doctor.py:264-266), stop truncating by position. Either (1) make the cap value-deterministic so both sides retain the SAME region of the key space - e.g. keep values whose stable hash falls in a fixed bucket, `uniques[uniques.map(lambda v: hash_stable(v) % 100 < k)]`, which yields an unbiased, order-independent, mutually-consistent overlap estimate that can be scaled back up - or (2) if truncation happens at all, record it on the KeyCandidate (`truncated: bool`) and make `headline()` say 'at least N IDs' rather than an exact count, cap `confidence` below 'high', and never let `suggest_best` withhold a key merely because the truncated coverage fell under threshold. Option (1) is preferable because it also restores the numeric agreement with `diagnose_join`, which correctly uses the full value sets.

```
--- f18.py (all six reported sub-cases) ---
has _prep: False | has _SAMPLE_ROWS: False | _MAX_DISTINCT: 200000

--- 1. reported case: 20k vs 20k, true 50% overlap ---
suggest_best: ('SEQN', 'SEQN', 'high')
coverage_l/r, n_matched, conf: 0.5 0.5 10000 high    (truth: 0.5 0.5 10000 high)
diagnose_join matched_keys: 10000 (truth 10000)

--- 2. mechanism check: token sets ---
token overlap: 10000  true overlap: 10000

--- 3. NHANES-shaped, labs a strict subset of demographics ---
'SEQN' and 'SEQN' share 8,366 IDs (82% of demographics, 100% of labs). | high
truth: 8,366 IDs (82% of demographics, 100% of labs)

--- 4. does the answer depend on ROW ORDER? ---
in-order  n_matched: 10175 (truth 10175)
shuffled  n_matched: 10175 (truth 10175)

--- 5. repeated-measures join ---
n candidates: 1
n_matched: 60000 (truth 60000) conf: high
suggest_best: ('SEQN', 'SEQN', 'high')

--- 6. very large: above _MAX_DISTINCT (200k) ---
n candidates: 1
n_matched: 50000 (truth 150000) cov_l: 0.25 cov_r: 0.25 conf: low
diagnose_join matched_keys: 150000 (truth 150000)

--- f18b.py (residual defect isolated) ---
=== residual: >_MAX_DISTINCT (200k) distinct keys -> head-truncation per side ===
suggest_best: None
headline: 'SEQN' and 'SEQN' share 50,000 IDs (25% of demographics, 25% of labs).
confidence: low  n_matched: 50000 (truth 150000)

=== does row ORDER change the answer above 200k distinct? ===
in-order  n_matched: 200000 (truth 250000)
shuffled  n_matched: 160005 (truth 250000)
shuffled  cov_l/cov_r/conf: [(0.8, 0.8, 'high')]

=== realistic string IDs, both shuffled, 240k distinct ===
headline: 'patient_id' and 'patient_id' share 166,746 IDs (83% of file A, 83% of file B). | conf: high
diagnose_join matched_keys: 240000 (truth 240000)

--- boundary probe ---
200000 -> n_matched 200000 cov 1.0 conf high (truth 200000 1.0 )
200001 -> n_matched 199999 cov 1.0 conf high (truth 200001 1.0 )
220000 -> n_matched 181794 cov 0.909 conf high (truth 220000 1.0 )
```

---

### 22 — Blank/NaN join keys are invisible to diagnose_join: inner join silently drops those rows with no warning, and predicted_rows is wrong for left/outer joins

**partially_fixed** · severity minor

**What remains:** Both symptoms named in the title are gone on the finding's own input: the blank rows are now announced by an explicit warning (they are no longer invisible), and predicted_rows is exact for all four join types (left 5=5, outer 6=6, previously 2 vs 5 and 3 vs 6). What remains is a narrower, still user-visible wrong statement in the SAME blank-key warning, reachable only when blanks exist on BOTH sides: ml/join_doctor.py:471 computes one keep/drop verdict for both counts — `kept = how in ('left','outer') and n_missing_left or how in ('right','outer') and n_missing_right` — and then applies it to the concatenated sentence. So for a LEFT join with 3 blank-ID rows left and 2 right, the app tells the researcher '3 in the first file and 2 in the second file row(s) have no ID at all ... They are kept but will have no matching information attached', when in fact the 2 right-hand rows are dropped (mirror-image error for a RIGHT join: it claims the 3 left rows are kept, and they are dropped). The delivered row count is still correct (5 and 4, verified against execute_join), so this is wrong prose about data loss rather than a wrong promised row count — minor, but it is the app asserting 'nothing was lost' about rows it did lose.

**Suggested fix:** In diagnose_join (ml/join_doctor.py:465-476) build the sentence per side instead of once: for each of n_missing_left/n_missing_right, decide kept = how in ('left','outer') for the left side and how in ('right','outer') for the right side, and emit '<n> row(s) in <file> have no ID at all — they are kept with no matching information attached' vs '... — they cannot be matched and will be dropped' separately (join with '; ' when both apply). Also fixes the 'N in the first file row(s)' word order.

```
$ python3 f22.py          # the finding's own input: blanks on the left only
Result: **2 rows** — matching on 2 shared IDs, keeping only IDs found in both files.
warnings: ["3 in the first file row(s) have no ID at all (blank or 'unknown'). They cannot be matched and will be dropped."]
notes:    ['1 row(s) of the second file have no match and will be dropped.']
unmatched_left: 0 (should reflect the 3 blank-ID rows)
inner  predicted=2 actual=2
left   predicted=5 actual=5
right  predicted=3 actual=3
outer  predicted=6 actual=6
(was: left predicted=2 actual=5 MISMATCH, outer predicted=3 actual=6 MISMATCH, warnings: [])

=== the merged frame keeps the blank-ID rows sensibly ===
    id  a    b
0    1  0  0.0
1    2  1  1.0
2  NaN  2  NaN
3  NaN  3  NaN
4  NaN  4  NaN

$ python3 f22_msg.py      # RESIDUAL: blanks on BOTH sides (3 left, 2 right)
--- inner: predicted=2 actual=2
    WARN: 3 in the first file and 2 in the second file row(s) have no ID at all (blank or 'unknown'). They cannot be matched and will be dropped.
--- left: predicted=5 actual=5
    WARN: 3 in the first file and 2 in the second file row(s) have no ID at all (blank or 'unknown'). They are kept but will have no matching information attached.
--- right: predicted=4 actual=4
    WARN: 3 in the first file and 2 in the second file row(s) have no ID at all (blank or 'unknown'). They are kept but will have no matching information attached.
--- outer: predicted=7 actual=7
    WARN: 3 in the first file and 2 in the second file row(s) have no ID at all (blank or 'unknown'). They are kept but will have no matching information attached.

truth for a LEFT join: the 3 left blanks are kept, the 2 right blanks are dropped
```

---

### 24 — find_key_candidates scores keys on two independent 5,000-row samples, so overlap counts and coverage percentages are wrong on any file over 5,000 rows — a perfect key is reported as 72% at 'high' confidence, or withheld entirely

**partially_fixed** · severity major

**What remains:** The 5,000-row sampling is genuinely gone — _key_tokens (ml/join_doctor.py:240-269) no longer samples rows, and all four cases in the finding, plus 50k/120k/partial-overlap variants, now report the exact ID count and exact coverage percentages. But the same failure CLASS survives above the new `_MAX_DISTINCT = 200_000` cap: ml/join_doctor.py:265-266 does `uniques = uniques.iloc[:_MAX_DISTINCT]`, i.e. it keeps the first 200,000 distinct values IN ROW ORDER on each side independently. When two files list the same subjects in different orders — the normal case — those are once again two different subsets, so the measured overlap collapses exactly as before, just at a 40x higher threshold. Observed: 250,000 x 250,000 frames with IDENTICAL, perfectly-matching ID sets are reported as "'SEQN' and 'SEQN' share 160,001 IDs (80% of the first file, 80% of the second file)" at 'high' confidence (truth: 250,000 / 100% / 100%), and at 1,000,000 x 1,000,000 every candidate falls to 'low' so suggest_best returns None and a perfect key is withheld entirely — verbatim both symptoms the finding names. Because only 'high' is pre-selected in the UI, the 250k case is the app asserting a wrong number, hence still major. Blast radius is smaller than before in one respect worth recording: diagnose_join is NOT truncated (it uses full value_counts), so the promised ROW COUNT above 200k is still exact — it is the key-suggestion headline and the confidence gate that lie. Also note the truncation makes 'age' vs 'glucose' (two plain 0..N-1 counters) score 200,000 matched / 100% / 100% in the 1M case, ranked above the real key.

**Suggested fix:** Do not truncate the distinct-value sets asymmetrically. Either (a) drop the row-order truncation and canonicalise all distinct values — the cost is linear and _key_tokens already rejects low-uniqueness columns first — or (b) if a cap is required for pathological files, make both sides deterministic on VALUE, not position: hash each canonical token and keep tokens whose hash falls in the same fixed fraction of the space on both sides (a shared-domain minhash), then scale n_matched back up by that fraction and mark the candidate as estimated. Whichever is chosen, when the cap actually bites, the headline must say the number is an estimate and confidence must be capped below 'high' so the UI does not pre-select it.

```
$ python3 f24.py          # the finding's own four cases — all now correct
4000 x 3500 -> ("'SEQN' and 'SEQN' share 3,500 IDs (88% of the first file, 100% of the second file).", 'high') | truth: 3500 shared IDs
9254 x 9254 -> ("'SEQN' and 'SEQN' share 9,254 IDs (100% of the first file, 100% of the second file).", 'high') | truth: 9254 shared IDs
9254 x 8000 -> ("'SEQN' and 'SEQN' share 8,000 IDs (86% of the first file, 100% of the second file).", 'high') | truth: 8000 shared IDs
20000 x 18000 -> ("'SEQN' and 'SEQN' share 18,000 IDs (90% of the first file, 100% of the second file).", 'high') | truth: 18000 shared IDs
(was: 9,254x9,254 -> 'share 5,000 IDs'; 9,254x8,000 -> 'share 3,614 IDs (72%, 72%)' high; 20,000x18,000 -> NO KEY PROPOSED)

=== larger / partial-overlap variants, still under the new cap ===
50000 x 50000 off=0 -> ('...share 50,000 IDs (100%, 100%).', 'high', 50000) | truth: 50000
120000 x 100000 off=0 -> ('...share 100,000 IDs (83%, 100%).', 'high', 100000) | truth: 100000
30000 x 30000 off=15000 -> ('...share 15,000 IDs (50%, 50%).', 'high', 15000) | truth: 15000

$ python3 f24_maxdistinct.py    # RESIDUAL: above _MAX_DISTINCT = 200_000
250000 x 250000 -> ("'SEQN' and 'SEQN' share 200,000 IDs (100% of the first file, 100% of the second file).", 'high') | truth: 250000 shared IDs   <-- COUNT/PCT WRONG
300000 x 220000 -> ("'SEQN' and 'SEQN' share 200,000 IDs (100% of the first file, 100% of the second file).", 'high') | truth: 220000 shared IDs   <-- COUNT/PCT WRONG
shuffled 250000 x 250000 -> ("'SEQN' and 'SEQN' share 160,001 IDs (80% of the first file, 80% of the second file).", 'high') | truth: 250000   <-- COUNT/PCT WRONG

$ python3 f24_big.py            # RESIDUAL: the 'withheld entirely' symptom returns
1,000,000 x 1,000,000 (identical ID sets, shuffled) -> NO KEY PROPOSED
truth: 1000000 shared IDs
   cand: SEQN SEQN 40166 20% 20% low
   cand: age glucose 200000 100% 100% low
   cand: SEQN glucose 39802 20% 20% low
```

---

### 26 — NUMERIC_SENTINELS omits the positive NHANES/SPSS 7/8/9, 77/88/99 and 7777/8888 missing-code families, so narrow-range coded columns pass silently

**partially_fixed** · severity major

**What remains:** The headline claim is fixed: NUMERIC_SENTINELS (import_doctor.py:35-40) now carries 7/8/9, 66/77/88/99, 666/777/888/999 and 6666/7777/8888/9999, gated by the new `coded` test (integral + <=15 distinct, import_doctor.py:318-326) so a continuous column containing a real 9 is still not flagged. The original repro's case 1 now yields a critical finding instead of 'No structural problems detected.', and the multi-code case that even the monkey-patch could not fix is now detected.

WHAT REMAINS: the sentinel VALUE LIST attached to the finding drops 7 on any 1-5 scale, so the one-click fix leaves the 'refused' code in the data. `rest` correctly excludes all candidates before measuring the spread, but the outlier test is strict `v > hi + 0.5 * spread` (import_doctor.py:335-339). For a 1-5 Likert lo=1, hi=5, spread=4, so the threshold is exactly 7.0 and `7 > 7` is False. Concretely, on the canonical NHANES pattern (1-5 answers, 7=refused, 9=don't know) the app says 'Found 9 (3x)', offers 'Treat 9 as missing', reports 'Recoded 9 as missing' — and the column mean afterwards is 3.364 instead of the true 3.000, with three 7s still counted as real answers. The user has been shown a critical finding, applied the app's own fix, and been told it succeeded, while the column is still contaminated and nothing says so. This contradicts the code's own stated intent in the comment at import_doctor.py:333-334 ('two sentinels mask each other'). Same defect for the 7/8/9 triple: proposed values are [8.0, 9.0], 7s remain, mean 3.25 vs true 3.00.

Separately (noted, not counted as the defect): a 7777/8888 code inside a wide-range column (income 12k-102k) is still missed because 7777 is not 0.5*spread below the minimum — 'No structural problems detected.' with a reported mean of 44,024 vs a true 58,300. Narrow-range 7777/8888 IS caught, so the code family is recognised; this is the deliberate 'far outside' tradeoff biting on wide columns rather than a missing-sentinel bug.

**Suggested fix:** In check_numeric_sentinels (ml/import_doctor.py:335-339), once ANY sentinel in a coded column clears the outlier test, sweep the rest of its code family in as well rather than testing each independently. E.g. after computing `hits`, add: if `coded` and hits, extend hits with every v in `present` that shares the same magnitude family as a hit (7/8/9; 66/77/88/99; 666/777/888/999; 6666/7777/8888/9999) and sits above `hi` (or below `lo`). Minimally, change the boundary to a gap test that does not exclude the smallest member of the family, e.g. `v > hi + min(0.5 * spread, ...)` combined with `v >= hi + 2`, so 7 on a 1-5 scale qualifies. Add a regression test asserting params['values'] == [7.0, 9.0] for pd.DataFrame({'q': [1,2,3,4,5]*6 + [7,7,7,9,9,9]}).

```
f26.py:
NUMERIC_SENTINELS = (-9999.0, -999.0, -99.0, -88.0, -77.0, -9.0, -8.0, -7.0, -1.0, 7.0, 8.0, 9.0, 66.0, 77.0, 88.0, 99.0, 666.0, 777.0, 888.0, 999.0, 6666.0, 7777.0, 8888.0, 9999.0, 99999.0)

=== CASE 1: 1-5 Likert with 9 = missing ===
summarize: Found 1 needing attention.
ids: ['sentinel_missing__diet_quality']
   detail: Found 9 (5x) — far outside the rest of the column (1 to 5). | severity: critical | confidence: medium

=== CASE 2: multi-code 7 AND 8 AND 9 in one item ===
sentinel ids: ['sentinel_missing__q']
   sentinel_missing__q | Found 8 (2x), 9 (2x) — far outside the rest of the column (1 to 5). | conf: medium

=== CASE 3: 77/88/99 family (2-digit codes) ===
   sentinel_missing__cigs_per_day | Found 77 (2x), 88 (2x), 99 (2x) — far outside the rest of the column (0 to 25).

=== CASE 5: false-positive guard ===
sentinel ids: []            <- continuous glucose column containing a real 7/8/9: correctly NOT flagged

f26b.py (the residual, with apply_fix run end-to-end):
=== NHANES classic: 1-5 Likert, 7=refused, 9=don't know ===
flagged: ['sentinel_missing__diet_quality']
  detail : Found 9 (3x) — far outside the rest of the column (1 to 5).
  label  : Treat 9 as missing in 'diet_quality'
  values : [9.0]
  applied: Recoded 9 as missing in 'diet_quality'.
  mean AFTER the app's fix : 3.3636363636363638
  TRUE mean (1-5 only)     : 3.0
  7s still in column       : 3

=== 7/8/9 all three, 1-5 scale ===
  values proposed: [8.0, 9.0] | detail: Found 8 (2x), 9 (2x) — far outside the rest of the column (1 to 5).
  mean after fix: 3.25 | true mean: 3.0 | 7s remaining: 2

=== 7777/8888 family, narrow real range ===
flagged: ['sentinel_missing__kcal'] ['Found 7777 (2x), 8888 (2x) — far outside the rest of the column (1200 to 2250).']

=== 7777/8888 family, wide real range (income) ===
flagged: [] []
reported mean: 44023.57142857143 | true mean: 58300.0
summarize: No structural problems detected.
```

---

### 28 — Key detection collapses above ~10,000 rows: `_prep` samples both sides independently, so the true key's measured overlap decays as 5000/N

**partially_fixed** · severity minor

**What remains:** The described collapse is genuinely fixed at the scale the finding named. `_prep` is gone; `_key_tokens` (ml/join_doctor.py:240-269) now cheaply rejects non-identifying columns and canonicalises only the DISTINCT values of survivors, with an explicit docstring saying why sampling was wrong. The recorded repro inverts completely: SEQN/SEQN now comes back at 'high' confidence with n_matched=17000 and coverage_right=1.0000, exactly agreeing with diagnose_join's ground truth of 17000, where it previously returned None / 'low' / 0.2512.

WHAT REMAINS: the same mechanism survives at a 40x-higher threshold. `_MAX_DISTINCT = 200_000` (ml/join_doctor.py:38) truncates with `uniques.iloc[:_MAX_DISTINCT]` — a head, not a random sample, so it is safe when both files happen to be in the same order, but for files ordered differently it is again two different subsets and the measured overlap decays as 200000/N. Two observable consequences:
  (1) At 250k rows per side with the right file sorted descending, the key IS still proposed at 'high' confidence, but the sentence the UI prints is wrong: utils/combine_ui.py:110 renders `chosen.headline(...)`, which here reads "'SEQN' and 'SEQN' share 150,000 IDs (75% of the first file, 75% of the second file)" when the truth is 250,000 IDs, 100% and 100%. Three lines later the same screen prints diagnose_join's exact figures, so the user sees two contradictory counts.
  (2) At 600k rows per side in differing order, coverage falls to 0.3330, the true key drops back to 'low', and suggest_best returns None — the original symptom, verbatim, just further out. The user is shown 'No shared ID was found… you can pick the columns yourself' (utils/combine_ui.py:90-94) for two files that share all 600,000 IDs.

Severity is only minor rather than major because (a) 200k+ distinct-key research files are well outside the NHANES-scale case the finding was about, (b) diagnose_join does NOT use the cap — it normalises the full column — so the row-count promise and plain_summary stay exact and no data is silently lost, and (c) the failure mode at (2) is to withhold and ask, which is the safe direction.

**Suggested fix:** Two independent fixes. For the wrong headline: stop letting truncated counts be presented as facts — when `len(uniques) > _MAX_DISTINCT`, set a `truncated=True` flag on the KeyCandidate and have `headline()` say 'at least 150,000 IDs (sampled)' rather than an exact figure, or have combine_ui.py:110 prefer diagnose_join's exact matched_keys once a key is chosen. For the residual collapse: make the truncation order-independent so both sides pick the SAME subset — e.g. keep values by a stable hash bucket (`hash(token) % k == 0` sized to ~200k expected) instead of `iloc[:_MAX_DISTINCT]`; identical values then survive on both sides regardless of row order and coverage stays unbiased at any N. Regression test: 600k rows per side, right side shuffled, assert suggest_best is not None and its coverage is within a few percent of 1.0.

```
f28.py — the recorded repro, verbatim scenario (nL=19942, nR=17000):
suggest_best: KeyCandidate(left_col='SEQN', right_col='SEQN', coverage_left=0.8524721692909437, coverage_right=1.0, n_matched=17000, left_unique=19942, right_unique=17000, ... name_similarity=1.0, index_like=False)
  -> SEQN SEQN high cov_l=0.8525 cov_r=1.0000 n_matched= 17000
  headline: 'SEQN' and 'SEQN' share 17,000 IDs (85% of the first file, 100% of the second file).
cand: SEQN SEQN high 0.8525 1.0 17000
truth (diagnose_join matched_keys): 17000
elapsed 0.60s

(was: suggest_best -> None; SEQN SEQN low 0.2512 1256)

f28b.py — probing the new _MAX_DISTINCT = 200_000 cap:
[A] nL=19942 nR=17000 right_order=asc
   suggest_best -> SEQN/SEQN conf=high cov_l=0.8525 cov_r=1.0000 n_matched=17000 | truth=17000
   elapsed 0.3s

[B] nL=250000 nR=250000 right_order=desc
   suggest_best -> SEQN/SEQN conf=high cov_l=0.7500 cov_r=0.7500 n_matched=150000 | truth=250000
   headline: 'SEQN' and 'SEQN' share 150,000 IDs (75% of the first file, 75% of the second file).
   elapsed 3.7s

[C] nL=600000 nR=600000 right_order=shuffled
   suggest_best -> None  (WITHHELD)   truth matched_keys = 600000
   low-conf cand: SEQN SEQN low cov_l=0.3330 cov_r=0.3330 n_matched=66600
   low-conf cand: sbp hba1c low cov_l=1.0000 cov_r=1.0000 n_matched=200000
   elapsed 10.6s
```

---

### 35 — coerce_numeric silently merges incompatible units (mg/dL + mmol/L, kg + lb) into one column at 'high' confidence, with a detail message that never discloses the mixing

**partially_fixed** · severity major

**What remains:** The recorded repro is fixed: a `_units_present` guard (ml/import_doctor.py lines 109-116) plus `if len(units) > 1` (line 427) now suppresses the numeric_as_text finding entirely and emits `mixed_units__<col>` at severity=critical, confidence=low, fix_kind='none', with a detail that explicitly names both units. Same for kg+lb. But the guard is both cardinality- and order-dependent, and the original silent merge still reproduces verbatim through two holes on the same input class:

(1) CASE E — `_units_present` only scans `s.dropna().astype(str).unique()[:500]`. A realistically-sized multi-site export (550 distinct glucose strings, site A's 510 mg/dL rows first, then mmol/L) makes `_units_present` return only {'mg/dl'}. The guard never fires; the app emits numeric_as_text at confidence='high' / auto_suggestable=True, shows three mg/dL-only examples as evidence, and silently merges the scales (mean 88.94 vs the true mg/dL mean 95.45). Nothing in title, detail, fix_label or the applied desc mentions a second unit. CASE F is the identical data with the rows reordered and the guard DOES fire — so whether the user gets a critical block or a pre-selected silent corruption depends on export row order.

(2) CASE D — the guard requires two *recognised* units, so the very common form where one site omits the unit ('95', '102', ...) and the other writes it ('5.3 mmol/L') passes at high/auto-suggestable confidence and produces the same clinically inverted mean of 63.6 that the original finding called out.

Both remaining paths are a pre-selected 'high'-confidence assertion that is wrong with zero disclosure, which is the silent-wrongness the contract ranks worst; by the stated standard ('a high-confidence recommendation that is wrong is CRITICAL') the residual is arguably critical, but I hold it at major since it now requires either >500 distinct values or an implicit unit rather than firing on every mixed-unit column.

**Suggested fix:** In ml/import_doctor.py `_units_present` (lines 109-116), drop the `[:500]` slice and scan the full distinct set — vectorise with `s.dropna().astype(str).str.strip().str.extract(_TRAILING_UNIT)` and take the non-null lowercased values, which is cheaper than the current Python loop anyway, so there is no reason to cap it. Separately, in `check_numeric_stored_as_text` (line 427), treat a partial unit as suspicious too: compute the fraction of non-null values carrying a recognised unit, and when that fraction is strictly between 0 and 1 do not return 'high' — either emit the same `mixed_units__<col>` block or, to avoid over-blocking legitimate '<0.01'-style rows, keep the coerce_numeric finding but force confidence='low' and add the counts to `detail` (e.g. "6 value(s) carry no unit and 4 carry 'mmol/L' — they may be on different scales") so it is never pre-selected and the mixing is disclosed.

```
=== CASE A: mg/dL + mmol/L, site-sorted (the strengthened repro) ===
all finding ids: [('mixed_units__glucose', 'critical', 'low', False)]
numeric_as_text findings: []
  -- mixed_units__glucose
     detail: Found mg/dl, mmol/l in the same column.
     label : Cannot fix automatically — convert to one unit first | kind: none
     applied desc: No automatic change is possible here; this needs a human decision.

=== CASE B: kg + lb ===
  mixed_units__weight | critical low False | Found kg, lb in the same column. | kind: none

=== CASE D: mixed units where one unit is IMPLICIT (bare numbers + mmol/L) ===
  numeric_as_text__glucose | warning high True | 100% of values parse as numbers after removing units, commas and comparison signs (e.g. '95', '102', '110').
     -> Converted 'glucose' to numeric (removing units, separators and comparison signs).
     values: [95.0, 102.0, 110.0, 99.0, 88.0, 120.0, 5.3, 5.7, 6.1, 4.9] | mean = 63.6

=== CASE E: the SAME two units, but >500 distinct strings before the 2nd unit appears ===
  distinct values: 550 | rows: 550
  _units_present() sees: ['mg/dl']
  numeric_as_text__glucose | warning high auto=True
     detail: 100% of values parse as numbers after removing units, commas and comparison signs (e.g. '70.0 mg/dL', '70.1 mg/dL', '70.2 mg/dL').
     applied: Converted 'glucose' to numeric (removing units, separators and comparison signs).
     MERGED mean = 88.94 | mg/dL-only mean = 95.45

=== CASE F: same as E but mmol/L rows come FIRST (so they are inside the 500 window) ===
  _units_present() sees: ['mg/dl', 'mmol/l']
  [('mixed_units__glucose', 'low', False)]
```

---

### 47 — Duplicated key column name crashes normalize_key and therefore diagnose_join / execute_join / repair_keys (AttributeError), and silently blanks find_key_candidates

**partially_fixed** · severity minor

**What remains:** Three of the four symptoms survive.

FIXED: diagnose_join (ml/join_doctor.py:362-367) now guards `isinstance(ls, pd.DataFrame)` and raises a clear, actionable ValueError — acceptable per the contract, not a defect.

STILL BROKEN 1 — execute_join and repair_keys are NOT guarded. repair_keys (line 507) calls normalize_key(l2[left_key]) on a DataFrame; normalize_key line 112-114 does `out.dropna().astype(str).str.lower()` and dies with `AttributeError: 'DataFrame' object has no attribute 'str'`. That is a raw internal traceback, exactly as originally reported. utils/combine_ui.py:129 wraps execute_join in try/except and surfaces it as "Could not attach X: 'DataFrame' object has no attribute 'str'" — meaningless to a researcher.

STILL BROKEN 2 — find_key_candidates still silently blanks the duplicated column (_key_tokens returns None at line 254 with no signal). f47b.A confirms other columns survive, but for the reported frame it returns [] and utils/combine_ui.py:89-99 then tells the user "**other** has no column that lines up with your data, so it cannot be attached" — a confidently wrong cause for a file whose SEQN lines up perfectly. The duplicate label is never named anywhere in the UI.

STILL BROKEN 3 — _slug (line 569) still truncates at 20 chars, so two 2019/2020 cohort files collide and execute_join delivers a frame with TWO columns literally named 'bmi_cohort_baseline_meas' — silently, with no warning, and diagnose_join's collision warning ('kept side by side with suffixes so nothing is overwritten') is now false. Worse, this is self-inflicted: attaching a third file to that merged frame hits the duplicate-label path from symptom 1, so execute_join dies with the same AttributeError on a frame the app itself produced.

**Suggested fix:** Guard repair_keys/execute_join the same way diagnose_join is guarded (or make normalize_key reject a DataFrame with the same ValueError); have _key_tokens record why a column was skipped so find_key_candidates can report 'SEQN appears twice' instead of returning []; and make _slug de-duplicate — e.g. append a short hash of the full name when the 20-char truncation collides, and assert the merged frame has unique column labels before returning.

```
left columns: ['SEQN', 'age', 'SEQN']
diagnose_join -> ValueError : The column 'SEQN' appears more than once in one of these files. Rename or remove the duplicate before joining.
execute_join -> AttributeError : 'DataFrame' object has no attribute 'str'
repair_keys -> AttributeError : 'DataFrame' object has no attribute 'str'
candidates: []
suggest_best: None

=== second half: _slug truncation collision ===
'cohort_baseline_meas' 'cohort_baseline_meas'
merged columns: ['SEQN', 'bmi_cohort_baseline_meas', 'bmi_cohort_baseline_meas']
  SEQN  bmi_cohort_baseline_meas  bmi_cohort_baseline_meas
0    1                      22.0                      22.5
1    2                      28.0                      28.5

--- f47b.py ---
=== A. duplicated label: does an ALTERNATE valid key still get proposed? ===
candidates: [('patient_id', 'patient_id', 'high')]

=== B. what the UI (utils/combine_ui.py) shows when the key label is duplicated ===
find_key_candidates -> []
UI message shown to user: "**right.csv** has no column that lines up with your data, so it cannot be attached."  <-- the real cause (duplicate column label 'SEQN') is never named

=== C. duplicate output columns from _slug collision break the next step ===
merged columns: ['SEQN', 'bmi_cohort_baseline_meas', 'bmi_cohort_baseline_meas']
merged['bmi_cohort_baseline_meas'] is a DataFrame
chained diagnose_join -> ValueError : The column 'bmi_cohort_baseline_meas' appears more than once in one of these files. Rename or remove the duplicate before joining.
chained execute_join -> AttributeError : 'DataFrame' object has no attribute 'str'
```

---

### 48 — find_key_candidates samples each side positionally (5,000 rows, random_state=42), so on files over 5,000 rows the reported overlap is a sampling artefact — wrong counts/percentages at "high" confidence, and above ~20k rows the true key is downgraded to "low" or dropped entirely

**partially_fixed** · severity major

**What remains:** The worst part is genuinely gone: the 5,000-row random_state=42 sample is removed. _key_tokens (ml/join_doctor.py:240-269) now canonicalises DISTINCT values, not a row sample, and its docstring names the old bug. All three cases from the finding's section A and the section-B shuffled-cohort case now report the exact truth (9,254 = 9,254 at 'high'; 20,000 no longer degraded to 'low'; the 200k pair no longer returns zero candidates), and results are deterministic across runs.

BUT the same class of defect survives at a higher threshold. _MAX_DISTINCT = 200_000 (line 38) is applied by positional head-truncation at line 265-266 (`uniques = uniques.iloc[:_MAX_DISTINCT]`), so above 200,000 distinct key values the reported overlap is again computed on a subset:

1. Wrong percentages at 'high' confidence, still. 200,000 x 260,000 — the finding's own third case — reports '100% of the second file' when the truth is 200000/260000 = 77%; 60,000 subjects will be dropped and the headline says none will.
2. Wrong counts at 'high' confidence. 300k x 260k reports 'share 200,000 IDs (100% of demographics.csv, 100% of labs.csv)' when the truth is 260,000 IDs, 87% and 100%. The UI (utils/combine_ui.py:110 st.caption(chosen.headline(...))) prints that line immediately above diagnose_join's correct warning that 40,000 rows (13%) will be dropped — the app contradicts itself on the same screen.
3. The true key can still be dropped entirely. Case E: 260,000 left subjects whose 60,000 matching IDs sit after the first 200,000 distinct values -> find_key_candidates returns 0 candidates and suggest_best returns None, so combine_ui.py:90-99 tells the user 'No shared ID was found between your data so far and X' for a join that in fact matches 60,000 subjects (100% of the second file).

Mitigation: diagnose_join itself is exact (no truncation), so a user who picks the key manually gets correct numbers; the wrongness is confined to the proposal/headline stage. Because case E makes the key unfindable in the UI, that mitigation does not always apply.

**Suggested fix:** Do not head-truncate the distinct values used for overlap. Either lift the cap for the intersection computation (a Python set of 260k strings is cheap — the 300k x 260k run above took 3.8s end to end), or, if a cap is kept for memory, propagate it: mark the KeyCandidate as estimated, cap confidence at 'medium' so it is not pre-selected, and change headline() to say 'at least N IDs (measured on the first 200,000 values)' rather than asserting an exact count and 100% coverage.

```
=== A. reported case: different lengths, 100% of left is in right ===
9254 12000 -> ("'SEQN' and 'SEQN' share 9,254 IDs (100% of the first file, 77% of the second file).", 'high') | true matched = 9254 | n candidates = 1 | 0.2s
20000 30000 -> ("'SEQN' and 'SEQN' share 20,000 IDs (100% of the first file, 67% of the second file).", 'high') | true matched = 20000 | n candidates = 1 | 0.4s
200000 260000 -> ("'SEQN' and 'SEQN' share 200,000 IDs (100% of the first file, 100% of the second file).", 'high') | true matched = 200000 | n candidates = 1 | 3.4s

=== B. equal lengths, identical cohort, only row order differs ===
identical 9,000-subject cohort, shuffled -> ("'SEQN' and 'SEQN' share 9,000 IDs (100% of the first file, 100% of the second file).", 'high') | true matched = 9000

=== C. above _MAX_DISTINCT (200,000) ===
250000 250000 -> ("'SEQN' and 'SEQN' share 200,000 IDs (100% of the first file, 100% of the second file).", 'high') | true matched = 250000 | 3.4s
300000 260000 -> ("'SEQN' and 'SEQN' share 200,000 IDs (100% of the first file, 100% of the second file).", 'high') | true matched = 260000 | 3.8s

=== D. repeatability (was random_state-dependent) ===
distinct results over 3 runs: {(9254, 1.0, 0.771167)}

--- f48b.py ---
=== E. >200,000 distinct, with the matching block NOT in the first 200,000 rows ===
suggest_best -> NONE
n candidates = 0
TRUE matched keys (diagnose_join) = 60000
TRUE coverage: left 23% right 100%

=== F. what the user is told vs what the join really does (300k x 260k) ===
proposed: 'SEQN' and 'SEQN' share 200,000 IDs (100% of demographics.csv, 100% of labs.csv). confidence = high
n_matched claimed = 200000 | coverage_left claimed = 100% | coverage_right claimed = 100%
TRUE matched = 260000 | TRUE coverage_left = 87% | TRUE coverage_right = 100%
diagnose_join predicted_rows = 260000
warnings: ['40,000 row(s) of demographics.csv (13%) have no match and will be dropped. Use a left join to keep them.']
```

---
