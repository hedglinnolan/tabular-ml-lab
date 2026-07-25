# The original 48 confirmed findings

Recovered from the stress-test workflow's journal (`wf_e5abb4fe-e32`), which
produced 102 raw findings across 8 families; 48 survived adversarial
verification (11 critical, 30 major, 7 minor).

This file exists because the list was never written down the first time. When
the question "did you address all of those?" came back, the answer had to be
reconstructed from agent transcripts on an ephemeral disk. That must not
happen twice, so the findings live in the repository now.

Each entry keeps the title, the severity as confirmed, the verifier's
reasoning, and the repro exactly as recorded. Dispositions are tracked in
`docs/FINDINGS_LEDGER.md`.

---


## Finding 01

## Title
Null join keys are matched to each other in execute_join, fabricating subjects, while diagnose_join predicts a row count that excludes them — silently wrong, with no missing-ID warning

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro is valid and reproduces verbatim (predicted 1, actual 5). Consolidated version at /tmp/claude-0/-home-user-tabular-ml-lab/07f184b8-6f8b-5f93-9930-b6e30849812e/scratchpad/corrected_repro.py:

import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd, numpy as np
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# 1. Reported case (object dtype, None keys)
L = pd.DataFrame({"seqn": [None, None, "3"], "diet": [11, 22, 33]})
R = pd.DataFrame({"seqn": [None, None, "3"], "chol": [91, 92, 93]})
d = diagnose_join(L, R, "seqn", "seqn", "inner")
m, _ = execute_join(L, R, "seqn", "seqn", "inner")
print(plain_summary(d)); print("actual rows:", len(m)); print(m)
# -> "Result: **1 rows** — matching on 2 shared IDs"; actual 5 rows

# 2. float64 NaN — same fan-out, and ZERO warnings/notes/blocking of any kind
L = pd.DataFrame({"seqn": [np.nan, np.nan, 3.0], "diet": [11, 22, 33]})
R = pd.DataFrame({"seqn": [np.nan, np.nan, 3.0], "chol": [91, 92, 93]})
d = diagnose_join(L, R, "seqn", "seqn", "inner"); m, _ = execute_join(L, R, "seqn", "seqn", "inner")
print(d.predicted_rows, len(m), d.warnings, d.notes, d.blocking)   # 1 5 [] [] []

# 3. Cohort scale: 200x200 with 10 blank IDs per side
n, miss = 200, 10
keys = [None]*miss + [str(i) for i in range(n-miss)]
L = pd.DataFrame({"seqn": pd.Series(keys, dtype="object"), "diet": range(n)})
R = pd.DataFrame({"seqn": pd.Series(keys, dtype="object"), "chol": range(n)})
d = diagnose_join(L, R, "seqn", "seqn", "inner"); m, _ = execute_join(L, R, "seqn", "seqn", "inner")
print(d.predicted_rows, len(m), d.warnings)   # 190 290 []  (+100 fabricated rows, silent)
print(len(execute_join(L, R, "seqn", "seqn", "inner", repair=False)[0]))   # 290 — not the repair step

# 4. Additional facets I found beyond the report
L = pd.DataFrame({"seqn": [None, None, "3", "4"], "diet": [1, 2, 3, 4]})
R = pd.DataFrame({"seqn": ["3", "4"], "chol": [9, 8]})
d = diagnose_join(L, R, "seqn", "seqn", "left"); m, _ = execute_join(L, R, "seqn", "seqn", "left")
print(d.predicted_rows, len(m))   # 2 vs 4 — LEFT join under-predicts with nulls on ONE side only
d = diagnose_join(L, R, "seqn", "seqn", "inner")
print(d.notes, d.warnings)        # [] [] — inner join silently drops 2 blank-ID rows, no note
```

---


## Finding 02

## Title
normalize_key blanks unparseable IDs in a ≥95%-numeric text key column; repair_keys turns the blanks into NaN, and pandas merges NaN to NaN — fusing unrelated subjects into fabricated rows the diagnosis never predicts or reports

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary, normalize_key

# --- 1. the reported case: 19 numeric + 1 alphanumeric = 95% numeric ---
L = pd.DataFrame({"subject": [str(i) for i in range(1,20)] + ["X1"], "diet": list(range(19)) + [111]})
R = pd.DataFrame({"subject": [str(i) for i in range(1,20)] + ["Y9"], "chol": list(range(19)) + [999]})

print(normalize_key(L["subject"]).tolist()[-2:])   # ['19', '']   X1 -> blank
print(normalize_key(R["subject"]).tolist()[-2:])   # ['19', '']   Y9 -> same blank

d = diagnose_join(L, R, "subject", "subject", "inner")
m, _ = execute_join(L, R, "subject", "subject", "inner")
print(plain_summary(d), d.warnings, d.notes, d.blocking, d.unmatched_left, d.unmatched_right)
# Result: **19 rows** ... [] [] [] 0 0
print("predicted", d.predicted_rows, "actual", len(m))   # predicted 19 actual 20
print(m.tail(1).to_string())                             # subject NaN, diet 111, chol 999

# --- 2. the tool's OWN repair causes it (plain pandas is correct) ---
print(len(L.merge(R, on="subject", how="inner")))                      # 19  correct
print(len(execute_join(L, R, "subject", "subject", repair=False)[0]))  # 19  correct
print(len(execute_join(L, R, "subject", "subject", repair=True)[0]))   # 20  corrupted (default)

# --- 3. it scales as a CARTESIAN PRODUCT, not one stray row ---
L2 = pd.DataFrame({"subject": [str(i) for i in range(1,96)] + [f"A{i}" for i in range(5)], "diet": range(100)})
R2 = pd.DataFrame({"subject": [str(i) for i in range(1,96)] + [f"B{i}" for i in range(5)], "chol": range(100)})
d2 = diagnose_join(L2, R2, "subject", "subject", "inner")
m2, _ = execute_join(L2, R2, "subject", "subject", "inner")
print(d2.predicted_rows, len(m2), d2.row_multiplication, d2.warnings)
# 95 120 False []   -> 25 fabricated rows, fan-out flag False, zero warnings

# --- 4. every join type is affected ---
for how in ("inner", "left", "right", "outer"):
    dd = diagnose_join(L, R, "subject", "subject", how)
    mm, _ = execute_join(L, R, "subject", "subject", how)
    print(how, dd.predicted_rows, len(mm))   # all predict 19, all return 20
```

---


## Finding 03

## Title
diagnose_join counts missing values as a shared key for text/date columns, inflating matched_keys and emitting a phantom "differ only by capitalisation" warning

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, plain_summary, execute_join

# A) reported case - byte-identical text key columns
L = pd.DataFrame({"id": ["A1", None], "a": [1, 2]})
R = pd.DataFrame({"id": ["A1", None], "b": [7, 8]})
d = diagnose_join(L, R, "id", "id")
print(d.matched_keys, d.needs_normalization, d.warnings)
# -> 2 True ['Some IDs differ only by capitalisation or stray spaces. ...']   (expected: 1 False [])

# B) datetime key - "capitalisation" advice about a column with no letters
L = pd.DataFrame({"visit": pd.to_datetime(["2020-01-01", None, None]), "a": [1,2,3]})
R = pd.DataFrame({"visit": pd.to_datetime(["2020-01-01", None, None]), "b": [7,8,9]})
d = diagnose_join(L, R, "visit", "visit")
print(d.matched_keys, d.needs_normalization, d.warnings)   # -> 2 True [phantom warning]

# C) ADDITIONAL defect found - missing keys wreck the row prediction too
L = pd.DataFrame({"pid": ["S%03d" % i for i in range(1,51)] + [None]*30, "a": range(80)})
R = pd.DataFrame({"pid": ["S%03d" % i for i in range(1,51)] + [None]*20, "b": range(70)})
d = diagnose_join(L, R, "pid", "pid")
print(d.matched_keys, d.predicted_rows, d.row_multiplication)  # -> 51, 50, False
print(len(execute_join(L, R, "pid", "pid")[0]))                # -> 650 actual rows
```

---


## Finding 04

## Title
join_doctor labels category/date/period/duration keys as "text", blocking working categorical-vs-numeric joins and emitting a self-contradictory message

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
/tmp/claude-0/-home-user-tabular-ml-lab/07f184b8-6f8b-5f93-9930-b6e30849812e/scratchpad/verify_join_dtype.py

import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, plain_summary

# Defect 1 - a join that works is blocked, with a false claim about storage.
L = pd.DataFrame({"cycle": pd.Categorical([1, 2, 3]), "a": [1, 2, 3]})
R = pd.DataFrame({"cycle": [1, 2, 3], "b": [7, 8, 9]})
d = diagnose_join(L, R, "cycle", "cycle")
assert len(L.merge(R, on="cycle")) == 3      # pandas merges this fine
assert d.matched_keys == 3                    # diagnose_join AGREES 3 match
assert not d.can_proceed                      # ...yet blocks the join
assert "stored as text" in d.blocking[0]      # ...calling a category of ints "text"
print("D1:", plain_summary(d)); print("   ", d.blocking[0])

# Defect 2 - for a genuinely impossible join the message contradicts itself.
L2 = pd.DataFrame({"k": pd.to_datetime(["2020-01-01", "2020-01-02"]), "a": [1, 2]})
R2 = pd.DataFrame({"k": [1, 2], "b": [7, 8]})
d2 = diagnose_join(L2, R2, "k", "k")          # blocking here is CORRECT (merge raises)
assert "look identical on screen" in d2.blocking[0] and "matches 0 IDs" in d2.blocking[0]
print("D2:", d2.blocking[0])
print("OK - both defects reproduce")
```

---


## Finding 05

## Title
join_doctor: datetime keys are compared by their stringified form, so a tz-aware/naive (or cross-timezone) pair is reported as "nothing to join on. Check you picked the right columns"

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
from datetime import timezone, timedelta
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join, normalize_key, find_key_candidates

# --- Case 1: exactly as reported (naive vs tz-aware) ---
L = pd.DataFrame({"visit_date": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"]), "a":[1,2,3]})
R = pd.DataFrame({"visit_date": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"]).tz_localize("UTC"), "b":[7,8,9]})
print(list(normalize_key(L["visit_date"])))   # ['2020-01-01', ...]
print(list(normalize_key(R["visit_date"])))   # ['2020-01-01 00:00:00+00:00', ...]
print(diagnose_join(L, R, "visit_date", "visit_date").blocking)
# ["None of the values in 'visit_date' appear in 'visit_date', so there is nothing to join on. Check you picked the right columns."]

# --- Case 2 (worse, not in the report): both tz-aware, different zones, SAME instants ---
base = pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"]).tz_localize("UTC")
L2 = pd.DataFrame({"visit_date": base, "a":[1,2,3]})
R2 = pd.DataFrame({"visit_date": base.tz_convert(timezone(timedelta(hours=-5))), "b":[7,8,9]})
print(len(L2.merge(R2, on="visit_date", how="inner")))            # 3  <- plain pandas joins fine
print(diagnose_join(L2, R2, "visit_date", "visit_date").blocking) # same "nothing to join on" blocker
print(len(execute_join(L2, R2, "visit_date", "visit_date")[0]))   # 0  <- repair=True destroys the join
print(len(execute_join(L2, R2, "visit_date", "visit_date", repair=False)[0]))  # 3

# --- Case 3: same failure with plain strings (Excel vs DB CSV export, no parse_dates) ---
L3 = pd.DataFrame({"visit_date": ["2020-01-01","2020-01-02","2020-01-03"], "a":[1,2,3]})
R3 = pd.DataFrame({"visit_date": ["2020-01-01 00:00:00+00:00","2020-01-02 00:00:00+00:00","2020-01-03 00:00:00+00:00"], "b":[7,8,9]})
print(diagnose_join(L3, R3, "visit_date", "visit_date").blocking)  # same blocker

# --- Control: date objects vs datetime64 midnight DOES match (blocking == []) ---
L4 = pd.DataFrame({"d": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"]).date, "a":[1,2,3]})
R4 = pd.DataFrame({"d": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"]), "b":[7,8,9]})
print(diagnose_join(L4, R4, "d", "d").blocking)  # []
```

---


## Finding 06

## Title
Blank/NaN join keys are cross-joined into fabricated rows; diagnose_join's predicted row count is wrong and no warning or blocker fires

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# (1) Both sides have blank IDs -> cross-join fabricates participants
left  = pd.DataFrame({"SEQN": [1, 2, 3, np.nan, np.nan],        "age":     [40, 55, 61, 70, 22]})
right = pd.DataFrame({"SEQN": [1, 2, 3, np.nan, np.nan, np.nan], "glucose": [95, 102, 110, 88, 90, 91]})
for how in ("inner", "left", "right", "outer"):
    for repair in (True, False):
        d = diagnose_join(left, right, "SEQN", "SEQN", how)
        m, _ = execute_join(left, right, "SEQN", "SEQN", how, repair=repair)
        print(how, repair, "predicted", d.predicted_rows, "actual", len(m),
              "warnings", d.warnings, "blocking", d.blocking)
# -> predicted 3, actual 9, no warnings, no blockers, in all 8 combinations
print(execute_join(left, right, "SEQN", "SEQN", "inner")[0])

# (2) Trailing blank rows from an Excel export -- the realistic trigger
import io
l = pd.read_csv(io.StringIO("SEQN,age\n1,40\n2,55\n3,61\n,\n,\n"))
r = pd.read_csv(io.StringIO("SEQN,glucose\n1,95\n2,102\n3,110\n,\n,\n"))
d = diagnose_join(l, r, "SEQN", "SEQN", "inner")
m, _ = execute_join(l, r, "SEQN", "SEQN", "inner")
print(plain_summary(d), "-> actual", len(m))   # promises 3 rows, returns 7 (4 all-NaN)

# (3) NOT in the original report: one-sided blanks break non-inner predictions too
l1 = pd.DataFrame({"SEQN": [1, 2, np.nan], "age": [1, 2, 3]})
r1 = pd.DataFrame({"SEQN": [1, 2, 3],      "lab": [9, 8, 7]})
for how in ("inner", "left", "outer"):
    d = diagnose_join(l1, r1, "SEQN", "SEQN", how)
    m, _ = execute_join(l1, r1, "SEQN", "SEQN", how)
    print(how, "predicted", d.predicted_rows, "actual", len(m))
# -> inner 2/2 (ok), left 2/3, outer 3/4

# (4) All-NaN key: diagnose blocks, but execute_join still returns 5x5
l2 = pd.DataFrame({"SEQN": [np.nan]*5, "age": range(5)})
r2 = pd.DataFrame({"SEQN": [np.nan]*5, "lab": range(5)})
d = diagnose_join(l2, r2, "SEQN", "SEQN", "inner")
print(d.predicted_rows, d.blocking, len(execute_join(l2, r2, "SEQN", "SEQN", "inner")[0]))  # 0, [msg], 25
```

---


## Finding 07

## Title
diagnose_join green-lights date-vs-text key pairs that execute_join(repair=False) cannot merge

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# --- Variant A: green-lit diagnosis, raw pandas ValueError (as reported) ---
L = pd.DataFrame({"d": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"]), "a":[1,2,3]})
R = pd.DataFrame({"d": ["2020-01-01","2020-01-02","2020-01-03"], "b":[7,8,9]})
d = diagnose_join(L, R, "d", "d")
print("can_proceed:", d.can_proceed, "| blocking:", d.blocking, "| warnings:", d.warnings)
print("dtype_mismatch:", d.dtype_mismatch, "| needs_norm:", d.needs_normalization)
print(plain_summary(d))            # "Result: **3 rows** ..."
try:
    execute_join(L, R, "d", "d", "inner", repair=False)
except ValueError as e:
    print("A) RAISED:", e)         # merge on datetime64[us] and str columns

# --- Variant B: same root cause, SILENT 0 rows instead of an error (not in the report) ---
L2 = pd.DataFrame({"d": ["2020-01-01","2020-01-02","2020-01-03"], "a":[1,2,3]})
R2 = pd.DataFrame({"d": [pd.Timestamp(v).date() for v in ["2020-01-01","2020-01-02","2020-01-03"]], "b":[7,8,9]})
d2 = diagnose_join(L2, R2, "d", "d")
merged, _ = execute_join(L2, R2, "d", "d", "inner", repair=False)
print("B) predicted:", d2.predicted_rows, "matched_keys:", d2.matched_keys,
      "can_proceed:", d2.can_proceed, "-> ACTUAL rows:", len(merged))   # 3 / 3 / True -> 0

# --- Control: the numeric equivalent IS correctly blocked ---
L3 = pd.DataFrame({"id": [1,2,3], "a":[1,2,3]})
R3 = pd.DataFrame({"id": ["001","002","003"], "b":[7,8,9]})
d3 = diagnose_join(L3, R3, "id", "id")
print("C) numeric can_proceed:", d3.can_proceed, "|", d3.blocking[0][:70])

# --- The default path is fine everywhere: repair=True joins all three correctly ---
print("D) repair=True rows:", len(execute_join(L, R, "d", "d", "inner")[0]),
      len(execute_join(L2, R2, "d", "d", "inner")[0]),
      len(execute_join(L3, R3, "id", "id", "inner")[0]))   # 3 3 3
```

---


## Finding 08

## Title
find_key_candidates draws an independent 5,000-row sample per file, so value overlap is measured between unrelated row subsets — the true key is dropped entirely above ~50k rows and mis-quoted 4x-150x too low below that

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import find_key_candidates, suggest_best, diagnose_join

# CASE A — 300-subject cohort x 200k lab export; all 300 IDs present on the right.
# (SEQN starts at 83732, so _looks_like_row_index is False — not an arange artifact.)
demo = pd.DataFrame({"SEQN": np.arange(83732, 83732+300), "age": np.zeros(300)})
labs = pd.DataFrame({"SEQN": np.arange(83732, 83732+200_000), "glucose": np.zeros(200_000)})
print(find_key_candidates(demo, labs), suggest_best(demo, labs))
print("truth:", diagnose_join(demo, labs, "SEQN", "SEQN").matched_keys)
# -> [] None      truth: 300

# CASE B — the module's flagship promise: differently-named key, string IDs,
# no row-counter structure anywhere. 800 of 800 left IDs genuinely match.
rng = np.random.default_rng(0)
ids = np.array([f"PT{n:07d}" for n in rng.choice(9_000_000, size=120_000, replace=False)])
right = pd.DataFrame({"patient_id": ids, "glucose": rng.normal(size=len(ids))})
left  = pd.DataFrame({"SEQN": rng.choice(ids, size=800, replace=False), "age": rng.normal(size=800)})
print(len(find_key_candidates(left, right)), suggest_best(left, right))
print("truth:", diagnose_join(left, right, "SEQN", "patient_id").matched_keys)
# -> 0 None       truth: 800

# CASE C — confidently wrong numbers rather than silence, and stale row counts.
for n, m in [(6_000, 5_000), (20_000, 19_000), (100_000, 90_000), (500_000, 400_000)]:
    A = pd.DataFrame({"SEQN": np.arange(n), "age": np.zeros(n)})
    B = pd.DataFrame({"SEQN": np.arange(m), "glucose": np.zeros(m)})
    cc = find_key_candidates(A, B)
    print(f"{n}/{m}:", (cc[0].headline() if cc else "NO CANDIDATES"),
          "| conf:", (cc[0].confidence if cc else "-"),
          "| left_rows:", (cc[0].left_rows if cc else "-"),
          "| truth:", diagnose_join(A, B, "SEQN", "SEQN").matched_keys)
# 6000/5000:     share 4,178 IDs (84%/84%)  conf high   left_rows 5000  truth 5000
# 20000/19000:   share 2,601 IDs (52%/52%)  conf high   left_rows 5000  truth 19000
# 100000/90000:  share   624 IDs (12%/12%)  conf low    left_rows 5000  truth 90000
# 500000/400000: NO CANDIDATES                                          truth 400000

# CASE D — fan-out signal on the candidate also vanishes with the candidate.
subs = np.arange(500_000, 500_000+250_000)
R = pd.DataFrame({"SEQN": np.repeat(subs, 2), "glucose": np.zeros(500_000)})
L = pd.DataFrame({"SEQN": subs[:3000], "age": np.zeros(3000)})
print(find_key_candidates(L, R))                       # -> []
d = diagnose_join(L, R, "SEQN", "SEQN")
print("truth:", d.matched_keys, d.row_multiplication)  # -> 3000 True

```

---


## Finding 09

## Title
Duplicate key column name makes every ml/join_doctor.py entry point raise AttributeError and makes find_key_candidates silently drop the true key — but the module has no UI callers and no loader can currently produce such a frame

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import (find_key_candidates, suggest_best, diagnose_join,
                            execute_join, repair_keys, normalize_key)

left  = pd.DataFrame([[1,40,1],[2,55,2],[3,61,3]], columns=["SEQN","age","SEQN"])
right = pd.DataFrame({"SEQN":[1,2,3], "glucose":[95,102,110]})

print(find_key_candidates(left, right), suggest_best(left, right))   # [] None
diagnose_join(left, right, "SEQN", "SEQN")   # AttributeError: 'DataFrame' object has no attribute 'str'

# Mechanism correction #1: only the duplicated label is skipped, not the whole frame.
l2 = pd.DataFrame([[1,40,1,"a1"],[2,55,2,"a2"],[3,61,3,"a3"]],
                  columns=["SEQN","age","SEQN","pid"])
r2 = pd.DataFrame({"SEQN":[1,2,3], "pid":["a1","a2","a3"], "glucose":[95,102,110]})
print([(c.left_col, c.right_col) for c in find_key_candidates(l2, r2)])
# -> [('pid','pid')]  : 'pid' is still proposed; the true key SEQN is dropped with no explanation

# Mechanism correction #2: execute_join(repair=False) never reaches normalize_key.
execute_join(l2, r2, "SEQN", "SEQN", repair=False)
# -> ValueError: The column label 'SEQN' is not unique.   (NOT AttributeError)
```

---


## Finding 10

## Title
drop_rows fix deletes rows by label, destroying unrelated rows on non-unique-index frames while reporting the intended (wrong) count

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro runs verbatim and reproduces exactly (pandas 3.0.3):

    import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
    import pandas as pd
    from ml.import_doctor import diagnose, apply_fix

    df = pd.DataFrame({"id": [1,2,3,4,5,6,7,8,"Total"], "v": [1,2,3,4,5,6,7,8,None]},
                      index=[0,1,2,3,4,5,6,7,0])
    f = [x for x in diagnose(df) if x.id == "footer_rows"][0]
    fixed, desc = apply_fix(df, f)
    print(desc, len(df), "->", len(fixed), fixed["id"].tolist())
    # Dropped 1 non-data row(s) from the bottom of the file. 9 -> 7 [2, 3, 4, 5, 6, 7, 8]

Reached through the app's own loader (no hand-built index needed) — pandas orient='split' JSON, which data_processor._json_obj_to_frame honours via `df.index = obj["index"]`:

    import json, sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
    from data_processor import load_tabular_data
    from ml.import_doctor import diagnose, apply_fix
    payload = {"columns": ["id","v"], "index": [0,1,2,3,4,5,6,7,0],
               "data": [[i,i] for i in range(1,9)] + [["Total", None]]}
    open("/tmp/export.json","w").write(json.dumps(payload))
    df = load_tabular_data("/tmp/export.json", filename="export.json")
    f = [x for x in diagnose(df) if x.id == "footer_rows"][0]
    print(apply_fix(df, f)[1], len(df), "->", len(apply_fix(df, f)[0]))
    # Dropped 1 non-data row(s) from the bottom of the file. 9 -> 7

Worse variant — longitudinal frame indexed by repeated subject ID, one footer row, three real rows destroyed while the message still says one:

    idx = ["S1","S1","S2","S2","S3","S3","S1"]
    df = pd.DataFrame({"subject":["S1","S1","S2","S2","S3","S3","Total"],
                       "bp":[120,118,130,128,140,138,None]}, index=idx)
    # -> "Dropped 1 non-data row(s)..." 7 -> 4; both S1 visits silently deleted
```

---


## Finding 11

## Title
MultiIndex (two-row header) columns: diagnose_join green-lights a join that execute_join always crashes on

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary, suggest_best

A = pd.DataFrame([[1,95],[2,102],[3,110]],
                 columns=pd.MultiIndex.from_tuples([("key","SEQN"),("labs","glucose")]))
B = pd.DataFrame([[1,40],[2,55],[3,61]],
                 columns=pd.MultiIndex.from_tuples([("key","SEQN"),("demo","age")]))

d = diagnose_join(A, B, ("key","SEQN"), ("key","SEQN"))
print(plain_summary(d), d.can_proceed, d.blocking)
# -> Result: **3 rows** - matching on 3 shared IDs, ... True []

# 1) execute_join crashes for every how, with repair=True and repair=False
for how in ("inner","left","right","outer"):
    try:
        execute_join(A, B, ("key","SEQN"), ("key","SEQN"), how=how)
    except ValueError as e:
        print(how, "->", str(e).splitlines()[0])   # The column label 'key' is not unique.

# 2) the suggested candidate cannot be fed back into the API
c = suggest_best(A, B)
print(repr(c.left_col), c.confidence)              # "('key', 'SEQN')" high  <- a label that does not exist
diagnose_join(A, B, c.left_col, c.right_col)       # KeyError: "('key', 'SEQN')"
```

---


## Finding 12

## Title
FALSE MERGE: non-numeric IDs in a mostly-numeric key column are blanked to NaN by repair_keys, then cross-joined with each other

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary, normalize_key

# --- Part 1: the reported case (same IDs on both sides) ---
ids = [f"{i:03d}" for i in range(1, 97)] + ["A01", "A02", "A03", "A04"]  # 96% numeric
demo = pd.DataFrame({"subject_id": ids, "age": range(100)})
labs = pd.DataFrame({"subject_id": ids, "glucose": range(1000, 1100)})

print(list(normalize_key(pd.Series(ids))[-4:]))        # ['', '', '', ''] -> identity destroyed
d = diagnose_join(demo, labs, "subject_id", "subject_id", "inner")
merged, _ = execute_join(demo, labs, "subject_id", "subject_id", "inner")   # repair=True default
print(plain_summary(d), d.blocking, d.warnings, d.notes)  # "96 rows", [], [], []
print("predicted", d.predicted_rows, "actual", len(merged))               # 96 vs 112
print(len(execute_join(demo, labs, "subject_id", "subject_id", "inner", repair=False)[0]))  # 100

# --- Part 2 (worse, not in the original report): the blanked IDs are DIFFERENT subjects ---
lids = [f"{i:03d}" for i in range(1, 97)] + ["A01", "A02", "A03", "A04"]
rids = [f"{i:03d}" for i in range(1, 97)] + ["B77", "B78", "B79", "B80"]
demo2 = pd.DataFrame({"subject_id": lids, "age": range(100)})
labs2 = pd.DataFrame({"subject_id": rids, "glucose": range(1000, 1100)})
d2 = diagnose_join(demo2, labs2, "subject_id", "subject_id", "inner")
m2, _ = execute_join(demo2, labs2, "subject_id", "subject_id", "inner")
print("predicted", d2.predicted_rows, "actual", len(m2), d2.blocking, d2.warnings)  # 96 vs 112, silent
print(m2.tail(4).to_string(index=False))   # A01..A04's ages carrying B77..B80's glucose, subject_id NaN
```

---


## Finding 13

## Title
Duplicate column labels crash three checks; diagnose() silently swallows the crashes and drops unrelated findings frame-wide

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.import_doctor import diagnose, summarize, apply_fix, ALL_CHECKS

good = pd.DataFrame({"bp":   [120,118,130,121,119,125,122,117,128,124,126],
                     "age":  [40,55,61,999,47,52,33,999,61,44,39],
                     "notes":[None]*11})
bad = good.copy(); bad["bp2"] = bad["bp"]
bad = bad[["bp","bp2","age","notes"]]; bad.columns = ["bp","bp","age","notes"]

print("unique names   ->", [x.id for x in diagnose(good)], "|", summarize(diagnose(good)))
print("duplicated 'bp'->", [x.id for x in diagnose(bad)],  "|", summarize(diagnose(bad)))

for chk in ALL_CHECKS:                      # per-check probe
    try:    print(f"  {chk.__name__:32s} OK   -> {[f.id for f in chk(bad)]}")
    except Exception as e: print(f"  {chk.__name__:32s} DIES {type(e).__name__}: {e}")

# recovery: the loss is temporary if the caller acts on the finding that IS shown
dup = [f for f in diagnose(bad) if f.id == "duplicate_columns"][0]
fixed, _ = apply_fix(bad, dup)
print("after dedupe fix ->", list(fixed.columns), [f.id for f in diagnose(fixed)])
```

---


## Finding 14

## Title
Unconditional case-folding in normalize_key silently merges distinct case-sensitive IDs, and diagnose_join misattributes the resulting fan-out to "several rows per ID"

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import normalize_key, diagnose_join, execute_join, plain_summary

# 8 distinct case-sensitive accession IDs; two pairs collide under case folding
ids = ["aB3", "Ab3", "xY9", "Xy9", "k7Q", "m2R", "p5T", "z8W"]
A = pd.DataFrame({"record_id": ids, "age":     [41,72,55,29,60,33,48,66]})
B = pd.DataFrame({"record_id": ids, "glucose": [88,190,101,140,95,110,120,133]})

print(normalize_key(A.record_id).tolist())
# -> ['ab3','ab3','xy9','xy9','k7q','m2r','p5t','z8w']   (8 subjects -> 6 tokens)

d = diagnose_join(A, B, "record_id", "record_id")
print(d.matched_keys, d.needs_normalization, d.predicted_rows)
# -> 6 False 12        (true answer: 8 matched keys, 8 rows)
print(plain_summary(d))
# -> "Result: **12 rows** — matching on 6 shared IDs, keeping only IDs found in both files."
print(d.blocking)   # -> []  (nothing blocks the join)
print(d.warnings)
# -> ["Both files have several rows per ID, so every combination is produced: 6 shared IDs
#     become 12 rows. This is usually a mistake — check whether one file should be
#     summarised to one row per subject first."]

print(execute_join(A, B, "record_id", "record_id")[0].to_string(index=False))
# 12 rows: ab3 pairs age 41 with glucose 190, age 72 with glucose 88, etc.

print(len(execute_join(A, B, "record_id", "record_id", repair=False)[0]))  # -> 8 (correct)
```

---


## Finding 15

## Title
FALSE MERGE: normalize_key coerces large numeric-looking IDs through float64, fabricating and collapsing keys (ml/join_doctor.py:59-65)

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd, numpy as np
from ml.join_doctor import normalize_key, diagnose_join, execute_join, plain_summary

# --- Case 1: report as filed (21-digit barcodes read as TEXT) ---
bc = ["123456789012345678901", "123456789012345678902"]
demo = pd.DataFrame({"barcode": bc, "age": [41, 72]})
labs = pd.DataFrame({"barcode": bc, "glucose": [88, 190]})
print(demo["barcode"].dtype)                       # str  -> nothing lost on import
print(normalize_key(demo["barcode"]).tolist())     # ['123456789012345683968', '123456789012345683968']
d = diagnose_join(demo, labs, "barcode", "barcode", "inner")
m, _ = execute_join(demo, labs, "barcode", "barcode", "inner")
print(plain_summary(d)); print(d.blocking, d.warnings)
print(m.to_string(index=False))                    # 4 rows, fabricated barcode, crossed glucose

# --- Case 2: one non-integer cell drags the column to float64 ---
print(normalize_key(pd.Series(["9007199254740993", "9007199254740992", "1.5"])).tolist())

# --- Case 3 (BROADER, not in the report): 18-digit accession IDs + ONE blank cell ---
ids = [str(900719925474099000 + i) for i in range(200)]
col = ids[:]; col[7] = ""                          # numeric_share 0.995 >= 0.95 -> float64 branch
demo2 = pd.DataFrame({"accession": col, "age": np.arange(200)})
labs2 = pd.DataFrame({"accession": col, "glucose": np.arange(200) + 50})
n = normalize_key(demo2["accession"])
print(demo2["accession"].nunique(), "->", n[n != ""].nunique())   # 200 -> 2
m2, _ = execute_join(demo2, labs2, "accession", "accession", "inner")
print(len(m2))                                     # 26882 rows from two 200-row files

# --- Root cause, isolated: errors="coerce" is what destroys the value ---
t = pd.Series(bc).astype(str).str.strip()
print(pd.to_numeric(t).dtype, pd.to_numeric(t).tolist())                  # object, EXACT ints
print(pd.to_numeric(t, errors="coerce").dtype,
      pd.to_numeric(t, errors="coerce").tolist())                          # float64, both 1.2345678901234568e+20
```

---


## Finding 16

## Title
Missing join keys are cross-joined into fabricated rows and counted as a matched ID; predicted row count is wrong (2 predicted vs 6 actual) — affects all key types, not just the text/NaN branch

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary, normalize_key

def check(label, A, B, how="inner"):
    d = diagnose_join(A, B, "id", "id", how)
    m, _ = execute_join(A, B, "id", "id", how)
    print(f"{label} [{how}]: predicted={d.predicted_rows} actual={len(m)} "
          f"matched_keys={d.matched_keys} unmatched_left={d.unmatched_left}")
    print("   ", plain_summary(d))
    if d.warnings: print("    warnings:", d.warnings)
    return d, m

# (1) EXACTLY AS REPORTED -- text keys, missing on both sides.
A = pd.DataFrame({"id": ["a1", "a2", None, None], "age": [40, 50, 60, 70]})
B = pd.DataFrame({"id": ["a1", "a2", None, None], "glucose": [90, 91, 92, 93]})
d, m = check("text NaN both sides", A, B)
print(m.to_string(index=False))
assert d.matched_keys == 3 and d.predicted_rows == 2 and len(m) == 6   # all reproduce

# Mechanism: pandas 3.0 astype(str) preserves NA, so the "" guard misses it,
# and NaN survives set intersection by identity while value_counts drops it.
ln = normalize_key(A["id"])
print("\nnormalize_key ->", ln.tolist(), "| dtype", ln.dtype)
print("after `ln[ln != \"\"]` guard ->", ln[ln != ""].tolist())   # NaN still present
print("lset & rset ->", set(ln[ln != ""].unique()) & set(ln[ln != ""].unique()))

# (2) NOT IN THE REPORT: fabrication is NOT confined to the text/NaN branch.
# normalize_key maps these to "" correctly and matched_keys is right, but
# repair_keys writes "" back to NaN and pandas still cross-joins them.
check("numeric NaN keys", pd.DataFrame({"id": [1.0, 2.0, np.nan, np.nan], "age": [40, 50, 60, 70]}),
                          pd.DataFrame({"id": [1.0, 2.0, np.nan, np.nan], "glucose": [90, 91, 92, 93]}))
check("empty-string keys", pd.DataFrame({"id": ["a1", "a2", "", ""], "age": [40, 50, 60, 70]}),
                           pd.DataFrame({"id": ["a1", "a2", "", ""], "glucose": [90, 91, 92, 93]}))
check("whitespace-only keys", pd.DataFrame({"id": ["a1", "a2", "  ", " "], "age": [40, 50, 60, 70]}),
                              pd.DataFrame({"id": ["a1", "a2", "  ", " "], "glucose": [90, 91, 92, 93]}))
# all three: predicted=2, actual=6

# (3) NOT IN THE REPORT: left joins under-predict even with NaN on one side only.
check("text NaN left only", pd.DataFrame({"id": ["a1", "a2", None], "age": [40, 50, 60]}),
                            pd.DataFrame({"id": ["a1", "a2"], "glucose": [90, 91]}), "left")
# predicted=2, actual=3

# (4) Realistic scale + a bogus warning as a side effect.
n = 100; ids = [f"s{i:03d}" for i in range(n)]
check("100 subjects, 6 blank IDs each side",
      pd.DataFrame({"id": ids[:94] + [None]*6, "age": np.arange(n)}),
      pd.DataFrame({"id": ids[:94] + [None]*6, "glucose": np.arange(n)*2}))
# predicted=94, actual=130 (36 fabricated rows), matched_keys=95,
# plus a false "Some IDs differ only by capitalisation or stray spaces" warning.

# (5) The same fabrication is LIVE in the shipping UI merge path
# (pages/01_Upload_and_Audit.py lines 682 & 691), independent of join_doctor:
print("\nUI 'Matching values' metric:",
      len(set(A["id"].dropna().unique()) & set(B["id"].dropna().unique())))   # 2
print("UI merged rows:", len(pd.merge(A, B, on="id", how="inner", suffixes=("", "_2"))))  # 6

```

---


## Finding 17

## Title
diagnose_join under-predicts row count whenever key values are missing/blank (not just when a key column is entirely missing)

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro runs verbatim and prints exactly the claimed numbers (pandas 3.0.3):

    right predicted 0 actual 2
    outer predicted 2 actual 4

But that exact case is the *least* harmful form, because `matched` is empty so `diagnose_join` sets a blocking message and `plain_summary()` returns "This join will not work yet — see below" — the wrong number is never shown above the Merge button. The damaging, version-independent case is PARTIAL missingness, where there is no blocker and the wrong number is displayed:

    import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
    import pandas as pd, numpy as np
    from ml.join_doctor import diagnose_join, execute_join, plain_summary

    A = pd.DataFrame({"id": [1.0, 2.0, np.nan, 3.0], "age": range(4)})
    B = pd.DataFrame({"id": [2.0, 3.0, np.nan, np.nan], "glucose": range(4)})
    for how in ("inner", "left", "right", "outer"):
        d = diagnose_join(A, B, "id", "id", how)
        m, _ = execute_join(A, B, "id", "id", how)
        print(how, "predicted", d.predicted_rows, "actual", len(m), "blocking:", bool(d.blocking))
    print(plain_summary(diagnose_join(A, B, "id", "id", "inner")))

Output — all four join types wrong, no blocker on any of them:

    inner  predicted 2 actual 4 blocking: False
    left   predicted 3 actual 5 blocking: False
    right  predicted 2 actual 4 blocking: False
    outer  predicted 3 actual 5 blocking: False
    Result: **2 rows** — matching on 2 shared IDs, keeping only IDs found in both files.

The user is told "2 rows" above the Merge button and gets 4.
```

---


## Finding 18

## Title
Large files: independent per-side row sampling in `_prep` destroys measured key overlap — real overlaps are reported as unlinkable or badly undercounted

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro is valid and reproduces verbatim (pandas 3.0.3 / numpy 2.4.6). Extended version that also isolates the mechanism and two harder consequences:

import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd, numpy as np
import ml.join_doctor as jd
from ml.join_doctor import find_key_candidates, suggest_best, diagnose_join

# --- 1. reported case: 20k vs 20k, true 50% overlap -----------------------
n = 20000
L = pd.DataFrame({"SEQN": np.arange(0, n), "age": 1})
R = pd.DataFrame({"SEQN": np.arange(n//2, n + n//2), "glucose": 2})
print(suggest_best(L, R))                      # None   (truth: SEQN<->SEQN)
c = find_key_candidates(L, R)[0]
print(c.coverage_left, c.coverage_right, c.n_matched, c.confidence)
#   0.1242 0.1242 621 low        truth: 0.5 0.5 10000 high
print(diagnose_join(L, R, "SEQN", "SEQN").matched_keys)   # 10000

# --- 2. the mechanism: the two samples are different SUBSETS of ROWS ------
sl, sr = jd._prep(L, "SEQN"), jd._prep(R, "SEQN")
print(len(set(sl) & set(sr)), len(set(L.SEQN) & set(R.SEQN)))   # 621 vs 10000

# --- 3. NHANES-shaped, labs a strict subset of demographics --------------
seqn = np.arange(83732, 83732 + 10175)
demo = pd.DataFrame({"SEQN": seqn, "age": 1})
labs = pd.DataFrame({"SEQN": seqn[:8366], "glucose": 2})
c2 = find_key_candidates(demo, labs)[0]
print(c2.headline("demographics", "labs"), c2.confidence)
#   "'SEQN' and 'SEQN' share 3,185 IDs (64% of demographics, 64% of labs)." high
#   truth: 8,366 IDs (82% of demographics, 100% of labs)

# --- 4. NEW: the answer depends on ROW ORDER, which is meaningless --------
a = pd.DataFrame({"SEQN": seqn, "age": 1})              # 10175 identical IDs
b = pd.DataFrame({"SEQN": seqn, "glucose": 2})          # 100% true overlap
print(find_key_candidates(a, b)[0].n_matched)                       # 5000
print(find_key_candidates(a, b.sample(frac=1, random_state=7)
                                .reset_index(drop=True))[0].n_matched)  # ~1174

# --- 5. NEW: legitimate repeated-measures join becomes a dead end ---------
subj = np.arange(60000)
left  = pd.DataFrame({"SEQN": subj, "age": 1})
right = pd.DataFrame({"SEQN": np.repeat(subj, 2), "visit": 1})
print(find_key_candidates(left, right)[0].n_matched)   # 432  (truth: 60000)
print(suggest_best(left, right))                       # None

# --- 6. counterfactual: the sampling is the whole cause -------------------
jd._SAMPLE_ROWS = 10**9
print(find_key_candidates(L, R)[0].n_matched, find_key_candidates(L, R)[0].confidence)  # 10000 high
print(find_key_candidates(demo, labs)[0].headline("demographics", "labs"))
#   "'SEQN' and 'SEQN' share 8,366 IDs (82% of demographics, 100% of labs)."
```

---


## Finding 19

## Title
Rows with a missing/blank join key are cross-joined into fabricated rows; predicted_rows is wrong by 5-25x and nothing warns

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# A) Report's headline case (blank "" on left, NaN on right) - reproduces ONLY with repair=True (default)
left  = pd.DataFrame({"SEQN": [f"A{i:03d}" for i in range(140)] + [""] * 60, "age": range(200)})
right = pd.DataFrame({"SEQN": [f"A{i:03d}" for i in range(140)] + [np.nan] * 60, "glucose": range(200)})
for how in ("inner", "left", "right", "outer"):
    d = diagnose_join(left, right, "SEQN", "SEQN", how)
    print(how, d.predicted_rows, len(execute_join(left, right, "SEQN", "SEQN", how)[0]),
          len(execute_join(left, right, "SEQN", "SEQN", how, repair=False)[0]), d.warnings)
# -> inner 140 3740 140 []   (repair=True fabricates 3600 rows; repair=False does not, here)

# B) Missing-on-both-sides - reproduces with repair BOTH True and False (pandas matches NA to NA itself)
l3 = pd.DataFrame({"id": ["a1", None, None], "age": [40, 55, 61]})
r3 = pd.DataFrame({"id": ["a1", None, None], "glucose": [95, 102, 110]})
print(diagnose_join(l3, r3, "id", "id").predicted_rows,
      len(execute_join(l3, r3, "id", "id")[0]),
      len(execute_join(l3, r3, "id", "id", repair=False)[0]))   # -> 1 5 5
# same 1 vs 5 for float-NaN keys and for nullable Int64 pd.NA keys

# C) Realistic scale, and the misleading advisory
rng = np.random.default_rng(0); ids = [f"{83000+i}" for i in range(1000)]
L = pd.DataFrame({"SEQN": ids[:940] + [np.nan]*60,  "age": rng.integers(20, 80, 1000)})
R = pd.DataFrame({"SEQN": ids[:930] + [np.nan]*70,  "glucose": rng.normal(95, 12, 1000)})
d = diagnose_join(L, R, "SEQN", "SEQN", "inner"); m, _ = execute_join(L, R, "SEQN", "SEQN", "inner")
print(d.predicted_rows, len(m), d.matched_keys, plain_summary(d), d.warnings)
# -> 930 5130 931 'Result: **930 rows** ...' and the ONLY warning is a false
#    "Some IDs differ only by capitalisation or stray spaces" advisory.
```

---


## Finding 20

## Title
execute_join drops right-only row identifiers on right/outer joins when key columns have different names

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import execute_join, diagnose_join

a = pd.DataFrame({"SEQN": [1, 2, 3], "age": [40, 55, 61]})
b = pd.DataFrame({"patient_id": [3, 4, 5], "glucose": [95, 102, 110]})

for how in ("inner", "left", "right", "outer"):
    merged, _ = execute_join(a, b, "SEQN", "patient_id", how)
    d = diagnose_join(a, b, "SEQN", "patient_id", how)
    print(how, "rows:", len(merged),
          "| predicted:", d.predicted_rows,
          "| rows with NULL identifier:", int(merged["SEQN"].isna().sum()),
          "| blocking/warnings:", d.blocking, d.warnings)

# how=right  -> 3 rows, 2 with NULL SEQN, predicted==actual, no warnings
# how=outer  -> 5 rows, 2 with NULL SEQN, predicted==actual, no warnings
# Control: identically-named keys coalesce correctly and lose nothing.
b2 = b.rename(columns={"patient_id": "SEQN"})
print(execute_join(a, b2, "SEQN", "SEQN", "outer")[0]["SEQN"].isna().sum())  # -> 0
```

---


## Finding 21

## Title
diagnose_join ignores rows with a blank key: predicted_rows under-counts left/right/outer joins (and misreports inner-join drops)

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

def check(tag, l, r):
    print("---", tag)
    for how in ("inner", "left", "right", "outer"):
        d = diagnose_join(l, r, "id", "id", how)
        actual = len(execute_join(l, r, "id", "id", how)[0])
        print(f"  {'OK ' if d.predicted_rows == actual else 'BAD'} {how:<5} predicted {d.predicted_rows}  actual {actual}")

# A. reported case: blanks on the left only, no blanks on the right (so no NaN cross-join)
l = pd.DataFrame({"id": [1, 2, None, None, None], "a": range(5)})
r = pd.DataFrame({"id": [1, 2, 3], "b": range(3)})
check("A blanks on left only", l, r)              # left 2/5 BAD, outer 3/6 BAD
d = diagnose_join(l, r, "id", "id", "inner")      # inner drops 3 of 5 left rows, silently:
print("  inner unmatched_left =", d.unmatched_left, "warnings =", d.warnings, "notes =", d.notes)
print("  left summary:", plain_summary(diagnose_join(l, r, "id", "id", "left")))
# -> "Result: **2 rows** ... keeping every row of the first file" for a 5-row left frame

# B. blanks on the right only (symmetric)
check("B blanks on right", pd.DataFrame({"id": [1,2,3], "a": range(3)}),
                           pd.DataFrame({"id": [1,None,None], "b": range(3)}))   # right 1/3, outer 3/5

# C. blanks on BOTH sides: pandas merges NaN to NaN, so even INNER is under-counted
check("C blanks both sides", pd.DataFrame({"id": [1,2,None,None], "a": range(4)}),
                             pd.DataFrame({"id": [1,3,None], "b": range(3)}))     # inner 1/3, outer 3/5

# D. key column entirely blank (float NaN, i.e. how a column of empties loads from CSV)
l = pd.DataFrame({"id": pd.Series([np.nan]*5, dtype="float64"), "a": range(5)})
r = pd.DataFrame({"id": [1, 2], "b": range(2)})
check("D all-blank key", l, r)                    # left 0/5, outer 2/7
print("  blocking:", diagnose_join(l, r, "id", "id", "left").blocking)
print("  blocking (object dtype):", diagnose_join(pd.DataFrame({"id": [None]*5, "a": range(5)}), r, "id", "id", "left").blocking)
```

---


## Finding 22

## Title
Blank/NaN join keys are invisible to diagnose_join: inner join silently drops those rows with no warning, and predicted_rows is wrong for left/outer joins

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

l = pd.DataFrame({"id": [1, 2, None, None, None], "a": range(5)})
r = pd.DataFrame({"id": [1, 2, 3], "b": range(3)})

d = diagnose_join(l, r, "id", "id", "inner")
print(plain_summary(d))
print("warnings:", d.warnings)
print("notes:   ", d.notes)
print("unmatched_left:", d.unmatched_left, "(should reflect the 3 blank-ID rows)")

# The prediction itself is wrong once blanks exist:
for how in ("inner", "left", "right", "outer"):
    d = diagnose_join(l, r, "id", "id", how)
    actual = len(execute_join(l, r, "id", "id", how)[0])
    flag = "" if d.predicted_rows == actual else "  <-- MISMATCH"
    print(f"{how:6s} predicted={d.predicted_rows} actual={actual}{flag}")

# Actual output:
# Result: **2 rows** - matching on 2 shared IDs, keeping only IDs found in both files.
# warnings: []
# notes:    ['1 row(s) of the second file have no match and will be dropped.']
# unmatched_left: 0 (should reflect the 3 blank-ID rows)
# inner  predicted=2 actual=2
# left   predicted=2 actual=5  <-- MISMATCH
# right  predicted=3 actual=3
# outer  predicted=3 actual=6  <-- MISMATCH
```

---


## Finding 23

## Title
diagnose_join returns can_proceed=True for a predicted 25,000,000-row many-to-many blow-up, and execute_join has no size cap

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys, time, resource; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary, suggest_best

L = pd.DataFrame({"id": ["A"] * 5000, "age": np.arange(5000)})
R = pd.DataFrame({"id": ["A"] * 5000, "glucose": np.arange(5000)})

d = diagnose_join(L, R, "id", "id")
print(f"{d.predicted_rows:,}", "can_proceed:", d.can_proceed, "blocking:", d.blocking)
print("warnings:", d.warnings)          # the m2m warning DOES fire
print(plain_summary(d))                 # but the headline line is neutral
print("recommender would propose:", suggest_best(L, R))   # -> None (see notes)

# Honest cost measurement (preferred over the rlimit trick):
def rss(): return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024
t0 = time.time(); m, _ = execute_join(L, R, "id", "id"); t1 = time.time()
print(f"rows={len(m):,} time={t1-t0:.2f}s deep_mem={m.memory_usage(deep=True).sum()/1024**3:.3f} GiB")
print("peak RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "MiB")

# The rlimit variant from the original report also reproduces verbatim, in a fresh process:
#   resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, resource.getrlimit(resource.RLIMIT_AS)[1]))
#   execute_join(L, R, "id", "id")
#   -> MemoryError: Unable to allocate 191. MiB for an array with shape (25000000,) and data type int64
```

---


## Finding 24

## Title
find_key_candidates scores keys on two independent 5,000-row samples, so overlap counts and coverage percentages are wrong on any file over 5,000 rows — a perfect key is reported as 72% at 'high' confidence, or withheld entirely

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import suggest_best, diagnose_join

# every right-hand ID is present on the left, i.e. a perfect key, in all four cases
for nl, nr in ((4000, 3500), (9254, 9254), (9254, 8000), (20000, 18000)):
    L = pd.DataFrame({"SEQN": np.arange(83732, 83732 + nl), "age": np.arange(nl)})
    R = pd.DataFrame({"SEQN": np.arange(83732, 83732 + nr), "glucose": np.arange(nr)})
    c = suggest_best(L, R)
    print(nl, "x", nr, "->", (c.headline(), c.confidence) if c else "NO KEY PROPOSED",
          "| truth:", diagnose_join(L, R, "SEQN", "SEQN").matched_keys, "shared IDs")

# Actual output (deterministic, pandas 3.0.3 / numpy 2.4.6):
# 4000 x 3500   -> "'SEQN' and 'SEQN' share 3,500 IDs (88% of the first file, 100% of the second file)." high | truth: 3500   <- correct, under the 5k threshold
# 9254 x 9254   -> "'SEQN' and 'SEQN' share 5,000 IDs (100% of the first file, 100% of the second file)." high | truth: 9254  <- ID COUNT WRONG (report says this case is clean; it is not)
# 9254 x 8000   -> "'SEQN' and 'SEQN' share 3,614 IDs (72% of the first file, 72% of the second file)." high | truth: 8000
# 20000 x 18000 -> NO KEY PROPOSED                                                                            | truth: 18000
```

---


## Finding 25

## Title
Blank/missing join keys: diagnose_join predicts rows as if blanks are dropped, execute_join merges NaN-to-NaN and fabricates cartesian participant rows

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# --- minimal case: predicted 1, actual 5 (4 fabricated rows) ---
left  = pd.DataFrame({"SEQN": [1, np.nan, np.nan], "age": [45, 38, 72]})
right = pd.DataFrame({"SEQN": [1, np.nan, np.nan], "glucose": [95, 110, 140]})
for how in ("inner", "left", "right", "outer"):
    d = diagnose_join(left, right, "SEQN", "SEQN", how)
    m, _ = execute_join(left, right, "SEQN", "SEQN", how)
    assert d.predicted_rows == len(m), (how, d.predicted_rows, len(m))  # fails for all four
print(plain_summary(diagnose_join(left, right, "SEQN", "SEQN", "inner")))
m, _ = execute_join(left, right, "SEQN", "SEQN", "inner")
print(len(m)); print(m.to_string())

# also broken without the repair step, and for text keys with empty strings
m2, _ = execute_join(left, right, "SEQN", "SEQN", "inner", repair=False)
assert len(m2) == 5
lt = pd.DataFrame({"SEQN": ["A1", "", " "], "age": [45, 38, 72]})
rt = pd.DataFrame({"SEQN": ["A1", "", ""],  "glucose": [95, 110, 140]})
assert diagnose_join(lt, rt, "SEQN", "SEQN").predicted_rows == 1
assert len(execute_join(lt, rt, "SEQN", "SEQN")[0]) == 5

# --- realistic scale: predicted 119, actual 1999, 1880 fabricated rows ---
rng = np.random.default_rng(7); n = 200
li = np.arange(1, n + 1).astype(float); ri = li.copy()
li[rng.random(n) < 0.2] = np.nan
ri[rng.random(n) < 0.2] = np.nan
L = pd.DataFrame({"SEQN": li, "age": rng.integers(20, 80, n)})
R = pd.DataFrame({"SEQN": ri, "glucose": rng.integers(60, 200, n)})
d = diagnose_join(L, R, "SEQN", "SEQN", "inner")
m, _ = execute_join(L, R, "SEQN", "SEQN", "inner")
print(plain_summary(d), "| predicted", d.predicted_rows, "| actual", len(m),
      "| fabricated", int(m["SEQN"].isna().sum()), "| row_multiplication", d.row_multiplication)
msgs = d.blocking + d.warnings + d.notes
assert not any(w in s.lower() for s in msgs
               for w in ("blank", "missing", "empty", "no id", "without an id"))
```

---


## Finding 26

## Title
NUMERIC_SENTINELS omits the positive NHANES/SPSS 7/8/9, 77/88/99 and 7777/8888 missing-code families, so narrow-range coded columns pass silently

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
import ml.import_doctor as idoc
from ml.import_doctor import diagnose, summarize

print(idoc.NUMERIC_SENTINELS)
# (-9999.0, -999.0, -99.0, -9.0, -8.0, -7.0, -1.0, 777.0, 888.0, 999.0, 9999.0, 99999.0)

df = pd.DataFrame({"diet_quality": [1,2,3,4,5,1,2,3,4,5,9,9,9,9,9]})  # 1-5 Likert, 9 = missing
print(summarize(diagnose(df)))        # -> "No structural problems detected."
print(df.diet_quality.mean(),         # 5.0 reported
      df.loc[df.diet_quality != 9].diet_quality.mean())   # 3.0 true

# Monkey-patching the list to include the positive families fixes THIS case:
idoc.NUMERIC_SENTINELS = (-9999.,-999.,-99.,-9.,-8.,-7.,-1.,
                          7.,8.,9.,77.,88.,99.,777.,888.,999.,7777.,8888.,9999.,99999.)
print([f.detail for f in diagnose(df)])
# -> ["Found 9 (5x) - far outside the rest of the column (1 to 5)."]

# ...but it does NOT fix the multi-code case the report also cites (7 AND 8 AND 9 in one item):
df2 = pd.DataFrame({"q": [1,2,3,4,5]*6 + [7,7,8,8,9,9]})
print([f.id for f in diagnose(df2) if f.id.startswith("sentinel_missing")])   # -> [] still
```

---


## Finding 27

## Title
check_numeric_stored_as_text silently skips text columns whose values are all plain numbers (the `raw_numeric >= 0.99` guard), so promote_header output gets a false all-clear

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import io, pandas as pd
from ml.import_doctor import diagnose, apply_fix, summarize, check_numeric_stored_as_text

# 1) minimal: a pure text-numeric column is invisible
print(diagnose(pd.DataFrame({"age": ["31","32","33","34","35","36","37"]})))
# -> []   ("No structural problems detected.")

# 2) proof it is the raw_numeric >= 0.99 guard, not the parser:
#    add ONE unit-bearing value and the same column is flagged
print([f.id for f in check_numeric_stored_as_text(
    pd.DataFrame({"age": ["31","32","33","34","35","36","37 yrs"]}))])
# -> ['numeric_as_text__age']

# 3) flagship Excel scenario, end to end
csv = "Nutrition Cohort Study 2024,,,\nExported 2024-03-14,,,\n,,,\nsubject_id,age,bmi,site\n" + \
      "".join(f"S{i:03d},{30+i},{22+i*0.4:.1f},Boston\n" for i in range(1,13)) + "Total,,,\n"
raw = pd.read_csv(io.StringIO(csv))
fixed, desc = apply_fix(raw, diagnose(raw)[0])      # "Promoted row 3 to column headers..."
print(fixed.dtypes.tolist())                        # all str
print([f.id for f in diagnose(fixed)])              # ['footer_rows', 'constant_columns'] - no age/bmi
print(summarize(diagnose(fixed)))                   # 'Found 1 worth checking, 1 note.'
print(pd.api.types.is_numeric_dtype(fixed["age"]))  # False
fixed["age"].mean()                                 # TypeError: Cannot perform reduction 'mean' with string dtype
```

---


## Finding 28

## Title
Key detection collapses above ~10,000 rows: `_prep` samples both sides independently, so the true key's measured overlap decays as 5000/N

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import suggest_best, find_key_candidates, diagnose_join

# Realistic form: two NHANES files, different row counts, both sorted ascending,
# right file a strict subset of the left. No reversal/shuffle trickery needed.
nL, nR = 19942, 17000
rs = np.random.RandomState(3)
lids = np.arange(1_000_000, 1_000_000 + nL)
rids = np.sort(rs.choice(lids, nR, replace=False))
a = pd.DataFrame({"SEQN": lids, "sbp":   rs.normal(size=nL)})
b = pd.DataFrame({"SEQN": rids, "hba1c": rs.normal(size=nR)})

print("suggest_best:", suggest_best(a, b))                       # -> None
for c in find_key_candidates(a, b):
    print(c.left_col, c.right_col, c.confidence, round(c.coverage_left, 4), c.n_matched)
                                                                 # -> SEQN SEQN low 0.2512 1256
print("truth:", diagnose_join(a, b, "SEQN", "SEQN").matched_keys) # -> 17000
```

---


## Finding 29

## Title
check_numeric_stored_as_text coerces a mixed-unit column onto one numeric scale at 'high' confidence, never naming the units it stripped

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

# Stronger form than the original report: the minority unit is invisible in the
# three examples the detail message shows, so nothing in the UI text hints at it.
df = pd.DataFrame({"chol": ["180 mg/dL","195 mg/dL","210 mg/dL","175 mg/dL","188 mg/dL",
                            "165 mg/dL","200 mg/dL","192 mg/dL","4.65 mmol/L","5.04 mmol/L"]})
f = [x for x in diagnose(df) if x.id.startswith("numeric_as_text")][0]
print(f.confidence, f.auto_suggestable)   # high True
print(f.detail)
# 100% of values parse as numbers after removing units, commas and comparison
# signs (e.g. '180 mg/dL', '195 mg/dL', '210 mg/dL').
out, desc = apply_fix(df, f)
print(desc)                                # "...(removing units, separators and comparison signs)."
print(out["chol"].tolist())                # [180.0,...,192.0, 4.65, 5.04]
print(out["chol"].mean())                  # 151.47  (true mg/dL mean: 187.97)
```

---


## Finding 30

## Title
check_text_missing_tokens marks 'none'/'unknown'/'not applicable' as high-confidence, auto-suggestable missing-value recodes, silently destroying legitimate reference categories

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

# 1. Reported case - reproduces verbatim
df = pd.DataFrame({"alcohol_use": ["None","Light","Moderate","None","Heavy","Light",
                                   "None","Moderate","None","Light","None","Heavy"]})
f = [x for x in diagnose(df) if x.id.startswith("text_missing")][0]
print(f.confidence, f.auto_suggestable, "|", f.detail)
print(apply_fix(df, f)[0]["alcohol_use"].tolist())
# -> high True | 5 cell(s) contain 'none'.
# -> [nan, 'Light', 'Moderate', nan, 'Heavy', 'Light', nan, 'Moderate', nan, 'Light', nan, 'Heavy']

# 2. Worse: obstetric skip-logic column, 9 of 12 real observations destroyed
df5 = pd.DataFrame({"pregnancy_complications": ["Not applicable"]*6 +
                    ["Pre-eclampsia","Gestational diabetes","None","None","Pre-eclampsia","None"]})
f5 = [x for x in diagnose(df5) if x.id.startswith("text_missing")][0]
print(f5.confidence, f5.auto_suggestable, "|", f5.detail)
print(apply_fix(df5, f5)[0]["pregnancy_complications"].tolist())
# -> high True | 9 cell(s) contain 'none', 'not applicable'.

# 3. All-or-nothing: cannot recode real missing without destroying 'None'
df3 = pd.DataFrame({"meds": ["None","Statin","N/A","None","Beta-blocker","None",
                             "Statin","","None","Metformin","None","Statin"]})
f3 = [x for x in diagnose(df3) if x.id.startswith("text_missing")][0]
print(f3.params["values"])          # ['', 'n/a', 'none'] - one bundled action
print(apply_fix(df3, f3)[0]["meds"].tolist())   # 7 of 12 -> nan

# 4. The damning internal inconsistency: 999 in an age column, which IS an
#    unambiguous sentinel and is validated by a distributional-outlier test,
#    is only 'medium' and NOT auto-suggestable.
df2 = pd.DataFrame({"age": [40,55,61,999,47,52,33,999,61,44,39]})
print([(x.id, x.confidence, x.auto_suggestable) for x in diagnose(df2)])
# -> [('sentinel_missing__age', 'medium', False)]
```

---


## Finding 31

## Title
melt_repeated overwrites an existing 'measurement' column and emits duplicate column names (and crashes outright on an existing 'value' column)

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```

```

---


## Finding 32

## Title
check_header_in_later_row false-positives on clean narrow frames with a blank header cell: emits the single critical/high-confidence 'promote_header' finding, which drops the first data row and hides all other findings

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys, io; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

csv = ",kcal\nApple,95\nBanana,105\nRice,205\nOats,150\nMilk,103\n"
df = pd.read_csv(io.StringIO(csv))                 # ['Unnamed: 0', 'kcal']
fs = diagnose(df)
assert [(f.id, f.severity, f.confidence, f.auto_suggestable) for f in fs] == \
       [("header_in_later_row", "critical", "high", True)]
out, desc = apply_fix(df, fs[0])
assert list(out.columns) == ["Apple", "95"] and len(out) == 4   # first data row destroyed

# masking: two genuine findings vanish purely because the header cell is blank
d = pd.DataFrame({"Unnamed: 0": ["Male","male ","Female","missing","Male","Female",
                                 "male","Female","Male","Female","Male","Female"],
                  "kcal": [95,105,205,150,103,88,120,140,99,101,111,131]})
assert [f.id for f in diagnose(d)] == ["header_in_later_row"]
assert [f.id for f in diagnose(d.rename(columns={"Unnamed: 0": "sex"}))] == \
       ["category_variants__sex", "text_missing__sex"]
print("BUG CONFIRMED")
```

---


## Finding 33

## Title
apply_fix('melt_repeated') raises an unhandled ValueError when the frame already has a column named 'value' (and silently produces duplicate columns when it has one named 'measurement')

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro is valid and runs verbatim. Extended version that also exposes the unreported sibling defect:

import sys; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

base = {'id': [1, 2, 3], 'bp_1': [120, 118, 130], 'bp_2': [122, 117, 133], 'bp_3': [119, 116, 131]}

def run(extra_col, label):
    df = pd.DataFrame({'id': base['id'], extra_col: [9, 8, 7],
                       'bp_1': base['bp_1'], 'bp_2': base['bp_2'], 'bp_3': base['bp_3']})
    f = [x for x in diagnose(df) if x.id == 'wide_repeated_measures'][0]
    try:
        out, _ = apply_fix(df, f)
        print(label, '-> OK, columns:', list(out.columns))
    except Exception as e:
        print(label, '->', type(e).__name__, e)

run('value', "column named 'value'")          # ValueError (the reported bug)
run('measurement', "column named 'measurement'")  # no error, but columns == ['id','measurement','measurement','value']

Actual output on this checkout (pandas 3.0.3):
column named 'value' -> ValueError: value_name (value) cannot match an element in the DataFrame columns.
column named 'measurement' -> OK, columns: ['id', 'measurement', 'measurement', 'value']
```

---


## Finding 34

## Title
'none'/'unknown' recoded to missing at HIGH (auto-suggestable) confidence, destroying legitimate categorical levels

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.import_doctor import diagnose, apply_fix, check_numeric_sentinels

# 1. Reported case: 'None' = "no medication", a real clinical level.
df = pd.DataFrame({'medication': ['None','Statin','Metformin','None','Insulin','None','Statin','None'],
                   'ldl': [130,95,110,145,88,150,101,138]})
f = [x for x in diagnose(df) if x.id.startswith('text_missing')][0]
print(f.confidence, f.auto_suggestable, f.severity, '|', f.detail)   # high True warning | 4 cell(s) contain 'none'.
out, desc = apply_fix(df, f)
print(desc, int(df.medication.notna().sum()), '->', int(out.medication.notna().sum()))  # 8 -> 4

# 2. No prevalence guard: a level that is 90% of the column is still wiped.
df2 = pd.DataFrame({'complications': ['None']*9 + ['Sepsis']})
f2 = [x for x in diagnose(df2) if x.id.startswith('text_missing')][0]
out2, _ = apply_fix(df2, f2)
print(f2.confidence, f2.auto_suggestable, int(df2.complications.notna().sum()), '->',
      int(out2.complications.notna().sum()))                          # high True 10 -> 1

# 3. 'Unknown' as a recorded level behaves the same.
df3 = pd.DataFrame({'smoking': ['Never','Former','Current','Unknown','Never','Current','Unknown','Former']})
f3 = [x for x in diagnose(df3) if x.id.startswith('text_missing')][0]
out3, _ = apply_fix(df3, f3)
print(f3.confidence, f3.auto_suggestable, int(df3.smoking.notna().sum()), '->',
      int(out3.smoking.notna().sum()))                                # high True 8 -> 6

# 4. The inconsistency: the numeric sibling proposes the SAME destructive fix with
#    STRONGER evidence (outlier-distance check) yet is only medium / not auto-suggestable.
n = check_numeric_sentinels(pd.DataFrame({'age':[40,55,61,999,47,52,33,999,61,44,39]}))[0]
print('numeric sibling:', n.confidence, n.auto_suggestable)           # medium False
```

---


## Finding 35

## Title
coerce_numeric silently merges incompatible units (mg/dL + mmol/L, kg + lb) into one column at 'high' confidence, with a detail message that never discloses the mixing

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro is valid and reproduces verbatim. Below is a STRENGTHENED version that removes the report's one accidental mitigation — in the original, the three example values printed in `detail` happened to include both units, so an alert user could notice. Sort the column the way a real multi-site export arrives (site A's rows, then site B's) and the diagnosis shows the user zero evidence that two units are present:

import sys; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

df = pd.DataFrame({'glucose': ['95 mg/dL','102 mg/dL','110 mg/dL','99 mg/dL','88 mg/dL','120 mg/dL',
                               '5.3 mmol/L','5.7 mmol/L','6.1 mmol/L','4.9 mmol/L']})
f = [x for x in diagnose(df) if x.id.startswith('numeric_as_text')][0]
print(f.confidence, f.auto_suggestable)  # high True
print(f.detail)
out, desc = apply_fix(df, f)
print(out.glucose.tolist(), out.glucose.mean())

Actual output:
  high True
  100% of values parse as numbers after removing units, commas and comparison signs (e.g. '95 mg/dL', '102 mg/dL', '110 mg/dL').
  [95.0, 102.0, 110.0, 99.0, 88.0, 120.0, 5.3, 5.7, 6.1, 4.9]  mean = 63.6

Every example shown is mg/dL. Nothing in the title, detail, why_it_matters, fix_label or the post-fix desc mentions that a second unit exists. The resulting mean of 63.6 "mg/dL" is not merely wrong, it is clinically inverted: it reads as hypoglycemia, when the mg/dL rows average ~102 and the mmol/L rows are ~5.5 mmol/L (~99 mg/dL) — i.e. an entirely normoglycemic cohort is rendered as hypoglycemic.
```

---


## Finding 36

## Title
coerce_numeric's methods-section description omits how many values it blanked (up to 20% of a column)

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

# high-confidence / auto-suggestable case: 96 numeric + 4 unparseable
df = pd.DataFrame({'sodium': [str(i) for i in range(96)] + ['not measured'] * 4})
f = [x for x in diagnose(df) if x.id.startswith('numeric_as_text')][0]
print(f.confidence, f.auto_suggestable)   # high True
print('DETAIL:', f.detail)                # DOES disclose: "96% of values parse ... Non-numeric leftovers: 'not measured'."
out, desc = apply_fix(df, f)
print('DESC  :', desc)                    # does NOT: "Converted 'sodium' to numeric (removing units, separators and comparison signs)."
print('non-null:', int(df.sodium.notna().sum()), '->', int(out.sodium.notna().sum()))  # 100 -> 96

# worst case allowed by the min_parse=0.8 gate: 81 numeric + 19 unparseable
df2 = pd.DataFrame({'sodium': [str(i) for i in range(81)] + ['not measured'] * 19})
f2 = [x for x in diagnose(df2) if x.id.startswith('numeric_as_text')][0]
print(f2.confidence, f2.auto_suggestable) # medium False  (NOT auto-suggestable at this end)
out2, desc2 = apply_fix(df2, f2)
print('DESC  :', desc2)                   # identical one-liner
print('non-null:', int(df2.sodium.notna().sum()), '->', int(out2.sodium.notna().sum()))  # 100 -> 81
```

---


## Finding 37

## Title
diagnose_join counts blank/NaN IDs as a shared key: predicts 2 rows, execute_join returns 8 with 6 fabricated rows, no blocking or fan-out warning

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd, numpy as np
from ml.join_doctor import diagnose_join, plain_summary, execute_join

# --- case 1: text key (as reported) ---
left  = pd.DataFrame({"pid": ["A01", "A02", None, None], "age": [40, 55, 61, 70]})
right = pd.DataFrame({"pid": ["A01", "A02", None, None, None], "glucose": [95, 102, 110, 120, 130]})
d = diagnose_join(left, right, "pid", "pid", "inner", "demographics", "labs")
print(plain_summary(d, "demographics", "labs"))   # Result: **2 rows** — matching on 3 shared IDs, ...
print("blocking:", d.blocking)                    # []
print("warnings:", d.warnings)                    # only the spurious capitalisation warning
merged, _ = execute_join(left, right, "pid", "pid", "inner")
print("actual rows:", len(merged))                # 8
print(merged.to_string())

# --- case 2: NUMERIC key — same fabrication, report missed this ---
l2 = pd.DataFrame({"pid": [1.0, 2.0, np.nan, np.nan], "age": [40, 55, 61, 70]})
r2 = pd.DataFrame({"pid": [1.0, 2.0, np.nan, np.nan, np.nan], "glucose": [95, 102, 110, 120, 130]})
d2 = diagnose_join(l2, r2, "pid", "pid", "inner", "demographics", "labs")
print(plain_summary(d2, "demographics", "labs"))  # Result: **2 rows** — matching on 2 shared IDs, ...
print("blocking:", d2.blocking, "warnings:", d2.warnings)   # [] []
print("actual rows:", len(execute_join(l2, r2, "pid", "pid", "inner")[0]))   # 8

# --- case 3: realistic CSV path (blank ID cells) reproduces identically ---
# demo.csv:  pid,age / A01,40 / A02,55 / ,61 / ,70
# labs.csv:  pid,glucose / A01,95 / A02,102 / ,110 / ,120 / ,130
# -> same "2 rows / 3 shared IDs" message, same 8-row merge

# execute_join(..., repair=False) also returns 8 rows: repair_keys is NOT the cause.
```

---


## Finding 38

## Title
diagnose_join drops missing/blank key rows from its row prediction, so predicted < actual for left/right/outer joins and plain_summary contradicts itself

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, plain_summary, execute_join

def check(name, left, right, lk="SEQN", rk="SEQN"):
    print("==", name)
    for how in ("inner", "left", "right", "outer"):
        d = diagnose_join(left, right, lk, rk, how, "demographics", "labs")
        m, _ = execute_join(left, right, lk, rk, how)
        flag = "MISMATCH" if len(m) != d.predicted_rows else ""
        print(f"  {how:6} pred={d.predicted_rows} actual={len(m)} {flag}")
        print(f"         {plain_summary(d, 'demographics', 'labs')}")
        print(f"         warnings={d.warnings} notes={d.notes}")

# 1. Reported case: NaN IDs on the left. left/outer predict 3, actually 5.
#    inner predicts 3 == 3 but silently drops 2 rows with NO warning at all.
check("NaN left numeric",
      pd.DataFrame({"SEQN": [1, 2, 3, np.nan, np.nan], "age": [40, 55, 61, 70, 80]}),
      pd.DataFrame({"SEQN": [1, 2, 3], "glucose": [95, 102, 110]}))

# 2. Mirror image: NaN on the right. right predicts 2 vs 3, outer 3 vs 4.
check("NaN right numeric",
      pd.DataFrame({"SEQN": [1, 2, 3], "age": [40, 55, 61]}),
      pd.DataFrame({"SEQN": [1, 2, np.nan], "glucose": [95, 102, 110]}))

# 3. Not NaN-specific: blank / whitespace-only string IDs. left/outer 2 vs 4.
check("blank string keys",
      pd.DataFrame({"SEQN": ["a01", "a02", "", "  "], "age": [40, 55, 61, 70]}),
      pd.DataFrame({"SEQN": ["a01", "a02"], "glucose": [95, 102]}))

# 4. Control: with no missing keys all four join types agree, confirming the
#    defect is specific to missing/blank keys rather than a general miscount.
check("clean control",
      pd.DataFrame({"SEQN": [1, 2, 3, 4], "age": [40, 55, 61, 70]}),
      pd.DataFrame({"SEQN": [1, 2, 3], "glucose": [95, 102, 110]}))
```

---


## Finding 39

## Title
Row-counter columns are rated "high" confidence and proposed as the best join key whenever both files use the same counter name; index_like is never surfaced to the user

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import numpy as np, pandas as pd
from ml.join_doctor import find_key_candidates, suggest_best, diagnose_join, plain_summary
RNG = np.random.RandomState(0)

# (a) two unrelated files that merely both carry a row counter
a = pd.DataFrame({"row": range(50), "age": RNG.randint(18, 80, 50)})
b = pd.DataFrame({"row": range(50), "gdp": RNG.normal(5e4, 1e4, 50)})
c = find_key_candidates(a, b)[0]
print(c.confidence, c.index_like, "|", c.headline("survey.csv", "economics.csv"))
# -> high True | 'row' and 'row' share 50 IDs (100% of survey.csv, 100% of economics.csv).
print("suggest_best ->", suggest_best(a, b).left_col)          # -> row

# (b) two sites that each numbered their participants 1..N
siteA = pd.DataFrame({"subject_id": range(1, 51), "age": RNG.randint(18, 80, 50)})
siteB = pd.DataFrame({"subject_id": range(1, 61), "glucose": RNG.normal(100, 20, 60)})
best = suggest_best(siteA, siteB)
print(best.confidence, best.index_like, "|", best.headline("site A", "site B"))
# -> high True | 'subject_id' and 'subject_id' share 50 IDs (100% of site A, 83% of site B).
d = diagnose_join(siteA, siteB, "subject_id", "subject_id", "inner", "site A", "site B")
print(plain_summary(d, "site A", "site B"), "| blocking:", d.blocking, "| warnings:", d.warnings)
# -> Result: **50 rows** ... | blocking: [] | warnings: []

# (c) NEW, stronger than the report: the counter BEATS a genuine key
seqn = RNG.choice(np.arange(80000, 90000), 60, replace=False)
A = pd.DataFrame({"row": range(60), "SEQN": seqn,       "age": RNG.randint(18, 80, 60)})
B = pd.DataFrame({"row": range(60), "patient_id": seqn, "glucose": RNG.normal(100, 20, 60)})
for k in find_key_candidates(A, B)[:2]:
    print(f"{k.left_col} <-> {k.right_col} score={k.score:.3f} conf={k.confidence} index_like={k.index_like}")
# -> row  <-> row        score=1.000 conf=high    index_like=True   <- proposed
# -> SEQN <-> patient_id score=0.688 conf=medium  index_like=False  <- the real key, demoted
print("suggest_best ->", suggest_best(A, B).left_col)          # -> row

# (d) NEW: every generic counter name triggers the same name-agreement rescue
for nm in ("index", "id", "n", "Unnamed: 0", "rownum", "obs"):
    x = pd.DataFrame({nm: range(40), "a": RNG.rand(40)})
    y = pd.DataFrame({nm: range(40), "b": RNG.rand(40)})
    k = [c for c in find_key_candidates(x, y) if c.left_col == nm][0]
    print(nm, k.confidence, bool(suggest_best(x, y)))          # -> all "high True"
```

---


## Finding 40

## Title
diagnose_join: dtype-mismatch blocker fires ahead of the zero-overlap check, producing "Fixing this matches 0 IDs" plus a contradictory "use a left join" warning

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, plain_summary, execute_join

left  = pd.DataFrame({"SEQN": ["A01", "A02", "A03"], "age": [40, 55, 61]})   # text IDs
right = pd.DataFrame({"SEQN": [1, 2, 3], "glucose": [95, 102, 110]})        # different study, numeric IDs
d = diagnose_join(left, right, "SEQN", "SEQN", "inner", "demographics.csv", "labs.xlsx")
print(plain_summary(d, "demographics.csv", "labs.xlsx"))
for b in d.blocking: print("BLOCKING:", b)
for w in d.warnings: print("WARNING :", w)
for n in d.notes:    print("NOTE    :", n)
print("matched_keys =", d.matched_keys)   # -> 0

# what the recommended action actually yields:
m, _ = execute_join(left, right, "SEQN", "SEQN", "left", "demographics.csv", "labs.xlsx")
print(m); print("glucose all-NaN:", m["glucose"].isna().all())   # -> True

# the "use a left join" advice is ALSO emitted on the honest zero-overlap branch
# (no dtype mismatch), so fixing only the if/elif ordering does not remove it:
a = pd.DataFrame({"SEQN": ["A01","A02","A03"], "age": [40,55,61]})
b = pd.DataFrame({"SEQN": ["Z9","Z8","Z7"],   "glucose": [95,102,110]})
d3 = diagnose_join(a, b, "SEQN", "SEQN", "inner", "demographics.csv", "labs.xlsx")
print(d3.blocking); print(d3.warnings)   # blocking says "nothing to join on"; warning says "Use a left join to keep them."
```

---


## Finding 41

## Title
diagnose_join emits no warning about rows that will be blank-filled on left/right/outer joins

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro is valid and reproduces verbatim. Slightly extended version (adds ground truth + the right join, which is affected identically):

import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, plain_summary

left  = pd.DataFrame({"subject_id": range(1, 101), "age": range(100)})
right = pd.DataFrame({"subject_id": range(50, 151), "glucose": range(101)})
for how in ("left", "right", "outer", "inner"):
    d = diagnose_join(left, right, "subject_id", "subject_id", how, "screening", "followup")
    print(how, "|", plain_summary(d, "screening", "followup"))
    print("   blocking:", d.blocking, "warnings:", d.warnings, "notes:", d.notes)

m = left.merge(right, on="subject_id", how="left")
print("actual left rows:", len(m), "blank glucose:", int(m["glucose"].isna().sum()))
m2 = left.merge(right, on="subject_id", how="outer")
print("actual outer rows:", len(m2), "blank glucose:", int(m2["glucose"].isna().sum()),
      "blank age:", int(m2["age"].isna().sum()))

Actual output:
left   | Result: **100 rows** - matching on 51 shared IDs, keeping every row of screening.   blocking: [] warnings: [] notes: []
right  | Result: **101 rows** - matching on 51 shared IDs, keeping every row of followup.    blocking: [] warnings: [] notes: []
outer  | Result: **150 rows** - matching on 51 shared IDs, keeping every row of both files.  blocking: [] warnings: [] notes: []
inner  | Result: **51 rows** ...  warnings: ['49 row(s) of screening (49%) have no match and will be dropped. Use a left join to keep them.']  notes: ['50 row(s) of followup have no match and will be dropped.']
actual left rows: 100 blank glucose: 49
actual outer rows: 150 blank glucose: 49 blank age: 50
```

---


## Finding 42

## Title
diagnose_join suppresses genuine column collisions whenever a key name also exists in the other frame (cross-name joins), so no suffix warning fires and execute_join's methods description names a column that no longer exists

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, "/home/user/tabular-ml-lab")
import pandas as pd
from ml.join_doctor import diagnose_join, execute_join

def show(tag, left, right, lk, rk):
    d = diagnose_join(left, right, lk, rk, "inner", "demo", "labs")
    m, desc = execute_join(left, right, lk, rk, "inner", "demo", "labs")
    print(f"--- {tag}")
    print("  collisions:", d.column_collisions, "| warnings:", d.warnings)
    print("  merged    :", list(m.columns))
    print("  desc tail :", desc.split(". ")[-1])

# Case 1 (as reported): both key names present in both files
show("both names in both files",
     pd.DataFrame({"SEQN":[1,2,3], "patient_id":["x","y","z"], "age":[40,55,61]}),
     pd.DataFrame({"patient_id":[1,2,3], "SEQN":["q","r","s"], "glucose":[95,102,110]}),
     "SEQN", "patient_id")
# collisions: []  warnings: []
# merged: ['SEQN_demo','patient_id_demo','age','patient_id_labs','SEQN_labs','glucose']
# desc:   "... on 'SEQN' ..."  (no SEQN column exists)

# Case 2 (broader trigger, NOT in the original report): only left_key collides.
# This is the more damaging shape - the join key is still renamed, one line of setup.
show("only left_key name exists in right",
     pd.DataFrame({"SEQN":[1,2,3], "age":[40,55,61]}),
     pd.DataFrame({"patient_id":[1,2,3], "SEQN":["q","r","s"], "glucose":[95,102,110]}),
     "SEQN", "patient_id")
# collisions: []  warnings: []
# merged: ['SEQN_demo','age','SEQN_labs','glucose']   <- key renamed, still no warning
# desc:   "... on 'SEQN' ..."

# Case 3: only right_key collides -> key survives, but the collision is still hidden
# AND execute_join's `drop(columns=[right_key])` cleanup silently fails to fire,
# leaving a redundant patient_id_labs duplicating the key.
show("only right_key name exists in left",
     pd.DataFrame({"SEQN":[1,2,3], "patient_id":["x","y","z"], "age":[40,55,61]}),
     pd.DataFrame({"patient_id":[1,2,3], "glucose":[95,102,110]}),
     "SEQN", "patient_id")
# merged: ['SEQN','patient_id_demo','age','patient_id_labs','glucose']
```

---


## Finding 43

## Title
Missing/blank join keys are matched to each other, fabricating participants; predicted_rows disagrees with the actual merge

## Severity when confirmed
critical

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tabular-ml-lab')
import numpy as np, pandas as pd
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# 1. Fabrication: rows with no ID are joined to each other.
demo = pd.DataFrame({'SEQN': ['1001', '1002', None], 'age': [41, 52, 63]})
labs = pd.DataFrame({'SEQN': ['1001', None, None], 'glucose': [95, 210, 300]})
d = diagnose_join(demo, labs, 'SEQN', 'SEQN', 'inner')
print(plain_summary(d), '| matched_keys =', d.matched_keys)   # '1 rows', 2 shared IDs (only 1 is real)
merged, _ = execute_join(demo, labs, 'SEQN', 'SEQN', 'inner')
print(len(merged)); print(merged)   # 3 rows; age 63 paired with BOTH unidentified lab rows

# 2. Not repair-specific: pandas .merge matches NaN to NaN on its own.
print(len(execute_join(demo, labs, 'SEQN','SEQN','inner', repair=False)[0]))   # 3

# 3. Every join type disagrees (numeric keys; ''/'  ' cells behave the same).
L = pd.DataFrame({'id': [1, 2, np.nan], 'a': [1, 2, 3]})
R = pd.DataFrame({'id': [1, np.nan, np.nan], 'b': [4, 5, 6]})
for how in ('inner','left','right','outer'):
    print(how, diagnose_join(L,R,'id','id',how).predicted_rows, len(execute_join(L,R,'id','id',how)[0]))
# inner 1 3 | left 2 4 | right 1 3 | outer 2 4

# 4. Diagnosis says the join is impossible; merge returns 64 fabricated rows.
demo2 = pd.DataFrame({'SEQN': [1001.0, 1002.0] + [np.nan]*8, 'age': range(10)})
labs2 = pd.DataFrame({'SEQN': ['A1', 'A2'] + [None]*8, 'glucose': range(10)})
d2 = diagnose_join(demo2, labs2, 'SEQN', 'SEQN', 'inner')
print(d2.can_proceed, d2.predicted_rows, len(execute_join(demo2,labs2,'SEQN','SEQN','inner')[0]))
# False 0 64

# 5. Realistic scale: 4,000 participants, 5% of IDs missing per file.
rng = np.random.default_rng(0); n = 4000
ids = [str(i) for i in range(1, n+1)]
dl = pd.DataFrame({'SEQN': ids, 'age': rng.integers(18, 80, n)})
dr = pd.DataFrame({'SEQN': ids, 'glucose': rng.normal(95, 20, n)})
dl.loc[rng.random(n) < 0.05, 'SEQN'] = None
dr.loc[rng.random(n) < 0.05, 'SEQN'] = None
d3 = diagnose_join(dl, dr, 'SEQN', 'SEQN', 'inner')
m3, _ = execute_join(dl, dr, 'SEQN', 'SEQN', 'inner')
print(d3.predicted_rows, len(m3))   # 3598 vs 46274 -> 42,676 fabricated participant rows
```

---


## Finding 44

## Title
'Convert to numbers' strips the decimal comma, silently rescaling European-format numeric columns — at 'high' confidence, marked auto-suggestable

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

for name, vals in {
    "bmi (1-digit decimal comma)": ['22,5','28,4','31,0','24,5','19,8','27,1'],
    "glucose (2-digit decimal comma)": ['5,55','6,10','4,98','7,25','5,04','6,33'],
    "de/at export (dot thousands + comma decimal)": ['1.234,5','2.100,7','980,3','1.050,25','3.400,0','2.750,8'],
    "true thousands separators (control)": ['45,000','52,300','61,000','48,000','55,500','39,900'],
}.items():
    df = pd.DataFrame({'v': vals})
    f = [x for x in diagnose(df) if x.id.startswith('numeric_as_text')][0]
    out, desc = apply_fix(df, f)
    print(f"{name}: confidence={f.confidence} auto_suggestable={f.auto_suggestable} -> {out['v'].tolist()}")

# Observed output:
# bmi (1-digit decimal comma): confidence=high auto_suggestable=True -> [225, 284, 310, 245, 198, 271]
# glucose (2-digit decimal comma): confidence=high auto_suggestable=True -> [555, 610, 498, 725, 504, 633]
# de/at export (dot thousands + comma decimal): confidence=high auto_suggestable=True -> [1.2345, 2.1007, 9803.0, 1.05025, 3.4, 2.7508]
# true thousands separators (control): confidence=high auto_suggestable=True -> [45000, 52300, 61000, 48000, 55500, 39900]
```

---


## Finding 45

## Title
join_doctor.normalize_key treats text missing-codes ('unknown', 'missing', '.', 'NA') as real shared IDs, and diagnose_join describes the resulting cross-product as legitimate repeated measures

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys, warnings, io; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tabular-ml-lab')
from data_processor import load_csv          # the app's own CSV loader
from ml.join_doctor import diagnose_join, execute_join, plain_summary

# Tokens that survive pandas' default NA parsing, so they reach join_doctor as text.
demo = load_csv(io.BytesIO(b'SEQN,age\n1001,41\nunknown,52\nunknown,63\nmissing,70\n'))
labs = load_csv(io.BytesIO(b'SEQN,glucose\n1001,95\nunknown,210\nmissing,300\n'))

d = diagnose_join(demo, labs, 'SEQN', 'SEQN', 'inner', 'demographics.csv', 'labs.csv')
m, _ = execute_join(demo, labs, 'SEQN', 'SEQN', 'inner')
print(plain_summary(d))
print('predicted', d.predicted_rows, 'actual', len(m), 'matched_keys', d.matched_keys)
print('blocking', d.blocking); print('warnings', d.warnings)
print(m.to_string())

# OBSERVED:
# Result: **4 rows** - matching on 3 shared IDs, keeping only IDs found in both files.
# predicted 4 actual 4 matched_keys 3
# blocking []
# warnings ['demographics.csv has several rows per ID (e.g. repeated visits), so 3
#           subjects become 4 rows. That is correct for repeated measures, ...']
#       SEQN  age  glucose
# 0     1001   41       95
# 1  unknown   52      210
# 2  unknown   63      210     <- two different people handed the same glucose
# 3  missing   70      300     <- an unidentified person handed a lab result

# --- Variant B: real NaN keys (what load_csv makes of literal 'NA'/'null'/'None') ---
import pandas as pd, numpy as np
demo2 = pd.DataFrame({'SEQN': ['1001', np.nan, np.nan, 'A02'], 'age': [41,52,63,70]})
labs2 = pd.DataFrame({'SEQN': ['1001', np.nan, 'A02'], 'glucose': [95,210,300]})
d2 = diagnose_join(demo2, labs2, 'SEQN', 'SEQN', 'inner')
m2, _ = execute_join(demo2, labs2, 'SEQN', 'SEQN', 'inner')
print('matched_keys', d2.matched_keys, 'predicted', d2.predicted_rows, 'actual', len(m2))
print(d2.warnings)
# OBSERVED: matched_keys 3 (NaN counted as a shared ID), predicted 2, actual 4,
# plus a spurious "Some IDs differ only by capitalisation or stray spaces" warning.
```

---


## Finding 46

## Title
'Convert to numbers' strips trailing letters off alphanumeric IDs, silently collapsing distinct participants — at 'high' confidence

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.import_doctor import diagnose, apply_fix

df = pd.DataFrame({'subject_id': ['101A','101B','102A','102B','103A','103B'],
                   'bmi': [22.1,28.4,31.0,24.5,19.8,27.1]})
f = [x for x in diagnose(df) if x.id.startswith('numeric_as_text')][0]
print(f.confidence, f.auto_suggestable, '|', f.detail)
out, _ = apply_fix(df, f)
print(out['subject_id'].tolist(), '| unique:', out['subject_id'].nunique(), 'of', len(out))

# Actual output (verbatim):
# high True | 100% of values parse as numbers after removing units, commas and
#   comparison signs (e.g. '101A', '101B', '102A').
# [101, 101, 102, 102, 103, 103] | unique: 3 of 6

# Discriminator for a correct fix (do NOT gate on uniqueness — see notes):
from ml.import_doctor import _clean_numeric_text
inc = pd.Series(["45,000","52,300","61,000","48,000","55,500","39,900",
                 "72,000","50,000","47,250","60,100","41,000"])   # legit, 100% unique
ids = pd.Series(['101A','101B','102A','102B','103A','103B'])
for n, s in (("inc", inc), ("ids", ids)):
    print(n, s.nunique(), '->', _clean_numeric_text(s).nunique())
# inc 11 -> 11   (no collapse)
# ids 6 -> 3     (collapse == information loss)
```

---


## Finding 47

## Title
Duplicated key column name crashes normalize_key and therefore diagnose_join / execute_join / repair_keys (AttributeError), and silently blanks find_key_candidates

## Severity when confirmed
minor

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
import sys, traceback; sys.path.insert(0, '/home/user/tabular-ml-lab')
import pandas as pd
from ml.join_doctor import (diagnose_join, execute_join, repair_keys,
                            find_key_candidates, suggest_best, _slug)

left  = pd.DataFrame([[1,10,1],[2,20,2]], columns=['SEQN','age','SEQN'])
right = pd.DataFrame({'SEQN':[1,2], 'glucose':[95,102]})
for name, call in (('diagnose_join', lambda: diagnose_join(left,right,'SEQN','SEQN')),
                   ('execute_join',  lambda: execute_join(left,right,'SEQN','SEQN')),
                   ('repair_keys',   lambda: repair_keys(left,right,'SEQN','SEQN'))):
    try: call(); print(name, 'ok')
    except Exception as e: print(name, '->', type(e).__name__, e)
print('candidates:', find_key_candidates(left, right))   # []
print('suggest_best:', suggest_best(left, right))        # None

# Second half of the report — also reproduces, but it is a DIFFERENT defect:
print(_slug('cohort_baseline_measurements_2019.csv'),
      _slug('cohort_baseline_measurements_2020.csv'))    # both 'cohort_baseline_meas'
merged, _ = execute_join(pd.DataFrame({'SEQN':[1,2],'bmi':[22.,28.]}),
                         pd.DataFrame({'SEQN':[1,2],'bmi':[22.5,28.5]}),
                         'SEQN','SEQN', left_name='cohort_baseline_measurements_2019.csv',
                         right_name='cohort_baseline_measurements_2020.csv')
print(list(merged.columns))   # ['SEQN', 'bmi_cohort_baseline_meas', 'bmi_cohort_baseline_meas']
```

---


## Finding 48

## Title
find_key_candidates samples each side positionally (5,000 rows, random_state=42), so on files over 5,000 rows the reported overlap is a sampling artefact — wrong counts/percentages at "high" confidence, and above ~20k rows the true key is downgraded to "low" or dropped entirely

## Severity when confirmed
major

## Verifier's reasoning
(none)

## Corrected repro (as recorded then)
```
The reported repro is valid and reproduces verbatim on first run. Below is a deterministic version (no unseeded np.random) that also covers two cases the report missed.

import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tabular-ml-lab')
import numpy as np, pandas as pd
from ml.join_doctor import suggest_best, find_key_candidates, diagnose_join

# A. Reported case — different lengths, 100% of left is in right
for nl, nr in [(9254, 12000), (20000, 30000), (200000, 260000)]:
    a = pd.DataFrame({'SEQN': np.arange(83732, 83732+nl)})
    b = pd.DataFrame({'SEQN': np.arange(83732, 83732+nr)})
    c = suggest_best(a, b)
    print(nl, nr, '->', 'NONE' if c is None else (c.headline(), c.confidence),
          '| true matched =', diagnose_join(a, b, 'SEQN', 'SEQN').matched_keys,
          '| n candidates =', len(find_key_candidates(a, b)))

# B. Stronger case the report missed: EQUAL lengths, IDENTICAL cohort, only row
#    order differs (a plain 1:1 join that cannot fail).
ids = [f"PT-{i:06d}" for i in range(9000)]
a2 = pd.DataFrame({'SEQN': ids})
b2 = pd.DataFrame({'SEQN': list(np.random.default_rng(0).permutation(ids))})
c = suggest_best(a2, b2)
print('identical 9,000-subject cohort, shuffled ->', c.headline(), c.confidence,
      '| true matched =', diagnose_join(a2, b2, 'SEQN', 'SEQN').matched_keys)

Observed (matches the report exactly on A):
  9254 12000  -> "'SEQN' and 'SEQN' share 2,762 IDs (55% of the first file, 55% of the second file)." confidence='high' | true matched = 9254
  20000 30000 -> NONE (sole candidate cov 0.267/0.267, confidence='low') | true matched = 20000
  200000 260000 -> NONE, and find_key_candidates returns [] — zero candidates | true matched = 200000
  identical 9,000-subject cohort, shuffled -> "'SEQN' and 'SEQN' share 2,765 IDs (55% ..., 55% ...)" confidence='high' | true matched = 9000
```

---
