# M2 wide-data benchmark (M2_CONTRACT §5)

Two synthetic tables went through the real server: HTTP via FastAPI's TestClient, with
`TURBOTAB_WORKERS=2` worker processes, exactly as the browser drives it. Stage times run from the
`running` event to the `fresh` event for the stage's final key. Request times are wall-clock.

- **wide**: 500 samples × 20,000 negative-binomial gene counts, plus `sample_id`, `age`, `sex`,
  `batch` and a continuous outcome `bmi` driven by 20 genes. That is 20,005 columns in a 25 MB
  CSV. The lens is genomics; roles are accepted as proposed; complete cases; a 20% holdout;
  5 folds; elastic net.
- **tall**: 1,000,000 rows × 30 columns of a dietary-style table (137 MB CSV). The lens is
  dietary and the outcome is `glucose`. The exclusion is `kcal` within 500–5,000.

The machine is an Apple M4 (4 performance and 6 efficiency cores) with 16 GB, running macOS 27,
Python 3.13.0, DuckDB 1.5.5, pyarrow 24 and scikit-learn 1.9. Other agents shared it during every
run (load average 2.5–3.9), so treat single numbers as ±20%.

```
venv/bin/python -m turbotab.core.bench.synth --out DIR          # writes both CSVs (seeded)
TURBOTAB_WORKERS=2 venv/bin/python -m turbotab.core.bench.run --data DIR --which both \
    --fit-timeout 420 --json after.json                          # ~6 min, 5 of them the fit
```

The raw results are in `bench-before.json` and `bench-after.json`.

## 500 × 20,000

| Step | Before | After | What changed |
|---|---:|---:|---|
| Ingest (request → table ready) | 10.96 s | **6.26 s** | Per-column counts come from Arrow instead of 100 DuckDB batches (4.9 s → 0.75 s) |
| Profile stage (summaries + lens hints) | 24.74 s | **7.18 s** | Pyarrow materialize (9.4 s → 0.3 s); `packs.suggest` reads each shape once, not twice |
| Summaries, cold | 4.70 s | **1.02 s** | Computed from Arrow columns with numpy |
| `GET /columns` (20,005 summaries, cached) | 0.12 s | 0.12 s | — |
| `GET /table` 100 rows × 40 visible columns | 0.06 s | **0.04 s** | — |
| `GET /table` 100 rows × all 20,005 columns | 0.75 s | 0.77 s | (the client sends its visible columns) |
| Roles stage | 1.80 s | 1.83 s | — |
| Findings stage | 135.5 s | **17.5 s** | `packs.reframe` recounted the count matrix once per finding (257 times); now once |
| Record the roles (`POST set_roles`, 20,004 columns) | 0.66 s | 0.78 s | — |
| Preview: complete cases (energy-free) | 0.08 s | 0.06 s | — |
| Preview: roles (lineage) | 7.20 s | **0.14 s** | A set was rebuilt inside a list comprehension: 20,000² |
| Cohort stage | 0.12 s | 0.10 s | — |
| Split stage | 0.15 s | 0.20 s | — |
| Design stage | 14.34 s | **1.29 s** | Pyarrow materialize, plus two quadratic list lookups (lineage, numeric predictors), and only non-numeric columns normalized |
| **Fit: elastic net, 5-fold CV + refit** | **≈ 21 min** (est.) | **5.2 min** (310.7 s, CV R² 0.499) | Float32 and four threads for p > n |

The "before" fit did not finish its first fold within 150 s on the server, so the run cancelled
it. Its estimate comes from one outer fold of the same pipeline measured in-process: 214 s. The
fit stage makes six such fits (five folds and the refit). The "after" fit comes from the run
recorded as `after` in `bench-after.json`. The final run (`after-final`) skipped the fit, and no
model code changed between the two runs.

### Elastic net at p > n: what each setting costs and changes

One outer fold (320 training rows × 20,004 columns) of the design's own elastic-net pipeline was
measured in-process: `ElasticNetCV`, 6 lasso-ridge mixes × 100 penalties × 5 inner folds.

| Precision, tolerance, threads | Seconds | Penalty | Mix | Genes kept | Fold R² |
|---|---:|---:|---:|---:|---:|
| float64, 1e-4, 1 (before) | 214.4 | 0.2857 | 1.0 | 77 | 0.4611 |
| float32, 1e-4, 1 | 62.7 | 0.2857 | 1.0 | 77 | 0.4611 |
| **float32, 1e-4, 4 (shipped)** | **43.8** | 0.2857 | 1.0 | 77 | 0.4611 |
| float64, 1e-3, 1 | 41.2 | 0.2857 | 1.0 | 77 | 0.4608 |
| float32, 1e-3, 1 | 14.1 | 0.2857 | 1.0 | 77 | 0.4608 |

Float32 does not change the answer. On three more p > n tables (150 × 800) it chose the same
penalty, mix and genes as float64; `test_the_wide_elastic_net_chooses_what_the_default_chooses`
holds this. A tolerance of 1e-3 would be about 4× faster again, but it does change the answer.
On one of those three tables it moved the penalty from 0.2308 to 0.3271 and the support from 40
genes to 24, with the same outer R² (0.4293 against 0.4295). It is therefore **not** shipped. It
is a choice for Nolan, filed in the inbox. The four threads scaled by only 1.4× because other jobs
held the performance cores. On an idle machine expect more.

## 1,000,000 × 30

| Step | Before | After |
|---|---:|---:|
| Ingest (request → table ready) | 2.35 s | 1.67 s |
| Profile stage | 2.46 s | 2.39 s |
| Summaries, cold | 0.72 s | 0.75 s |
| Roles stage | 0.09 s | 0.06 s |
| Findings stage | 15.1 s | 14.8 s |
| Record an exclusion | 0.04 s | 0.05 s |
| **Cohort stage** | 0.06 s | **0.04 s** |
| **Preview: exclusions** (again) | 0.38 s (0.15 s) | **0.32 s** (0.15 s) |
| Preview: complete cases (again) | 0.35 s (0.34 s) | 0.31 s (0.30 s) |

The tall table was already fast, because DuckDB's projections cost little at 30 columns. It
stays on DuckDB for summaries and statistics. Materializing through Arrow is as fast or faster
at every size measured: a 5,000-row sample, a 980,000-row cohort, and every row.

## Extremely wide CSVs

A 40,000-column × 3-row CSV with long names has a 2.4 MB header line, over DuckDB's default
`max_line_size` of 2 MB. DuckDB refuses such a line, as `test_a_line_over_two_megabytes_is_read`
shows when the limit is not raised. The ingest now reads the header, raises the limit (twice
the header, and at least 64 bytes per field), and ingests the file in 21 s. **Peak memory was
4.25 GB, all of it DuckDB's per-column state.** At 150,000 columns DuckDB alone holds about
2.7 GB just to count the rows. Tables that wide should be transposed or read feature-major,
and that is the `oriented` stage's job.

## What still dominates, and why it stays

1. **The elastic-net fit at 20,000 columns: 5 minutes.** Nested CV makes six fits of 30 paths
   each. The remaining levers either change the answer (the tolerance, the penalty grid) or
   belong to the fit stage (running the outer folds in parallel). Both are in the inbox.
2. **Findings at 20,000 columns: 17.5 s, and at 1,000,000 rows: 14.8 s.** All of it is legacy
   checks in `ml/import_doctor.py`, `ml/binary_text.py` and `turbotab/packs.py`: per-column
   pandas work, with string normalization of every value at 1M rows. That code is Classic's or
   the packs', so it is wrapped here, not edited.
3. **Lens hints at 20,000 columns: about 6 s of the profile stage**, inside `packs.likert_block`.
   It is the same kind of legacy loop.
4. **The ingest copy at 20,000 columns: about 4 s.** DuckDB's `row_number() OVER ()` adds about
   2 s by materializing every column. Adding `__row_id` in a pyarrow pass afterwards would save
   that.
