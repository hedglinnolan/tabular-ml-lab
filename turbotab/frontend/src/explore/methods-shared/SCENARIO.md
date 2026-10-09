# The shared scenario of the living-methods prototypes

All three prototypes of the living methods section (BLUEPRINT §11.4) — A, the paper
(`/lab/methods-document`); B, the quest log (`/lab/methods-questlog`); C, the map
(`/lab/methods-map`) — show one analysis. Each captured it from the real server by driving
`scenario.py` (`run_inference`, `run_prediction`), so every sentence, guess and number they show is
the same engine output on the same answers. The only difference between them is the design.

## The analysis

- **Data.** The NHANES dietary export (`_tt_tmp_nhanes.csv`, 21,849 rows × 29 columns; the
  acceptance tests find it through `TURBOTAB_NHANES_CSV`).
- **Opening.** The dietary lens; the outcome `glucose`, a regression; the purpose inference.
- **Roles and readings.** The roles as the engine proposes them. Every reading is answered from
  the fixture's declared truth (`turbotab/core/tests/truths.py`): `kcal` is one day's energy in
  kcal (asked by the screens); three role readings are confirmed one at a time (`bp_di`, `bp_sys`,
  `cycle_begin_year`, the first three the card lists), which unlocks the block confirm of the
  other 13; the fit's card asks whether `age` and `cycle_begin_year` hold amounts or codes, and
  one block answers both.
- **Rows.** Every row is kept for the primary analysis. Two screens are declared as secondary
  analyses, the same model on their own rows: Willett 2013 by sex (20,235 rows) and NHS/HPFS by sex
  (20,430 rows). Complete cases. No holdout under inference: 5-fold cross-validation, seed 0.
- **The question.** The exposure `sugar`, its total effect, as a substitution (in place of other
  energy sources at fixed total energy), on the mean-difference scale.
- **The adjustment set.** The pack's guess for each group it guesses (age and gender confounders,
  the six other nutrients confounders, body size of unknown timing: left out of the primary and
  added in Model 3). The seven covariates with no guess are answered from the truth:
  `cycle_begin_year` a confounder, and the blood pressures, HDL, triglycerides and the two
  medications mediators, left out of a total effect.
- **Energy.** The engine's default, the method it ranks first among those that run on these
  columns: the standard (multivariate) model.
- **Models.** The declared sequence with Model 1 adjusted for `age`, `gender` and `kcal`; the
  linear family. The first estimate served locks the plan (SHA-256 `c9efee9fb0b2…`).
- **Results.** Table 2 for `sugar`, and "Which of my decisions mattered?" across the declared
  alternatives (the model sequence and the two screens).

| Model | Difference in mean `glucose` per unit of `sugar` | 95% interval | Rows |
| --- | --- | --- | --- |
| Unadjusted | −0.0345 | −0.0417 to −0.0273 | 21,849 |
| Model 1 (age, gender, kcal) | −0.0262 | −0.0357 to −0.0167 | 21,849 |
| **Model 2, primary** | **−0.0199** | **−0.0327 to −0.00718** | 21,849 |
| Model 3 (plus body size) | −0.0223 | −0.0349 to −0.00978 | 21,849 |

## The prediction variant

The same table and the same opening and readings under prediction: every row kept, missing values
imputed in each training fold (the engine's first method), a 20% holdout (seed 0) with 5-fold
cross-validation on the rest, the standard energy model, and the linear model beside
gradient-boosted trees.

## Reproducing it

```
TURBOTAB_HOME=$(mktemp -d) TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 \
    venv/bin/python -m turbotab.server --port 8961                       # repo root
venv/bin/python turbotab/frontend/src/explore/methods-shared/scenario.py --base http://127.0.0.1:8961
```

It prints each moment of the path and Table 2. Each prototype's own capture script drives the same
path and takes what its design shows at each moment.
