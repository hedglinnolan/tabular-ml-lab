# `genomics_tmm_cpm.csv` — TMM-scaled CPM

**Derived** from the CPM matrix above by multiplying each sample by a scaling
factor drawn from N(1, 0.06), seeded at 50.

- rows are **samples** (60), gene columns 495
- row sums cluster **near 1e6 and none equals it**
- non-integer, non-negative

**Signature** — §02 row 4: *sums roughly but not exactly equal near 1e6 →
TMM- or median-of-ratios-scaled CPM.*

**Why it is a separate fixture from `genomics_cpm.csv`**: the two differ only
in whether the sum is exact, and a classifier that collapses them is wrong in a
way that matters — §04 records that TMM and median-of-ratios controlled the
false-positive rate where total-count scaling did not (Dillies 2013).
