# `genomics_estimated_counts.csv` — estimated counts

**Derived** by perturbing each cell of the raw counts by a factor drawn from
U(0.97, 1.03), seeded at 50 — which is what fractional read assignment
looks like at gene level.

- rows are **samples** (60), gene columns 495
- **non-integer**, non-negative, library sizes still vary widely

**Signature** — §02 row 2: *non-integer ≥0, sums vary, max ≫1e4 → estimated
counts (salmon/kallisto/RSEM) — **ask**.* The pack says ask, so the classifier
may not silently pick between this and raw counts.

**The ambiguity this fixture is FOR.** Estimated counts and FPKM are both
non-integer, non-negative, with varying sums, and §02's table separates them
only by max and skew — which overlap. The measured difference on these two
fixtures is the library-size spread: this matrix keeps the raw matrix's
coefficient of variation (~0.27) because nothing normalized it, and
`genomics_fpkm.csv` sits near 0.10 because FPKM divided the library size out.
A classifier that cannot tell them apart should **say both and ask**, which is
what §02 already requires for this row.
