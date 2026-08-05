# `genomics_cpm.csv` — CPM/TPM-shaped

**Derived from `genomics_expression.csv`** by
`make_genomics_siblings.py`: each sample's counts divided by that sample's
library size and multiplied by 1e6, rounded to 6 decimals.

- rows are **samples** (60), gene columns 495
- every row sums to **1e6** (to within rounding)
- non-integer, non-negative, no negatives

**Signature it exercises** — `GENOMICS_PACK.md` §02, row 3: *every column sums
to 1e6 ±1e-3 → CPM or TPM, indistinguishable.* The classifier must return BOTH
names and must not choose between them; the matrix does not carry the
information that would let it.

**What it must NOT be classified as**: raw counts (it is not integer) or
estimated counts (its library sizes do not vary).
