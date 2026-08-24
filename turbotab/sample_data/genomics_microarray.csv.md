# `genomics_microarray.csv` — microarray log2 intensity (RMA)

**Derived** by rank-transforming each sample's counts onto the 2–16 interval an
RMA-summarized Affymetrix matrix occupies, with small Gaussian jitter (seeded at
50), then clipping to [2, 16]. Columns are renamed to **probe-set IDs**
(`1000000_at` …) because that is half the signature.

- rows are **samples** (60), probe columns 495
- continuous **2–16**, **no zeros**, no negatives

**Signature** — §02 row 8: *continuous 2–16, no zeros, probe-style IDs →
microarray log2 intensity (RMA).*

**The coaching** — §02, `SETTLED`: the whole count toolchain does not apply;
use limma. A classifier that reports "microarray" and still offers a
negative-binomial route has produced the label and withheld the consequence.

**Honest limit**: the values are rank-derived, so they carry the counts'
*ordering* and not a real hybridization intensity distribution. That is enough
for a shape classifier and it is not enough for anything that models the values.
