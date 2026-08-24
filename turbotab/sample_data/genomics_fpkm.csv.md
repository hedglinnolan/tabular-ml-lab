# `genomics_fpkm.csv` — FPKM/RPKM-shaped

**Derived** by dividing each sample's counts by its library size and by a
per-gene length drawn once from a lognormal centered near 1.6 kb (seeded at
50), then scaling by 1e9 — the FPKM formula.

- rows are **samples** (60), gene columns 495
- non-negative, **non-integer**, row sums **not constant**
- heavy right skew

**Signature** — §02 row 5: *non-negative, sums not constant, max 1e3–1e5, heavy
skew, non-integer → FPKM/RPKM.*

**Why the length vector is applied per gene and not per cell**: that is what
FPKM does, and it is precisely why FPKM is not comparable across samples *even
in principle* — §02 cites Wagner, Kim & Lynch (*Theory Biosci* 131:281, 2012)
for the invariance it violates. The row sums disagreeing here is that fact,
visible.
