"""L50-F1 — the four sibling matrices, DERIVED from the counts fixture.

`AGENT_ONBOARD.md` §07 trap #4: *verifying against the fixture that works.*
`GENOMICS_PACK.md` §02 specifies **nine** signatures and this repository shipped
one matrix that could exercise **one** of them — `genomics_expression.csv`, 60
samples × 496 genes, 100% integer, max 2.65e4, which is raw counts. A classifier
tested only on the class it already recognizes is not tested.

**Derived rather than invented, and that is the whole design.** Each sibling is
the same 60 × 496 matrix put through the transform its name claims, so the
signatures below are *consequences* of a real operation rather than numbers
chosen to match a table. A fixture written to satisfy the classifier would be
trap #3 — the fixture manufacturing the thing whose absence is the defect —
and it would pass while proving nothing.

**Orientation, said once because it decides every statistic.** The pack's table
is written for the field convention, **genes in rows**. This app's tables are
samples in rows, so *"per column"* in §02 is **per sample** here — a row sum is
a library size. The transforms below all operate per sample for that reason.

Run: `venv/bin/python turbotab/sample_data/make_genomics_siblings.py`
Deterministic: seeded, and re-running writes byte-identical files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "genomics_expression.csv"
#: Everything that is not an expression column. Derived, not listed, because
#: the fixture carries `condition` as well and a hand list missed it — which
#: is trap #5 in the file whose subject is trap #4.
META = ("sample_id", "batch", "sex", "age", "condition")
SEED = 50


def _counts():
    frame = pd.read_csv(SOURCE)
    genes = [c for c in frame.columns
             if c not in META and frame[c].dtype.kind in "iuf"]
    return frame, genes


def _write(frame: pd.DataFrame, name: str, prose: str) -> None:
    frame.to_csv(HERE / name, index=False)
    (HERE / f"{name}.md").write_text(prose, encoding="utf-8")
    print(f"wrote {name}")


def main() -> None:
    frame, genes = _counts()
    counts = frame[genes].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)

    # ── CPM/TPM · every sample sums to exactly 1e6 ──────────────────────────
    # §02: "Every column sums to 1e6 ±1e-3 → CPM or TPM, INDISTINGUISHABLE."
    # The pack is explicit that the two cannot be told apart from the matrix,
    # and the classifier must say so rather than pick one.
    lib = counts.sum(axis=1, keepdims=True)
    cpm = counts / lib * 1e6
    out = frame.copy()
    out[genes] = np.round(cpm, 6)
    _write(out, "genomics_cpm.csv", f"""# `genomics_cpm.csv` — CPM/TPM-shaped

**Derived from `genomics_expression.csv`** by
`make_genomics_siblings.py`: each sample's counts divided by that sample's
library size and multiplied by 1e6, rounded to 6 decimals.

- rows are **samples** (60), gene columns {len(genes)}
- every row sums to **1e6** (to within rounding)
- non-integer, non-negative, no negatives

**Signature it exercises** — `GENOMICS_PACK.md` §02, row 3: *every column sums
to 1e6 ±1e-3 → CPM or TPM, indistinguishable.* The classifier must return BOTH
names and must not choose between them; the matrix does not carry the
information that would let it.

**What it must NOT be classified as**: raw counts (it is not integer) or
estimated counts (its library sizes do not vary).
""")

    # ── Estimated counts · non-integer, library sizes still vary ───────────
    # §02 row 2. salmon/kallisto/RSEM assign reads FRACTIONALLY across
    # transcripts that share sequence, so a gene-level count comes out with a
    # decimal part while the library size is still a library size. That is the
    # whole difference from row 1, and §02 says to ASK rather than assume.
    #
    # It is the hardest sibling to tell from FPKM and the fixture exists to
    # make that difficulty visible rather than to hide it: both are
    # non-integer, non-negative, with sums that vary. What separates them is
    # that estimated counts keep the library-size SPREAD of the raw matrix
    # (sumCV ~0.27) and FPKM does not (~0.10), because FPKM divided it out.
    est = counts * rng.uniform(0.97, 1.03, size=counts.shape)
    out = frame.copy()
    out[genes] = np.round(est, 3)
    _write(out, "genomics_estimated_counts.csv",
           f"""# `genomics_estimated_counts.csv` — estimated counts

**Derived** by perturbing each cell of the raw counts by a factor drawn from
U(0.97, 1.03), seeded at {SEED} — which is what fractional read assignment
looks like at gene level.

- rows are **samples** (60), gene columns {len(genes)}
- **non-integer**, non-negative, library sizes still vary widely

**Signature** — §02 row 2: *non-integer ≥0, sums vary, max ≫1e4 → estimated
counts (salmon/kallisto/RSEM) — **ask**.* The pack says ask, so the classifier
may not silently pick between this and raw counts.

**The ambiguity this fixture is FOR.** Estimated counts and FPKM are both
non-integer, non-negative, with varying sums, and §02's table separates them
only by max and skew — which overlap. The measured difference on these two
fixtures is the **library-size spread**: this matrix keeps the raw matrix's,
because nothing normalized it, and `genomics_fpkm.csv`'s is markedly smaller
because FPKM divided the library size out. A classifier that cannot tell them
apart should **say both and ask**, which is what §02 already requires here.

**No number is quoted, deliberately.** This generator draws from one seeded
stream in file order, so inserting a block above this one moves every draw
below it — which happened, and a brief that had quoted these figures went stale
the same afternoon (`TEST-056`). The spread is asserted where it can be
re-measured rather than stated where it can rot:
`turbotab/test_the_fixture_constants_match_the_fixtures.py`.
""")

    # ── TMM-scaled CPM · sums near 1e6 but not equal ────────────────────────
    # §02 row 4. The distinction from row 3 is that a scaling factor has been
    # applied per sample, so the sums cluster near 1e6 rather than hitting it.
    factors = rng.normal(1.0, 0.06, size=(counts.shape[0], 1))
    out = frame.copy()
    out[genes] = np.round(cpm * factors, 6)
    _write(out, "genomics_tmm_cpm.csv", f"""# `genomics_tmm_cpm.csv` — TMM-scaled CPM

**Derived** from the CPM matrix above by multiplying each sample by a scaling
factor drawn from N(1, 0.06), seeded at {SEED}.

- rows are **samples** (60), gene columns {len(genes)}
- row sums cluster **near 1e6 and none equals it**
- non-integer, non-negative

**Signature** — §02 row 4: *sums roughly but not exactly equal near 1e6 →
TMM- or median-of-ratios-scaled CPM.*

**Why it is a separate fixture from `genomics_cpm.csv`**: the two differ only
in whether the sum is exact, and a classifier that collapses them is wrong in a
way that matters — §04 records that TMM and median-of-ratios controlled the
false-positive rate where total-count scaling did not (Dillies 2013).
""")

    # ── FPKM/RPKM · length-normalized, sums NOT constant ────────────────────
    # §02 row 5. Gene lengths are the thing FPKM divides by, so a plausible
    # length vector is drawn once and applied to every sample — which is what
    # makes FPKM non-comparable ACROSS samples (Wagner 2012, §02's own
    # citation) and is visible here as row sums that do not agree.
    lengths = rng.lognormal(mean=7.4, sigma=0.6, size=len(genes))  # ~1.6 kb
    fpkm = counts / lib * 1e9 / lengths
    out = frame.copy()
    out[genes] = np.round(fpkm, 4)
    _write(out, "genomics_fpkm.csv", f"""# `genomics_fpkm.csv` — FPKM/RPKM-shaped

**Derived** by dividing each sample's counts by its library size and by a
per-gene length drawn once from a lognormal centered near 1.6 kb (seeded at
{SEED}), then scaling by 1e9 — the FPKM formula.

- rows are **samples** (60), gene columns {len(genes)}
- non-negative, **non-integer**, row sums **not constant**
- heavy right skew

**Signature** — §02 row 5: *non-negative, sums not constant, max 1e3–1e5, heavy
skew, non-integer → FPKM/RPKM.*

**Why the length vector is applied per gene and not per cell**: that is what
FPKM does, and it is precisely why FPKM is not comparable across samples *even
in principle* — §02 cites Wagner, Kim & Lynch (*Theory Biosci* 131:281, 2012)
for the invariance it violates. The row sums disagreeing here is that fact,
visible.
""")

    # ── VST · variance-stabilized, homoscedastic, a repeated floor ──────────
    # §02 row 6. DESeq2's VST maps counts onto a log2-like scale with a FLOOR
    # that many low-count genes share — the repeated floor is the signature,
    # so it is produced rather than added: log2 of a size-factor-normalized
    # count plus a pseudocount does exactly this.
    normed = counts / (lib / lib.mean())
    vst = np.log2(normed + 4.0)
    out = frame.copy()
    out[genes] = np.round(vst, 4)
    floor = float(np.round(np.log2(4.0), 4))
    _write(out, "genomics_vst.csv", f"""# `genomics_vst.csv` — VST-shaped

**Derived** by size-factor-normalizing the counts and taking
`log2(x + 4)` — which reproduces VST's two defining features without pretending
to be DESeq2's estimator.

- rows are **samples** (60), gene columns {len(genes)}
- continuous, max well under 25, **no negatives**
- a **repeated floor at {floor}**, shared by every zero count

**Signature** — §02 row 6: *continuous, max ~15–25, repeated floor, roughly
homoscedastic → VST.*

**The coaching this fixture must trigger is the load-bearing part.** §02:
VST and rlog are for visualization, clustering and PCA, and are **never** the
input to a DE test — `SETTLED`. A classifier that recognizes the shape and does
not say that has produced a label rather than a diagnosis.

**Not covered**: rlog, §02 row 7, which is *as above but small negatives
permitted*. `wide_assay.csv` is symmetric around zero and exercises row 9
instead; a true rlog fixture would need a negative floor and is not shipped.
""")

    # ── Microarray log2 RMA · 2–16, no zeros, probe-style IDs ───────────────
    # §02 row 8. The probe IDs are the tell, so the columns are RENAMED — a
    # microarray matrix that kept gene symbols would be missing half its own
    # signature.
    ranks = counts.argsort(axis=1).argsort(axis=1) / max(len(genes) - 1, 1)
    rma = 2.5 + ranks * 12.5 + rng.normal(0, 0.25, size=counts.shape)
    rma = np.clip(rma, 2.0, 16.0)
    probes = [f"{1000000 + i * 7}_at" for i in range(len(genes))]
    out = frame[list(META)].copy()
    for probe, column in zip(probes, range(len(genes))):
        out[probe] = np.round(rma[:, column], 4)
    _write(out, "genomics_microarray.csv", f"""# `genomics_microarray.csv` — microarray log2 intensity (RMA)

**Derived** by rank-transforming each sample's counts onto the 2–16 interval an
RMA-summarized Affymetrix matrix occupies, with small Gaussian jitter (seeded at
{SEED}), then clipping to [2, 16]. Columns are renamed to **probe-set IDs**
(`1000000_at` …) because that is half the signature.

- rows are **samples** (60), probe columns {len(genes)}
- continuous **2–16**, **no zeros**, no negatives

**Signature** — §02 row 8: *continuous 2–16, no zeros, probe-style IDs →
microarray log2 intensity (RMA).*

**The coaching** — §02, `SETTLED`: the whole count toolchain does not apply;
use limma. A classifier that reports "microarray" and still offers a
negative-binomial route has produced the label and withheld the consequence.

**Honest limit**: the values are rank-derived, so they carry the counts'
*ordering* and not a real hybridization intensity distribution. That is enough
for a shape classifier and it is not enough for anything that models the values.
""")


if __name__ == "__main__":
    main()
