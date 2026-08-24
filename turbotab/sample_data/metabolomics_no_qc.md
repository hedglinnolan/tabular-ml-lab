# `metabolomics_no_qc.csv` — the export with the QCs and the run order removed

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_siblings.py`: the 8 pooled-QC rows dropped, and the
`run_order` and `batch` columns dropped. **No value was altered.**

72 rows x 398 columns; 392 features; 4,037 blank cells; 0 zeros; range 51.33 to 4.541e+06 (ratio 8.846e+04)

This is the commonest shape a metabolomics table actually arrives in: an
analyst exports "the data", meaning the biological samples and their
measurements, and the injections that were the instrument checking itself go
with the acquisition metadata. Both absences are **unrecoverable from what is
left** — which is precisely why `METABOLOMICS_PACK.md` §01 asks the tool to say
so rather than proceed quietly.

---

## Must surface

1. **No pooled QC samples** — `pack::metabolomics::no_pooled_qc`. §01's coaching
   sentence, quoted: without pooled QCs, QC-RSD, the D-ratio and drift
   correction cannot be computed *by any tool* from this file, and QCs that were
   run but not exported cannot be reconstructed later.
2. **No run order at all** — `pack::metabolomics::no_run_order`. Said loudly,
   per §01, and naming the three diagnostics that become impossible: drift,
   QC-RLSC, the run-order PCA overlay.
3. **Missingness is still detection-limit shaped** —
   `pack::metabolomics::left_censored`. Unchanged by the row removal, and it is
   here to show that dropping the QCs did not cost the left-censoring reading.

## Must NOT surface

- **No run-order finding.** `pack::metabolomics::run_order` reads a column that
  is a permutation of the row positions; there is no longer one, and the
  name-blind reading and the name-based one agree on that because they are the
  same reading (`packs._permutation_column`).
- **No pooled-QC finding.** `sample_type` is now constant, so the variance
  reading has no minority level to compare, and the naming reading finds
  nothing. Two detectors, one silence, no contradiction between them.
- **No acquisition inventory.** There is nothing left to inventory.
