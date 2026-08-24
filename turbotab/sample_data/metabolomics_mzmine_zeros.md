# `metabolomics_mzmine_zeros.csv` — non-detections written as zeros

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_siblings.py`: every blank intensity replaced by `0.0`.
Nothing else changed. This is what the same run looks like out of MZmine,
MaxQuant or Progenesis — `METABOLOMICS_PACK.md` §01 names all three as writing
`0` where XCMS `fillPeaks` writes a small number instead.

80 rows x 400 columns; 392 features; 0 blank cells; 4,316 zeros; range 50.95 to 4.541e+06 (ratio 8.912e+04)

---

## Must surface

1. **Zeros, and nothing assumed about them** —
   `pack::metabolomics::zeros_or_missing`. The count, the fact that there are
   now **no blank cells at all**, and the four vendors' disagreement, carried in
   the payload as a list rather than flattened into the sentence. The finding is
   `offered` and says outright that the app has not defaulted, because §01 is
   explicit that defaulting wrong corrupts every downstream step.

## Must NOT surface

2. **No left-censoring finding, and that is the point of this file.**
   `pack::metabolomics::left_censored` reads a rank correlation between a
   feature's *missing rate* and its abundance. There are no missing values here,
   so the reading is unavailable — **the export destroyed the evidence for it**,
   which is the concrete cost of the vendor disagreement §01 describes. A
   generic tool sees a complete table.
- **No all-zero features and no all-zero samples.** The highest per-feature
  missing rate in the source is 55%, so no feature and no sample became empty.
  `pack::metabolomics::empty_blocks` is silent here and fires on
  `metabolomics_merged_modes.csv` instead.
