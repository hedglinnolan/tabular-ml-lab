# `metabolomics_paired_logged.csv` — already log2, and subject ids that repeat

**The pairing in this file is a LABEL, not a discovery.** The intensities are
`metabolomics_untargeted.csv`'s, log2-transformed and nothing else. The
`subject_id` and `timepoint` columns were **written by
`make_metabolomics_siblings.py`**, assigning consecutive participant rows to
36 subjects of two samples each. There is no biological pairing in these
numbers and **no claim about paired biology may rest on this file.** It exists
to exercise the design detectors, which read the roster's shape and are correct
about the shape.

**Derived from `metabolomics_untargeted.csv`** by two operations:

1. every intensity replaced by its base-2 logarithm, rounded to 6 decimals —
   what a user who normalized in MetaboAnalyst and re-exported uploads;
2. `subject_id` and `timepoint` inserted as described above. The 8 pooled-QC
   rows carry neither, which is right: a QC injection has no subject.

80 rows x 402 columns; 392 features; 4,316 blank cells; 0 zeros; range 5.671 to 22.11 (ratio 3.9)

---

## Must surface

1. **Already transformed** — `pack::metabolomics::already_transformed`, at
   `critical`. §01: a max below ~40 with a positive minimum and a low dynamic
   range. Here the whole block runs 5.67 to 22.11. A second log
   transform on this is the silent catastrophe the pack names.
   The marker is `offered` rather than `derived`: there are no negatives, and a
   compressed range has an innocent reading (a targeted panel in µM), so the
   app offers the reading instead of asserting it.
2. **Subject ids repeat** — `pack::metabolomics::repeated_subjects`. 72 samples
   from 36 subjects. The finding **routes to the grain question the lockbox
   already holds** and asks nothing: `params["routes_to"] == "set_grain"`, and
   `params["group_column"]` is taken from `grain.suggestion(df)["columns"]`
   rather than chosen here, so the two cannot name different columns.
3. **The acquisition inventory** now includes a timepoint column.

## Must NOT surface

- **No second card about the range.** The compressed range is a SIGNAL of
  `already_transformed`, not a finding beside it: §01's two bullets are one
  reading, where the second states the threshold the first uses. Two cards at
  different strengths about one fact is the app hedging.
- **No second grain question.** The pack finding carries `fix_kind="none"`, so
  it is structurally incapable of adding one.
