# `metabolomics_merged_modes.csv` — the two polarities merged, badly

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_siblings.py`, through five operations. Each is a thing a real
two-polarity merge really does; none was chosen to make a detector fire, and
each is stated here so a later reader can check that.

| # | operation | consequence |
|---|---|---|
| 1 | every blank intensity replaced by `0.0` | the merge writes zeros for non-detections |
| 2 | features 1–196 renamed `…_pos`, features 197–392 renamed `…_neg` | both acquisitions in one table |
| 3 | 8 `_neg` features set to `0.0` throughout | carried in from the other polarity's feature list, never detected |
| 4 | 5 `_pos` features set to the constant `79.17` | XCMS `fillPeaks` writes a small number, not a zero (§01) — the block's 1st percentile |
| 5 | row 79 (`S080`) zeroed across both blocks except the 5 gap-filled features | a failed injection the merger still emitted a row for; the gap filler wrote its constant into that row too |
| 6 | row 40's `sample_id` set to `S040` | a sample re-injected after a failed acquisition, exported under the original id |
| 7 | 6 `_pos` columns written a second time under the same name | the merge emitted them twice |

80 rows x 406 columns; 392 features; 0 blank cells; 5,194 zeros; range 50.95 to 4.541e+06 (ratio 8.912e+04)

**Operation 7 is the one worth reading twice.** `read_csv` renames a repeated
column label to `name.1`, so a duplicate feature id **never arrives as a
duplicate** — it arrives silently renamed, and every "we measured N metabolites"
count downstream is 6 too high. The detector reads both forms: a
literal repeat, which only a constructed frame carries, and the renaming
signature, which is what any real CSV carries.

---

## Must surface

1. **Both ion modes in one table** — `pack::metabolomics::ion_modes`, at `info`,
   marked **CONVENTION**. That the polarities are normalized separately and
   merged afterwards is what the field does, not a result anybody established,
   and the badge is the difference between saying so and asserting it.
2. **Duplicate features and a duplicate sample** —
   `pack::metabolomics::duplicate_ids`. 6 renamed columns and one
   `sample_id` on two rows.
3. **Rows and columns carrying no information** —
   `pack::metabolomics::empty_blocks`. 8 all-zero features,
   5 constant features, 1 empty sample. All three in one finding
   because they are one condition, counted separately in the payload because
   they are repaired differently.
4. **Zeros, and nothing assumed** — `pack::metabolomics::zeros_or_missing`, as
   on `metabolomics_mzmine_zeros.csv`.

## Must NOT surface

- **No left-censoring finding.** Same as the zero-filled sibling: the blanks are
  gone, so the reading has nothing to rest on.
- **No claim about which merge strategy is right.** §01 says the strategy
  affects the results and does not say which to use, so neither does the app.
