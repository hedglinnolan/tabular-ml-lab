"""L50-D — four sibling tables, DERIVED from `metabolomics_untargeted.csv`.

`AGENT_ONBOARD.md` §07 trap #4: *verifying against the fixture that works.*
`METABOLOMICS_PACK.md` §01 specifies three diagnostic families and this
repository shipped **one** metabolomics table — 80 samples, 392 features, pooled
QCs present, a run-order column present, no zeros, no negatives, no duplicates,
one polarity. Six of §01's diagnostics had nothing on disk to fire on, and a
detector verified only against the file it was written for is not verified.

**Derived rather than invented, and that is the whole design.** Each sibling is
the same 80 x 392 matrix put through an operation a real export really performs,
so what the detectors then see is a *consequence* of that operation rather than
a number chosen to make a detector fire. A fixture written to satisfy a detector
is trap #3 — the fixture manufacturing the thing whose absence is the defect —
and it passes while proving nothing.

**Where a sibling carries a LABEL rather than a transform, the companion says
so in its first paragraph.** `metabolomics_paired_logged.csv` carries a subject
roster that is a design label added here, not a biological pairing discovered in
the data, and no claim about paired biology may rest on it. That distinction is
the difference between a derived fixture and a fabricated one, and it is stated
on the file rather than in this docstring, because the file is what a later loop
will read.

Run: `venv/bin/python turbotab/sample_data/make_metabolomics_siblings.py`
Deterministic: seeded, and re-running writes byte-identical files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "metabolomics_untargeted.csv"
#: Everything that is not an intensity column. Derived from the prefix rather
#: than hand-listed, because the hand list is what missed `condition` in
#: `make_genomics_siblings.py` — trap #5 in the file whose subject is trap #4.
FEATURE_PREFIX = "mz_"
SEED = 50


def _source():
    frame = pd.read_csv(SOURCE)
    features = [c for c in frame.columns if str(c).startswith(FEATURE_PREFIX)]
    return frame, features


def _write(frame: pd.DataFrame, name: str, prose: str) -> None:
    frame.to_csv(HERE / name, index=False)
    (HERE / f"{name.rsplit('.', 1)[0]}.md").write_text(prose, encoding="utf-8")
    print(f"wrote {name}  {frame.shape[0]} x {frame.shape[1]}")


def _stats(frame: pd.DataFrame, features) -> str:
    values = frame[features].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    return (f"{frame.shape[0]} rows x {frame.shape[1]} columns; "
            f"{len(features)} features; "
            f"{int(np.isnan(values).sum()):,} blank cells; "
            f"{int((finite == 0).sum()):,} zeros; "
            f"range {positive.min():,.4g} to {positive.max():,.4g} "
            f"(ratio {positive.max() / positive.min():,.4g})")


def main() -> None:
    frame, features = _source()
    rng = np.random.default_rng(SEED)

    # ── 1 · the QC injections and the acquisition metadata, removed ──────────
    #
    # The commonest way a metabolomics table arrives: an analyst exports "the
    # data" and means the biological samples and their measurements. The QC
    # injections were run and are not in the file, and the injection order went
    # with them. Both absences are unrecoverable from what is left, which is
    # exactly why §01 asks the tool to say so out loud.
    no_qc = frame[frame["sample_type"] == "participant"].copy()
    no_qc = no_qc.drop(columns=["run_order", "batch"]).reset_index(drop=True)
    _write(no_qc, "metabolomics_no_qc.csv", f"""\
# `metabolomics_no_qc.csv` — the export with the QCs and the run order removed

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_siblings.py`: the 8 pooled-QC rows dropped, and the
`run_order` and `batch` columns dropped. **No value was altered.**

{_stats(no_qc, features)}

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
""")

    # ── 2 · an analyst's working file: log2, and a paired design ─────────────
    #
    # TWO OPERATIONS, and they are not the same KIND of operation. The log2 is a
    # transform of the values and is fully derived. The subject roster is a
    # LABEL — see the companion's first paragraph — and it is honest only
    # because the file says so.
    logged = frame.copy()
    logged[features] = np.round(np.log2(frame[features].to_numpy(dtype=float)), 6)
    participants = logged.index[logged["sample_type"] == "participant"].tolist()
    subject = pd.Series(pd.NA, index=logged.index, dtype="object")
    timepoint = pd.Series(pd.NA, index=logged.index, dtype="object")
    for rank, position in enumerate(participants):
        subject.iloc[position] = f"SUBJ{rank // 2 + 1:03d}"
        timepoint.iloc[position] = "pre" if rank % 2 == 0 else "post"
    logged.insert(1, "subject_id", subject)
    logged.insert(2, "timepoint", timepoint)
    _write(logged, "metabolomics_paired_logged.csv", f"""\
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

{_stats(logged, features)}

---

## Must surface

1. **Already transformed** — `pack::metabolomics::already_transformed`, at
   `critical`. §01: a max below ~40 with a positive minimum and a low dynamic
   range. Here the whole block runs {logged[features].to_numpy(dtype=float)[np.isfinite(logged[features].to_numpy(dtype=float))].min():.2f} to {logged[features].to_numpy(dtype=float)[np.isfinite(logged[features].to_numpy(dtype=float))].max():.2f}. A second log
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
""")

    # ── 3 · the export that writes zeros for non-detections ──────────────────
    #
    # ONE operation: every blank becomes 0. §01 names four tools and what each
    # writes into a cell it could not quantify; MZmine, MaxQuant and Progenesis
    # all write 0, and this is what their output looks like.
    zeros = frame.copy()
    zeros[features] = frame[features].fillna(0.0)
    _write(zeros, "metabolomics_mzmine_zeros.csv", f"""\
# `metabolomics_mzmine_zeros.csv` — non-detections written as zeros

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_siblings.py`: every blank intensity replaced by `0.0`.
Nothing else changed. This is what the same run looks like out of MZmine,
MaxQuant or Progenesis — `METABOLOMICS_PACK.md` §01 names all three as writing
`0` where XCMS `fillPeaks` writes a small number instead.

{_stats(zeros, features)}

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
""")

    # ── 4 · the two polarities merged, badly ─────────────────────────────────
    #
    # FOUR operations, each a thing a real merge really does, and each named in
    # the companion. This is the widest sibling and it is the one most likely to
    # be read as manufactured, so every number below is stated on the file.
    merged = frame.copy()
    merged[features] = frame[features].fillna(0.0)
    half = len(features) // 2
    pos, neg = features[:half], features[half:]
    renamed = {c: f"{c}_pos" for c in pos}
    renamed.update({c: f"{c}_neg" for c in neg})
    merged = merged.rename(columns=renamed)
    pos_names = [renamed[c] for c in pos]
    neg_names = [renamed[c] for c in neg]

    # 4a · features carried into the merged list that were never detected in the
    #      polarity they were aligned into: all-zero columns.
    never_detected = neg_names[:8]
    merged[never_detected] = 0.0

    # 4b · XCMS `fillPeaks` writes a small number rather than a zero (§01), so a
    #      feature below the limit in every sample comes out CONSTANT and
    #      non-zero. The value is the block's 1st percentile, rounded.
    block = merged[pos_names + neg_names].to_numpy(dtype=float)
    filled_value = float(np.round(np.percentile(block[block > 0], 1), 2))
    gap_filled = pos_names[:5]
    merged[gap_filled] = filled_value

    # 4c · a failed injection the merger still emitted a row for: zero across
    #      both blocks, EXCEPT the gap-filled features, because a gap filler
    #      writes its constant into every row including that one. That is what
    #      makes the constant columns genuinely constant, and it is also the
    #      case that made `_empty_blocks` read the failed injection as fine
    #      until the detector was taught that a constant column is not evidence
    #      a sample was measured.
    failed_row = len(merged) - 1
    failed_sample = str(merged["sample_id"].iloc[failed_row])
    zeroed = [c for c in pos_names + neg_names if c not in gap_filled]
    merged.loc[merged.index[failed_row], zeroed] = 0.0

    # 4d · a sample re-injected after a failed acquisition, exported under the
    #      original id. Two rows, one id, two run orders.
    reinjected_row = 40
    reinjected_from = str(merged["sample_id"].iloc[reinjected_row - 1])
    merged.loc[merged.index[reinjected_row], "sample_id"] = reinjected_from

    # 4e · six features the merge emitted twice under one name. `to_csv` writes
    #      the repeated header and `read_csv` renames the second to `name.1`, so
    #      a duplicate feature id NEVER arrives as a duplicate — it arrives
    #      silently renamed, and every "how many features" count is six high.
    repeated = pos_names[10:16]
    merged = pd.concat([merged, merged[repeated]], axis=1)
    _write(merged, "metabolomics_merged_modes.csv", f"""\
# `metabolomics_merged_modes.csv` — the two polarities merged, badly

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_siblings.py`, through five operations. Each is a thing a real
two-polarity merge really does; none was chosen to make a detector fire, and
each is stated here so a later reader can check that.

| # | operation | consequence |
|---|---|---|
| 1 | every blank intensity replaced by `0.0` | the merge writes zeros for non-detections |
| 2 | features 1–{half} renamed `…_pos`, features {half + 1}–{len(features)} renamed `…_neg` | both acquisitions in one table |
| 3 | {len(never_detected)} `_neg` features set to `0.0` throughout | carried in from the other polarity's feature list, never detected |
| 4 | {len(gap_filled)} `_pos` features set to the constant `{filled_value}` | XCMS `fillPeaks` writes a small number, not a zero (§01) — the block's 1st percentile |
| 5 | row {failed_row} (`{failed_sample}`) zeroed across both blocks except the {len(gap_filled)} gap-filled features | a failed injection the merger still emitted a row for; the gap filler wrote its constant into that row too |
| 6 | row {reinjected_row}'s `sample_id` set to `{reinjected_from}` | a sample re-injected after a failed acquisition, exported under the original id |
| 7 | {len(repeated)} `_pos` columns written a second time under the same name | the merge emitted them twice |

{_stats(merged, pos_names + neg_names)}

**Operation 7 is the one worth reading twice.** `read_csv` renames a repeated
column label to `name.1`, so a duplicate feature id **never arrives as a
duplicate** — it arrives silently renamed, and every "we measured N metabolites"
count downstream is {len(repeated)} too high. The detector reads both forms: a
literal repeat, which only a constructed frame carries, and the renaming
signature, which is what any real CSV carries.

---

## Must surface

1. **Both ion modes in one table** — `pack::metabolomics::ion_modes`, at `info`,
   marked **CONVENTION**. That the polarities are normalized separately and
   merged afterwards is what the field does, not a result anybody established,
   and the badge is the difference between saying so and asserting it.
2. **Duplicate features and a duplicate sample** —
   `pack::metabolomics::duplicate_ids`. {len(repeated)} renamed columns and one
   `sample_id` on two rows.
3. **Rows and columns carrying no information** —
   `pack::metabolomics::empty_blocks`. {len(never_detected)} all-zero features,
   {len(gap_filled)} constant features, 1 empty sample. All three in one finding
   because they are one condition, counted separately in the payload because
   they are repaired differently.
4. **Zeros, and nothing assumed** — `pack::metabolomics::zeros_or_missing`, as
   on `metabolomics_mzmine_zeros.csv`.

## Must NOT surface

- **No left-censoring finding.** Same as the zero-filled sibling: the blanks are
  gone, so the reading has nothing to rest on.
- **No claim about which merge strategy is right.** §01 says the strategy
  affects the results and does not say which to use, so neither does the app.
""")


if __name__ == "__main__":
    main()
