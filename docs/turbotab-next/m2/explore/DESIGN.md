# M2 design prototype: the reshape storyboard and the seal moment

Route `/lab/m2` (scenes S1–S5, URL parameters pick a state). Every number comes from
`capture.py`, which writes `turbotab/frontend/src/explore/m2/fixture.json` from
`dietary_recalls.csv`, `clinical_longitudinal.csv` and the feature-major copy of
`metabolomics_untargeted.csv`, using the real ingest, `turbotab.repeats`, `turbotab.orientation`,
`engine.draw_holdout` and the production model families. Captures: `src/explore/m2/capture.mjs`
→ `screens/` (1440×900, light and dark, all under 250 KB), `frames/` (strips at 0, 150, 300, 450,
600 and 900 ms on a hand-stepped clock; `frames.json`), `words.json`.

## Rationale

Both moments follow §05.2: animate only correspondences that exist, and carry one or two
identities, not a table.

The reshape plays the method's real steps on a window of four participants: gather by
`participant_id`, combine, settle. Rows persist from file to settled table, so a pair visibly
meets and becomes one row. The "from rows" column (`0+1`) is the row map made visible. Values
never interpolate; they roll from one real state's value to the next. First and last do not
combine, so the leaving recall is struck instead. The row flow draws "300 folded in" differently
from "−300": same count, different claim. A strip of all 600 rows halves to 300, and a column strip
accounts for every column (changed, folded away, passing through). Window and strips read the same
at 17 and 20,002 columns; only the counts change.

The seal draws one cell per row moving to its side, so its basis is a picture. Grouped seals move
whole people. An abandoned grouping leaves amber holes joined to the visits that crossed. An
undetermined seal is dashed, carries a question mark, and reports what `participant_id` suggests
without claiming it. A chronological seal first lays subjects out by last visit.

Opening the seal is the one irreversible act, so it takes CONSEQUENCE's silhouette and grammar but
not `--stop`, which would claim the numbers cannot be trusted. After opening, a change is marked
post-seal and the first-opened scores stay beside the new ones.

The coach annotates the energy view on two lines with straight leaders, saying what the bars
cannot: where the under-reported days went, and how much variance was day-to-day.

## Visible words (light, 1440×900; serif = app, sans = action, mono = data)

| Screen | Total | App | Action | Data |
|---|---:|---:|---:|---:|
| reshape-mean | 376 | 132 | 102 | 142 |
| wide-20000-columns | 357 | 115 | 101 | 141 |
| orientation-turned | 314 | 68 | 96 | 150 |
| seal-grouped-recorded | 205 | 68 | 71 | 66 |
| seal-undetermined | 254 | 107 | 100 | 47 |
| results-sealed | 283 | 121 | 104 | 58 |
| results-post-seal | 316 | 148 | 85 | 83 |

All thirteen screens are in `words.json`. The reshape screens are the heaviest, and most of their
words are data (table cells).

## Honest weaknesses

- `dietary_recalls.csv` is already ordered by person, so its "gather" is mostly frames opening. Rows
  only travel in a stacked export (S2, `wide-stacked-gather-strip.png`). NHANES day-1 / day-2 files
  are stacked, which is the more common case.
- The table body keeps its tallest (gathered) height so nothing reflows during a flip. The settled
  table therefore leaves about 100 px of empty card.
- As stills, mid-segment frames are busy. At 450 ms two rows overlap showing the same mean, and in a
  stacked gather travelling rows cover the rows they pass.
- Value rolls and cell tints run on wall time while positions follow the player. When a step is
  slowed or paused they finish early. Production should drive them from the player.
- The 600-row strip is a texture (0.36 px per row), and its window ticks are tiny.
- The attest exit is a first-person button, not §09's typed sentence. The signal word "Opened once"
  is new. Both are readings of §09 for Nolan to rule on.
- Coach notes are placed from an estimate of their text width. A long note on a narrow card could
  overflow, so production needs measured layout.
- After the seal is recorded, the flip still says "With this choice (preview)". Production returns
  to live.
- In this fixture `hba1c` is noise by construction, so every family loses to the baseline. The
  held-out picture is honest but dull.
- These are my proposals, not contract numbers: CV-only ordered first below 100 rows, and "RMSE known
  to about ±1.96/√(2n)".
- The 20,000-column table is a seeded synthetic table. The post-seal change is a shortcut (the
  energy sentence's "change" switches residual to density).
- The check mark in the closed seal could read as "approved".

## Worth lifting into production

1. `reshape.ts` and `ReshapeTable`: row identity across states, the row-map column, value rolls,
   and record columns folding. A reshape mode for `TableFocus`, fed by a `TableFrame` that carries
   each row's unit and source rows.
2. Combined versus dropped in the row flow: `RowStep.combined: int`, drawn as a fold, not a hatch.
3. `ColumnStrip` and `ColumnRank`: the column account for wide data.
4. `seal.ts`, `SealFork` and `SealGlyph`: three bases, holes and straddle lines, the chronological
   layout by last visit, and the glyph in the banner and on the decision sentence.
5. `SealedResults`: the sealed, opened and post-seal states, the CONSEQUENCE card, and the
   first-opened ghost dot.
6. The coach band: notes on fixed lines with straight leaders to `range`, `points` or the spread.
7. `useMorph`: option-to-option morphs for views drawn outside React, without replaying the
   storyboard.
8. `TurnTable` and `SpreadStrips` for the orientation preview.

## Findings from the capture (for the backend)

- Under `change`, `repeats.aggregate` differences columns that are constant per person, so `age`
  and `bmi` become 0. The capture passes within-unit constants through, and `working` should do the
  same.
- Under `mean`, `recall_number` averages to 1.5 and `recall_date` keeps the first value. Record-level
  columns should fold away under mean and change, and the sentence should say so.
- All 12 recalls under 500 kcal are first recalls, so "first" keeps all 12 and "last" keeps none.
- Undetermined seal: 100 of the 110 `participant_id` values among the held-out recalls also train.
