"""L51 — `metabolomics_redundant.csv`, the ionization-product sibling.

`METABOLOMICS_PACK.md` §01, *"Redundancy detection — a real differentiator"*:

> Untargeted features are not independent. Ionization produces adducts
> (`[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`, `[M-H]-`, `[M+HCOO]-`),
> isotopologues (Δm/z 1.00336), dimers, and in-source fragments. […] If 5,000
> features collapse to ~1,200 clusters, the user's "5,000 metabolites" claim is
> wrong by ~4×.

`metabolomics_untargeted.csv` has **no redundancy at all** — its 392 features are
drawn independently, and the largest off-diagonal Pearson correlation in the
whole matrix is **0.87**, below any threshold the pack would use. It is a
perfectly good negative control and there was nothing for the diagnostic to find.

**Derived, and the derivation is the physics.** An adduct is not a new
measurement — it is the *same molecules* leaving the source as a different ion,
so its abundance is the parent's times an ionization-efficiency ratio, times
technical noise. That is exactly how every product column here is produced: from
a real parent column of `metabolomics_untargeted.csv`, by that multiplication.
Nothing is drawn to be correlated with something; the correlation is a
consequence of the columns being the same compound.

**No retention time, and that is the point rather than an omission.** The pack's
diagnostic clusters on RT **and** correlation. This app cannot read an RT —
`mz_0001` is an ordinal index, not a mass, and no RT exists anywhere in the
source fixture — so shipping one here would manufacture the thing whose absence
is the defect (`AGENT_ONBOARD.md` §07 trap #3) and would let a half-built
diagnostic look whole. The RT half is a failing test, not a fixture column.

Run: `venv/bin/python turbotab/sample_data/make_metabolomics_redundant.py`
Deterministic: seeded, and re-running writes byte-identical files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "metabolomics_untargeted.csv"
META = ("sample_id", "sample_type", "run_order", "batch", "age", "sex", "bmi")
OUTCOME = "responder"
SEED = 51

#: How many source features become parent compounds. The rest are dropped, so
#: the fixture stays the size of the one it is derived from.
N_PARENTS = 100

#: A parent is only chosen from features that are mostly OBSERVED, and that is
#: physical rather than convenient: an adduct of a compound you detect in half
#: your samples is a column that is missing in at least half your samples, and a
#: correlation nobody can compute is not evidence of anything.
MAX_PARENT_MISSING = 0.20

#: The ionization products, with the abundance ratio each carries relative to
#: the parent ion. `label` is recorded in this file and in the companion; it is
#: **not** written into the fixture, because an untargeted feature table does not
#: come annotated — that is the whole reason the diagnostic has to be
#: correlational.
PRODUCTS = (
    ("[M+Na]+", 0.05, 0.60),
    ("[M+K]+", 0.02, 0.25),
    ("[M+NH4]+", 0.03, 0.40),
    ("[2M+H]+ dimer", 0.01, 0.15),
    ("in-source fragment", 0.02, 0.30),
)

#: The M+1 isotopologue is not drawn from a range — it is arithmetic. Carbon-13
#: is 1.1% of natural carbon, so a molecule with `n` carbons shows an M+1 peak at
#: about `0.011 n` of the monoisotopic peak. The carbon count is what is drawn.
CARBON_RANGE = (10, 40)

#: Technical noise on the log scale. Small on purpose: two ions of one molecule
#: track each other closely, and the fixture's job is to carry redundancy that a
#: correlational reading can see, not to test the threshold's edge.
LOG_NOISE_SD = 0.06


def main() -> None:
    frame = pd.read_csv(SOURCE)
    features = [c for c in frame.columns if c.startswith("mz_")]
    rng = np.random.default_rng(SEED)

    missing = frame[features].isna().mean()
    eligible = [c for c in features if missing[c] <= MAX_PARENT_MISSING]
    assert len(eligible) >= N_PARENTS, (
        f"only {len(eligible)} features are observed often enough to parent an "
        f"adduct series")
    parents = [eligible[i] for i in
               rng.choice(len(eligible), size=N_PARENTS, replace=False)]

    #: The detection floor, read off the source rather than chosen: the smallest
    #: intensity the instrument recorded anywhere in it. A product that lands
    #: below it is a non-detection, which is what left-censoring IS.
    floor = float(np.nanmin(frame[features].to_numpy(dtype=float)))

    columns, truth = [], []
    for parent in parents:
        base = frame[parent].to_numpy(dtype=float)
        series = [("[M+H]+ parent", base)]
        n_products = int(rng.integers(2, 5))               # 2, 3 or 4
        picks = rng.choice(len(PRODUCTS) + 1, size=n_products, replace=False)
        for pick in picks:
            if pick == len(PRODUCTS):
                carbons = int(rng.integers(*CARBON_RANGE))
                label, ratio = f"M+1 isotopologue (C{carbons})", 0.011 * carbons
            else:
                label, low, high = PRODUCTS[pick]
                ratio = float(rng.uniform(low, high))
            noise = rng.normal(0.0, LOG_NOISE_SD, size=len(base))
            series.append((label, base * ratio * np.exp(noise)))
        cluster = []
        for label, values in series:
            values = np.where(values < floor, np.nan, values)
            cluster.append(len(columns))
            columns.append((label, parent, np.round(values, 2)))
        truth.append(cluster)

    order = rng.permutation(len(columns))
    placed = {original: position for position, original in enumerate(order)}

    block = pd.DataFrame(
        {f"mz_{placed[i] + 1:04d}": columns[i][2] for i in range(len(columns))})
    block = block[[f"mz_{i + 1:04d}" for i in range(len(columns))]]
    out = pd.concat([frame[list(META)], block, frame[[OUTCOME]]], axis=1)
    out.to_csv(HERE / "metabolomics_redundant.csv", index=False)

    sizes = [len(c) for c in truth]
    ratio = len(columns) / N_PARENTS
    roster = "\n".join(
        f"| `mz_{placed[i] + 1:04d}` | {columns[i][0]} | "
        f"`mz_{placed[cluster[0]] + 1:04d}` |"
        for cluster in truth[:3] for i in cluster)

    (HERE / "metabolomics_redundant.csv.md").write_text(f"""\
# `metabolomics_redundant.csv` — {N_PARENTS} compounds wearing {len(columns)} feature names

**{len(out)} rows × {len(out.columns)} columns.** {len(columns)} intensity
features `mz_0001`…`mz_{len(columns):04d}`, the seven metadata columns
`{"`, `".join(META)}`, and the binary outcome `{OUTCOME}`.

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_redundant.py` (seed {SEED}). {N_PARENTS} of its features are
kept as parent compounds; each is joined by {min(sizes) - 1}–{max(sizes) - 1}
ionization products computed from it. The rows, the run order, the batch labels,
the pooled QC injections and the outcome are unchanged.

---

## The number this fixture exists to produce

> **{len(columns)} features. {N_PARENTS} compounds. A table described as
> "{len(columns)} metabolites" overstates by {ratio:.1f}×.**

That is `METABOLOMICS_PACK.md` §01's own illustration — *"if 5,000 features
collapse to ~1,200 clusters, the user's '5,000 metabolites' claim is wrong by
~4×"* — built to the same ratio deliberately, and said here rather than left for
a reader to infer.

---

## How a product column is produced

An adduct is the same molecules leaving the source as a different ion, so:

    product = parent × ionization_ratio × exp(N(0, {LOG_NOISE_SD}))

then any value below the source file's own detection floor ({floor:g}, the
smallest intensity recorded anywhere in it) is set to blank, because a product
too faint to detect is a non-detection.

| Product | Abundance ratio to the parent ion |
|---|---|
{chr(10).join(f"| `{label}` | U({low:g}, {high:g}) |" for label, low, high in PRODUCTS)}
| M+1 isotopologue | **arithmetic**: 0.011 × carbons, carbons drawn from {CARBON_RANGE[0]}–{CARBON_RANGE[1]} |

The isotopologue ratio is not drawn from a range. Carbon-13 is 1.1% of natural
carbon, so an `n`-carbon molecule shows an M+1 peak at about `0.011 n` of the
monoisotopic one. The carbon count is what is drawn.

**Parents are chosen only from features missing in ≤{MAX_PARENT_MISSING:.0%} of
samples.** That is physical, not convenient: the adducts of a compound detected
in half your samples are missing in at least half your samples, and a correlation
nobody can compute is not evidence of anything.

**Column order is shuffled.** A parent and its products are scattered across the
feature block, so nothing can find them by assuming adjacency. The first three
clusters, for a reader who wants to check one:

| column | what it is | parent column |
|---|---|---|
{roster}

---

## What this fixture deliberately does NOT carry

**No retention time, and no m/z.** `mz_0001` is an ordinal index — that is true
of `metabolomics_untargeted.csv` too, and it is worth saying out loud because the
prefix reads like a mass. §01's diagnostic clusters on near-identical RT
(±0.05–0.1 min) **and** correlation (r > 0.9); only the second half is computable
against any table this repository has. Shipping a fabricated RT column would make
a half-built diagnostic look whole, which is the fixture manufacturing the thing
whose absence is the defect.

The consequence is stated in the finding and is not a caveat, and it is **not a
one-sided bound**: clustering on correlation alone merges *more* than clustering
on correlation **and** co-elution would, so supplying a retention time could only
split groups and would raise the effective count — while a column observed in too
few samples for any correlation to be computed is counted as independent here,
which raises it for the opposite reason. This fixture carries exactly one of the
second kind, which is why the app's answer is **105 rather than 104**: `mz_0015`
is a faint product observed in 5 of 80 samples.

---

## Must surface

Under the **metabolomics** lens:

1. **`pack::metabolomics::redundancy`** — measured on the whole numeric block
   (the {len(columns)} features plus `run_order`, `age`, `bmi` and `{OUTCOME}`,
   which sit in it as singletons): **408 columns, 105 groups, 3.9×**, with
   {N_PARENTS} multi-member groups of 3–5 — one per compound, exactly.
   `offered`, never applied: collapsing features changes what is analyzed.
2. **`pack::metabolomics::left_censored`** — inherited from the source, and
   reinforced, because the products are fainter than their parents.
3. **`pack::metabolomics::run_order`** — `run_order` is unchanged, and products
   inherit their parent's drift.
4. **`pack::metabolomics::pooled_qc`** — the eight QC injections are unchanged.

## Must NOT surface

- **No collapse applied.** The finding reports an effective count. Dropping
  {len(columns) - N_PARENTS} columns changes the analysis and is the user's call.
- **No claim that any two named features ARE one compound.** The reading is
  correlational and the finding says so; without RT it cannot exclude two
  co-varying compounds.

## Shapes not covered

- **A table that carries m/z and RT.** The whole reason the RT half is a failing
  test rather than a feature.
- **Negative-mode adducts (`[M-H]-`, `[M+HCOO]-`) as a separate ion mode.** The
  ratios above are positive-mode; §01's ion-mode split is a different diagnostic.
- **In-source fragments that are NOT proportional to the parent.** Every product
  here is proportional by construction, which is the easy case.
""", encoding="utf-8")
    print(f"wrote metabolomics_redundant.csv  ({len(out)}×{len(out.columns)}; "
          f"{N_PARENTS} compounds, {len(columns)} features, {ratio:.2f}x, "
          f"cluster sizes {min(sizes)}-{max(sizes)})")


if __name__ == "__main__":
    main()
