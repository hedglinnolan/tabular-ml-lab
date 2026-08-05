# `metabolomics_redundant.csv` — 100 compounds wearing 404 feature names

**80 rows × 412 columns.** 404 intensity
features `mz_0001`…`mz_0404`, the seven metadata columns
`sample_id`, `sample_type`, `run_order`, `batch`, `age`, `sex`, `bmi`, and the binary outcome `responder`.

**Derived from `metabolomics_untargeted.csv`** by
`make_metabolomics_redundant.py` (seed 51). 100 of its features are
kept as parent compounds; each is joined by 2–4
ionization products computed from it. The rows, the run order, the batch labels,
the pooled QC injections and the outcome are unchanged.

---

## The number this fixture exists to produce

> **404 features. 100 compounds. A table described as
> "404 metabolites" overstates by 4.0×.**

That is `METABOLOMICS_PACK.md` §01's own illustration — *"if 5,000 features
collapse to ~1,200 clusters, the user's '5,000 metabolites' claim is wrong by
~4×"* — built to the same ratio deliberately, and said here rather than left for
a reader to infer.

---

## How a product column is produced

An adduct is the same molecules leaving the source as a different ion, so:

    product = parent × ionization_ratio × exp(N(0, 0.06))

then any value below the source file's own detection floor (50.95, the
smallest intensity recorded anywhere in it) is set to blank, because a product
too faint to detect is a non-detection.

| Product | Abundance ratio to the parent ion |
|---|---|
| `[M+Na]+` | U(0.05, 0.6) |
| `[M+K]+` | U(0.02, 0.25) |
| `[M+NH4]+` | U(0.03, 0.4) |
| `[2M+H]+ dimer` | U(0.01, 0.15) |
| `in-source fragment` | U(0.02, 0.3) |
| M+1 isotopologue | **arithmetic**: 0.011 × carbons, carbons drawn from 10–40 |

The isotopologue ratio is not drawn from a range. Carbon-13 is 1.1% of natural
carbon, so an `n`-carbon molecule shows an M+1 peak at about `0.011 n` of the
monoisotopic one. The carbon count is what is drawn.

**Parents are chosen only from features missing in ≤20% of
samples.** That is physical, not convenient: the adducts of a compound detected
in half your samples are missing in at least half your samples, and a correlation
nobody can compute is not evidence of anything.

**Column order is shuffled.** A parent and its products are scattered across the
feature block, so nothing can find them by assuming adjacency. The first three
clusters, for a reader who wants to check one:

| column | what it is | parent column |
|---|---|---|
| `mz_0168` | [M+H]+ parent | `mz_0168` |
| `mz_0125` | in-source fragment | `mz_0168` |
| `mz_0398` | [M+NH4]+ | `mz_0168` |
| `mz_0144` | [M+H]+ parent | `mz_0144` |
| `mz_0181` | [2M+H]+ dimer | `mz_0144` |
| `mz_0321` | in-source fragment | `mz_0144` |
| `mz_0330` | [M+H]+ parent | `mz_0330` |
| `mz_0256` | [M+K]+ | `mz_0330` |
| `mz_0148` | [M+NH4]+ | `mz_0330` |
| `mz_0350` | [M+Na]+ | `mz_0330` |

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
   (the 404 features plus `run_order`, `age`, `bmi` and `responder`,
   which sit in it as singletons): **408 columns, 105 groups, 3.9×**, with
   100 multi-member groups of 3–5 — one per compound, exactly.
   `offered`, never applied: collapsing features changes what is analyzed.
2. **`pack::metabolomics::left_censored`** — inherited from the source, and
   reinforced, because the products are fainter than their parents.
3. **`pack::metabolomics::run_order`** — `run_order` is unchanged, and products
   inherit their parent's drift.
4. **`pack::metabolomics::pooled_qc`** — the eight QC injections are unchanged.

## Must NOT surface

- **No collapse applied.** The finding reports an effective count. Dropping
  304 columns changes the analysis and is the user's call.
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
