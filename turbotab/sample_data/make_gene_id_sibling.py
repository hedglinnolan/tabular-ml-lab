"""L51 — `genomics_gene_ids.csv`, the identifier sibling of the counts fixture.

`GENOMICS_PACK.md` §01, *"Gene IDs"*, names four diagnostics — version suffixes,
duplicates after mapping, mixed vocabularies, and Excel corruption — and this
repository shipped **nothing any of them could fire on**.
`genomics_expression.csv` names its 495 columns `gene_0001`…, which belongs to no
vocabulary at all, and `genomics_microarray.csv` uses `1000000_at` probe IDs,
which is one clean vocabulary with no versions and no duplicates. Three of the
four diagnostics had no shape to read.

**Derived where it can be, curated where it cannot, and the line is stated.**

* **The VALUES are derived.** Every column is a column of
  `genomics_expression.csv`, unchanged. Nothing here is a new measurement.
* **The IDENTIFIERS are reference data.** Real HGNC symbols, and Ensembl
  accessions that are *structurally* valid and deliberately **do not resolve** —
  see below.
* **The CORRUPTION is derived**, and that is the part that matters.
  `1-Mar` is not typed into this file. It is produced by applying Excel's
  conversion rule to the real symbol `MARCH1`, and the serial `44621` is produced
  by taking the resulting date's offset from Excel's 1899-12-30 epoch. A fixture
  with `1-Mar` written into it would be `AGENT_ONBOARD.md` §07 trap #3 — the
  fixture manufacturing the thing whose absence is the defect — and it would pass
  while proving that a string equals itself.

**The Ensembl accessions name nothing on purpose.** They are `ENSG` plus eleven
digits drawn from a seeded counter, so they satisfy every structural test and
resolve to no gene in any release. A fixture carrying `ENSG00000141510` invites a
reader to check TP53 biology that these counts do not have; this one cannot be
misread that way, and the vocabulary detector reads structure rather than
membership so nothing is lost.

Run: `venv/bin/python turbotab/sample_data/make_gene_id_sibling.py`
Deterministic: seeded, and re-running writes byte-identical files.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "genomics_expression.csv"
META = ("sample_id", "batch", "sex", "age", "condition")
SEED = 51

#: Excel's day-zero. `1899-12-30` rather than `1900-01-00` because that is the
#: offset the serial numbers actually use, including the deliberate 1900 leap-year
#: bug — which is why `44621` is 2022-03-01 and not 2022-03-03.
EXCEL_EPOCH = _dt.date(1899, 12, 30)

#: The year an autoconverted symbol lands in is *the year the file was opened*,
#: which is what makes the serial unrecoverable: nothing in `44621` says whether
#: the source symbol was `MARCH1` or a date somebody meant. Fixed here so the
#: fixture is reproducible, and named so the number is checkable.
CORRUPTION_YEAR = 2022

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "MARCH": 3, "APR": 4, "MAY": 5,
           "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10,
           "NOV": 11, "DEC": 12}

#: Excel's rule, as a rule. A symbol whose leading letters spell a month and
#: whose trailing digits are a valid day is parsed as a date. This is why HGNC
#: renamed `SEPT*`→`SEPTIN*` and `MARCH*`→`MARCHF*`.
_EXCEL_RULE = re.compile(
    r"^(JAN|FEB|MARCH|MAR|APR|MAY|JUN|JUL|AUG|SEPT|SEP|OCT|NOV|DEC)(\d{1,2})$")

#: Real HGNC symbols that Excel destroys. Every one of these is a gene.
CORRUPTIBLE = ("MARCH1", "MARCH2", "MARCH3", "MARCH5", "MARCH6", "MARCH7",
               "MARCH8", "SEPT1", "SEPT2", "SEPT7", "SEPT9", "SEPT11",
               "SEP15", "DEC1")

#: The two that end up as bare serials rather than as date strings. Both forms
#: occur in the wild — the string is what a `.csv` export carries and the serial
#: is what survives a round trip through a numeric cell format.
AS_SERIAL = ("SEPT1", "MARCH7")

#: Real HGNC symbols Excel leaves alone. Reference data, and the only hand-listed
#: thing in this file.
CLEAN_SYMBOLS = (
    "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "NRAS", "HRAS", "MYC", "MYCN",
    "PTEN", "RB1", "VHL", "APC", "CDKN2A", "CDKN1A", "CDKN1B", "MDM2", "ATM",
    "ATR", "CHEK1", "CHEK2", "BRAF", "PIK3CA", "AKT1", "AKT2", "MTOR", "TSC1",
    "TSC2", "STK11", "SMAD4", "TGFB1", "TGFBR2", "NOTCH1", "NOTCH2", "JAG1",
    "DLL1", "WNT1", "CTNNB1", "AXIN1", "GSK3B", "FOXO1", "FOXO3", "FOXP3",
    "GATA1", "GATA3", "RUNX1", "SPI1", "CEBPA", "TAL1", "LMO2", "IKZF1",
    "PAX5", "EBF1", "IRF4", "BCL2", "BCL6", "BCL2L1", "BAX", "BAK1", "CASP3",
    "CASP8", "CASP9", "APAF1", "CYCS", "FAS", "FASLG", "TNF", "TNFRSF1A",
    "IL6", "IL10", "IL1B", "CXCL8", "CCL2", "STAT1", "STAT3", "STAT5A",
    "JAK1", "JAK2", "SOCS1", "SOCS3", "IFNG", "IL2RA", "CD4", "CD8A", "CD19",
    "CD34", "PTPRC", "ITGAM", "ITGB2", "SELL", "VCAM1", "ICAM1", "ACTB",
    "GAPDH", "TUBB", "RPL13A", "B2M")

#: How many Ensembl-vocabulary columns to write, and how many of them repeat a
#: base accession at a second version. Three pairs, because one pair reads as a
#: typo and three read as what it is: two annotation releases merged.
N_ENSEMBL = 70
N_DUPLICATE_PAIRS = 3


def excel_convert(symbol: str, year: int = CORRUPTION_YEAR):
    """`MARCH1` → `('1-Mar', 44621)`, or `None` where Excel leaves it alone."""
    match = _EXCEL_RULE.match(symbol.upper())
    if match is None:
        return None
    month, day = _MONTHS[match.group(1)], int(match.group(2))
    try:
        when = _dt.date(year, month, day)
    except ValueError:                                     # 31-Feb and friends
        return None
    return f"{day}-{when.strftime('%b')}", (when - EXCEL_EPOCH).days


def main() -> None:
    frame = pd.read_csv(SOURCE)
    genes = [c for c in frame.columns
             if c not in META and frame[c].dtype.kind in "iuf"]
    rng = np.random.default_rng(SEED)

    # ── the identifier list, built in the order the vocabularies are decided ──
    names, notes = [], []

    for symbol in CLEAN_SYMBOLS:
        names.append(symbol)
        notes.append(("hgnc_symbol", symbol))

    for symbol in CORRUPTIBLE:
        converted = excel_convert(symbol)
        assert converted is not None, f"{symbol} is in CORRUPTIBLE and survives"
        text, serial = converted
        if symbol in AS_SERIAL:
            names.append(str(serial))
            notes.append(("excel_serial", symbol))
        else:
            names.append(text)
            notes.append(("excel_date_string", symbol))

    # Ensembl, versioned. The accession body is a seeded counter, so it is
    # structurally an `ENSG` and is an accession of nothing.
    bodies = rng.choice(np.arange(1, 999_999), size=N_ENSEMBL, replace=False)
    ensembl = []
    for body in bodies:
        version = int(rng.integers(1, 18))
        ensembl.append((f"ENSG{int(body):011d}", version))
    # The duplicates: the LAST three bases repeat the FIRST three at a second
    # version, which is what merging two annotation releases on symbol produces.
    for i in range(N_DUPLICATE_PAIRS):
        base, version = ensembl[i]
        ensembl[-(i + 1)] = (base, version + 1)
    for base, version in ensembl:
        names.append(f"{base}.{version}")
        notes.append(("ensembl_versioned", base))

    assert len(names) == len(set(names)), "an identifier repeats verbatim"
    assert len(names) <= len(genes), (
        f"{len(names)} identifiers and only {len(genes)} count columns")

    # Deterministic interleave, so the two vocabularies are not contiguous and
    # nothing can pass by assuming a block layout.
    order = rng.permutation(len(names))
    names = [names[i] for i in order]

    block = frame[genes[:len(names)]].copy()
    block.columns = names
    out = pd.concat([frame[[c for c in META if c in frame.columns]], block],
                    axis=1)

    out.to_csv(HERE / "genomics_gene_ids.csv", index=False)

    n_date = sum(1 for kind, _ in notes if kind == "excel_date_string")
    n_serial = sum(1 for kind, _ in notes if kind == "excel_serial")
    n_symbol = sum(1 for kind, _ in notes if kind == "hgnc_symbol")
    pairs = ", ".join(
        f"`{sym}` → `{excel_convert(sym)[0]}`"
        for sym in CORRUPTIBLE if sym not in AS_SERIAL)
    serials = ", ".join(
        f"`{sym}` → `{excel_convert(sym)[1]}`" for sym in AS_SERIAL)

    (HERE / "genomics_gene_ids.csv.md").write_text(f"""\
# `genomics_gene_ids.csv` — the four gene-ID diagnostics, in one table

**{len(out)} rows × {len(out.columns)} columns.** {len(names)} feature columns
and the five metadata columns `{"`, `".join(META)}`.

**Derived from `genomics_expression.csv`** by
`make_gene_id_sibling.py` (seed {SEED}). Every feature column holds an
**unchanged** column of that file; only the column NAMES differ. No count in this
table is new, and re-running the generator writes it byte for byte.

---

## What is derived, what is curated, and where the line is

| Part | Origin |
|---|---|
| the {len(names)} count columns | `genomics_expression.csv`, unchanged |
| the {n_symbol} clean symbols | real HGNC gene symbols, hand-listed |
| the {n_date + n_serial} corrupted identifiers | **computed** by applying Excel's conversion rule to real symbols |
| the {N_ENSEMBL} Ensembl accessions | `ENSG` + eleven seeded digits — structurally valid, **deliberately not real** |

The Ensembl accessions resolve to nothing in any release. That is on purpose: an
accession that resolved would invite a reader to check biology these counts do
not have, and the vocabulary diagnostic reads structure rather than membership,
so nothing is lost by it.

---

## The corruption, and how it was produced

`GENOMICS_PACK.md` §01: *"Excel corruption is SETTLED and measured: Ziemann,
Eren & El-Osta (Genome Biology 17:177, 2016) found gene-name conversion errors in
~20% of papers with supplementary Excel gene lists; Abeysooriya et al. (PLoS
Comput Biol 2021) found the rate had risen. HGNC renamed `SEPT*`→`SEPTIN*` and
`MARCH*`→`MARCHF*` partly because of this."*

Not typed in. `excel_convert()` applies the rule — leading letters spelling a
month, trailing digits a valid day — and takes the result's offset from Excel's
`1899-12-30` epoch for the serial form. Fixed at year **{CORRUPTION_YEAR}**,
because the year a converted symbol lands in is the year the file was opened, and
fixing it is what makes this reproducible.

- **{n_date} as date strings**: {pairs}
- **{n_serial} as bare serials**: {serials}

Both forms occur: the string is what a `.csv` export carries and the serial is
what survives a round trip through a numeric cell format.

---

## Must surface

Under the **genomics** lens, all four of §01's gene-ID diagnostics:

1. **Excel corruption — {n_date + n_serial} identifiers.** `critical`, and a
   **hard stop**: the pack says *"Never auto-repair — report and stop."* The
   finding proposes no action and pre-selects nothing. `MARCH1` and `MARCH2`
   both become `1-Mar` and `2-Mar` reversibly, but nothing in `44805`
   distinguishes the symbol `SEPT1` from a date somebody meant, and a repair
   that guessed would put a wrong gene in a results table.
2. **Version suffixes — {N_ENSEMBL} of {len(names)} identifiers carry one.**
   A join against unversioned annotation fails silently and drops them.
3. **Duplicates after collapsing versions — {N_DUPLICATE_PAIRS} base accessions
   appear twice.** Two annotation releases merged: the same gene twice, at two
   versions, counted twice by anything that sums.
4. **Mixed vocabularies — HGNC symbols and Ensembl accessions in one
   identifier set.** They join to different annotation tables and cannot be
   reconciled without knowing which release produced each.

Plus `pack::genomics::counts_p_over_n`, because the values are still counts and
{len(names)} of them against {len(out)} samples is still p ≫ n.

---

## Must NOT surface

- **No repair, on any of the four.** Every one is `fix_kind="none"`. §01's
  sentence is the whole content of the first finding.
- **No claim that a symbol was recovered.** The finding names the *shape* of the
  damage and the count; it never says what `44805` used to be.

---

## Shapes this fixture does NOT cover

- **Entrez identifiers.** Deliberately absent, and the absence is load-bearing:
  a bare 5-digit integer is both an Excel serial and an Entrez gene ID, and the
  only thing separating them is the company it keeps. This table has no
  out-of-window integer identifiers, which is what licenses reading
  `{excel_convert(AS_SERIAL[0])[1]}` as a serial at all. A table mixing Entrez
  IDs with symbols is the case where that reading must be withheld.
- **RefSeq (`NM_…`) accessions**, which also carry versions.
- **Genes in rows.** This app's tables are samples in rows, so the identifiers
  are column names. A file with an identifier COLUMN is the orientation §01
  handles before this diagnostic runs.
- **A file corrupted only into serials.** With no date strings and no symbol
  majority left, nothing in the numbers would say the column names were ever
  gene symbols.
""", encoding="utf-8")
    print(f"wrote genomics_gene_ids.csv  "
          f"({len(out)}×{len(out.columns)}; {n_symbol} symbols, "
          f"{n_date} date strings, {n_serial} serials, "
          f"{N_ENSEMBL} versioned Ensembl, "
          f"{N_DUPLICATE_PAIRS} duplicate bases)")


if __name__ == "__main__":
    main()
