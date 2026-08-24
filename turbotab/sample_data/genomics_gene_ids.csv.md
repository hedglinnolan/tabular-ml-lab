# `genomics_gene_ids.csv` — the four gene-ID diagnostics, in one table

**60 rows × 186 columns.** 181 feature columns
and the five metadata columns `sample_id`, `batch`, `sex`, `age`, `condition`.

**Derived from `genomics_expression.csv`** by
`make_gene_id_sibling.py` (seed 51). Every feature column holds an
**unchanged** column of that file; only the column NAMES differ. No count in this
table is new, and re-running the generator writes it byte for byte.

---

## What is derived, what is curated, and where the line is

| Part | Origin |
|---|---|
| the 181 count columns | `genomics_expression.csv`, unchanged |
| the 97 clean symbols | real HGNC gene symbols, hand-listed |
| the 14 corrupted identifiers | **computed** by applying Excel's conversion rule to real symbols |
| the 70 Ensembl accessions | `ENSG` + eleven seeded digits — structurally valid, **deliberately not real** |

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
`1899-12-30` epoch for the serial form. Fixed at year **2022**,
because the year a converted symbol lands in is the year the file was opened, and
fixing it is what makes this reproducible.

- **12 as date strings**: `MARCH1` → `1-Mar`, `MARCH2` → `2-Mar`, `MARCH3` → `3-Mar`, `MARCH5` → `5-Mar`, `MARCH6` → `6-Mar`, `MARCH8` → `8-Mar`, `SEPT2` → `2-Sep`, `SEPT7` → `7-Sep`, `SEPT9` → `9-Sep`, `SEPT11` → `11-Sep`, `SEP15` → `15-Sep`, `DEC1` → `1-Dec`
- **2 as bare serials**: `SEPT1` → `44805`, `MARCH7` → `44627`

Both forms occur: the string is what a `.csv` export carries and the serial is
what survives a round trip through a numeric cell format.

---

## Must surface

Under the **genomics** lens, all four of §01's gene-ID diagnostics:

1. **Excel corruption — 14 identifiers.** `critical`, and a
   **hard stop**: the pack says *"Never auto-repair — report and stop."* The
   finding proposes no action and pre-selects nothing. `MARCH1` and `MARCH2`
   both become `1-Mar` and `2-Mar` reversibly, but nothing in `44805`
   distinguishes the symbol `SEPT1` from a date somebody meant, and a repair
   that guessed would put a wrong gene in a results table.
2. **Version suffixes — 70 of 181 identifiers carry one.**
   A join against unversioned annotation fails silently and drops them.
3. **Duplicates after collapsing versions — 3 base accessions
   appear twice.** Two annotation releases merged: the same gene twice, at two
   versions, counted twice by anything that sums.
4. **Mixed vocabularies — HGNC symbols and Ensembl accessions in one
   identifier set.** They join to different annotation tables and cannot be
   reconciled without knowing which release produced each.

Plus `pack::genomics::counts_p_over_n`, because the values are still counts and
181 of them against 60 samples is still p ≫ n.

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
  `44805` as a serial at all. A table mixing Entrez
  IDs with symbols is the case where that reading must be withheld.
- **RefSeq (`NM_…`) accessions**, which also carry versions.
- **Genes in rows.** This app's tables are samples in rows, so the identifiers
  are column names. A file with an identifier COLUMN is the orientation §01
  handles before this diagnostic runs.
- **A file corrupted only into serials.** With no date strings and no symbol
  majority left, nothing in the numbers would say the column names were ever
  gene symbols.
