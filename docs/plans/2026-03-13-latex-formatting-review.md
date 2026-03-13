# LaTeX Formatting Review

## Goal
Review the LaTeX/PDF output specifically for document-formatting quality issues that are independent of content correctness.

## Focus areas
- tables cut off / overflow beyond margins
- long cell content causing layout breakage
- section spacing / page breaks
- wide tables needing resize/longtable/tabularx/landscape treatment
- figure/table placement quality
- readability of generated PDF as a manuscript draft

## Scope
### In scope
- `ml/latex_report.py`
- any LaTeX generation helpers directly affecting layout
- generated LaTeX structure as it would impact compiled PDF

### Out of scope
- scientific content quality except where it directly causes formatting/layout failures
- broader export/manuscript logic unrelated to layout

## Deliverable
- concrete formatting/layout issues observed or likely from current generation
- exact files/functions causing them
- recommended bounded fixes in priority order
- whether the formatting pass can be safely done in parallel with the procedural-correctness pass
