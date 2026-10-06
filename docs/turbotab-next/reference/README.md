# Reference — the binding material carried over from the legacy app

These documents were written for the legacy TurboTab app and still bind TurboTab v2. They were
curated here when that app was retired (BLUEPRINT §9.1). The contract is `../BLUEPRINT.md`, and
`../MODELING_SEQUENCE.md` covers everything after the seal. Where either of them disagrees with a
document here, they win.

| Document | What binds | Cited by |
|---|---|---|
| `OPENING_SEQUENCE.md` | Everything before the seal: the order of the questions, their copy, and their firing rules. Nothing may be resequenced. | M2_CONTRACT; the Router (`turbotab/core/interview.py`) |
| `LOCKBOX_CONSTITUTION.md` | What the app may know, and when (§01–§08): the fixed pre-seal order, grain asked rather than inferred, the seal's three states, eligibility and trims, declaration and execution, missingness. Its appendix is the routing constitution: fact, choice, consequence. | `turbotab/core/seal.py`, `decisions.py`, `repairs.py`, `coach.py`; the seal components |
| `DESIGN_LANGUAGE.md` | §02 color tokens, §03 the three voices, §04 components, §05 motion, §09 question grammar, §11 the evidence badge | BLUEPRINT §7; `turbotab/frontend/CLAUDE.md` |
| `prototypes/design-language.html` | The design language, drawn. Open it in a browser. | `DESIGN_LANGUAGE.md` |
| `PRODUCT_VISION.md` | The product owner's rulings in §06b ("correct, surfaced, beautiful") and §06c (the substitution curve and its five marks). Its two-door sections (§04b, §05) describe the legacy app. | `turbotab/core/methods/substitution.py`; M1_CONTRACT; DRIVE_RUBRIC |
| `DOMAIN_PACKS.md` | What a domain pack may change and what it may never change, its three guards, and the lens question (§01) | the lens-contradiction finding (`turbotab/core/detectors/lenses.py`) |
| `research/*_PACK.md` | The domain science behind every evidence badge: nutrition, clinical and survey, metabolomics, genomics, and interaction design | teaching content, detectors, the legacy packs |

**Evidence sources are written relative to this folder.** A badge cites
`research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature` or
`DOMAIN_PACKS.md#01 · The opening question`. `turbotab/core/tests/test_docs_paths.py` checks two
things: every cited file exists here with the cited section among its headings, and every docs path
the code reads exists. It checks that a source is named and can be found. It does not check that the
claim is faithful to the source; the claims ledger (`../audit/claims-ledger.md`) covers that.

**These documents also describe history.** They mention the Guided door, `turbotab/api.py`,
`web/index.html`, the loops, the findings ledger and the pre-commit gates. All of that was retired.
Documents they name that are not in this folder are in `docs/turbotab/archive/`, except
`DOMAIN_SCIENCE.md`, `data/` and `tools/`, which stay in `docs/turbotab/` because Classic's tests read
them. Git keeps the code.

Reference material that v2 produces itself goes in this folder too. `METHODS_REFERENCE.md` is
generated from the method contracts by `python -m turbotab.core.reference.methods` and is never
edited by hand; `turbotab/core/tests/test_reference.py` fails when it is stale. The expert review
packets built from the same contracts are in `../review-packets/`.
