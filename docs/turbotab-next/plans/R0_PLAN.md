# R0: the claims registry's scaffold and its first 15 claims

Planned 2026-10-16 on turbotab-next @ a78e535a, from CLAIMS_REGISTRY.md §7 (ruled 2026-10-15).
Two waves, each kept under Nolan's 3M line (ORCHESTRATOR §2): R0a builds the registry with no
real claim in it; R0b registers the 15 S1 claims and splits the colliding constants. R0b is sized
from R0a's measured spend before it starts. At stage 1's rate (about 0.3M fresh tokens per agent
run), R0a is about 1.2M and R0b about 2.5M.

## R0a · the scaffold (one package)

| Owns | Tier | Agents |
|---|---|---|
| `turbotab/core/claims/**` (new), `turbotab/core/reference/claims.py` (new), `docs/turbotab-next/reference/CLAIMS_REFERENCE.md` (generated), `turbotab/core/tests/test_claims_*.py` (new) | Opus builder, Opus adversarial verifier | builder, verifier, up to 2 repair rounds |

It touches no engine module, so no fitted number and no sentence a user reads can change.

**What it builds** (CLAIMS_REGISTRY.md §1, §1.1, §2.1, §2.2, §2.4, §3, §5's history, §6):
- The schema: `Claim`, `Part`, `Regime`, `Param`, `Basis`, `Fires` and `Silent` with their declared truth, kinds and grades as §1 and §2.4 give them.
- A `Registry` object, with one module-level default. Claims modules under `core/claims/` are found by a package scan, so no list of modules exists for two later packages to share.
- `card(id, ctx=None)`, `methods(id, ctx=None)` and `value(id, param)`. Strict mode, on in tests, raises when `ctx` fails a regime predicate. Outside it, the named fallback claim renders. A proposed claim never renders.
- Predicates and enforcers as `"module:function"` strings, resolved only when checked. Importing `turbotab.core.claims` loads no other `turbotab.core` module.
- Status derived, as §6 gives it: proposed, built, or retracted (the only declared one).
- Meta-tests 1–6 (§6) as functions over a registry. One test file runs them on the default registry, and each is also run on a planted registry that breaks it and on one that keeps it.
- The digit gate (§3), static and typed, over every claim template outside its placeholders, every `Relation.says` and every teaching string. It starts from an allowlist of today's digit-bearing strings, each tagged, and refuses a new one.
- The generator, `python -m turbotab.core.reference.claims [--check]`, which writes CLAIMS_REFERENCE.md and appends a claim's old version to `core/claims/history.jsonl` when its hash changes. "Ruled, not built" lists the proposed claims.

**Acceptance (test first, both ways):**
1. Each meta-test fails on a planted claim that breaks it and passes on a conforming one. Meta-test 2's planted test imports its claim's constant.
2. Strict mode raises outside the regime; non-strict renders the fallback; a proposed claim refuses to render.
3. A claim whose `enforced_by` does not resolve, or whose test does not exist, is proposed; resolving both makes it built.
4. A planted string with a new digit fails the gate, a placeholder passes, and a digit inside a citation or a tagged name passes.
5. `--check` exits 1 on a stale reference and 0 on a current one; a changed hash appends the old version to a temporary history file.
6. A subprocess import of `turbotab.core.claims` loads no other `turbotab.core` module.
7. The fast core tier passes, and `git diff` touches only the files R0a owns.

**Stop when** the seven hold and the verifier lists no problem; at most 2 repair rounds, then report.

## R0b · the 15 S1 claims and the splits (sized after R0a)

- The 15 S1 claims of notes/claims-backtest/FIXES.md, re-located in today's code (stage 1 fixed them), each with its parts, regime predicates, `Fires` and `Silent` tests (stage 1's known-truth tests, plus a `Silent` test wherever one is missing) and `enforced_by`. D-086 is a design-doc claim and stays proposed.
- The splits: `trials.few_clusters` (10, Li & Redden 2015) and `inference.few_clusters` (30, Cameron & Miller); `tuning.small_sample_standard` (300) and the boosted-trees and forest rule now named `SMALL_N` (500). Each constant takes its value from its claim, and each sentence that states it is filled from the claim.
- A-001's "300" in the family reason strings is generated from its claim. The TuningDecl fields stay, and a test pins every tuning plan's seed before and after.
- Grades: a builder grades only judgment or unsupported. Its verifier, who did not write the claim, proposes each higher part's grade with its evidence, and the repair round records it with `graded_by`.
- Probably two packages split by area, so no two own one engine file: inference, trials, causal, survival and designs in one; tuning, baseline, materiality, data recognizers, omics and usual intake in the other. Then one integrator, who regenerates METHODS_REFERENCE and the packets.

## Rulings

None needed from Nolan: the registry's three questions were ruled on 2026-10-15, and the rest is methods and architecture, which he delegated.
