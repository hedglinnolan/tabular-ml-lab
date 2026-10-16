# Critique of the claims registry, draft 1

An independent, read-only reviewer tested draft 1 of plans/CLAIMS_REGISTRY.md against the claims backtest on 2026-10-15, on feat/s1-int before fix/s1-inference merged. This is its report, condensed. Draft 2 was rewritten from it.

## Verdict

As written, draft 1 would have caught **none of the 15 S1 defects** with certainty and partly exposed 5. Its two novel gates were the digit gate and the mutation check. They target invented precision and self-pinning tests. Both problems exist in the code, but neither caused S1 harm. The S1 harms came from four other failures:
- guards that did not fire;
- rules stated for a wider regime than their evidence;
- a path that bypassed a guard;
- a grade nobody checked independently.

## Backtest fit

The S1 grades from the basis audit: unsupported ×3, judgment ×6, sourced ×3, proved ×2, certified ×1.

| id | failure | draft 1 | why |
|---|---|---|---|
| A-001 | ruled fix unbuilt; test pins the code to itself | partial | §6 would list the standard as "ruled, not built", but the old defaults claim stays live; the bit-for-bit pin passes the mutation check |
| A-029 | parameter on the wrong scale for rare-event losses | neither | 0.01 is already a parameter; every gate passes |
| A-102 | concern lines borrowed from the linear family | neither | nothing ties a claim's sentence to its `enforced_by` |
| A-124 | proxy "calibrated" on 9 in-sample cases | neither, worse | "calibrated with cases → replicated" would grade it replicated |
| B-211 | an exit path drops the design's refusals | neither | the tests cover the direct path only |
| B-281 | false mathematical reason for a refusal | neither | graded judgment; draft 1 required known-truth tests only above judgment |
| B-416 | Kutner misapplied to positivity | neither | the key resolves, so "sourced" passes |
| B-435 | Cox guard never fires | partial | judgment, so it was exempt from Fires tests |
| BX-040 | UL fact wrong for 5 nutrients | neither | teaching claims needed no tests |
| C-123 | blanks dropped before counting | partial | the predicate would be the same buggy `constant_values` |
| C-143 | missing 2021–23 phlebotomy weight | neither | domain knowledge |
| C-170 | contradicts BEATS_PREDICT M4 | partial | only if someone else grades it |
| C-193 | misses the repo's own "ask" fixture | neither | judgment; a Fires test already existed |
| D-058 | certified for linear, wrong for binary | neither | predicates were for proved claims only |
| D-086 | identity proved, threshold wrong | partial | one grade would badge it proved; "more than half" escapes the digit gate |

**S1: 0 prevented, 0 caught, 5 partial, 10 neither.**

In a sample of 10 S2s:
- D-021 (a SETTLED badge sourced to an internal pack section) is caught.
- D-085 (an FWL identity) is caught if it is graded proved.
- A-055 is prevented if both of its sites render one template.
- C-047 is partly exposed.
- Six are caught by nothing.

## Feasibility

- **Digit gate.**
  - Of 298 `Relation.says`, 30 carry non-citation digits, and only about 6 are real thresholds. The rest are names (CR2, SHA-256), guideline items, 0/1, 95%.
  - In teaching/content.py's 66 digit-bearing sentences, about a third are real parameters.
  - The gate needs a typed-literal grammar and `[0-9]`, not `str.isdigit()`, which is true for "²". It works statically on template literals outside placeholders; rendering voice sentences needs runtime context.
- **Static gates going blind.** Real for `cited_texts`, which walks live contract objects. If `says` became an id, it would scan the id. The cure is to fill `says` from the claim at registration. `test_plain_words` is an AST scan and stays sighted.
- **Mutation check.**
  - Constants bind at import, so each mutant needs a fresh process.
  - 129 test files import upper-case constants by name. Those tests move with the code and cannot fail, which is correct to flag.
  - It rewards boundary pins, which are self-pinning. It fails honest tests of conventions.
  - Halving or doubling can degenerate (CV 0.6 → 1.2).
  - Cost: tens of minutes for R0, hours for the whole layer. It belongs in CI only, narrowed, and later.
- **Refusing to render proposed claims.** Easy, but nearly vacuous, because unbuilt code never calls it. The real harm was a built claim staying live beside its ruled replacement (A-001).
- **Claim ids on DecisionRecord.**
  - Mechanically safe: `extra="forbid"` with a defaulted field still parses old logs.
  - But openapi, generated.ts and the mocks regenerate, and the defects fire in stage outputs, not records.
  - A claims-in-force manifest in provenance.json is the better layer. Old versions need a history.
- **Import order.**
  - `DECLARING_MODULES` covers the home modules of about 3 of the 15 S1 claims.
  - The codebase avoids top-level imports on purpose: contracts register lazily.
  - So the registry should be dependency-free modules by area.

## Changes it asked for, most important first

1. Fires and Silent known-truth tests for every rule that refuses, settles or sets a default, whatever its grade.
2. Runtime predicates for every grade above judgment, set to the regime the evidence covers.
3. Grades per part (value, direction, scope); show the weakest part the sentence asserts.
4. `graded_by` someone other than the author.
5. "Replicated" needs held-out cases from two or more fixtures.
6. A standalone registry: contracts are the pattern, not the home. Fill `says` at registration.
7. Derived status, `supersedes`, and a conflict meta-test.
8. Two registers per claim: card and methods.
9. A narrowed mutation check, moved to R2.
10. A static, typed digit gate.
11. Calm: "your call" for unsupported refusals and high-materiality defaults only.
12. Seeding from `user_sees` and the audit's corrected statements, not the inventory's paraphrases.
13. R0 after fix/s1-inference merges. Resolve the tuning contradiction (generate the reason text and pin the seeds). Defer the r ≥ 0.3 merge, the mutation check and record fields.

With changes 1–5, about 8 of the 15 S1s become caught or likely caught: A-001, A-124, B-281, B-416, B-435, C-170, D-058 and D-086.
