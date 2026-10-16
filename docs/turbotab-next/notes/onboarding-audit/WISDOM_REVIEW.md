# Onboarding guidance: a review for wisdom

A third, independent pass (Fable 5.1, 2026-10-15) over the onboarding guidance after `AUDIT.md`
and `REVISION2.md`. Nolan asked for a reviewer "that can embody wisdom": not a third checklist, but
the judgment of the lead whose habits newcomers absorb. The question is whether an agent that
follows these files behaves well in the cases they never name.

## 1 · Verdict

The guidance is sound and mostly wise, and it is already proportionate: 72 lines in every agent's
context, and nothing in them that a role file could carry instead. Its rules carry their reasons,
so they generalize; its hard rules are literal where the stakes are Nolan's; and it already makes
stopping a normal act. What it lacked was small and in one place: the stance toward one's own
report. An agent was told to doubt the task, the spec and the code, and to list three doubts, but
not that a report naming what it did not do is a good report. Six sentences fix that and three
places where a defect could close silently. The root file does not grow.

## 2 · What is wise, and should be protected from later improvement

- **"It holds only the rules any agent can break."** The file's own charter, in its third sentence.
  Every future addition should answer it.
- **A missing field is returned, not invented.** This turns the commonest agent failure (a guessed
  branch, a guessed scope) into a cheap round trip. Keep "return at once" literal.
- **The premise is a hypothesis; the code is the truth.** One principle that covers INBOX lines,
  fix lists, stale "what the code does today" sections, stamps and the orchestrator's own rulings.
- **The trust bar as four imperatives,** each the negation of a backtest failure. Known truth,
  guards both ways, a number right or refused, sentences from the value the code uses. These stay
  literal: the stakes are a researcher's.
- **The Never list carries reasons** where a reason generalizes ("a `v*` tag publishes a release",
  "heavy runs are scheduled with the owner, so report the need") and stays bare where judgment
  failed (`pkill`, `--no-verify`). That is the right asymmetry.
- **No "approve, but"; a note triggers nothing.** The verifier contract was learned at cost (E1p)
  and is now mechanical. Do not soften it for a friendlier tone.
- **The stopping rule is set before round one, and the fallback is the honest refusal.** B-281's
  six rounds are the reason, and they are written beside the rule.
- **The role files begin with what the prompt must name.** The contract between orchestrator and
  agent is visible on both sides (`PREAMBLE_TEMPLATE.md` mirrors it).
- **ORCHESTRATOR.md §3 separates what Nolan rules from what the orchestrator owns,** with his words
  and dates, and says never to hand a methods question back. This is the project's spine.
- **§0's first reply:** what is true now, what you will do without asking, what you need from him.
  Three clauses that respect his time.

## 3 · What is not, and the change made

Each change names the behavior it buys. Nothing else was touched.

| Where | Was | Now | Behavior |
|---|---|---|---|
| `CLAUDE.md` First 3 | "Check it in the code first, and report where it was wrong" | "...the fix follows the code, not the diagnosis; a task whose premise fails may not need doing, so say so before you build" | The builder probe would have built on a diagnosis the code contradicted and noted it at the end. An agent now returns before building when the premise fails outright. |
| `CLAUDE.md` Report | "...with your three most doubtful claims" | "...and what you did not run or could not check. A short true report beats a green one; a result you did not read is not reported" | The honesty principle behind three role rules (the fabricated CI result, the deselected test read as green, the stamp trusted) stated once for every role. Reports name their gaps. |
| `CLAUDE.md` Environment | two bullets, four lines each | three lines each; the constraints path and the four document names dropped | Pays for the two lines above. `scripts/py`'s own comment names the pins; the prompt's READ names the documents. |
| `builder.md` step 2 | the premise check ends in the report | "If there is no defect, or the right fix needs a choice the prompt did not make (which rule to relax, what a sentence should say), return with what you found and the options rather than choose" | The builder probe listed four ways to relax a guard, each trading against a known hole, and would have picked "the narrowest that passes". A choice that moves a number or a sentence now goes back up. |
| `verifier.md` | the builder's report "is optional but worth reading" | "read the three doubts first: in the backtest, 8 of 9 such picks were defects" | The verifier's first effort lands where defects are. |
| `integrator.md` step 3 | confirm each problem is fixed | "One that is not is reported before any push, never merged around" | A known defect cannot reach `ci/` through an integrator being helpful. |
| `ORCHESTRATOR.md` line 6 | rulings "are not re-argued" | "...when evidence shows one wrong, he is shown the evidence once and re-rules (he has said his statements are guidance, not infallible orders, 2026-10-02)" | Resolves "not re-argued" against his own stance and the 2026-10-10 practice (the critique withdrew two rulings; he re-ruled). Neither obey a wrong ruling nor relitigate: show once. |
| `ORCHESTRATOR.md` §2 Autonomy | a short status per step | "...which also says plainly what went wrong on your watch ... and what changed because of it" | Incidents are told in the status, as the pkill and email incidents were in practice; the habit is now written. |
| `ORCHESTRATOR.md` §4.5 | the builder may "dispute with evidence" (builder.md) but no one rules | "A problem the builder disputes with evidence is ruled by you or a fresh verifier; a dispute closes nothing by itself" | A dispute cannot close a defect. This default is mine, from the fresh-review practice; flagged in §7. |
| `ORCHESTRATOR.md` §5 | memory and rules, no rule for growth | "An incident becomes a memory first. It becomes a rule in the root CLAUDE.md when it repeats or when the stakes are Nolan's..." | Protects the root file's proportion. It describes the present Never list exactly. |
| `HANDOFF.md` | | this review named; §6's questions under Open for Nolan | The orchestrator finds them where it looks. |

Not changed, and why: the Never list (literal, the stakes are his); the heavy-run definition (four
lines, each clause settled a probe's confusion); `turbotab/frontend/CLAUDE.md` (reaches only
frontend editors and is a map, not rules); every dated ruling.

## 4 · The tensions, and the tie-breakers actually chosen

- **Thoroughness against velocity.** Nolan, 2026-09-27: velocity over ceremony; 2026-10-09: "I do
  not want the quality of work to be sacrificed". The backtest's lesson: test different things, not
  more. Tie-breaker: thoroughness where a number, a guard or a leak is at stake (the trust bar);
  everywhere else, do the task, stop at the stopping rule, file the rest. Carried in CLAUDE.md
  (trust bar, First 4) and ORCHESTRATOR.md §2.
- **Autonomy against asking.** Nolan, 2026-09-27: launch the next step without waiting; the
  orchestrator is "the arbiter of taste". Boundaries: his pause points, heavy runs, his rulings
  (product, pedagogy, outward-facing, publishing), usage near its limit. For executors: never ask
  Nolan; a missing field or an open choice that moves a number returns to the orchestrator. Chosen;
  §2 Autonomy, CLAUDE.md First 2, builder.md step 2.
- **Rigor against walkability.** Nolan, 2026-10-15: the product walkable sooner; rigor serves the
  product and does not run ahead of it; pull, not push. Tie-breaker: rigor for what the walkthrough
  puts on screen; the rest waits for the friction log. Chosen; §2.
- **Completeness against calm.** Nolan, 2026-10-05: calm over complete; one obvious click. Chosen;
  §1 and FOUNDATION.
- **Following the plan against noticing it is wrong.** The orchestrator's choice, from the probes:
  the premise is a hypothesis and the code is the truth; a wrong diagnosis is reported and the fix
  follows the code; a premise that fails stops the task. For Nolan's own rulings: shown the evidence
  once, then he re-rules (his 2026-10-02 stance, the 2026-10-10 practice). Chosen, with the line-6
  amendment above; the amendment is confirmed with him in §6.
- **Unresolved, left to Nolan:** how much a wave may spend before he hears of it in advance (the
  orchestrator set "the backtest's scale" as the threshold); whether he wants cost and rework in each
  step's status.

## 5 · The debt behind the debt

In the order a lead would take them.

1. **Rule growth.** The root file will be the place every incident wants to land. The §5 line is
   the guard; the orchestrator should apply it at every wave end and prune as often as it adds.
2. **The orchestrator learns by incident, not by retrospective.** Each incident is now a rule with
   its story beside it, which is good. What is missing is the three-line retrospective per wave:
   what it cost (agent runs, rounds, tokens), what the verifier caught after build, and what the
   wave taught. Put it in the step's status to Nolan, so he can judge velocity against ceremony
   himself instead of discovering usage at 91%.
3. **HANDOFF's state is hand-written and drifted** (it said stage 1 had merged). §0 says check it
   against git; a ten-line script that prints the heads and ancestry of the branches HANDOFF names
   would make the check mechanical and the drift visible.
4. **Review loops are all-or-nothing and expensive.** The backtest showed that doubts triage well
   (8 of 9) and that one referee alone is 44–60% precise. The builder's doubts now lead the
   verifier; next, let a verifier's report say what it did not check, so a second verifier is spent
   only where coverage is thin.
5. **Plans that have landed still instruct** (`WAVE_C6A_PLAN.md:375` tells builders to pip
   install). A landed plan gets a one-line header: landed at `<sha>`; its rules are superseded by
   CLAUDE.md. Prompts' READ fields then cannot resurrect a stale rule.
6. **Documents that lie about paths** (the methods documents cite `tests/acceptance/...` relative
   to `turbotab/core/`) are warned about in every agent's context. Fix the four documents once and
   delete the warning.
7. **Model tier is enforced only by the call.** A Sonnet builder on a guard can do nothing but say
   so. The plan's tier column (§4.1) is the right place; the builder's report should state its
   model so the orchestrator sees a mismatch.
8. **About 110 stale worktrees** cost every agent that lists them. §5 covers the rule; the cleanup
   is overdue.

## 6 · Questions only Nolan can answer

1. When one of his rulings meets evidence that it is wrong, is the right habit to show him the
   evidence once and let him re-rule, as on 2026-10-10? The orchestrator's file now says so, from
   his words of 2026-10-02; he should confirm it.
2. Should each step's status carry its cost and rework in one line (agent runs, rounds, defects
   caught after build), so that he, not the usage meter, decides when ceremony is too much?
3. What may a wave spend before he is told in advance? The orchestrator set the backtest's scale
   (about 60 agent runs, 7–8M subagent tokens). Is that his number?

## 7 · Doubts

1. The honesty sentence in CLAUDE.md's Report may be tone that agents already carry, in which
   case it costs a line in every context and buys little; I judged the fabricated CI result and
   the deselected-test case to be evidence that it does not come for free.
2. "A task whose premise fails may not need doing, so say so before you build" could make an agent
   return on a diagnosis that is merely imprecise, the common case, instead of fixing the real
   defect. The same sentence says the fix follows the code; whether agents read both halves is
   untested.
3. The §4.5 dispute default and the §5 growth rule are inferred from practice (fresh review until
   approved; the present Never list), not ruled by Nolan or chosen by the orchestrator in words.
