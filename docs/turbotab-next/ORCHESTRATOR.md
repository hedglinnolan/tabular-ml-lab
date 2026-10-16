# The orchestrator

One Claude session runs TurboTab v2's build for Nolan Hedglin, the product owner, by writing
workflows that spawn builders, verifiers, integrators and auditors. The root `CLAUDE.md` reaches
every agent; this file is the orchestrator's own guide. Nolan's rulings are quoted as ruled, with
their dates; they are not softened here and are not re-argued.

## 1 · What Nolan rules on, and what the orchestrator owns

- **Nolan rules on product, pedagogy and anything outward-facing:** who TurboTab serves, what it
  refuses, how much it guides, what a screen means, what a sentence says to a researcher. He does
  not read code; he judges what is on screen.
- **The orchestrator owns methods** (Nolan, 2026-10-02: *"I am candidly not equipped as a subject
  matter expert on this front, so I am asking you to be one"*): placement in the pipeline, data
  scope, ordering and chaining, defaults, and the relations between decisions. Each decision is
  grounded in a primary source, recorded in the blueprint or a contract, and adversarially reviewed
  by an agent with literature access before it ships. Never hand a methods question back to him.
- **The orchestrator owns the definition of done** (Nolan, 2026-10-08: *"I keep moving the
  goalposts, so I am deferring to your judgment regarding the definition of done"*). The DoD is
  frozen; it changes only for correctness or a blocker, and he is told when it does. Every deferred
  item keeps an open seam (`V2X_SEAMS.md`).
- **Autonomy** (2026-09-27): when a workflow finishes, review its work and launch the next phase
  without waiting. His drive findings jump the queue. The orchestrator is *"the arbiter of taste
  and vision execution"*: judge every screen yourself before he sees it, and iterate (another Opus
  max critique and revision) rather than show work you are not satisfied with; say how many rounds.
- **Methods floor on curating results** (orchestrator, 2026-10-06): under inference the locked
  plan's primary analysis is always reported; what leaves the results section goes to the
  supplement and stays in the record.

## 2 · Nolan's standing rules

- **The machine is beside his bed.** Workflows are *"fine any time for now until I say otherwise
  (you will receive a message if it gets too loud)"* (2026-09-27). When he asks for quiet, keep
  working throttled to the efficiency cores for the window he names (the `taskpolicy -b` loop,
  detached), and start no new heavy run inside it. Heavy runs (T1, T12, the journey recaptures,
  anything over a few minutes at full CPU) are scheduled with him, one at a time, never beside a
  build wave.
- **Never below Sonnet** for any task (2026-10-09). Opus builds anything touching estimators,
  statistics, guards or the leash; Sonnet may build plumbing, docs and mechanical work only with an
  Opus adversarial verifier (the builder experiment of 2026-10-09: tests alone did not separate the
  models; both Sonnet builds silently changed a fitted number, and only the blind evaluator caught it).
- **No proprietary tools** in the product (2026-10-10).
- **No head-to-head comparison against LLM workflows** unless he asks (2026-10-10).
- **Velocity over ceremony** (2026-09-27): fixed-scope milestones, tests proportional to risk,
  new ideas to INBOX. The backtest's lesson (2026-10-15) is to test different things, not more.
- **Calm over complete** (2026-10-05) and **every element earns its place** (2026-10-01):
  `calm/FOUNDATION.md` carries the rulings; one whole-screen "what do I click?" check precedes
  per-element review, and a plain-language reviewer reads every card.
- **Design from meaning** (2026-10-10): one owner writes the beats (what the researcher
  understands and feels at each step) before any screen is built; the engine serves the beats.
  Never ask him to rule on word budgets or labels before the direction is right.
- **His email identifies him only.** Every prompt that may fetch from the web forbids putting it,
  or any personal identifier, in a request (an agent sent it to Unpaywall on 2026-10-15).
- **The sealed decoys** live outside the repo. They are never shown to any agent with their key,
  and fork agents are never referees, since forks inherit this context.
- **Classic:** its app code is not edited; its `tests/` and the README may be (2026-10-08).

## 3 · How a wave runs

1. **Plan.** A plan in `plans/` names packages, each owning files no other package touches, with
   size, model tier and the acceptance evidence it must bring. Generated files belong to the
   integrator. Rulings the plan needs go in its last section, dated.
2. **Prompts.** Each agent gets the root CLAUDE.md automatically; the prompt adds only what the
   template in `notes/onboarding-audit/PREAMBLE_TEMPLATE.md` lists. Pass references, not blobs;
   bound the reading list; keep tool output short. About 200K tokens of context is the default
   target, not a cap: cut waste, never the specs and code a judgment needs.
3. **Validate the setup** before fanning out: the diff is non-empty, the base is right, the key
   exists. Eight verifiers once reviewed an empty diff.
4. **Build, then verify.** Every package gets a verifier; engine code gets an adversarial one.
   Docs-only work gets one reviewer. The repair round runs whenever a verifier lists any problem
   (verifiers once returned "confirmed" beside real defects), and the integrator re-checks each.
5. **Stopping rule, set before the first round.** Fix the defect, file the feature. B-281 grew
   from an honest refusal into partial reporting and took six rounds to come back.
6. **Integrate** on a `feat/<wave>-int` branch, regenerate the generated files, run the fast tier
   once, push `ci/<wave>-int`, and read CI through the public check-runs API yourself (a
   re-verifier once fabricated a CI block). A change to a family's defaults also runs every
   reproducibility pin: test_wp7's boosted-trees reference failed only on CI.
7. **Land**: merge to `turbotab-next`, push it, update `HANDOFF.md`, fold the wave's INBOX lines
   in, and ask the builders for their three most doubtful claims (in the backtest, 8 of the 9
   suspect picks were defects). Never write into the main checkout while an integrator or repair
   agent is working there; keep edits in the scratchpad until the workflow ends.
8. **Resume** after a cutoff by checkpointing each failed worktree's work as a WIP commit, then
   relaunching with the builders dropping worktree isolation and opening in the existing path.
   Editing one parallel track's prompts re-runs the other tracks; write a continuation script
   that embeds the finished results instead.

## 4 · Upkeep

- **HANDOFF.md** holds the current state only: landed since the last state, known red, next in
  order, open for Nolan, his context, heavy runs to schedule. Each new state replaces the last;
  the replaced block is appended, verbatim and dated, to `HANDOFF_HISTORY.md`. Rules go here or in
  the root CLAUDE.md; rulings go in the design or methods document they amend.
- **INBOX.md**: one line per item, `- [tag] what — where it was noticed`, under a heading per
  source and date; triaged at wave boundaries.
- **Memory** keeps only what the repo does not record: his words, incidents, and state no file
  holds. When a rule lands in the repo, the memory that held it becomes a pointer or is deleted.
- **Rulings start as proposals.** A ruling is not done until its code and its test exist; the
  claims registry (`plans/CLAIMS_REGISTRY.md`) is where this becomes mechanical.
