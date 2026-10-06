# The understanding layer

Nolan, 2026-10-05: the thread from exploration to every later decision is "the specific thing
about this app that has to be magic", and without it TurboTab "would just become auto-ml but with
more steps". This folder holds the design that answers him, and the evidence behind it.

## Read in this order

1. `UNDERSTANDING_LAYER.md`, the design. It covers:
   - the thread contract, parallel to the method contract (BLUEPRINT §13);
   - how threads extend the readings ledger (§14.1);
   - the lifecycle in the calm design;
   - the 17 families and their sentinels;
   - the coverage report;
   - the engine work packages;
   - the prototype plan.
2. `catalogs/<LENS>.md`, one readable index per lens, and `catalogs/<lens>.json`, the full thread declarations: detector, meaning, consumers by stage and purpose, the failure a score-driven pipeline commits, the moment, the layout, the sentence and the sources.
3. `research/WHAT_EXPLORATION_MAY_DECIDE.md`, the verified rule table: what a view may inform, by what it looks at and by purpose. Every claim was checked against primary sources by reviewers told to refute it.
4. `research/FIRST_LOOK_BRIEF.md`, the exploration stage. It has six fixed groups in every lens, "Worth a look" ranked by consequence and never by the outcome, and the walk on the first visit (Nolan's ruling).

## How it was made

1. Ten finders swept separate source families, producing 627 candidates:
   - the engine's intelligence;
   - the engine's method contracts;
   - reporting checklists;
   - bias and leakage taxonomies;
   - a data-structure grid;
   - five field methodologists;
   - the real fixtures.
2. A merger and gap hunter per lens consolidated them.
3. One independent critic per lens judged each thread: AutoML skeptic, evidence, and tier.
4. A final catalog per lens, then the synthesis.
5. A completeness critic found 34 more threads, including seven shared threads lost when one merged catalog was truncated.

There are 367 threads in total: shared 119, dietary 58, clinical 55, survey 47, genomics 45 and metabolomics 43. Threads the completeness critic added are marked † in the indexes.

## Known follow-ups

- **Coverage report (§4).** `UNDERSTANDING_LAYER.md` §4 was written before the completeness critic ran. Some TRIPOD+AI, PROBAST, leakage and ROBINS-E items it answers "by construction", or with weaker threads, now have their own threads: `shared-winner-optimism`, `shared-overfit-calibration-slope`, `shared-target-population-shift`, `shared-decisions-after-viewing`, `shared-score-fitted-on-these-rows`, `shared-rashomon-disagreement` and `shared-learned-interaction`. The coverage registry test, work package 1, settles the mapping from data.
- **Plain language.** The threads' moments and sentences are written in the methods register. On screen they follow the two-registers rule (`.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md` §2): the card speaks plain language, and the technical name stays as a quiet label. The manuscript keeps the methods register.
- **Owner questions.** `UNDERSTANDING_LAYER.md` §7 has three open questions, each with a recommendation. The orchestrator put them to Nolan on 2026-10-05.
