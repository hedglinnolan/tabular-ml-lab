# TurboTab Next — frontend

Vite + React 19 + TypeScript (strict). The root CLAUDE.md carries the environment, the test rules
and how to commit. Before changing anything visible, read `docs/turbotab-next/BLUEPRINT.md` §0 and
§7, and `docs/turbotab-next/calm/FOUNDATION.md`: §1 the five-second test, §2 the calm budget and
the two registers, §4 color by role, §5 the tapestry's grammar, §10 the display-order rule.
FOUNDATION amends `reference/DESIGN_LANGUAGE.md` §02–§04 and BLUEPRINT §11; where they differ,
FOUNDATION wins.

## Layout

```
src/api/generated.ts  the JSON contract, generated from turbotab/server/openapi.json (never edit)
src/api/schema.ts     thin named aliases over generated.ts, the enum lists, typed StageResult, SSE events
src/api/client.ts     the ONLY module that calls fetch; one typed function per route; 409 -> RefusalError;
                      401 (server mode, no session) -> an event main.tsx answers with the sign-in page
src/api/queries.ts    TanStack Query hooks; every project key starts with the pid: [pid, ...]
src/api/events.ts     SSE -> cache (applyProjectEvent is pure; useProjectEvents opens the stream)
src/motion/           the motion primitives: prefs (durations, reduced motion), Arrive, StaleVeil,
                      NumberTween. Settle is a shared layoutId on record/blocks.tsx
src/api/m1-types.ts, m1-stage-types.ts  the M1 aliases over generated.ts (Record, banner, stage)
src/api/m3-types.ts   the M3 aliases: the ask card, readings, methods, plan, files, codebooks, the
                      WP17 cards, the causal and time-varying lanes, the ten newer stages
src/state/focus.tsx   the one StageFocus the Record, the banner and the stage share (per project)
src/components/record the Record: the Router's questions -> server sentences, teaching, findings;
                      generic/ composes any open step without a bespoke card (no step is blank:
                      a new Router key needs a bespoke card or a composer, generic.test.tsx);
                      ask/AskCard.tsx is the ledger's one ask card on the open step (§14.2)
src/components/banner the pipeline banner: rows, columns, models, result (derive.ts is pure)
src/components/stage  the stage: previews with the transform player, evidence, live scenes,
                      the Results, save/export (journal-style SVG and PNG)
src/components/views  the view kinds (FOUNDATION §5 rule 9): one typed input per view, a table
                      alternative for each; /lab/views renders them all in both themes
src/quest/            the quest shell at /p/:pid (W1_PART2_PLAN §2 "W1-E"): QuestScreen (the header,
                      the quest line and progress bar, the manuscript rail, the card, the tapestry),
                      log.ts (pure: the open stage, segments, the one primary action, Confirm all's
                      lines), StageLine (the line, the bar, the hover panel), StageCard, Tapestry,
                      Fit; contracts.ts is the interface W1-F and W1-G implement
src/quest/cards/      CardFor, the dispatcher: every quest line's card by its source, never a blank
                      (renderers.ts): today's AskFor, W1-F's DECLARATION_CARDS (cards/predict),
                      W1-G's ResultsCards (src/quest/results), a finding's own card, reading rows,
                      else GenericAsk in the line's words. CardFor.test.tsx is the renderer
                      contract: every (source, key) in fixtures/quest-contract.json, which
                      turbotab/core/tests/test_quest_renderer_contract.py writes from the path fuzzer
src/kit/              the product's calm kit, copied from src/explore/calm-kit as the shell uses it:
                      tokens (scoped to [data-kit]: a part carries the attribute; today's cards keep
                      src/styles/tokens.css), the Source Sans 3 faces, kit.module.css, Why, Foot,
                      the manuscript rail and overlay (props-driven, over GET /methods/sections)
src/screens/          Start, Project (today's screen, at /p/:pid/classic until W2 closes), Lab (/lab
                      demonstrates every motion primitive; /lab/stage hosts the stage beside a
                      stand-in Record; /lab/m3 lists the M3 journeys). Every /lab route exists only
                      in dev:mock; a production build drops them
src/explore/          dev-only prototypes and the calm kit (calm-kit, calm-quest, the methods-* and
                      stage prototypes), built by their own vite configs (build:protos, build:calm)
src/mocks/            MSW handlers + an in-memory server with a stage graph (dev:mock only);
                      m3.ts replays the real server's captured journeys (fixtures/m3-*.json, from
                      docs/turbotab-next/m3/capture_fixtures.py) as projects m3~<journey>[~<n>];
                      quest.ts serves the quest log and its reads for every project, and Fit
src/styles/tokens.css the palette and the type tokens of today's screens; base.css global rules
e2e/                  one Playwright journey per milestone (mock or real server), review screenshots;
                      M1: m1-journey.spec.ts (the whole journey), m1-record, m1-stage;
                      M2: m2-journeys.spec.ts (every lens and NHANES, real server only), m2-record,
                      m2-stage (the mock lab at /lab/stage/m2);
                      M3: m3-no-dead-end.spec.ts (six journeys, every open question answerable,
                      real server only). Each opens its project in the shell, then drives today's
                      screen through classic.ts's toClassic
```

A project's two routes share one event stream (App.tsx `Project`): `/p/:pid` is the quest shell,
whose header links to `/p/:pid/classic`, today's screen.

## Rules

- **Server state lives only in TanStack Query.** No polling: SSE events patch or invalidate.
  Small UI state stays in components. No global mutable singletons.
- **Network access only through `src/api/client.ts`** (eslint enforces `fetch`).
- **The contract is generated.** After any server model change run, from the repo root,
  `scripts/py -m turbotab.server.openapi --write`, then
  `npm run gen:api`. `generated.test.ts` fails when generated.ts drifts from openapi.json; a Python
  test fails when openapi.json drifts from the app. Import types from `schema.ts`, never from
  generated.ts.
- **Presentation follows FOUNDATION:** color by role (§4), one type family (§2), plain words on the
  card with the technical name as a quiet label (§2), nothing quietly filled in (§10). Style through
  tokens (`src/styles/tokens.css`; the kit's `docs/turbotab-next/calm/tokens.css`, which
  `src/kit/tokens.css` holds value for value under `[data-kit]`); never hard-code a color. The
  M1–M3 components in `record`, `banner` and `stage` were built to DESIGN_LANGUAGE §02–§05 (three
  voices, chips) before the calm ruling of 2026-10-05, and keep that idiom inside the quest shell
  until W2; match them when fixing them, and build new screens to FOUNDATION with `src/kit`.
- **Motion is identity continuity only** (DESIGN_LANGUAGE §05.2): settle, arrive, propagate
  (StaleVeil), the working table under a reshape, and numbers tweening. Nothing else moves;
  disclosure is instant. Durations 150–300 ms, from `DUR` / `useTransitions()`. Reduced motion
  (system or the /lab toggle) makes all of it instant — use the primitives, not ad-hoc animation,
  so that holds.
- **Never assert falsely.** Every rendered sentence traces to a decision or a computed fact.
  Detection suggests beside an option; it never pre-selects. Stale content is veiled and
  `inert`, never deleted. No estimate and no score before the lock or Fit (FOUNDATION §5 rule 6).
- **Accessibility:** real roles and aria, visible focus, `inert` for veiled content, keyboard
  operable chips and pickers. Wide content scrolls in its own container.

## Commands

```
npm run dev        # Vite; proxies /api to the Python server on 127.0.0.1:8787
npm run dev:mock   # Vite + MSW mock API (no server needed)
npm run check      # tsc + eslint + vitest — keep it under a minute
npm run test:e2e   # Playwright against the mock dev server (port E2E_PORT, default 5391)
E2E_BASE_URL=http://127.0.0.1:8792 npm run test:e2e   # a running server (screens: real-*)
npm run build      # typecheck + bundle to dist/ (served by turbotab/server)
npm run gen:api    # openapi-typescript ../../turbotab/server/openapi.json -> src/api/generated.ts
```

Playwright's browser is installed on this machine; if it is missing, stop and report it rather
than installing it. Screenshots are written to
`docs/turbotab-next/m<N>/screens/<prefix>-*.png`; keep each under 300 KB, and restore any the mock
run rewrote before committing. The M1 journey against the real server: start it with
`TURBOTAB_WORKERS=2 TURBOTAB_HOME=$(mktemp -d)` on port 8812, then
`E2E_BASE_URL=http://127.0.0.1:8812 npx playwright test m1-journey` (screens: `real-*`). The M2
journeys the same way on port 8842: `E2E_BASE_URL=http://127.0.0.1:8842 npx playwright test
m2-journeys` (screens: `docs/turbotab-next/m2/screens/real-*`; timings in
`docs/turbotab-next/m2/journeys-real.json`). The heavy m3 journeys (metabolomics and survey, about
half an hour each) are not run locally.

Space belongs to the stage's flip, except in a multi-select list (the lens, the models), where it
chooses; Enter records.

## Tests (proportional — BLUEPRINT §8)

Vitest for real logic only: client refusal parsing, event -> cache patching (and late responses
merging newest-wins), column filtering, contract drift, a view's scale math and its empty state.
Do not add a test per visual tweak.
