# TurboTab Next — frontend

Vite + React 19 + TypeScript (strict). Read `docs/turbotab-next/BLUEPRINT.md` (§0, §7) and
`docs/turbotab/DESIGN_LANGUAGE.md` (§02–§06, §09) before changing anything visible.

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
src/screens/          Start, Project, Lab (/lab demonstrates every motion primitive; /lab/stage
                      hosts the stage beside a stand-in Record; /lab/m3 lists the M3 journeys).
                      Every /lab route exists only in dev:mock; a production build drops them
src/mocks/            MSW handlers + an in-memory server with a stage graph (dev:mock only);
                      m3.ts replays the real server's captured journeys (fixtures/m3-*.json, from
                      docs/turbotab-next/m3/capture_fixtures.py) as projects m3~<journey>[~<n>]
src/styles/tokens.css the palette and the three voices; base.css global rules
e2e/                  one Playwright journey per milestone (mock or real server), review screenshots;
                      M1: m1-journey.spec.ts (the whole journey), m1-record, m1-stage;
                      M2: m2-journeys.spec.ts (every lens and NHANES, real server only), m2-record,
                      m2-stage (the mock lab at /lab/stage/m2);
                      M3: m3-no-dead-end.spec.ts (six journeys, every open question answerable,
                      real server only)
```

## Rules

- **Server state lives only in TanStack Query.** No polling: SSE events patch or invalidate.
  Small UI state stays in components. No global mutable singletons.
- **Network access only through `src/api/client.ts`** (eslint enforces `fetch`).
- **The contract is generated.** After any server model change run
  `venv/bin/python -m turbotab.server.openapi --write` (repo root), then `npm run gen:api`.
  `generated.test.ts` fails when generated.ts drifts from openapi.json; a Python test fails
  when openapi.json drifts from the app. Import types from `schema.ts`, never from generated.ts.
- **Every hue is a claim** (§02): `--accent` now/primary, `--ok` recorded, `--warn` the coach and
  staleness, `--stop` only the blocker band. Style through tokens; never hard-code colors.
- **Three voices** (§03): the app speaks serif (`--serif`), the user acts sans (`--sans`), data is
  mono in a chip (`.v` / `<V>`; `<Prose>` turns backticks into chips).
- **Motion is identity continuity only** (§05.2): settle, arrive, propagate (StaleVeil), the
  working table under a reshape, and numbers tweening. Nothing else moves; disclosure is instant.
  Durations 150–300 ms, from `DUR` / `useTransitions()`. Reduced motion (system or the /lab toggle)
  makes all of it instant — use the primitives, not ad-hoc animation, so that holds.
- **Never assert falsely.** Every rendered sentence traces to a decision or a computed fact.
  Detection suggests beside an option; it never pre-selects. Stale content is veiled and
  `inert`, never deleted.
- **Accessibility:** real roles and aria, visible focus, `inert` for veiled content, keyboard
  operable chips and pickers. Wide content scrolls in its own container.
- **American spelling** in code, copy and docs (color, behavior, normalize, modeling, labeled,
  artifact, analyze). The repo's pre-commit gate checks tracked .md and .py files.

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

First e2e run on a machine: `npx playwright install chromium`. Screenshots are written to
`docs/turbotab-next/m<N>/screens/<prefix>-*.png`; keep each under 300 KB. The M1 journey against
the real server: start it with `TURBOTAB_WORKERS=2 TURBOTAB_HOME=$(mktemp -d)` on port 8812, then
`E2E_BASE_URL=http://127.0.0.1:8812 npx playwright test m1-journey` (screens: `real-*`). The M2
journeys the same way on port 8842: `E2E_BASE_URL=http://127.0.0.1:8842 npx playwright test
m2-journeys` (screens: `docs/turbotab-next/m2/screens/real-*`; timings in
`docs/turbotab-next/m2/journeys-real.json`).

Space belongs to the stage's flip, except in a multi-select list (the lens, the models), where it
chooses; Enter records.

## Tests (proportional — BLUEPRINT §8)

Vitest for real logic only: client refusal parsing, event -> cache patching (and late responses
merging newest-wins), column filtering, contract drift.
Do not add a test per visual tweak. Never run the legacy Python suites from here.
