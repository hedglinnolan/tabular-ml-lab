# TurboTab Next — frontend

Vite + React 19 + TypeScript (strict). Read `docs/turbotab-next/BLUEPRINT.md` (§0, §7) and
`docs/turbotab/DESIGN_LANGUAGE.md` (§02–§06, §09) before changing anything visible.

## Layout

```
src/api/schema.ts     the JSON contract (hand-written until `npm run gen:api` replaces it)
src/api/client.ts     the ONLY module that calls fetch; one typed function per route; 409 -> RefusalError
src/api/queries.ts    TanStack Query hooks; every project key starts with the pid: [pid, ...]
src/api/events.ts     SSE -> cache (applyProjectEvent is pure; useProjectEvents opens the stream)
src/motion/           the motion primitives: prefs (durations, reduced motion), Arrive, StaleVeil,
                      NumberTween. Settle is a shared layoutId on record/blocks.tsx
src/components/record the Record: question blocks -> decision sentences, findings
src/components/pipeline  Rows / Columns / Results panel, 2-D virtualized table preview
src/screens/          Start, Project, Lab (/lab demonstrates every motion primitive)
src/mocks/            MSW handlers + an in-memory server with a stage graph (dev:mock only)
src/styles/tokens.css the palette and the three voices; base.css global rules
e2e/                  one Playwright journey per milestone, writes review screenshots
```

## Rules

- **Server state lives only in TanStack Query.** No polling: SSE events patch or invalidate.
  Small UI state stays in components. No global mutable singletons.
- **Network access only through `src/api/client.ts`** (eslint enforces `fetch`).
- **Contract names match the server exactly.** When the server's OpenAPI lands, run
  `npm run gen:api` and make `schema.ts` re-export the generated types.
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
E2E_BASE_URL=http://127.0.0.1:5173 E2E_SCREEN_PREFIX=real npm run test:e2e   # a running app
npm run build      # typecheck + bundle to dist/ (served by turbotab/server)
npm run gen:api    # openapi-typescript ../../turbotab/server/openapi.json -> src/api/generated.ts
```

First e2e run on a machine: `npx playwright install chromium`. Screenshots are written to
`docs/turbotab-next/m0/screens/<prefix>-*.png`; keep each under 300 KB.

## Tests (proportional — BLUEPRINT §8)

Vitest for real logic only: client refusal parsing, event -> cache patching, column filtering.
Do not add a test per visual tweak. Never run the legacy Python suites from here.
