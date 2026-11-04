/**
 * "Which of my decisions mattered?" in Estimate Results v0 (the front-end audit's G10; the owner's
 * Q10), against what the real server served (fixtures/mattered-estimate.json, written by
 * fixtures/capture_mattered.py; nothing in it by hand): the NHANES journey under Estimate before
 * the plan's lock and once Fit locked it (the specification curve, one specification not
 * estimated, and the materiality ledger with a row that moved more than predicted); journey B, a
 * time to an event, whose curve the server withholds; and the stage's refusal of an exposure
 * family.
 *
 * The truth is the served field: each printed number is read back off the screen and compared with
 * the field it shows, to the three significant digits the views print (views/format.ts), and each
 * sentence is the engine's, read from the capture. Where a test changes a served status (a stage
 * computing, a stale curve, a stage the ledger reads turning fresh), it changes only that field of
 * a served view.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Ledger, ProjectView } from "../../api/schema";
import { ENGINE_LABELS } from "../../components/fixtures/stages";
import type { ServedCurve } from "../../components/views/matteredAdapter";
import capture from "./fixtures/mattered-estimate.json";
import { ResultsTapestry } from "./index";
import { Mattered } from "./mattered";

// ── the capture ──────────────────────────────────────────────────────────────

type Json = Record<string, unknown>;
interface Served {
  curve: ServedCurve | null;
  says: string | null;
  methods: string | null;
  reported: { estimate: number } | null;
  refused: { message: string; exits: { label: string; stage?: string }[] } | null;
  withheld?: string | null;
}
interface StageResult {
  stage: string;
  key: string | null;
  fresh: boolean;
  status: string;
  artifact: Served | Json | null;
}
interface Snapshot {
  view: Json & { id: string; state: Json; decisions: Json[]; stages: Record<string, Json & { status: string; key: string | null }> };
  quest: Json & { fit: Json & { locked: boolean }; stages: { key: string; name: string }[] };
  stages: Record<string, StageResult>;
  materiality: Ledger;
  route?: { status: number; body: Json };
}
const C = capture as unknown as {
  nhanes: { before: Snapshot; locked: Snapshot & { fit_row: { estimate: number; ci_low: number; ci_high: number } } };
  cox: { locked: Snapshot };
  refused: Served;
};
const LOCKED = C.nhanes.locked;
const servedOf = (s: Snapshot) => s.stages.mattered!.artifact as Served;
const CURVE = servedOf(LOCKED).curve!;
const LEDGER = LOCKED.materiality;
const unticked = (s: string) => s.replaceAll("`", "");
/** The locked snapshot with `artifact` served as the mattered stage's. */
const curveServed = (artifact: Served): Snapshot => ({
  ...LOCKED,
  stages: { ...LOCKED.stages, mattered: { ...LOCKED.stages.mattered!, artifact } },
});

// ── reading numbers back off the screen (as estimate.test.tsx reads them) ────

function numbersIn(text: string): number[] {
  return [...text.matchAll(/[−-]?\d[\d,]*(?:\.\d+)?/g)].map((m) => Number(m[0].replace("−", "-").replace(/,/g, "")));
}

/** A served value printed to three significant digits (whole numbers from 1,000) reads back within
 *  half a unit of its last printed digit. */
function shows(printed: number | undefined, served: number | null | undefined): boolean {
  if (printed === undefined || served === null || served === undefined) return false;
  if (served === 0) return printed === 0;
  const a = Math.abs(served);
  const unit = a >= 1000 ? 1 : 10 ** (Math.floor(Math.log10(a)) - 2);
  return Math.abs(printed - served) <= unit / 2 + 1e-9 * a;
}

function expectShows(text: string | null | undefined, served: (number | null | undefined)[], where: string) {
  const got = numbersIn(text ?? "");
  const want = served.filter((v): v is number => v !== null && v !== undefined);
  expect(got.length, `${where}: "${text}"`).toBe(want.length);
  want.forEach((v, i) => expect(shows(got[i], v), `${where}: printed ${got[i]} for served ${v} in "${text}"`).toBe(true));
}

/** Every number the curve serves: each specification's estimate and interval. */
const curveNumbers = CURVE.specs.flatMap((s) => [s.estimate, s.low, s.high]);

// ── a server that answers as the real one did ────────────────────────────────

/** The requests made, by path. */
let asked: string[] = [];

function serve(s: Snapshot) {
  asked = [];
  const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const path = url.replace(/^.*\/api\/projects\/[^/?]+/, "").replace(/\?.*$/, "");
      asked.push(path);
      if (method !== "GET") return json(500, { detail: `nothing is recorded from the tapestry: ${method} ${path}` });
      if (path === "") return json(200, s.view);
      if (path === "/quest") return json(200, s.quest);
      if (path === "/materiality") return json(200, s.materiality);
      const stage = /^\/stages\/(\w+)$/.exec(path)?.[1];
      if (stage && s.stages[stage]) return json(200, s.stages[stage]);
      return json(404, { detail: `not in the capture: ${method} ${path}` });
    }),
  );
}

const count = (path: string) => asked.filter((p) => p === path).length;

function mount(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const out = render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
  return { ...out, rerender: (n: ReactNode) => out.rerender(<QueryClientProvider client={qc}>{n}</QueryClientProvider>) };
}

/** The Estimate view, drawn once what it reads is served. */
async function estimateView(s: Snapshot): Promise<HTMLElement> {
  serve(s);
  mount(<ResultsTapestry pid={s.view.id} />);
  return screen.findByTestId("results-estimate");
}

/** The section, once its curve and its ledger are read. */
async function matteredSection(s: Snapshot): Promise<HTMLElement> {
  const view = await estimateView(s);
  const section = await within(view).findByTestId("results-mattered");
  await waitFor(() => expect(count("/materiality")).toBeGreaterThan(0));
  return section;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// ── the capture itself ───────────────────────────────────────────────────────

describe("the capture", () => {
  it("is the NHANES journey under Estimate: the stage waits on the lock before Fit, the curve and the ledger after it", () => {
    expect(C.nhanes.before.quest.fit.locked).toBe(false);
    expect(C.nhanes.before.view.stages.mattered!.status).toBe("blocked");
    expect(C.nhanes.before.stages.mattered!.artifact).toBeNull();
    expect(C.nhanes.before.route!.status).toBe(409);
    expect(LOCKED.quest.fit.locked).toBe(true);
    expect(LOCKED.stages.mattered!.status).toBe("fresh");
    // The curve's primary is the fit's own row of `sugar`, as GET …/stages/fit serves it.
    const primary = CURVE.specs.find((s) => s.primary)!;
    expect(primary.estimate).toBeCloseTo(LOCKED.fit_row.estimate, 12);
    expect(primary.low).toBeCloseTo(LOCKED.fit_row.ci_low, 12);
    expect(primary.high).toBeCloseTo(LOCKED.fit_row.ci_high, 12);
    // The stage route serves what GET /mattered serves.
    expect((LOCKED.route!.body.curve as ServedCurve).specs).toEqual(CURVE.specs);
    expect(CURVE.not_estimated).toHaveLength(1);
    expect(LEDGER.locked).toBe(true);
    expect(LEDGER.rows.filter((r) => r.label)).toHaveLength(1);
    expect(LEDGER.rows.filter((r) => r.realized)).toHaveLength(1);
  });

  it("withholds journey B's curve in the words GET /mattered refuses it with, and refuses a family's", () => {
    const cox = servedOf(C.cox.locked);
    expect(cox.curve).toBeNull();
    expect(cox.withheld).toBe((C.cox.locked.route!.body.error as { message: string }).message);
    expect(C.refused.curve).toBeNull();
    expect(C.refused.refused!.exits).toHaveLength(1);
  });
});

// ── the lock's gate (FOUNDATION §5 rule 6) ───────────────────────────────────

describe("the lock's gate", () => {
  it("before the lock draws no part of it and reads neither the curve nor the ledger", async () => {
    const view = await estimateView(C.nhanes.before);
    expect(within(view).queryByTestId("results-mattered")).toBeNull();
    expect(count("/stages/mattered")).toBe(0);
    expect(count("/materiality")).toBe(0);
  });

  it("fails closed: with no lock in the quest log, the served curve and ledger are not drawn", async () => {
    for (const quest of [{ ...LOCKED.quest, fit: null }, C.nhanes.before.quest]) {
      const view = await estimateView({ ...LOCKED, quest: quest as Snapshot["quest"] });
      expect(within(view).queryByTestId("results-mattered")).toBeNull();
      expect(within(view).queryByTestId("results-ledger")).toBeNull();
      expect(numbersIn(view.textContent ?? "").some((n) => curveNumbers.some((v) => shows(n, v)))).toBe(false);
      cleanup();
    }
  });

  it("after the lock draws it under Table 2", async () => {
    const section = await matteredSection(LOCKED);
    const table2 = screen.getByTestId("results-table2");
    expect(table2.compareDocumentPosition(section) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(section).getByRole("heading", { level: 3 })).toHaveTextContent("Which of my decisions mattered?");
  });
});

// ── the specification curve ──────────────────────────────────────────────────

describe("the specification curve", () => {
  it("draws every served specification, the primary marked, and prints each one's served numbers in sorted order", async () => {
    const section = await matteredSection(LOCKED);
    expect(section.querySelectorAll("[data-spec]")).toHaveLength(CURVE.specs.length);
    const primary = CURVE.specs.find((s) => s.primary)!;
    expect(section.querySelector('[data-primary="true"]')).toHaveAttribute("data-spec", primary.key);
    const rows = [...within(section).getByTestId("table-alternative").querySelectorAll("tbody tr")];
    const sorted = [...CURVE.specs].sort((a, b) => a.estimate - b.estimate || (a.key < b.key ? -1 : 1));
    expect(rows).toHaveLength(sorted.length);
    sorted.forEach((sp, i) => {
      const cells = [...rows[i]!.children].map((c) => c.textContent ?? "");
      expect(cells[0]).toBe(`${i + 1}${sp.primary ? " (primary)" : ""}`);
      expectShows(cells[1], [sp.estimate, sp.low, sp.high], `rank ${i + 1}'s estimate`);
      expectShows(cells[2], [sp.n], `rank ${i + 1}'s rows`);
    });
    // The table is headed by the served level.
    expect(within(section).getByTestId("table-alternative").querySelector("thead")).toHaveTextContent(`Estimate (${CURVE.level! * 100}% interval)`);
  });

  it("names the specification not estimated by the choice it changed, with the engine's reason", async () => {
    const section = await matteredSection(LOCKED);
    const [held] = CURVE.not_estimated;
    const choice = CURVE.choices.find((c) => held!.picks[c.key] !== c.primary)!;
    const option = choice.options.find((o) => o.key === held!.picks[choice.key])!;
    const items = within(section).getAllByTestId("results-not-estimated");
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent(unticked(choice.label));
    expect(items[0]).toHaveTextContent(unticked(option.label));
    expect(items[0]).toHaveTextContent(unticked(held!.reason));
  });

  it("says the stage's methods sentence beneath it", async () => {
    const section = await matteredSection(LOCKED);
    expect(within(section).getByTestId("results-mattered-methods")).toHaveTextContent(unticked(servedOf(LOCKED).methods!));
  });

  it("says a withheld curve in the engine's words and draws no specification", async () => {
    const section = await matteredSection(C.cox.locked);
    expect(section).toHaveTextContent(unticked(servedOf(C.cox.locked).withheld!));
    expect(section.querySelector("svg")).toBeNull();
    expect(section.querySelector("[data-spec]")).toBeNull();
    expect(within(section).queryByTestId("results-mattered-methods")).toBeNull();
  });

  it("says a refused curve in the engine's words with its way forward named, never pressed, in the quest stage it opens", async () => {
    const section = await matteredSection(curveServed(C.refused));
    expect(section).toHaveTextContent(unticked(C.refused.refused!.message));
    const exit = C.refused.refused!.exits[0]!;
    const at = LOCKED.quest.stages.find((st) => st.key === exit.stage)!.name;
    expect(within(section).getByTestId("results-mattered-exit")).toHaveTextContent(`${exit.label}, in ${at}.`);
    expect(within(section).queryByRole("button")).toBeNull();
    expect(section.querySelector("svg")).toBeNull();
  });

  it("names a way forward by its label alone where it names no quest stage", async () => {
    // GET /mattered's way forward before the lock names a press, not a quest stage.
    const exit = (C.nhanes.before.route!.body.error as { exits: { label: string; stage?: string }[] }).exits[0]!;
    expect(exit.stage).toBeUndefined();
    const section = await matteredSection(curveServed({ ...C.refused, refused: { message: C.refused.refused!.message, exits: [exit] } }));
    expect(within(section).getByTestId("results-mattered-exit").textContent).toBe(exit.label);
  });

  it("while the stage computes says the engine's label for the work, and draws no curve", async () => {
    const running = { ...LOCKED.view.stages.mattered!, status: "running", fresh: false };
    const section = await matteredSection({
      ...LOCKED,
      view: { ...LOCKED.view, stages: { ...LOCKED.view.stages, mattered: running } },
      stages: { ...LOCKED.stages, mattered: { ...LOCKED.stages.mattered!, status: "running", fresh: false, key: null, artifact: null } },
    });
    expect(section).toHaveTextContent(`${ENGINE_LABELS.mattered}…`);
    expect(section.querySelector("svg")).toBeNull();
  });

  it("says nothing of a curve whose stage is blocked, and draws nothing with no ledger row either", async () => {
    /** `s` with its mattered stage blocked on an answer (as under Describe, which records no
     *  estimand) and nothing served for it. */
    const blocked = (s: Snapshot): Snapshot => ({
      ...s,
      view: { ...s.view, stages: { ...s.view.stages, mattered: { ...s.view.stages.mattered!, status: "blocked", fresh: false, key: null, missing: ["estimand"] } } },
      stages: { ...s.stages, mattered: { ...s.stages.mattered!, status: "blocked", fresh: false, key: null, artifact: null } },
    });
    expect(C.cox.locked.materiality.rows).toEqual([]);
    const view = await estimateView(blocked(C.cox.locked));
    await waitFor(() => expect(count("/materiality") && count("/stages/mattered")).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 50));
    expect(within(view).queryByTestId("results-mattered")).toBeNull();
    cleanup();
    // With ledger rows the section draws them, and says nothing of the curve.
    const section = await matteredSection(blocked(LOCKED));
    await within(section).findByTestId("results-ledger");
    expect(within(section).queryByRole("note")).toBeNull();
    expect(section.querySelector("[data-spec]")).toBeNull();
  });

  it("says a stopped curve as stopped, its curve veiled, and no failure where none is", async () => {
    const stopped = { ...LOCKED.view.stages.mattered!, status: "stale", fresh: false, cancelled: true };
    const section = await matteredSection({
      ...LOCKED,
      // The failure line reads the interview for the answer behind it (none is offered here).
      view: { ...LOCKED.view, interview: [], stages: { ...LOCKED.view.stages, mattered: stopped } },
      stages: { ...LOCKED.stages, mattered: { ...LOCKED.stages.mattered!, status: "stale", fresh: false } },
    });
    expect(within(section).getByTestId("failure-mattered")).toHaveTextContent(`You stopped ${ENGINE_LABELS.mattered!.toLowerCase()}`);
    await waitFor(() => expect(section.querySelector('[data-veil="stopped"] [inert] [data-spec]')).not.toBeNull());
    cleanup();
    expect(within(await matteredSection(LOCKED)).queryByTestId("failure-mattered")).toBeNull();
  });

  it("veils a curve of earlier answers while it is computed again, and only then", async () => {
    let section = await matteredSection(LOCKED);
    expect(section.querySelector('[data-veil="fresh"]')).not.toBeNull();
    expect(section.querySelector("[inert]")).toBeNull();
    cleanup();
    const stale = { ...LOCKED.view.stages.mattered!, status: "stale", fresh: false };
    section = await matteredSection({
      ...LOCKED,
      view: { ...LOCKED.view, stages: { ...LOCKED.view.stages, mattered: stale } },
      stages: { ...LOCKED.stages, mattered: { ...LOCKED.stages.mattered!, status: "stale", fresh: false } },
    });
    await waitFor(() => expect(section.querySelectorAll("[data-spec]")).toHaveLength(CURVE.specs.length));
    expect(section.querySelector('[data-veil="stale"]')).not.toBeNull();
    expect(section.querySelector("[inert] [data-spec]")).not.toBeNull();
  });

  /** `s` with its mattered stage at `status` and its artifact of earlier answers still served, as
   *  the stage route serves the engine's last artifact, not fresh, while it is computed again. */
  const computedAgain = (s: Snapshot, status: "stale" | "running" | "queued"): Snapshot => ({
    ...s,
    view: { ...s.view, stages: { ...s.view.stages, mattered: { ...s.view.stages.mattered!, status, fresh: false } } },
    stages: { ...s.stages, mattered: { ...s.stages.mattered!, status, fresh: false } },
  });
  /** The section's text outside every inert body: what it says as current. */
  const current = (section: HTMLElement) => {
    const copy = section.cloneNode(true) as HTMLElement;
    copy.querySelectorAll("[inert]").forEach((el) => el.remove());
    return copy.textContent ?? "";
  };
  const AGAIN = [
    ["stale", "stale"],
    ["running", "recomputing"],
    ["queued", "recomputing"],
  ] as const;

  it("veils a refusal of earlier answers and its ways forward, inert, while it is computed again, and only then", async () => {
    const message = unticked(C.refused.refused!.message);
    const exit = C.refused.refused!.exits[0]!.label;
    const fresh = await matteredSection(curveServed(C.refused));
    expect(fresh.querySelector('[data-veil="fresh"]')).not.toBeNull();
    expect(fresh.querySelector("[inert]")).toBeNull();
    expect(current(fresh)).toContain(message);
    expect(current(fresh)).toContain(exit);
    cleanup();
    for (const [status, veil] of AGAIN) {
      const section = await matteredSection(computedAgain(curveServed(C.refused), status));
      await waitFor(() => expect(section.querySelector(`[data-veil="${veil}"] [inert] [data-testid="results-mattered-exit"]`), status).not.toBeNull());
      expect(section.querySelector(`[data-veil="${veil}"] [inert]`), status).toHaveTextContent(message);
      expect(current(section), status).not.toContain(message);
      expect(current(section), status).not.toContain(exit);
      cleanup();
    }
  });

  it("veils a withheld curve of earlier answers, inert, while it is computed again, and only then", async () => {
    const withheld = unticked(servedOf(C.cox.locked).withheld!);
    const fresh = await matteredSection(C.cox.locked);
    expect(fresh.querySelector('[data-veil="fresh"]')).not.toBeNull();
    expect(fresh.querySelector("[inert]")).toBeNull();
    expect(current(fresh)).toContain(withheld);
    cleanup();
    for (const [status, veil] of AGAIN) {
      const section = await matteredSection(computedAgain(C.cox.locked, status));
      await waitFor(() => expect(section.querySelector(`[data-veil="${veil}"] [inert]`), status).toHaveTextContent(withheld));
      expect(current(section), status).not.toContain(withheld);
      cleanup();
    }
  });
});

// ── the materiality ledger ───────────────────────────────────────────────────

describe("the materiality ledger", () => {
  it("draws each served row, read-only, with its alternative, its line, what was done and its instruments' names", async () => {
    const section = await matteredSection(LOCKED);
    const ledger = await within(section).findByTestId("results-ledger");
    const rows = within(ledger).getAllByTestId("results-ledger-row");
    expect(rows.map((r) => r.getAttribute("data-thread"))).toEqual(LEDGER.rows.map((r) => r.thread));
    LEDGER.rows.forEach((row, i) => {
      const text = rows[i]!.textContent ?? "";
      expect(text.toLowerCase()).toContain(unticked(row.alternative).toLowerCase());
      expect(text).toContain(unticked(row.sentence));
      if (row.done) expect(text).toContain(unticked(row.done));
      expect(text).toContain(row.predicted.label);
      if (row.realized) expect(text).toContain(row.realized.label);
    });
    expect(within(ledger).queryByRole("button")).toBeNull();
  });

  it("says that a row moved more than predicted, its realized movement once, and no relabel where it did not", async () => {
    const section = await matteredSection(LOCKED);
    const ledger = await within(section).findByTestId("results-ledger");
    for (const row of LEDGER.rows) {
      const el = ledger.querySelector(`[data-thread="${row.thread}"]`)!;
      const relabel = el.querySelector('[data-testid="results-ledger-relabel"]');
      if (!row.label) {
        expect(relabel, row.thread).toBeNull();
        continue;
      }
      const words = unticked(row.realized!.words);
      const label = unticked(row.label);
      const at = label.indexOf(words);
      expect(at).toBeGreaterThan(0); // the relabel quotes the realized movement
      expect((el.textContent ?? "").split(words)).toHaveLength(2); // said once
      expect(relabel).toHaveTextContent(label.slice(0, at).trim());
      expect(relabel).toHaveTextContent(label.slice(at + words.length).trim());
    }
  });

  it("is read again when a stage its realized rows read changes, and not when another stage does", async () => {
    serve(LOCKED);
    const pid = LOCKED.view.id;
    const view = LOCKED.view as unknown as ProjectView;
    const { rerender } = mount(<Mattered pid={pid} view={view} />);
    await screen.findByTestId("results-ledger");
    expect(count("/materiality")).toBe(1);
    const at = (stage: string, change: Json): ProjectView =>
      ({ ...view, stages: { ...view.stages, [stage]: { ...view.stages[stage]!, ...change } } }) as ProjectView;
    // Another stage moves: the ledger is not read again.
    rerender(<Mattered pid={pid} view={at("explain", { status: "running" })} />);
    await new Promise((r) => setTimeout(r, 50));
    expect(count("/materiality")).toBe(1);
    // The sensitivity stage, whose refits the ledger quotes, is fresh for other answers.
    rerender(<Mattered pid={pid} view={at("sensitivity", { key: "another-key" })} />);
    await waitFor(() => expect(count("/materiality")).toBe(2));
  });
});
