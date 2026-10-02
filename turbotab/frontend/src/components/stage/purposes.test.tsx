/**
 * The purpose gate (BLUEPRINT §11.2, M2_CONTRACT §11): a view kind, a composed picture, a coach
 * anchor kind, a stage element or a Record component that renders without a declared purpose
 * fails here. Two ways in: every kind the source can render (its `data-view` and `data-purpose`
 * literals, every exported Record component), and every kind actually rendered for one view of
 * each shape the server sends.
 */
import { render } from "@testing-library/react";
import type { ReactNode } from "react";
import type { ConsequenceView } from "../../api/m1-stage-types";
import type { CoachNote } from "../../api/m2-stage-types";
import { renderTrack } from "./PreviewGrid";
import {
  ANSWER_WORDS,
  COACH_ANCHOR_KINDS,
  COACH_ANCHOR_PURPOSES,
  isStructural,
  QUESTIONS,
  RECORD_PURPOSES,
  SERVER_VIEW_KINDS,
  STAGE_PURPOSES,
  VIEW_PURPOSES,
} from "./purposes";
import { storyboardOf, trackOf } from "./tracks";
import { createPlayerStore, PlayerContext } from "./usePlayer";

const SOURCES = import.meta.glob("../**/*.tsx", { query: "?raw", import: "default", eager: true }) as Record<
  string,
  string
>;

const literals = (attr: string) => {
  const found = new Map<string, string>();
  const re = new RegExp(`${attr}="([a-z_]+)"`, "g");
  for (const [path, text] of Object.entries(SOURCES)) {
    if (path.endsWith(".test.tsx")) continue;
    for (const m of text.matchAll(re)) found.set(m[1]!, path);
  }
  return found;
};

describe("every rendered kind declares the question it answers", () => {
  it("covers every view kind the server sends and every coach anchor kind", () => {
    for (const k of SERVER_VIEW_KINDS) expect(VIEW_PURPOSES[k], k).toBeDefined();
    for (const k of COACH_ANCHOR_KINDS) expect(COACH_ANCHOR_PURPOSES[k], k).toBeDefined();
  });

  it("covers every data-view and data-purpose the stage can render", () => {
    const views = literals("data-view");
    expect(views.size).toBeGreaterThanOrEqual(8); // a vacuous scan would pass anything
    for (const [kind, path] of views)
      expect((VIEW_PURPOSES as Record<string, unknown>)[kind], `data-view="${kind}" in ${path} has no purpose`).toBeDefined();
    const elements = literals("data-purpose");
    expect(elements.size).toBeGreaterThan(5);
    for (const [kind, path] of elements)
      expect(STAGE_PURPOSES[kind], `data-purpose="${kind}" in ${path} has no purpose`).toBeDefined();
    // No orphans: every registered stage kind is still rendered somewhere.
    for (const kind of Object.keys(VIEW_PURPOSES)) expect(views.has(kind), `${kind} is registered but never rendered`).toBe(true);
    for (const kind of Object.keys(STAGE_PURPOSES))
      expect(elements.has(kind), `${kind} is registered but never rendered`).toBe(true);
  });

  it("covers every component the Record exports", () => {
    const exported = new Map<string, string>();
    for (const [path, text] of Object.entries(SOURCES)) {
      if (!path.startsWith("../record/") || path.endsWith(".test.tsx")) continue;
      for (const m of text.matchAll(/export function ([A-Z][A-Za-z0-9]*)\s*[(<]/g)) exported.set(m[1]!, path);
    }
    expect(exported.size).toBeGreaterThan(10);
    for (const [name, path] of exported)
      expect(RECORD_PURPOSES[name], `<${name}> in ${path} has no purpose (src/components/stage/purposes.ts)`).toBeDefined();
    for (const name of Object.keys(RECORD_PURPOSES)) expect(exported.has(name), `${name} is registered but not exported`).toBe(true);
  });

  it("says each purpose in one short sentence, naming one of the five questions", () => {
    expect(Object.keys(QUESTIONS)).toHaveLength(5);
    const all = { ...VIEW_PURPOSES, ...COACH_ANCHOR_PURPOSES, ...STAGE_PURPOSES, ...RECORD_PURPOSES };
    for (const [kind, p] of Object.entries(all)) {
      const text = isStructural(p) ? p.structural : p.answer;
      expect(text.split(/\s+/).length, kind).toBeLessThanOrEqual(ANSWER_WORDS);
      if (!isStructural(p)) expect(QUESTIONS[p.question], kind).toBeDefined();
    }
  });
});

// ── one view of each shape, rendered ─────────────────────────────────────────

const note = (kind: CoachNote["anchor"]["kind"], ref: CoachNote["anchor"]["ref"]): CoachNote => ({
  text: `A note on \`${String(ref)}\``,
  anchor: { kind, ref },
});
const step = (key: string, n: number, dropped = 0) => ({ key, label: key, n, dropped, reason: null, decision_id: null });
const hist = { edges: [0, 1, 2], counts: [3, 4], n_missing: 0 };
const lineage = {
  nodes: [
    { id: "raw:a", column: "a", lane: "raw", role: "exposure", label: "a", formula: null, group: null, count: 1 },
    { id: "mx:a", column: "a", lane: "matrix", role: "exposure", label: "a", formula: null, group: null, count: 1 },
  ],
  links: [{ source: "raw:a", target: "mx:a", operation: "kept" }],
  collapsed: false,
};
const base = { title: "t", caption: "c", emphasis: [] as string[] };

const SAMPLES = [
  { ...base, kind: "row_flow", coach: [note("step", "x")], before: [step("loaded", 10)], after: [step("loaded", 10), step("x", 8, 2)], story: [], seal: null },
  { ...base, kind: "lineage", coach: [note("column", "a")], before: null, after: lineage, story: [] },
  {
    ...base,
    kind: "distribution",
    coach: [note("range", [0, 1])],
    column: "a",
    before: hist,
    after: { ...hist, counts: [2, 4] },
    before_label: "a",
    after_label: "a",
    marks: [],
    story: [],
  },
  {
    ...base,
    kind: "relationship",
    coach: [note("points", [0])],
    x_label: "kcal",
    y_label_before: "a",
    y_label_after: "a_adj",
    points_before: [[1, 2]],
    points_after: [[1, 0]],
    r_before: 0.5,
    r_after: 0,
    story: [],
  },
  {
    ...base,
    kind: "table_focus",
    coach: [note("column", "a")],
    columns_before: ["a"],
    columns_after: ["a"],
    rows: [{ row_id: 0, before: { a: null }, after: { a: 1 } }],
    changed: [[0, "a"]],
    n_affected_columns: 1,
    story: [],
  },
  // The reshape (many records to one row per unit, with the row map).
  {
    ...base,
    kind: "table_focus",
    coach: [],
    columns_before: ["id", "a"],
    columns_after: ["id", "a"],
    rows: [0, 1].map((r) => ({ row_id: r, before: { id: "u", a: r }, after: { id: "u", a: 0.5 } })),
    changed: [],
    n_affected_columns: 1,
    story: [
      { label: "records", columns: ["id", "a"], rows: [0, 1].map((r) => ({ row_id: r, unit: "u", sources: [r], values: { id: "u", a: r } })) },
      { label: "combined", columns: ["id", "a"], rows: [{ row_id: 0, unit: "u", sources: [0, 1], values: { id: "u", a: 0.5 } }] },
    ],
  },
  // The turn (the last frame is the first one transposed).
  {
    ...base,
    kind: "table_focus",
    coach: [],
    columns_before: ["feature", "s1"],
    columns_after: ["sample_id", "f1"],
    rows: [{ row_id: 0, before: { sample_id: "s1", f1: 3 }, after: { sample_id: "s1", f1: 3 } }],
    changed: [],
    n_affected_columns: 0,
    story: [
      { label: "supplied", columns: ["feature", "s1"], rows: [{ row_id: 0, unit: null, sources: [], values: { feature: "f1", s1: 3 } }] },
      { label: "turned", columns: ["sample_id", "f1"], rows: [{ row_id: 0, unit: null, sources: [], values: { sample_id: "s1", f1: 3 } }] },
    ],
  },
  // The seal (a row flow carrying its cells).
  {
    ...base,
    kind: "row_flow",
    coach: [],
    before: [step("loaded", 4)],
    after: [step("loaded", 4), step("train", 2, 2), step("holdout", 2)],
    story: [],
    seal: {
      state: "undetermined",
      label: "undetermined",
      exploratory: true,
      column: null,
      chronological: false,
      time_column: null,
      boundary: null,
      time_start: null,
      time_end: null,
      n_rows: 4,
      n_holdout: 2,
      n_units: null,
      n_holdout_units: null,
      straddle: null,
      evidence: null,
      hold: [0, 1, 0, 1],
      unit: null,
      unit_time: null,
    },
  },
] as unknown as ConsequenceView[];

class NoResize {
  observe() {}
  unobserve() {}
  disconnect() {}
}

describe("the rendered stage", () => {
  beforeAll(() => {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver ??= NoResize;
    HTMLCanvasElement.prototype.getContext = (() => null) as unknown as HTMLCanvasElement["getContext"];
  });

  it("draws only kinds with a declared purpose, for every shape of view", () => {
    const seen = new Set<string>();
    const store = createPlayerStore();
    const wrap = (node: ReactNode) => <PlayerContext.Provider value={store}>{node}</PlayerContext.Provider>;
    for (const view of SAMPLES) {
      const track = trackOf(view);
      const story = storyboardOf([track]);
      const { container, unmount } = render(wrap(renderTrack(track, story.last, false)));
      for (const el of container.querySelectorAll("[data-view]")) seen.add(el.getAttribute("data-view")!);
      for (const el of container.querySelectorAll("[data-purpose]")) {
        const p = el.getAttribute("data-purpose")!;
        expect(STAGE_PURPOSES[p], `rendered data-purpose="${p}"`).toBeDefined();
      }
      unmount();
    }
    for (const kind of seen) expect((VIEW_PURPOSES as Record<string, unknown>)[kind], `rendered data-view="${kind}"`).toBeDefined();
    // Each shape drew its own picture: the five views and the three composed ones.
    expect([...seen].sort()).toEqual(
      ["distribution", "lineage", "relationship", "reshape_table", "row_flow", "seal_fork", "table_focus", "turn_table"].sort(),
    );
  });
});
