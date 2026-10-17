/**
 * The lineage keeps its labels apart and inside its box (finding 5 of the 2026-10-17 walkthrough,
 * and W1-N's leftover V3).
 *
 * Its narrowest clear width (`lineageMinWidth`, which the live Columns card sets as its inner
 * box's minimum and scrolls below) counts only what keeps the labels apart: the raw labels' room,
 * the widest adjusted label with its pill's padding, and the matrix labels' room; the pill itself
 * is free to squeeze, as the layout always let it. At that width every adjusted label clears the
 * lane dots on either side by the pill's padding and every matrix label ends inside the drawing;
 * one pixel narrower, the widest does not clear them. The layout reads a monospace character as
 * 0.61 of the type size (Lineage.tsx `laneWidths`), which these checks use too. The NHANES roles
 * lineage the Columns card draws (the capture's predictors, grouped by role) fits the 634 px box
 * the card has at a 1280 px window (measured by W1-N's reviewer and again for W1-R), so it shows
 * whole there without a sideways scroll; the old rule asked 675.
 *
 * Its rows never draw below its box: where the box is shorter than its rows need at one row per
 * type size (a pitch at which no label runs into the next), the drawing is as tall as they need
 * and its box scrolls; where it is tall enough, the drawing fills the box and nothing scrolls.
 *
 * The lineage is the server's NHANES residual preview (src/mocks/m1-stage-fixture.json).
 */
import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import type {
  Lineage as LineageData,
  LineageView,
  PreviewResult,
} from "../../../api/m1-stage-types";
import stageFixture from "../../../mocks/m1-stage-fixture.json";
import { artifactAt, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { predictorsLineage } from "../LiveScenes";
import { Lineage, lineageMinWidth } from "./Lineage";
import { placeLineage } from "./lineageLayout";

const energy = (
  stageFixture as unknown as {
    previews: { energy_adjustment: { decision: { method?: string }; body: PreviewResult }[] };
  }
).previews.energy_adjustment;
const LINEAGE = (
  energy
    .find((x) => x.decision.method === "residual")!
    .body.views.find((v) => v.kind === "lineage") as LineageView
).after;

/** The full-size type, and the layout's reading of a monospace character's width. */
const FS = 11;
const CW = FS * 0.61;

let observed: ResizeObserverCallback[] = [];
beforeEach(() => {
  observed = [];
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(private cb: ResizeObserverCallback) {}
      observe() {
        observed.push(this.cb);
      }
      unobserve() {}
      disconnect() {}
    },
  );
});
afterEach(() => vi.unstubAllGlobals());

/** The lineage drawn in a box `width` by `height`. */
function draw(lineage: LineageData, width: number, height = 420) {
  const r = render(<Lineage lineage={lineage} />);
  act(() => {
    for (const cb of observed)
      cb([{ contentRect: { width, height } } as ResizeObserverEntry], {} as ResizeObserver);
  });
  observed = [];
  return r;
}

/** Where the three lanes sit (their heads' x) and the drawing's width, as drawn. */
function lanes(lineage: LineageData, width: number) {
  const r = draw(lineage, width);
  const heads = [...r.container.querySelectorAll("svg > g > g text")];
  const x = (word: string) =>
    Number(heads.find((t) => t.textContent?.startsWith(word))!.getAttribute("x"));
  const out = {
    x0: x("RAW"),
    x1: x("ADJUSTED"),
    x2: x("MATRIX"),
    w: Number(r.container.querySelector("svg")!.getAttribute("width")),
  };
  r.unmount();
  return out;
}

const labels = (lineage: LineageData, lane: string) =>
  placeLineage(lineage)
    .nodes.filter((n) => n.lane === lane)
    .map((n) => (n.node.count > 1 ? n.node.label : (n.node.column ?? n.node.label)));
const widest = (words: string[]) => Math.max(...words.map((s) => s.length)) * CW;

/** How far the widest adjusted label stays from the lane dots on its two sides, and how far the
 *  widest matrix label ends inside the drawing. */
function clearance(lineage: LineageData, width: number) {
  const { x0, x1, x2, w } = lanes(lineage, width);
  const half = widest(labels(lineage, "adjusted")) / 2;
  return {
    adjusted: Math.min(x1 - half - x0, x2 - (x1 + half)),
    matrix: w - (x2 + 7 + widest(labels(lineage, "matrix"))),
  };
}

describe("the lineage's narrowest clear width", () => {
  /** The pill's padding on either side of its label (laneWidths: the label's width plus 16). */
  const PAD = 8;
  /** The matrix lane's room after its label (laneWidths: the label's width plus 12, 7 before it). */
  const END = 5;
  const clear = (c: { adjusted: number; matrix: number }) =>
    c.adjusted >= PAD - 1e-9 && c.matrix >= END - 1e-9;

  it("keeps every adjusted label clear of the lane dots by the pill's padding, and every matrix label inside", () => {
    const min = lineageMinWidth(LINEAGE);
    expect(clear(clearance(LINEAGE, min))).toBe(true);
    expect(clear(clearance(LINEAGE, min - 1))).toBe(false);
  });

  it("is set by the widest adjusted label where that is wider than the lanes' least span", () => {
    // The lineage with one adjusted column's label made long, by construction.
    const one = LINEAGE.nodes.find((n) => n.lane === "adjusted" && n.count === 1 && n.column)!;
    const wide: LineageData = {
      ...LINEAGE,
      nodes: LINEAGE.nodes.map((n) =>
        n === one ? { ...n, column: "protein_per_kcal_of_total_energy_intake" } : n,
      ),
    };
    expect(widest(labels(wide, "adjusted")) + 2 * PAD).toBeGreaterThan(120);
    const min = lineageMinWidth(wide);
    expect(clear(clearance(wide, min))).toBe(true);
    const narrower = clearance(wide, min - 1);
    expect(narrower.adjusted).toBeLessThan(PAD);
  });

  it("is narrow enough for the NHANES roles lineage at a 1280 px window's Columns card", () => {
    const FIXTURES = import.meta.glob<M3Fixture>(
      "../../../mocks/fixtures/m3-nhanes-prediction.json",
      { eager: true, import: "default" },
    );
    const journey = Object.values(FIXTURES)[0]!;
    // The walk where the families question is open, as the Columns card draws it (LiveScenes.tsx
    // ColumnsCard: the design's lineage once computed, else the predictors as recorded).
    const view = viewsOf(journey).find((v) =>
      v.interview.some((s) => s.key === "models" && s.status === "open"),
    )!;
    const design =
      view.stages.design?.status === "fresh"
        ? (artifactAt(journey, "design", view.stages.design.key) as { lineage: LineageData })
        : null;
    const cohort = artifactAt(journey, "cohort", view.stages.cohort!.key) as {
      predictors: string[];
    };
    const roles =
      design?.lineage ??
      predictorsLineage(cohort.predictors, view.state.roles as Record<string, string>);
    // The card W1-N's reviewer measured: the 21 predictors grouped by role.
    expect(roles.collapsed).toBe(true);
    expect(labels(roles, "matrix")).toEqual([
      "13 covariate columns",
      "1 energy column",
      "7 exposure columns",
    ]);
    expect(lineageMinWidth(roles)).toBeLessThanOrEqual(634);
    const at = clearance(roles, 634);
    expect(at.adjusted).toBeGreaterThanOrEqual(PAD);
    expect(at.matrix).toBeGreaterThanOrEqual(0);
  });
});

describe("the lineage's rows in a short box", () => {
  const rows = placeLineage(LINEAGE).rows;
  const HEAD = 26; // the lane heads' band (Lineage.tsx)
  const FOOT = 6; // below the last row, with no note under it
  const need = HEAD + rows * FS + FOOT;

  it("draws as tall as its rows need, one row per type size, and its box scrolls, where the box is shorter", () => {
    expect(rows).toBeGreaterThan(15);
    const r = draw(LINEAGE, 900, 220);
    const svg = r.container.querySelector("svg")!;
    expect(Number(svg.getAttribute("height"))).toBe(need);
    expect(need).toBeGreaterThan(220);
    expect((r.container.firstElementChild as HTMLElement).style.overflowY).toBe("auto");
  });

  it("fills the box, and nothing scrolls, where the box is tall enough", () => {
    const r = draw(LINEAGE, 900, need + 40);
    expect(Number(r.container.querySelector("svg")!.getAttribute("height"))).toBe(need + 40);
    expect((r.container.firstElementChild as HTMLElement).style.overflowY).toBe("");
  });
});
