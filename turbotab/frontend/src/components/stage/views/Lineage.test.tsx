/**
 * The lineage keeps its labels apart and inside its box (finding 5 of the 2026-10-17 walkthrough,
 * W1-N's leftover V3, and the W1-R verifier's V4).
 *
 * An adjusted pill is drawn whole or not at all: never narrower than its label with its padding
 * (the label's width plus 16, Lineage.tsx `laneWidths`), and never closer to the lane dots on
 * either side than a link's run of 40 px. Where the lanes are too close for the widest pill, the
 * adjusted lane draws its dots alone, as a thumbnail does. The narrowest whole width
 * (`lineageMinWidth`, which the live Columns card sets as its inner box's minimum and scrolls
 * below) is the raw labels' room, the widest pill with a link's run on either side, and the matrix
 * labels' room: at that width every pill is whole and every matrix label ends inside the drawing,
 * and one pixel narrower the pills give way to dots. The layout reads a monospace character as
 * 0.61 of the type size, which these checks use too. The NHANES roles lineage the Columns card
 * draws (the capture's predictors, grouped by role) fits whole in the 634 px box the card has at a
 * 1280 px window (measured by W1-N's reviewer and again for W1-R), without a sideways scroll.
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

/** Where the three lanes sit (their heads' x) and the drawing's width, in a drawing. */
function lanesIn(container: HTMLElement) {
  const heads = [...container.querySelectorAll("svg > g > g text")];
  const x = (word: string) =>
    Number(heads.find((t) => t.textContent?.startsWith(word))!.getAttribute("x"));
  return {
    x0: x("RAW"),
    x1: x("ADJUSTED"),
    x2: x("MATRIX"),
    w: Number(container.querySelector("svg")!.getAttribute("width")),
  };
}

const labels = (lineage: LineageData, lane: string) =>
  placeLineage(lineage)
    .nodes.filter((n) => n.lane === lane)
    .map((n) => (n.node.count > 1 ? n.node.label : (n.node.column ?? n.node.label)));
const widest = (words: string[]) => Math.max(...words.map((s) => s.length)) * CW;

/** Each adjusted pill as drawn: its label and its width (the rect just before its label; a group's
 *  deck of cards lies behind it), and where the lanes sit, all read off one drawing. */
function pills(lineage: LineageData, width: number) {
  const r = draw(lineage, width);
  const { x0, x1, x2 } = lanesIn(r.container);
  const drawn = [...r.container.querySelectorAll("svg g")].flatMap((g) => {
    const kids = [...g.children];
    const text = kids.find((k) => k.tagName.toLowerCase() === "text");
    const rects = kids.filter((k) => k.tagName.toLowerCase() === "rect");
    return text && rects.length
      ? [{ label: text.textContent ?? "", w: Number(rects[rects.length - 1]!.getAttribute("width")) }]
      : [];
  });
  r.unmount();
  return { x0, x1, x2, drawn };
}

/** The pill's padding on either side of its label (laneWidths: the label's width plus 16). */
const PAD = 8;
/** A link's least run between a pill and the lane dot on either side. */
const LINK = 40;
/** The matrix lane's room after its label (laneWidths: the label's width plus 12, 7 before it). */
const END = 5;
const whole = (p: { label: string; w: number }) => p.w >= p.label.length * CW + 2 * PAD - 1e-9;

/** Whether, at `width`, every adjusted node is a whole pill clear of both lanes' dots by a link's
 *  run, and every matrix label ends inside the drawing. */
function wholeAt(lineage: LineageData, width: number) {
  const { x0, x1, x2, drawn } = pills(lineage, width);
  const widestPill = Math.max(...drawn.map((p) => p.w));
  const matrix = width - (x2 + 7 + widest(labels(lineage, "matrix")));
  return (
    drawn.length === labels(lineage, "adjusted").length &&
    drawn.every(whole) &&
    x1 - widestPill / 2 - x0 >= LINK - 1e-9 &&
    x2 - (x1 + widestPill / 2) >= LINK - 1e-9 &&
    matrix >= END - 1e-9
  );
}

describe("the lineage's adjusted pills", () => {
  it("are whole at the narrowest whole width, clear of the lane dots, and give way to dots one pixel narrower", () => {
    const min = lineageMinWidth(LINEAGE);
    expect(labels(LINEAGE, "adjusted").length).toBeGreaterThan(5);
    expect(wholeAt(LINEAGE, min)).toBe(true);
    expect(pills(LINEAGE, min - 1).drawn).toEqual([]);
  });

  // It draws the lineage once at every seventh pixel from 200 to 400 past the narrowest whole width
  // (one full jsdom render a width, under a second on this machine): its timeout fits that
  // work on a slower runner (CI's timed out at the 5 s default, W1-T), never cutting a width.
  it("are never drawn narrower than their labels, at any width", { timeout: 30_000 }, () => {
    const min = lineageMinWidth(LINEAGE);
    const all = labels(LINEAGE, "adjusted").length;
    let widths = 0;
    for (let width = 200; width <= min + 400; width += 7) {
      const { drawn } = pills(LINEAGE, width);
      expect(drawn.filter((p) => !whole(p)), `${width} px`).toEqual([]);
      expect([0, all], `${width} px`).toContain(drawn.length);
      widths += 1;
    }
    expect(widths).toBe(Math.floor((min + 400 - 200) / 7) + 1);
  });

  it("set the narrowest whole width by the widest adjusted label where that is wider than the lanes' least span", () => {
    // The lineage with one adjusted column's label made long, by construction.
    const one = LINEAGE.nodes.find((n) => n.lane === "adjusted" && n.count === 1 && n.column)!;
    const wide: LineageData = {
      ...LINEAGE,
      nodes: LINEAGE.nodes.map((n) =>
        n === one ? { ...n, column: "protein_per_kcal_of_total_energy_intake" } : n,
      ),
    };
    expect(widest(labels(wide, "adjusted")) + 2 * PAD + 2 * LINK).toBeGreaterThan(120);
    const min = lineageMinWidth(wide);
    expect(min).toBeGreaterThan(lineageMinWidth(LINEAGE));
    expect(wholeAt(wide, min)).toBe(true);
    expect(pills(wide, min - 1).drawn).toEqual([]);
  });

  it("are whole in the NHANES roles lineage at a 1280 px window's Columns card", () => {
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
    expect(wholeAt(roles, 634)).toBe(true);
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
