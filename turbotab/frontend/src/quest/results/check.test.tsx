/**
 * The plan's check of the exposure's shape beside Table 2 (check.tsx) against what the real effects
 * stage serves (fixtures/estimate-check.json, written by fixtures/capture_check.py from the engine
 * test's own constructions; nothing in it by hand): a straight line kept on a cohort with a known
 * bend (the check: the spline's curve against the line, and its test), a spline kept (the line's
 * one coefficient), the line's check refused under multiple imputation, and no form declared (no
 * check). The truth is statsmodels' (the capture checks every served field against it and keeps it
 * beside the artifact): the drawn curve is read back off the screen against its own axis ticks and
 * compared with statsmodels' curve, the table alternative's numbers with it to the digits printed,
 * and each number a sentence says with statsmodels' to the digits it prints; each sentence is the
 * engine's, read from the capture.
 */
import { cleanup, render, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { EffectsArtifact } from "../../api/m3-types";
import { ShapeCheck } from "./check";
import capture from "./fixtures/estimate-check.json";

type Check = NonNullable<EffectsArtifact["families"][number]["check"]>;
interface Point {
  x: number;
  curve: number;
  line: number;
}
interface Captured {
  artifact: EffectsArtifact;
  truth?: {
    test?: { statistic: number; df_num: number; df_den: number; p: number };
    reference?: number;
    points?: Point[];
    line?: { estimate: number; ci_low: number; ci_high: number; p: number };
  };
  refused?: string;
}
const K = capture as unknown as Record<"line" | "spline" | "refused" | "none", Captured>;
const checkOf = (c: Captured) => c.artifact.families[0]!.check ?? null;
const LINE = checkOf(K.line)!;
const SPLINE = checkOf(K.spline)!;
const REFUSED = checkOf(K.refused)!;
const unticked = (s: string) => s.replaceAll("`", "");

/** Every number printed in `text`: "F(3, 591) = 107.20" → [3, 591, 107.2]; a real minus sign read
 *  as one. */
function numbersIn(text: string): number[] {
  return [...text.matchAll(/[−-]?\d[\d,]*(?:\.\d+)?/g)].map((m) => Number(m[0].replace("−", "-").replace(/,/g, "")));
}

/** A value printed to `digits` significant digits (whole numbers from 1,000) reads back within
 *  half a unit of its last printed digit. */
function shows(printed: number | undefined, truth: number, digits = 3): boolean {
  if (printed === undefined) return false;
  if (truth === 0) return printed === 0;
  const a = Math.abs(truth);
  const unit = a >= 1000 ? 1 : 10 ** (Math.floor(Math.log10(a)) - (digits - 1));
  return Math.abs(printed - truth) <= unit / 2 + 1e-9 * a;
}

/** A path's points ("M48,32L314,100…") in pixels. */
function pathPoints(d: string): [number, number][] {
  return [...d.matchAll(/[ML](-?[\d.]+),(-?[\d.]+)/g)].map((m) => [Number(m[1]), Number(m[2])]);
}

/** An axis as a reader reads it: its first and last ticks' values at their pixels, and the line
 *  through them from pixels back to values (the ticks are the axis's own labels). */
function axisOf(svg: SVGElement, which: "x" | "y"): { at: (px: number) => number; perPx: number } {
  const ticks = [...svg.querySelectorAll(`[data-axis="${which}"] [data-tick]`)].map((t) => {
    const v = Number(t.getAttribute("data-tick"));
    const px = which === "x" ? Number(t.getAttribute("x")) : Number(t.querySelector("line")!.getAttribute("y1"));
    return { v, px };
  });
  expect(ticks.length, `${which} axis ticks`).toBeGreaterThanOrEqual(2);
  const [a, b] = [ticks[0]!, ticks[ticks.length - 1]!];
  const perPx = (b.v - a.v) / (b.px - a.px);
  return { at: (px) => a.v + (px - a.px) * perPx, perPx: Math.abs(perPx) };
}

afterEach(cleanup);

describe("the capture", () => {
  it("holds a straight line kept, a spline kept, the line's check refused and no check, each the engine's", () => {
    expect([LINE.kept, LINE.shape, LINE.refused, LINE.points.length]).toEqual(["linear", "spline", null, 50]);
    expect(LINE.test!.distribution).toBe("F");
    expect([SPLINE.kept, SPLINE.shape, SPLINE.refused, SPLINE.points]).toEqual(["spline", "linear", null, []]);
    expect(SPLINE.line).not.toBeNull();
    expect([REFUSED.kept, REFUSED.shape, REFUSED.refused, REFUSED.points, REFUSED.test, REFUSED.line]).toEqual(["linear", "spline", K.refused.refused, [], null, null]);
    expect(checkOf(K.none)).toBeNull();
    // The served curve is statsmodels' (the capture asserts it to 1e-7; read back here).
    const truth = K.line.truth!.points!;
    LINE.points.forEach((p, i) => {
      expect(Math.abs(p.x - truth[i]!.x)).toBeLessThan(1e-9 * Math.abs(truth[i]!.x));
      expect(Math.abs(p.curve - truth[i]!.curve)).toBeLessThan(1e-6);
      expect(Math.abs(p.line! - truth[i]!.line)).toBeLessThan(1e-6);
    });
  });
});

describe("a straight line kept: the spline's curve against it", () => {
  it("draws each served point of the curve and the line, read back against the axes as statsmodels gives them", () => {
    const { container } = render(<ShapeCheck check={LINE} family="linear" />);
    const svg = container.querySelector("svg")!;
    expect(svg).not.toBeNull();
    const [x, y] = [axisOf(svg, "x"), axisOf(svg, "y")];
    const truth = K.line.truth!.points!;
    for (const [key, field] of [["spline", "curve"], ["linear", "line"]] as const) {
      const paths = svg.querySelectorAll(`[data-line="${key}"] [data-mark="line"]`);
      expect(paths, `${key}: one unbroken line`).toHaveLength(1);
      const pts = pathPoints(paths[0]!.getAttribute("d")!);
      expect(pts, `${key}: every served point`).toHaveLength(truth.length);
      pts.forEach(([px, py], i) => {
        // A path's pixels are rounded to 0.01 px: within that, each point is statsmodels'.
        expect(Math.abs(x.at(px) - truth[i]!.x), `${key} x at ${i}`).toBeLessThanOrEqual(0.006 * x.perPx + 1e-9);
        expect(Math.abs(y.at(py) - truth[i]![field]), `${key} ${field} at ${i}`).toBeLessThanOrEqual(0.006 * y.perPx + 1e-9);
      });
    }
    // No band: the served points carry no interval, and the view says none is drawn.
    expect(svg.querySelector('[data-mark="band"]')).toBeNull();
    expect(container.textContent).toContain("No interval is drawn around either line.");
    // Both lines named, never by color alone.
    expect(within(container.querySelector('[aria-label="Legend"]') as HTMLElement).getAllByRole("listitem")).toHaveLength(2);
  });

  it("leaves the straight line out where the engine serves it at no point, and draws the curve alone", () => {
    // As the engine serves the check where the primary reports no slope (`check_curve`, slope None).
    const bare = { ...LINE, points: LINE.points.map((p) => ({ ...p, line: null })) } as Check;
    const { container } = render(<ShapeCheck check={bare} family="linear" />);
    const svg = container.querySelector("svg")!;
    expect(svg.querySelectorAll('[data-line="spline"] [data-mark="line"]')).toHaveLength(1);
    expect(svg.querySelector('[data-line="linear"]')).toBeNull();
    expect(container.querySelector('[aria-label="Legend"]')).toBeNull();
    expect(container.textContent).toContain("No interval is drawn around the curve.");
    expect(container.textContent).not.toContain("either line");
  });

  it("names the exposure's column and the served reference on its axes, the reference read back as statsmodels' lowest knot", () => {
    const { container } = render(<ShapeCheck check={LINE} family="linear" />);
    const svg = container.querySelector("svg")!;
    const yTitle = svg.querySelector('[data-axis="y"] > text')!.textContent!;
    const xTitle = svg.querySelector('[data-axis="x"] > text:last-of-type')!.textContent!;
    expect(xTitle).toBe(LINE.column);
    expect(yTitle).toContain(LINE.column);
    const [ref] = numbersIn(yTitle);
    expect(shows(ref, K.line.truth!.reference!, 4), `"${yTitle}"`).toBe(true);
    // The caption names the same value as the lowest knot, to the same digits.
    expect(unticked(LINE.caption)).toContain(`at ${String(ref)},`);
  });

  it("lists every served point in its table alternative, to the digits it prints", () => {
    const { container } = render(<ShapeCheck check={LINE} family="linear" />);
    const rows = [...within(container).getByTestId("table-alternative").querySelectorAll("tbody tr")];
    const truth = K.line.truth!.points!;
    expect(rows).toHaveLength(truth.length);
    rows.forEach((tr, i) => {
      const got = [...tr.children].map((c) => numbersIn(c.textContent ?? "")[0]);
      const want = [truth[i]!.x, truth[i]!.curve, truth[i]!.line];
      want.forEach((v, j) => expect(shows(got[j], v), `row ${i}, column ${j}: ${got[j]} for ${v}`).toBe(true));
    });
  });

  it("says the check in the engine's caption, its test statsmodels' F to the digits printed, and composes no sentence of its own", () => {
    const { container } = render(<ShapeCheck check={LINE} family="linear" />);
    const section = within(container).getByTestId("results-form-check");
    const caption = within(section).getByTestId("results-form-check-caption");
    expect(caption.textContent).toBe(unticked(LINE.caption));
    const t = K.line.truth!.test!;
    const said = /F\((\d+), (\d+)\) = ([\d.]+), p = (.+?)\./.exec(caption.textContent!)!;
    expect(said, caption.textContent!).not.toBeNull();
    expect([Number(said[1]), Number(said[2])]).toEqual([t.df_num, t.df_den]);
    expect(shows(Number(said[3]), t.statistic, 5)).toBe(true);
    // "p = 2 × 10⁻⁵⁵": statsmodels' p to its one printed digit.
    const sup = "⁰¹²³⁴⁵⁶⁷⁸⁹";
    const [, m, e] = /^(\d) × 10⁻([⁰-⁹¹²³]+)$/.exec(said[4]!)!;
    const printed = Number(m) * 10 ** -Number([...e!].map((c) => sup.indexOf(c)).join(""));
    expect(shows(printed, t.p, 1), `${said[4]} for ${t.p}`).toBe(true);
    // Beside the figure, the section says its heading and the engine's caption, nothing else.
    const figure = section.querySelector("figure")!;
    const outside = [...section.childNodes].filter((n) => n !== figure).map((n) => n.textContent).join("");
    expect(outside).toBe(`The plan's check of the shape${unticked(LINE.caption)}`);
  });
});

describe("a spline kept: the straight line's one coefficient", () => {
  it("says it in the engine's caption, statsmodels' coefficient and interval to the digits printed, and draws no curve", () => {
    const { container } = render(<ShapeCheck check={SPLINE} family="linear" />);
    const section = within(container).getByTestId("results-form-check");
    expect(section.getAttribute("data-shape")).toBe("linear");
    const caption = within(section).getByTestId("results-form-check-caption");
    expect(caption.textContent).toBe(unticked(SPLINE.caption));
    const t = K.spline.truth!.line!;
    const said = /coefficient ([−-]?[\d.]+) per unit, 95% CI ([−-]?[\d.]+) to ([−-]?[\d.]+), p = ([\d.]+)\./.exec(caption.textContent!)!;
    expect(said, caption.textContent!).not.toBeNull();
    const [est, lo, hi, p] = said.slice(1).map(Number);
    expect([shows(est, t.estimate, 4), shows(lo, t.ci_low, 4), shows(hi, t.ci_high, 4), shows(p, t.p, 2)]).toEqual([true, true, true, true]);
    expect(section.querySelector("svg")).toBeNull();
    expect(section.querySelector("figure")).toBeNull();
    expect(section.querySelector("table")).toBeNull();
    expect(section.textContent).toBe(`The plan's check of the shape${unticked(SPLINE.caption)}`);
  });
});

describe("the check refused", () => {
  it("says the refusal in the engine's words and draws no number", () => {
    const { container } = render(<ShapeCheck check={REFUSED} family="linear" />);
    const section = within(container).getByTestId("results-form-check");
    expect(section.getAttribute("data-refused")).toBe("true");
    const caption = within(section).getByTestId("results-form-check-caption");
    expect(caption.textContent).toBe(unticked(`The plan's check of the straight line could not be made: ${K.refused.refused!}`));
    expect(section.querySelector("svg")).toBeNull();
    expect(section.querySelector("figure")).toBeNull();
    expect(section.querySelector("table")).toBeNull();
    expect(section.textContent).toBe(`The plan's check of the shape${unticked(REFUSED.caption)}`);
  });

  it("draws no curve from a refused check even with points beside it (the guard's other side)", () => {
    // The served kept-line check, its refusal set as the engine sets one: the points are not drawn.
    const { container } = render(<ShapeCheck check={{ ...LINE, refused: REFUSED.refused, caption: REFUSED.caption } as Check} family="linear" />);
    expect(container.querySelector("svg")).toBeNull();
    expect(within(container).getByTestId("results-form-check-caption").textContent).toBe(unticked(REFUSED.caption));
  });
});

describe("no check", () => {
  it("draws nothing where the family's check is null", () => {
    const { container } = render(<ShapeCheck check={checkOf(K.none)} family="linear" />);
    expect(container.innerHTML).toBe("");
  });
});
