import { act, render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { NOW, OneLine, TableAlternative, tipLeft } from "./parts";
import { insetRange, linear, ticksIn, widen } from "./scale";
import { ViewsLab } from "../lab/ViewsLab";
import { entries } from "../lab/curves.lab";

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const lum = (rgb: number[]) => {
  const [r, g, b] = rgb.map((c) => {
    const v = c / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
};
const contrast = (a: number[], b: number[]) => {
  const [x, y] = [lum(a), lum(b)];
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
};
const token = (block: string, name: string) => new RegExp(`${name}:\\s*(#[0-9A-Fa-f]{6})`).exec(block)![1]!;

describe("the shared scale math", () => {
  it("names a single value as its only tick, never the widening around it", () => {
    expect(widen([100, 100])).toEqual([90, 110]);
    expect(ticksIn([100, 100])).toEqual([100]);
    expect(ticksIn([0.21, 0.21])).toEqual([0.21]);
    expect(ticksIn([0.013, 0.87])).toEqual([0.2, 0.4, 0.6, 0.8]);
  });

  it("insets the pixel range so a mark at the domain's edge sits off the axis", () => {
    expect(insetRange([40, 588], 8)).toEqual([48, 580]);
    expect(insetRange([262, 24], 8)).toEqual([254, 32]);
    const x = linear([0, 10], [40, 588], 8);
    expect(x(0)).toBe(48);
    expect(x(10)).toBe(580);
  });
});

describe("the tooltip's place", () => {
  it("centres on the point, and a wide tooltip near an edge is pushed inside by its own width", () => {
    expect(tipLeft(216, 432, 180)).toBe(216);
    // a 260 px spec-curve tooltip over the right-hand columns of a 432 px pane
    expect(tipLeft(420, 432, 260)).toBe(432 - 130 - 4);
    expect(tipLeft(10, 432, 260)).toBe(134);
    // wider than the view: centred on it
    expect(tipLeft(10, 200, 260)).toBe(100);
  });
});

describe("the data-now gray", () => {
  const tokens = readFileSync(resolve(__dirname, "../../../explore/calm-kit/tokens.css"), "utf8");
  it("is at least 3:1 on the canvas in both themes, and stays apart from the fit's ink", () => {
    const light = /:root\s*\{([^}]*)\}/.exec(tokens)![1]!;
    const dark = /:root\[data-theme="dark"\]\s*\{([^}]*)\}/.exec(tokens)![1]!;
    const share = Number(/var\(--data-context\) (\d+)%/.exec(NOW)![1]) / 100;
    for (const block of [light, dark]) {
      const ctx = hex(token(block, "--data-context"));
      const ink = hex(token(block, "--canvas-ink"));
      const mixed = ctx.map((c, i) => Math.round(c * share + ink[i]! * (1 - share)));
      expect(contrast(mixed, hex(token(block, "--canvas")))).toBeGreaterThanOrEqual(3);
      expect(contrast(mixed, hex(token(block, "--data-fit")))).toBeGreaterThanOrEqual(2.5);
    }
  });
});

describe("every view in the lab", () => {
  it("draws every marker at least 9 px across (radius 4.5, the ring painted under the fill)", () => {
    const { container } = render(<ViewsLab entries={entries} />);
    const marks = [...container.querySelectorAll("circle")].filter((c) => !c.getAttribute("class")?.includes("hit"));
    expect(marks.length).toBeGreaterThan(20);
    for (const c of marks) expect(Number(c.getAttribute("r"))).toBeGreaterThanOrEqual(4.5);
  });

  it("gives every drawn view a table alternative, the sealed curve included", () => {
    const { container } = render(<ViewsLab entries={entries} />);
    for (const fig of container.querySelectorAll("figure:not([data-empty])")) expect(fig.querySelector('[data-testid="table-alternative"]')).toBeTruthy();
    expect(container.querySelectorAll('figure[data-sealed="true"] [data-testid="table-alternative"]').length).toBeGreaterThan(0);
  });
});

describe("the shared parts", () => {
  it("says one line instead of an empty frame, and draws nothing", () => {
    const { container } = render(<OneLine view="curve" text="There is no curve to draw." />);
    expect(container.querySelector('[data-exhibit-view="curve"]')).toHaveAttribute("data-empty", "true");
    expect(container.querySelector("svg")).toBeNull();
    expect(container.querySelector('[role="note"]')).toHaveTextContent("There is no curve to draw.");
  });

  it("renders a lazy table alternative only once it is opened", () => {
    const { container, getByTestId } = render(<TableAlternative>{() => <table data-testid="lazy" />}</TableAlternative>);
    expect(container.querySelector('[data-testid="lazy"]')).toBeNull();
    const d = getByTestId("table-alternative") as HTMLDetailsElement;
    act(() => {
      d.open = true;
      d.dispatchEvent(new Event("toggle"));
    });
    expect(container.querySelector('[data-testid="lazy"]')).not.toBeNull();
  });
});
