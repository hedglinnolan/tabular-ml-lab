import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { ExhibitTable } from "./ExhibitTable";
import { DOT, Forest, plotWidth, ROW, TableWithForest } from "./Forest";
import { focusOf, PagePreview } from "./PagePreview";
import { exhibitModel, forest, forestMulti, forestRatio, table1, table2, table2Multi } from "./lab/fixtures";
import { pageFromExhibits } from "./adapters";
import { forestScale } from "./scale";
import type { ForestData, PageData } from "./types";

const GATE = "Estimates open when Fit is pressed.";

describe("the table view", () => {
  it("prints Table 2 as the paper does: caption, primary in weight, quiet footnote marks", () => {
    const { container } = render(<ExhibitTable data={table2} gate={null} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("Table 2.").tagName).toBe("B");
    const rows = [...container.querySelectorAll("tbody tr[data-row]")];
    expect(rows.map((r) => r.getAttribute("data-row"))).toEqual(["crude", "model_2", "model_3"]);
    const primary = container.querySelector('tr[data-primary="true"]')!;
    expect(primary).toHaveTextContent("−0.0199 (−0.0327 to −0.00718)");
    expect(primary.querySelector("th")).toHaveTextContent(/^Model 2 \(primary\)a/);
    expect(screen.getByTestId("footnotes")).toHaveTextContent("Model 3: adjusted for kcal");
    // The p and n columns align on their digits; the estimate column reads left to right.
    const heads = [...table.querySelectorAll("thead th")];
    expect(heads.map((h) => h.className.includes("num"))).toEqual([false, false, true, true]);
  });

  it("is its own table alternative: header cells carry scopes", () => {
    render(<ExhibitTable data={table1} gate={null} />);
    expect(screen.getAllByRole("columnheader").length).toBe(2);
    expect(screen.getAllByRole("rowheader").length).toBeGreaterThan(5);
    expect(screen.getByText("11,195 (51.2)")).toBeInTheDocument();
  });

  it("says one line instead of an empty frame, and draws no estimate before its gate", () => {
    const empty = render(<ExhibitTable data={{ ...table2, rows: [] }} gate={null} />);
    expect(empty.queryByRole("table")).toBeNull();
    expect(empty.getByRole("note")).toHaveTextContent("no estimate");
    empty.unmount();
    const gated = render(<ExhibitTable data={table2} gate={GATE} />);
    expect(gated.queryByRole("table")).toBeNull();
    expect(gated.getByRole("note")).toHaveTextContent(GATE);
    expect(gated.container).not.toHaveTextContent("0.0199");
  });

  it("lights a row from the keyboard as well as the pointer", () => {
    const lit: (string | null)[] = [];
    const { container } = render(<ExhibitTable data={table2} gate={null} onLit={(k) => lit.push(k)} />);
    const row = container.querySelector('tr[data-row="model_3"]') as HTMLElement;
    expect(row.tabIndex).toBe(0);
    fireEvent.focus(row);
    fireEvent.blur(row);
    expect(lit).toEqual(["model_3", null]);
  });
});

const known: ForestData = {
  stub: "Model",
  measure: "Difference per unit",
  axis: "linear",
  reference: 0,
  rows: [
    { key: "a", label: "A", est: -5, lo: -10, hi: -1 },
    { key: "b", label: "B", est: -2, lo: -4, hi: 0, primary: true },
  ],
};

describe("the forest view", () => {
  it("draws a known estimate at its pixel, each row on its row's line, ticks the data reaches", () => {
    // Width 256: domain [−10.6, 0.6] onto [16, 240], 20 px a unit (scale.test.ts).
    const { container } = render(<Forest data={known} gate={null} width={256} />);
    const dots = [...container.querySelectorAll("g[data-mark] circle[data-dot]")];
    const at = dots.map((d) => [Number(d.getAttribute("cx")), Number(d.getAttribute("cy"))]);
    expect(at[0]![0]).toBeCloseTo(128);
    expect(at[1]![0]).toBeCloseTo(188);
    expect(at.map((p) => p[1])).toEqual([ROW / 2, ROW + ROW / 2]);
    // Every dot is at least 8 px across, with its ring painted outside it, never over its edge.
    for (const d of dots) {
      expect(Number(d.getAttribute("r")) * 2).toBeGreaterThanOrEqual(8);
      expect(d.getAttribute("stroke-width")).toBeNull();
      const ring = d.previousElementSibling!;
      expect(ring.getAttribute("data-ring")).toBe("true");
      expect(Number(ring.getAttribute("r"))).toBe(Number(d.getAttribute("r")) + DOT.ring);
    }
    const line = container.querySelector('g[data-mark="a"] line')!;
    expect(Number(line.getAttribute("x1"))).toBeCloseTo(28);
    expect(Number(line.getAttribute("x2"))).toBeCloseTo(208);
    expect(line.getAttribute("stroke-width")).toBe("2");
    const ticks = [...container.querySelectorAll("text[data-tick]")].map((t) => t.textContent);
    expect(ticks).toEqual(["−10", "−5", "0"]);
    expect(Number(container.querySelector('[data-testid="reference"]')!.getAttribute("x1"))).toBeCloseTo(228);
  });

  it("aligns its rows to Table 2's by key and order, and links the pointed row", () => {
    const lit: (string | null)[] = [];
    const { container } = render(<Forest data={forest} gate={null} onLit={(k) => lit.push(k)} />);
    expect([...container.querySelectorAll("div[data-row][tabindex]")].map((r) => r.getAttribute("data-row"))).toEqual(
      table2.rows.flatMap((r) => (r.kind === "row" ? [r.key] : [])),
    );
    fireEvent.pointerEnter(container.querySelector('g[data-mark="model_2"] rect')!);
    // The printed estimate is part of the row: it links and shows the tooltip too.
    fireEvent.pointerEnter(container.querySelector('[data-est="crude"]')!);
    fireEvent.pointerMove(container.querySelector('[data-est="crude"]')!, { clientX: 5, clientY: 5 });
    expect(lit).toEqual(["model_2", "crude"]);
    expect(screen.getByTestId("view-tip")).toHaveTextContent("Unadjusted");
  });

  it("lights the whole row, stub, estimate and band, and names its stub from the data", () => {
    const { container } = render(<Forest data={{ ...forest, stub: "Subgroup" }} gate={null} lit="model_3" />);
    expect(container.querySelector('[data-row="model_3"]')).toHaveAttribute("data-lit", "true");
    expect(container.querySelector('[data-est="model_3"]')).toHaveAttribute("data-lit", "true");
    expect(container.querySelector('g[data-mark="model_3"] rect')).toHaveAttribute("data-lit", "true");
    expect(container.querySelector('[data-row="crude"]')).not.toHaveAttribute("data-lit");
    expect(container).toHaveTextContent("Subgroup");
    expect(container).not.toHaveTextContent(/^Model/);
  });

  it("says both sides of the reference in plain words, wherever the reference sits", () => {
    for (const data of [forest, forestRatio]) {
      const r = render(<Forest data={data} gate={null} width={274} />);
      const sides = r.getByTestId("forest-sides");
      expect(sides).toHaveTextContent(`← ${data.sides![0]}`);
      expect(sides).toHaveTextContent(`${data.sides![1]} →`);
      r.unmount();
    }
  });

  it("measures its width once it draws, though it first mounted behind a closed gate", () => {
    const observed: { el: Element; cb: ResizeObserverCallback }[] = [];
    const Before = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class {
      constructor(private cb: ResizeObserverCallback) {}
      observe(el: Element) {
        observed.push({ el, cb: this.cb });
      }
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver;
    try {
      const r = render(<Forest data={known} gate="Not yet." />);
      expect(observed).toHaveLength(0);
      r.rerender(<Forest data={known} gate={null} />);
      expect(observed).toHaveLength(1);
      act(() => observed[0]!.cb([{ contentRect: { width: 512 } } as ResizeObserverEntry], {} as ResizeObserver));
      expect(r.container.querySelector("svg")).toHaveAttribute("width", "512");
    } finally {
      globalThis.ResizeObserver = Before;
    }
  });

  it("shows a tooltip on a row, value first, with a hit target the row's full height", () => {
    const { container } = render(<Forest data={known} gate={null} width={256} />);
    const band = container.querySelector('g[data-mark="b"] rect')!;
    expect(Number(band.getAttribute("height"))).toBe(ROW);
    fireEvent.pointerMove(band, { clientX: 10, clientY: 10 });
    expect(screen.getByTestId("view-tip")).toHaveTextContent("−2 (−4 to 0) · B");
  });

  it("puts a ratio on a log axis with 1 as its reference", () => {
    const { container } = render(<Forest data={forestRatio} gate={null} width={400} />);
    const ticks = [...container.querySelectorAll("text[data-tick]")].map((t) => Number(t.getAttribute("data-tick")));
    expect(ticks).toContain(1);
    for (const t of ticks) expect(t >= 1 && t <= 3.47).toBe(true);
  });

  it("draws one point, and says one line for no rows, a closed gate, no estimate or a ratio below zero", () => {
    const one = render(<Forest data={{ ...known, rows: [known.rows[1]!] }} gate={null} width={256} />);
    expect(one.container.querySelectorAll("g[data-mark] circle[data-dot]")).toHaveLength(1);
    one.unmount();
    const unfitted = forestRatio.rows.map((r) => ({ ...r, est: null, lo: null, hi: null }));
    for (const [data, gate, says] of [
      [{ ...known, rows: [] }, null, "No estimate to draw yet."],
      [known, GATE, GATE],
      [{ ...known, axis: "log", reference: 1 }, null, "ratio at or below zero"],
      // A ratio forest whose models did not converge has no estimate; it is not a sign problem.
      [{ ...forestRatio, rows: unfitted }, null, "No row has an estimate to draw."],
    ] as const) {
      const r = render(<Forest data={data as ForestData} gate={gate} />);
      expect(r.container.querySelector("svg")).toBeNull();
      expect(r.getByRole("note")).toHaveTextContent(says);
      r.unmount();
    }
  });

  it("has a table alternative and, for two series or more, a legend in the palette's order", () => {
    const two: ForestData = {
      ...known,
      series: [
        { key: "lin", label: "Linear" },
        { key: "rob", label: "Robust" },
      ],
      rows: known.rows.map((r, i) => ({ ...r, series: i ? "rob" : "lin" })),
    };
    const { container } = render(<Forest data={two} gate={null} width={256} />);
    expect(within(screen.getByTestId("table-alternative")).getByRole("table")).toBeInTheDocument();
    expect(screen.getByTestId("forest-legend")).toHaveTextContent("LinearRobust");
    const fills = [...container.querySelectorAll("g[data-mark] circle[data-dot]")].map((c) => (c as SVGElement).style.fill);
    expect(fills).toEqual(["var(--cat-1)", "var(--cat-2)"]);
  });

  it("colors each model by its place in the sequence, never by rank, with a direct label each", () => {
    const { container } = render(<Forest data={forestMulti} gate={null} width={400} />);
    const fill = (key: string) => (container.querySelector(`g[data-mark="${key}"] circle[data-dot]`) as SVGElement).style.fill;
    expect(["sugar:crude", "sugar:model_2", "sugar:model_3", "protein:model_2", "carb:model_3"].map(fill)).toEqual([
      "var(--cat-1)",
      "var(--cat-2)",
      "var(--cat-3)",
      "var(--cat-2)",
      "var(--cat-3)",
    ]);
    expect(screen.getByTestId("forest-legend")).toHaveTextContent("UnadjustedModel 2 (primary)Model 3");
    expect([...container.querySelectorAll("[data-direct-label]")].map((t) => t.textContent)).toEqual(["Unadjusted", "Model 2 (primary)", "Model 3"]);
  });

  it("refuses six series, or a row naming no declared series, in one line, its rows still in a table", () => {
    const six: ForestData = {
      ...known,
      series: "abcdef".split("").map((k) => ({ key: k, label: k })),
      rows: known.rows.map((r, i) => ({ ...r, series: "ab"[i] })),
    };
    const stray: ForestData = { ...six, series: six.series!.slice(0, 2), rows: [{ ...known.rows[0]!, series: "a" }, { ...known.rows[1]!, series: "z" }] };
    for (const [data, says] of [
      [six, "6 series are more than the comparison palette's 5 colors"],
      [stray, "The row “B” names no declared series"],
    ] as const) {
      const r = render(<Forest data={data} gate={null} width={256} />);
      expect(r.container.querySelector("svg")).toBeNull();
      expect(r.getByRole("note")).toHaveTextContent(says);
      expect(within(r.getByTestId("table-alternative")).getByRole("table")).toBeInTheDocument();
      r.unmount();
    }
  });
});

describe("the forest beside its table", () => {
  it("is the table's last column: one mark per row, on the table's scale, each estimate printed once", () => {
    const W = 720;
    const { container } = render(<TableWithForest table={table2} forest={forest} gate={null} width={W} />);
    expect(container.querySelectorAll("table")).toHaveLength(1);
    const P = plotWidth(W);
    const s = forestScale(forest.rows, { axis: "linear", reference: 0, width: P, inset: 16 })!;
    for (const r of forest.rows) {
      const dot = container.querySelector(`tr[data-row="${r.key}"] svg[data-mark="${r.key}"] circle[data-dot]`)!;
      expect(Number(dot.getAttribute("cx"))).toBeCloseTo(s.x(r.est!));
      expect(dot.getAttribute("cy")).toBe("50%");
    }
    expect(screen.getAllByText("−0.0199 (−0.0327 to −0.00718)")).toHaveLength(1);
    expect(screen.getAllByText("Unadjusted")).toHaveLength(1);
    expect(screen.getByTestId("forest-sides")).toHaveTextContent("← Lower mean glucose");
  });

  it("gives group rows the grid alone, so a multi-exposure table keeps its rows aligned", () => {
    const { container } = render(<TableWithForest table={table2Multi} forest={forestMulti} gate={null} width={720} />);
    const groups = [...container.querySelectorAll('tr[data-group="true"]')];
    expect(groups.map((g) => g.textContent?.startsWith("Per unit of"))).toEqual([true, true, true]);
    for (const g of groups) {
      expect(g.querySelector("svg [data-testid=reference]")).not.toBeNull();
      expect(g.querySelector("circle")).toBeNull();
    }
    expect(container.querySelectorAll("tbody tr[data-row] circle[data-dot]")).toHaveLength(forestMulti.rows.length);
    expect(screen.getByTestId("forest-legend")).toBeInTheDocument();
  });

  it("shows nothing but the gate's line while the gate is closed", () => {
    const { container } = render(<TableWithForest table={table2} forest={forest} gate={GATE} />);
    expect(container.querySelector("svg, table")).toBeNull();
    expect(screen.getByRole("note")).toHaveTextContent(GATE);
  });
});

const page = (): PageData => pageFromExhibits(exhibitModel(), "figure1")!;
const slotOf = (region: string, text: string) => within(screen.getByRole("region", { name: region })).getByText(text).closest("[data-slot]");

describe("the page view", () => {
  it("places the exhibit with its caption and its number in placement order", () => {
    const { container } = render(<PagePreview data={page()} gate={null} />);
    expect(slotOf("Supplement", "Figure S1.")).toHaveAttribute("data-this", "true");
    expect(within(screen.getByRole("region", { name: "Supplement" })).getByText(/Difference in mean glucose per unit of sugar, with 95% intervals/)).toBeInTheDocument();
    expect(container.querySelector('[data-exhibit-view="page"]')).toHaveAttribute("data-placement", "supplement");
    // Every exhibit is a block of number and caption at the page's type size; no shrunken copy.
    expect(container.querySelector("svg")).toBeNull();
  });

  it("lists a left-out analysis unnumbered in the supplement, and renumbers what is left", () => {
    const model = exhibitModel({ table1: "left_out" });
    expect(model.exhibits.map((e) => e.number)).toEqual([null, "Table 1", "Figure S1"]);
    expect(model.text.results[0]).toMatch(/\(Table 1\)\.$/);
    const { container } = render(<PagePreview data={pageFromExhibits(model, "table1")!} gate={null} />);
    const left = container.querySelector('[data-slot="table1"]')!;
    expect(left).toHaveAttribute("data-this", "true");
    expect(left.querySelector("b")).toBeNull();
    expect(within(screen.getByRole("region", { name: "Supplement" })).getByText("Analyses left out")).toBeInTheDocument();
  });

  it("moves the exhibit to the pointed placement in the choice's color, renumbered, gray where it sits now", () => {
    const { container } = render(<PagePreview data={page()} gate={null} preview="results" />);
    expect(slotOf("Main text", "Figure 1.")).toHaveAttribute("data-touched", "true");
    expect(slotOf("Supplement", "Where it sits now.")).toHaveAttribute("data-was", "true");
    expect(container.querySelector('[data-exhibit-view="page"]')).toHaveAttribute("data-placement", "results");
  });

  it("renumbers the drafted sentences' references with the page when a move renumbers it", () => {
    const t1 = pageFromExhibits(exhibitModel(), "table1")!;
    const { container } = render(<PagePreview data={t1} gate={null} preview="supplement" />);
    expect(slotOf("Supplement", "Table S1.")).toHaveAttribute("data-touched", "true");
    expect(slotOf("Main text", "Table 1.")?.getAttribute("data-slot")).toBe("table2");
    expect(container).toHaveTextContent("adjusted for 10 characteristics (Table 1).");
    expect(container).not.toHaveTextContent("(Table 2)");
  });

  it("holds an exhibit the floor fixes, saying why in one line", () => {
    const t2 = pageFromExhibits(exhibitModel(), "table2")!;
    const { container } = render(<PagePreview data={t2} gate={null} preview="supplement" />);
    expect(screen.getByRole("note")).toHaveTextContent("Table 2 stays in Results: the locked primary always stays in Results.");
    expect(container.querySelector("[data-touched]")).toBeNull();
  });

  it("holds an exhibit pointed at a placement the floor does not allow, saying where it can go", () => {
    const t1 = pageFromExhibits(exhibitModel(), "table1")!;
    expect(focusOf(t1).allowed).toEqual(["results", "supplement", "left_out"]);
    const { container } = render(<PagePreview data={t1} gate={null} preview="discussion" />);
    expect(screen.getByRole("note")).toHaveTextContent(
      "Table 1 cannot be placed in the Discussion: the methods floor allows Results or the Supplement, or leaving it out.",
    );
    expect(container.querySelector("[data-touched]")).toBeNull();
    expect(container.querySelector('[data-exhibit-view="page"]')).toHaveAttribute("data-placement", "results");
  });

  it("opens the Discussion's sentences on a capital, with no backtick left", () => {
    const { discussion } = exhibitModel().text;
    for (const t of discussion) {
      expect(t[0]).toBe(t[0]!.toUpperCase());
      expect(t).not.toContain("`");
    }
    expect(discussion[1]).toMatch(/^Sugar, protein, carb and 4 more are/);
  });

  it("shows no drafted sentence, so no estimate, while the gate is closed", () => {
    const t2 = pageFromExhibits(exhibitModel(), "table2")!;
    const { container } = render(<PagePreview data={t2} gate={GATE} />);
    expect(screen.getByTestId("page-gate")).toHaveTextContent(GATE);
    expect(container).not.toHaveTextContent("0.0199");
    expect(container).not.toHaveTextContent("95% CI");
    expect(slotOf("Main text", "Table 2.")).toHaveAttribute("data-this", "true");
  });

  it("says a section has nothing drafted, and has a table alternative", () => {
    const p = page();
    render(<PagePreview data={{ ...p, exhibits: [focusOf(p)], text: { results: [], discussion: [] } }} gate={null} />);
    expect(screen.getAllByText("Nothing drafted here yet.")).toHaveLength(2);
    expect(within(screen.getByTestId("table-alternative")).getByRole("table")).toHaveTextContent("Supplement");
  });
});
