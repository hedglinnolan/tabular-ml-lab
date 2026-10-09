import { fireEvent, render, screen, within } from "@testing-library/react";
import { ExhibitTable } from "./ExhibitTable";
import { Forest, ROW } from "./Forest";
import { PagePreview } from "./PagePreview";
import { exhibitModel, forest, forestRatio, table1, table2 } from "./lab/fixtures";
import { pageFromExhibits } from "./adapters";
import type { ForestData, PageData } from "./types";

const GATE = "Estimates open when Fit is pressed.";

describe("the table view", () => {
  it("prints Table 2 as the paper does: caption, primary in weight, quiet footnote marks", () => {
    const { container } = render(<ExhibitTable data={table2} />);
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
    render(<ExhibitTable data={table1} />);
    expect(screen.getAllByRole("columnheader").length).toBe(2);
    expect(screen.getAllByRole("rowheader").length).toBeGreaterThan(5);
    expect(screen.getByText("11,195 (51.2)")).toBeInTheDocument();
  });

  it("says one line instead of an empty frame, and draws no estimate before its gate", () => {
    const empty = render(<ExhibitTable data={{ ...table2, rows: [] }} />);
    expect(empty.queryByRole("table")).toBeNull();
    expect(empty.getByRole("note")).toHaveTextContent("no estimate");
    empty.unmount();
    const gated = render(<ExhibitTable data={table2} gate={GATE} />);
    expect(gated.queryByRole("table")).toBeNull();
    expect(gated.getByRole("note")).toHaveTextContent(GATE);
    expect(gated.container).not.toHaveTextContent("0.0199");
  });
});

const known: ForestData = {
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
    const { container } = render(<Forest data={known} width={256} />);
    const dots = [...container.querySelectorAll("g[data-mark] circle")];
    const at = dots.map((d) => [Number(d.getAttribute("cx")), Number(d.getAttribute("cy"))]);
    expect(at[0]![0]).toBeCloseTo(128);
    expect(at[1]![0]).toBeCloseTo(188);
    expect(at.map((p) => p[1])).toEqual([ROW / 2, ROW + ROW / 2]);
    expect(Number(dots[1]!.getAttribute("r")) * 2).toBeGreaterThanOrEqual(8);
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
    const { container } = render(<Forest data={forest} onLit={(k) => lit.push(k)} />);
    expect([...container.querySelectorAll("div[data-row][tabindex]")].map((r) => r.getAttribute("data-row"))).toEqual(
      table2.rows.flatMap((r) => (r.kind === "row" ? [r.key] : [])),
    );
    fireEvent.pointerEnter(container.querySelector('g[data-mark="model_2"] rect')!);
    expect(lit).toEqual(["model_2"]);
  });

  it("shows a tooltip on a row, value first, with a hit target the row's full height", () => {
    const { container } = render(<Forest data={known} width={256} />);
    const band = container.querySelector('g[data-mark="b"] rect')!;
    expect(Number(band.getAttribute("height"))).toBe(ROW);
    fireEvent.pointerMove(band, { clientX: 10, clientY: 10 });
    expect(screen.getByTestId("view-tip")).toHaveTextContent("−2 (−4 to 0) · B");
  });

  it("puts a ratio on a log axis with 1 as its reference", () => {
    const { container } = render(<Forest data={forestRatio} width={400} />);
    const ticks = [...container.querySelectorAll("text[data-tick]")].map((t) => Number(t.getAttribute("data-tick")));
    expect(ticks).toContain(1);
    for (const t of ticks) expect(t >= 1 && t <= 3.47).toBe(true);
  });

  it("draws one point, and says one line for no rows, a closed gate or a ratio below zero", () => {
    const one = render(<Forest data={{ ...known, rows: [known.rows[1]!] }} width={256} />);
    expect(one.container.querySelectorAll("g[data-mark] circle")).toHaveLength(1);
    one.unmount();
    for (const [data, gate, says] of [
      [{ ...known, rows: [] }, undefined, "No estimate to draw yet."],
      [known, GATE, GATE],
      [{ ...known, axis: "log", reference: 1 }, undefined, "ratio at or below zero"],
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
    const { container } = render(<Forest data={two} width={256} />);
    expect(within(screen.getByTestId("table-alternative")).getByRole("table")).toBeInTheDocument();
    expect(screen.getByTestId("forest-legend")).toHaveTextContent("LinearRobust");
    const fills = [...container.querySelectorAll("g[data-mark] circle")].map((c) => (c as SVGElement).style.fill);
    expect(fills).toEqual(["var(--cat-1)", "var(--cat-2)"]);
  });
});

const page = (): PageData => pageFromExhibits(exhibitModel(), "figure1")!;

describe("the page view", () => {
  it("places the exhibit with its caption, and lists left-out analyses in the supplement", () => {
    const { container } = render(<PagePreview data={page()} />);
    const supplement = screen.getByRole("region", { name: "Supplement" });
    expect(within(supplement).getByText("Figure 1.").closest("[data-slot]")).toHaveAttribute("data-this", "true");
    expect(within(supplement).getByText(/Difference in mean glucose per unit of sugar, with 95% intervals/)).toBeInTheDocument();
    expect(container.querySelector('[data-exhibit-view="page"]')).toHaveAttribute("data-placement", "supplement");
    const left = render(<PagePreview data={{ ...page(), exhibit: { ...page().exhibit, placement: "left_out" } }} />);
    expect(within(left.getAllByRole("region", { name: "Supplement" })[0]!).getByText("Analyses left out")).toBeInTheDocument();
  });

  it("moves the exhibit to the pointed placement in the choice's color, gray where it sits now", () => {
    const { container } = render(<PagePreview data={page()} preview="results" />);
    expect(within(screen.getByRole("region", { name: "Main text" })).getByText("Figure 1.").closest("[data-slot]")).toHaveAttribute("data-touched", "true");
    expect(within(screen.getByRole("region", { name: "Supplement" })).getByText("Figure 1.").closest("[data-slot]")).toHaveAttribute("data-was", "true");
    expect(container.querySelector('[data-exhibit-view="page"]')).toHaveAttribute("data-placement", "results");
  });

  it("holds an exhibit the floor fixes, saying why in one line", () => {
    const t2 = pageFromExhibits(exhibitModel(), "table2")!;
    const { container } = render(<PagePreview data={t2} preview="supplement" />);
    expect(screen.getByRole("note")).toHaveTextContent("Table 2 stays in Results: the locked primary always stays in Results.");
    expect(container.querySelector("[data-touched]")).toBeNull();
  });

  it("says a section has nothing drafted, and has a table alternative", () => {
    render(<PagePreview data={{ exhibit: page().exhibit, text: { results: [], discussion: [] } }} />);
    expect(screen.getAllByText("Nothing drafted here yet.")).toHaveLength(2);
    expect(within(screen.getByTestId("table-alternative")).getByRole("table")).toHaveTextContent("Supplement");
  });
});
