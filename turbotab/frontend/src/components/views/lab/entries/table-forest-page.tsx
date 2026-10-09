/** /lab/views: the table, forest and page view kinds, each with its fixture and its states. */
import { useState } from "react";
import { ExhibitTable, TABLE_PURPOSE } from "../../ExhibitTable";
import { Forest, FOREST_PURPOSE, TableWithForest } from "../../Forest";
import { PagePreview, PAGE_PURPOSE } from "../../PagePreview";
import { pageFromExhibits } from "../../adapters";
import type { ForestData, Placement, TableData } from "../../types";
import type { LabEntry } from "../entry";
import { exhibitModel, forest, forestMulti, forestRatio, GATE, table1, table2, table2Multi, table2Ratio } from "../fixtures";

/** A table with its forest as the last column: pointing at or focusing a row lights it whole. */
function Linked({ table, data }: { table: TableData; data: ForestData }) {
  const [lit, setLit] = useState<string | null>(null);
  return <TableWithForest table={table} forest={data} gate={null} lit={lit} onLit={setLit} />;
}

/** A forest alone, lit by pointing at a row. */
function Lit({ data }: { data: ForestData }) {
  const [lit, setLit] = useState<string | null>(null);
  return <Forest data={data} gate={null} lit={lit} onLit={setLit} />;
}

const OPTIONS: Placement[] = ["results", "discussion", "supplement", "left_out"];

/** The page with the card's placement options: pointing at one previews it. */
function Placing({ exhibit }: { exhibit: string }) {
  const page = pageFromExhibits(exhibitModel(), exhibit)!;
  const [preview, setPreview] = useState<Placement | null>(null);
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 14, color: "var(--canvas-muted)" }}>
        Point at a placement:
        {OPTIONS.map((p) => (
          <button
            key={p}
            type="button"
            onPointerEnter={() => setPreview(p)}
            onPointerLeave={() => setPreview(null)}
            onFocus={() => setPreview(p)}
            onBlur={() => setPreview(null)}
            style={{ background: "none", border: 0, padding: 0, cursor: "pointer", textDecoration: "underline", color: "var(--canvas-ink)" }}
          >
            {p === "left_out" ? "Left out" : p[0]!.toUpperCase() + p.slice(1)}
          </button>
        ))}
      </div>
      <PagePreview data={page} gate={null} preview={preview} />
    </div>
  );
}

const onePoint = { ...forest, rows: forest.rows.filter((r) => r.primary) };

export const entries: LabEntry[] = [
  {
    kind: "table",
    order: 1,
    purpose: TABLE_PURPOSE,
    samples: [
      { label: "Table 1, overall, from the profile (NHANES, 21,849 rows)", render: () => <ExhibitTable data={table1} gate={null} /> },
      { label: "Table 2, a difference, from the effects (NHANES: sugar and glucose)", render: () => <ExhibitTable data={table2} gate={null} /> },
      { label: "Table 2, an odds ratio, from the effects (time-varying: DASH and CVD)", render: () => <ExhibitTable data={table2Ratio} gate={null} /> },
      {
        label: "Table 2, three exposures in group rows (NHANES's fit: sugar, protein, carbohydrate)",
        render: () => <ExhibitTable data={table2Multi} gate={null} />,
      },
      { label: "No rows", render: () => <ExhibitTable data={{ ...table2, rows: [] }} gate={null} /> },
      { label: "Before Fit (rule 6)", render: () => <ExhibitTable data={table2} gate={GATE} /> },
    ],
  },
  {
    kind: "forest",
    order: 2,
    purpose: FOREST_PURPOSE,
    samples: [
      { label: "A difference on a linear axis (NHANES); point at a row", render: () => <Lit data={forest} /> },
      { label: "An odds ratio on a log axis (time-varying)", render: () => <Forest data={forestRatio} gate={null} /> },
      {
        label: "Three models as series, sage, plum and ochre in the sequence's order (NHANES's fit, three exposures)",
        render: () => <Lit data={forestMulti} />,
      },
      { label: "Beside Table 2, as its last column: rows aligned, estimates printed once", render: () => <Linked table={table2} data={forest} />, wide: true },
      { label: "Beside Table 2 with group rows and three series", render: () => <Linked table={table2Multi} data={forestMulti} />, wide: true },
      { label: "One point", render: () => <Forest data={onePoint} gate={null} /> },
      { label: "No rows", render: () => <Forest data={{ ...forest, rows: [] }} gate={null} /> },
      {
        label: "No row has an estimate (a model that did not converge), on a ratio axis",
        render: () => <Forest data={{ ...forestRatio, rows: forestRatio.rows.map((r) => ({ ...r, est: null, lo: null, hi: null })) }} gate={null} />,
      },
      {
        label: "Six series: refused in one line, the rows still in a table",
        render: () => (
          <Forest
            data={{
              ...forest,
              series: ["a", "b", "c", "d", "e", "f"].map((k) => ({ key: k, label: `Series ${k.toUpperCase()}` })),
              rows: forest.rows.map((r, i) => ({ ...r, series: "abcdef"[i] })),
            }}
            gate={null}
          />
        ),
      },
      { label: "Before Fit (rule 6)", render: () => <Forest data={forest} gate={GATE} /> },
    ],
  },
  {
    kind: "page",
    order: 10,
    purpose: PAGE_PURPOSE,
    samples: [
      { label: "Table 2 in Results; the floor holds it there", render: () => <Placing exhibit="table2" />, wide: true },
      { label: "Table 1 in Results; the floor keeps it out of the Discussion", render: () => <Placing exhibit="table1" />, wide: true },
      { label: "Figure S1 in the Supplement; point to move it", render: () => <Placing exhibit="figure1" />, wide: true },
      {
        label: "Figure S1 pointed at Results: renumbered Figure 1",
        render: () => <PagePreview data={pageFromExhibits(exhibitModel(), "figure1")!} gate={null} preview="results" />,
        wide: true,
      },
      {
        label: "Table 1 pointed at the Discussion: held, with where it can go",
        render: () => <PagePreview data={pageFromExhibits(exhibitModel(), "table1")!} gate={null} preview="discussion" />,
        wide: true,
      },
      {
        label: "Table 1 left out: unnumbered, still listed in the supplement; Table 2 becomes Table 1",
        render: () => <PagePreview data={pageFromExhibits(exhibitModel({ table1: "left_out" }), "table1")!} gate={null} />,
        wide: true,
      },
      {
        label: "Nothing drafted, no other exhibit",
        render: () => {
          const p = pageFromExhibits(exhibitModel(), "table2")!;
          return <PagePreview data={{ ...p, exhibits: p.exhibits.filter((e) => e.key === "table2"), text: { results: [], discussion: [] } }} gate={null} />;
        },
        wide: true,
      },
      { label: "Before Fit (rule 6): placements and captions, no drafted sentence", render: () => <PagePreview data={pageFromExhibits(exhibitModel(), "table2")!} gate={GATE} />, wide: true },
    ],
  },
];
