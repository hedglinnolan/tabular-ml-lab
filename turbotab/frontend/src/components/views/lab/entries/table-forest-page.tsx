/** /lab/views: the table, forest and page view kinds, each with its fixture and its states. */
import { useState } from "react";
import { ExhibitTable, TABLE_PURPOSE } from "../../ExhibitTable";
import { Forest, FOREST_PURPOSE } from "../../Forest";
import { PagePreview, PAGE_PURPOSE } from "../../PagePreview";
import { pageFromExhibits } from "../../adapters";
import type { Placement } from "../../types";
import type { LabEntry } from "../entry";
import { exhibitModel, forest, forestRatio, table1, table2, table2Ratio } from "../fixtures";

const GATE = "Estimates open when Fit is pressed: the plan is fixed before any estimate is shown.";

/** Table 2 beside its forest: pointing at a row in either lights it in both. */
function Linked() {
  const [lit, setLit] = useState<string | null>(null);
  return (
    <div style={{ display: "grid", gap: 20 }}>
      <ExhibitTable data={table2} lit={lit} onLit={setLit} />
      <Forest data={forest} lit={lit} onLit={setLit} />
    </div>
  );
}

/** The page with the card's placement options: pointing at one previews it. */
function Placing({ exhibit }: { exhibit: string }) {
  const page = pageFromExhibits(exhibitModel(), exhibit)!;
  const [preview, setPreview] = useState<Placement | null>(null);
  const options: Placement[] = ["results", "discussion", "supplement", "left_out"];
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 14, color: "var(--canvas-muted)" }}>
        Point at a placement:
        {options.map((p) => (
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
      <PagePreview data={page} preview={preview} thumb={exhibit === "figure1" ? <Forest data={forest} width={420} /> : <ExhibitTable data={table2} />} />
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
      { label: "Table 1, overall, from the profile (NHANES, 21,849 rows)", render: () => <ExhibitTable data={table1} /> },
      { label: "Table 2, a difference, from the effects (NHANES: sugar and glucose)", render: () => <ExhibitTable data={table2} /> },
      { label: "Table 2, an odds ratio, from the effects (time-varying: DASH and CVD)", render: () => <ExhibitTable data={table2Ratio} /> },
      { label: "No rows", render: () => <ExhibitTable data={{ ...table2, rows: [] }} /> },
      { label: "Before Fit (rule 6)", render: () => <ExhibitTable data={table2} gate={GATE} /> },
    ],
  },
  {
    kind: "forest",
    order: 2,
    purpose: FOREST_PURPOSE,
    samples: [
      { label: "A difference on a linear axis (NHANES)", render: () => <Forest data={forest} /> },
      { label: "An odds ratio on a log axis (time-varying)", render: () => <Forest data={forestRatio} /> },
      { label: "Rows aligned to Table 2, linked", render: () => <Linked /> },
      { label: "One point", render: () => <Forest data={onePoint} /> },
      { label: "No rows", render: () => <Forest data={{ ...forest, rows: [] }} /> },
      { label: "Before Fit (rule 6)", render: () => <Forest data={forest} gate={GATE} /> },
    ],
  },
  {
    kind: "page",
    order: 10,
    purpose: PAGE_PURPOSE,
    samples: [
      { label: "Table 2 in Results; the floor holds it there", render: () => <Placing exhibit="table2" />, wide: true },
      { label: "Figure 1 in the Supplement; point to move it", render: () => <Placing exhibit="figure1" />, wide: true },
      {
        label: "Figure 1 pointed at Results",
        render: () => <PagePreview data={pageFromExhibits(exhibitModel(), "figure1")!} preview="results" thumb={<Forest data={forest} width={420} />} />,
        wide: true,
      },
      {
        label: "Table 1 left out: still listed in the supplement",
        render: () => {
          const p = pageFromExhibits(exhibitModel(), "table1")!;
          return <PagePreview data={{ ...p, exhibit: { ...p.exhibit, placement: "left_out" } }} />;
        },
        wide: true,
      },
      {
        label: "Nothing drafted, no other exhibit",
        render: () => {
          const p = pageFromExhibits(exhibitModel(), "table2")!;
          return <PagePreview data={{ exhibit: p.exhibit, text: { results: [], discussion: [] } }} />;
        },
        wide: true,
      },
    ],
  },
];
