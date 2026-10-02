/**
 * The reshape storyboard (Tier A: row identity): the frames come in the method's own order — your
 * data now in file order, each unit's records gathered, combined or kept, one row per unit — and a
 * row is the same row from the file to the settled table, standing for exactly its unit's records.
 */
import type { TableFocusView } from "../../../api/m1-stage-types";
import { trackOf } from "../tracks";
import { changedAt, keeps, layout, reshapeOf, RH, sourceRows, valuesAt } from "./reshape";
import { ringT } from "./TurnTable";

type Rec = { row: number; unit: string; kcal: number; date: string };

/** A set_aggregation preview as turbotab/core/structure_previews.py builds it. */
function preview(records: Rec[], method: "mean" | "first" | "last"): TableFocusView {
  const units = [...new Set(records.map((r) => r.unit))];
  const shown = ["participant_id", "recall_date", "energy_kcal"];
  const settled = units.map((u) => {
    const recs = records.filter((r) => r.unit === u); // in the method's order (time, then file)
    if (method === "mean") {
      const kcal = recs.reduce((a, r) => a + r.kcal, 0) / recs.length;
      const ids = recs.map((r) => r.row).sort((a, b) => a - b);
      return { row_id: ids[0]!, unit: u, sources: ids, values: { participant_id: u, recall_date: recs[0]!.date, energy_kcal: kcal } };
    }
    const kept = method === "first" ? recs[0]! : recs[recs.length - 1]!;
    return { row_id: kept.row, unit: u, sources: [kept.row], values: { participant_id: u, recall_date: kept.date, energy_kcal: kept.kcal } };
  });
  const after = new Map(settled.map((s) => [s.unit, s.values]));
  return {
    kind: "table_focus",
    title: "2 units' records, combined",
    caption: "Numeric columns are averaged; other columns keep the first record.",
    emphasis: shown.slice(1),
    coach: [],
    columns_before: shown,
    columns_after: shown,
    rows: records.map((r) => ({
      row_id: r.row,
      before: { participant_id: r.unit, recall_date: r.date, energy_kcal: r.kcal },
      after: after.get(r.unit)!,
    })),
    changed: [],
    n_affected_columns: 9,
    story: [
      {
        label: "Each participant_id's records",
        columns: shown,
        rows: records.map((r) => ({
          row_id: r.row,
          unit: r.unit,
          sources: [r.row],
          values: { participant_id: r.unit, recall_date: r.date, energy_kcal: r.kcal },
        })),
      },
      { label: "Combined into one row each", columns: shown, rows: settled },
    ],
  } as unknown as TableFocusView;
}

const ADJACENT: Rec[] = [
  { row: 0, unit: "P001", kcal: 3063, date: "2023-01-01" },
  { row: 1, unit: "P001", kcal: 1913, date: "2023-01-08" },
  { row: 2, unit: "P002", kcal: 1228, date: "2023-01-02" },
  { row: 3, unit: "P002", kcal: 312, date: "2023-01-09" },
];

// A stacked export (all first recalls, then all second ones): a unit's rows are far apart.
const STACKED: Rec[] = [
  { row: 0, unit: "P001", kcal: 3063, date: "2023-01-01" },
  { row: 300, unit: "P001", kcal: 1913, date: "2023-01-08" },
  { row: 1, unit: "P002", kcal: 1228, date: "2023-01-02" },
  { row: 301, unit: "P002", kcal: 312, date: "2023-01-09" },
];

describe("the reshape storyboard", () => {
  it("is read from the preview's frames, in the method's own order of four real states", () => {
    const view = preview(ADJACENT, "mean");
    const track = trackOf(view);
    expect(track.states.map((s) => s.label)).toEqual([
      "Your data now",
      "Each participant_id's records",
      "Combined into one row each",
      "With this choice",
    ]);
    const m = reshapeOf(view)!;
    expect(m.idColumn).toBe("participant_id");
    expect(m.kind).toBe("combine");
    expect(m.units).toEqual(["P001", "P002"]);
    expect(m.rows.map((r) => [r.row, r.k])).toEqual([
      [0, 0],
      [1, 1],
      [2, 0],
      [3, 1],
    ]);
  });

  it("starts in file order, gathers each unit, and settles to one row per unit", () => {
    for (const records of [ADJACENT, STACKED]) {
      for (const method of ["mean", "first", "last"] as const) {
        const m = reshapeOf(preview(records, method))!;
        const s0 = layout(m, 0);
        // State 0: the rows in the file's own order (row ids increasing down the table).
        const byY = m.rows.map((r, i) => ({ row: r.row, y: s0.rows[i]!.y })).sort((a, b) => a.y - b.y);
        expect(byY.map((x) => x.row)).toEqual([...m.rows.map((r) => r.row)].sort((a, b) => a - b));
        // State 1: each unit's records sit together, in the method's order.
        const s1 = layout(m, 1);
        for (const [u, unit] of m.units.entries()) {
          const ys = m.rows.filter((r) => r.unit === unit).map((r) => [r.k, s1.rows[m.rows.indexOf(r)]!.y] as const);
          const sorted = [...ys].sort((a, b) => a[0] - b[0]).map(([, y]) => y);
          sorted.forEach((y, k) => expect(y).toBe(sorted[0]! + k * RH));
          expect(s1.frames[u]!.alpha).toBe(1);
        }
        // State 3: exactly one visible row per unit, one line each, in unit order.
        const s3 = layout(m, 3);
        const visible = m.rows.filter((_, i) => s3.rows[i]!.alpha > 0.5);
        expect(visible.map((r) => r.unit)).toEqual(m.units);
        visible.forEach((r) => expect(s3.rows[m.rows.indexOf(r)]!.y).toBe(m.units.indexOf(r.unit) * RH));
        expect(s3.height).toBe(m.units.length * RH);
        // Stacked rows are elided in file order, then travel to meet their partner.
        expect(s0.gaps.length).toBe(records === STACKED ? 1 : 0);
      }
    }
  });

  it("keeps each row's identity: a combined row stands for its unit's records, a kept one is that record", () => {
    const mean = reshapeOf(preview(STACKED, "mean"))!;
    const p1 = mean.rows.find((r) => r.row === 0)!;
    expect(sourceRows(mean, p1, 0)).toBe("0");
    expect(sourceRows(mean, p1, 3)).toBe("0+300");
    expect(valuesAt(mean, p1, 1).energy_kcal).toBe(3063); // gathered: still its own values
    expect(valuesAt(mean, p1, 3).energy_kcal).toBe(2488); // combined: its unit's mean
    expect(changedAt(mean, p1, 3, "energy_kcal")).toBe(true);
    expect(changedAt(mean, p1, 3, "participant_id")).toBe(false);
    // The settled row is the unit's first record's element, never a neighbor's.
    const s3 = layout(mean, 3);
    const shown = mean.rows.filter((_, i) => s3.rows[i]!.alpha > 0.5).map((r) => r.row);
    expect(shown).toEqual([0, 1]);

    const last = reshapeOf(preview(STACKED, "last"))!;
    expect(last.kind).toBe("keep");
    const kept = last.rows.filter((r) => keeps(last, r));
    expect(kept.map((r) => r.row)).toEqual([300, 301]);
    const l3 = layout(last, 3);
    // The kept record stays visible as itself; the record that leaves is struck, then gone.
    for (const r of last.rows) {
      const g = l3.rows[last.rows.indexOf(r)]!;
      expect(g.alpha).toBe(keeps(last, r) ? 1 : 0);
      expect(g.struck).toBe(keeps(last, r) ? 0 : 1);
      expect(valuesAt(last, r, 3)).toBe(r.values); // a kept or struck record shows its own values
      expect(sourceRows(last, r, 3)).toBe(String(r.row));
    }
    expect(layout(last, 2).rows.every((g) => g.alpha > 0)).toBe(true); // struck, still seen leaving
  });

  it("is not read into a table that does not reshape", () => {
    const view = preview(ADJACENT, "mean");
    expect(reshapeOf({ ...view, story: view.story.slice(0, 1) })).toBeNull();
    const noMap = {
      ...view,
      story: view.story.map((f) => ({ ...f, rows: f.rows.map((r) => ({ ...r, unit: null, sources: [] })) })),
    } as TableFocusView;
    expect(reshapeOf(noMap)).toBeNull();
  });
});

describe("the orientation turn", () => {
  // Cell (i, j) of an 8 × 8 corner on a square grid, at the turn's progress t.
  const at = (i: number, j: number, t: number, rings: number) => {
    const u = ringT(t, Math.abs(i - j), rings);
    return [j + (i - j) * u, i + (j - i) * u] as const;
  };

  it("moves every cell from its place to its mirror, and only forward", () => {
    for (let d = 0; d < 8; d++) {
      expect(ringT(0, d, 8)).toBe(0);
      expect(ringT(1, d, 8)).toBe(1);
      let prev = 0;
      for (let k = 0; k <= 100; k++) {
        const u = ringT(k / 100, d, 8);
        expect(u).toBeGreaterThanOrEqual(prev);
        prev = u;
      }
    }
  });

  it("never piles the cells up mid-turn: a pair swaps past at most the diagonal cell between them", () => {
    const n = 8;
    // The most cells sharing one spot (a quarter cell) at any moment of the turn.
    const pile = (move: (i: number, j: number, t: number) => readonly [number, number]) => {
      let most = 0;
      for (let k = 1; k < 200; k++) {
        const count = new Map<string, number>();
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) {
            const [x, y] = move(i, j, k / 200);
            const key = `${Math.round(x * 4)},${Math.round(y * 4)}`;
            count.set(key, (count.get(key) ?? 0) + 1);
          }
        most = Math.max(most, ...count.values());
      }
      return most;
    };
    expect(pile((i, j, t) => at(i, j, t, n))).toBeLessThanOrEqual(3);
    // Every cell at once would meet its whole anti-diagonal on the diagonal: the mess it replaces.
    expect(pile((i, j, t) => [j + (i - j) * t, i + (j - i) * t])).toBe(n);
  });
});
