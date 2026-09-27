/**
 * S4 — a transform touching 495 columns of a 60 × 500 genomics table. The table shows the 12 whose
 * shape changed most (the fixture's ranking), and a minimap of all 495 ties each shown column to
 * where it sits in the file: the view is the same size at 495 columns or 20,000.
 */
import { useMemo, useState } from "react";
import { Prose } from "../../../components/Prose";
import { WIDE_OPTIONS, WIDE_Q } from "../copy";
import { FX, fmtNum, view, type Lineage, type TableFocusView } from "../data";
import { useScrub } from "../engine/scrub";
import { Question, type ShelfOption } from "../Question";
import { PipelineStrip, RecordBar, ScrubBar, StageFrame, StageSection, type StripState } from "../Stage";
import { ScrubHistogram, type HistState } from "../views/Histogram";
import { ScrubLineage, type LineageState } from "../views/Lineage";
import { MorphTable, type TableState } from "../views/MorphTable";
import s from "../Scenario.module.css";
import w from "./Wide.module.css";

const B = FX.scenario_b;
const TF = view(B.preview.views, "table_focus") as TableFocusView;
const DIST = view(B.preview.views, "distribution")!;
const LIN = view(B.preview.views, "lineage")!;
const N = B.count_columns.n;
const FOCUS = DIST.column;

function tableStates(): Record<string, TableState> {
  const cols = (side: "before" | "after", status: "same" | "changed"): TableState["cols"] =>
    Object.fromEntries(
      TF.columns_before.map((c) => [
        c,
        { name: c, values: TF.rows.map((r) => fmtNum(r[side][c] as number)), status },
      ]),
    );
  const first = TF.columns_before[0]!;
  const last = TF.columns_before[TF.columns_before.length - 1]!;
  return {
    now: { cols: cols("before", "same"), notes: [] },
    log: { cols: cols("after", "changed"), notes: [{ id: "f", text: B.transform.formula, from: first, to: last, tone: "formula" }] },
    keep: { cols: cols("before", "same"), notes: [{ id: "u", text: "unchanged", from: first, to: last }] },
  };
}

function histStates(): Record<string, HistState> {
  const rugBefore = TF.rows.map((r) => ({ row: r.row_id, value: r.before[FOCUS] as number }));
  const rugAfter = TF.rows.map((r) => ({ row: r.row_id, value: r.after[FOCUS] as number }));
  const before: HistState = {
    edges: DIST.before.edges,
    counts: DIST.before.counts,
    domain: "counts",
    label: DIST.before_label,
    marks: [],
    after: false,
    rug: rugBefore,
    note: "skewness 3.2",
  };
  return {
    now: before,
    keep: { ...before, after: true },
    log: {
      edges: DIST.after.edges,
      counts: DIST.after.counts,
      domain: "log",
      label: DIST.after_label,
      marks: [],
      after: true,
      rug: rugAfter,
      note: "skewness 0.1; median 70 becomes 6.15",
    },
  };
}

function lineageStates(): Record<string, LineageState> {
  return {
    now: { lineage: LIN.before as Lineage, emphasis: [], after: false },
    keep: { lineage: LIN.before as Lineage, emphasis: [], after: true },
    log: { lineage: LIN.after, emphasis: LIN.emphasis, after: true },
  };
}

function strip(): Record<string, StripState> {
  const mk = (sub: string, touched: boolean): StripState => ({
    rows: { value: String(B.dataset.n_rows), sub: "samples" },
    columns: { value: String(N), sub },
    results: { value: "—", sub: "not fitted yet" },
    touched: touched ? ["columns"] : [],
  });
  return { now: mk("count columns, raw", false), keep: mk("count columns, raw", false), log: mk("count columns, log2", true) };
}

/** Where the shown columns sit among all 495: one tick per column, the shown ones joined up. */
function Minimap({ pos, width }: { pos: Record<string, { x: number; w: number }>; width: number }) {
  const pad = 8;
  const full = Math.max(width, 600);
  const x = (i: number) => pad + (i / (N - 1)) * (full - 2 * pad);
  const shown = TF.columns_before.map((c) => ({ c, i: Number(c.replace(/\D/g, "")) - 1 }));
  return (
    <div className={w.minimap}>
      <svg width={full} height={46} aria-hidden="true">
        {Array.from({ length: N }, (_, i) => (
          <line key={i} x1={x(i)} x2={x(i)} y1={30} y2={40} className={w.tick} />
        ))}
        {shown.map(({ c, i }) => {
          const p = pos[c];
          if (!p) return null;
          const cx = p.x + p.w / 2;
          return (
            <g key={c}>
              <path d={`M${cx},0 C${cx},16 ${x(i)},14 ${x(i)},28`} className={w.leader} />
              <line x1={x(i)} x2={x(i)} y1={27} y2={42} className={w.shown} />
            </g>
          );
        })}
      </svg>
      <span className={w.caption}>
        <span className={w.count}>12 / {N}</span> <Prose text="count columns shown, most changed first; the rest change the same way" />
      </span>
    </div>
  );
}

export function WideScenario() {
  const { active } = useScrub();
  const [recorded, setRecorded] = useState<string | null>(null);
  const [hotRow, setHotRow] = useState<number | null>(null);
  const table = useMemo(() => tableStates(), []);
  const hist = useMemo(() => histStates(), []);
  const lineage = useMemo(() => lineageStates(), []);
  const pipe = useMemo(() => strip(), []);
  const sources = useMemo(() => (LIN.before as Lineage).nodes.filter((n) => n.lane === "raw").map((n) => n.label), []);
  const options: ShelfOption[] = [
    { key: "log", label: WIDE_OPTIONS.log!.label, cols: ["0.1", String(N)], consequence: WIDE_OPTIONS.log!.consequence },
    { key: "keep", label: WIDE_OPTIONS.keep!.label, cols: ["3.2", "0"], consequence: WIDE_OPTIONS.keep!.consequence },
  ];
  const recordable = (k: string) => k in WIDE_OPTIONS;

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <Question
          id="wide"
          kicker={WIDE_Q.kicker}
          question={WIDE_Q.question}
          why={WIDE_Q.why}
          columns={["`gene_0430` skew", "changed"]}
          colWidths="112px 56px"
          options={options}
          recorded={recorded ? { key: recorded, sentence: WIDE_OPTIONS[recorded]!.sentence } : null}
          onRecord={setRecorded}
          onChange={() => setRecorded(null)}
          recordable={recordable}
        />
      </div>
      <StageFrame label="Consequence preview">
        <PipelineStrip states={pipe} />
        <ScrubBar
          afterLabel={active ? `With ${WIDE_OPTIONS[active]!.short}` : null}
          recorded={!!recorded}
          unchanged={active === "keep"}
        />
        <StageSection kicker={`Working table · ${TF.rows.length} of ${B.dataset.n_rows} samples`} aside={<Prose text={`+ ${N - TF.columns_before.length} more count columns`} />}>
          <MorphTable
            slots={TF.columns_before}
            rowIds={TF.rows.map((r) => r.row_id)}
            states={table}
            rowH={23}
            fontPx={11.5}
            span={[0, 0.8]}
            stableWidths
            label="The 12 most-changed count columns"
            hotRow={hotRow}
            onRowHover={setHotRow}
            footer={(pos, width) => <Minimap pos={pos} width={width} />}
          />
        </StageSection>
        <div className={s.split}>
          <StageSection kicker="Most changed column" aside={<Prose text={`\`${FOCUS}\``} />}>
            <ScrubHistogram states={hist} height={176} span={[0.15, 1]} hotRow={hotRow} label={DIST.title} />
          </StageSection>
          <StageSection kicker="Columns into the model">
            <ScrubLineage states={lineage} sources={sources} rowH={25} span={[0.25, 1]} />
          </StageSection>
        </div>
        <RecordBar
          basis={B.preview.basis}
          action={active && recordable(active) ? `Record: ${WIDE_OPTIONS[active]!.label}` : null}
          onRecord={() => active && setRecorded(active)}
          recorded={!!recorded}
          onChange={() => setRecorded(null)}
        />
      </StageFrame>
    </div>
  );
}
