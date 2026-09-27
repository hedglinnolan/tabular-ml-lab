/**
 * S2 — the exclusions question: three rules, each a row-flow step and a cut kcal distribution.
 * Ordered from cutting nothing to cutting most, so each ↓ excludes more and the flip teaches it.
 */
import { useMemo, useState } from "react";
import { EXCL_OPTIONS, EXCL_Q } from "../copy";
import { EXCLUSIONS, FX, fmtInt, view, type DistributionView } from "../data";
import { useScrub } from "../engine/scrub";
import { Question, type ShelfOption } from "../Question";
import { PipelineStrip, RecordBar, ScrubBar, StageFrame, StageSection, type StripState } from "../Stage";
import { ScrubHistogram, type CutMark, type HistState } from "../views/Histogram";
import { ScrubRowFlow, type FlowState } from "../views/RowFlow";
import s from "../Scenario.module.css";

function flowStates(): Record<string, FlowState> {
  const first = view(EXCLUSIONS[0]!.preview.views, "row_flow")!;
  const out: Record<string, FlowState> = { now: { steps: first.before, after: false } };
  for (const o of EXCLUSIONS) {
    const rf = view(o.preview.views, "row_flow")!;
    const cut = rf.after.find((x) => x.key.startsWith("exclude"));
    out[o.key] = {
      steps: rf.after,
      after: true,
      note: cut
        ? { step: cut.key, text: EXCL_OPTIONS[o.key]!.sidenote }
        : { step: rf.after[rf.after.length - 1]!.key, text: EXCL_OPTIONS[o.key]!.sidenote },
    };
  }
  return out;
}

/** Each cut value carries the rows it removes on its outer side (the fixture's by-level counts). */
function marksOf(d: DistributionView, byLevel: Record<string, { below: number; above: number }>): CutMark[] {
  const groups = new Map<string | null, number[]>();
  for (const m of d.marks) groups.set(m.group, [...(groups.get(m.group) ?? []), m.value]);
  return d.marks.map((m) => {
    const vals = groups.get(m.group)!;
    const side = m.value === Math.min(...vals) ? "below" : "above";
    const level = byLevel[m.group ?? "all"];
    return { value: m.value, label: m.label, group: m.group, side, cut: level ? level[side] : 0 };
  });
}

function histStates(): { states: Record<string, HistState>; breakAt: number } {
  const d0 = view(EXCLUSIONS[0]!.preview.views, "distribution")!;
  const out: Record<string, HistState> = {
    now: {
      edges: d0.before.edges,
      counts: d0.before.counts,
      domain: "kcal",
      label: d0.before_label,
      marks: [],
      after: false,
    },
  };
  for (const o of EXCLUSIONS) {
    const d = view(o.preview.views, "distribution")!;
    out[o.key] = {
      edges: d.after.edges,
      counts: d.after.counts,
      ghost: d.before.counts,
      domain: "kcal",
      label: d.after_label,
      marks: marksOf(d, o.counts.by_level),
      after: true,
    };
  }
  // Past 1.5× the highest cut the tail is a few rows per bin: draw it narrow, after an axis break.
  const top = Math.max(...EXCLUSIONS.flatMap((o) => view(o.preview.views, "distribution")!.marks.map((m) => m.value)));
  const breakAt = d0.before.edges.find((e) => e >= top * 1.5) ?? d0.before.edges[d0.before.edges.length - 1]!;
  return { states: out, breakAt };
}

function strip(): Record<string, StripState> {
  const cols = FX.scenario_a.setup.predictors.length;
  const mk = (n: number, touched: boolean): StripState => ({
    rows: { value: fmtInt(n), sub: "rows in the cohort" },
    columns: { value: String(cols), sub: "predictors" },
    results: { value: "—", sub: "not fitted yet" },
    touched: touched ? ["rows"] : [],
  });
  const out: Record<string, StripState> = { now: mk(FX.scenario_a.dataset.n_rows, false) };
  for (const o of EXCLUSIONS) out[o.key] = mk(o.counts.final, o.counts.excluded > 0);
  return out;
}

export function ExclusionScenario() {
  const { active } = useScrub();
  const [recorded, setRecorded] = useState<string | null>(null);
  const flow = useMemo(() => flowStates(), []);
  const hist = useMemo(() => histStates(), []);
  const pipe = useMemo(() => strip(), []);
  const options: ShelfOption[] = useMemo(
    () =>
      EXCLUSIONS.map((o) => ({
        key: o.key,
        label: EXCL_OPTIONS[o.key]!.label,
        cols: [o.counts.excluded ? `−${fmtInt(o.counts.excluded)}` : "0", fmtInt(o.counts.final)],
        consequence: EXCL_OPTIONS[o.key]!.consequence,
        badge: o.evidence?.status,
        why: o.evidence?.quote,
      })),
    [],
  );
  const lanes = useMemo(() => ["female", "male"], []);
  const recordable = (k: string) => k in EXCL_OPTIONS;
  const label = active ? `With ${EXCL_OPTIONS[active]!.short}` : null;

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <Question
          id="exclusions"
          kicker={EXCL_Q.kicker}
          question={EXCL_Q.question}
          why={EXCL_Q.why}
          columns={["rows cut", "kept"]}
          colWidths="56px 58px"
          options={options}
          recorded={recorded ? { key: recorded, sentence: EXCL_OPTIONS[recorded]!.sentence } : null}
          onRecord={setRecorded}
          onChange={() => setRecorded(null)}
          recordable={recordable}
        />
      </div>
      <StageFrame label="Consequence preview">
        <PipelineStrip states={pipe} />
        <ScrubBar afterLabel={label} recorded={!!recorded} unchanged={active === "keep_all"} />
        <StageSection kicker="Rows kept at each step">
          <ScrubRowFlow states={flow} span={[0, 0.8]} />
        </StageSection>
        <StageSection
          kicker="kcal, with the cuts marked"
          aside={
            <span className={s.legend}>
              <i className={s.swKept} /> kept <i className={s.swCut} /> cut
            </span>
          }
        >
          <ScrubHistogram
            states={hist.states}
            breakAt={hist.breakAt}
            lanes={lanes}
            height={250}
            span={[0.15, 1]}
            label="kcal distribution with the cut values marked"
          />
        </StageSection>
        <RecordBar
          basis={EXCLUSIONS[0]!.preview.basis}
          action={active && recordable(active) ? `Record: ${EXCL_OPTIONS[active]!.label}` : null}
          onRecord={() => active && setRecorded(active)}
          recorded={!!recorded}
          onChange={() => setRecorded(null)}
        />
      </StageFrame>
    </div>
  );
}
