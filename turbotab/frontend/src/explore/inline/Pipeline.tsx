/**
 * The pipeline panel, structure only: Rows (the participant flow) and Columns (raw → model
 * matrix). Values and relationships live in the question's stage; this panel answers "what
 * would this option touch?" by marking the steps and tracks it changes. A preview is marked in
 * the accent (now) and says nothing was recorded; a recorded answer wears the green keyline.
 */
import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { NumberTween } from "../../motion/NumberTween";
import { useTransitions } from "../../motion/prefs";
import { Prose } from "../../components/Prose";
import { fmtInt } from "./format";
import type { Track } from "./lineage";
import s from "./inline.module.css";

export type PanelStatus = "idle" | "preview" | "recorded";

export interface FlowStep {
  key: string;
  label: string;
  n: number | null;
  dropped?: number;
  /** "touched": the previewed option changes this step. */
  mark?: "touched" | "open" | "waiting" | "none";
  note?: string;
}

export interface SplitNode {
  train: number;
  holdout: number;
  fitHere: boolean;
}

export function Panel({
  status,
  statusText,
  children,
}: {
  status: PanelStatus;
  statusText?: string;
  children: ReactNode;
}) {
  return (
    <aside className={s.panel} aria-label="Pipeline" data-status={status}>
      <div className={s.panelHead}>
        <span className={s.panelTitle}>Pipeline</span>
        {status !== "idle" ? (
          <span className={s.panelStatus} data-status={status}>
            {statusText ?? (status === "preview" ? "preview · nothing recorded" : "recorded")}
          </span>
        ) : null}
      </div>
      {children}
    </aside>
  );
}

export function RowsSection({
  steps,
  split,
  status,
}: {
  steps: FlowStep[];
  split?: SplitNode | { waiting: true } | null;
  status: PanelStatus;
}) {
  return (
    <section className={s.section} aria-label="Rows">
      <div className={s.sectionHead}>
        <span className="kicker">Rows</span>
      </div>
      <ol className={s.flow}>
        {steps.map((st) => (
          <li key={st.key} className={s.step} data-mark={st.mark ?? "none"} data-status={status}>
            <span className={s.stepDot} aria-hidden="true" />
            <span className={s.stepLabel}>
              <Prose text={st.label} />
            </span>
            {st.note ? <span className={s.stepNote}>{st.note}</span> : null}
            {st.dropped ? (
              <span className={s.stepDrop}>
                −<NumberTween value={st.dropped} format={fmtInt} />
              </span>
            ) : null}
            {st.n !== null ? (
              <NumberTween value={st.n} format={fmtInt} className={s.stepN} />
            ) : (
              <span className={s.stepN} />
            )}
          </li>
        ))}
      </ol>
      {split && "waiting" in split ? (
        <div className={s.splitWaiting}>Split · waiting</div>
      ) : split ? (
        <div className={s.split}>
          <div className={s.splitNode} data-fit={split.fitHere} data-status={status}>
            <span className={s.splitLabel}>Training</span>
            <span className={s.splitN}>{fmtInt(split.train)}</span>
            {split.fitHere ? <span className={s.fitTag}>fit here</span> : null}
          </div>
          <div className={s.splitNode} data-sealed="true">
            <span className={s.splitLabel}>Held out</span>
            <span className={s.splitN}>{fmtInt(split.holdout)}</span>
            <span className={s.sealTag}>sealed</span>
          </div>
        </div>
      ) : null}
    </section>
  );
}

const ROW_H = 22;
const RAW_W = 112;
const LINK_W = 60;

function isAdjusting(op: string) {
  return op !== "kept" && op !== "none" && op !== "feeds" && op !== "pass-through";
}

export function ColumnsSection({
  tracks,
  touchedIds,
  status,
  summary,
  untouchedNote,
}: {
  tracks: Track[];
  touchedIds: Set<string>;
  status: PanelStatus;
  summary?: string;
  /** When the option touches no column: one line instead of the full lineage. */
  untouchedNote?: string;
}) {
  const t = useTransitions();
  const index = new Map(tracks.map((tr, i) => [tr.id, i]));
  const height = tracks.length * ROW_H;
  const fans: { from: number; to: number; key: string }[] = [];
  tracks.forEach((tr, i) => {
    for (const inp of tr.inputs) {
      const j = index.get(inp);
      if (j !== undefined) fans.push({ from: j, to: i, key: `${inp}->${tr.id}` });
    }
  });

  return (
    <section className={s.section} aria-label="Columns">
      <div className={s.sectionHead}>
        <span className="kicker">Columns</span>
        {summary ? (
          <span className={s.sectionSum} data-status={status}>
            {summary}
          </span>
        ) : null}
      </div>
      {untouchedNote ? (
        <p className={s.untouched}>{untouchedNote}</p>
      ) : (
        <div className={s.lineage} style={{ height }}>
          <svg
            className={s.lineageLinks}
            width={LINK_W}
            height={height}
            style={{ left: RAW_W }}
            aria-hidden="true"
          >
            {tracks.map((tr, i) => {
              const y = i * ROW_H + ROW_H / 2;
              const on = touchedIds.has(tr.id);
              const adj = isAdjusting(tr.op);
              if (!tr.outs.length) {
                return (
                  <line
                    key={tr.id}
                    x1={2}
                    x2={tr.op === "feeds" ? 10 : 18}
                    y1={y}
                    y2={y}
                    className={s.link}
                    data-on={on}
                  />
                );
              }
              return (
                <g key={tr.id}>
                  <line
                    x1={2}
                    x2={LINK_W - 4}
                    y1={y}
                    y2={y}
                    className={adj ? s.linkAdj : s.link}
                    data-on={on}
                    data-status={status}
                  />
                  {adj ? (
                    <circle
                      cx={LINK_W / 2}
                      cy={y}
                      r={2.6}
                      className={s.linkDot}
                      data-status={status}
                    />
                  ) : null}
                </g>
              );
            })}
            <AnimatePresence initial={false}>
              {fans.map((f) => {
                const y0 = f.from * ROW_H + ROW_H / 2;
                const y1 = f.to * ROW_H + ROW_H / 2;
                return (
                  <motion.path
                    key={f.key}
                    d={`M2 ${y0} C ${LINK_W * 0.35} ${y0}, ${LINK_W * 0.2} ${y1}, ${LINK_W / 2} ${y1}`}
                    className={s.fan}
                    data-status={status}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={t.arrive}
                  />
                );
              })}
            </AnimatePresence>
          </svg>
          <ul className={s.tracks}>
            {tracks.map((tr) => {
              const on = touchedIds.has(tr.id);
              const outText = tr.outs.length
                ? tr.outs.length > 1
                  ? `${tr.outs[0]} +${tr.outs.length - 1}`
                  : tr.outs[0]!
                : "";
              return (
                <li
                  key={tr.id}
                  className={s.track}
                  data-on={on}
                  data-status={status}
                  title={tr.formula ?? undefined}
                >
                  <span className={s.trackRaw} style={{ width: RAW_W }} title={tr.raw}>
                    {tr.count > 1 ? (
                      <>
                        {/* a collapsed group: its kind, then how many columns it stands for */}
                        {tr.raw.replace(/^[\d,]+\s+/, "").replace(/\s+columns?$/, "")}
                        <span className={s.count}>×{fmtInt(tr.count)}</span>
                      </>
                    ) : (
                      tr.raw
                    )}
                  </span>
                  <span className={s.trackOut} style={{ marginLeft: LINK_W }}>
                    {/* The track is the same column; its new name fades in where the old one was. */}
                    <AnimatePresence initial={false}>
                      <motion.span
                        key={outText || `gone-${tr.op}`}
                        className={outText ? s.outName : s.outGone}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={t.arrive}
                      >
                        {outText || (tr.op === "none" ? "not in model" : "leaves")}
                      </motion.span>
                    </AnimatePresence>
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}

export function RoleSection({
  groups,
  highlight,
}: {
  groups: { role: string; columns: string[] }[];
  highlight: Set<string>;
}) {
  return (
    <section className={s.section} aria-label="Columns">
      <div className={s.sectionHead}>
        <span className="kicker">Columns</span>
        <span className={s.sectionSum}>proposed roles</span>
      </div>
      <ul className={s.roles}>
        {groups.map((g) => {
          const lit = g.columns.filter((c) => highlight.has(c));
          const rest = g.columns.length - lit.length;
          return (
            <li key={g.role} className={s.roleRow} data-on={lit.length > 0}>
              <span className={s.roleName}>{g.role}</span>
              <span className={s.roleCols}>
                {lit.map((c) => (
                  <span key={c} className={s.roleCol}>
                    {c}
                  </span>
                ))}
                {rest > 0 && g.columns.length === 1 ? (
                  <span className={s.roleOne}>{g.columns[0]}</span>
                ) : rest > 0 ? (
                  <span className={s.roleMore}>{lit.length ? `+${rest}` : `${rest}`}</span>
                ) : null}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
