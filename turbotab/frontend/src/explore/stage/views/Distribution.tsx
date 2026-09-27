/**
 * distribution — one column's values before and after, with any cut values marked.
 *
 * Three drawings of the same view kind, chosen by what changed:
 *   overlay  the same column loses rows: kept bars, the removed part hatched on top
 *   stacked  the column itself is rewritten (a new unit): recorded above, after below
 *   levels   a true/false or categorical column: one horizontal bar per level
 * Bars keep their identity (bin i, level i) across options, so they grow and shrink.
 */
import { useId, useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";
import { scaleLinear } from "d3-scale";
import { max } from "d3-array";
import { clean, fmtInt, fmtTick } from "../format";
import type { DistributionView, HistogramData, Mark } from "../types";
import { useSize } from "./hooks";
import s from "./views.module.css";
import { useStageTransitions } from "../motion";

interface Props {
  view: DistributionView;
  stacked: boolean;
  compact?: boolean;
  evidence?: boolean;
}

const GROUP_NAME: Record<string, string> = { female: "women", male: "men" };

function markClass(m: Mark) {
  if (m.group === "female") return s.markA;
  if (m.group === "male") return s.markB;
  return s.markAll;
}

function Bars({
  hist,
  box,
  yMax,
  className,
  removed,
  pattern,
}: {
  hist: HistogramData;
  box: { x0: number; x1: number; y0: number; y1: number };
  yMax: number;
  className: string | undefined;
  removed?: number[];
  pattern?: string;
}) {
  const t = useStageTransitions();
  const n = hist.counts.length;
  const bw = (box.x1 - box.x0) / n;
  const y = scaleLinear().domain([0, yMax]).range([box.y1, box.y0]);
  return (
    <g>
      {hist.counts.map((c, i) => {
        const top = c > 0 ? Math.min(y(c), box.y1 - 1.5) : y(c);
        const gone = removed?.[i] ?? 0;
        const topGone = gone > 0 ? Math.min(y(c + gone), top - 1.5) : top;
        return (
          <g key={i}>
            <motion.rect
              x={box.x0 + i * bw + 0.5}
              width={Math.max(0, bw - 1)}
              initial={false}
              animate={{ y: top, height: Math.max(0, box.y1 - top) }}
              transition={t.arrive}
              className={className}
            />
            {removed ? (
              <motion.rect
                x={box.x0 + i * bw + 0.5}
                width={Math.max(0, bw - 1)}
                initial={false}
                animate={{ y: topGone, height: Math.max(0, top - topGone) }}
                transition={t.arrive}
                fill={`url(#${pattern})`}
                className={s.removed}
              />
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function XTicks({
  lo,
  hi,
  box,
  count,
}: {
  lo: number;
  hi: number;
  box: { x0: number; x1: number; y1: number };
  count: number;
}) {
  const x = scaleLinear().domain([lo, hi]).range([box.x0, box.x1]);
  const ticks = x.ticks(count).filter((v) => v >= lo && v <= hi);
  return (
    <g className={s.axis}>
      <line x1={box.x0} x2={box.x1} y1={box.y1} y2={box.y1} className={s.baseline} />
      {ticks.map((v) => (
        <text key={v} x={x(v)} y={box.y1 + 13} textAnchor="middle">
          {fmtTick(v)}
        </text>
      ))}
    </g>
  );
}

function Levels({ view, compact }: { view: DistributionView; compact: boolean }) {
  const t = useStageTransitions();
  const levels = view.levels ?? [];
  const rows = levels.map((l, i) => ({ label: l, n: view.before.counts[i] ?? 0, blank: false }));
  if (view.before.n_missing > 0) rows.push({ label: "blank", n: view.before.n_missing, blank: true });
  const total = rows.reduce((a, r) => a + r.n, 0);
  const top = max(rows, (r) => r.n) ?? 1;
  return (
    <div className={compact ? s.levelsCompact : s.levels} role="img" aria-label={view.title}>
      {rows.map((r, i) => (
        <div key={i} className={s.levelRow}>
          <span className={s.levelLabel}>{r.label}</span>
          <span className={s.levelTrack}>
            <motion.span
              className={r.blank ? s.levelBlank : s.levelBar}
              initial={false}
              animate={{ width: `${(100 * r.n) / top}%` }}
              transition={t.arrive}
            />
          </span>
          <span className={s.levelCount}>
            {fmtInt(r.n)}
            <span className={s.levelPct}>{((100 * r.n) / total).toFixed(1)}%</span>
          </span>
        </div>
      ))}
    </div>
  );
}

export function Distribution({ view, stacked, compact = false, evidence = false }: Props) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const t = useStageTransitions();
  const pid = useId().replace(/:/g, "");
  const pattern = `hatch-${pid}`;

  const geo = useMemo(() => {
    if (w < 60 || h < 50) return null;
    const pad = { l: compact ? 8 : 12, r: compact ? 8 : 12, t: compact ? 30 : 40, b: 20 };
    return { pad };
  }, [w, h, compact]);

  if (view.levels) {
    return (
      <div ref={ref} className={s.fill} data-view="distribution">
        <Levels view={view} compact={compact} />
      </div>
    );
  }

  const b = view.before;
  const a = view.after;
  const lo = clean(b.edges[0]!);
  const hi = b.edges[b.edges.length - 1]!;

  let body = null;
  if (geo && !stacked) {
    const { pad } = geo;
    const box = { x0: pad.l, x1: w - pad.r, y0: pad.t, y1: h - pad.b };
    const removed = b.counts.map((c, i) => Math.max(0, c - (a.counts[i] ?? 0)));
    const yMax = max(b.counts) ?? 1;
    const x = scaleLinear().domain([lo, hi]).range([box.x0, box.x1]);
    body = (
      <svg width={w} height={h} className={s.svg} role="img" aria-label={view.title}>
        <defs>
          <pattern id={pattern} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" className={s.hatchLine} />
          </pattern>
        </defs>
        <Bars
          hist={a}
          box={box}
          yMax={yMax}
          className={evidence ? s.barBefore : s.barAfter}
          removed={removed}
          pattern={pattern}
        />
        <XTicks lo={lo} hi={hi} box={box} count={compact ? 4 : 7} />
        <AnimatePresence initial={false}>
          {view.marks.map((m) => {
            const row = m.group === "male" ? 1 : 0;
            const mx = x(m.value);
            const name = m.group ? `${GROUP_NAME[m.group] ?? m.group} ` : "";
            const label = compact ? `${name}${fmtInt(m.value)}` : `${name}${m.label}`;
            const ly = (compact ? 10 : 13) + row * (compact ? 11 : 14);
            return (
              <motion.g
                key={`${m.group}|${m.value}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={t.arrive}
                className={markClass(m)}
              >
                <line x1={mx} x2={mx} y1={ly + 3} y2={box.y1} className={s.markLine} />
                <text x={mx + 3} y={ly} className={s.markText}>
                  {label}
                </text>
              </motion.g>
            );
          })}
        </AnimatePresence>
      </svg>
    );
  } else if (geo && stacked) {
    const { pad } = geo;
    const gap = compact ? 38 : 46;
    const t0 = compact ? 16 : 22;
    const ph = (h - t0 - gap - pad.b) / 2;
    const top = { x0: pad.l, x1: w - pad.r, y0: t0, y1: t0 + ph };
    const bot = { x0: pad.l, x1: w - pad.r, y0: top.y1 + gap, y1: top.y1 + gap + ph };
    const alo = clean(a.edges[0]!);
    const ahi = a.edges[a.edges.length - 1]!;
    body = (
      <svg width={w} height={h} className={s.svg} role="img" aria-label={view.title}>
        <text x={top.x0} y={top.y0 - 6} className={s.stackLabelBefore}>
          {view.before_label}
        </text>
        <Bars hist={b} box={top} yMax={max(b.counts) ?? 1} className={s.barBefore} />
        <XTicks lo={lo} hi={hi} box={top} count={compact ? 3 : 5} />
        <AnimatePresence initial={false}>
          <motion.g
            key={`${view.after_label}|${alo}|${ahi}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0 } }}
            transition={t.arrive}
          >
            <text x={bot.x0} y={bot.y0 - 6} className={s.stackLabelAfter}>
              {view.after_label}
            </text>
            <XTicks lo={alo} hi={ahi} box={bot} count={compact ? 3 : 5} />
          </motion.g>
        </AnimatePresence>
        <Bars hist={a} box={bot} yMax={max(a.counts) ?? 1} className={evidence ? s.barBefore : s.barAfter} />
      </svg>
    );
  }

  return (
    <div ref={ref} className={s.fill} data-view="distribution">
      {body}
    </div>
  );
}
