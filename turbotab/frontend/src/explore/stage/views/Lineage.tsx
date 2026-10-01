/**
 * lineage — raw columns → adjusted → the model matrix.
 *
 * Nodes keep their identity across options (see lineageLayout.ts), so switching the
 * energy method relabels `protein_adj` to `protein_per_kcal` in place, and the links from
 * `kcal` re-route: into every nutrient (residual, density), into `kcal_from_other`
 * (partition), or straight through (standard). A wide table arrives already collapsed
 * into group nodes with counts, so the picture has the same size at 11 or 20,000 columns.
 */
import { useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";
import { fmtInt } from "../format";
import type { Lineage as LineageData } from "../types";
import { placeLineage, type PlacedNode } from "./lineageLayout";
import { useSize } from "./hooks";
import s from "./views.module.css";
import { useStageTransitions } from "../motion";

interface Props {
  lineage: LineageData;
  compact?: boolean;
  /** Raw columns to pick out (a finding's subject). */
  emphasis?: string[];
  /** The open question acts on the adjusted lane (live pipeline only). */
  openLabel?: string;
}

const HEAD = 26;

function label(n: PlacedNode): string {
  return n.node.count > 1 ? n.node.label : (n.node.column ?? n.node.label);
}

export function Lineage({ lineage, compact: small = false, openLabel, emphasis }: Props) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const t = useStageTransitions();
  const placed = useMemo(() => placeLineage(lineage), [lineage]);
  // A thumbnail given a lot of room draws like the large view.
  const compact = small && !(h - HEAD > placed.rows * 22 && w > 560);
  const picked = new Set(emphasis ?? []);

  const fs = compact ? 9.5 : 11;
  const cw = fs * 0.61;
  const rawW =
    Math.max(...placed.nodes.filter((n) => n.lane === "raw").map((n) => label(n).length)) * cw + 10;
  const mxW = Math.max(
    ...placed.nodes.filter((n) => n.lane === "matrix").map((n) => label(n).length),
    ...placed.notes.map((n) => n.text.length),
  );
  const mxLabelW = mxW * cw + 12;
  const cap = compact ? (placed.rows <= 8 ? 21 : 16) : 28;
  const rowH = Math.max(10, Math.min(cap, (h - HEAD - (openLabel ? 30 : 6)) / Math.max(placed.rows, 1)));
  const x0 = rawW + 4;
  const x2 = Math.max(x0 + 120, w - mxLabelW);
  const x1 = (x0 + x2) / 2;
  const adjW = compact ? 0 : Math.min(x2 - x1 - 40, Math.max(...placed.nodes.filter((n) => n.lane === "adjusted").map((n) => label(n).length)) * cw + 16);
  const top = HEAD + (compact ? 2 : 4);
  const yOf = (row: number) => top + row * rowH + rowH / 2;
  const xOf = (lane: string) => (lane === "raw" ? x0 : lane === "adjusted" ? x1 : x2);
  const edgeOut = (n: PlacedNode) => (n.lane === "adjusted" && adjW ? x1 + adjW / 2 : xOf(n.lane));
  const edgeIn = (n: PlacedNode) => (n.lane === "adjusted" && adjW ? x1 - adjW / 2 : xOf(n.lane));
  const ready = w > 80 && h > 60;

  const path = (a: PlacedNode, b: PlacedNode) => {
    const xa = edgeOut(a);
    const xb = edgeIn(b);
    const ya = yOf(a.row);
    const yb = yOf(b.row);
    const m = (xa + xb) / 2;
    return `M${xa.toFixed(1)},${ya.toFixed(1)}C${m.toFixed(1)},${ya.toFixed(1)} ${m.toFixed(1)},${yb.toFixed(1)} ${xb.toFixed(1)},${yb.toFixed(1)}`;
  };

  return (
    <div ref={ref} className={s.fill} data-view="lineage">
      {ready ? (
        <svg width={w} height={h} className={s.svg} role="img" aria-label="Column lineage">
          <g className={s.laneHeads}>
            <text x={x0} y={12} textAnchor="end">
              RAW <tspan className={s.laneCount}>{fmtInt(placed.counts.raw)}</tspan>
            </text>
            <text x={x1} y={12} textAnchor="middle">
              ADJUSTED
            </text>
            <text x={x2} y={12} textAnchor="start">
              MATRIX <tspan className={s.laneCount}>{fmtInt(placed.counts.matrix)}</tspan>
            </text>
          </g>

          {openLabel ? (
            <g className={s.openLane}>
              <rect
                x={x1 - (adjW || 26) / 2 - 10}
                y={HEAD - 8}
                width={(adjW || 26) + 20}
                height={placed.rows * rowH + 14}
                rx={9}
              />
            </g>
          ) : null}

          <AnimatePresence initial={false}>
            {placed.links.map((k) => (
              <motion.path
                key={k.key}
                className={k.rewrite ? s.linkRewrite : s.link}
                initial={{ opacity: 0, pathLength: 0, d: path(k.from, k.to) }}
                animate={{ opacity: 1, pathLength: 1, d: path(k.from, k.to) }}
                exit={{ opacity: 0 }}
                transition={t.arrive}
              />
            ))}
          </AnimatePresence>

          <AnimatePresence initial={false}>
            {placed.nodes.map((n) => {
              const y = yOf(n.row);
              const x = xOf(n.lane);
              const group = n.node.count > 1;
              const emph =
                n.changed ||
                picked.has(n.identity) ||
                (n.lane === "matrix" && placed.nodes.some((m) => m.changed && m.identity === n.identity));
              return (
                <motion.g
                  key={n.key}
                  initial={{ opacity: 0, x, y }}
                  animate={{ opacity: 1, x, y }}
                  exit={{ opacity: 0 }}
                  transition={t.arrive}
                  className={emph ? s.nodeChanged : n.node.role === "identifier" ? s.nodeMuted : s.node}
                >
                  {n.lane === "adjusted" && adjW ? (
                    <>
                      {group ? (
                        <rect x={-adjW / 2 + 3} y={-rowH / 2 + 1} width={adjW} height={rowH - 4} rx={5} className={s.pillDeck} />
                      ) : null}
                      <rect x={-adjW / 2} y={-rowH / 2 + 3} width={adjW} height={rowH - 6} rx={5} className={s.pill} />
                      <text x={0} y={0} dy="0.34em" textAnchor="middle" style={{ fontSize: fs }}>
                        {label(n)}
                      </text>
                    </>
                  ) : (
                    <>
                      <circle r={group ? 4 : 2.6} className={s.dot} />
                      {n.lane === "raw" ? (
                        <text x={-7} dy="0.34em" textAnchor="end" style={{ fontSize: fs }}>
                          {label(n)}
                        </text>
                      ) : n.lane === "matrix" ? (
                        <text x={7} dy="0.34em" textAnchor="start" style={{ fontSize: fs }}>
                          {label(n)}
                        </text>
                      ) : null}
                    </>
                  )}
                </motion.g>
              );
            })}
          </AnimatePresence>

          <AnimatePresence initial={false}>
            {placed.notes.map((note) => (
              <motion.text
                key={`note-${note.identity}`}
                x={x2 + 7}
                y={yOf(note.row)}
                dy="0.34em"
                className={s.laneNote}
                style={{ fontSize: fs }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={t.arrive}
              >
                {note.text}
              </motion.text>
            ))}
          </AnimatePresence>

          {openLabel ? (
            <text x={x1} y={placed.rows * rowH + HEAD + 20} textAnchor="middle" className={s.openLabel}>
              {openLabel}
            </text>
          ) : null}
        </svg>
      ) : null}
    </div>
  );
}
