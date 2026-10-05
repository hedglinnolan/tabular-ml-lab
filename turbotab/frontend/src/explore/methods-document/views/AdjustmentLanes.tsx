/**
 * The adjustment set as a lineage of derived roles (MODELING_SEQUENCE §3: "lineage with derived
 * roles (confounder, mediator, collider lanes)"): each covariate on the left, linked to where its
 * answers send it — into the primary model, beside it in a declared secondary model, or out — so
 * the user sees what the guesses would change in the model before confirming them. A guess is a
 * dashed link (proposed, nothing recorded); an answered covariate is a solid one. Columns the pack
 * has no guess for are not linked at all: each is asked.
 *
 * Read from the server's adjustment card (proposals stage): groups, guesses, derived roles.
 */
import { Fragment } from "react";
import { useSize } from "../../../components/stage/views/geometry";
import type { AdjustmentCard } from "../fixture";
import s from "../doc.module.css";

type Lane = "confounder" | "timing_unknown" | "mediator" | "asked";

const LANES: { lane: Lane; title: string; note: string }[] = [
  { lane: "confounder", title: "Primary model", note: "confounders, adjusted for" },
  { lane: "timing_unknown", title: "Declared beside it", note: "timing unknown: with it" },
  { lane: "mediator", title: "Left out", note: "mediators of a total effect" },
  { lane: "asked", title: "Asked of you", note: "no guess: one by one" },
];

const ROW = 19;
const HEAD = 20;
const GAP = 8;

function laneOf(derived: string | null | undefined): Lane {
  if (derived === "confounder" || derived === "timing_unknown" || derived === "mediator") return derived;
  return "asked";
}

export function AdjustmentLanes({ card, outcome, focusGroup }: { card: AdjustmentCard; outcome: string; focusGroup: string | null }) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  // Left: each group, a heading then its columns.
  let y = 0;
  const left: { key: string; label?: string; column?: string; y: number; lane: Lane; group: string; proposed: boolean }[] = [];
  for (const g of card.groups) {
    left.push({ key: `h:${g.key}`, label: g.label, y, lane: laneOf(g.derived), group: g.key, proposed: true });
    y += HEAD;
    for (const c of g.columns) {
      const answered = card.answered?.[c];
      left.push({
        key: c,
        column: c,
        y,
        lane: laneOf(answered ? null : g.derived),
        group: g.key,
        proposed: !answered,
      });
      y += ROW;
    }
    y += GAP;
  }
  const height = y;
  // Right: one box per lane, spaced over the height.
  const boxH = 50;
  const slots = LANES.length;
  const boxY = (i: number) => (height - boxH) * (i / Math.max(1, slots - 1));
  const W = Math.max(420, w || 560);
  const leftW = 160;
  const boxW = Math.min(230, Math.round(W * 0.36));
  const boxX = W - boxW;
  return (
    <figure className={s.lanes} data-purpose="adjustment_lanes">
      <div className={s.lanesHead}>
        <code className="v">{card.exposure}</code>
        <span className={s.lanesArrow} aria-hidden="true">
          →
        </span>
        <code className="v">{outcome}</code>
        <span className={s.lanesEffect}>{card.effect === "direct" ? "direct effect" : "total effect"}</span>
      </div>
      <div ref={ref}>
      <svg width={W} height={height} viewBox={`0 0 ${W} ${height}`} className={s.lanesSvg} role="img" aria-label="Where each covariate's answers send it">
        {left
          .filter((n) => n.column && n.lane !== "asked")
          .map((n) => {
            const i = LANES.findIndex((l) => l.lane === n.lane);
            const ty = boxY(i) + boxH / 2;
            const sy = n.y + ROW / 2;
            const dim = focusGroup !== null && focusGroup !== n.group;
            return (
              <path
                key={`l:${n.key}`}
                d={`M ${leftW} ${sy} C ${leftW + 90} ${sy}, ${boxX - 90} ${ty}, ${boxX} ${ty}`}
                className={n.proposed ? s.laneLinkProposed : s.laneLink}
                data-lane={n.lane}
                data-dim={dim || undefined}
              />
            );
          })}
        {left.map((n) =>
          n.label ? (
            <text key={n.key} x={0} y={n.y + 15} className={s.laneGroup} data-dim={focusGroup !== null && focusGroup !== n.group ? true : undefined}>
              {n.label}
            </text>
          ) : (
            <Fragment key={n.key}>
              <rect
                x={0}
                y={n.y + 2}
                width={leftW}
                height={ROW - 4}
                rx={4}
                className={s.laneChip}
                data-asked={n.lane === "asked" || undefined}
                data-dim={focusGroup !== null && focusGroup !== n.group ? true : undefined}
              />
              <text x={8} y={n.y + 13.5} className={s.laneChipText}>
                {n.column}
              </text>
              {n.lane === "asked" ? (
                <text x={leftW - 8} y={n.y + 13.5} textAnchor="end" className={s.laneAsk}>
                  ?
                </text>
              ) : null}
            </Fragment>
          ),
        )}
        {LANES.map((l, i) => {
          const count = left.filter((n) => n.column && n.lane === l.lane).length;
          return (
            <g key={l.lane} transform={`translate(${boxX} ${boxY(i)})`}>
              <rect width={boxW} height={boxH} rx={8} className={s.laneBox} data-lane={l.lane} />
              <text x={12} y={20} className={s.laneBoxTitle}>
                {l.title}
              </text>
              <text x={boxW - 12} y={20} textAnchor="end" className={s.laneBoxCount}>
                {count}
              </text>
              <text x={12} y={37} className={s.laneBoxNote}>
                {l.note}
              </text>
            </g>
          );
        })}
      </svg>
      </div>
      <figcaption className={s.lanesCaption}>
        Dashed: the pack&rsquo;s guess, nothing recorded. The role is derived from three answers per covariate, never
        from the data ({card.source}).
      </figcaption>
    </figure>
  );
}
