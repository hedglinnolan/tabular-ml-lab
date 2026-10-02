/**
 * The seal's glyph (lifted from /lab/m2), in the three states its basis can take (ROADMAP lockbox
 * constitution §03). Only a verified basis is drawn closed. Grouping abandoned is a ring with a gap;
 * undetermined is a dashed ring around a question mark — never a clean lock. A recorded seal wears
 * --ok (sealed is a recorded claim); a preview wears ink; an exploratory basis wears the coach's
 * amber, as its label.
 */
import type { GlyphState } from "./phase";
import s from "./seal.module.css";

export function SealGlyph({
  state,
  recorded,
  size = 22,
  className,
}: {
  state: GlyphState;
  recorded: boolean;
  size?: number;
  className?: string;
}) {
  const tone = state === "closed" ? (recorded ? "ok" : "ink") : "warn";
  const r = 9;
  return (
    <svg
      width={size}
      height={size}
      viewBox="-12 -12 24 24"
      className={className ? `${s.glyph} ${className}` : s.glyph}
      data-seal={state}
      data-tone={tone}
      role="img"
      aria-label={
        state === "closed"
          ? recorded
            ? "Sealed"
            : "Would seal"
          : state === "abandoned"
            ? "Sealed by row: grouping abandoned"
            : "Sealed by row: basis undetermined"
      }
    >
      {state === "closed" ? (
        <>
          <circle r={r} className={s.ring} />
          <circle r={5.2} className={s.core} />
          <path d="M-2.6,0.2 L-0.6,2.2 L2.8,-1.8" className={s.mark} />
        </>
      ) : state === "abandoned" ? (
        <>
          <path
            d={`M${r * Math.cos(-1.2)},${r * Math.sin(-1.2)} A${r},${r} 0 1,1 ${r * Math.cos(-0.35)},${r * Math.sin(-0.35)}`}
            className={s.ring}
          />
          <circle r={5.2} className={s.coreOpen} />
        </>
      ) : (
        <>
          <circle r={r} className={s.ring} strokeDasharray="2.6 2.2" />
          <text y={3.6} textAnchor="middle" className={s.q}>
            ?
          </text>
        </>
      )}
    </svg>
  );
}
