/**
 * The seal's glyph, lifted from the M2 prototype (src/explore/m2/views/SealGlyph.tsx), in the
 * three shapes its basis can take (lockbox constitution §03). Only a grouped seal, or a
 * verified one row per unit, is drawn closed. Grouping abandoned is a ring with a gap;
 * undetermined is a dashed ring around a question mark — never a clean lock. A recorded seal wears
 * --ok (sealed is a recorded claim); a preview wears ink; an exploratory basis wears the coach's
 * amber, as its label.
 */
import type { SealBasisState } from "../../../api/m2-types";
import s from "./SealGlyph.module.css";

export type SealShape = "grouped" | "abandoned" | "undetermined";

export function sealShape(state: SealBasisState | null | undefined): SealShape {
  if (state === "grouped" || state === "one_row_per_unit") return "grouped";
  if (state === "abandoned") return "abandoned";
  return "undetermined";
}

const LABEL: Record<SealShape, string> = {
  grouped: "Sealed",
  abandoned: "Sealed by row: grouping abandoned",
  undetermined: "Sealed by row: basis undetermined",
};

export function SealGlyph({
  state,
  recorded,
  size = 18,
  className,
}: {
  state: SealBasisState | null | undefined;
  recorded: boolean;
  size?: number;
  className?: string;
}) {
  const shape = sealShape(state);
  const tone = shape === "grouped" ? (recorded ? "ok" : "ink") : "warn";
  const r = 9;
  return (
    <svg
      width={size}
      height={size}
      viewBox="-12 -12 24 24"
      className={[s.glyph, className].filter(Boolean).join(" ")}
      data-seal={shape}
      data-tone={tone}
      role="img"
      aria-label={LABEL[shape]}
    >
      {shape === "grouped" ? (
        <>
          <circle r={r} className={s.ring} />
          <circle r={5.2} className={s.core} />
          <path d="M-2.6,0.2 L-0.6,2.2 L2.8,-1.8" className={s.mark} />
        </>
      ) : shape === "abandoned" ? (
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
