/**
 * The seal's glyph, in the three states its basis can take (ROADMAP lockbox constitution §03).
 * Only a grouped seal is drawn closed. Grouping abandoned is a ring with a gap; undetermined is a
 * dashed ring around a question mark — never a clean lock. Recorded seals wear --ok (sealed is a
 * recorded claim); a preview wears ink; an exploratory basis wears the coach's amber, as its label.
 */
export type SealState = "grouped" | "abandoned" | "undetermined";

export function sealStateOf(basis: string): SealState {
  if (basis === "grouped" || basis === "cross_sectional") return "grouped";
  if (basis === "repetition_found_grouping_abandoned") return "abandoned";
  return "undetermined";
}

export function SealGlyph({
  state,
  recorded,
  size = 22,
  className,
}: {
  state: SealState;
  recorded: boolean;
  size?: number;
  className?: string;
}) {
  const tone = state === "grouped" ? (recorded ? "ok" : "ink") : "warn";
  const r = 9;
  return (
    <svg
      width={size}
      height={size}
      viewBox="-12 -12 24 24"
      className={className}
      data-seal={state}
      data-tone={tone}
      role="img"
      aria-label={
        state === "grouped"
          ? "Sealed"
          : state === "abandoned"
            ? "Sealed by row: grouping abandoned"
            : "Sealed by row: basis undetermined"
      }
    >
      {state === "grouped" ? (
        <>
          <circle r={r} className="sealRing" />
          <circle r={5.2} className="sealCore" />
          <path d="M-2.6,0.2 L-0.6,2.2 L2.8,-1.8" className="sealMark" />
        </>
      ) : state === "abandoned" ? (
        <>
          <path d={`M${r * Math.cos(-1.2)},${r * Math.sin(-1.2)} A${r},${r} 0 1,1 ${r * Math.cos(-0.35)},${r * Math.sin(-0.35)}`} className="sealRing" />
          <circle r={5.2} className="sealCoreOpen" />
        </>
      ) : (
        <>
          <circle r={r} className="sealRing" strokeDasharray="2.6 2.2" />
          <text y={3.6} textAnchor="middle" className="sealQ">
            ?
          </text>
        </>
      )}
    </svg>
  );
}
