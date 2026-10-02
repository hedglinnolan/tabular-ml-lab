/**
 * The recorded seal under the live row flow: its glyph and its basis, named (M2_CONTRACT §3: the
 * basis is rendered on the seal, in three states, never two). An exploratory basis carries its
 * label; it is never drawn as a clean lock.
 */
import type { SplitArtifact } from "../../../api/m1-stage-types";
import { Rich } from "../text";
import { glyphOf, isExploratory } from "./phase";
import { SealGlyph } from "./SealGlyph";
import s from "./seal.module.css";

export function LiveSeal({ split }: { split: SplitArtifact }) {
  if (!split.n_holdout) return null;
  const state = split.basis?.state;
  const exploratory = isExploratory(state, split.exploratory);
  const chron = split.chronology?.drawn ? `chronological by \`${split.chronology.time_column}\`, ` : "";
  return (
    <p className={s.liveSeal} data-exploratory={exploratory || undefined} data-testid="live-seal" data-purpose="seal_basis" data-basis={state ?? "unknown"}>
      <SealGlyph state={glyphOf(state)} recorded size={16} />
      <span className={s.basisKicker}>{exploratory ? "Sealed by row · exploratory" : "Sealed"}</span>
      <span>
        <Rich text={`${chron}${split.basis?.label ?? "undetermined"}`} />
      </span>
    </p>
  );
}
