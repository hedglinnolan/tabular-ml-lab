/**
 * The kit's small parts, props-driven: "Why does this matter?", the card's one place to act (Foot,
 * from Direction A's quest log, design/quest-meaning-first cards.tsx) and the pointing grace. Each
 * keeps the calm kit's CSS (kit.module.css); a part renders inside a `data-kit` region, where the
 * kit's tokens apply (tokens.css).
 */
import type { ReactNode } from "react";
import k from "./kit.module.css";
import f from "./foot.module.css";

/** How long a pointed line or stage keeps its panel after the pointer leaves it, so crossing the
 *  gap to the next one does not flicker (src/explore/calm-kit/parts.tsx, Nolan 2026-10-05). */
export const POINT_GRACE_MS = 160;

/** Engine text with its backticks dropped: the calm budget prints column names as plain text. */
export const plain = (s: string | null | undefined): string => (s ?? "").replaceAll("`", "");

export function Plain({ text }: { text: string | null | undefined }) {
  return <>{plain(text)}</>;
}

/** Teaching behind one disclosure (FOUNDATION §2). */
export function Why({ text }: { text: string }) {
  return (
    <details className={k.why}>
      <summary>Why does this matter?</summary>
      <p>
        <Plain text={text} />
      </p>
    </details>
  );
}

/** The one primary action, always at the foot of the card (FOUNDATION §1), with a quiet line on
 *  its left: what the press waits for, or what the last one did. */
export function Foot({
  label,
  onClick,
  disabled,
  left,
  testId = "primary",
}: {
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  left?: ReactNode;
  testId?: string;
}) {
  return (
    <div className={`${k.actRow} ${f.actRow}`} data-kit>
      <span className={k.hint}>{left ?? ""}</span>
      <button type="button" className={k.primary} disabled={disabled} onClick={onClick} data-testid={testId}>
        {label}
      </button>
    </div>
  );
}
