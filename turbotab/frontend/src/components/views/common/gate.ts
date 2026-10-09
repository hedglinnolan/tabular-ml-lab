/**
 * The outcome's gate, enforced by every view that could read it (FOUNDATION §5 rule 6): before its
 * gate opens, the outcome beside another column stays out of every view. A view's input declares
 * the outcome and whether its gate is open, and its layout refuses, in one line, to draw anything
 * that reads the outcome while the gate is shut. The caller cannot leave it unsaid: the field is
 * required where the view reads columns the user chose (embedding, matrix).
 */

export interface OutcomeGate {
  /** the outcome's column, as the user named it */
  name: string;
  /**
   * whether the outcome beside another column has opened: after the lock under Estimate and
   * Describe, after the held-out draw under Predict, the strictest gate under several goals
   */
  gate_open: boolean;
}

/**
 * The one line a view shows instead of drawing, when any column it reads is the outcome and its
 * gate is shut; null when the view may draw. `outcome: null` means no outcome is named yet, when
 * every column is a column (§5 rule 6, before the outcome is named).
 */
export function gateRefusal(outcome: OutcomeGate | null | undefined, reads: readonly (string | null | undefined)[]): string | null {
  if (!outcome || outcome.gate_open) return null;
  if (!reads.includes(outcome.name)) return null;
  return `${outcome.name} is the outcome, and the outcome beside other columns waits for its gate, so this view is not drawn yet.`;
}
