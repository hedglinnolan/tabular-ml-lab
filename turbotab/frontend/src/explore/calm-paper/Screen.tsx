/**
 * The paper (#/paper): a placeholder until its structure agent replaces this file. It shows the
 * kit's reference wiring (../calm-kit/ReferenceScreen.tsx) so the walker contract
 * (e2e/calm-protos/paper.ts) and the cross-check of the four Table 2s run from the start. Build it
 * from ../calm-kit only: the only difference between the four structures is structure.
 */
import { ReferenceScreen } from "../calm-kit/ReferenceScreen";

/** Marks this file as the placeholder; the chooser reads it. Delete it with the placeholder. */
export const PLACEHOLDER = true;

export function Screen() {
  return <ReferenceScreen name="The shared reference walk (this structure is not built yet)" />;
}
