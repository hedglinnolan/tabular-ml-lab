/**
 * Keyboard traversal of a question's options (BLUEPRINT §11.1: one key per option, so
 * rifling is fast). Arrow keys move and stop at the ends; Home and End jump. Returns the
 * next index, or null when the key is not a movement.
 */
export function nextIndex(key: string, current: number, count: number): number | null {
  if (count <= 0) return null;
  const at = Math.max(0, Math.min(count - 1, current));
  switch (key) {
    case "ArrowDown":
    case "ArrowRight":
      return Math.min(count - 1, at + 1);
    case "ArrowUp":
    case "ArrowLeft":
      return Math.max(0, at - 1);
    case "Home":
    case "PageUp":
      return 0;
    case "End":
    case "PageDown":
      return count - 1;
    default:
      return null;
  }
}

/** What a key does on an option list, beyond moving. */
export type OptionKeyAction = "record" | "toggle" | "escape" | null;

/**
 * Enter records. Space chooses in a multi-select list (its ARIA meaning); in a single-choice
 * list it is the stage's: it flips the preview between your data now and with this choice
 * (M1_CONTRACT §11), so it never records.
 */
export function keyAction(key: string, mode: "single" | "multi"): OptionKeyAction {
  if (key === "Escape") return "escape";
  if (key === "Enter") return "record";
  if (key === " " || key === "Spacebar") return mode === "multi" ? "toggle" : null;
  return null;
}
