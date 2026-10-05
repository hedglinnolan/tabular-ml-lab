/**
 * Engine text in the calm voice: backticks mark data values in the engine's sentences; the calm
 * budget prints column names as plain text, never boxed chips (FOUNDATION §2).
 */
export { fmtInt, fmtNum, fmtR, fmtTick } from "../../components/stage/format";
export { fmtCI, fmtEst } from "../methods-shared/results";

export const plain = (s: string | null | undefined): string => (s ?? "").replaceAll("`", "");

/** Engine text with its backticks dropped. */
export function Plain({ text }: { text: string | null | undefined }) {
  return <>{plain(text)}</>;
}
