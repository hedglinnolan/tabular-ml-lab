import { plain } from "../kit/parts";

/** The engine's running-text name ("the lens question") as a line's own heading, its data values
 *  in plain text (the calm budget boxes no column names: backticks dropped). */
export function said(name: string): string {
  const text = plain(name);
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}
