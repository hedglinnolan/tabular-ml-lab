/**
 * Server sentences mark data terms with backticks (`hba1c`). The app's voice is
 * serif and data speaks mono, so each backticked span becomes a mono chip.
 */
import { Fragment, type ReactNode } from "react";

export function Prose({ text }: { text: string }) {
  const parts = text.split("`");
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <code key={i} className="v">
            {part}
          </code>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

/** A data value inline with prose. */
export function V({ children }: { children: ReactNode }) {
  return <code className="v">{children}</code>;
}
