/**
 * A line's reading items as quiet rows (ruling 7: from what the quest line already serves): each
 * reading's words and evidence, and the other readings the engine offers for it, each a button that
 * records the decision the item carries. A line with no items says its own reason.
 *
 * Roles the roles stage proposed with one proposal (ruling R-g: one judgment of the recognizer),
 * which the fit's refusal asks whole, are one row: the group read whole, its one press, and its
 * columns a click away, each with its own presses (roles/roles.ts `readingBlocks`).
 */
import { useId, useState } from "react";
import { isRefusalError } from "../../api/client";
import { useDecide } from "../../api/queries";
import type { Decision, QuestLine, ReadItem } from "../../api/schema";
import { Plain } from "../../kit";
import { said } from "../words";
import c from "./cards.module.css";
import { readingBlocks } from "./roles/roles";
import r from "./roles/roles.module.css";

export function Readings({ pid, line }: { pid: string; line: QuestLine }) {
  const decide = useDecide(pid);
  const refused = decide.error && isRefusalError(decide.error) ? decide.error.refusal : null;
  const record = (d: Decision) => decide.mutate(d);
  return (
    <section className={c.lineCard} data-testid={`reading-${line.key}`} aria-label={said(line.name)}>
      <p className={c.kicker}>{said(line.name)}</p>
      {line.items.length === 0 ? (
        line.reason || line.would_change || line.changes_nothing ? (
          <p className={c.quiet}>
            <Plain text={line.reason ?? line.would_change ?? line.changes_nothing} />
          </p>
        ) : null
      ) : (
        <ul className={c.rows}>
          {readingBlocks(line.items).map(({ head, members }) => (
            <li key={`${head.kind}:${head.column}`} className={c.row}>
              <Row item={head} pending={decide.isPending} record={record} />
              {members.length ? <Members members={members} pending={decide.isPending} record={record} /> : null}
            </li>
          ))}
        </ul>
      )}
      {refused ? (
        <p className={c.note} role="alert">
          <Plain text={refused.error.message} />
        </p>
      ) : null}
    </section>
  );
}

interface RowProps {
  item: ReadItem;
  pending: boolean;
  record: (d: Decision) => void;
}

/** One reading: its words, its evidence and the engine's ways forward for it. */
function Row({ item, pending, record }: RowProps) {
  return (
    <>
      <span className={c.rowWords}>
        <Plain text={item.words} />
      </span>
      {item.evidence ? (
        <span className={c.quiet}>
          <Plain text={item.evidence} />
        </span>
      ) : null}
      {item.options.length ? (
        <span className={c.rowOptions}>
          {item.options.map((o) =>
            o.decision ? (
              <button key={o.label} type="button" className={c.small} disabled={pending} onClick={() => record(o.decision as unknown as Decision)}>
                {o.label}
              </button>
            ) : (
              <span key={o.label} className={c.quiet}>
                {o.label}
              </span>
            ),
          )}
        </span>
      ) : null}
    </>
  );
}

/** A group's columns, a click away, each with its own presses (which move it alone). */
function Members({ members, pending, record }: { members: ReadItem[]; pending: boolean; record: (d: Decision) => void }) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  return (
    <>
      <button type="button" className={c.small} aria-expanded={open} aria-controls={listId} onClick={() => setOpen((o) => !o)}>
        {open ? "Hide the columns" : `List the ${members.length.toLocaleString("en-US")} columns`}
      </button>
      {open ? (
        <ul id={listId} className={r.members}>
          {members.map((m) => (
            <li key={m.column} className={c.row}>
              <Row item={{ ...m, words: `\`${m.column}\` ${m.words}` }} pending={pending} record={record} />
            </li>
          ))}
        </ul>
      ) : null}
    </>
  );
}
