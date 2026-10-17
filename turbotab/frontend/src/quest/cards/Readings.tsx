/**
 * A line's reading items as quiet rows (ruling 7: from what the quest line already serves): each
 * reading's words and evidence, and the other readings the engine offers for it, each a button that
 * records the decision the item carries. A line with no items says its own reason.
 */
import { isRefusalError } from "../../api/client";
import { useDecide } from "../../api/queries";
import type { Decision, QuestLine } from "../../api/schema";
import { Plain } from "../../kit";
import { said } from "../words";
import c from "./cards.module.css";

export function Readings({ pid, line }: { pid: string; line: QuestLine }) {
  const decide = useDecide(pid);
  const refused = decide.error && isRefusalError(decide.error) ? decide.error.refusal : null;
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
          {line.items.map((item) => (
            <li key={`${item.kind}:${item.column}`} className={c.row}>
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
                      <button
                        key={o.label}
                        type="button"
                        className={c.small}
                        disabled={decide.isPending}
                        onClick={() => decide.mutate(o.decision as unknown as Decision)}
                      >
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
