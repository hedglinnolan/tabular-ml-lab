/**
 * One column's role, confirmed on its own (the walkthrough of 2026-10-17, finding 1; BLUEPRINT §14
 * rule 2). The card shows the column, the role it was proposed as, the evidence the roles stage gave
 * for it, and why it waits (the engine's words on the line), and records one `confirm_role` per
 * press: the role as proposed, or another the engine offers for it. No press confirms several
 * columns, a refusal's ways forward included (roles.ts `ownOptions`, `ownRefusal`). Beneath, how
 * many columns still wait, from the quest log.
 */
import { useState } from "react";
import { isRefusalError } from "../../../api/client";
import { useDecide, useQuest } from "../../../api/queries";
import type { Decision, Refusal } from "../../../api/schema";
import { FailureNote, RefusalNote } from "../../../components/record/Refusal";
import { Plain, plain } from "../../../kit";
import type { LineCardProps } from "../../contracts";
import { said } from "../../words";
import c from "../cards.module.css";
import { Readings } from "../Readings";
import { ownOptions, ownRefusal, roleItem, waitingRoles, waitingSentence } from "./roles";
import r from "./roles.module.css";

export function RoleCard({ pid, line }: LineCardProps) {
  const decide = useDecide(pid);
  const log = useQuest(pid).data;
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const item = roleItem(line);
  // A role line without its reading is drawn as the reading rows it carries, never a blank.
  if (!item) return <Readings pid={pid} line={line} />;
  const options = ownOptions(item);
  const waiting = log ? waitingSentence(waitingRoles(log).length) : null;
  const record = (decision: Decision) => {
    setRefusal(null);
    setFailure(null);
    decide.mutate(decision, {
      onError: (err) => (isRefusalError(err) ? setRefusal(err.refusal) : setFailure(err instanceof Error ? err.message : String(err))),
    });
  };
  return (
    <section className={c.lineCard} data-testid={`role-${item.column}`} data-status={line.status} aria-label={said(line.name)}>
      <p className={c.kicker}>{said(line.name)}</p>
      <p className={c.rowWords}>
        <Plain text={`\`${item.column}\` ${item.words}`} />
      </p>
      {item.evidence ? (
        <p className={c.quiet} data-testid="role-evidence">
          <Plain text={item.evidence} />
        </p>
      ) : null}
      {line.reason ? (
        <p className={c.quiet} data-testid="role-reason">
          <Plain text={line.reason} />
        </p>
      ) : null}
      {options.length ? (
        <div className={c.rowOptions}>
          {options.map(({ option, role, confirms }) => (
            <button
              key={role}
              type="button"
              className={confirms ? r.confirm : c.small}
              disabled={decide.isPending}
              data-testid={`role-option-${role}`}
              onClick={() => record(option.decision as unknown as Decision)}
            >
              {plain(option.label)}
            </button>
          ))}
        </div>
      ) : null}
      {refusal ? (
        <RefusalNote
          refusal={ownRefusal(refusal, item.column)}
          onDismiss={() => setRefusal(null)}
          onExit={(exit) => (exit.decision ? record(exit.decision) : setRefusal(null))}
        />
      ) : null}
      {failure ? <FailureNote message={failure} onDismiss={() => setFailure(null)} /> : null}
      {waiting ? (
        <p className={c.quiet} data-testid="roles-waiting">
          {waiting}
        </p>
      ) : null}
    </section>
  );
}
