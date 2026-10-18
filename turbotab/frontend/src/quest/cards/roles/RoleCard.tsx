/**
 * One column's role, confirmed on its own (the walkthrough of 2026-10-17, finding 1; BLUEPRINT §14
 * rule 2). The card shows the column, the role it was proposed as, the evidence the roles stage gave
 * for it, and why it waits (the engine's words on the line), and records one `confirm_role` per
 * press: the role as proposed, or another the engine offers for it. No press confirms several
 * columns, a refusal's ways forward included (roles.ts `ownOptions`, `ownRefusal`).
 *
 * A group of columns the roles stage proposed with one proposal (ruling R-g: one role, one
 * confidence, one reason word for word) is one card: how many columns, their role, the confidence
 * and the shared reason, the first columns named and every column a click away, and one press that
 * confirms them all (the engine's `confirm_readings` of exactly the group's columns: roles.ts
 * `groupPress`). Each listed column keeps its own presses, which move it alone; the rest stay one
 * group. Beneath either, how many columns proposed as predictors still wait, from the quest log.
 */
import { useId, useState, type ReactNode } from "react";
import { isRefusalError } from "../../../api/client";
import { useDecide, useQuest } from "../../../api/queries";
import type { Decision, QuestLine, ReadItem, Refusal } from "../../../api/schema";
import { FailureNote, RefusalNote } from "../../../components/record/Refusal";
import { Plain, plain } from "../../../kit";
import type { LineCardProps } from "../../contracts";
import { said } from "../../words";
import c from "../cards.module.css";
import { Readings } from "../Readings";
import {
  groupItem,
  groupPress,
  groupRefusal,
  memberItems,
  ownOptions,
  ownRefusal,
  roleItem,
  waitingPredictorColumns,
  waitingSentence,
} from "./roles";
import r from "./roles.module.css";

export function RoleCard({ pid, line }: LineCardProps) {
  const decide = useDecide(pid);
  const log = useQuest(pid).data;
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const item = roleItem(line);
  // A role line without its reading is drawn as the reading rows it carries, never a blank.
  if (!item) return <Readings pid={pid} line={line} />;
  // How many columns proposed as predictors wait, as the Columns panel says it (W1-T: a flag or a
  // column proposed as left out is no predictor once confirmed, so it is not counted).
  const waiting = log ? waitingSentence(waitingPredictorColumns(log)) : null;
  const record = (decision: Decision) => {
    setRefusal(null);
    setFailure(null);
    decide.mutate(decision, {
      onError: (err) => (isRefusalError(err) ? setRefusal(err.refusal) : setFailure(err instanceof Error ? err.message : String(err))),
    });
  };
  const notes = (
    <>
      {failure ? <FailureNote message={failure} onDismiss={() => setFailure(null)} /> : null}
      {waiting ? (
        <p className={c.quiet} data-testid="roles-waiting">
          {waiting}
        </p>
      ) : null}
    </>
  );
  if (groupItem(line)) {
    return (
      <GroupCard line={line} pending={decide.isPending} record={record}>
        {refusal ? (
          <RefusalNote
            refusal={groupRefusal(refusal, line)}
            onDismiss={() => setRefusal(null)}
            onExit={(exit) => (exit.decision ? record(exit.decision) : setRefusal(null))}
          />
        ) : null}
        {notes}
      </GroupCard>
    );
  }
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
      <Moves item={item} pending={decide.isPending} record={record} />
      {refusal ? (
        <RefusalNote
          refusal={ownRefusal(refusal, item.column)}
          onDismiss={() => setRefusal(null)}
          onExit={(exit) => (exit.decision ? record(exit.decision) : setRefusal(null))}
        />
      ) : null}
      {notes}
    </section>
  );
}

/** One column's own presses (roles.ts `ownOptions`): each one `confirm_role` of that column. */
function Moves({ item, pending, record }: { item: ReadItem; pending: boolean; record: (d: Decision) => void }) {
  const options = ownOptions(item);
  if (!options.length) return null;
  return (
    <div className={c.rowOptions}>
      {options.map(({ option, role, confirms }) => (
        <button
          key={role}
          type="button"
          className={confirms ? r.confirm : c.small}
          disabled={pending}
          data-testid={`role-option-${role}`}
          onClick={() => record(option.decision as unknown as Decision)}
        >
          {plain(option.label)}
        </button>
      ))}
    </div>
  );
}

/**
 * A group's card (ruling R-g): the group read whole (its first item: how many, the role, the
 * confidence, below high, the shared reason as its evidence), why it waits, its one press, and its
 * columns a click away, each with its own presses.
 */
function GroupCard({
  line,
  pending,
  record,
  children,
}: {
  line: QuestLine;
  pending: boolean;
  record: (d: Decision) => void;
  children: ReactNode;
}) {
  const head = groupItem(line)!;
  const members = memberItems(line);
  const press = groupPress(line);
  const [open, setOpen] = useState(false);
  const listId = useId();
  return (
    <section className={c.lineCard} data-testid={`role-group-${head.column}`} data-status={line.status} aria-label={said(line.name)}>
      <p className={c.kicker}>{said(line.name)}</p>
      <p className={c.rowWords} data-testid="role-group-words">
        <Plain text={head.words} />
      </p>
      {head.evidence ? (
        <p className={c.quiet} data-testid="role-evidence">
          <Plain text={head.evidence} />
        </p>
      ) : null}
      {line.reason ? (
        <p className={c.quiet} data-testid="role-reason">
          <Plain text={line.reason} />
        </p>
      ) : null}
      {press ? (
        <div className={c.rowOptions}>
          <button
            type="button"
            className={r.confirm}
            disabled={pending}
            data-testid="role-group-confirm"
            onClick={() => record(press.decision as unknown as Decision)}
          >
            {plain(press.label)}
          </button>
        </div>
      ) : null}
      <button
        type="button"
        className={c.small}
        aria-expanded={open}
        aria-controls={listId}
        data-testid="role-group-columns"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "Hide the columns" : `List the ${members.length.toLocaleString("en-US")} columns`}
      </button>
      {open ? (
        <ul id={listId} className={r.members} data-testid="role-group-list">
          {members.map((m) => (
            <li key={m.column} data-testid={`role-${m.column}`}>
              <p className={c.rowWords}>
                <Plain text={`\`${m.column}\` ${m.words}`} />
              </p>
              <Moves item={m} pending={pending} record={record} />
            </li>
          ))}
        </ul>
      ) : null}
      {children}
    </section>
  );
}
