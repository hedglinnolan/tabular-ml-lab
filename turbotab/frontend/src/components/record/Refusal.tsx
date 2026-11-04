/**
 * A press that did not record answers at the control (DRIVE_RUBRIC §2.1): a refusal says
 * why and offers the server's exits; a failure says the server did not record it. Neither
 * looks like a dead control. No exit confirms a column's role with other readings in one press
 * (BLUEPRINT §14 rule 2; besideAsk.ts `oneRoleAtATime`). An exit that names no decision but the
 * card it opens (`target`, record/target.ts) opens that card in the quest shell and closes the
 * note (G12); every other exit is the caller's press.
 */
import { useId } from "react";
import type { Refusal } from "../../api/schema";
import { Prose } from "../Prose";
import { oneRoleAtATime } from "./besideAsk";
import s from "./Refusal.module.css";
import { targetName, targetOf, useOpenTarget } from "./target";

export function RefusalNote({
  refusal: served,
  onExit,
  onDismiss,
}: {
  refusal: Refusal;
  onExit: (exit: Refusal["error"]["exits"][number]) => void;
  onDismiss: () => void;
}) {
  const refusal = oneRoleAtATime(served);
  const id = useId();
  const opener = useOpenTarget();
  return (
    <div className={s.refusal} role="alert" aria-labelledby={id} data-testid="refusal">
      <p id={id} className={s.text}>
        <span className={s.label}>Not recorded</span> <Prose text={refusal.error.message} />
      </p>
      <div className={s.exits}>
        {refusal.error.exits.map((exit) => {
          const open = opener(exit);
          const to = open ? targetOf(exit) : null;
          return (
            <button
              key={exit.label}
              type="button"
              className={exit.decision ? s.exit : s.ghost}
              onClick={() => {
                if (!open) return onExit(exit);
                onDismiss();
                open();
              }}
              data-testid="refusal-exit"
              data-target={to ? targetName(to) : undefined}
            >
              <Prose text={exit.label} />
              {exit.decision ? " instead" : ""}
            </button>
          );
        })}
        {refusal.error.exits.length === 0 ? (
          <button type="button" className={s.ghost} onClick={onDismiss}>
            Understood
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function FailureNote({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className={s.refusal} role="alert" data-testid="failure">
      <p className={s.text}>
        <span className={s.label}>Not recorded</span> The server did not record this: {message}
      </p>
      <div className={s.exits}>
        <button type="button" className={s.ghost} onClick={onDismiss}>
          Understood
        </button>
      </div>
    </div>
  );
}
