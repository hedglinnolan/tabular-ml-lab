/**
 * A press that did not record answers at the control (DRIVE_RUBRIC §2.1): a refusal says
 * why and offers the server's exits; a failure says the server did not record it. Neither
 * looks like a dead control.
 */
import { useId } from "react";
import type { Refusal } from "../../api/schema";
import { Prose } from "../Prose";
import s from "./Refusal.module.css";

export function RefusalNote({
  refusal,
  onExit,
  onDismiss,
}: {
  refusal: Refusal;
  onExit: (exit: Refusal["error"]["exits"][number]) => void;
  onDismiss: () => void;
}) {
  const id = useId();
  return (
    <div className={s.refusal} role="alert" aria-labelledby={id} data-testid="refusal">
      <p id={id} className={s.text}>
        <span className={s.label}>Not recorded</span> <Prose text={refusal.error.message} />
      </p>
      <div className={s.exits}>
        {refusal.error.exits.map((exit) => (
          <button
            key={exit.label}
            type="button"
            className={exit.decision ? s.exit : s.ghost}
            onClick={() => onExit(exit)}
            data-testid="refusal-exit"
          >
            <Prose text={exit.label} />
            {exit.decision ? " instead" : ""}
          </button>
        ))}
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
