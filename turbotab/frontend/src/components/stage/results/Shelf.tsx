/**
 * The shelf (the banner's "models" segment): every family that can model this task, in the
 * shelf's order, each with its inductive bias and its concerns stated outright — never hidden,
 * never removed. Families chosen for fitting are marked; a fitted family that does worse than the
 * baseline says so here too.
 */
import type { FitArtifact, ShelfArtifact } from "../../../api/m1-stage-types";
import { Rich } from "../text";
import s from "./results.module.css";

interface Props {
  shelf: ShelfArtifact;
  chosen: string[] | null;
  fit: FitArtifact | null;
}

const FIT_WORD: Record<string, string> = { good: "good fit", fair: "fair fit", poor: "poor fit" };

export function Shelf({ shelf, chosen, fit }: Props) {
  return (
    <div className={s.shelf} data-testid="shelf">
      <ol className={s.shelfList}>
        {shelf.families.map((f) => {
          const fitted = fit?.models.find((m) => m.family === f.key);
          const concerns = [...f.concerns, ...(fitted?.concerns ?? []).filter((c) => !f.concerns.includes(c))];
          const on = chosen?.includes(f.key) ?? false;
          return (
            <li key={f.key} className={s.shelfItem} data-chosen={on || undefined}>
              <div className={s.shelfHead}>
                <span className={s.rank}>{f.rank}</span>
                <span className={s.family}>{f.label}</span>
                <span className={s.fitTag} data-fit={f.fit}>
                  {FIT_WORD[f.fit] ?? f.fit}
                </span>
                {on ? <span className={s.chosen}>chosen</span> : null}
              </div>
              <p className={s.bias}>{f.inductive_bias}</p>
              {concerns.map((c, i) => (
                <p key={i} className={s.concern}>
                  <Rich text={c} />
                </p>
              ))}
            </li>
          );
        })}
      </ol>
      <p className={s.basis}>
        <Rich text={shelf.basis} />
      </p>
    </div>
  );
}
