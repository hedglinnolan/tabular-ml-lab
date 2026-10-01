/**
 * The donor × recipient navigator — the substitution question's control (§13). Each cell is a
 * pair the design allows (a parent and its own part are not offered); pressing one records
 * `set_substitution`, and the press is acknowledged at the cell until the curve arrives.
 */
import type { DesignArtifact } from "../../../api/m1-stage-types";
import s from "./results.module.css";

interface Props {
  pairs: DesignArtifact["substitution_pairs"];
  current: { donor: string; recipient: string } | null;
  pending: { donor: string; recipient: string } | null;
  onPick: (donor: string, recipient: string) => void;
  disabled?: boolean;
}

export function PairNavigator({ pairs, current, pending, onPick, disabled }: Props) {
  const names: string[] = [];
  for (const p of pairs) {
    if (!names.includes(p.donor)) names.push(p.donor);
    if (!names.includes(p.recipient)) names.push(p.recipient);
  }
  const allowed = new Set(pairs.map((p) => `${p.donor}>${p.recipient}`));
  if (names.length < 2) return null;
  return (
    <div className={s.nav} data-testid="pair-navigator">
      <table className={s.navTable}>
        <caption className={s.navCaption}>Move energy from (row) to (column)</caption>
        <thead>
          <tr>
            <th scope="col" className={s.navCorner}>
              from ↓ to →
            </th>
            {names.map((n) => (
              <th key={n} scope="col" className={s.navHead}>
                <span>{n}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {names.map((d) => (
            <tr key={d}>
              <th scope="row" className={s.navRowHead}>
                {d}
              </th>
              {names.map((r) => {
                if (d === r) return <td key={r} className={s.navSelf} aria-hidden="true" />;
                const ok = allowed.has(`${d}>${r}`);
                const on = current?.donor === d && current?.recipient === r;
                const wait = pending?.donor === d && pending?.recipient === r;
                return (
                  <td key={r} className={s.navCell}>
                    {ok ? (
                      <button
                        type="button"
                        className={s.navButton}
                        aria-pressed={on}
                        data-pending={wait || undefined}
                        disabled={disabled || on}
                        onClick={() => onPick(d, r)}
                        aria-label={`Move energy from ${d} to ${r}`}
                        title={`${d} → ${r}`}
                      />
                    ) : (
                      <span className={s.navNo} title={`${d} → ${r} is not offered for this design`} />
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
