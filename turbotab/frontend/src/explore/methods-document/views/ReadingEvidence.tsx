/**
 * A reading's evidence on the user's own columns (§11 rule 6: their data before theory). The
 * server's preview of a role confirmation draws nothing ("Nothing about this choice can be shown on
 * your data yet"), so the canvas shows what the guess rests on instead: each column the slot
 * would settle, the value it would take, and the column summary a reader checks it against — for
 * a flag, how many of its rows are set and how many blanks its base column has to explain; for
 * anything else, its distinct values and its blanks. Then what each answer changes, from the roles
 * teaching.
 *
 * Every number is the server's column summary (GET /columns); every sentence is the server's.
 */
import { Rich } from "../../../components/stage/text";
import type { RoleProposal } from "../../../api/m1-types";
import { columnSummary, teaching } from "../fixture";
import s from "../doc.module.css";

export interface EvidenceItem {
  column: string;
  value: string;
}

const n = (x: number) => x.toLocaleString("en-US");

function trueCount(column: string): number | null {
  const top = columnSummary(column)?.top;
  const hit = top?.find((t) => t.value === true);
  return hit ? hit.count : top ? 0 : null;
}

function Values({ item, proposal }: { item: EvidenceItem; proposal: RoleProposal | null }) {
  const sum = columnSummary(item.column);
  if (!sum) return <>—</>;
  const base = item.value === "flag" ? (proposal?.linked_to ?? null) : null;
  if (base) {
    const set = trueCount(item.column);
    const blanks = columnSummary(base)?.n_missing ?? null;
    return (
      <>
        <span className="num">{set === null ? "—" : n(set)}</span> rows set;{" "}
        <code className="v">{base}</code> has{" "}
        <span className={s.flagBlanks} data-zero={blanks === 0 || undefined}>
          {blanks === null ? "—" : n(blanks)}
        </span>{" "}
        blanks
      </>
    );
  }
  return (
    <>
      <span className="num">{n(sum.n_unique)}</span> distinct, <span className="num">{n(sum.n_missing)}</span> blank
    </>
  );
}

export function ReadingEvidence({
  items,
  proposals,
  title,
}: {
  items: EvidenceItem[];
  proposals: RoleProposal[];
  title: string;
}) {
  const proposal = (c: string) => proposals.find((p) => p.column === c) ?? null;
  const flag = items.find((i) => i.value === "flag" && proposal(i.column)?.reason);
  const role = teaching("roles");
  const consequence = (v: string) => role?.options.find((o) => o.value === v)?.consequence ?? null;
  const values = [...new Set(items.map((i) => i.value))];
  // What the other answer would change, where a single answer is on the table.
  const alternative = values.length === 1 && values[0] === "flag" ? "covariate" : null;
  return (
    <figure className={s.evidence} data-purpose="reading_evidence">
      <p className={s.evidenceLead}>
        <Rich text={title} />
      </p>
      <div className={s.evidenceTableWrap}>
        <table className={s.evidenceTable}>
          <thead>
            <tr>
              <th scope="col">Column</th>
              <th scope="col">Would be</th>
              <th scope="col">What its values show</th>
            </tr>
          </thead>
          <tbody>
            {items.map((i) => (
              <tr key={i.column}>
                <td>
                  <code className="v">{i.column}</code>
                </td>
                <td>{i.value}</td>
                <td>
                  <Values item={i} proposal={proposal(i.column)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {flag ? (
        <p className={s.evidenceReason}>
          <span className={s.evidenceKicker}>The guess rests on</span>
          <Rich text={proposal(flag.column)!.reason} />
        </p>
      ) : null}
      <dl className={s.evidenceWhat}>
        {values.map((v) => (
          <div key={v} className={s.evidenceWhatRow}>
            <dt>As {v}s</dt>
            <dd>{consequence(v)}</dd>
          </div>
        ))}
        {alternative ? (
          <div className={s.evidenceWhatRow}>
            <dt>As {alternative}s instead</dt>
            <dd>{consequence(alternative)}</dd>
          </div>
        ) : null}
      </dl>
    </figure>
  );
}
