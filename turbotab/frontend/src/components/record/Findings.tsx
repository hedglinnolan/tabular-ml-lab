/**
 * What the lens noticed: finding cards (§04), bounded to five with a counted
 * expander. The shelf is never shortened — the rest are one press away and the
 * count says exactly how many. Expanding is disclosure, not consequence, so it is
 * instant (§05.2 ruling, 2026-08-03).
 */
import { useId, useState } from "react";
import type { Finding, FindingsArtifact } from "../../api/schema";
import { fmtInt } from "../../util/format";
import { Prose } from "../Prose";
import styles from "./Findings.module.css";

const BOUND = 5;
const COLUMN_BOUND = 6;

const BADGE: Record<string, string> = {
  settled: styles.settled!,
  convention: styles.convention!,
  disputed: styles.disputed!,
};

function FindingCard({ f }: { f: Finding }) {
  const [allCols, setAllCols] = useState(false);
  const cols = allCols ? f.affected_columns : f.affected_columns.slice(0, COLUMN_BOUND);
  const hidden = f.affected_columns.length - cols.length;
  return (
    <li className={styles.card} data-severity={f.severity} data-testid={`finding-${f.id}`}>
      <div className={styles.top}>
        <span className={styles.severity} data-severity={f.severity}>
          {f.severity}
        </span>
        <span className={styles.source}>
          {f.source}
          {f.lens ? ` · ${f.lens}` : ""}
        </span>
        {f.evidence ? (
          <span
            className={`${styles.badge} ${BADGE[f.evidence.status.toLowerCase()] ?? ""}`}
            title={`Source: ${f.evidence.source}`}
          >
            {f.evidence.status.toUpperCase()}
          </span>
        ) : null}
      </div>
      <h3 className={styles.title}>
        <Prose text={f.title} />
      </h3>
      <p className={styles.detail}>
        <Prose text={f.detail} />
      </p>
      {f.why_it_matters ? (
        <p className={styles.why}>
          <span className={styles.whyLabel}>Consequence</span> <Prose text={f.why_it_matters} />
        </p>
      ) : null}
      {f.affected_columns.length ? (
        <div className={styles.cols} aria-label="Affected columns">
          {cols.map((col) => (
            <code key={col} className="v">
              {col}
            </code>
          ))}
          {hidden > 0 ? (
            <button type="button" className={styles.more} onClick={() => setAllCols(true)}>
              {fmtInt(hidden)} more
            </button>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

export function FindingsList({ artifact }: { artifact: FindingsArtifact }) {
  const [expanded, setExpanded] = useState(false);
  const listId = useId();
  const { findings } = artifact;
  const shown = expanded ? findings : findings.slice(0, BOUND);
  const rest = findings.length - shown.length;
  if (findings.length === 0) {
    return (
      <p className={styles.none} data-testid="findings-none">
        Nothing was noticed. {artifact.basis}
      </p>
    );
  }
  return (
    <>
      <ul id={listId} className={styles.list}>
        {shown.map((f) => (
          <FindingCard key={f.id} f={f} />
        ))}
      </ul>
      <div className={styles.footer}>
        {rest > 0 ? (
          <button
            type="button"
            className={styles.expand}
            aria-controls={listId}
            aria-expanded={false}
            onClick={() => setExpanded(true)}
            data-testid="findings-more"
          >
            {fmtInt(rest)} more {rest === 1 ? "finding" : "findings"}
          </button>
        ) : findings.length > BOUND ? (
          <button
            type="button"
            className={styles.expand}
            aria-controls={listId}
            aria-expanded={true}
            onClick={() => setExpanded(false)}
          >
            Show the first {BOUND}
          </button>
        ) : null}
        <span className={styles.basis}>{artifact.basis}</span>
      </div>
    </>
  );
}
