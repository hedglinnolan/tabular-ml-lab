/**
 * The live pipeline panel (BLUEPRINT §0): always visible beside the Record.
 * Rows — participant flow, n at each step. Columns — raw columns toward the model
 * matrix. Results — what has been fit. When a decision changes, the sections it
 * reaches go stale and veil in order, then clear as fresh results arrive.
 */
import { useState, type ReactNode } from "react";
import type {
  DatasetInfo,
  Dtype,
  ProjectView,
  StageResult,
  TargetInfoArtifact,
} from "../../api/schema";
import { DTYPES } from "../../api/schema";
import { NumberTween } from "../../motion/NumberTween";
import { StaleVeil, veilFor, type VeilState } from "../../motion/StaleVeil";
import { cx, fmtInt } from "../../util/format";
import { V } from "../Prose";
import { MiniHistogram } from "./MiniHistogram";
import { TablePreview } from "./TablePreview";
import styles from "./PipelinePanel.module.css";

interface Props {
  pid: string;
  view: ProjectView;
  ingest?: StageResult<DatasetInfo>;
  targetInfo?: StageResult<TargetInfoArtifact>;
}

const NAME_BOUND = 8;

function Section({
  title,
  aside,
  children,
  testId,
}: {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
  testId: string;
}) {
  const id = `pipeline-${testId}`;
  return (
    <section className={styles.section} aria-labelledby={id} data-testid={testId}>
      <div className={styles.sectionHead}>
        <h2 id={id} className={styles.kicker}>
          {title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function ingestVeil(view: ProjectView, ingest?: StageResult<DatasetInfo>): VeilState {
  return veilFor(view.stages.ingest, ingest);
}

function Rows({ view, ingest }: { view: ProjectView; ingest?: StageResult<DatasetInfo> }) {
  const n = ingest?.artifact?.n_rows ?? view.summary.n_rows;
  return (
    <Section title="Rows" testId="rows">
      <StaleVeil state={ingestVeil(view, ingest)} order={2}>
        <ol className={styles.flow} aria-label="Participant flow">
          <li className={styles.node} data-testid="rows-node-loaded">
            <span className={styles.dot} aria-hidden="true" />
            <div className={styles.nodeText}>
              <span className={styles.nodeLabel}>Loaded table</span>
              <span className={styles.nodeSub}>{view.summary.source_name}</span>
            </div>
            <span className={styles.nodeCount}>
              {n === null || n === undefined ? (
                <span className={styles.muted}>
                  {view.stages.ingest?.status === "error" ? "could not be read" : "reading…"}
                </span>
              ) : (
                <>
                  <NumberTween value={n} data-testid="rows-n" />{" "}
                  <span className={styles.unit}>rows</span>
                </>
              )}
            </span>
          </li>
        </ol>
        <p className={styles.note}>
          Nothing has been excluded. Eligibility and the split will add their steps here.
        </p>
      </StaleVeil>
    </Section>
  );
}

function Columns({ pid, view, ingest, targetInfo }: Props) {
  const [open, setOpen] = useState<Partial<Record<Dtype, boolean>>>({});
  const info = ingest?.artifact;
  const target = view.state.target;
  const ti = targetInfo?.artifact;
  const tiVeil = target ? veilFor(view.stages.target_info, targetInfo) : "fresh";
  const veil: VeilState = ingestVeil(view, ingest) !== "fresh" ? ingestVeil(view, ingest) : tiVeil;
  const cols = info?.columns.filter((c) => c.name !== "__row_id") ?? [];
  const groups = DTYPES.map((d) => ({ dtype: d, cols: cols.filter((c) => c.dtype === d) })).filter(
    (g) => g.cols.length > 0,
  );

  return (
    <Section
      title="Columns"
      testId="columns"
      aside={
        info ? (
          <span className={styles.headCount}>
            <NumberTween value={cols.length} /> columns
          </span>
        ) : null
      }
    >
      {!info ? (
        <p className={styles.note}>Columns appear once the file has been read.</p>
      ) : (
        <StaleVeil state={veil} order={3} testId="veil-columns">
          {target ? (
            <div className={styles.target} data-testid="columns-target">
              <div className={styles.targetText}>
                <span className={styles.targetLabel}>Outcome</span> <V>{target}</V>
                {ti && ti.column === target ? (
                  <>
                    {" "}
                    · <V>{ti.task}</V>
                  </>
                ) : null}
              </div>
              {ti?.histogram && ti.column === target ? (
                <MiniHistogram histogram={ti.histogram} label={`Distribution of ${target}`} />
              ) : null}
            </div>
          ) : (
            <p className={styles.note}>No outcome chosen yet; it will be marked here.</p>
          )}
          <ul className={styles.groups} aria-label="Columns by type">
            {groups.map((g) => {
              const expanded = open[g.dtype] ?? false;
              const names = expanded ? g.cols : g.cols.slice(0, NAME_BOUND);
              const targetInGroup = target && g.cols.some((c) => c.name === target);
              // Keep the outcome visible in its group even when the list is bounded.
              const shown =
                targetInGroup && !names.some((c) => c.name === target)
                  ? [...names.slice(0, NAME_BOUND - 1), g.cols.find((c) => c.name === target)!]
                  : names;
              const rest = g.cols.length - shown.length;
              return (
                <li key={g.dtype} className={styles.group} data-testid={`group-${g.dtype}`}>
                  <div className={styles.groupHead}>
                    <span className={styles.dtype}>{g.dtype}</span>
                    <NumberTween value={g.cols.length} className={styles.groupCount} />
                  </div>
                  <div className={styles.names}>
                    {shown.map((c) => (
                      <code
                        key={c.name}
                        className={cx(styles.name, c.name === target && styles.isTarget)}
                        data-target={c.name === target || undefined}
                      >
                        {c.name}
                      </code>
                    ))}
                    {rest > 0 ? (
                      <button
                        type="button"
                        className={styles.more}
                        onClick={() => setOpen((o) => ({ ...o, [g.dtype]: true }))}
                      >
                        {fmtInt(rest)} more
                      </button>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
          <TablePreview
            pid={pid}
            columns={cols.map((c) => c.name)}
            dtypes={cols.map((c) => c.dtype)}
            nRows={info.n_rows}
            target={target}
          />
        </StaleVeil>
      )}
    </Section>
  );
}

function Results({ view }: { view: ProjectView }) {
  const { state } = view;
  const waits: { label: string; done: boolean }[] = [
    { label: "a lens", done: state.lens !== null },
    { label: "an outcome", done: state.target !== null },
    { label: "a purpose", done: state.purpose !== null },
    { label: "a train/test split (asked in a later version)", done: false },
  ];
  return (
    <Section title="Results" testId="results">
      <div className={styles.empty}>
        <p className={styles.emptyLead}>Nothing has been fit yet.</p>
        <p className={styles.emptyBody}>
          Held-out performance and a comparison of models will appear here. They wait on:
        </p>
        <ul className={styles.waits}>
          {waits.map((w) => (
            <li key={w.label} data-done={w.done || undefined}>
              <span className={styles.mark} aria-hidden="true" />
              {w.label}
              <span className="visually-hidden">{w.done ? " — recorded" : " — waiting"}</span>
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}

export function PipelinePanel(props: Props) {
  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <span className={styles.panelTitle}>Pipeline</span>
        <span className={styles.panelSub}>what is true of the working data now</span>
      </div>
      <Rows view={props.view} ingest={props.ingest} />
      <Columns {...props} />
      <Results view={props.view} />
    </div>
  );
}
