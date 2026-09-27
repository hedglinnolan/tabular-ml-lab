import { useColumnSummaries, useProjectView, useStageResult } from "../api/queries";
import { useProjectEvents } from "../api/events";
import type { ProjectView } from "../api/schema";
import { Header } from "../components/Header";
import { JobChips } from "../components/JobChips";
import { PipelinePanel } from "../components/pipeline/PipelinePanel";
import { Record } from "../components/record/Record";
import { NumberTween } from "../motion/NumberTween";
import { Link } from "../router";
import { readingText } from "../util/format";
import styles from "./ProjectScreen.module.css";

export function ProjectScreen({ pid }: { pid: string }) {
  const stream = useProjectEvents(pid);
  const viewQ = useProjectView(pid);

  if (viewQ.isPending) {
    return (
      <>
        <Header />
        <main className={styles.message}>Opening the project…</main>
      </>
    );
  }
  if (viewQ.isError) {
    return (
      <>
        <Header />
        <main className={styles.message}>
          <p>This project could not be opened: {viewQ.error.message}</p>
          <Link href="/">Back to the start</Link>
        </main>
      </>
    );
  }
  return <Loaded pid={pid} view={viewQ.data} stream={stream} />;
}

function Loaded({ pid, view, stream }: { pid: string; view: ProjectView; stream: string }) {
  const { stages, state, summary } = view;
  const ran = (s: string) => stages[s] !== undefined && stages[s].status !== "idle";
  const ingest = useStageResult(pid, "ingest", ran("ingest"));
  const profile = useStageResult(pid, "profile", ran("profile"));
  const targetInfo = useStageResult(
    pid,
    "target_info",
    state.target !== null || ran("target_info"),
  );
  const findings = useStageResult(pid, "findings", state.lens !== null);
  const ingested = stages.ingest?.status === "fresh";
  const summaries = useColumnSummaries(pid, ingested);

  return (
    <>
      <Header jobs={<JobChips pid={pid} stages={stages} />}>
        <span className={styles.name} title={summary.source_name}>
          {summary.name}
        </span>
        {summary.n_rows !== null && summary.n_cols !== null ? (
          <span className={styles.size} data-testid="dataset-size">
            <NumberTween value={summary.n_rows} /> rows × <NumberTween value={summary.n_cols} />{" "}
            columns
          </span>
        ) : (
          <span className={styles.size} data-testid="dataset-size">
            {readingText(stages.ingest)}
          </span>
        )}
        {stream === "reconnecting" ? (
          <span className={styles.stream} role="status">
            reconnecting to the server…
          </span>
        ) : null}
      </Header>
      <div className={styles.layout}>
        <main className={styles.record} aria-label="The record">
          <Record
            pid={pid}
            view={view}
            ingest={ingest.data}
            profile={profile.data}
            targetInfo={targetInfo.data}
            findings={findings.data}
            summaries={summaries.data}
          />
        </main>
        <aside className={styles.pipeline} aria-label="Pipeline">
          <PipelinePanel pid={pid} view={view} ingest={ingest.data} targetInfo={targetInfo.data} />
        </aside>
      </div>
    </>
  );
}
