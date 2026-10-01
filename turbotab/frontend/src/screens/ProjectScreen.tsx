/**
 * A project (M1_CONTRACT §10): the header, the pipeline banner, then the working window —
 * the Record on the left and the stage on the right. What the stage shows is one focus,
 * shared by the Record, the banner and the stage, held in a context this screen creates.
 */
import { useRef, type FocusEvent, type KeyboardEvent } from "react";
import { useProjectView } from "../api/queries";
import { useProjectEvents } from "../api/events";
import type { ProjectView } from "../api/schema";
import { Banner } from "../components/banner/Banner";
import { Header } from "../components/Header";
import { JobChips } from "../components/JobChips";
import { Record } from "../components/record/Record";
import { Stage } from "../components/stage/Stage";
import { NumberTween } from "../motion/NumberTween";
import { Link } from "../router";
import { StageFocusProvider, useStageFocus } from "../state/focus";
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
  return (
    <StageFocusProvider key={pid}>
      <Loaded pid={pid} view={viewQ.data} stream={stream} />
    </StageFocusProvider>
  );
}

function Loaded({ pid, view, stream }: { pid: string; view: ProjectView; stream: string }) {
  const { stages, summary } = view;
  const { focus, setFocus, release, reset, keepPreview, endPreview } = useStageFocus();
  const working = useRef<HTMLDivElement>(null);

  // Leaving the working window (keyboard focus gone elsewhere) ends a preview or evidence
  // view; moving between the Record and the stage keeps it, so the stage can be used.
  const onBlur = (e: FocusEvent<HTMLDivElement>) => {
    const next = e.relatedTarget as Node | null;
    if (next && working.current?.contains(next)) return;
    if (focus.kind === "option" || focus.kind === "finding") release(focus);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape" && !e.defaultPrevented && focus.kind !== "live") reset();
  };

  return (
    <div className={styles.screen}>
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
      <Banner pid={pid} view={view} />
      <div className={styles.window} ref={working} onBlur={onBlur} onKeyDown={onKeyDown}>
        <main className={styles.record} aria-label="The record">
          <Record pid={pid} view={view} />
        </main>
        <aside
          className={styles.stage}
          aria-label="The stage"
          onPointerEnter={keepPreview}
          onPointerLeave={endPreview}
        >
          <Stage pid={pid} view={view} focus={focus} onFocus={setFocus} />
        </aside>
      </div>
    </div>
  );
}
