/**
 * The Record: the interview as a growing document. Each question is asked in
 * turn; an answered question settles into its decision sentence; "change"
 * reopens it and a new decision is appended (the old sentence stays in the
 * history). Below, what the chosen lenses noticed.
 */
import { useId, useMemo, useState, type ReactNode } from "react";
import { LayoutGroup, motion } from "motion/react";
import { isRefusalError } from "../../api/client";
import { useDecide } from "../../api/queries";
import type {
  ColumnSummary,
  DatasetInfo,
  Decision,
  DecisionRecord,
  FindingsArtifact,
  ProfileArtifact,
  ProjectView,
  Refusal,
  Slot,
  StageResult,
  StageStatus,
  TargetInfoArtifact,
} from "../../api/schema";
import { SLOTS } from "../../api/schema";
import { useRunStage } from "../../api/queries";
import { useTransitions } from "../../motion/prefs";
import { StaleVeil, veilFor } from "../../motion/StaleVeil";
import { Link } from "../../router";
import { cx, fmtClock, fmtInt } from "../../util/format";
import { V } from "../Prose";
import { StageRetry, needsRetry } from "../StageRetry";
import { DecisionSentence, History, Pending, QuestionBlock, SkipRow } from "./blocks";
import { FindingsList } from "./Findings";
import { LensAnswers, PurposeAnswers, RefusalNote, TargetAnswers, TaskAnswers } from "./questions";
import { sentence, slotOf } from "./sentences";
import c from "./controls.module.css";
import styles from "./Record.module.css";

interface Props {
  pid: string;
  view: ProjectView;
  ingest?: StageResult<DatasetInfo>;
  profile?: StageResult<ProfileArtifact>;
  targetInfo?: StageResult<TargetInfoArtifact>;
  findings?: StageResult<FindingsArtifact>;
  summaries?: ColumnSummary[];
}

/**
 * What a section says while its stage has no result to show. Never claims work that is
 * not happening; work that will not restart by itself says why and offers `retry`.
 */
function waiting(
  status: StageStatus | undefined,
  working: ReactNode,
  retry?: ReactNode,
): ReactNode {
  switch (status?.status) {
    case "queued":
    case "running":
      return working;
    case "error":
      return (
        <>
          This did not finish: {status.error ?? "the server gave no reason"}. {retry}
        </>
      );
    case "blocked":
      return <>Waits on an answer to: {status.missing.join(", ")}.</>;
    default:
      if (status?.cancelled) return <>You stopped this before it finished. {retry}</>;
      return <>Not computed yet.</>;
  }
}

/**
 * The file could not be read, or reading it was stopped: nothing can be asked about a
 * table that does not exist yet, so this stands where the first question would.
 */
function Unread({ pid, status }: { pid: string; status: StageStatus }) {
  const run = useRunStage(pid);
  const failed = status.status === "error";
  return (
    <QuestionBlock
      layoutId="ingest-unread"
      kicker="The file"
      title={failed ? "This file could not be read." : "Reading the file was stopped."}
      why={
        failed
          ? "Nothing can be asked about a table TurboTab has not read. The reader stopped here:"
          : "You stopped it before the table was ready. Nothing can be asked about the table until it is read."
      }
      testId="ingest-unread"
    >
      {failed ? (
        <pre className={styles.reason}>{status.error ?? "The server gave no reason."}</pre>
      ) : null}
      <div className={c.actions}>
        <button
          type="button"
          className={c.primary}
          disabled={run.isPending}
          onClick={() => run.mutate("ingest")}
          title="Reads the same file again, from where it is on disk."
        >
          {failed ? "Try reading it again" : "Read it again"}
        </button>
        <Link href="/" className={cx(c.ghost, styles.linkButton)}>
          Open another file
        </Link>
      </div>
    </QuestionBlock>
  );
}

const SUBJECT: Record<Slot, string> = {
  lens: "the lens",
  target: "the outcome",
  task: "the task",
  purpose: "the purpose",
  roles: "the column roles",
  energy_adjustment: "the energy adjustment",
  exclusions: "the exclusions",
  missing: "the missing values",
  split: "the split",
  models: "the models",
  substitution: "the substitution",
};

export function Record({ pid, view, ingest, profile, targetInfo, findings, summaries }: Props) {
  const decide = useDecide(pid);
  const t = useTransitions();
  const targetHeading = useId();
  const [reopened, setReopened] = useState<Partial<Record<Slot, boolean>>>({});
  const [refusal, setRefusal] = useState<{ slot: Slot; refusal: Refusal } | null>(null);
  const { state, decisions, stages } = view;

  const bySlot = useMemo(() => {
    const out = Object.fromEntries(SLOTS.map((s) => [s, []])) as unknown as Record<
      Slot,
      DecisionRecord[]
    >;
    for (const r of decisions) {
      const slot = slotOf(r.decision, decisions);
      if (slot) out[slot].push(r);
    }
    return out;
  }, [decisions]);

  const summaryMap = useMemo(
    () => (summaries ? new Map(summaries.map((s) => [s.name, s])) : undefined),
    [summaries],
  );

  const submit = (slot: Slot, decision: Decision) => {
    setRefusal(null);
    decide.mutate(decision, {
      onSuccess: () => setReopened((r) => ({ ...r, [slot]: false })),
      onError: (err) => {
        if (isRefusalError(err)) setRefusal({ slot, refusal: err.refusal });
      },
    });
  };
  const reopen = (slot: Slot) => {
    setRefusal(null);
    setReopened((r) => ({ ...r, [slot]: true }));
  };
  const keep = (slot: Slot) => () => {
    setRefusal(null);
    setReopened((r) => ({ ...r, [slot]: false }));
  };

  const refusalFor = (slot: Slot) =>
    refusal?.slot === slot ? (
      <RefusalNote
        refusal={refusal.refusal}
        onDismiss={() => setRefusal(null)}
        onExit={(exit) => (exit.decision ? submit(slot, exit.decision) : setRefusal(null))}
      />
    ) : null;

  /**
   * The record behind a slot's current value. For most slots that is the latest one. A
   * task answer names its column and counts only while that column is the outcome, so
   * the task's is the latest answer for this outcome with this value (none when unset).
   */
  const current = (slot: Slot): DecisionRecord | undefined => {
    const recs = bySlot[slot];
    if (slot !== "task") return recs[recs.length - 1];
    if (state.task === null) return undefined;
    return recs.findLast(
      (r) =>
        r.decision.kind === "set_task" &&
        r.decision.column === state.target &&
        r.decision.task === state.task,
    );
  };

  const historyFor = (slot: Slot) => {
    const now = current(slot);
    const earlier =
      slot === "task" ? bySlot.task.filter((r) => r !== now) : bySlot[slot].slice(0, -1);
    return earlier.map((r) => ({
      id: r.id,
      seq: r.seq,
      when: fmtClock(r.at),
      sentence: sentence(r, decisions),
      tag:
        r.decision.kind === "set_task" && r.decision.column !== state.target
          ? "another outcome"
          : undefined,
    }));
  };

  const settled = (slot: Slot, value: unknown) =>
    value !== null && !reopened[slot] && current(slot);
  const arrive = (slot: Slot) => bySlot[slot].length === 0;

  const hints = profile?.artifact?.lens_hints ?? [];
  const columns = ingest?.artifact?.columns.filter((col) => col.name !== "__row_id");
  // No table, and none on its way: the file failed to read, or reading it was stopped.
  const unread = !columns && needsRetry(stages.ingest) ? stages.ingest : undefined;
  const ti = targetInfo?.artifact ?? null;
  const tiVeil = veilFor(stages.target_info, targetInfo);
  const taskResolved =
    state.task !== null ||
    (ti !== null && ti.column === state.target && ti.confidence === "high" && tiVeil === "fresh");

  // ─── blocks ────────────────────────────────────────────────────────────────
  const lensBlock = (() => {
    const rec = settled("lens", state.lens);
    if (!rec && !columns) {
      // Questions about the table wait for the table.
      return unread ? null : (
        <Pending testId="pending-lens">
          Reading the file. The first question comes once its columns are known.
        </Pending>
      );
    }
    if (rec) {
      return (
        <DecisionSentence
          layoutId="slot-lens"
          subject={SUBJECT.lens}
          onChange={() => reopen("lens")}
          meta={`#${rec.seq}`}
          testId="decision-lens"
        >
          {sentence(rec, decisions)}
        </DecisionSentence>
      );
    }
    return (
      <QuestionBlock
        layoutId="slot-lens"
        kicker="Lens"
        title="What kind of measurements are in this table?"
        why="Pick all that apply. This changes what TurboTab looks for and what it suggests — it never limits what you can do."
        consumer="The structural diagnosis reads it first, because what looks malformed to a general-purpose import check is the expected shape for an assay panel. After that it sets priors on missingness, model ranking and which figure answers a question. Every default it raises states its reason and can be overturned."
        testId="question-lens"
      >
        <LensAnswers
          current={state.lens}
          hints={hints}
          pending={decide.isPending}
          onSubmit={(lenses) => submit("lens", { kind: "set_lens", lenses })}
          onKeep={state.lens ? keep("lens") : undefined}
        />
        {needsRetry(stages.profile) ? (
          <p className={styles.hintsMissing} data-testid="hints-missing">
            {stages.profile!.status === "error"
              ? `No lens hints: summarizing the columns did not finish (${stages.profile!.error ?? "no reason given"}).`
              : "No lens hints: you stopped the column summaries they come from."}{" "}
            <StageRetry pid={pid} status={stages.profile} />
          </p>
        ) : null}
        {refusalFor("lens")}
      </QuestionBlock>
    );
  })();

  const targetBlock = (() => {
    const rec = settled("target", state.target);
    if (rec) {
      return (
        <DecisionSentence
          layoutId="slot-target"
          subject={SUBJECT.target}
          onChange={() => reopen("target")}
          meta={`#${rec.seq}`}
          testId="decision-target"
        >
          {sentence(rec, decisions)}
        </DecisionSentence>
      );
    }
    return (
      <QuestionBlock
        layoutId="slot-target"
        kicker="Outcome"
        title={
          <span id={targetHeading}>
            Which column is the outcome you want to explain or predict?
          </span>
        }
        why="Everything downstream is built around it: the split, the models, and the findings that read the outcome."
        consumer="Task detection reads it next, then every stage that fits anything. Findings that depend on the outcome — energy adjustment, for one — are recomputed when it changes."
        arrive={arrive("target")}
        testId="question-target"
      >
        {columns ? (
          <TargetAnswers
            columns={columns}
            summaries={summaryMap}
            current={state.target}
            pending={decide.isPending}
            labelId={targetHeading}
            onSubmit={(column) => submit("target", { kind: "set_target", column })}
            onKeep={state.target ? keep("target") : undefined}
          />
        ) : unread ? (
          <Pending>There are no columns to choose from until the file is read.</Pending>
        ) : (
          <Pending>The file is still being read; its columns appear here when it is done.</Pending>
        )}
        {refusalFor("target")}
      </QuestionBlock>
    );
  })();

  const taskBlock = (() => {
    const rec = settled("task", state.task);
    let body: ReactNode;
    if (rec) {
      body = (
        <DecisionSentence
          layoutId="slot-task"
          subject={SUBJECT.task}
          onChange={() => reopen("task")}
          meta={`#${rec.seq}`}
          testId="decision-task"
        >
          {sentence(rec, decisions)}
          {ti && ti.column === state.target && ti.detected_task !== state.task ? (
            <>
              {" "}
              TurboTab had detected <V>{ti.detected_task}</V>.
            </>
          ) : null}
        </DecisionSentence>
      );
    } else if (ti && (reopened.task || ti.confidence !== "high")) {
      body = (
        <QuestionBlock
          layoutId="slot-task"
          kicker="Task"
          title={
            <>
              What kind of prediction is <V>{ti.column}</V>?
            </>
          }
          why="The task decides which models and which metrics apply."
          arrive={arrive("task") && !reopened.task}
          testId="question-task"
        >
          <TaskAnswers
            info={ti}
            current={state.task}
            pending={decide.isPending}
            onSubmit={(task) => submit("task", { kind: "set_task", column: ti.column, task })}
            onKeep={reopened.task ? keep("task") : undefined}
          />
          {refusalFor("task")}
        </QuestionBlock>
      );
    } else if (ti) {
      body = (
        <SkipRow layoutId="slot-task" onAsk={() => reopen("task")} testId="skip-task">
          <span className={styles.notAsked}>Not asked:</span> <V>{ti.column}</V> read as{" "}
          <V>{ti.detected_task}</V> — <V>{ti.confidence}</V> confidence, {ti.reason}.
        </SkipRow>
      );
    } else {
      body = (
        <Pending testId="pending-task">
          {waiting(
            stages.target_info,
            <>
              Reading <V>{state.target}</V> to detect the task…
            </>,
            <StageRetry pid={pid} status={stages.target_info} />,
          )}
        </Pending>
      );
    }
    return (
      <StaleVeil
        state={ti ? tiVeil : "fresh"}
        order={0}
        testId="veil-task"
        action={<StageRetry pid={pid} status={stages.target_info} />}
      >
        {body}
      </StaleVeil>
    );
  })();

  const purposeBlock = (() => {
    const rec = settled("purpose", state.purpose);
    if (rec) {
      return (
        <DecisionSentence
          layoutId="slot-purpose"
          subject={SUBJECT.purpose}
          onChange={() => reopen("purpose")}
          meta={`#${rec.seq}`}
          testId="decision-purpose"
        >
          {sentence(rec, decisions)}
        </DecisionSentence>
      );
    }
    return (
      <QuestionBlock
        layoutId="slot-purpose"
        kicker="Purpose"
        title="Is this analysis for prediction or for inference?"
        why="The two lead to different models, different checks and a different methods section."
        consumer="Model ranking and the Results panel read it: prediction is judged on held-out rows, inference on estimates and their uncertainty."
        arrive={arrive("purpose")}
        testId="question-purpose"
      >
        <PurposeAnswers
          current={state.purpose}
          pending={decide.isPending}
          onSubmit={(purpose) => submit("purpose", { kind: "set_purpose", purpose })}
          onKeep={state.purpose ? keep("purpose") : undefined}
        />
        {refusalFor("purpose")}
      </QuestionBlock>
    );
  })();

  const findingsBody = (() => {
    const f = findings?.artifact;
    const status = stages.findings;
    if (f) {
      return (
        <StaleVeil
          state={veilFor(status, findings)}
          order={1}
          testId="veil-findings"
          action={<StageRetry pid={pid} status={status} />}
        >
          <FindingsList artifact={f} />
        </StaleVeil>
      );
    }
    return (
      <Pending testId="pending-findings">
        {waiting(
          status,
          "Checking the table against the chosen lenses…",
          <StageRetry pid={pid} status={status} />,
        )}
      </Pending>
    );
  })();

  const slots: { slot: Slot; show: boolean; node: ReactNode }[] = [
    { slot: "lens", show: true, node: lensBlock },
    { slot: "target", show: state.lens !== null || state.target !== null, node: targetBlock },
    { slot: "task", show: state.target !== null, node: taskBlock },
    {
      slot: "purpose",
      show: state.purpose !== null || (state.target !== null && taskResolved),
      node: purposeBlock,
    },
  ];

  const nFindings = findings?.artifact?.findings.length;
  const allAnswered = state.lens && state.target && state.purpose && taskResolved;

  return (
    <LayoutGroup id="record">
      <div className={styles.record}>
        <header className={styles.opener}>
          <h1 className={styles.h1}>The record</h1>
          <p className={styles.lede}>
            Every answer is written down here as a sentence you could publish. Change any of them;
            nothing earlier is deleted.
          </p>
        </header>
        {unread ? <Unread pid={pid} status={unread} /> : null}
        {slots
          .filter((s) => s.show)
          .map((s) => (
            <motion.div
              key={s.slot}
              layout="position"
              transition={{ layout: t.settle }}
              className={styles.slot}
              data-slot={s.slot}
            >
              {s.node}
              <History items={historyFor(s.slot)} />
            </motion.div>
          ))}
        {allAnswered ? (
          <motion.p layout="position" transition={{ layout: t.settle }} className={styles.closing}>
            That is everything this version asks. Eligibility and the train/test split come next.
          </motion.p>
        ) : null}
        {state.lens ? (
          <motion.section
            layout="position"
            transition={{ layout: t.settle }}
            className={styles.findings}
            aria-labelledby="findings-heading"
            data-testid="findings"
          >
            <div className={styles.sectionHead}>
              <h2 id="findings-heading" className={styles.h2}>
                Noticed in this table
              </h2>
              {nFindings !== undefined ? (
                <span className={styles.count}>{fmtInt(nFindings)}</span>
              ) : null}
            </div>
            {findingsBody}
          </motion.section>
        ) : null}
      </div>
    </LayoutGroup>
  );
}
