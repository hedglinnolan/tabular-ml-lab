/**
 * The Record: the interview as a growing document, rendered from the server's Router
 * (ProjectView.interview). The client never decides the order; it renders what the Router
 * says — the one open question, the answered ones settled into the sentences the server
 * authored, the skipped and the not applicable with their reasons, and what comes next.
 *
 * "change" reopens an answer; recording appends a new decision and the old sentence stays
 * in the history. A press that does not record answers at the control. After a settle the
 * arriving question takes the keyboard focus, and the recorded sentence is announced.
 * Below the questions: what the lenses noticed, each finding a claim plus its lever.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { LayoutGroup, motion } from "motion/react";
import { isRefusalError } from "../../api/client";
import { useColumnSummaries, useDecide, useRunStage, useTeaching } from "../../api/queries";
import type { InterviewStep, QuestionKey, Role, TeachingEntry } from "../../api/m1-types";
import type { Decision, DecisionRecord, ProjectView, Refusal, StageStatus } from "../../api/schema";
import { DUR, useTransitions } from "../../motion/prefs";
import { StaleVeil, veilFor } from "../../motion/StaleVeil";
import { Link } from "../../router";
import { useStageFocus } from "../../state/focus";
import { useStage } from "../../state/stages";
import { fmtClock, fmtInt } from "../../util/format";
import { Prose, V } from "../Prose";
import { STAGE_LABEL } from "../JobChips";
import { REOPEN_EVENT, StageFailure, rootFailure } from "../Failure";
import { DesignWarnings, warningsAbout } from "../stage/DesignWarnings";
import { StageRetry, needsRetry } from "../StageRetry";
import { DecisionSentence, History, Pending, SkipRow } from "./blocks";
import {
  EnergyAsk,
  ExclusionsAsk,
  MissingAsk,
  ModelsAsk,
  SplitAsk,
  SubstitutionAsk,
} from "./ask/ChoiceQuestions";
import type { AskProps } from "./ask/common";
import { LensAsk, PurposeAsk, TargetAsk, TaskAsk } from "./ask/FactQuestions";
import { RolesAsk } from "./ask/RolesAsk";
import { FindingsCards } from "./Findings";
import { FailureNote, RefusalNote } from "./Refusal";
import { sentence, sentenceText, slotOf } from "./sentences";
import { ConceptDrawer } from "./teach";
import s from "./Record.module.css";

/** Each question's name in running text ("Change the column roles"). */
const SUBJECT: Record<QuestionKey, string> = {
  lens: "the lens",
  target: "the outcome",
  task: "the task",
  purpose: "the purpose",
  roles: "the column roles",
  exclusions: "the exclusions",
  missing: "the missing values",
  split: "the split",
  energy_adjustment: "the energy adjustment",
  models: "the models",
  substitution: "the substitution",
};

/** What each stage is doing while a question waits on it: the job chip's own words. */
const STAGE_WORK = STAGE_LABEL;

/** The stage that carries out each answer: where "it did not run" is read. */
const RUNS_IN: Partial<Record<QuestionKey, string>> = {
  exclusions: "cohort",
  missing: "cohort",
  split: "split",
  energy_adjustment: "design",
  models: "fit",
  substitution: "substitution",
};
/** The stages each of those waits on: a stop inherited from them is said there, not here. */
const STAGE_DEPS: Record<string, string[]> = {
  cohort: [],
  split: ["cohort"],
  design: ["split"],
  fit: ["design", "split"],
  substitution: ["fit", "design"],
};
/** Sentences whose counts are made under earlier answers, and which ones. */
const COUNTED_UNDER: Partial<Record<QuestionKey, QuestionKey[]>> = {
  exclusions: ["target"],
  missing: ["target", "roles", "exclusions"],
};

interface Answer {
  key: QuestionKey;
  at: string;
  refusal?: Refusal;
  failure?: string;
}

const titleOf = (entries: Map<string, TeachingEntry>, key: QuestionKey) =>
  entries.get(key)?.title ?? key.replace(/_/g, " ");

export function Record({ pid, view }: { pid: string; view: ProjectView }) {
  const decide = useDecide(pid);
  const teaching = useTeaching();
  const t = useTransitions();
  const { reset } = useStageFocus();
  const { state, decisions, stages, interview } = view;

  const ingest = useStage(pid, view, "ingest");
  const profile = useStage(pid, view, "profile");
  const targetInfo = useStage(pid, view, "target_info");
  const findings = useStage(pid, view, "findings");
  const roles = useStage(pid, view, "roles");
  const proposals = useStage(pid, view, "proposals");
  const shelf = useStage(pid, view, "shelf");
  const design = useStage(pid, view, "design");
  const summaries = useColumnSummaries(pid, stages.ingest?.status === "fresh").data;

  const [reopened, setReopened] = useState<Partial<Record<QuestionKey, boolean>>>({});
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [arrival, setArrival] = useState<{ from: QuestionKey } | null>(null);
  const [announce, setAnnounce] = useState("");
  const [drawer, setDrawer] = useState<QuestionKey | null>(null);
  const [flash, setFlash] = useState<QuestionKey | null>(null);
  const slots = useRef(new Map<QuestionKey, HTMLElement>());
  const onArrived = useCallback(() => setArrival(null), []);

  const entries = useMemo(
    () => new Map((teaching.data ?? []).map((e) => [e.key, e])),
    [teaching.data],
  );
  const byId = useMemo(() => new Map(decisions.map((r) => [r.id, r])), [decisions]);
  const bySlot = useMemo(() => {
    const out = new Map<string, DecisionRecord[]>();
    for (const r of decisions) {
      const slot = slotOf(r.decision, decisions);
      if (!slot) continue;
      out.set(slot, [...(out.get(slot) ?? []), r]);
    }
    return out;
  }, [decisions]);
  const summaryMap = useMemo(
    () => (summaries ? new Map(summaries.map((x) => [x.name, x])) : undefined),
    [summaries],
  );

  const stepOf = (key: QuestionKey) => interview.find((st) => st.key === key);
  const openStep = interview.find((st) => st.status === "open");
  const reopenedKeys = interview.filter((st) => reopened[st.key]).map((st) => st.key);
  const nowKey = reopenedKeys[0] ?? openStep?.key ?? null;
  const arrivingKey =
    arrival && openStep && openStep.key !== arrival.from && !reopenedKeys.length
      ? openStep.key
      : null;

  /** Move to a question: the user asked for it (a lever, "change"). */
  const goTo = (key: QuestionKey, focusHeading: boolean) => {
    window.setTimeout(() => {
      const el = slots.current.get(key);
      if (!el) return;
      el.scrollIntoView({ block: "nearest", behavior: t.reduced ? "auto" : "smooth" });
      const target = focusHeading ? el.querySelector<HTMLElement>("h2[tabindex]") : el;
      target?.focus({ preventScroll: true });
    }, 30);
  };

  const record = (key: QuestionKey, decision: Decision, at = "") => {
    setAnswer(null);
    const wasReopened = !!reopened[key] && stepOf(key)?.status !== "open";
    decide.mutate(decision, {
      onSuccess: (next) => {
        setReopened((r) => ({ ...r, [key]: false }));
        const latest = next.decisions.reduce<DecisionRecord | null>(
          (m, r) => (m === null || r.seq > m.seq ? r : m),
          null,
        );
        if (latest) setAnnounce(`Recorded: ${sentenceText(latest)}`);
        reset();
        if (wasReopened) {
          // A changed earlier answer settles in place; focus stays there, nothing scrolls away.
          window.setTimeout(
            () => {
              slots.current
                .get(key)
                ?.querySelector<HTMLElement>('[data-block="decision"] button')
                ?.focus({ preventScroll: true });
            },
            t.reduced ? 0 : DUR.settle * 1000,
          );
        } else {
          setArrival({ from: key });
        }
      },
      onError: (err) => {
        if (isRefusalError(err)) setAnswer({ key, at, refusal: err.refusal });
        else setAnswer({ key, at, failure: err instanceof Error ? err.message : String(err) });
      },
    });
  };

  const reopen = (key: QuestionKey) => {
    setAnswer(null);
    setArrival(null); // the user went somewhere: nothing arrives behind their back
    setReopened((r) => ({ ...r, [key]: true }));
    goTo(key, true);
  };
  const keepFor = (key: QuestionKey) =>
    reopened[key]
      ? () => {
          setAnswer(null);
          setReopened((r) => ({ ...r, [key]: false }));
          reset();
        }
      : undefined;

  const route = (key: QuestionKey) => {
    const st = stepOf(key);
    if (!st) return;
    setArrival(null);
    if (st.status === "answered" || st.status === "skipped") {
      setReopened((r) => ({ ...r, [key]: true }));
    }
    setFlash(key);
    window.setTimeout(() => setFlash((f) => (f === key ? null : f)), 1600);
    goTo(key, st.status !== "waiting" && st.status !== "not_applicable");
  };
  // "Change the energy adjustment" pressed on the stage or the banner reopens it here.
  const routeRef = useRef(route);
  useEffect(() => {
    routeRef.current = route;
  });
  useEffect(() => {
    const onReopen = (e: Event) => routeRef.current((e as CustomEvent<QuestionKey>).detail);
    window.addEventListener(REOPEN_EVENT, onReopen);
    return () => window.removeEventListener(REOPEN_EVENT, onReopen);
  }, []);

  const answerFor = (key: QuestionKey): AskProps["answerAt"] => {
    if (!answer || answer.key !== key) return null;
    const dismiss = () => setAnswer(null);
    const node = answer.refusal ? (
      <RefusalNote
        refusal={answer.refusal}
        onDismiss={dismiss}
        onExit={(exit) => (exit.decision ? record(key, exit.decision, answer.at) : dismiss())}
      />
    ) : (
      <FailureNote message={answer.failure ?? "no reason given"} onDismiss={dismiss} />
    );
    return { key: answer.at, node };
  };

  const props = (key: QuestionKey): AskProps => ({
    entry: entries.get(key),
    pending: decide.isPending,
    record: (d, at) => record(key, d, at),
    keep:
      keepFor(key) && (currentRecord(key) || stepOf(key)?.status === "skipped")
        ? keepFor(key)
        : undefined,
    answerAt: answerFor(key),
    shell: {
      qkey: key,
      now: nowKey === key,
      arriving: arrivingKey === key,
      onArrived,
      onOpenDrawer: () => setDrawer(key),
      reopened: !!reopened[key],
    },
  });

  /** The live record behind a question's answer: the Router names it. */
  const currentRecord = (key: QuestionKey): DecisionRecord | undefined => {
    const id = stepOf(key)?.decision_id;
    if (id) return byId.get(id);
    const recs = bySlot.get(key) ?? [];
    return key === "task" ? undefined : recs[recs.length - 1];
  };

  const historyFor = (key: QuestionKey) => {
    const now = currentRecord(key);
    return (bySlot.get(key) ?? [])
      .filter((r) => r !== now)
      .map((r) => ({
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

  // ─── the open (or reopened) question, by key ──────────────────────────────
  const columns = ingest?.artifact?.columns.filter((col) => col.name !== "__row_id");
  const ti = targetInfo?.artifact ?? null;
  const waitingFor = (stage: string, then: string): ReactNode => (
    <Pending>
      {STAGE_WORK[stage] ?? "Computing"}… {then}
    </Pending>
  );

  const ask = (key: QuestionKey): ReactNode => {
    const p = props(key);
    switch (key) {
      case "lens":
        return (
          <LensAsk
            {...p}
            current={state.lens}
            hints={profile?.artifact?.lens_hints ?? []}
            hintsNote={
              needsRetry(stages.profile) ? (
                <p className={s.hintsMissing}>
                  No lens hints: summarizing the columns did not finish.{" "}
                  <StageRetry pid={pid} status={stages.profile} />
                </p>
              ) : undefined
            }
          />
        );
      case "target":
        return <TargetAsk {...p} columns={columns} summaries={summaryMap} current={state.target} />;
      case "task":
        return ti ? (
          <TaskAsk {...p} info={ti} current={state.task} />
        ) : (
          waitingFor("target_info", "Then the task is read from it.")
        );
      case "purpose":
        return <PurposeAsk {...p} current={state.purpose} />;
      case "roles":
        return roles?.artifact ? (
          <RolesAsk
            key={roles.key ?? "roles"}
            {...p}
            artifact={roles.artifact}
            current={(state.roles as Record<string, Role> | null) ?? null}
          />
        ) : (
          waitingFor("roles", "Then each column's role is asked.")
        );
      case "exclusions":
        return (
          <ExclusionsAsk
            {...p}
            proposals={proposals?.artifact ?? undefined}
            current={state.exclusions}
            numericColumns={(columns ?? [])
              .filter((col) => col.dtype === "numeric" || col.dtype === "integer")
              .map((col) => col.name)}
          />
        );
      case "missing":
        return (
          <MissingAsk
            {...p}
            proposals={proposals?.artifact ?? undefined}
            current={state.missing}
          />
        );
      case "split":
        return <SplitAsk {...p} roles={roles?.artifact ?? undefined} current={state.split} />;
      case "energy_adjustment":
        return (
          <EnergyAsk
            {...p}
            reading={proposals?.artifact?.energy}
            current={state.energy_adjustment}
            leftOut={state.missing?.drop_columns ?? []}
          />
        );
      case "models":
        return shelf?.artifact ? (
          <ModelsAsk {...p} shelf={shelf.artifact} current={state.models} />
        ) : (
          waitingFor("shelf", "Then the model families are offered.")
        );
      case "substitution":
        return (
          <SubstitutionAsk {...p} pairs={design?.artifact?.substitution_pairs.length ?? null} />
        );
    }
  };

  // ─── one slot per step, in the Router's order ─────────────────────────────
  /** What is true of a recorded answer now that its sentence cannot say. */
  const sentenceNote = (key: QuestionKey, rec: DecisionRecord): ReactNode => {
    const stage = RUNS_IN[key];
    if (stage) {
      const failure = rootFailure(stages, stage);
      if (failure && failure.stage === stage) {
        return (
          <StageFailure
            pid={pid}
            view={view}
            stage={stage}
            compact
            changeable={false}
            lead="Recorded, but it did not run."
            testId={`sentence-failure-${key}`}
          />
        );
      }
      const inherited = (STAGE_DEPS[stage] ?? []).some((u) => stages[u]?.cancelled);
      if (stages[stage]?.cancelled && !inherited) {
        return (
          <StageFailure
            pid={pid}
            view={view}
            stage={stage}
            compact
            changeable={false}
            lead="Recorded."
            testId={`sentence-stopped-${key}`}
          />
        );
      }
    }
    if (key === "energy_adjustment" && design?.artifact && design.fresh && design.key === stages.design?.key) {
      // The design's energy concerns, beside the answer that raised them.
      const about = warningsAbout(design.artifact.warnings, "energy");
      if (about.length) return <DesignWarnings warnings={about} title="Concerns" testId="energy-warnings" />;
    }
    const under = COUNTED_UNDER[key];
    if (under) {
      const later = decisions
        .filter((r) => r.seq > rec.seq && under.includes(slotOf(r.decision, decisions) as QuestionKey))
        .sort((a, b) => b.seq - a.seq)[0];
      if (later) {
        const slot = slotOf(later.decision, decisions) as QuestionKey;
        return (
          <>
            Its counts were made before #{later.seq} changed {SUBJECT[slot]}; the banner shows the
            rows as they are now.
          </>
        );
      }
    }
    return null;
  };

  const settled = (key: QuestionKey, rec: DecisionRecord) => (
    <DecisionSentence
      layoutId={`q-${key}`}
      subject={SUBJECT[key]}
      onChange={() => reopen(key)}
      meta={`#${rec.seq}`}
      note={sentenceNote(key, rec)}
      testId={`decision-${key}`}
    >
      {sentence(rec, decisions)}
    </DecisionSentence>
  );

  const slotBody = (st: InterviewStep): ReactNode => {
    const key = st.key;
    if (reopened[key] && st.status !== "waiting") return ask(key);
    switch (st.status) {
      case "open":
        return ask(key);
      case "answered": {
        const rec = currentRecord(key);
        return rec ? settled(key, rec) : null;
      }
      case "skipped":
        return (
          <SkipRow layoutId={`q-${key}`} onAsk={() => reopen(key)} testId={`skip-${key}`}>
            <span className={s.notAsked}>Not asked:</span>{" "}
            {ti && key === "task" ? (
              <>
                <V>{ti.column}</V> read as <V>{ti.task}</V>, <V>{ti.confidence}</V> confidence.{" "}
              </>
            ) : null}
            {st.reason ? <Prose text={st.reason} /> : null}
          </SkipRow>
        );
      case "not_applicable":
        return (
          <div className={s.na} data-testid={`na-${key}`}>
            <span className={s.naTitle}>{titleOf(entries, key)}</span>
            <span className={s.naTag}>not applicable</span>
            {st.reason ? (
              <span className={s.naText}>
                <Prose text={st.reason} />
              </span>
            ) : null}
          </div>
        );
      case "waiting":
        return null;
    }
  };

  // Answers always stay in place (nothing earlier is deleted). The first unanswered question,
  // when it waits on a stage, stands in place as a pending row; every later unanswered or
  // inapplicable step is listed under "Then".
  const firstAt = interview.findIndex((st) => st.status === "open" || st.status === "waiting");
  const firstUnanswered = firstAt === -1 ? undefined : interview[firstAt];
  const pendingStep =
    firstUnanswered?.status === "waiting" &&
    firstUnanswered.waiting_on.every((w) => !interview.some((x) => x.key === w))
      ? firstUnanswered
      : null;
  const later = (st: InterviewStep, i: number) =>
    st !== pendingStep &&
    (st.status === "waiting" || (st.status === "not_applicable" && firstAt !== -1 && i > firstAt));
  const inline = interview.filter((st, i) => !later(st, i) && st !== pendingStep);
  const next = interview.filter(later);

  /** A finding is settled once the question its lever routes to has a recorded answer. */
  const answeredBy = (f: { routes_to: QuestionKey | null }) => {
    if (!f.routes_to || stepOf(f.routes_to)?.status !== "answered") return null;
    const rec = currentRecord(f.routes_to);
    if (!rec) return null;
    const text = rec.sentence ?? sentenceText(rec);
    const head = text.split(/(?<=[.;:])\s/)[0]!.replace(/[.;:]$/, "");
    const words = head.split(/\s+/);
    return { seq: rec.seq, said: words.length > 14 ? `${words.slice(0, 14).join(" ")}…` : head };
  };
  const unread = !columns && needsRetry(stages.ingest) ? stages.ingest : undefined;
  const findingsBody = (() => {
    const f = findings?.artifact;
    if (f) return <FindingsCards artifact={f} onRoute={route} answeredBy={answeredBy} />;
    return (
      <Pending testId="pending-findings">
        {waitingText(
          stages.findings,
          "Checking the table against the chosen lenses…",
          <StageRetry pid={pid} status={stages.findings} />,
        )}
      </Pending>
    );
  })();
  const nFindings = findings?.artifact?.findings.length;
  const drawerEntry = drawer ? entries.get(drawer) : undefined;

  return (
    <LayoutGroup id="record">
      <div className={s.record}>
        <header className={s.opener}>
          <h1 className={s.h1}>The record</h1>
          <p className={s.lede}>
            Every answer is written down as a sentence you could publish. Change any of them;
            nothing earlier is deleted.
          </p>
        </header>
        <p className="visually-hidden" aria-live="polite" role="status" data-testid="announce">
          {announce}
        </p>
        {unread ? <Unread pid={pid} status={unread} /> : null}
        {interview.length === 0 && !unread ? (
          <Pending>Reading the file. The first question comes once its columns are known.</Pending>
        ) : null}
        {interview.map((st) => {
          if (st !== pendingStep && !inline.includes(st)) return null;
          const waitsOn = st.waiting_on[0] ?? "";
          const stopped = rootFailure(stages, waitsOn) !== null || !!stages[waitsOn]?.cancelled;
          const body =
            st === pendingStep ? (
              stopped ? (
                // Never "Fitting the models…" over a stage that failed or was stopped.
                <StageFailure
                  pid={pid}
                  view={view}
                  stage={waitsOn}
                  after={<>{titleOf(entries, st.key)} waits on it.</>}
                  testId={`pending-failure-${st.key}`}
                />
              ) : (
                <Pending testId={`pending-${st.key}`}>
                  {STAGE_WORK[waitsOn] ?? "Computing"}… {titleOf(entries, st.key)} comes next.
                </Pending>
              )
            ) : (
              slotBody(st)
            );
          if (!body) return null;
          return (
            <motion.div
              key={st.key}
              ref={(el: HTMLDivElement | null) => {
                if (el) slots.current.set(st.key, el);
                else slots.current.delete(st.key);
              }}
              layout="position"
              transition={{ layout: t.settle }}
              className={s.slot}
              data-slot={st.key}
              data-status={st.status}
              data-flash={flash === st.key || undefined}
            >
              {body}
              <History items={historyFor(st.key)} />
            </motion.div>
          );
        })}
        {next.length > 0 ? (
          <motion.section
            layout="position"
            transition={{ layout: t.settle }}
            className={s.next}
            aria-label="Asked next"
            data-testid="next"
          >
            <h2 className={s.nextHead}>Then</h2>
            <ol className={s.nextList}>
              {next.map((st) => (
                <li
                  key={st.key}
                  ref={(el) => {
                    if (el) slots.current.set(st.key, el);
                    else slots.current.delete(st.key);
                  }}
                  tabIndex={-1}
                  className={s.nextItem}
                  data-slot={st.key}
                  data-status={st.status}
                  data-flash={flash === st.key || undefined}
                >
                  <span className={s.nextTitle}>{titleOf(entries, st.key)}</span>
                  {st.status === "not_applicable" ? (
                    <span className={s.naTag}>not applicable</span>
                  ) : null}
                  {flash === st.key && st.status === "waiting" && firstUnanswered ? (
                    <span className={s.nextNote}>
                      asked after {titleOf(entries, firstUnanswered.key).toLowerCase()}
                    </span>
                  ) : null}
                </li>
              ))}
            </ol>
          </motion.section>
        ) : null}
        {state.lens ? (
          <motion.section
            layout="position"
            transition={{ layout: t.settle }}
            className={s.findings}
            aria-labelledby="findings-heading"
            data-testid="findings"
          >
            <StaleVeil
              state={veilFor(stages.findings, findings)}
              order={1}
              testId="veil-findings"
              action={<StageRetry pid={pid} status={stages.findings} />}
            >
              <div className={s.sectionHead}>
                <h2 id="findings-heading" className={s.h2}>
                  Noticed in this table
                </h2>
                {nFindings !== undefined ? (
                  <span className={s.count} data-testid="findings-count">
                    {fmtInt(nFindings)}
                  </span>
                ) : null}
              </div>
              {findingsBody}
            </StaleVeil>
          </motion.section>
        ) : null}
      </div>
      {drawerEntry?.drawer ? (
        <ConceptDrawer entry={drawerEntry} onClose={() => setDrawer(null)} />
      ) : null}
    </LayoutGroup>
  );
}

/** What a section says while its stage has no result; never claims work that is not happening. */
function waitingText(
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

/** The file could not be read, or reading it was stopped: nothing can be asked yet. */
function Unread({ pid, status }: { pid: string; status: StageStatus }) {
  const run = useRunStage(pid);
  const failed = status.status === "error";
  return (
    <section className={s.unread} data-testid="ingest-unread" aria-labelledby="unread-h">
      <div className={s.unreadKicker}>The file</div>
      <h2 id="unread-h" className={s.unreadTitle}>
        {failed ? "This file could not be read." : "Reading the file was stopped."}
      </h2>
      <p className={s.unreadWhy}>
        {failed
          ? "Nothing can be asked about a table TurboTab has not read. The reader stopped here:"
          : "You stopped it before the table was ready. Nothing can be asked about the table until it is read."}
      </p>
      {failed ? (
        <pre className={s.reason}>{status.error ?? "The server gave no reason."}</pre>
      ) : null}
      <div className={s.unreadActions}>
        <button
          type="button"
          className={s.primary}
          disabled={run.isPending}
          onClick={() => run.mutate("ingest")}
          title="Reads the same file again, from where it is on disk."
        >
          {failed ? "Try reading it again" : "Read it again"}
        </button>
        <Link href="/" className={s.ghost}>
          Open another file
        </Link>
      </div>
    </section>
  );
}
