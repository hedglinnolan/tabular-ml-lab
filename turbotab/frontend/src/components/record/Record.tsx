/**
 * The Record: the interview as a growing document, rendered from the server's Router
 * (ProjectView.interview). The client never decides the order; it renders what the Router
 * says — the one open question, the answered ones settled into the sentences the server
 * authored, the skipped and the not applicable with their reasons, and what comes next.
 *
 * M2 (M2_CONTRACT §10): the opening sequence's questions in the Router's order; the repairs on
 * findings before the outcome question (each previews on the stage, then apply, hold for its
 * question, or keep as is); findings held for a question resurface inside it, pre-checked; the
 * seal states its basis; and opening the seal is the Router's last step, a CONSEQUENCE card.
 *
 * "change" reopens an answer; recording appends a new decision and the old sentence stays
 * in the history. A press that does not record answers at the control. After a settle the
 * arriving question takes the keyboard focus, and the recorded sentence is announced.
 * Below the questions: what the lenses noticed, each finding a claim plus its lever.
 */
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { LayoutGroup, motion } from "motion/react";
import { isRefusalError } from "../../api/client";
import { useColumnSummaries, useDecide, useRunStage, useTeaching } from "../../api/queries";
import type { InterviewStep, QuestionKey, Role, TeachingEntry } from "../../api/m1-types";
import { NOT_YET } from "../../api/m2-types";
import type {
  Decision,
  DecisionRecord,
  Finding,
  ProjectView,
  Refusal,
  StageStatus,
} from "../../api/schema";
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
import { DecisionSentence, History, Pending } from "./blocks";
import {
  EnergyAsk,
  ExclusionsAsk,
  MissingAsk,
  ModelsAsk,
  SubstitutionAsk,
  SurveyAsk,
} from "./ask/ChoiceQuestions";
import type { AskProps } from "./ask/common";
import { LensAsk, PurposeAsk, TargetAsk, TaskAsk } from "./ask/FactQuestions";
import { RolesAsk } from "./ask/RolesAsk";
import { OpenSealStep, SealAsk } from "./ask/SealQuestions";
import {
  AggregationAsk,
  EventAsk,
  GrainAsk,
  OrientationAsk,
  RepeatKindAsk,
  TemporalAsk,
  UnitAsk,
} from "./ask/SequenceQuestions";
import { AskCard } from "./ask/AskCard";
import { FindingsCards } from "./Findings";
import { layoutFlow } from "./flow";
import { compose, taskFollowup } from "./generic/compose";
import { GenericAsk } from "./generic/GenericAsk";
import { findingState, followUps, heldFor, type DeferredChoice } from "./findingState";
import { FailureNote, RefusalNote } from "./Refusal";
import { RepairsSection, Resurfaced, type RepairAnswer } from "./Repairs";
import { SealGlyph } from "./seal/SealGlyph";
import { sentence, sentenceText, slotOf } from "./sentences";
import { StatedSkip } from "./StatedSkip";
import { ConceptDrawer } from "./teach";
import s from "./Record.module.css";
import k from "./ask/seal.module.css";

/** Each question's name in running text ("Change the column roles"). */
export const SUBJECT: Record<QuestionKey, string> = {
  lens: "the lens",
  orientation: "the table's orientation",
  target: "the outcome",
  event: "the event level",
  task: "the task",
  follow_up: "the follow-up",
  purpose: "the purpose",
  grain: "the grain",
  repeat_kind: "repeats or time points",
  unit: "the unit of analysis",
  aggregation: "how rows are combined",
  temporal: "temporal prediction",
  roles: "the column roles",
  clusters: "the grouping",
  survey: "the survey answer",
  estimand: "the exposure and its effect",
  adjustment: "the adjustment set",
  time_varying: "the time-varying exposure",
  exclusions: "the eligibility",
  missing: "the missing values",
  split: "the seal",
  energy_adjustment: "the energy adjustment",
  causal: "the causal estimate",
  models: "the models",
  substitution: "the substitution",
  open_seal: "opening the seal",
};

/** What each stage is doing while a question waits on it: the job chip's own words. */
const STAGE_WORK = STAGE_LABEL;

/** The stage that carries out each answer: where "it did not run" is read. */
const RUNS_IN: Partial<Record<QuestionKey, string>> = {
  orientation: "oriented",
  aggregation: "working",
  exclusions: "cohort",
  missing: "cohort",
  split: "split",
  energy_adjustment: "design",
  models: "fit",
  substitution: "substitution",
};
/** The stages each of those waits on: a stop inherited from them is said there, not here. */
const STAGE_DEPS: Record<string, string[]> = {
  oriented: [],
  working: ["oriented"],
  cohort: ["working"],
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
/** The slot each step's answer is written to (the seal's opening writes `seal_opened`). */
const slotKey = (key: QuestionKey): string => (key === "open_seal" ? "seal_opened" : key);

interface Answer {
  /** A question key, "repairs", or "open_seal". */
  key: string;
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
  const { state, decisions, stages } = view;
  // §12.1: the Router's last step is "open the seal".
  const interview: readonly InterviewStep[] = view.interview;

  const ingest = useStage(pid, view, "ingest");
  const oriented = useStage(pid, view, "oriented");
  const structure = useStage(pid, view, "structure");
  const profile = useStage(pid, view, "profile");
  const targetInfo = useStage(pid, view, "target_info");
  const findings = useStage(pid, view, "findings");
  const roles = useStage(pid, view, "roles");
  const proposals = useStage(pid, view, "proposals");
  const sealPlan = useStage(pid, view, "seal_plan");
  const split = useStage(pid, view, "split");
  const shelf = useStage(pid, view, "shelf");
  const design = useStage(pid, view, "design");
  // The generic question's cards (compose.ts): the causal lane's and the time-varying lane's.
  const causalDesign = useStage(pid, view, "causal_design");
  const timeVarying = useStage(pid, view, "time_varying");
  const summaries = useColumnSummaries(pid, stages.ingest?.status === "fresh").data;

  const [reopened, setReopened] = useState<Partial<Record<QuestionKey, boolean>>>({});
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [arrival, setArrival] = useState<{ from: QuestionKey } | null>(null);
  const [announce, setAnnounce] = useState("");
  const [drawer, setDrawer] = useState<QuestionKey | null>(null);
  const [flash, setFlash] = useState<QuestionKey | null>(null);
  const [held, setHeld] = useState<Record<string, DeferredChoice | undefined>>({});
  const slots = useRef(new Map<QuestionKey, HTMLElement>());
  const onArrived = useCallback(() => setArrival(null), []);

  const entries = useMemo(
    () => new Map<string, TeachingEntry>((teaching.data ?? []).map((e) => [e.key, e])),
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
  /** The question the Router asks first: where an answer refused as `not_yet` is sent. */
  const firstOpen = interview.find((st) => st.status === "open" || st.status === "waiting")?.key;
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

  /** Record a decision. `key` is the question it answers, or "repairs" for a finding's
   *  disposition (which settles in place: nothing arrives, the open question keeps its place). */
  const record = (key: QuestionKey | "repairs", decision: Decision, at = "") => {
    setAnswer(null);
    const stays = key === "repairs" || (!!reopened[key] && stepOf(key)?.status !== "open");
    decide.mutate(decision, {
      onSuccess: (next) => {
        if (key !== "repairs") setReopened((r) => ({ ...r, [key]: false }));
        const latest = next.decisions.reduce<DecisionRecord | null>(
          (m, r) => (m === null || r.seq > m.seq ? r : m),
          null,
        );
        if (latest) setAnnounce(`Recorded: ${sentenceText(latest)}`);
        reset();
        if (at === "undo") {
          // An answer taken back: its question is where the user looks next (asked again, or
          // settled on the answer before it).
          goTo(key as QuestionKey, true);
          return;
        }
        if (stays) {
          // A changed earlier answer, or a repair, settles in place; focus stays there.
          window.setTimeout(
            () => {
              const scope =
                key === "repairs"
                  ? document.querySelector<HTMLElement>('[data-testid="repairs"]')
                  : slots.current.get(key);
              const next =
                scope?.querySelector<HTMLElement>('[role="option"][tabindex="0"]') ??
                scope?.querySelector<HTMLElement>('[data-block="decision"] button');
              next?.focus({ preventScroll: true });
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

  /** The server's answer to a press that did not record, shown at the control it answers. */
  const answerFor = (key: QuestionKey | "repairs"): AskProps["answerAt"] => {
    if (!answer || answer.key !== key) return null;
    const at = answer.at;
    return {
      key: at,
      node: (
        <AnswerNote
          answer={answer}
          onDismiss={() => setAnswer(null)}
          onRetry={(d) => record(key, d, at)}
          // §12.2: an answer that waits behind an unanswered question; its exit goes there.
          onGo={
            answer.refusal?.error.code === NOT_YET && firstOpen
              ? () => {
                  setAnswer(null);
                  window.dispatchEvent(new CustomEvent(REOPEN_EVENT, { detail: firstOpen }));
                }
              : undefined
          }
        />
      ),
    };
  };

  // ── findings: the ones with repairs, the ones held for a question ──────────
  const allFindings: Finding[] = useMemo(() => findings?.artifact?.findings ?? [], [findings]);
  const findingsReady = !!findings?.artifact;
  const resurfacedFor = (key: QuestionKey): ReactNode => {
    const ids = stepOf(key)?.deferred_findings;
    const list = heldFor(key, allFindings, decisions, ids && ids.length ? ids : undefined);
    if (!list.length) return null;
    return (
      <Resurfaced
        findings={list}
        decisions={decisions}
        choices={held}
        onChoice={(id, choice) => setHeld((h) => ({ ...h, [id]: choice }))}
      />
    );
  };

  // Recording a question that findings were held for also records what each resurfaced finding
  // was set to: its checked repair, or a dismissal (M2_CONTRACT §4). This follows the answer from
  // wherever it was recorded (the Record, the stage's record button), for answers made on this
  // visit only, so reopening a project never applies a repair nobody saw.
  const [since] = useState(() => decisions.reduce((m, r) => Math.max(m, r.seq), 0));
  const handled = useRef(new Set<string>());
  const decideAsync = decide.mutateAsync;
  useEffect(() => {
    if (!findingsReady) return;
    const todo: { key: QuestionKey; decision: Decision }[] = [];
    for (const st of interview) {
      if (st.key === "open_seal" || st.status !== "answered" || !st.decision_id) continue;
      const answerRec = byId.get(st.decision_id);
      if (!answerRec || answerRec.seq <= since) continue;
      for (const d of followUps(st.key, answerRec, allFindings, decisions, held)) {
        const tag = `${answerRec.id}:${JSON.stringify(d)}`;
        if (handled.current.has(tag)) continue;
        handled.current.add(tag);
        todo.push({ key: st.key, decision: d });
      }
    }
    if (!todo.length) return;
    void (async () => {
      for (const item of todo) {
        try {
          const next = await decideAsync(item.decision);
          const latest = next.decisions[next.decisions.length - 1];
          if (latest) setAnnounce((a) => `${a} ${sentenceText(latest)}`.trim());
        } catch (err) {
          if (isRefusalError(err)) setAnswer({ key: item.key, at: "held", refusal: err.refusal });
        }
      }
    })();
  }, [interview, decisions, allFindings, findingsReady, byId, held, decideAsync, since]);

  const coachFor = (key: QuestionKey): ReactNode => {
    if (key === "missing") return null; // its blanks and their reason are already on the card
    const note = proposals?.artifact?.coach?.[key];
    if (!note) return null;
    // Eligibility withholds the outcome's distribution (constitution §04): no note about it.
    if (note.anchor.kind === "column" && note.anchor.ref === state.target) return null;
    return <Prose text={note.text} />;
  };

  /** The ledger's one ask card on the open step that asks (BLUEPRINT §14.2), inside its question. */
  const askFor = (key: QuestionKey): ReactNode => {
    const st = stepOf(key);
    if (!st?.ask || st.status !== "open") return null;
    return (
      <AskCard
        card={st.ask}
        pending={decide.isPending}
        record={(d, at) => record(key, d, at)}
        answerAt={answerFor(key)}
        // §11.4 rule 4: the block that confirms every line unlocks once a guess was confirmed.
        mastered={decisions.some(
          (r) => r.decision.kind === "confirm_reading" || r.decision.kind === "confirm_readings",
        )}
      />
    );
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
      coach: coachFor(key),
      resurfaced: resurfacedFor(key),
      ask: askFor(key),
    },
  });

  /** The live record behind a question's answer: the Router names it. */
  const currentRecord = (key: QuestionKey): DecisionRecord | undefined => {
    const id = stepOf(key)?.decision_id;
    if (id) return byId.get(id);
    const recs = bySlot.get(slotKey(key)) ?? [];
    return key === "task" || key === "event" ? undefined : recs[recs.length - 1];
  };

  const historyFor = (key: QuestionKey) => {
    const now = currentRecord(key);
    return (bySlot.get(slotKey(key)) ?? [])
      .filter((r) => r !== now)
      .map((r) => ({
        id: r.id,
        seq: r.seq,
        when: fmtClock(r.at),
        sentence: sentence(r, decisions),
        tag:
          (r.decision.kind === "set_task" || r.decision.kind === "set_event") &&
          r.decision.column !== state.target
            ? "another outcome"
            : undefined,
      }));
  };

  // ─── the open (or reopened) question, by key ──────────────────────────────
  // The outcome is chosen from the table as it stands the right way round (M2 §2).
  const columns = (oriented?.artifact?.columns ?? ingest?.artifact?.columns)?.filter(
    (col) => col.name !== "__row_id",
  );
  const columnInfo = useMemo(
    () => new Map((columns ?? []).map((col) => [col.name, col])),
    [columns],
  );
  const ti = targetInfo?.artifact ?? null;
  const waitingFor = (stage: string, then: string): ReactNode => (
    <Pending>
      {STAGE_WORK[stage] ?? "Computing"}… {then}
    </Pending>
  );
  const structureArtifact = structure?.artifact ?? null;

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
      case "orientation":
        return (
          <OrientationAsk
            {...p}
            oriented={oriented?.artifact ?? null}
            current={state.orientation}
          />
        );
      case "target":
        return <TargetAsk {...p} columns={columns} summaries={summaryMap} current={state.target} />;
      case "event":
        return ti && ti.column === state.target ? (
          <EventAsk {...p} info={ti} current={state.event} />
        ) : (
          waitingFor("target_info", "Then the outcome's levels are read.")
        );
      case "task": {
        // WP18: what the task still asks (the outcome's scale, an ordinal outcome's order).
        const st = stepOf("task");
        const asks = st ? taskFollowup({ key, step: st, state, targetInfo: ti }) : null;
        if (asks) return <GenericAsk key={`task:${st?.followup}`} {...p} q={asks} />;
        return ti ? (
          <TaskAsk {...p} info={ti} current={state.task} />
        ) : (
          waitingFor("target_info", "Then the task is read from it.")
        );
      }
      case "purpose":
        return <PurposeAsk {...p} current={state.purpose} />;
      case "grain":
        return (
          <GrainAsk
            {...p}
            structure={structureArtifact}
            target={state.target}
            columns={(columns ?? []).map((col) => col.name)}
            current={state.grain}
          />
        );
      case "repeat_kind":
        return <RepeatKindAsk {...p} structure={structureArtifact} current={state.repeat_kind} />;
      case "unit":
        return (
          <UnitAsk
            {...p}
            structure={structureArtifact}
            nRows={oriented?.artifact?.n_rows ?? ingest?.artifact?.n_rows ?? null}
            current={state.unit}
          />
        );
      case "aggregation":
        return (
          <AggregationAsk
            {...p}
            structure={structureArtifact}
            task={state.task ?? ti?.task ?? null}
            current={state.aggregation}
          />
        );
      case "temporal":
        return <TemporalAsk {...p} structure={structureArtifact} current={state.temporal} />;
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
            // Eligibility is asked in scientific terms, with the outcome withheld (§04).
            numericColumns={(columns ?? [])
              .filter(
                (col) =>
                  (col.dtype === "numeric" || col.dtype === "integer") && col.name !== state.target,
              )
              .map((col) => col.name)}
          />
        );
      case "missing":
        return (
          <MissingAsk
            {...p}
            proposals={proposals?.artifact ?? undefined}
            columns={columnInfo}
            purpose={state.purpose}
            current={state.missing}
          />
        );
      case "split":
        return (
          <SealAsk
            {...p}
            plan={sealPlan?.artifact ?? null}
            roles={roles?.artifact ?? undefined}
            current={state.split}
          />
        );
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
          <ModelsAsk
            {...p}
            shelf={shelf.artifact}
            current={state.models}
          />
        ) : (
          waitingFor("shelf", "Then the model families are offered.")
        );
      case "substitution":
        return (
          <SubstitutionAsk {...p} pairs={design?.artifact?.substitution_pairs.length ?? null} />
        );
      case "survey":
        return (
          <SurveyAsk
            {...p}
            proposal={proposals?.artifact?.survey ?? undefined}
            current={state.survey}
          />
        );
      default: {
        // No step is ever blank: every other key is the generic question, composed from the
        // server's words and the cards its stage serves (generic/compose.ts).
        const st = stepOf(key);
        if (!st) return null;
        const q = compose({
          key,
          step: st,
          state,
          entry: entries.get(key),
          targetInfo: ti,
          roles: roles?.artifact,
          proposals: proposals?.artifact,
          causalDesign: causalDesign?.artifact,
          timeVarying: timeVarying?.artifact,
          columns,
        });
        const subject = SUBJECT[key];
        return (
          <GenericAsk
            key={`${key}:${st.status}`}
            {...p}
            q={q}
            fallbackTitle={`${subject.charAt(0).toUpperCase()}${subject.slice(1)}`}
          />
        );
      }
    }
  };

  const openSeal = (): ReactNode => (
    <OpenSealStep
      entry={entries.get("open_seal")}
      split={split?.artifact ?? null}
      now={nowKey === "open_seal"}
    />
  );

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
    if (key === "split") {
      const basis = split?.artifact?.basis;
      if (basis?.exploratory)
        return <>Held-out scores carry an exploratory label: {basis.label}.</>;
    }
    if (
      key === "energy_adjustment" &&
      design?.artifact &&
      design.fresh &&
      design.key === stages.design?.key
    ) {
      // The design's energy concerns, beside the answer that raised them.
      const about = warningsAbout(design.artifact.warnings, "energy");
      if (about.length)
        return <DesignWarnings warnings={about} title="Concerns" testId="energy-warnings" />;
    }
    const under = COUNTED_UNDER[key];
    if (under) {
      const later = decisions
        .filter(
          (r) => r.seq > rec.seq && under.includes(slotOf(r.decision, decisions) as QuestionKey),
        )
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

  /** The post-seal mark (M2_CONTRACT §3): a decision recorded after the seal was opened. */
  const postSeal = (rec: DecisionRecord): ReactNode =>
    rec.post_seal ? (
      <span
        className={k.postTag}
        title="Recorded after the held-out scores were seen"
        data-testid="post-seal-mark"
      >
        post-seal
      </span>
    ) : null;

  const settled = (key: QuestionKey, rec: DecisionRecord) => {
    const glyph =
      key === "split" || key === "open_seal" ? (
        <SealGlyph
          state={split?.artifact?.basis.state ?? sealPlan?.artifact?.basis?.state}
          recorded
          size={14}
        />
      ) : null;
    const undone = answerFor(key);
    const note = sentenceNote(key, rec);
    return (
      <DecisionSentence
        layoutId={`q-${key}`}
        subject={SUBJECT[key]}
        // The seal opens once (constitution §05): its sentence has no "change" and no "undo".
        onChange={key === "open_seal" ? undefined : () => reopen(key)}
        // INBOX 41: take the answer back with the engine's revert; nothing is deleted.
        onUndo={
          key === "open_seal"
            ? undefined
            : () => record(key, { kind: "revert", decision_id: rec.id }, "undo")
        }
        meta={
          <span className={k.metaSeal}>
            {postSeal(rec)}
            {glyph}#{rec.seq}
          </span>
        }
        note={
          undone?.key === "undo" ? (
            <>
              {note}
              {undone.node}
            </>
          ) : (
            note
          )
        }
        testId={`decision-${key}`}
      >
        {sentence(rec, decisions)}
      </DecisionSentence>
    );
  };

  const slotBody = (st: InterviewStep): ReactNode => {
    const key = st.key;
    if (key === "open_seal") {
      if (st.status === "open") return openSeal();
      if (st.status === "answered") {
        const rec = currentRecord(key);
        return rec ? settled(key, rec) : null;
      }
    } else if (reopened[key] && st.status !== "waiting") return ask(key);
    switch (st.status) {
      case "open":
        return key === "open_seal" ? null : ask(key);
      case "answered": {
        const rec = currentRecord(key);
        return rec ? settled(key, rec) : null;
      }
      case "skipped":
        return (
          <StatedSkip
            qkey={key}
            reason={st.reason}
            onAsk={() => reopen(key)}
            lead={
              ti && key === "task" ? (
                <>
                  <V>{ti.column}</V> read as <V>{ti.task}</V>, <V>{ti.confidence}</V>{" "}
                  confidence.{" "}
                </>
              ) : null
            }
          />
        );
      case "not_applicable": {
        // Questions an earlier answer made moot, side by side, are said once (§11.7: no walls).
        const run = naRuns.get(key);
        if (run === null) return null;
        if (run && run.length > 1)
          return (
            <NaGroup
              steps={run.map((x) => ({
                key: x.key,
                title: titleOf(entries, x.key),
                reason: x.reason,
              }))}
            />
          );
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
      }
      case "waiting":
        return null;
    }
  };

  // Where each step stands: in the flow, a pending row, or under "Then" (flow.ts).
  const { firstAt, pendingStep, inline, next, naRuns } = layoutFlow(interview, reopened);
  const firstUnanswered = firstAt === -1 ? undefined : interview[firstAt];

  // ── findings: repairs before the outcome; the rest noticed below ──────────
  const orientationStep = stepOf("orientation");
  // A turned-around table's findings read across the wrong axis: they wait for its orientation.
  const awaitingOrientation =
    orientationStep?.status === "open" || orientationStep?.status === "waiting";
  const repairFindings = allFindings.filter((f) => f.repairs.length > 0);
  const noticed = findings?.artifact
    ? { ...findings.artifact, findings: allFindings.filter((f) => f.repairs.length === 0) }
    : null;
  const deferTarget = (f: Finding): QuestionKey | null => {
    const to = f.routes_to;
    if (!to) return null;
    const st = stepOf(to)?.status;
    return st === "open" || st === "waiting" ? to : null;
  };
  const repairAnswer: RepairAnswer | null = answerFor("repairs");
  const repairs =
    state.lens && !awaitingOrientation && repairFindings.length ? (
      <motion.div
        key="repairs"
        layout="position"
        transition={{ layout: t.settle }}
        className={s.slot}
        data-slot="repairs"
      >
        <StaleVeil state={veilFor(stages.findings, findings)} order={1} testId="veil-repairs">
          <RepairsSection
            findings={allFindings}
            decisions={decisions}
            entry={entries.get("repairs")}
            deferTarget={deferTarget}
            subject={(q) => SUBJECT[q]}
            record={(d, at) => record("repairs", d, at)}
            pending={decide.isPending}
            answerAt={repairAnswer}
            marks={postSeal}
          />
        </StaleVeil>
      </motion.div>
    ) : null;
  const targetInline = inline.some((st) => st.key === "target") || pendingStep?.key === "target";

  /** A finding is settled once the question its lever routes to has a recorded answer. */
  const answeredBy = (f: Finding) => {
    const st = findingState(f, decisions);
    if (st.kind !== "open") return settlementOf(st.recordId);
    // The server says when an answer settles a finding; without it, the routed question's answer.
    if (f.answered_by === undefined && f.routes_to && stepOf(f.routes_to)?.status === "answered") {
      const rec = currentRecord(f.routes_to);
      return rec ? settlementOf(rec.id) : null;
    }
    return null;
  };
  const settlementOf = (id: string) => {
    const rec = byId.get(id);
    if (!rec) return null;
    const text = rec.sentence ?? sentenceText(rec);
    const head = text.split(/(?<=[.;:])\s/)[0]!.replace(/[.;:]$/, "");
    const words = head.split(/\s+/);
    return { seq: rec.seq, said: words.length > 14 ? `${words.slice(0, 14).join(" ")}…` : head };
  };
  const unread = !columns && needsRetry(stages.ingest) ? stages.ingest : undefined;
  const findingsBody = (() => {
    if (awaitingOrientation)
      return (
        <Pending testId="findings-await-orientation">
          What this table holds is read once its orientation is answered: on a table turned around,
          every check reads across the wrong axis.
        </Pending>
      );
    if (noticed)
      return <FindingsCards artifact={noticed} onRoute={route} answeredBy={answeredBy} />;
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
  const nFindings = noticed?.findings.length;
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
          // The repairs come before the outcome question (OPENING_SEQUENCE §01).
          const before = st.key === "target" ? repairs : null;
          if (!body) return before ? <Fragment key={st.key}>{before}</Fragment> : null;
          return (
            <Fragment key={st.key}>
              {before}
              <motion.div
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
            </Fragment>
          );
        })}
        {!targetInline ? repairs : null}
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
                  ) : st.status === "skipped" ? (
                    <span className={s.naTag} data-testid={`next-stated-${st.key}`}>
                      not asked
                    </span>
                  ) : null}
                  {st.key === "target" && st.waiting_on.includes("orientation") ? (
                    // OPENING_SEQUENCE §01: withheld while the table may be turned around.
                    <span className={s.nextNote} data-testid="target-withheld">
                      after the orientation: turned around, the columns are samples
                    </span>
                  ) : flash === st.key && st.status === "waiting" && firstUnanswered ? (
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
                {nFindings !== undefined && !awaitingOrientation ? (
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

/** Adjacent questions that do not apply, said once: their names, their reasons one press away. */
function NaGroup({ steps }: { steps: { key: string; title: string; reason: string | null }[] }) {
  const [open, setOpen] = useState(false);
  const reasons = [...new Set(steps.map((x) => x.reason).filter((r): r is string => !!r))];
  return (
    <div className={s.na} data-testid="na-group" data-keys={steps.map((x) => x.key).join(" ")}>
      <span className={s.naTitle}>{steps.map((x) => x.title).join(" · ")}</span>
      <span className={s.naTag}>not applicable</span>
      {reasons.length === 1 ? (
        <span className={s.naText}>
          <Prose text={reasons[0]!} />
        </span>
      ) : (
        <>
          <button
            type="button"
            className={s.naWhy}
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
          >
            why?
          </button>
          {open ? (
            <ul className={s.naList}>
              {steps.map((x) => (
                <li key={x.key}>
                  <span className={s.naItem}>{x.title}:</span> <Prose text={x.reason ?? ""} />
                </li>
              ))}
            </ul>
          ) : null}
        </>
      )}
    </div>
  );
}

/** A refusal with its exits (an exit records its decision, or goes to the question an answer
 *  waits behind), or a failure, at its control. */
function AnswerNote({
  answer,
  onDismiss,
  onRetry,
  onGo,
}: {
  answer: Answer;
  onDismiss: () => void;
  onRetry: (d: Decision) => void;
  onGo?: (() => void) | undefined;
}) {
  return answer.refusal ? (
    <RefusalNote
      refusal={answer.refusal}
      onDismiss={onDismiss}
      onExit={(exit) => (exit.decision ? onRetry(exit.decision) : onGo ? onGo() : onDismiss())}
    />
  ) : (
    <FailureNote message={answer.failure ?? "no reason given"} onDismiss={onDismiss} />
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
