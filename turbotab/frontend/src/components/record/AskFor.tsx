/**
 * AskFor: the answer card of one Router step. It reads the stages that step's card needs and
 * renders the bespoke component for the key (or, for a key without one, the generic question
 * composed from the server's words), inside the Question shell.
 *
 * The Record passes its own wiring (the recording, the refusal at the control, the arrival and
 * reopen state, the findings held for the question). Given only the project, its view and the
 * step key, AskFor wires itself plainly, so another shell (the quest log) can render today's
 * cards: it records through its own mutation, and a press the server refuses or fails to record
 * answers at its control with the same note the Record shows (the refusal and its exits, or the
 * failure), never a dead control. Arrival, reopening, the drawer and held findings are the
 * shell's.
 */
import { useMemo, useState, type ReactNode } from "react";
import { isRefusalError } from "../../api/client";
import { useColumnSummaries, useDecide, useTeaching } from "../../api/queries";
import type { QuestionKey, Role } from "../../api/m1-types";
import { NOT_YET } from "../../api/m2-types";
import type { Decision, ProjectView, Refusal } from "../../api/schema";
import { useStage } from "../../state/stages";
import { REOPEN_EVENT } from "../Failure";
import { STAGE_LABEL } from "../JobChips";
import { Prose } from "../Prose";
import { StageRetry, needsRetry } from "../StageRetry";
import { Pending } from "./blocks";
import { FailureNote, RefusalNote } from "./Refusal";
import {
  EnergyAsk,
  ExclusionsAsk,
  MissingAsk,
  ModelsAsk,
  SubstitutionAsk,
  SurveyAsk,
} from "./ask/ChoiceQuestions";
import { AskCard } from "./ask/AskCard";
import { besideAsk } from "./besideAsk";
import type { AskProps } from "./ask/common";
import { LensAsk, PurposeAsk, TargetAsk, TaskAsk } from "./ask/FactQuestions";
import { RolesAsk } from "./ask/RolesAsk";
import { SealAsk } from "./ask/SealQuestions";
import {
  AggregationAsk,
  EventAsk,
  GrainAsk,
  OrientationAsk,
  RepeatKindAsk,
  TemporalAsk,
  UnitAsk,
} from "./ask/SequenceQuestions";
import { compose, taskFollowup } from "./generic/compose";
import { GenericAsk } from "./generic/GenericAsk";
import s from "./Record.module.css";

/** Each question's name in running text ("Change the column roles"). */
export const SUBJECT: Record<QuestionKey, string> = {
  lens: "the lens",
  orientation: "the table's orientation",
  target: "the outcome",
  event: "the event level",
  task: "the task",
  follow_up: "the follow-up",
  design: "the study design",
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
  form: "the functional form",
  modification: "the effect modifiers",
  causal: "the causal estimate",
  models: "the models",
  substitution: "the substitution",
  open_seal: "opening the seal",
};

/** What the Record knows that a step's card cannot read from the project alone. */
export interface AskWiring {
  /** A decision is being recorded. */
  pending: boolean;
  /** Record a decision; `at` names the option a refusal should answer under. */
  record: (decision: Decision, at?: string) => void;
  /** Reopened with an answer on record: close and keep it. Absent otherwise. */
  keep?: (() => void) | undefined;
  /** The server's answer to a press that did not record, at its option. */
  answerAt: AskProps["answerAt"];
  now: boolean;
  arriving: boolean;
  onArrived: () => void;
  onOpenDrawer: () => void;
  reopened: boolean;
  /** Findings held for this question, resurfacing inside it (M2_CONTRACT §4). */
  resurfaced?: ReactNode;
}

const NOOP = () => {};

/** The plain wiring's answer to a press that did not record: where it was pressed, and why. */
interface PlainAnswer {
  at: string;
  refusal?: Refusal;
  failure?: string;
}

/* A const, not a function declaration: purposes.test.tsx registers every function the record/
 * folder exports, and AskFor is a structural wrapper (each card it renders is registered). When the
 * quest-log shell lands, register it in stage/purposes.ts and this can become a function. */
export const AskFor = ({
  pid,
  view,
  stepKey,
  wiring,
}: {
  pid: string;
  view: ProjectView;
  stepKey: QuestionKey;
  /** Absent: the plain wiring (records through its own mutation and answers a refusal at its
   *  control; no drawer, arrival, reopening or held findings). */
  wiring?: AskWiring;
}): ReactNode => {
  const key = stepKey;
  const { state, decisions, stages } = view;
  const decide = useDecide(pid);
  // The roles card asks for the project's goal (GET /teaching?purpose=): the server words it by
  // goal, its predictors under prediction (W1-R finding 3c). Every other card reads the words with
  // no goal named, which the server words alike under every goal.
  const teaching = useTeaching(key === "roles" ? state.purpose : null);
  const [plainAnswer, setPlainAnswer] = useState<PlainAnswer | null>(null);
  const entry = useMemo(
    () => (teaching.data ?? []).find((e) => e.key === key),
    [teaching.data, key],
  );

  const ingest = useStage(pid, view, "ingest");
  const oriented = useStage(pid, view, "oriented");
  const structure = useStage(pid, view, "structure");
  const profile = useStage(pid, view, "profile");
  const targetInfo = useStage(pid, view, "target_info");
  const roles = useStage(pid, view, "roles");
  const proposals = useStage(pid, view, "proposals");
  const sealPlan = useStage(pid, view, "seal_plan");
  const shelf = useStage(pid, view, "shelf");
  const design = useStage(pid, view, "design");
  // The generic question's cards (compose.ts): the causal lane's and the time-varying lane's.
  const causalDesign = useStage(pid, view, "causal_design");
  const timeVarying = useStage(pid, view, "time_varying");
  // FORM: the form question's card.
  const forms = useStage(pid, view, "forms");
  const summaries = useColumnSummaries(pid, stages.ingest?.status === "fresh").data;

  const summaryMap = useMemo(
    () => (summaries ? new Map(summaries.map((x) => [x.name, x])) : undefined),
    [summaries],
  );
  // The outcome is chosen from the table as it stands the right way round (M2 §2).
  const columns = (oriented?.artifact?.columns ?? ingest?.artifact?.columns)?.filter(
    (col) => col.name !== "__row_id",
  );
  const columnInfo = useMemo(
    () => new Map((columns ?? []).map((col) => [col.name, col])),
    [columns],
  );
  const ti = targetInfo?.artifact ?? null;
  const structureArtifact = structure?.artifact ?? null;
  const stepOf = (k: QuestionKey) => view.interview.find((st) => st.key === k);
  const waitingFor = (stage: string, then: string): ReactNode => (
    <Pending>
      {STAGE_LABEL[stage] ?? "Computing"}… {then}
    </Pending>
  );

  /** The plain wiring's recording: a refusal or a failure answers at the control pressed. */
  const plainRecord = (d: Decision, at = "") => {
    setPlainAnswer(null);
    decide.mutate(d, {
      onError: (err) =>
        setPlainAnswer(
          isRefusalError(err)
            ? { at, refusal: err.refusal }
            : { at, failure: err instanceof Error ? err.message : String(err) },
        ),
    });
  };
  /** As the Record's: a refusal with its exits, or the failure; `not_yet` leads to the first open
   *  question. Beside the ledger's ask card on this question, a press refused for the readings it
   *  asks says what it waits for, and the readings stay that card's alone (`besideAsk`). */
  const plainAnswerAt = ((): AskProps["answerAt"] => {
    if (wiring || !plainAnswer) return null;
    const { at, failure } = plainAnswer;
    const asking = stepOf(key)?.status === "open" && !!stepOf(key)?.ask && !at.startsWith("ask:");
    const refusal =
      plainAnswer.refusal && asking ? besideAsk(plainAnswer.refusal) : plainAnswer.refusal;
    const dismiss = () => setPlainAnswer(null);
    const firstOpen = view.interview.find(
      (st) => st.status === "open" || st.status === "waiting",
    )?.key;
    const onGo =
      refusal?.error.code === NOT_YET && firstOpen
        ? () => {
            setPlainAnswer(null);
            window.dispatchEvent(new CustomEvent(REOPEN_EVENT, { detail: firstOpen }));
          }
        : undefined;
    return {
      key: at,
      node: refusal ? (
        <RefusalNote
          refusal={refusal}
          onDismiss={dismiss}
          onExit={(exit) =>
            exit.decision ? plainRecord(exit.decision, at) : onGo ? onGo() : dismiss()
          }
        />
      ) : (
        <FailureNote message={failure ?? "no reason given"} onDismiss={dismiss} />
      ),
    };
  })();

  const w: AskWiring = wiring ?? {
    pending: decide.isPending,
    record: plainRecord,
    answerAt: plainAnswerAt,
    now: stepOf(key)?.status === "open",
    arriving: false,
    onArrived: NOOP,
    onOpenDrawer: NOOP,
    reopened: false,
  };

  /** At most one coach line per card (M2_CONTRACT §10). */
  const coach = ((): ReactNode => {
    if (key === "missing") return null; // its blanks and their reason are already on the card
    const note = proposals?.artifact?.coach?.[key];
    if (!note) return null;
    // Eligibility withholds the outcome's distribution (constitution §04): no note about it.
    if (note.anchor.kind === "column" && note.anchor.ref === state.target) return null;
    return <Prose text={note.text} />;
  })();

  /** The ledger's one ask card on the open step that asks (BLUEPRINT §14.2), inside its question. */
  const ledgerCard = ((): ReactNode => {
    const st = stepOf(key);
    if (!st?.ask || st.status !== "open") return null;
    return (
      <AskCard
        card={st.ask}
        pending={w.pending}
        record={w.record}
        answerAt={w.answerAt}
        // §11.4 rule 4: the block that confirms every line unlocks once a guess was confirmed.
        mastered={decisions.some(
          (r) => r.decision.kind === "confirm_reading" || r.decision.kind === "confirm_readings",
        )}
      />
    );
  })();

  const p: AskProps = {
    entry,
    pending: w.pending,
    record: w.record,
    keep: w.keep,
    answerAt: w.answerAt,
    shell: {
      qkey: key,
      now: w.now,
      arriving: w.arriving,
      onArrived: w.onArrived,
      onOpenDrawer: w.onOpenDrawer,
      reopened: w.reopened,
      coach,
      resurfaced: w.resurfaced,
      ask: ledgerCard,
    },
  };

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
        <OrientationAsk {...p} oriented={oriented?.artifact ?? null} current={state.orientation} />
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
          purpose={state.purpose}
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
        <ModelsAsk {...p} shelf={shelf.artifact} current={state.models} />
      ) : (
        waitingFor("shelf", "Then the model families are offered.")
      );
    case "substitution":
      return <SubstitutionAsk {...p} pairs={design?.artifact?.substitution_pairs.length ?? null} />;
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
        entry: entry,
        targetInfo: ti,
        roles: roles?.artifact,
        proposals: proposals?.artifact,
        causalDesign: causalDesign?.artifact,
        timeVarying: timeVarying?.artifact,
        forms: forms?.artifact,
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
