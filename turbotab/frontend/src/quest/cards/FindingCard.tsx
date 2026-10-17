/**
 * One noticing as its own card (ruling 7): today's findings card, made by filtering the findings
 * artifact to the line's finding, without editing Findings.tsx. A finding with repairs is today's
 * repairs card for it alone (apply one, hold it for its question, or keep the table as is); any
 * other is the findings card, whose lever opens the stage that asks its question, with today's
 * "Keep as is", which records that it was seen and set aside.
 */
import { useState } from "react";
import { isRefusalError } from "../../api/client";
import type { QuestionKey } from "../../api/m1-types";
import { useDecide, useTeaching } from "../../api/queries";
import type { Decision, Finding, ProjectView, QuestLine, Refusal } from "../../api/schema";
import { STAGE_LABEL } from "../../components/JobChips";
import { SUBJECT } from "../../components/record/AskFor";
import { FindingsCards } from "../../components/record/Findings";
import { RepairsSection } from "../../components/record/Repairs";
import { useStage } from "../../state/stages";
import { Plain } from "../../kit";
import { said } from "../words";
import c from "./cards.module.css";

export function FindingCard({
  pid,
  view,
  line,
  onRoute,
}: {
  pid: string;
  view: ProjectView;
  line: QuestLine;
  /** Open the stage that asks the question a finding's lever routes to. */
  onRoute: (question: string) => void;
}) {
  const findings = useStage(pid, view, "findings");
  const teaching = useTeaching();
  const decide = useDecide(pid);
  const [refused, setRefused] = useState<Refusal | null>(null);
  const artifact = findings?.artifact ?? null;
  const finding: Finding | undefined = artifact?.findings.find((f) => f.id === line.key);

  const record = (d: Decision) => {
    setRefused(null);
    decide.mutate(d, { onError: (err) => isRefusalError(err) && setRefused(err.refusal) });
  };
  const head = <p className={c.kicker}>{said(line.name)}</p>;
  let body;
  if (!artifact) {
    body = <p className={c.quiet}>{STAGE_LABEL.findings}…</p>;
  } else if (!finding) {
    body = <p className={c.quiet}>This noticing is no longer among the findings.</p>;
  } else if (finding.repairs.length) {
    const deferTarget = (f: Finding): QuestionKey | null => {
      const to = f.routes_to;
      if (!to) return null;
      const st = view.interview.find((s) => s.key === to)?.status;
      return st === "open" || st === "waiting" ? to : null;
    };
    body = (
      <RepairsSection
        findings={[finding]}
        decisions={view.decisions}
        entry={(teaching.data ?? []).find((e) => e.key === "repairs")}
        deferTarget={deferTarget}
        subject={(k) => SUBJECT[k]}
        record={(d) => record(d)}
        pending={decide.isPending}
        answerAt={null}
      />
    );
  } else {
    body = (
      <>
        <FindingsCards
          artifact={{ ...artifact, findings: [finding] }}
          onRoute={(to) => onRoute(to)}
        />
        {line.status === "open" ? (
          <button
            type="button"
            className={c.small}
            disabled={decide.isPending}
            onClick={() => record({ kind: "dismiss_finding", finding_id: finding.id, reason: null })}
            data-testid={`keep-as-is-${finding.id}`}
          >
            Keep as is
          </button>
        ) : null}
      </>
    );
  }
  return (
    <section className={c.lineCard} data-testid={`finding-card-${line.key}`} aria-label={said(line.name)}>
      {head}
      {body}
      {refused ? (
        <p className={c.note} role="alert">
          <Plain text={refused.error.message} />
        </p>
      ) : null}
    </section>
  );
}
