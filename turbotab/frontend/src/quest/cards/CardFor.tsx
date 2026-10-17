/**
 * The card dispatcher (plans/W1_PART2_PLAN.md §2 "W1-E", rulings 6 and 7): each quest line's card in
 * the shell's card column, by its source (renderers.ts). A Router question is today's card
 * (AskFor); a declaration is W1-F's card for its kind (DECLARATION_CARDS); a finding is a card of
 * its own; reading items are quiet rows; anything else is the fallback card in the line's own words
 * (LineGeneric). Results, once reached, is not drawn line by line: its column is W1-G's
 * ResultsCards (StageCard.tsx, renderers.ts `columnOf`). It never renders a blank: until the
 * project's view arrives, the line's name stands with what it waits for.
 */
import type { QuestionKey } from "../../api/m1-types";
import { useProjectView, useQuest } from "../../api/queries";
import type { ProjectView } from "../../api/schema";
import { AskFor } from "../../components/record/AskFor";
import { BESPOKE_KEYS } from "../../components/record/generic/compose";
import type { CardForProps, DeclarationCards } from "../contracts";
import { DECLARATION_CARDS } from "./predict";
import { FindingCard } from "./FindingCard";
import { LineGeneric } from "./LineGeneric";
import { Readings } from "./Readings";
import { rendererOf } from "./renderers";
import { said } from "../words";
import c from "./cards.module.css";

export function CardFor({ pid, line, onOpenStage, cards = DECLARATION_CARDS }: CardForProps & { cards?: DeclarationCards }) {
  const view = useProjectView(pid).data;
  const log = useQuest(pid).data;
  const kind = rendererOf(line, cards);
  const routeTo = (question: string) => {
    const at = log?.stages.find((s) => s.lines.some((l) => l.source === "question" && l.key === question));
    if (at) onOpenStage?.(at.key);
  };

  switch (kind) {
    case "declaration": {
      const Card = cards[line.key]!;
      return <Card pid={pid} line={line} />;
    }
    case "reading":
      return <Readings pid={pid} line={line} />;
    case "generic":
      return <LineGeneric line={line} />;
    case "finding":
      return view ? (
        <FindingCard pid={pid} view={view} line={line} onRoute={routeTo} />
      ) : (
        <Waiting name={line.name} />
      );
    case "ask":
      return view ? <Ask pid={pid} view={view} stepKey={line.key as QuestionKey} line={line} /> : <Waiting name={line.name} />;
  }
}

/** Today's card for a Router question; the line's own words where the Router lists no step for it
 *  (AskFor composes a step it does not render bespoke, and has nothing to compose without one). */
function Ask({
  pid,
  view,
  stepKey,
  line,
}: {
  pid: string;
  view: ProjectView;
  stepKey: QuestionKey;
  line: CardForProps["line"];
}) {
  const listed = view.interview.some((s) => s.key === stepKey);
  if (!listed && !BESPOKE_KEYS.includes(stepKey)) return <LineGeneric line={line} />;
  return (
    <div data-testid={`ask-${stepKey}`}>
      <AskFor pid={pid} view={view} stepKey={stepKey} />
    </div>
  );
}

function Waiting({ name }: { name: string }) {
  return (
    <section className={c.lineCard} aria-label={said(name)}>
      <p className={c.kicker}>{said(name)}</p>
      <p className={c.quiet}>Opening the project…</p>
    </section>
  );
}
