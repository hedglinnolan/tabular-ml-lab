/**
 * The words of the Record: option labels, and each decision's sentence.
 * Decision sentences are past tense and exact (§06): a sentence that could not
 * appear in a methods section is not a decision sentence.
 */
import type { ReactNode } from "react";
import type { Decision, DecisionRecord, Lens, Purpose, Slot, Task } from "../../api/schema";
import { V } from "../Prose";

export const LENS_LABEL: Record<Lens, string> = {
  metabolomics: "Metabolomics or proteomics",
  genomics: "Genomics or transcriptomics",
  dietary: "Dietary intake",
  clinical: "Clinical measurements and labs",
  survey: "Survey or questionnaire instruments",
};

export const TASK_TEXT: Record<Task, { label: string; body: string }> = {
  regression: { label: "Regression", body: "The outcome is a quantity; models predict its value." },
  binary: { label: "Binary", body: "The outcome has two classes; models predict which one." },
  multiclass: {
    label: "Multiclass",
    body: "The outcome has several unordered classes; models predict which one.",
  },
};

export const PURPOSE_TEXT: Record<Purpose, { label: string; body: string; clause: string }> = {
  prediction: {
    label: "Prediction",
    body: "Models are judged on rows they never saw; their coefficients are not interpreted.",
    clause: "models are judged on rows they never saw",
  },
  inference: {
    label: "Inference",
    body: "Associations are estimated with their uncertainty; the model family favors ones you can report.",
    clause: "associations are estimated with their uncertainty",
  },
};

export function slotOf(d: Decision, records: DecisionRecord[]): Slot | null {
  switch (d.kind) {
    case "set_lens":
      return "lens";
    case "set_target":
      return "target";
    case "set_task":
      return "task";
    case "set_purpose":
      return "purpose";
    case "revert": {
      const undone = records.find((r) => r.id === d.decision_id);
      return undone ? slotOf(undone.decision, records) : null;
    }
  }
}

function joinNodes(items: ReactNode[]): ReactNode[] {
  const out: ReactNode[] = [];
  items.forEach((it, i) => {
    if (i > 0) out.push(i === items.length - 1 ? (items.length > 2 ? ", and " : " and ") : ", ");
    out.push(it);
  });
  return out;
}

export function sentence(record: DecisionRecord, records: DecisionRecord[]): ReactNode {
  const d = record.decision;
  switch (d.kind) {
    case "set_lens": {
      const chips = d.lenses.map((l) => <V key={l}>{l}</V>);
      return (
        <>
          The table was read through the {joinNodes(chips)}{" "}
          {d.lenses.length === 1 ? "lens" : "lenses"}.
        </>
      );
    }
    case "set_target":
      return (
        <>
          <V>{d.column}</V> was chosen as the outcome.
        </>
      );
    case "set_task":
      return (
        <>
          The outcome was modeled as a <V>{d.task}</V> task.
        </>
      );
    case "set_purpose":
      return (
        <>
          The analysis was declared for <V>{d.purpose}</V>: {PURPOSE_TEXT[d.purpose].clause}.
        </>
      );
    case "revert": {
      const undone = records.find((r) => r.id === d.decision_id);
      return (
        <>
          Decision <V>#{undone?.seq ?? "?"}</V> was reverted, restoring the answer before it.
        </>
      );
    }
  }
}
