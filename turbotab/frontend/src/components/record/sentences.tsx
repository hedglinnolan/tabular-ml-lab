/**
 * The words of the Record: option labels, and each decision's sentence.
 * Decision sentences are past tense and exact (§06): a sentence that could not
 * appear in a methods section is not a decision sentence.
 */
import type { ReactNode } from "react";
import type { Decision, DecisionRecord, Lens, Purpose, Slot, Task } from "../../api/schema";
import { Prose, V } from "../Prose";

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
    case "set_roles":
      return "roles";
    case "set_energy_adjustment":
      return "energy_adjustment";
    case "set_exclusions":
      return "exclusions";
    case "set_missing":
      return "missing";
    case "set_split":
      return "split";
    case "select_models":
      return "models";
    case "set_substitution":
      return "substitution";
    case "set_orientation":
      return "orientation";
    case "set_event":
      return "event";
    case "set_grain":
      return "grain";
    case "set_repeat_kind":
      return "repeat_kind";
    case "set_unit":
      return "unit";
    case "set_aggregation":
      return "aggregation";
    case "set_temporal":
      return "temporal";
    case "open_seal":
      return "seal_opened";
    case "apply_repair":
    case "defer_finding":
    case "dismiss_finding":
      return "findings";
    case "set_feature_table":
      return "feature_table";
    case "set_categorical":
      return "categorical";
    case "set_survey":
      return "survey";
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

/**
 * The record's sentence. The server authors it when the decision is recorded and the
 * Record quotes it verbatim (DESIGN_LANGUAGE §05.1); only records made before the server
 * wrote sentences are composed here.
 */
export function sentence(record: DecisionRecord, records: DecisionRecord[]): ReactNode {
  if (record.sentence) return <Prose text={record.sentence} />;
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
          <V>{d.column}</V> was modeled as a <V>{d.task}</V> task.
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
    default:
      // M1 kinds: the server authors the sentence (DecisionRecord.sentence).
      return null;
  }
}

/** The sentence as plain words, for a screen reader's announcement. */
export function sentenceText(record: DecisionRecord): string {
  if (record.sentence) return record.sentence.replace(/`/g, "");
  const d = record.decision;
  switch (d.kind) {
    case "set_lens":
      return `The table was read through the ${d.lenses.join(" and ")} ${d.lenses.length === 1 ? "lens" : "lenses"}.`;
    case "set_target":
      return `${d.column} was chosen as the outcome.`;
    case "set_task":
      return `${d.column} was modeled as a ${d.task} task.`;
    case "set_purpose":
      return `The analysis was declared for ${d.purpose}.`;
    default:
      return "The answer was recorded.";
  }
}
