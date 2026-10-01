/**
 * PLACEHOLDER. The stage agent owns src/components/stage/** (M1_CONTRACT §14); the merge
 * keeps their real <Stage>. This stand-in honors the same props so the Record and the
 * banner can be built and driven against it: it says what the focus asks the stage to show.
 */
import type { ProjectView } from "../../api/schema";
import type { StageFocus } from "../../state/focus";
import { Prose } from "../Prose";

interface Props {
  pid: string;
  view: ProjectView;
  focus: StageFocus;
  onFocus: (f: StageFocus) => void;
}

function describe(focus: StageFocus): { pill: string; text: string } {
  switch (focus.kind) {
    case "option":
      return { pill: "Preview", text: `${focus.label} — nothing is recorded.` };
    case "finding":
      return { pill: "Evidence", text: `The evidence for finding \`${focus.findingId}\`.` };
    case "banner":
      return { pill: "Pipeline", text: `The ${focus.segment} of the pipeline, in full.` };
    case "live":
      return { pill: "Your data now", text: "The row flow and lineage of the current state." };
  }
}

export function Stage({ pid, focus, onFocus }: Props) {
  const d = describe(focus);
  return (
    <section
      aria-label="Stage"
      data-testid="stage"
      data-focus={focus.kind}
      data-pid={pid}
      style={{
        height: "100%",
        minHeight: 320,
        border: "1px dashed var(--line)",
        borderRadius: "var(--radius-lg)",
        background: "var(--surface)",
        padding: "16px 18px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="kicker">{d.pill}</span>
        {focus.kind !== "live" ? (
          <button
            type="button"
            onClick={() => onFocus({ kind: "live" })}
            style={{
              marginLeft: "auto",
              border: 0,
              background: "none",
              color: "var(--muted)",
              fontSize: 12,
              textDecoration: "underline",
            }}
          >
            Back to your data now
          </button>
        ) : null}
      </div>
      <p
        data-testid="stage-text"
        style={{ fontFamily: "var(--serif)", fontSize: 16, margin: 0, color: "var(--ink)" }}
      >
        <Prose text={d.text} />
      </p>
      <p style={{ margin: 0, fontSize: 12, color: "var(--faint)" }}>
        Placeholder: the stage component arrives with the stage agent&apos;s merge.
      </p>
    </section>
  );
}
