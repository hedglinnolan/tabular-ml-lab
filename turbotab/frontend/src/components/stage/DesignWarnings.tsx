/**
 * The design's warnings (design.warnings), said where they apply (BLUEPRINT: concerns stated,
 * never hidden; DRIVE_RUBRIC §5.13): every one under the column lineage, the energy ones beside the
 * energy decision, the substitution ones beside the curves. In the coach's amber.
 */
import { Rich } from "./text";
import s from "./Stage.module.css";

export type WarningTopic = "energy" | "substitution" | "columns";

const ENERGY = /kcal_from_other|energy explains|strata apply/i;
const SUBSTITUTION = /substitution|moves? (?:it|them) in proportion/i;

/** The warnings about one topic ("columns": every warning). */
export function warningsAbout(warnings: readonly string[], topic: WarningTopic): string[] {
  if (topic === "columns") return [...warnings];
  const pattern = topic === "energy" ? ENERGY : SUBSTITUTION;
  return warnings.filter((w) => pattern.test(w));
}

export function DesignWarnings({
  warnings,
  title = "Concerns about this design",
  testId = "design-warnings",
}: {
  warnings: readonly string[];
  title?: string;
  testId?: string;
}) {
  if (!warnings.length) return null;
  return (
    <section className={s.warnings} aria-label={title} data-testid={testId}>
      <h4 className={s.warningsHead}>{title}</h4>
      <ul className={s.warningsList}>
        {warnings.map((w) => (
          <li key={w}>
            <Rich text={w} />
          </li>
        ))}
      </ul>
    </section>
  );
}
