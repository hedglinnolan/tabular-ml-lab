/**
 * The plan's check of the exposure's shape, beside a family's Table 2 in Estimate Results (batch 1;
 * the owner's rulings of 2026-11-04; calm/BEATS_MODELS_RESULTS.md M4: "Whichever you choose, the
 * other shape is set in your plan, and Results reports it beside your estimate"), as the effects
 * stage serves it beside Model 2 (core/stages/effects.py `FormCheck`, `EffectsFamily.check`).
 * Nothing is computed here and no verdict is composed: the caption is the engine's (its refit, its
 * knots and its test, or the line's one coefficient, or why the check could not be made), and the
 * curve is the served points, drawn by the curve view.
 *
 *   - a straight line kept (`shape` "spline"): the caption, and the check's curve, each served
 *     point's difference from the served reference (`curve`) beside the straight line's (`line`),
 *     on the scale of the model's coefficients, points only and no band (`curveFromCheck`);
 *   - a restricted cubic spline kept (`shape` "linear"): the caption, which says the line's one
 *     coefficient; the engine serves no curve for it;
 *   - refused (`refused` set): the caption, the refusal's words, and nothing drawn as a number.
 *
 * Nothing at all where the family's check is null, as the engine serves it where the plan sets none
 * (core/methods/exposure_form.py `check_shape`: Predict, no single exposure declared, no form
 * declared for it, or another form, such as quintiles). Every sentence drawn is the engine's, save
 * the heading and the curve's labels, which name what is drawn (`CHECK_TITLE`, `CURVE_LABEL`,
 * `LINE_LABEL`, the axes, and the line saying no interval is drawn).
 */
import type { EffectsArtifact } from "../../api/m3-types";
import { Rich } from "../../components/stage/text";
import type { CurveData, CurveLine } from "../../components/views/curve";
import { CurveView } from "../../components/views/CurveView";
import { fmtNum } from "../../components/views/common/parts";
import c from "./check.module.css";
import s from "./results.module.css";

export type FormCheck = NonNullable<EffectsArtifact["families"][number]["check"]>;

/** The section's heading: what it is, the plan's check of the declared exposure's shape (core/
 *  stages/effects.py `FormCheck`), never what the check found. */
export const CHECK_TITLE = "The plan's check of the shape";
/** The two lines' names in the legend and the table: the spline the check fit, and the straight
 *  line the primary model keeps (the served `line`, Model 2's own slope from the reference). */
export const CURVE_LABEL = "The check's curve";
export const LINE_LABEL = "Your straight line";

/**
 * The check's curve as the curve view draws it: each served point's `x` against the spline's
 * difference from the reference (`curve`) and the straight line's (`line`), as served. The x axis
 * names the model-matrix column the shape is of (`column`), the y axis the served reference (to the
 * four significant digits the caption names the knots in) and the scale the points are on. Null
 * where there is no curve to draw: the check refused, a spline kept (its check is the line's one
 * coefficient), or no point served. The straight line is left out where the engine served it at no
 * point (the primary reported no slope).
 */
export function curveFromCheck(check: FormCheck): CurveData | null {
  if (check.refused !== null || check.shape !== "spline" || !check.points.length || check.reference === null) return null;
  const x = check.points.map((p) => p.x);
  const lines: CurveLine[] = [{ key: "spline", label: CURVE_LABEL, role: "series", slot: 1, x, y: check.points.map((p) => p.curve) }];
  if (check.points.some((p) => p.line !== null)) {
    lines.push({ key: "linear", label: LINE_LABEL, role: "series", slot: 2, x, y: check.points.map((p) => p.line) });
  }
  return {
    xLabel: check.column,
    xName: check.column,
    yLabel: `Difference from ${check.column} = ${fmtNum(check.reference, 4)}, on the model's coefficient scale`,
    lines,
    zero: true,
    // The served points carry no interval (core/stages/effects.py `CheckPoint`).
    basis: lines.length > 1 ? "No interval is drawn around either line." : "No interval is drawn around the curve.",
    sealed: null,
  };
}

/** The family's check beside its Table 2, or nothing where it has none. */
export function ShapeCheck({ check, family }: { check: FormCheck | null | undefined; family: string }) {
  if (!check) return null;
  const curve = curveFromCheck(check);
  return (
    <section
      className={s.exhibit}
      data-purpose="form_check"
      data-testid="results-form-check"
      data-family={family}
      data-kept={check.kept}
      data-shape={check.shape}
      data-refused={check.refused !== null ? "true" : undefined}
    >
      <h3 className={s.exhibitTitle}>{CHECK_TITLE}</h3>
      <p className={c.caption} data-testid="results-form-check-caption">
        <Rich text={check.caption} />
      </p>
      {curve ? <CurveView data={curve} /> : null}
    </section>
  );
}
