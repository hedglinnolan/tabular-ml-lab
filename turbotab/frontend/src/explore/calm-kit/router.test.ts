import { describe, expect, it } from "vitest";
import { STEPS } from "./fixture";
import { footprint, route, viewsFor } from "./router";

/** The layout the router picks for every option of every decision (the kit's routing table). */
export const ROUTING: Record<string, Record<string, string>> = Object.fromEntries(
  STEPS.map((s) => [s.id, Object.fromEntries(s.options.map((o) => [o.id, route(o.preview, { disabled: o.disabled })]))]),
);

describe("the canvas router", () => {
  it("routes every option of every decision by its footprint", () => {
    expect(ROUTING).toEqual({
      unit: { kcal_1: "focus", kcal_2: "focus", kj_1: "focus" },
      exclusions: {
        none: "none",
        willett_2013_by_sex: "flow",
        nhs_hpfs_by_sex: "flow",
        sex_neutral_500_5000: "flow",
        sex_neutral_500_3500: "flow",
        goldberg_schofield: "refused",
      },
      sensitivity: { both: "flow", willett: "flow", nhs: "flow", none: "none" },
      missing: { complete_case: "none", multiple_imputation: "none" },
      "single:bp_di": { covariate: "routing", excluded: "none" },
      "single:bp_sys": { covariate: "routing", excluded: "none" },
      "single:cycle_begin_year": { covariate: "routing", excluded: "none" },
      block: { confirm: "flow" },
      exposure: { sugar: "none", protein: "none", carb: "none", fat_total: "none", fat_sat: "none", fat_mon: "none", fat_poly: "none" },
      effect: { total: "angles", direct: "angles" },
      contrast: { substitution: "angles", addition: "angles" },
      "adjust:demographic": { confounder: "routing", timing_unknown: "routing", mediator: "routing", not_a_cause: "routing" },
      "adjust:dietary": { confounder: "routing", timing_unknown: "routing", mediator: "routing", not_a_cause: "routing" },
      "adjust:body": { timing_unknown: "routing", confounder: "routing", mediator: "routing", not_a_cause: "routing" },
      "adjust:unguessed:0": { confounder: "routing", timing_unknown: "routing", mediator: "routing", not_a_cause: "routing" },
      "adjust:unguessed:1": { confounder: "routing", timing_unknown: "routing", mediator: "routing", not_a_cause: "routing" },
      energy: {
        standard: "none",
        residual: "strip",
        density_multivariate: "strip",
        residual_energy_dropped: "strip",
        density: "strip",
        none: "refused",
        all_components: "refused",
        partition: "refused",
      },
      // Model 1 is what the question decides: "nothing" leaves it `sugar` alone, as it reads now
      model1: { guess: "routing", empty: "none" },
      codes: { confirm: "focus" },
      lock: { lock: "none" },
    });
  });

  it("measures rows, columns and routing from the engine's views", () => {
    const willett = STEPS.find((s) => s.id === "exclusions")!.options.find((o) => o.id === "willett_2013_by_sex")!;
    expect(footprint(willett.preview).rows).toEqual([expect.objectContaining({ n: 20235, dropped: 1614 })]);
    const residual = STEPS.find((s) => s.id === "energy")!.options.find((o) => o.id === "residual")!;
    const fp = footprint(residual.preview);
    expect(fp.columns.map((c) => c.column)).toEqual(["carb", "fat_total", "fat_mon", "fat_sat", "protein", "fat_poly", "sugar"]);
    expect(fp.routing.filter((r) => r.change === "enters").map((r) => r.column)).toContain("sugar_adj");
    const bp = STEPS.find((s) => s.id === "single:bp_di")!.options[0]!;
    expect(footprint(bp.preview).routing).toEqual([{ column: "bp_di", change: "enters" }]);
  });

  it("shows at most three views and puts the rest behind More angles", () => {
    for (const s of STEPS)
      for (const o of s.options) {
        const layout = route(o.preview, { disabled: o.disabled });
        const { shown, more } = viewsFor(layout, o.preview);
        expect(shown.length).toBeLessThanOrEqual(3);
        expect(shown.length + more.length).toBe(layout === "angles" || layout === "none" || layout === "refused" ? 0 : o.preview.views.length);
      }
  });

  it("gives every option a picture or one line that says why there is none", () => {
    for (const s of STEPS)
      for (const o of s.options) {
        const layout = route(o.preview, { disabled: o.disabled });
        if (layout === "refused") expect(o.refusal ?? o.preview.refusal?.message, `${s.id}/${o.id}`).toBeTruthy();
        else if (layout === "none") {
          const line = o.preview.note ?? o.preview.views[0]?.caption;
          expect(line, `${s.id}/${o.id}`).toBeTruthy();
        } else expect(o.preview.views.length + (o.preview.angles?.length ?? 0), `${s.id}/${o.id}`).toBeGreaterThan(0);
      }
  });
});
