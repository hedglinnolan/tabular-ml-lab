/**
 * The mock server's words for M1: the sentence each decision records (after
 * turbotab/core/voice.py), the refusals the M1 validators answer with (after
 * turbotab/core/decisions.py), and each finding's one-line summary and lever (after
 * turbotab/core/stages/finding_words.py). Backticks mark data; the client renders chips.
 */
import type { EnergyMethod, Role } from "../api/m1-types";
import type {
  Decision,
  DecisionRecord,
  Finding,
  FindingsArtifact,
  ProjectState,
  Refusal,
  Scalar,
} from "../api/schema";
import type { MockDataset } from "./datasets";
import {
  cohort,
  energyBearing,
  familyLabel,
  proposalsArtifact,
  rolesArtifact,
  ruleExcludes,
} from "./m1-stages";
import { findColumn, isNumericDtype, nMissing } from "./stats";

const tick = (v: unknown) => `\`${v}\``;
const count = (n: number) => tick(Math.round(n).toLocaleString("en-US"));
const num = (x: number | null) =>
  x === null ? "" : Number.isInteger(x) ? String(x) : String(+x.toFixed(6));

/** `a`, `a and b`, `a, b, c and 3 more` (voice.listing). */
function listing(items: string[], limit = 4, ticked = true): string {
  let shown = items.map((i) => (ticked ? tick(i) : i));
  if (shown.length > limit) {
    const rest = shown.length - (limit - 1);
    shown = [...shown.slice(0, limit - 1), `${rest} more`];
  }
  if (shown.length <= 1) return shown.join("");
  return `${shown.slice(0, -1).join(", ")} and ${shown[shown.length - 1]}`;
}

const NUMBER_WORD = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];

const PURPOSE_CLAUSE = {
  prediction: "models are judged on rows they never saw",
  inference: "associations are estimated with their uncertainty",
};

const METHOD_NAME: Record<EnergyMethod, string> = {
  none: "no adjustment",
  standard: "standard multivariate model",
  residual: "residual method",
  density_multivariate: "multivariate nutrient density model",
  density: "nutrient density model",
  partition: "energy partition model",
};

const ROLE_GROUP: Record<Role, [string, string]> = {
  energy: ["energy", "energy"],
  exposure: ["exposure", "exposures"],
  covariate: ["covariate", "covariates"],
  identifier: ["identifier", "identifiers"],
  flag: ["flag", "flags"],
  time: ["time column", "time columns"],
  design: ["design column", "design columns"],
  excluded: ["excluded", "excluded"],
};

const finish = (s: string) => (/[.!?]$/.test(s) ? s : `${s}.`);

/** §12.4: the columns the latest missing-values answer leaves out. */
function dropOf(records: DecisionRecord[]): string[] {
  const rec = [...records].reverse().find((r) => r.decision.kind === "set_missing");
  return (rec?.decision as { drop_columns?: string[] } | undefined)?.drop_columns ?? [];
}

function measuredCount(ds: MockDataset, target: string | null): number {
  const col = target ? findColumn(ds, target) : undefined;
  if (!col) return ds.nRows;
  return ds.nRows - nMissing(col);
}

/** The sentence a decision records, given the state before it. */
export function sentenceFor(
  ds: MockDataset,
  d: Decision,
  before: ProjectState,
  records: DecisionRecord[],
): string {
  switch (d.kind) {
    case "set_lens":
      return finish(
        `The table was read through the ${listing(d.lenses, 5)} ${d.lenses.length === 1 ? "lens" : "lenses"}`,
      );
    case "set_target": {
      const m = measuredCount(ds, d.column);
      return finish(
        m === ds.nRows
          ? `${tick(d.column)} was chosen as the outcome; it is measured on all ${count(ds.nRows)} rows`
          : `${tick(d.column)} was chosen as the outcome; it is measured on ${count(m)} of ${count(ds.nRows)} rows`,
      );
    }
    case "set_task":
      return finish(`${tick(d.column)} was modeled as a ${tick(d.task)} task`);
    case "set_purpose":
      return finish(
        `The analysis was declared for ${tick(d.purpose)}: ${PURPOSE_CLAUSE[d.purpose]}`,
      );
    case "revert": {
      const undone = records.find((r) => r.id === d.decision_id);
      return finish(
        `Decision ${tick(`#${undone?.seq ?? "?"}`)} was reverted, restoring the answer before it`,
      );
    }
    case "set_roles": {
      const roles = d.roles as Record<string, Role>;
      const prev = before.roles as Record<string, Role> | null;
      const n = Object.keys(roles).length;
      if (prev) {
        const changed = Object.keys(roles).filter((c) => prev[c] !== roles[c]);
        if (changed.length === 0)
          return finish(`The roles of all ${count(n)} columns were confirmed unchanged`);
        if (changed.length <= 3) {
          const moves = changed.map(
            (c) =>
              `${tick(c)} became ${ROLE_GROUP[roles[c]!][0] === "excluded" ? "excluded" : `${/^[aeiou]/.test(ROLE_GROUP[roles[c]!][0]) ? "an" : "a"} ${ROLE_GROUP[roles[c]!][0]}`}`,
          );
          return finish(`${listing(moves, 4, false)}; every other role is unchanged`);
        }
      }
      const order: Role[] = [
        "energy",
        "exposure",
        "covariate",
        "identifier",
        "flag",
        "time",
        "design",
        "excluded",
      ];
      const groups = order
        .map((r) => {
          const cols = Object.keys(roles).filter((c) => roles[c] === r);
          if (!cols.length) return null;
          return `${ROLE_GROUP[r][cols.length === 1 ? 0 : 1]} ${listing(cols)}`;
        })
        .filter(Boolean);
      return finish(`Column roles were set for ${count(n)} columns: ${groups.join("; ")}`);
    }
    case "set_exclusions": {
      if (d.rules.length === 0) {
        return finish(
          `No rows were excluded: ${before.target ? `every row with ${tick(before.target)} measured` : "every row"} stays in the analysis`,
        );
      }
      const measured = Array.from({ length: ds.nRows }, (_, i) => {
        const col = before.target ? findColumn(ds, before.target) : undefined;
        const v = col?.values[i];
        return !col || !(v === null || v === undefined || v === "");
      });
      const removed = new Uint8Array(ds.nRows);
      const clauses = d.rules.map((rule) => {
        let n = 0;
        for (let i = 0; i < ds.nRows; i++) {
          if (measured[i] && !removed[i] && ruleExcludes(ds, rule, i)) {
            removed[i] = 1;
            n += 1;
          }
        }
        let range: string;
        if (rule.by) {
          const parts = Object.entries(rule.by.ranges).map(
            ([level, [lo, hi]], i) =>
              `outside ${tick(num(lo))}–${tick(num(hi))} for ${i === 0 ? `${tick(rule.by!.column)} ${tick(level)}` : tick(level)}`,
          );
          range = listing(parts, 6, false);
        } else if (rule.low !== null && rule.high !== null) {
          range = `outside ${tick(num(rule.low))}–${tick(num(rule.high))}`;
        } else if (rule.low !== null) range = `below ${tick(num(rule.low))}`;
        else range = `above ${tick(num(rule.high))}`;
        return `${count(n)} ${n === 1 ? "row" : "rows"} with ${tick(rule.column)} ${range} ${n === 1 ? "was" : "were"} excluded as ${rule.reason}`;
      });
      return finish(clauses.join("; "));
    }
    case "set_missing": {
      const drop = (d as { drop_columns?: string[] }).drop_columns ?? [];
      if (d.strategy === "impute") {
        return finish(
          `${drop.length ? `${listing(drop)} ${drop.length === 1 ? "was" : "were"} left out of the predictors; ` : ""}Missing predictor values were imputed from the training rows only, so no row is dropped for a blank`,
        );
      }
      const state = { ...before, missing: d.strategy } as ProjectState;
      const c = cohort(ds, state, records, drop).artifact;
      const step = c.steps.find((st) => st.key === "complete_cases");
      const prior = step ? step.n + step.dropped : c.n_final;
      const tail = `${count(c.n_final)} of ${count(prior)} rows remain`;
      return finish(
        drop.length
          ? `${listing(drop)} ${drop.length === 1 ? "was" : "were"} left out of the predictors, then rows missing any other predictor were dropped (a complete-case analysis): ${tail}`
          : `Rows missing any predictor were dropped (a complete-case analysis): ${tail}`,
      );
    }
    case "set_split": {
      const roles = rolesArtifact(ds, before);
      const how = [`seed ${tick(d.seed)}`];
      if (roles.repeats) how.push(`keeping each ${tick(roles.repeats.column)}'s rows together`);
      const folds = `${tick(d.folds)}-fold cross-validation`;
      if (d.holdout === 0)
        return finish(
          `No rows were held out; performance was estimated by ${folds} (${how.join(", ")})`,
        );
      const n = before.target
        ? cohort(ds, before, records, dropOf(records)).artifact.n_final
        : null;
      const pool = n !== null ? ` of the ${count(n)} rows` : " of rows";
      return finish(
        `A random ${tick(`${Math.round(d.holdout * 100)}%`)}${pool} (${how.join(", ")}) was held out for one final score; models were compared by ${folds} on the rest`,
      );
    }
    case "set_energy_adjustment": {
      const who = d.nutrients.length ? listing(d.nutrients) : "each nutrient";
      const many = d.nutrients.length !== 1;
      const where = d.strata ? ` within levels of ${tick(d.strata)}` : "";
      const energy = d.energy_column ? tick(d.energy_column) : "total energy";
      switch (d.method) {
        case "none":
          return finish(
            `No energy adjustment was applied: ${who} ${many ? "enter" : "enters"} the models as absolute intakes`,
          );
        case "residual":
          return finish(
            `Energy was adjusted by the residual method: ${who} ${many ? "were each" : "was"} regressed on ${energy}${where} on training rows and replaced by the residual plus the nutrient's mean`,
          );
        case "standard":
          return finish(
            `Energy was adjusted by the standard multivariate model: ${energy} enters the models beside ${who}, so each nutrient's effect is at fixed total energy`,
          );
        case "density_multivariate":
          return finish(
            `Energy was adjusted by the multivariate nutrient density model: ${who} ${many ? "were each" : "was"} divided by ${energy}${where}, which stays in the models as its own term`,
          );
        case "density":
          return finish(
            `Energy was adjusted by the nutrient density model: ${who} ${many ? "were each" : "was"} divided by ${energy}${where}, which leaves the models`,
          );
        case "partition":
          return finish(
            `Energy was partitioned: ${energy} was split into kcal from ${who} and kcal from everything else, each its own term`,
          );
      }
      return "";
    }
    case "select_models": {
      const n = d.models.length;
      const head = NUMBER_WORD[n] ?? String(n);
      const labels = d.models.map((m) => familyLabel(m, before.task).toLowerCase());
      return finish(
        `${head} model ${n === 1 ? "family was" : "families were"} chosen: ${listing(labels, 8, false)}`,
      );
    }
    case "set_substitution": {
      const energy = before.energy_adjustment?.energy_column;
      return finish(
        `The substitution studied is ${tick(d.donor)} replaced by ${tick(d.recipient)}, in steps of ${tick(num(d.step_kcal))} kcal ${energy ? `with ${tick(energy)} held fixed` : "at the same total energy"}`,
      );
    }
  }
}

// ── refusals (decisions.py validators) ───────────────────────────────────────

type Exit = Refusal["error"]["exits"][number];
const refuse = (code: string, message: string, exits: Exit[] = []): Refusal => ({
  error: { code, message, exits },
});

const METHOD_LABEL: Record<EnergyMethod, string> = {
  none: "No energy adjustment",
  standard: "Standard (multivariate) model",
  residual: "Willett residual model",
  density_multivariate: "Multivariate nutrient density model",
  density: "Nutrient density alone",
  partition: "Energy partition model",
};

const FAMILIES = ["linear", "elastic_net", "boosted_trees"];

export function validateM1(ds: MockDataset, d: Decision, state: ProjectState): Refusal | null {
  const has = (c: string) => !!findColumn(ds, c);
  switch (d.kind) {
    case "set_roles": {
      const unknown = Object.keys(d.roles).find((c) => !has(c));
      if (unknown)
        return refuse("unknown_column", `There is no column named ${tick(unknown)} in this table.`);
      if (state.target && state.target in d.roles) {
        const rest = Object.fromEntries(
          Object.entries(d.roles).filter(([c]) => c !== state.target),
        );
        return refuse(
          "target_has_role",
          `${tick(state.target)} is the outcome; it cannot also be a predictor or any other role.`,
          [
            {
              label: "Leave the outcome out of the roles",
              decision: { kind: "set_roles", roles: rest },
            },
          ],
        );
      }
      if (
        !Object.values(d.roles).some((r) => r === "exposure" || r === "covariate" || r === "energy")
      ) {
        return refuse(
          "no_predictors",
          "No column would enter the models: at least one must be an exposure, a covariate or energy.",
          [{ label: "Mark at least one column as an exposure or a covariate", decision: null }],
        );
      }
      return null;
    }
    case "set_exclusions": {
      for (const rule of d.rules) {
        const col = findColumn(ds, rule.column);
        if (!col || !isNumericDtype(col.dtype)) {
          return refuse(
            "not_numeric",
            `${tick(rule.column)} is not a numeric column, so a range cannot apply to it.`,
            [
              {
                label: `Drop the rule on ${tick(rule.column)}`,
                decision: { kind: "set_exclusions", rules: d.rules.filter((x) => x !== rule) },
              },
            ],
          );
        }
        if (rule.low !== null && rule.high !== null && rule.low >= rule.high) {
          return refuse(
            "empty_range",
            `The range keeps nothing: ${tick(num(rule.low))} is not below ${tick(num(rule.high))}.`,
            [
              {
                label: "Swap the bounds",
                decision: {
                  kind: "set_exclusions",
                  rules: d.rules.map((x) =>
                    x === rule ? { ...x, low: rule.high, high: rule.low } : x,
                  ),
                },
              },
            ],
          );
        }
      }
      return null;
    }
    case "set_energy_adjustment": {
      if (d.method === "none") return null;
      const roles = (state.roles ?? {}) as Record<string, Role>;
      const withMethod = (method: EnergyMethod): Decision => ({ ...d, method });
      if (!d.energy_column || roles[d.energy_column] !== "energy") {
        return refuse(
          "energy_role",
          `${d.energy_column ? tick(d.energy_column) : "No column"} does not have the energy role in the recorded roles.`,
          [
            { label: "Confirm the column roles", decision: null },
            { label: "Do not adjust for energy", decision: withMethod("none") },
          ],
        );
      }
      if (d.nutrients.length === 0 || d.nutrients.some((n) => roles[n] !== "exposure")) {
        return refuse("nutrients", "The nutrients to adjust must all be exposures.", [
          { label: "Choose the nutrients to adjust", decision: null },
        ]);
      }
      const reading = proposalsArtifact(ds, state).energy;
      const verdict = reading?.applicability[d.method];
      if (verdict && !verdict.ok) {
        const ok = (Object.keys(METHOD_LABEL) as EnergyMethod[]).filter(
          (m) => m !== "none" && m !== d.method && reading?.applicability[m]?.ok,
        );
        const sentences = verdict.reason.match(/[^.!?]+[.!?]+(?:\s|$)/g) ?? [verdict.reason];
        return refuse(
          "method_not_applicable",
          `The ${METHOD_NAME[d.method]} cannot run on these columns: ${sentences[sentences.length - 1]!.trim().replace(/^./, (ch) => ch.toLowerCase())}`,
          ok.map((m) => ({ label: METHOD_LABEL[m], decision: withMethod(m) })),
        );
      }
      if (d.strata) {
        const col = findColumn(ds, d.strata);
        const levels = new Set(col?.values.filter((v) => v !== null && v !== ""));
        if (!col || isNumericDtype(col.dtype) || levels.size > 10) {
          return refuse(
            "strata",
            `${tick(d.strata)} is not a categorical column with at most 10 levels.`,
            [{ label: "Adjust without strata", decision: { ...d, strata: null } }],
          );
        }
      }
      return null;
    }
    case "select_models": {
      const unknown = d.models.filter((m) => !FAMILIES.includes(m));
      if (unknown.length) {
        const known = d.models.filter((m) => FAMILIES.includes(m));
        return refuse("unknown_family", `There is no model family named ${listing(unknown)}.`, [
          ...(known.length
            ? [
                {
                  label: "Keep the families that exist",
                  decision: { kind: "select_models" as const, models: known },
                },
              ]
            : []),
          { label: "Choose from the shelf", decision: null },
        ]);
      }
      return null;
    }
    case "set_substitution": {
      if (d.donor === d.recipient) {
        return refuse(
          "same_nutrient",
          "A substitution moves calories between two different nutrients.",
          [{ label: "Choose a different recipient", decision: null }],
        );
      }
      const roles = (state.roles ?? {}) as Record<string, Role>;
      const bad = [d.donor, d.recipient].find((c) => roles[c] !== "exposure" || !energyBearing(c));
      if (bad) {
        return refuse(
          "not_energy_bearing",
          `${tick(bad)} is not an energy-bearing nutrient among the predictors.`,
          [{ label: "Choose two energy-bearing nutrients", decision: null }],
        );
      }
      return null;
    }
    default:
      return null;
  }
}

// ── findings: one line each, with a lever ────────────────────────────────────

const NO_LEVER = "No control for this yet.";

interface Voiced {
  summary: string;
  routes_to: Finding["routes_to"];
  lever_label: string | null;
  group: string | null;
  severity?: Finding["severity"];
}

function shareOf(ds: MockDataset, column: string): { n: number; share: number } {
  const col = findColumn(ds, column);
  const n = col ? nMissing(col) : 0;
  return { n, share: ds.nRows ? n / ds.nRows : 0 };
}

function pearsonTop(
  ds: MockDataset,
  state: ProjectState,
): { nutrient: string; energy: string; r: number } | null {
  const energy = proposalsArtifact(ds, state).energy;
  const top = Object.entries(energy?.r_with_energy ?? {}).sort((a, b) => b[1] - a[1])[0];
  return top && energy?.energy_column
    ? { nutrient: top[0], energy: energy.energy_column, r: top[1] }
    : null;
}

function speak(ds: MockDataset, f: Finding, state: ProjectState): Voiced {
  const kind = f.id.split("__")[0]!;
  const col = f.affected_columns[0] ?? "";
  switch (kind) {
    case "implausible_energy": {
      const energy = col;
      const values = (findColumn(ds, energy)?.values ?? []).filter(
        (v): v is number => typeof v === "number",
      );
      const low = values.filter((v) => v < 500).length;
      const high = values.filter((v) => v > 5000).length;
      return {
        summary: `${(low + high).toLocaleString("en-US")} rows report an implausible day: ${low.toLocaleString("en-US")} below 500 kcal and ${high.toLocaleString("en-US")} above 5,000.`,
        routes_to: "exclusions",
        lever_label: "Choose an exclusion rule",
        group: null,
      };
    }
    case "energy_adjustment": {
      const top = pearsonTop(ds, state);
      return {
        summary: top
          ? `Every nutrient tracks total energy: ${tick(top.nutrient)} correlates ${top.r.toFixed(2)} with ${tick(top.energy)}.`
          : "Nutrient associations are confounded by total energy.",
        routes_to: "energy_adjustment",
        lever_label: "Adjust for energy",
        group: null,
      };
    }
    case "missing": {
      const { n, share } = shareOf(ds, col);
      const often = share >= 0.5;
      return {
        summary: often
          ? `${tick(col)} is blank on ${Math.round(share * 100)}% of rows; blanks there likely mean the question was not asked.`
          : `${tick(col)} is blank on ${n.toLocaleString("en-US")} rows (${(share * 100).toFixed(1)}%).`,
        routes_to: "missing",
        lever_label: "Choose how blanks are handled",
        group: "missing",
        severity: often ? "warning" : "info",
      };
    }
    case "repeats":
      return {
        summary: finishSummary(f.title),
        routes_to: "roles",
        lever_label: "Review in Roles",
        group: null,
      };
    case "compositional_macros":
      return {
        summary: `${finishSummary(f.title)} ${NO_LEVER}`,
        routes_to: null,
        lever_label: null,
        group: null,
      };
    case "binary_text":
      return {
        summary: `${finishSummary(f.title)} The record will say which became 1. ${NO_LEVER}`,
        routes_to: null,
        lever_label: null,
        group: "binary_text",
      };
    default:
      return {
        summary: `${finishSummary(f.title)}${f.severity === "info" ? "" : ` ${NO_LEVER}`}`,
        routes_to: null,
        lever_label: null,
        group: null,
      };
  }
}

function finishSummary(text: string): string {
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

function appFindings(ds: MockDataset, state: ProjectState): Finding[] {
  const out: Finding[] = [];
  const lens = state.lens ?? [];
  const base = (
    id: string,
    severity: Finding["severity"],
    title: string,
    columns: string[],
  ): Finding => ({
    id,
    severity,
    title,
    detail: title,
    why_it_matters: null,
    affected_columns: columns,
    source: "structural",
    lens: null,
    evidence: null,
    summary: title,
    routes_to: null,
    lever_label: null,
    group: null,
  });
  const roles = rolesArtifact(ds, state);
  for (const r of roles.columns) {
    if (r.proposed === "identifier") {
      out.push({
        ...base(
          `voice::identifier__${r.column}`,
          "info",
          `${tick(r.column)} names each respondent`,
          [r.column],
        ),
        summary: `${tick(r.column)} names each respondent; it is an identifier, never a predictor.`,
        routes_to: "roles",
        lever_label: "Review in Roles",
      });
    }
  }
  for (const r of roles.columns) {
    if (r.proposed !== "flag" || !r.linked_to) continue;
    const col = findColumn(ds, r.column);
    const on = (col?.values ?? []).filter((v: Scalar) => v === true || v === "True").length;
    out.push({
      ...base(`voice::flag__${r.column}`, "info", `${tick(r.column)} flags filled-in values`, [
        r.column,
        r.linked_to,
      ]),
      summary: `${tick(r.column)} marks ${count(on)} filled-in ${tick(r.linked_to)} values; it stays out of the model.`,
      routes_to: "roles",
      lever_label: "Review the flags",
      group: "flags",
    });
  }
  const cycle = ds.columns.find((c) => /cycle|year/i.test(c.name) && isNumericDtype(c.dtype));
  if (cycle && findColumn(ds, "SEQN")) {
    const years = [
      ...new Set(cycle.values.filter((v): v is number => typeof v === "number")),
    ].sort();
    out.push({
      ...base(`voice::cycles__${cycle.name}`, "info", `${tick(cycle.name)} pools survey cycles`, [
        cycle.name,
      ]),
      summary: `${tick(cycle.name)} pools ${count(years.length)} survey cycles, ${years[0]}–${years[years.length - 1]}. ${NO_LEVER}`,
    });
    if (lens.includes("dietary") && !ds.columns.some((c) => /^(wt|sdmv)/i.test(c.name))) {
      out.push({
        ...base("voice::survey_design", "warning", "No survey design columns", []),
        summary: `No survey weights or design columns: estimates describe these rows, not the population. ${NO_LEVER}`,
      });
    }
  }
  return out;
}

const ORDER: Record<string, number> = {
  energy_adjustment: 0,
  implausible_energy: 1,
  missing: 2,
  "voice::survey_design": 3,
  "voice::flag": 4,
  "voice::identifier": 5,
};

/** The M1 voice over the mock's findings: summary, lever, pager group (M1_CONTRACT §6). */
export function voiceFindings(
  ds: MockDataset,
  artifact: FindingsArtifact,
  state: ProjectState,
): FindingsArtifact {
  const voiced = artifact.findings.map((f) => {
    const v = speak(ds, f, state);
    return { ...f, ...v, severity: v.severity ?? f.severity };
  });
  const all = [...voiced, ...appFindings(ds, state)];
  const rank = (f: Finding) => ORDER[f.id.split("__")[0]!] ?? 9;
  all.sort((a, b) => rank(a) - rank(b));
  // A pager key only where two or more findings share it.
  const sizes = new Map<string, number>();
  for (const f of all) if (f.group) sizes.set(f.group, (sizes.get(f.group) ?? 0) + 1);
  return {
    ...artifact,
    findings: all.map((f) =>
      f.group && (sizes.get(f.group) ?? 0) < 2 ? { ...f, group: null } : f,
    ),
  };
}
