/**
 * The roles question: a compact, grouped confirmation of what TurboTab read each column to
 * be, editable per column. Predictors first (exposures, energy, covariates); the roles that
 * stay out of the models after. A nested nutrient shows its parent (`sugar` ⊂ `carb`), a flag
 * the column it marks. Pressing a column opens its role menu in place, with the reason for
 * the proposal; every edit previews on the stage before anything is recorded.
 */
import { useId, useMemo, useState } from "react";
import type { Role, RolesArtifact } from "../../../api/m1-types";
import { PREDICTOR_ROLES, ROLES } from "../../../api/m1-types";
import type { Decision } from "../../../api/schema";
import { useStageFocus, type StageFocus } from "../../../state/focus";
import { buildColumnIndex, filterColumns } from "../../../util/filterColumns";
import { cx } from "../../../util/format";
import { predictorGroups } from "../../../quest/cards/roles/roles";
import { V } from "../../Prose";
import { Question } from "../Question";
import { Taught } from "../teach";
import { Actions, Keep, RecordButton, fmtCount, taught, type AskProps } from "./common";
import c from "./ask.module.css";

const GROUP: Record<Role, [one: string, many: string]> = {
  exposure: ["Exposure", "Exposures"],
  energy: ["Energy", "Energy"],
  covariate: ["Covariate", "Covariates"],
  identifier: ["Identifier", "Identifiers"],
  cluster: ["Cluster", "Clusters"],
  design: ["Survey design", "Survey design"],
  time: ["Time", "Time"],
  flag: ["Flag", "Flags"],
  excluded: ["Excluded", "Excluded"],
};

/**
 * The card's own words under Estimate (and with no goal), as they stood: what you study, energy and
 * the covariates become predictors.
 */
const ESTIMATE = {
  become: (n: number) => (n === 1 ? "column becomes a predictor" : "columns become predictors"),
  note: "predictors",
  title: "Records every column's role. Only exposures, covariates and energy become predictors.",
};

/**
 * Under prediction every predictor role is a predictor (the roles ruling of 2026-10-17): the card
 * counts the columns that enter the models as predictors, and names each predictor role's group by
 * the project's lenses (nutrients under the dietary lens, assay features under an assay lens), its
 * note naming each step that treats it differently only where the project can run it
 * (core/readings.py `predictor_groups`): the energy-aware fill while a column holds the energy role,
 * under any lens; the screened elastic net's screen where the project's task puts that family on the
 * shelf (a regression or a yes/no outcome); normalization and batch correction under an assay lens.
 * The words are the engine's (roles/fixture.json).
 */
const PREDICT = {
  become: (n: number) => (n === 1 ? "column enters the models as a predictor" : "columns enter the models as predictors"),
};

/** "Nutrients, total energy and characteristics": the predictor groups in a sentence. */
const listed = (words: string[]) =>
  words.length < 3 ? words.join(" and ") : `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;

/** Chips shown per group before "N more". */
const SHOWN = 14;
/** Above this many columns, the roles question gets a search box (M2_CONTRACT §10). */
export const SEARCH_ABOVE = 200;

export function RolesAsk({
  artifact,
  current,
  purpose = null,
  lenses = null,
  task = null,
  ...p
}: AskProps & {
  artifact: RolesArtifact;
  current: Record<string, Role> | null;
  /** The project's goal: under prediction the card says predictors (`PREDICT`). */
  purpose?: string | null;
  /** The project's lenses: under prediction they name the predictor roles' groups. */
  lenses?: readonly string[] | null;
  /** The project's task, as answered or as the outcome reads: under prediction it sets whether the
   *  screened elastic net's screen is on the shelf, so whether a group's note names it. */
  task?: string | null;
}) {
  const predict = purpose === "prediction";
  /** The predictor groups under prediction with these roles: a note names the energy-aware fill
   *  only while one of them holds the energy role. */
  const groupsOf = (roles: Record<string, Role>) => (predict ? predictorGroups(lenses, { task, roles }) : null);
  const groupIn = (groups: ReturnType<typeof groupsOf>, role: Role) =>
    groups && (role === "exposure" || role === "energy" || role === "covariate") ? groups[role] : null;
  const proposed = useMemo(
    () => Object.fromEntries(artifact.columns.map((col) => [col.column, col.proposed])),
    [artifact],
  );
  // The baseline is what is on record when reopened, else the proposal.
  const baseline = useMemo<Record<string, Role>>(
    () =>
      Object.fromEntries(
        artifact.columns.map((col) => [col.column, current?.[col.column] ?? col.proposed]),
      ),
    [artifact, current],
  );
  const [draft, setDraft] = useState<Record<string, Role>>(baseline);
  const [open, setOpen] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<ReadonlySet<Role>>(new Set());
  const menuId = useId();
  const { setFocus, preview, endPreview } = useStageFocus();
  const byColumn = useMemo(
    () => new Map(artifact.columns.map((col) => [col.column, col])),
    [artifact],
  );
  const named = groupsOf(draft);
  const group = (role: Role) => groupIn(named, role);
  const groupName = (role: Role, n: number) => (group(role)?.head ?? GROUP[role])[n === 1 ? 0 : 1]!;

  const roleLabel = (r: Role) => group(r)?.head[0] ?? taught(p.entry, r)?.label ?? GROUP[r][0];
  /** What a role does, on its button: under prediction a predictor role's group note in the roles
   *  the press would leave (`roles`), none where the project runs no step that sets it apart. */
  const roleTitle = (r: Role, roles: Record<string, Role>) => {
    const g = groupIn(groupsOf(roles), r);
    return g ? g.note || undefined : taught(p.entry, r)?.consequence;
  };
  // ``unconfirmed`` is the server's to fill (BLUEPRINT §14): the proposals below high this answer
  // records as proposed, read from the roles it showed.
  const decisionFor = (roles: Record<string, Role>): Decision => ({
    kind: "set_roles",
    roles,
    unconfirmed: [],
  });
  const focusFor = (roles: Record<string, Role>, label: string): StageFocus => ({
    kind: "option",
    decision: decisionFor(roles),
    label,
  });

  // Wide tables (M2_CONTRACT §5, §10): above 200 columns the roles are searched, not scrolled.
  const wide = artifact.columns.length > SEARCH_ABOVE;
  const [query, setQuery] = useState("");
  const index = useMemo(
    () => (wide ? buildColumnIndex(artifact.columns.map((col) => col.column)) : null),
    [artifact, wide],
  );
  const matching = useMemo(() => {
    if (!index || !query.trim()) return null;
    return new Set(filterColumns(index, query).map((i) => index.names[i]!));
  }, [index, query]);

  const changed = Object.keys(draft).filter((col) => draft[col] !== baseline[col]);
  const edited = Object.keys(draft).filter((col) => draft[col] !== proposed[col]);
  const groups = ROLES.map((role) => ({
    role,
    columns: artifact.columns
      .filter((col) => draft[col.column] === role && (!matching || matching.has(col.column)))
      .map((col) => col.column),
  })).filter((g) => g.columns.length > 0);
  const nPredictors = Object.values(draft).filter((r) => PREDICTOR_ROLES.includes(r)).length;
  const count = (r: Role) => Object.values(draft).filter((x) => x === r).length;
  const energyCols = Object.keys(draft).filter((col) => draft[col] === "energy");

  const choose = (column: string, role: Role) => {
    const next = { ...draft, [column]: role };
    setDraft(next);
    setOpen(null);
    setFocus(focusFor(next, `${column} as ${roleLabel(role).toLowerCase()}`));
  };

  const recordAll = () => p.record(decisionFor(draft), "confirm");
  const confirmLabel =
    changed.length === 0
      ? current
        ? "Record these roles again"
        : `Confirm the roles of ${fmtCount(artifact.columns.length)} columns`
      : `Record these roles (${changed.length} changed)`;

  const counted = (role: Role, estimate: [string, string]) => {
    const n = count(role);
    if (!n) return null;
    const words = group(role)?.head.map((w) => w.toLowerCase()) ?? estimate;
    return `${fmtCount(n)} ${words[n === 1 ? 0 : 1]}`;
  };
  const summary = [
    counted("exposure", ["exposure", "exposures"]),
    energyCols.length ? `${named ? named.energy.head[0].toLowerCase() : "energy"} \`${energyCols[0]}\`` : null,
    counted("covariate", ["covariate", "covariates"]),
  ].filter(Boolean);
  const become = (n: number) => (predict ? PREDICT : ESTIMATE).become(n);
  const title = named
    ? `Records every column's role. ${listed([named.exposure.head[1], named.energy.head[0].toLowerCase(), named.covariate.head[1].toLowerCase()])} enter the models as predictors.`
    : ESTIMATE.title;

  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        <>
          <Taught
            text={`\`${fmtCount(nPredictors)}\` ${become(nPredictors)}${summary.length ? `: ${summary.join(", ")}` : ""}.`}
          />
          {artifact.repeats ? (
            <>
              {" "}
              <V>{artifact.repeats.column}</V> repeats: <V>{fmtCount(artifact.repeats.n_units)}</V>{" "}
              people, up to <V>{fmtCount(artifact.repeats.max_rows_per_unit)}</V> rows each.
            </>
          ) : null}
        </>
      }
    >
      {wide ? (
        <div className={c.search} role="search">
          <input
            type="search"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(null);
            }}
            placeholder={`Find among ${fmtCount(artifact.columns.length)} columns`}
            aria-label="Find a column by name"
            data-testid="roles-search"
          />
          <span className={c.searchCount} aria-live="polite" data-testid="roles-search-count">
            {matching
              ? `${fmtCount(matching.size)} of ${fmtCount(artifact.columns.length)}`
              : `${fmtCount(artifact.columns.length)} columns`}
          </span>
        </div>
      ) : null}
      {matching && matching.size === 0 ? (
        <p className={c.reason}>No column name holds those words.</p>
      ) : null}
      <div className={c.groups} data-testid="roles-groups">
        {groups.map(({ role, columns }) => {
          const predictor = PREDICTOR_ROLES.includes(role);
          const all = expanded.has(role);
          const shown = all ? columns : columns.slice(0, SHOWN);
          const openHere = open !== null && columns.includes(open);
          const col = openHere ? byColumn.get(open!) : undefined;
          return (
            <div
              key={role}
              className={c.group}
              data-role={role}
              data-predictor={predictor || undefined}
            >
              <div className={c.groupHead}>
                <span className={c.groupName}>{groupName(role, columns.length)}</span>
                <span className={c.groupCount}>
                  {matching
                    ? `${fmtCount(columns.length)} of ${fmtCount(count(role))}`
                    : fmtCount(columns.length)}
                </span>
                <span className={c.groupNote}>
                  {group(role)?.note ?? (predictor ? ESTIMATE.note : "kept out of the models")}
                </span>
              </div>
              <ul className={cx(c.chips, all && columns.length > 60 && c.chipsScroll)}>
                {shown.map((name) => {
                  const info = byColumn.get(name);
                  const isOpen = open === name;
                  return (
                    <li key={name}>
                      <button
                        type="button"
                        className={c.chip}
                        aria-expanded={isOpen}
                        aria-controls={isOpen ? menuId : undefined}
                        aria-label={`${name}: ${roleLabel(draft[name]!)}. Change its role.`}
                        data-changed={draft[name] !== proposed[name] || undefined}
                        data-confidence={info?.confidence}
                        data-testid={`role-chip-${name}`}
                        onClick={() => setOpen(isOpen ? null : name)}
                      >
                        <span className={c.chipName}>{name}</span>
                        {info?.nested_in ? (
                          <span className={c.chipNote} title={`Part of ${info.nested_in}`}>
                            ⊂ {info.nested_in}
                          </span>
                        ) : null}
                        {info?.linked_to ? (
                          <span className={c.chipNote} title={`Marks ${info.linked_to}`}>
                            → {info.linked_to}
                          </span>
                        ) : null}
                        {info && info.confidence !== "high" ? (
                          <span
                            className={c.unsure}
                            aria-hidden="true"
                            title={`${info.confidence} confidence`}
                          />
                        ) : null}
                      </button>
                    </li>
                  );
                })}
                {columns.length > SHOWN ? (
                  <li>
                    <button
                      type="button"
                      className={c.more}
                      aria-expanded={all}
                      onClick={() =>
                        setExpanded((x) => {
                          const next = new Set(x);
                          if (next.has(role)) next.delete(role);
                          else next.add(role);
                          return next;
                        })
                      }
                    >
                      {all ? "show fewer" : `${fmtCount(columns.length - SHOWN)} more`}
                    </button>
                  </li>
                ) : null}
              </ul>
              {col ? (
                <div
                  id={menuId}
                  className={c.menu}
                  role="group"
                  aria-label={`Role of ${col.column}`}
                  data-testid="role-menu"
                  onPointerLeave={endPreview}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") {
                      e.stopPropagation();
                      setOpen(null);
                    }
                  }}
                >
                  <p className={c.menuWhy}>
                    <V>{col.column}</V> was read as{" "}
                    <strong>{roleLabel(col.proposed).toLowerCase()}</strong>, {col.confidence}{" "}
                    confidence: <Taught text={col.reason} />
                    {col.unit ? (
                      <>
                        {" "}
                        Unit <V>{col.unit}</V>.
                      </>
                    ) : null}
                  </p>
                  <div className={c.roleButtons}>
                    {ROLES.map((r) => {
                      const next = { ...draft, [col.column]: r };
                      const f = focusFor(next, `${col.column} as ${roleLabel(r).toLowerCase()}`);
                      return (
                        <button
                          key={r}
                          type="button"
                          className={c.roleButton}
                          aria-pressed={draft[col.column] === r}
                          title={roleTitle(r, next)}
                          onPointerMove={() => preview(f)}
                          onFocus={() => setFocus(f)}
                          onClick={() => choose(col.column, r)}
                          data-testid={`role-${r}`}
                        >
                          {roleLabel(r)}
                          {r === col.proposed ? (
                            <span className={c.proposedMark}>proposed</span>
                          ) : null}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <Actions>
        <span
          onPointerMove={() => preview(focusFor(draft, "These roles"))}
          onPointerLeave={endPreview}
          onFocus={() => setFocus(focusFor(draft, "These roles"))}
        >
          <RecordButton
            disabled={p.pending}
            onClick={recordAll}
            title={title}
            testId="record-roles"
          >
            {confirmLabel}
          </RecordButton>
        </span>
        {edited.length > 0 ? (
          <button
            type="button"
            className={c.ghost}
            onClick={() => {
              setDraft(proposed);
              setOpen(null);
            }}
            title="Puts every column back to what TurboTab proposed. Nothing is recorded."
          >
            Back to the proposal
          </button>
        ) : null}
        <Keep keep={p.keep} />
      </Actions>
      {p.answerAt?.key === "confirm" ? <div className={c.answer}>{p.answerAt.node}</div> : null}
    </Question>
  );
}
