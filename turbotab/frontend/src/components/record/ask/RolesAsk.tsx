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
  design: ["Survey design", "Survey design"],
  time: ["Time", "Time"],
  flag: ["Flag", "Flags"],
  excluded: ["Excluded", "Excluded"],
};

/** Chips shown per group before "N more". */
const SHOWN = 14;
/** Above this many columns, the roles question gets a search box (M2_CONTRACT §10). */
export const SEARCH_ABOVE = 200;

export function RolesAsk({
  artifact,
  current,
  ...p
}: AskProps & { artifact: RolesArtifact; current: Record<string, Role> | null }) {
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

  const roleLabel = (r: Role) => taught(p.entry, r)?.label ?? GROUP[r][0];
  const decisionFor = (roles: Record<string, Role>): Decision => ({ kind: "set_roles", roles });
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

  const summary = [
    count("exposure")
      ? `${fmtCount(count("exposure"))} ${count("exposure") === 1 ? "exposure" : "exposures"}`
      : null,
    energyCols.length ? `energy \`${energyCols[0]}\`` : null,
    count("covariate")
      ? `${fmtCount(count("covariate"))} ${count("covariate") === 1 ? "covariate" : "covariates"}`
      : null,
  ].filter(Boolean);

  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        <>
          <Taught
            text={`\`${fmtCount(nPredictors)}\` ${nPredictors === 1 ? "column becomes a predictor" : "columns become predictors"}${summary.length ? `: ${summary.join(", ")}` : ""}.`}
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
                <span className={c.groupName}>{GROUP[role][columns.length === 1 ? 0 : 1]}</span>
                <span className={c.groupCount}>
                  {matching
                    ? `${fmtCount(columns.length)} of ${fmtCount(count(role))}`
                    : fmtCount(columns.length)}
                </span>
                <span className={c.groupNote}>
                  {predictor ? "predictors" : "kept out of the models"}
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
                          title={taught(p.entry, r)?.consequence}
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
            title="Records every column's role. Only exposures, covariates and energy become predictors."
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
