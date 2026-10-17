/** The small controls the Predict declaration cards share: a number, a select, a check, a column list. */
import { useId, useState, type ReactNode } from "react";
import { parseTyped, type Bounds } from "./payloads";
import c from "./declaration.module.css";

/** A number the card takes, typed as text: blank is none, unless the model needs one (`required`),
 *  and then it is invalid; outside the model's bounds it is invalid too. An invalid number is
 *  marked so, and the card does not post it (it says why at the option pressed). */
export function NumberField({
  name,
  label,
  text,
  onChange,
  bounds,
  integer,
  required = false,
}: {
  name: string;
  label: string;
  text: string;
  onChange: (text: string) => void;
  bounds: Bounds;
  integer: boolean;
  required?: boolean;
}) {
  const typed = parseTyped(text, bounds, integer, required);
  return (
    <label className={c.field}>
      <span className={c.fieldLabel}>{label}</span>
      <input
        type="number"
        className={c.input}
        data-testid={`field-${name}`}
        value={text}
        min={bounds.ge ?? bounds.gt}
        max={bounds.le ?? bounds.lt}
        step={integer ? 1 : "any"}
        aria-invalid={typed === "invalid" || undefined}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function SelectField({
  name,
  label,
  value,
  options,
  onChange,
  invalid = false,
}: {
  name: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  invalid?: boolean;
}) {
  return (
    <label className={c.field}>
      <span className={c.fieldLabel}>{label}</span>
      <select
        className={`${c.input} ${c.select}`}
        data-testid={`field-${name}`}
        value={value}
        aria-invalid={invalid || undefined}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function CheckField({
  name,
  label,
  checked,
  onChange,
}: {
  name: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className={c.check}>
      <input
        type="checkbox"
        data-testid={`field-${name}`}
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}

/** Fields that belong together, under one quiet label. */
export function FieldGroup({ legend, children }: { legend: string; children: ReactNode }) {
  return (
    <fieldset className={c.group}>
      <legend className={c.legend}>{legend}</legend>
      <div className={c.fields}>{children}</div>
    </fieldset>
  );
}

const SHOWN = 100;

/** Columns to choose any number of, as plain names with a check. A long table shows the first
 *  hundred that match what is typed, so a wide one stays light. */
export function ColumnChecks({
  group,
  legend,
  columns,
  chosen,
  onToggle,
}: {
  group: string;
  legend: string;
  columns: string[];
  chosen: ReadonlySet<string>;
  onToggle: (column: string) => void;
}) {
  const [query, setQuery] = useState("");
  const searchId = useId();
  const q = query.trim().toLowerCase();
  const matches = q ? columns.filter((col) => col.toLowerCase().includes(q)) : columns;
  const shown = matches.slice(0, SHOWN);
  return (
    <fieldset className={c.picker}>
      <legend className={c.legend}>{legend}</legend>
      {columns.length > 12 ? (
        <input
          id={searchId}
          type="search"
          className={c.input}
          placeholder="Find a column"
          aria-label={`Find a column for ${legend.toLowerCase()}`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      ) : null}
      <div className={c.picks}>
        {shown.map((col) => (
          <label key={col} className={c.check}>
            <input
              type="checkbox"
              data-testid={`pick-${group}-${col}`}
              checked={chosen.has(col)}
              onChange={() => onToggle(col)}
            />
            {col}
          </label>
        ))}
      </div>
      {matches.length > shown.length ? (
        <p className={c.more}>
          {shown.length.toLocaleString("en-US")} of {matches.length.toLocaleString("en-US")} shown;
          type to narrow.
        </p>
      ) : null}
    </fieldset>
  );
}
