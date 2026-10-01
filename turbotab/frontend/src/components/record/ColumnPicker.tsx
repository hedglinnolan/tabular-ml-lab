/**
 * A searchable, virtualized column picker (ARIA combobox + listbox). Only the
 * visible options are in the DOM, and filtering is one pass over pre-lowercased
 * names, so it stays fast at 20,000 columns.
 *
 * Keys: ↑/↓ move, PageUp/PageDown jump, Enter chooses, Escape clears the search.
 */
import { useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { ColumnInfo, ColumnSummary } from "../../api/schema";
import { buildColumnIndex, filterColumns } from "../../util/filterColumns";
import { codeLike, cx, fmtCode, fmtInt, fmtStat } from "../../util/format";
import styles from "./ColumnPicker.module.css";

const ROW = 34;

interface Props {
  columns: ColumnInfo[];
  value: string | null;
  onChange: (name: string) => void;
  /** Enter on the already-chosen option. */
  onCommit?: (name: string) => void;
  summaries?: Map<string, ColumnSummary>;
  labelId?: string;
}

export function ColumnPicker({ columns, value, onChange, onCommit, summaries, labelId }: Props) {
  const id = useId();
  const listId = `${id}-list`;
  const [query, setQuery] = useState("");
  const index = useMemo(() => buildColumnIndex(columns.map((c) => c.name)), [columns]);
  const matches = useMemo(() => filterColumns(index, query), [index, query]);
  const [activeRaw, setActive] = useState(0);
  const active = matches.length === 0 ? -1 : Math.min(activeRaw, matches.length - 1);
  const scrollRef = useRef<HTMLDivElement>(null);

  // eslint-disable-next-line react-hooks/incompatible-library -- TanStack Virtual is the chosen virtualizer
  const virtualizer = useVirtualizer({
    count: matches.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW,
    overscan: 8,
  });

  const optionId = (i: number) => `${id}-opt-${i}`;
  const move = (to: number) => {
    if (matches.length === 0) return;
    const j = Math.max(0, Math.min(matches.length - 1, to));
    setActive(j);
    virtualizer.scrollToIndex(j, { align: "auto" });
  };

  const choose = (i: number) => {
    const ci = matches[i];
    if (ci === undefined) return;
    const name = columns[ci]!.name;
    if (name === value && onCommit) onCommit(name);
    else onChange(name);
    setActive(i);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    const page = Math.max(1, Math.floor((scrollRef.current?.clientHeight ?? 260) / ROW) - 1);
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        move(active + 1);
        break;
      case "ArrowUp":
        e.preventDefault();
        move(active - 1);
        break;
      case "PageDown":
        e.preventDefault();
        move(active + page);
        break;
      case "PageUp":
        e.preventDefault();
        move(active - page);
        break;
      case "Enter":
        e.preventDefault();
        if (active >= 0) choose(active);
        break;
      case "Escape":
        if (query) {
          e.preventDefault();
          setQuery("");
          setActive(0);
        }
        break;
    }
  };

  // Describe the option under the keyboard while searching; otherwise the chosen one.
  const activeCol = active >= 0 ? columns[matches[active]!] : undefined;
  const detailName = activeCol?.name ?? value;
  const detail = detailName ? summaries?.get(detailName) : undefined;
  const detailIsActiveOnly = detail !== undefined && detail.name !== value;

  return (
    <div className={styles.picker}>
      <div className={styles.searchRow}>
        <input
          className={styles.search}
          type="text"
          role="combobox"
          aria-expanded="true"
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={active >= 0 ? optionId(active) : undefined}
          aria-labelledby={labelId}
          aria-label={labelId ? undefined : "Search columns"}
          placeholder={`Search ${fmtInt(columns.length)} columns`}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onKeyDown={onKeyDown}
          spellCheck={false}
          autoComplete="off"
        />
        <span className={styles.count} aria-live="polite" data-testid="picker-count">
          {matches.length === columns.length
            ? `${fmtInt(columns.length)} columns`
            : `${fmtInt(matches.length)} of ${fmtInt(columns.length)}`}
        </span>
      </div>
      <div ref={scrollRef} className={styles.scroll}>
        <div
          id={listId}
          role="listbox"
          aria-label="Columns"
          className={styles.list}
          style={{ height: virtualizer.getTotalSize() }}
        >
          {virtualizer.getVirtualItems().map((item) => {
            const col = columns[matches[item.index]!]!;
            const selected = col.name === value;
            return (
              <div
                key={item.key}
                id={optionId(item.index)}
                role="option"
                aria-selected={selected}
                data-active={item.index === active || undefined}
                className={cx(styles.option, selected && styles.selected)}
                style={{ transform: `translateY(${item.start}px)`, height: ROW }}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => choose(item.index)}
              >
                <span className={styles.name}>{col.name}</span>
                <span className={styles.dtype}>{col.dtype}</span>
                <span className={styles.stats}>
                  {fmtInt(col.n_unique)} distinct
                  {col.n_missing ? ` · ${fmtInt(col.n_missing)} missing` : ""}
                </span>
              </div>
            );
          })}
        </div>
        {matches.length === 0 ? (
          <p className={styles.empty}>No column name contains “{query}”.</p>
        ) : null}
      </div>
      {detail ? (
        <p
          className={cx(styles.detail, detailIsActiveOnly && styles.activeOnly)}
          data-testid="picker-detail"
        >
          <span className="v">{detail.name}</span>{" "}
          {detail.median !== null
            ? codeLike(detail.name, detail)
              ? `values ${fmtCode(detail.min)} – ${fmtCode(detail.max)}`
              : `median ${fmtStat(detail.median)}, range ${fmtStat(detail.min)} – ${fmtStat(detail.max)}`
            : detail.top?.length
              ? `most common: ${detail.top
                  .slice(0, 3)
                  .map((t) => `${String(t.value)} (${fmtInt(t.count)})`)
                  .join(", ")}`
              : ""}
        </p>
      ) : null}
    </div>
  );
}
