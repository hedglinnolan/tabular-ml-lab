/**
 * S3 — the 13 real NHANES findings, within the doctrine (§11.7): three pushed and ranked, the rest
 * counted and typed; same-kind findings share one paged card; each is a one-line claim plus its
 * lever. The focused finding's evidence is on the stage, on the user's data, and scrubs to what its
 * lever's usual answer (or the pipeline as it stands) would do.
 */
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { Prose } from "../../../components/Prose";
import { cx } from "../../../util/format";
import { EXCL_OPTIONS, FINDING_COPY, FINDING_GROUPS, FINDINGS_PUSHED, type FindingCopy } from "../copy";
import { ENERGY, EXCLUSIONS, FINDINGS, FX, fmtInt, fmtR, view, type Finding } from "../data";
import { useScrub } from "../engine/scrub";
import { RecordBar, ScrubBar, StageFrame, StageSection } from "../Stage";
import { ScrubHistogram, type CutMark, type HistState } from "../views/Histogram";
import { MorphTable, type TableState } from "../views/MorphTable";
import { ScrubScatter, type ScatterState } from "../views/Scatter";
import s from "../Scenario.module.css";
import f from "./Findings.module.css";

const BY_ID = new Map(FINDINGS.map((x) => [x.id, x]));
const ENERGY_ID = "pack::dietary::energy_adjustment";
const INTAKE_ID = "pack::dietary::implausible_intake";

/** Level counts, read from the finding's own detail ("'True' (306 rows) and 'False' (21,543 rows)"). */
function levels(fd: Finding): { level: string; n: number }[] {
  const out = [...fd.detail.matchAll(/'([^']+)' \(([\d,]+) rows\)/g)].map((m) => ({
    level: m[1]!,
    n: Number(m[2]!.replace(/,/g, "")),
  }));
  const blank = /with ([\d,]+) blank/.exec(fd.detail);
  if (blank) out.push({ level: "", n: Number(blank[1]!.replace(/,/g, "")) });
  return out;
}

/** A column's levels as the model reads them: one-hot for a predictor, excluded otherwise. */
function levelTable(fd: Finding): { states: Record<string, TableState>; labels: string[]; slot: string; after: string } {
  const col = fd.affected_columns[0]!;
  // boolean_as_text details carry no per-level counts: its binary_text twin (same column) does.
  const src = levels(fd).length > 1 ? fd : (BY_ID.get(`binary_text__${col}`) ?? fd);
  const lv = levels(src);
  const role = FX.scenario_a.setup.state.roles[col];
  const dummy = FX.scenario_a.energy_adjustment.options[0]!.matrix_columns!.find((c) => c.startsWith(`${col}_`));
  const shown = (l: string) => (l === "" ? "blank" : l);
  const now: TableState = {
    cols: { [col]: { name: col, values: lv.map((l) => shown(l.level)), status: "same" } },
    notes: [],
  };
  let after: TableState;
  let label: string;
  if (dummy) {
    const one = dummy.slice(col.length + 1);
    after = {
      cols: { [col]: { name: dummy, values: lv.map((l) => (l.level === one ? "1" : "0")), status: "changed" } },
      notes: [{ id: "d", text: `one-hot: \`${one}\` = 1`, from: col, to: col }],
    };
    label = "As the model reads it";
  } else if (role === "excluded") {
    after = { cols: { [col]: { ...now.cols[col]!, status: "dropped" } }, notes: [{ id: "x", text: "excluded (role)", from: col, to: col }] };
    label = "As this model reads it";
  } else {
    after = {
      cols: {
        [col]: {
          name: col,
          values: lv.map((l) => (l.level === "True" ? "1" : l.level === "False" ? "0" : shown(l.level))),
          status: "changed",
        },
      },
      notes: [{ id: "b", text: "read as binary", from: col, to: col }],
    };
    label = "Read as binary";
  }
  return {
    states: { now, [fd.id]: after },
    labels: lv.map((l) => fmtInt(l.n)),
    slot: col,
    after: label,
  };
}

function energyStates(): Record<string, ScatterState> {
  const rel = view(ENERGY.residual!.preview!.views, "relationship")!;
  return {
    now: { ys: rel.points_before.map((p) => p[1]), yLabel: rel.y_label_before, r: rel.r_before, note: "climbs with kcal", after: false },
    [ENERGY_ID]: { ys: rel.points_after.map((p) => p[1]), yLabel: rel.y_label_after, r: rel.r_after, note: "the usual: residual", after: true },
  };
}

function intakeStates(): { states: Record<string, HistState>; breakAt: number } {
  const o = EXCLUSIONS.find((x) => x.key === "kcal_500_5000")!;
  const d = view(o.preview.views, "distribution")!;
  const lvl = o.counts.by_level.all!;
  const marks: CutMark[] = d.marks.map((m, i) => ({
    value: m.value,
    label: m.label,
    group: null,
    side: i === 0 ? "below" : "above",
    cut: i === 0 ? lvl.below : lvl.above,
  }));
  return {
    states: {
      now: { edges: d.before.edges, counts: d.before.counts, domain: "kcal", label: d.before_label, marks, after: false },
      [INTAKE_ID]: { edges: d.after.edges, counts: d.after.counts, ghost: d.before.counts, domain: "kcal", label: d.after_label, marks, after: true },
    },
    breakAt: d.before.edges.find((e) => e >= 7500) ?? d.before.edges[d.before.edges.length - 1]!,
  };
}

function Pager({ ids, onFocus, activeId }: { ids: string[]; onFocus: (id: string) => void; activeId: string | null }) {
  const [i, setI] = useState(0);
  const cur = Math.max(0, ids.indexOf(activeId ?? ""));
  const at = ids.includes(activeId ?? "") ? cur : i;
  const go = (d: number) => {
    const n = (at + d + ids.length) % ids.length;
    setI(n);
    onFocus(ids[n]!);
  };
  return (
    <span className={f.pager}>
      <button
        type="button"
        className={f.pageButton}
        onClick={(e) => {
          e.stopPropagation();
          go(-1);
        }}
        aria-label="Previous"
      >
        ‹
      </button>
      <span className="num">
        {at + 1} / {ids.length}
      </span>
      <button
        type="button"
        className={f.pageButton}
        onClick={(e) => {
          e.stopPropagation();
          go(1);
        }}
        aria-label="Next"
      >
        ›
      </button>
    </span>
  );
}

function Lever({ copy, onLever }: { copy: FindingCopy; onLever: (c: FindingCopy) => void }) {
  if (!copy.lever) return <span className={f.noLever}>no lever in this version</span>;
  return (
    <button
      type="button"
      className={f.lever}
      onClick={(e) => {
        e.stopPropagation();
        onLever(copy);
      }}
    >
      {copy.lever} →
    </button>
  );
}

export function FindingsScenario() {
  const { active, focus, flip } = useScrub();
  const [open, setOpen] = useState(false);
  const [receipt, setReceipt] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const energy = useMemo(() => energyStates(), []);
  const intake = useMemo(() => intakeStates(), []);
  const current = active ?? FINDINGS_PUSHED[0]!;
  const fd = BY_ID.get(current)!;
  const copy = FINDING_COPY[current]!;
  const table = useMemo(() => (current === ENERGY_ID || current === INTAKE_ID ? null : levelTable(fd)), [current, fd]);
  const rest = FINDING_GROUPS.reduce((n, g) => n + g.ids.length, 0);

  // Open on the first pushed finding, showing the user's data (not a preview).
  useEffect(() => {
    if (!active) focus(FINDINGS_PUSHED[0]!, "now");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keyboard order: the pushed cards, then (when open) the first page of each group.
  const order = [...FINDINGS_PUSHED, ...(open ? FINDING_GROUPS.map((g) => (g.ids.includes(current) ? current : g.ids[0]!)) : [])];

  const onLever = (c: FindingCopy) => {
    if (c.routes_to === "energy_adjustment" || c.routes_to === "exclusions") {
      const sKey = c.routes_to === "energy_adjustment" ? "energy" : "exclusions";
      window.history.pushState(null, "", `/lab/explore/scrub?s=${sKey}`);
      window.dispatchEvent(new Event("turbotab:navigate"));
      return;
    }
    setReceipt("The roles question opens here in the app; it is not part of this prototype.");
  };

  const onKey = (e: KeyboardEvent) => {
    const i = order.indexOf(current);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = order[Math.max(0, Math.min(order.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))];
      if (n) focus(n, "now");
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      if (!active) focus(current, "now");
      flip("after");
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      flip("now");
    } else if (e.key === "Enter" && copy.lever) {
      e.preventDefault();
      onLever(copy);
    }
  };

  const card = (id: string, extra?: ReactNode) => {
    const x = BY_ID.get(id)!;
    const c = FINDING_COPY[id]!;
    const on = current === id;
    return (
      <div
        key={id}
        role="option"
        aria-selected={on}
        className={cx(f.card, on && f.on)}
        data-severity={x.severity}
        onClick={() => {
          focus(id, "now");
          box.current?.focus();
        }}
      >
        {extra}
        <p className={f.claim}>
          <Prose text={c.summary} />
        </p>
        <div className={f.actions}>
          <Lever copy={c} onLever={onLever} />
          {x.evidence ? <span className={f.badge}>{x.evidence.status}</span> : null}
        </div>
      </div>
    );
  };

  const stageNote =
    current === ENERGY_ID
      ? "`fat_total` against `kcal`, 800 training rows"
      : current === INTAKE_ID
        ? "`kcal`, every loaded row"
        : `\`${fd.affected_columns[0]}\`, one row per value`;

  const stageAfter =
    current === ENERGY_ID
      ? "With the usual: the Willett residual"
      : current === INTAKE_ID
        ? `With ${EXCL_OPTIONS.kcal_500_5000!.short} excluded`
        : table?.after ?? null;

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <section className={f.panel} aria-label="Findings">
          <div className={f.head}>
            <span className={f.kicker}>Findings</span>
            <span className={f.total}>
              {FINDINGS.length} on this table · {FINDINGS_PUSHED.length} shown
            </span>
          </div>
          <div
            ref={box}
            role="listbox"
            tabIndex={0}
            aria-label="Findings"
            className={f.list}
            onKeyDown={onKey}
            data-testid="findings-options"
          >
            {FINDINGS_PUSHED.map((id) => card(id))}
            <button type="button" className={f.more} aria-expanded={open} onClick={() => setOpen(!open)}>
              <span className={f.moreCount}>{rest} more</span>
              <span className={f.moreKinds}>
                {FINDING_GROUPS.map((g) => `${g.ids.length} ${g.title.toLowerCase()}`).join(" · ")}
              </span>
              <span className={f.moreToggle}>{open ? "hide" : "show"}</span>
            </button>
            {open
              ? FINDING_GROUPS.map((g) => {
                  const id = g.ids.includes(current) ? current : g.ids[0]!;
                  return card(
                    id,
                    <div className={f.groupHead}>
                      <span>{g.title}</span>
                      <Pager
                        ids={g.ids}
                        activeId={current}
                        onFocus={(n) => {
                          focus(n, "now");
                          box.current?.focus();
                        }}
                      />
                    </div>,
                  );
                })
              : null}
          </div>
          {receipt ? (
            <p className={f.receipt} role="status">
              {receipt}
            </p>
          ) : null}
        </section>
      </div>
      <StageFrame label="The finding on your data">
        <ScrubBar afterLabel={active ? stageAfter : null} recorded={false} />
        <StageSection kicker="On your data" aside={<Prose text={stageNote} />}>
          {current === ENERGY_ID ? (
            <div className={f.energy}>
              <ScrubScatter xs={view(ENERGY.residual!.preview!.views, "relationship")!.points_before.map((p) => p[0])} xLabel="kcal" states={energy} height={330} noteWidth={170} />
              <div className={f.rList} aria-label="Each nutrient's correlation with kcal">
                <span className={f.rHead}>
                  r with <code className="v">kcal</code>
                </span>
                {FX.scenario_a.energy_adjustment.correlation_with_energy.map((c) => (
                  <div key={c.column} className={f.rRow}>
                    <span className={f.rName}>{c.column}</span>
                    <span className={f.rBar}>
                      <span style={{ width: `${c.r * 100}%` }} />
                    </span>
                    <span className={f.rVal}>{fmtR(c.r)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : current === INTAKE_ID ? (
            <ScrubHistogram states={intake.states} breakAt={intake.breakAt} height={300} label="kcal with 500 and 5,000 marked" />
          ) : table ? (
            <div className={f.levels}>
              <MorphTable
                slots={[table.slot]}
                rowIds={table.labels.map((_, i) => i)}
                rowLabels={table.labels}
                gutterHead="rows"
                states={table.states}
                fontPx={15}
                rowH={38}
                label={`${table.slot} levels`}
              />
            </div>
          ) : null}
        </StageSection>
        <RecordBar
          basis={FX.scenario_a.findings.basis}
          action={copy.lever ? `${copy.lever} →` : null}
          onRecord={() => onLever(copy)}
          recorded={false}
          onChange={() => {}}
        />
      </StageFrame>
    </div>
  );
}
