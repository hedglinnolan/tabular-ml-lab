/**
 * /lab — every motion primitive the app is allowed, on mock data, with a
 * reduced-motion toggle. This is where motion is reviewed before it ships.
 * The closed list (DESIGN_LANGUAGE §05.2): settle, arrive, propagate, and the
 * working table under a reshape. Nothing else moves.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, LayoutGroup, motion } from "motion/react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { Header } from "../components/Header";
import { V } from "../components/Prose";
import { DecisionSentence, Pending, QuestionBlock } from "../components/record/blocks";
import c from "../components/record/controls.module.css";
import { Arrive } from "../motion/Arrive";
import { NumberTween } from "../motion/NumberTween";
import { useMotionPrefs, useTransitions } from "../motion/prefs";
import { StaleVeil, type VeilState } from "../motion/StaleVeil";
import { fmtInt } from "../util/format";
import styles from "./LabScreen.module.css";

function Demo({ title, rule, children }: { title: string; rule: string; children: ReactNode }) {
  return (
    <section className={styles.demo}>
      <div className={styles.demoHead}>
        <h2 className={styles.h2}>{title}</h2>
        <p className={styles.rule}>{rule}</p>
      </div>
      <div className={styles.stage}>{children}</div>
    </section>
  );
}

function SettleDemo() {
  const [answer, setAnswer] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  return (
    <LayoutGroup id="lab-settle">
      {open || !answer ? (
        <QuestionBlock
          layoutId="lab-settle"
          kicker="Fact"
          title={
            <>
              Is <V>ward_id</V> categorical?
            </>
          }
          why="Numbers that name things are not quantities; a model must not average them."
        >
          <div className={c.cards}>
            {["categorical", "numeric"].map((a) => (
              <button
                key={a}
                type="button"
                className={c.card}
                aria-pressed={answer === a}
                onClick={() => {
                  setAnswer(a);
                  setOpen(false);
                }}
              >
                <span className={c.cardTitle}>
                  {a === "categorical" ? "Yes, it names wards" : "No, it is a quantity"}
                </span>
                <span className={c.cardBody}>
                  {a === "categorical" ? "Encoded as categories." : "Used as a number."}
                </span>
              </button>
            ))}
          </div>
        </QuestionBlock>
      ) : (
        <DecisionSentence
          layoutId="lab-settle"
          subject="ward_id's type"
          onChange={() => setOpen(true)}
          meta="#1"
        >
          <V>ward_id</V> was recorded as <V>{answer}</V>.
        </DecisionSentence>
      )}
    </LayoutGroup>
  );
}

function ArriveDemo() {
  const [answered, setAnswered] = useState(false);
  return (
    <div className={styles.column}>
      <div className={styles.row}>
        <button
          type="button"
          className={c.primary}
          onClick={() => setAnswered(true)}
          disabled={answered}
        >
          Answer: people repeat
        </button>
        <button type="button" className={c.ghost} onClick={() => setAnswered(false)}>
          Reset
        </button>
      </div>
      {answered ? (
        <DecisionSentence layoutId="lab-arrive-cause" subject="grain">
          Rows repeat within <V>participant_id</V>.
        </DecisionSentence>
      ) : (
        <Pending>Not yet answered: does a person appear on more than one row?</Pending>
      )}
      {answered ? (
        <Arrive className={styles.arrived}>
          <p className={styles.arrivedTitle}>Which column identifies a person?</p>
          <p className={styles.arrivedBody}>
            This section exists because of the answer above it, so it grows from there.
          </p>
        </Arrive>
      ) : null}
    </div>
  );
}

const SECTIONS = ["Task", "Findings", "Columns", "Results"] as const;

function PropagateDemo() {
  const [states, setStates] = useState<VeilState[]>(SECTIONS.map(() => "fresh"));
  const [n, setN] = useState([600, 7, 17, 0]);
  const timers = useRef<number[]>([]);
  useEffect(() => () => timers.current.forEach((t) => window.clearTimeout(t)), []);

  const change = () => {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
    setStates(SECTIONS.map(() => "stale"));
    SECTIONS.forEach((_, i) => {
      timers.current.push(
        window.setTimeout(
          () => setStates((s) => s.map((v, j) => (j === i ? "recomputing" : v))),
          700 + i * 160,
        ),
        window.setTimeout(
          () => {
            setStates((s) => s.map((v, j) => (j === i ? "fresh" : v)));
            setN((cur) =>
              cur.map((v, j) =>
                j === i ? (i === 0 ? (v === 600 ? 555 : 600) : i === 1 ? (v === 7 ? 5 : 7) : v) : v,
              ),
            );
          },
          1300 + i * 260,
        ),
      );
    });
  };
  return (
    <div className={styles.column}>
      <div className={styles.row}>
        <button type="button" className={c.primary} onClick={change}>
          Change an upstream answer
        </button>
        <span className={styles.hint}>
          Sections veil in document order, then clear one by one as their results arrive.
        </span>
      </div>
      {SECTIONS.map((name, i) => (
        <StaleVeil key={name} state={states[i]!} order={i} testId={`lab-veil-${i}`}>
          <div className={styles.fake}>
            <span className={styles.fakeTitle}>{name}</span>
            <span className={styles.fakeValue}>
              <NumberTween value={n[i]!} />
            </span>
            <button type="button" className={c.ghost}>
              A control that goes inert while stale
            </button>
          </div>
        </StaleVeil>
      ))}
    </div>
  );
}

function NumberDemo() {
  const [v, setV] = useState(600);
  return (
    <div className={styles.column}>
      <p className={styles.big}>
        <NumberTween value={v} /> <span className={styles.unit}>rows</span>
      </p>
      <div className={styles.row}>
        <button type="button" className={c.ghost} onClick={() => setV(555)}>
          Exclude 45 implausible recalls
        </button>
        <button type="button" className={c.ghost} onClick={() => setV(600)}>
          Restore all 600
        </button>
        <button type="button" className={c.ghost} onClick={() => setV(1_204_332)}>
          Load 1,204,332 rows
        </button>
      </div>
    </div>
  );
}

interface LabRow {
  row_id: number;
  participant_id: string;
  recall: number;
  energy_kcal: number;
  hba1c: number;
}

const ROWS: LabRow[] = [
  { row_id: 0, participant_id: "P001", recall: 1, energy_kcal: 3063, hba1c: 5.1 },
  { row_id: 1, participant_id: "P001", recall: 2, energy_kcal: 1913, hba1c: 5.1 },
  { row_id: 2, participant_id: "P002", recall: 1, energy_kcal: 1228, hba1c: 5.3 },
  { row_id: 3, participant_id: "P002", recall: 2, energy_kcal: 312, hba1c: 5.3 },
  { row_id: 4, participant_id: "P003", recall: 1, energy_kcal: 2410, hba1c: 6.2 },
  { row_id: 5, participant_id: "P003", recall: 2, energy_kcal: 6840, hba1c: 6.2 },
  { row_id: 6, participant_id: "P004", recall: 1, energy_kcal: 1788, hba1c: 5.7 },
  { row_id: 7, participant_id: "P004", recall: 2, energy_kcal: 2204, hba1c: 5.7 },
];

const col = createColumnHelper<LabRow>();
const COLUMNS = [
  col.accessor("row_id", { header: "#" }),
  col.accessor("participant_id", { header: "participant_id" }),
  col.accessor("recall", { header: "recall" }),
  col.accessor("energy_kcal", { header: "energy_kcal", cell: (i) => fmtInt(i.getValue()) }),
  col.accessor("hba1c", { header: "hba1c" }),
];

function TableMorphDemo() {
  const t = useTransitions();
  const [excluded, setExcluded] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([]);
  const data = useMemo(
    () => (excluded ? ROWS.filter((r) => r.energy_kcal >= 500 && r.energy_kcal <= 5000) : ROWS),
    [excluded],
  );
  // eslint-disable-next-line react-hooks/incompatible-library -- TanStack Table is the chosen table model
  const table = useReactTable({
    data,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getRowId: (r) => String(r.row_id),
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });
  return (
    <div className={styles.column}>
      <div className={styles.row}>
        <button type="button" className={c.ghost} onClick={() => setExcluded((e) => !e)}>
          {excluded ? "Restore the excluded recalls" : "Exclude recalls outside 500–5,000 kcal"}
        </button>
        <button
          type="button"
          className={c.ghost}
          onClick={() => setSorting((s) => (s.length ? [] : [{ id: "energy_kcal", desc: true }]))}
        >
          {sorting.length ? "Back to file order" : "Sort by energy"}
        </button>
        <span className={styles.hint}>
          Rows keep their identity (<V>__row_id</V>): they move, leave and return; they are never
          redrawn as different rows.
        </span>
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => (
                  <th key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {table.getRowModel().rows.map((row) => (
                <motion.tr
                  key={row.id}
                  layout="position"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={t.row}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>
      <p className={styles.hint}>
        n = <NumberTween value={data.length} />
      </p>
    </div>
  );
}

export function LabScreen() {
  const { reduced, system, override, setOverride } = useMotionPrefs();
  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.intro}>
          <h1 className={styles.h1}>Motion lab</h1>
          <p className={styles.lede}>
            Motion here has one job: to keep an object's identity across a change, so you never lose
            track of what became what. These are the only motions the app makes.
          </p>
          <div className={styles.toggleRow}>
            <button
              type="button"
              role="switch"
              aria-checked={reduced}
              className={styles.switch}
              onClick={() => setOverride(!reduced)}
              data-testid="reduced-motion"
            >
              <span className={styles.knob} aria-hidden="true" />
              Reduce motion
            </button>
            <span className={styles.hint}>
              {override === null
                ? `Following the system setting (${system ? "reduce" : "no preference"}).`
                : `Overriding the system setting (${system ? "reduce" : "no preference"}).`}
            </span>
            {override !== null ? (
              <button type="button" className={styles.link} onClick={() => setOverride(null)}>
                Follow the system again
              </button>
            ) : null}
          </div>
        </div>
        <Demo
          title="Settle"
          rule="An answered question becomes its decision sentence: the same element, not a swap."
        >
          <SettleDemo />
        </Demo>
        <Demo title="Arrive" rule="A section caused by an answer grows downward from that answer.">
          <ArriveDemo />
        </Demo>
        <Demo
          title="Propagate"
          rule="Staleness sweeps downstream in order; stale content is veiled, tagged and inert, never deleted."
        >
          <PropagateDemo />
        </Demo>
        <Demo
          title="Number tween"
          rule="A changed count tweens from its old value so it reads as the same quantity."
        >
          <NumberDemo />
        </Demo>
        <Demo
          title="Working table under a reshape"
          rule="Rows that survive a change keep their place in the eye; excluded rows leave, restored rows return."
        >
          <TableMorphDemo />
        </Demo>
      </main>
    </>
  );
}
