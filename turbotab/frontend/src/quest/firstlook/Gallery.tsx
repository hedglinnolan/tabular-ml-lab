/**
 * First look's gallery, on the canvas (FOUNDATION §3 stage 3, "a guide plus a gallery"; §5 rule 6;
 * FIRST_LOOK_BRIEF §1.3–§1.4, §6), in the guide's order (Guide.tsx): each column as it is, on every
 * row of the table; what Explore found, outcome-free, each look with "Decide now" where its lever
 * records; and the outcome last, behind its gates: its distribution, how it moves with each
 * predictor, and the first rows with it beside the other columns.
 *
 * An outcome view opens by its own press, which posts the record the server handed back for it
 * (`view_outcome`: GET /columns' for the distribution, GET /table's for the rows, the explore
 * finding's for a relationship), and is drawn only once recorded; before its gate it says, in the
 * server's words, when it opens, and offers nothing to press. That press is the tapestry's one
 * record (Tapestry.tsx): a look is not a choice. "Decide now" records nothing here: it hands the
 * look to the card through the stage focus (`{ kind: "finding" }`), which asks its lever's question.
 */
import { useState, type ReactNode } from "react";
import { useDecide } from "../../api/queries";
import type { ProjectView, QuestLog } from "../../api/schema";
import { fmtPct } from "../../components/views/common/parts";
import { Plain, kit as k } from "../../kit";
import { StaleVeil } from "../../motion/StaleVeil";
import { useStageFocus } from "../../state/focus";
import { fmtInt, fmtValue, listJoin } from "../../util/format";
import { type FirstLook, useColumnHistogram, useFirstLook, useOutcomeHistogram } from "./data";
import { type ColumnLook, type Look, ROWS_SHOWN, TITLE, decideNow, exploreLine, listedColumns, outcomeTitle, rowsWords, sectionId } from "./model";
import { Bars, Curve, GroupShares, Levels } from "./views";
import s from "./firstlook.module.css";

export function FirstLookGallery({ pid, view, log }: { pid: string; view: ProjectView; log: QuestLog }) {
  const fl = useFirstLook(pid, view, log);
  return (
    <div className={s.gallery} data-testid="first-look-gallery">
      <Columns pid={pid} fl={fl} />
      {fl.target ? <Explore fl={fl} /> : null}
      {fl.target ? <Outcome pid={pid} fl={fl} view={view} /> : null}
    </div>
  );
}

function Section({ id, title, caption, children }: { id: string; title: string; caption?: string | null; children: ReactNode }) {
  const at = sectionId(id);
  return (
    <section id={at} className={s.section} tabIndex={-1} aria-labelledby={`${at}-title`} data-testid={at}>
      <h3 id={`${at}-title`} className={s.sectionTitle}>
        {title}
      </h3>
      {caption ? <p className={s.caption}>{caption}</p> : null}
      {children}
    </section>
  );
}

/** What a press that did not record says, at the press. */
function Refused({ message }: { message: string }) {
  return (
    <p className={s.refused} role="alert" data-testid="look-refused">
      Not recorded: <Plain text={message} />
    </p>
  );
}

/** A read the server did not answer, said where its drawing would be. */
function Unread({ message }: { message: string }) {
  return (
    <p className={s.refused} data-testid="look-unread">
      Not read: <Plain text={message} />
    </p>
  );
}

// ── each column, as it is ────────────────────────────────────────────────────

function Columns({ pid, fl }: { pid: string; fl: FirstLook }) {
  const { columns, drawn } = fl;
  if (!fl.read) {
    return (
      <Section id="columns" title={TITLE.columns}>
        <p className={s.quiet}>Reading the columns…</p>
      </Section>
    );
  }
  const caption =
    drawn.length < columns.length
      ? `The first ${fmtInt(drawn.length)} of ${fmtInt(columns.length)}, in the table's order, each on every row of the table.`
      : "In the table's order, each on every row of the table.";
  return (
    <Section id="columns" title={TITLE.columns} caption={caption}>
      <ul className={s.grid}>
        {drawn.map((look) => (
          <ColumnCell key={look.name} pid={pid} look={look} table={fl.table} />
        ))}
      </ul>
    </Section>
  );
}

function ColumnCell({ pid, look, table }: { pid: string; look: ColumnLook; table: string }) {
  const d = look.draw;
  return (
    <li className={s.cell} data-testid={`look-column-${look.name}`}>
      <div className={s.cellHead}>
        <span className={s.cellName} title={look.name}>
          {look.name}
        </span>
        <span className={s.cellKind}>{look.kind}</span>
      </div>
      {look.blank ? (
        <p className={s.quiet}>
          {fmtInt(look.blank)} blank ({fmtPct(look.blank / (look.n + look.blank))})
        </p>
      ) : null}
      {d.kind === "histogram" ? (
        <ColumnBars pid={pid} column={look.name} table={table} />
      ) : d.kind === "levels" ? (
        <Levels levels={d.levels} n={look.n} more={d.more} label={look.name} />
      ) : (
        <p className={s.quiet}>{d.line}</p>
      )}
    </li>
  );
}

function ColumnBars({ pid, column, table }: { pid: string; column: string; table: string }) {
  const h = useColumnHistogram(pid, column, table);
  if (h.data) return <Bars histogram={h.data} label={column} />;
  if (h.error) return <Unread message={h.error.message} />;
  return <p className={s.quiet}>Reading its values…</p>;
}

// ── what Explore found ───────────────────────────────────────────────────────

function Explore({ fl }: { fl: FirstLook }) {
  const { looks, artifact, veil, beat } = fl.explore;
  const { setFocus } = useStageFocus();
  if (!artifact || !looks) {
    return (
      <Section id="explore" title={TITLE.explore}>
        <p className={s.quiet} data-testid="first-look-explore-line">
          {exploreLine(beat)}
        </p>
      </Section>
    );
  }
  return (
    <Section id="explore" title={TITLE.explore} caption={exploreLine(beat)}>
      <StaleVeil state={veil} label={TITLE.explore}>
        {looks.free.length ? (
          <ul className={s.grid}>
            {looks.free.map((look) => (
              <FreeCell key={look.finding.id} look={look} onDecide={() => setFocus({ kind: "finding", findingId: look.finding.id })} />
            ))}
          </ul>
        ) : (
          <p className={s.quiet}>Explore found nothing outcome-free to show.</p>
        )}
      </StaleVeil>
    </Section>
  );
}

function DecideNow({ id, onDecide }: { id: string; onDecide: () => void }) {
  return (
    <button type="button" className={`${k.linkish} ${s.act}`} onClick={onDecide} data-testid={`look-${id}-decide`}>
      Decide now
    </button>
  );
}

function FreeCell({ look, onDecide }: { look: Look; onDecide: () => void }) {
  const f = look.finding;
  const listed = listedColumns(f);
  return (
    <li className={s.cell} data-testid={`look-${f.id}`}>
      <p className={s.claim}>
        <Plain text={f.summary} />
      </p>
      {listed?.names.length ? (
        <p className={s.quiet} data-testid={`look-${f.id}-columns`}>
          {listJoin(listed.more ? [...listed.names, `${fmtInt(listed.more)} more`] : listed.names)}
        </p>
      ) : null}
      {f.groups.length ? <GroupShares groups={f.groups} column={f.columns[0] ?? ""} /> : null}
      {f.kind === "collinear" && f.detail ? (
        <ul className={s.pairs}>
          {f.detail.split("; ").map((pair) => (
            <li key={pair}>
              <Plain text={pair} />
            </li>
          ))}
        </ul>
      ) : f.detail ? (
        <p className={s.quiet}>
          <Plain text={f.detail} />
        </p>
      ) : null}
      {look.options.length ? <DecideNow id={f.id} onDecide={onDecide} /> : null}
    </li>
  );
}

// ── the outcome, behind its gates ────────────────────────────────────────────

function Outcome({ pid, fl, view }: { pid: string; fl: FirstLook; view: ProjectView }) {
  const target = fl.target!;
  const { setFocus } = useStageFocus();
  const relationships = fl.explore.looks?.relationships ?? [];
  // Relationships that wait for the same reason are said once, with the predictors they are of.
  const waiting = new Map<string, string[]>();
  for (const l of relationships) {
    if (l.state.kind !== "waits") continue;
    const line = l.state.line ?? "Too few rows with both values to draw it.";
    waiting.set(line, [...(waiting.get(line) ?? []), ...l.finding.columns]);
  }
  // Explore draws the outcome's mean in each tenth: for a yes-or-no outcome with its event named,
  // the event coded 1, so the share of rows with it (stages/modeling.py `coded_outcome`).
  const event = view.state.event;
  const share = fl.task === "binary" && !!event;
  const y = share ? `Share of rows where ${target} is ${event}` : `Mean of ${target}`;
  const more = fl.explore.looks?.more ?? 0;
  return (
    <Section id="outcome" title={outcomeTitle(target)} caption="Opening a view of the outcome records it as looked at.">
      <ul className={s.grid}>
        <DistributionCell pid={pid} fl={fl} onDecide={(id) => setFocus({ kind: "finding", findingId: id })} />
      </ul>
      {relationships.length ? (
        // Explore's: veiled while it computes again (a look recorded is one of its inputs).
        <StaleVeil state={fl.explore.veil} label="How the outcome moves with each predictor">
          <ul className={s.grid}>
            {relationships
              .filter((l) => l.state.kind !== "waits")
              .map((l) => (
                <RelationshipCell
                  key={l.finding.id}
                  pid={pid}
                  look={l}
                  y={y}
                  share={share}
                  onDecide={() => setFocus({ kind: "finding", findingId: l.finding.id })}
                />
              ))}
            {[...waiting].map(([line, columns]) => (
              <li key={line} className={s.cell} data-testid="look-relationships-waiting">
                <div className={s.cellHead}>
                  <span className={s.cellName}>Each predictor</span>
                  <span className={s.cellKind}>with the outcome</span>
                </div>
                <p className={s.quiet}>{listJoin(columns)}</p>
                <p className={s.quiet} data-testid="look-relationships-waiting-line">
                  <Plain text={line} />
                </p>
              </li>
            ))}
          </ul>
          {more ? (
            <p className={s.quiet}>
              and {more === 1 ? "1 more continuous predictor" : `${fmtInt(more)} more continuous predictors`}, not drawn
            </p>
          ) : null}
        </StaleVeil>
      ) : null}
      <ul className={s.grid}>
        <RowsCell pid={pid} fl={fl} />
      </ul>
    </Section>
  );
}

function DistributionCell({ pid, fl, onDecide }: { pid: string; fl: FirstLook; onDecide: (id: string) => void }) {
  const target = fl.target!;
  const { door, outcome } = fl;
  const decide = useDecide(pid);
  const [refused, setRefused] = useState<string | null>(null);
  const numeric = !!outcome && (outcome.dtype === "numeric" || outcome.dtype === "integer");
  const open = door?.kind === "open";
  const histogram = useOutcomeHistogram(pid, target, fl.gate, open && numeric);
  const lever = fl.explore.looks?.distribution ?? null;
  let body: ReactNode;
  if (!door || !outcome) body = <p className={s.quiet}>Reading when it opens…</p>;
  else if (door.kind === "shut") {
    body = (
      <p className={s.quiet} data-testid="look-outcome-distribution-line">
        <Plain text={door.line} />
      </p>
    );
  } else if (door.kind === "record") {
    body = (
      <>
        <button
          type="button"
          className={`${k.linkish} ${s.act}`}
          disabled={decide.isPending}
          onClick={() => {
            setRefused(null);
            decide.mutate(door.record, { onError: (err) => setRefused(err.message) });
          }}
          data-testid="open-outcome-distribution"
        >
          Open the outcome&apos;s distribution
        </button>
        <p className={s.quiet} data-testid="look-outcome-distribution-line">
          <Plain text={door.line} />
        </p>
      </>
    );
  } else {
    body = (
      <>
        <p className={s.quiet} data-testid="look-outcome-distribution-rows">
          On the {fmtInt(outcome.n)} {rowsWords(door.rows)} with a value
        </p>
        {numeric ? (
          histogram.data ? (
            <Bars histogram={histogram.data} label={target} />
          ) : histogram.error ? (
            <Unread message={histogram.error.message} />
          ) : (
            <p className={s.quiet}>Reading its values…</p>
          )
        ) : outcome.top?.length ? (
          <Levels
            levels={outcome.top.map((t) => ({ label: String(t.value), count: t.count }))}
            n={outcome.n}
            more={Math.max(0, outcome.n_unique - outcome.top.length)}
            label={target}
          />
        ) : (
          <p className={s.quiet}>{fmtInt(outcome.n_unique)} distinct values</p>
        )}
        {lever && decideNow(lever).length ? (
          // Its lever is Explore's, and records the slot as Explore computed it: veiled and inert
          // with Explore's other looks while it computes again, so an earlier answer is never undone.
          <StaleVeil state={fl.explore.veil} compact>
            <DecideNow id={lever.id} onDecide={() => onDecide(lever.id)} />
          </StaleVeil>
        ) : null}
      </>
    );
  }
  return (
    <li className={s.cell} data-testid="look-outcome-distribution">
      <div className={s.cellHead}>
        <span className={s.cellName} title={target}>
          {target}
        </span>
        <span className={s.cellKind}>its distribution</span>
      </div>
      {body}
      {refused ? <Refused message={refused} /> : null}
    </li>
  );
}

function RelationshipCell({ pid, look, y, share, onDecide }: { pid: string; look: Look; y: string; share: boolean; onDecide: () => void }) {
  const f = look.finding;
  const column = listJoin(f.columns);
  const decide = useDecide(pid);
  const [refused, setRefused] = useState<string | null>(null);
  const state = look.state;
  return (
    <li className={s.cell} data-testid={`look-${f.id}`} data-state={state.kind}>
      <div className={s.cellHead}>
        <span className={s.cellName} title={column}>
          {column}
        </span>
        <span className={s.cellKind}>with the outcome</span>
      </div>
      {state.kind === "record" ? (
        <>
          <p className={s.quiet}>
            <Plain text={f.summary} />
          </p>
          <button
            type="button"
            className={`${k.linkish} ${s.act}`}
            disabled={decide.isPending}
            aria-label={`Open how the outcome moves with ${column}`}
            onClick={() => {
              setRefused(null);
              decide.mutate(state.record, { onError: (err) => setRefused(err.message) });
            }}
            data-testid={`look-${f.id}-open`}
          >
            Open
          </button>
        </>
      ) : (
        <>
          <Curve points={f.points} column={column} y={y} share={share} />
          {look.options.length ? <DecideNow id={f.id} onDecide={onDecide} /> : null}
        </>
      )}
      {refused ? <Refused message={refused} /> : null}
    </li>
  );
}

function RowsCell({ pid, fl }: { pid: string; fl: FirstLook }) {
  const target = fl.target!;
  const { rows, window } = fl;
  const decide = useDecide(pid);
  const [refused, setRefused] = useState<string | null>(null);
  return (
    <li className={`${s.cell} ${s.wide}`} data-testid="look-outcome-rows">
      <div className={s.cellHead}>
        <span className={s.cellName}>The first {fmtInt(ROWS_SHOWN)} rows</span>
        <span className={s.cellKind}>the outcome beside the other columns</span>
      </div>
      {!rows || !window ? (
        <p className={s.quiet}>Reading the rows…</p>
      ) : (
        <>
          {rows.kind === "record" ? (
            <button
              type="button"
              className={`${k.linkish} ${s.act}`}
              disabled={decide.isPending}
              onClick={() => {
                setRefused(null);
                decide.mutate(rows.record, { onError: (err) => setRefused(err.message) });
              }}
              data-testid="open-outcome-rows"
            >
              Open the outcome beside the other columns
            </button>
          ) : null}
          {(rows.kind === "open" ? rows.notes : [rows.line]).map((line, i) => (
            <p key={i} className={s.quiet} data-testid="look-outcome-rows-line">
              <Plain text={line} />
            </p>
          ))}
          <div className={s.rows}>
            <table>
              <thead>
                <tr>
                  {window.columns.map((c) => (
                    <th key={c} scope="col" data-outcome={c === target || undefined}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {window.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((v, j) => (
                      <td key={j} data-outcome={window.columns[j] === target || undefined}>
                        {fmtValue(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {refused ? <Refused message={refused} /> : null}
    </li>
  );
}
