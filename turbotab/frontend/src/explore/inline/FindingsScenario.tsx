/**
 * S3 — the 13 real NHANES findings under the doctrine (§11.7): three pushed, ranked; the rest
 * counted and typed, same-kind findings paged in one card. Each finding is a one-line claim built
 * from the file's own numbers, its evidence drawn at sparkline scale (the same pictures the
 * questions use), and its lever. Pointing at a finding marks the columns it is about in the panel.
 */
import { useState } from "react";
import { Prose } from "../../components/Prose";
import {
  ENERGY,
  EXCLUSIONS,
  FINDING_SET,
  FIXTURE,
  type FindingCard,
  type FindingGroup,
  type Route,
} from "./fixture";
import { fmtInt } from "./format";
import { Panel, RoleSection, RowsSection } from "./Pipeline";
import { cx, every } from "./util";
import { Histogram } from "./views/Histogram";
import { Scatter } from "./views/Scatter";
import { LevelBars, ShareBar } from "./views/Small";
import s from "./inline.module.css";

const SPARK = every(ENERGY.xs.length, 220);
const PIC_W = 128;
const PIC_H = 50;

const ROLE_ORDER: [string, string][] = [
  ["outcome", "outcome"],
  ["identifier", "identifier"],
  ["time", "time"],
  ["energy", "energy"],
  ["exposure", "exposures"],
  ["covariate", "covariates"],
  ["flag", "flags"],
  ["excluded", "not used"],
];

function roleGroups() {
  const roles = FINDING_SET.roles;
  const groups = ROLE_ORDER.map(([role, label]) => ({
    role: label,
    columns:
      role === "outcome"
        ? [FINDING_SET.outcome]
        : FINDING_SET.columns.filter((c) => roles[c] === role),
  }));
  return groups.filter((g) => g.columns.length);
}
const GROUPS = roleGroups();

export function FindingsScenario({ onRoute }: { onRoute: (r: Route) => void }) {
  const [hover, setHover] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const { pushed, groups, total } = FINDING_SET;
  const rest = total - pushed.length;
  const steps = FIXTURE.scenario_a.setup.cohort_steps.slice(0, 2);

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <section className={s.findings} aria-labelledby="findings-h">
          <div className={s.qKicker}>Findings · {total}</div>
          <h2 id="findings-h" className={s.qTitle}>
            Three things to settle before modeling
          </h2>
          <p className={s.qWhy}>
            Read from all <span className="v">{fmtInt(FINDING_SET.nRows)}</span> rows under the
            dietary lens, with <span className="v">{FINDING_SET.outcome}</span> as the outcome.
          </p>
          <ol className={s.pushed}>
            {pushed.map((f, i) => (
              <li key={f.id}>
                <Finding f={f} rank={i + 1} onRoute={onRoute} onHover={(cols) => setHover(cols)} />
              </li>
            ))}
          </ol>
          <div className={s.rest}>
            <span className={s.restCount}>
              {rest} more, {groups.length} kinds
            </span>
            {groups.map((g) => (
              <button
                key={g.key}
                type="button"
                className={cx(s.restChip, open === g.key && s.restChipOn)}
                aria-expanded={open === g.key}
                onClick={() => setOpen(open === g.key ? null : g.key)}
              >
                <span className="num">{g.size}</span> {g.type}
              </button>
            ))}
          </div>
          {groups.map((g) =>
            open === g.key ? (
              <Paged key={g.key} g={g} onRoute={onRoute} onHover={(cols) => setHover(cols)} />
            ) : null,
          )}
        </section>
      </div>
      <Panel status="idle">
        <RowsSection
          status="idle"
          steps={steps.map((st) => ({
            key: st.key,
            label: st.key === "loaded" ? "Rows loaded" : `\`${FINDING_SET.outcome}\` measured`,
            n: st.n,
          }))}
          split={null}
        />
        <RoleSection groups={GROUPS} highlight={new Set(hover)} />
      </Panel>
    </div>
  );
}

function Picture({ f }: { f: FindingCard }) {
  const p = f.picture;
  switch (p.kind) {
    case "scatter":
      return (
        <Scatter
          xs={ENERGY.xs}
          state={ENERGY.base}
          width={PIC_W}
          height={PIC_H}
          variant="spark"
          sample={SPARK}
          title={`${ENERGY.focus} against ${ENERGY.energyColumn}`}
        />
      );
    case "tails": {
      const all = EXCLUSIONS.options[0]!.dist.before;
      const cut = EXCLUSIONS.options.find((o) => o.byLevel[0]?.level === "all");
      return (
        <Histogram
          hist={all}
          kept={cut?.dist.after ?? null}
          marks={cut?.dist.marks ?? []}
          width={PIC_W}
          height={PIC_H}
          variant="spark"
          xMax={6300}
          title="kcal, with the implausible tails"
        />
      );
    }
    case "levels":
      return (
        <LevelBars
          levels={p.levels}
          width={PIC_W}
          height={PIC_H}
          title={p.levels.map((l) => `${l.label} ${fmtInt(l.n)}`).join(", ")}
        />
      );
    case "share":
      return (
        <div className={s.sharePic}>
          <ShareBar
            parts={p.parts}
            width={PIC_W}
            height={12}
            title={p.parts.map((x) => `${x.label} ${fmtInt(x.n)}`).join(", ")}
          />
          <span className={s.shareKey}>
            {p.parts.map((x) => (
              <span key={x.label} data-tone={x.tone}>
                {x.label}
              </span>
            ))}
          </span>
        </div>
      );
  }
}

function Lever({ f, onRoute }: { f: FindingCard; onRoute: (r: Route) => void }) {
  const [said, setSaid] = useState(false);
  const live = f.route !== "roles";
  return (
    <span className={s.leverCell}>
      <button
        type="button"
        className={s.leverBtn}
        onClick={() => (live ? onRoute(f.route) : setSaid(true))}
      >
        {f.lever} <span aria-hidden="true">→</span>
      </button>
      {said ? (
        <span className={s.leverSaid} role="status">
          opens Roles, not in this prototype
        </span>
      ) : null}
    </span>
  );
}

function Finding({
  f,
  rank,
  onRoute,
  onHover,
}: {
  f: FindingCard;
  rank: number;
  onRoute: (r: Route) => void;
  onHover: (cols: string[]) => void;
}) {
  return (
    <article
      className={s.finding}
      data-rank={rank}
      onPointerEnter={() => onHover(f.columns)}
      onPointerLeave={() => onHover([])}
      onFocus={() => onHover(f.columns)}
      onBlur={() => onHover([])}
    >
      <div className={s.findingPic}>
        <Picture f={f} />
      </div>
      <p className={s.claim}>
        <Prose text={f.claim} />
      </p>
      <div className={s.findingSide}>
        {f.evidence ? (
          <span className={s.badge} title={f.evidence.source}>
            {f.evidence.status}
          </span>
        ) : null}
        <Lever f={f} onRoute={onRoute} />
      </div>
    </article>
  );
}

function Paged({
  g,
  onRoute,
  onHover,
}: {
  g: FindingGroup;
  onRoute: (r: Route) => void;
  onHover: (cols: string[]) => void;
}) {
  const [page, setPage] = useState(0);
  const f = g.pages[page]!;
  const n = g.pages.length;
  return (
    <section className={s.paged} aria-label={g.title}>
      <div className={s.pagedHead}>
        <span className={s.pagedTitle}>{g.title}</span>
        <span className={s.pager}>
          <button type="button" aria-label="Previous" onClick={() => setPage((page - 1 + n) % n)}>
            ‹
          </button>
          <span className="num">
            {page + 1} of {n}
          </span>
          <button type="button" aria-label="Next" onClick={() => setPage((page + 1) % n)}>
            ›
          </button>
        </span>
      </div>
      <Finding f={f} rank={0} onRoute={onRoute} onHover={onHover} />
    </section>
  );
}
