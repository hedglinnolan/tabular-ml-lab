/**
 * /lab/m3 — the M3 reference journeys, replayed from the real server's captures (dev:mock only;
 * src/mocks/m3.ts). Each journey opens as a project at any of its snapshots: the Record, the
 * banner and the stage read exactly what the server served there, so a surface can be built on
 * real shapes for every stage and endpoint. A step's answer moves the replay on when it is the
 * answer the capture recorded; anything else is refused with that answer as its exit.
 */
import { useEffect, useState } from "react";
import { Header } from "../components/Header";
import { Link, projectPath } from "../router";
import { M3_JOURNEYS, loadJourney, m3Pid, type M3Fixture } from "../mocks/m3";
import s from "./LabScreen.module.css";

export function M3LabScreen() {
  const [open, setOpen] = useState<string | null>(null);
  const [fixture, setFixture] = useState<M3Fixture | null>(null);
  useEffect(() => {
    if (!open) return;
    let live = true;
    void loadJourney(open).then((f) => live && setFixture(f));
    return () => {
      live = false;
    };
  }, [open]);
  return (
    <>
      <Header />
      <main className={s.main} data-testid="m3-lab">
        <header>
          <h1 className={s.h1}>The M3 journeys</h1>
          <p className={s.lede}>
            Reference journeys the real server ran, captured step by step. Open one at any step;
            the replay serves the views, stage results and refusals the server gave there.
          </p>
        </header>
        <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 10 }}>
          {M3_JOURNEYS.map((name) => (
            <li key={name}>
              <button
                type="button"
                onClick={() => setOpen(open === name ? null : name)}
                aria-expanded={open === name}
                data-testid={`m3-journey-${name}`}
                style={{ font: "inherit", fontWeight: 650, background: "none", border: 0, padding: 0, color: "var(--accent-ink)" }}
              >
                {name}
              </button>{" "}
              <Link href={projectPath(m3Pid(name))}>open at the start</Link>
              {open === name && fixture?.meta.journey === name ? (
                <div style={{ marginTop: 6, fontSize: 13 }}>
                  <p style={{ margin: "0 0 4px", fontFamily: "var(--serif)", fontSize: 14 }}>
                    {fixture.meta.label} <code className="v">{fixture.meta.source}</code>
                  </p>
                  <ol start={0} style={{ margin: 0, paddingLeft: 22, columns: 2 }}>
                    {fixture.snapshots.map((snap, i) => (
                      <li key={i}>
                        <Link href={projectPath(m3Pid(name, i))} data-testid={`m3-step-${name}-${i}`}>
                          {snap.open ? `${snap.open.replace(/_/g, " ")} open` : (snap.first ?? "the end")}
                        </Link>
                        {snap.after ? <span style={{ color: "var(--muted)" }}> after {snap.after}</span> : null}
                      </li>
                    ))}
                  </ol>
                  {fixture.meta.notes.length ? (
                    <p style={{ color: "var(--muted)", margin: "6px 0 0" }}>{fixture.meta.notes.join(" · ")}</p>
                  ) : null}
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      </main>
    </>
  );
}
