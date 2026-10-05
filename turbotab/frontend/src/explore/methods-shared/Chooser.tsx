/**
 * /lab/methods — the chooser for the three living-methods prototypes (BLUEPRINT §11.4). Each card
 * says the prototype's idea and how a newcomer and an expert move through it, and opens it. All
 * three show the one shared scenario (SCENARIO.md), so what differs between them is the design.
 *
 * Under dev:mock the cards open the /lab routes; the static build (protos-main.tsx) passes hash
 * links instead.
 */
import type { ReactNode } from "react";
import { Header } from "../../components/Header";
import { Link } from "../../router";
import s from "./chooser.module.css";

export type ProtoId = "document" | "questlog" | "map";

interface Proto {
  id: ProtoId;
  letter: string;
  title: string;
  idea: string;
  newcomer: string;
  expert: string;
}

export const PROTOS: Proto[] = [
  {
    id: "document",
    letter: "A",
    title: "The paper",
    idea: "The methods section is the screen: a typeset paper whose open slots and changeable phrases are filled in place, with the canvas beside it showing the focused one on the data.",
    newcomer: "presses Next and fills one open slot at a time, the rest of the paper dimmed to a map.",
    expert: "scans the paper and clicks any slot or phrase to answer or change it.",
  },
  {
    id: "questlog",
    letter: "B",
    title: "The quest log",
    idea: "The methods section as a list of objectives by reporting-guideline section, one focused card at a time, the canvas beside it and the whole document one press away.",
    newcomer: "follows the rail: each card says the sentence it will write and offers the engine's guess.",
    expert: "jumps to any objective in the rail, or opens the whole document.",
  },
  {
    id: "map",
    letter: "C",
    title: "The map",
    idea: "The analysis drawn as a map from the raw table to the estimate: every decision is a node where it acts, and the methods text reads alongside.",
    newcomer: "presses “Next asked” and walks the lit nodes in order.",
    expert: "clicks any node; hovering an option plays it on the map and the canvas.",
  },
];

export const DEV_HREFS: Record<ProtoId, string> = {
  document: "/lab/methods-document",
  questlog: "/lab/methods-questlog",
  map: "/lab/methods-map",
};

function Open({ href, hash, children }: { href: string; hash: boolean; children: ReactNode }) {
  // A hash link is the browser's own navigation (the static build); a path goes through the router.
  return hash ? (
    <a href={href} className={s.open}>
      {children}
    </a>
  ) : (
    <Link href={href} className={s.open}>
      {children}
    </Link>
  );
}

export function MethodsChooser({ hrefs = DEV_HREFS }: { hrefs?: Record<ProtoId, string> }) {
  const hash = hrefs.document.startsWith("#");
  return (
    <>
      <Header />
      <main className={s.main} data-testid="methods-chooser">
        <header className={s.head}>
          <p className={s.kicker}>Design prototypes · BLUEPRINT §11.4</p>
          <h1 className={s.h1}>The living methods section, three ways</h1>
          <p className={s.lede}>
            The Record drafts the methods section as you answer: written-in phrases you can change,
            slots that ask with a guess, and decisions that change no number left to the export.
            Three designs of that one idea follow. Each can be walked from the first draft to the
            locked Table 2 and “Which of my decisions mattered?”, and each has a Reset.
          </p>
        </header>

        <ul className={s.cards} aria-label="The prototypes">
          {PROTOS.map((p) => (
            <li key={p.id} className={s.card} data-testid="proto-card">
              <p className={s.letter} aria-hidden="true">
                {p.letter}
              </p>
              <h2 className={s.title}>
                <span className={s.sr}>{p.letter}: </span>
                {p.title}
              </h2>
              <p className={s.idea}>{p.idea}</p>
              <dl className={s.moves}>
                <dt>A newcomer</dt>
                <dd>{p.newcomer}</dd>
                <dt>An expert</dt>
                <dd>{p.expert}</dd>
              </dl>
              <Open href={hrefs[p.id]} hash={hash}>
                Open {p.title.replace(/^The /, "the ")}
              </Open>
            </li>
          ))}
        </ul>

        <section className={s.scenario} aria-labelledby="scenario-h">
          <h2 id="scenario-h" className={s.h2}>
            One scenario in all three
          </h2>
          <p>
            The NHANES dietary export, <code className="v">21,849</code> rows: the total effect of{" "}
            <code className="v">sugar</code> on fasting <code className="v">glucose</code> as a
            substitution for other energy sources at fixed total energy (inference, STROBE-nut).
            Every row is kept, with Willett 2013’s and NHS/HPFS’s sex-specific screens as secondary
            analyses; the adjustment set takes the pack’s guesses, with blood pressure, lipids and
            medications as mediators and the survey cycle as a confounder; the engine’s default
            energy model; Model 1 adjusted for age, gender and energy. Model 2, the primary:{" "}
            <code className="v">−0.0199</code> (95% interval <code className="v">−0.0327</code> to{" "}
            <code className="v">−0.00718</code>).
          </p>
          <p className={s.note}>
            All numbers are real engine output on NHANES: every guess, methods sentence and
            estimate was captured from the TurboTab server driven through this same scenario. The
            prototypes replay that capture with no server, so nothing you click is recorded.
          </p>
        </section>
      </main>
    </>
  );
}
