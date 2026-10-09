import { render } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { parseRoute } from "../../../router";
import { ANSWER_WORDS, QUESTIONS } from "../../stage/purposes";
import { words } from "../../stage/format";
import { scopedThemes } from "./themes";
import { anchorOf, ENTRIES, ViewsLab } from "./ViewsLab";

/** FOUNDATION §5 rule 9's view kinds, in its order. */
const RULE_9 = ["table", "forest", "curve", "calibration", "decision curve", "specification curve", "overlap", "embedding", "matrix", "page"];

describe("/lab/views", () => {
  it("exists only in the lab build", () => {
    expect(parseRoute("/lab/views", true)).toEqual({ name: "views-lab" });
    expect(parseRoute("/lab/views", false)).toEqual({ name: "missing", path: "/lab/views" });
  });

  it("scopes the one token file's light and dark blocks to a frame each", () => {
    // Read from disk: vitest serves CSS imports as empty text.
    const tokens = readFileSync("src/explore/calm-kit/tokens.css", "utf8");
    const css = scopedThemes(tokens);
    expect(css).toMatch(/\[data-lab-theme="light"\] \{[^}]*--canvas: #ECF1F6/);
    expect(css).toMatch(/\[data-lab-theme="dark"\] \{[^}]*--canvas: #080B11/);
  });

  it("shows every rule-9 view kind once, in rule 9's order", () => {
    expect(ENTRIES.map((e) => e.kind)).toEqual(RULE_9);
  });

  it("gives every view kind a purpose entry in at most the registry's words", () => {
    for (const e of ENTRIES) {
      expect(QUESTIONS[e.purpose.question], e.kind).toBeDefined();
      expect(words(e.purpose.answer), e.kind).toBeLessThanOrEqual(ANSWER_WORDS);
    }
  });

  it("renders every sample of every kind in the light and the dark theme", () => {
    const { container } = render(<ViewsLab />);
    for (const e of ENTRIES) {
      const section = container.querySelector(`section#${anchorOf(e.kind)}`)!;
      const frames = section.querySelectorAll("[data-lab-theme]");
      expect(frames).toHaveLength(e.samples.length * 2);
      for (const f of frames) expect(f.querySelector("[data-exhibit-view], [data-view]"), `${e.kind} in ${f.getAttribute("data-lab-theme")}`).not.toBeNull();
    }
  });
});
