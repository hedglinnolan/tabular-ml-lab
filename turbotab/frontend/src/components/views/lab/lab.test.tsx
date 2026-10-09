import { render } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { parseRoute } from "../../../router";
import { ANSWER_WORDS } from "../../stage/purposes";
import { LAB_ENTRIES, LabViews, scopedTokens } from "./LabViews";

describe("/lab/views", () => {
  it("exists only in the lab build", () => {
    expect(parseRoute("/lab/views", true)).toEqual({ name: "views-lab" });
    expect(parseRoute("/lab/views", false)).toEqual({ name: "missing", path: "/lab/views" });
  });

  it("renders every entry in both themes", () => {
    const { container } = render(<LabViews />);
    const kinds = new Set(LAB_ENTRIES.map((e) => e.kind));
    for (const k of ["overlap", "embedding", "matrix"]) expect(kinds.has(k)).toBe(true);
    for (const e of LAB_ENTRIES) {
      const section = container.querySelector(`#${e.id}`)!;
      expect(section.querySelectorAll("[data-lab-theme]")).toHaveLength(2);
      expect(section.querySelectorAll(`[data-exhibit-view="${e.kind}"]`)).toHaveLength(2);
    }
  });

  it("scopes the light and dark tokens from the one tokens file", () => {
    // the test runner stubs stylesheets, so the file is read as the page's ?raw import reads it
    const css = scopedTokens(readFileSync(`${process.cwd()}/src/explore/calm-kit/tokens.css`, "utf8"));
    expect(css).toMatch(/\[data-lab-theme="light"\]\{[^}]*--canvas: #ECF1F6/);
    expect(css).toMatch(/\[data-lab-theme="dark"\]\{[^}]*--canvas: #080B11/);
  });

  it("gives each view kind one short purpose", () => {
    for (const e of LAB_ENTRIES) expect(e.purpose.answer.split(/\s+/).length, e.kind).toBeLessThanOrEqual(ANSWER_WORDS);
  });
});
