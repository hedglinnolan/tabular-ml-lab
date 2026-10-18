/**
 * The product kit's tokens are the foundation's (docs/turbotab-next/calm/tokens.css) value for value,
 * light and dark, only scoped to `[data-kit]` so the classic screen keeps its own palette.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const here = resolve(__dirname);
const ours = readFileSync(resolve(here, "tokens.css"), "utf8");
const theirs = readFileSync(resolve(here, "../../../../docs/turbotab-next/calm/tokens.css"), "utf8");

/** Each block's declarations (`--name: value`), in order, by the block's selector. */
function blocks(css: string): Map<string, string[]> {
  const out = new Map<string, string[]>();
  const body = css.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const m of body.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const selector = m[1]!.replace(/@media[^{]*$/, "").replace(/\s+/g, " ").trim();
    const decls = m[2]!
      .split(";")
      .map((d) => d.trim())
      .filter(Boolean);
    out.set(selector, decls);
  }
  return out;
}

describe("the kit's tokens", () => {
  const a = blocks(theirs);
  const b = blocks(ours);

  it("holds the foundation's light, system-dark and chosen-dark blocks under [data-kit]", () => {
    expect(b.get("[data-kit]")).toEqual(a.get(":root"));
    expect(b.get(':root:not([data-theme="light"]) [data-kit]')).toEqual(
      a.get(':root:not([data-theme="light"])'),
    );
    expect(b.get(':root[data-theme="dark"] [data-kit]')).toEqual(a.get(':root[data-theme="dark"]'));
  });

  it("declares nothing on :root, so the classic screen's tokens stand", () => {
    expect([...b.keys()].filter((s) => /^:root\s*$|^:root[^ ]*$/.test(s))).toEqual([]);
    expect(a.get(":root")?.length).toBeGreaterThan(20);
  });
});

/**
 * One color for the primary action (FOUNDATION §1, §4; the walkthrough of 2026-10-17, minor 22):
 * inside the shell a card built before the kit records in the kit's action color, which today's
 * palette (src/styles/tokens.css) carries as the foundation's --action family, value for value, in
 * the light, system-dark and chosen-dark blocks.
 */
describe("today's palette's action color", () => {
  const today = blocks(readFileSync(resolve(here, "../styles/tokens.css"), "utf8"));
  const foundation = blocks(theirs);
  const action = (decls: string[] | undefined) => (decls ?? []).filter((d) => /^--action(-hover|-ink)?:/.test(d));

  it("is the foundation's, value for value, in each block", () => {
    for (const selector of [":root", ':root:not([data-theme="light"])', ':root[data-theme="dark"]']) {
      expect(action(foundation.get(selector)), selector).toHaveLength(3);
      expect(action(today.get(selector)), selector).toEqual(action(foundation.get(selector)));
    }
  });
});
