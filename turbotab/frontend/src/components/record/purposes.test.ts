/**
 * The Record's purpose gate (BLUEPRINT §11.2): a component the Record exports without a declared
 * purpose fails here, and so does an entry for a component that is no longer exported.
 */
import { ANSWER_WORDS, QUESTIONS, RECORD_PURPOSES, isStructural } from "./purposes";

const SOURCES = import.meta.glob("./**/*.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function exported(): Map<string, string> {
  const out = new Map<string, string>();
  for (const [path, text] of Object.entries(SOURCES)) {
    if (path.endsWith(".test.tsx")) continue;
    for (const m of text.matchAll(/export function ([A-Z][A-Za-z0-9]*)\s*[(<]/g))
      out.set(m[1]!, path);
  }
  return out;
}

describe("every Record component declares the question it answers", () => {
  it("covers every exported component, and lists none that is gone", () => {
    const found = exported();
    expect(found.size).toBeGreaterThan(30); // a vacuous scan would pass anything
    for (const [name, path] of found)
      expect(RECORD_PURPOSES[name], `<${name}> in ${path} has no purpose`).toBeDefined();
    for (const name of Object.keys(RECORD_PURPOSES))
      expect(found.has(name), `${name} is registered but not exported`).toBe(true);
  });

  it("says each purpose in one short sentence, naming one of the five questions", () => {
    for (const [name, p] of Object.entries(RECORD_PURPOSES)) {
      const text = isStructural(p) ? p.structural : p.answer;
      expect(text.split(/\s+/).length, name).toBeLessThanOrEqual(ANSWER_WORDS);
      if (!isStructural(p)) expect(QUESTIONS[p.question], name).toBeDefined();
    }
  });
});
