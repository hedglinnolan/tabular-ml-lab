/**
 * The map (calm.html#/map): its walk from the first draft to the locked Table 2 and what
 * mattered. A placeholder until the structure replaces src/explore/calm-map/Screen.tsx: until
 * then it is the kit's reference card walk. No request reaches /api/ on the way.
 */
import { expect } from "@playwright/test";
import { cardToMattered, cardWalk, watchApi } from "./scenario";
import type { Walker } from "./walker";

const seen = new WeakMap<object, string[]>();

export const walker: Walker = {
  name: "The map",
  path: "/calm.html#/map",
  toTable2: async (page) => {
    seen.set(page, watchApi(page));
    await cardWalk(page);
  },
  toMattered: async (page) => {
    await cardToMattered(page);
    expect(seen.get(page) ?? [], "requests to /api/ during the walk").toEqual([]);
  },
};
