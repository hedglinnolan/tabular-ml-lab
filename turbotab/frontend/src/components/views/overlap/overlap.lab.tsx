import { OVERLAP_ENGINE, OVERLAP_ONE_EACH, OVERLAP_ONE_GROUP, OVERLAP_RECORDED, OVERLAP_UNTRIMMED } from "../fixtures";
import type { LabEntry } from "../lab/entry";
import { OverlapView } from "./OverlapView";
import { OVERLAP_PURPOSE } from "./types";

const ENGINE = "The engine's OverlapView and the plan's trim, from the captured causal journey (m3-causal.json)";
const base = { kind: "overlap", purpose: OVERLAP_PURPOSE };

export const entries: LabEntry[] = [
  { ...base, id: "overlap-choice", title: "Overlap · a trim pointed at", source: ENGINE, render: () => <OverlapView input={OVERLAP_ENGINE} title="Who has a comparable row in the other group" /> },
  { ...base, id: "overlap-recorded", title: "Overlap · the trim recorded", source: ENGINE, render: () => <OverlapView input={OVERLAP_RECORDED} title="Who has a comparable row in the other group" /> },
  { ...base, id: "overlap-rest", title: "Overlap · at rest, nothing trimmed", source: ENGINE, render: () => <OverlapView input={OVERLAP_UNTRIMMED} /> },
  { ...base, id: "overlap-one", title: "Overlap · one row in each group", source: "Hand-made: the degenerate case", render: () => <OverlapView input={OVERLAP_ONE_EACH} /> },
  { ...base, id: "overlap-empty", title: "Overlap · one group has no rows", source: "Hand-made: the empty case", render: () => <OverlapView input={OVERLAP_ONE_GROUP} /> },
];
