import { OVERLAP_ENGINE, OVERLAP_ONE_EACH, OVERLAP_ONE_GROUP, OVERLAP_RECORDED, OVERLAP_UNTRIMMED } from "../fixtures";
import type { LabEntry, LabSample } from "../lab/entry";
import { OverlapView } from "./OverlapView";
import { OVERLAP_PURPOSE } from "./types";

const ENGINE = "The engine's OverlapView and the plan's trim, from the captured causal journey (m3-causal.json)";

const samples: LabSample[] = [
  { label: "A trim pointed at", source: ENGINE, render: () => <OverlapView input={OVERLAP_ENGINE} title="Who has a comparable row in the other group" /> },
  { label: "The trim recorded", source: ENGINE, render: () => <OverlapView input={OVERLAP_RECORDED} title="Who has a comparable row in the other group" /> },
  { label: "At rest, nothing trimmed", source: ENGINE, render: () => <OverlapView input={OVERLAP_UNTRIMMED} /> },
  { label: "One row in each group", source: "Hand-made: the degenerate case", render: () => <OverlapView input={OVERLAP_ONE_EACH} /> },
  { label: "One group has no rows", source: "Hand-made: the empty case", render: () => <OverlapView input={OVERLAP_ONE_GROUP} /> },
];

export const entries: LabEntry[] = [{ kind: "overlap", order: 7, purpose: OVERLAP_PURPOSE, samples }];
