/**
 * The stage's M2 contract (M2_CONTRACT §3, §6, §11): thin aliases over the generated types, as
 * m1-stage-types.ts does for M1. Nothing here is hand-shaped; regenerate after a server change.
 */
import type { components } from "./generated";

type S = components["schemas"];

/** A coach note (≤ 12 words) and what it points at: a column, a range, points, a row-flow step. */
export type CoachNote = S["CoachNote"];
export type CoachAnchor = S["CoachAnchor"];
export type CoachAnchorKind = CoachAnchor["kind"];

/** The seal drawn one cell per row: the split question's preview (a row flow's `seal`). */
export type SealCells = S["SealCells"];
export type SealBasisState = SealCells["state"];
export type SealBasis = S["SealBasis"];
export type Chronology = S["Chronology"];
export type SealPlan = S["SealPlan"];
export type HoldoutOption = S["HoldoutOption"];

/** A storyboard frame's row, with the reshape's row map: its unit and the source rows it stands for. */
export type FrameRow = S["FrameRow"];
