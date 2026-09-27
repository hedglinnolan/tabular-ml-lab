/**
 * Keyed scenes — the one mechanism every scrubbed view uses.
 *
 * A view describes each state it can show ("your data now", "with residual", …) as a Scene: a map
 * from a stable key to a few numbers (position, width, opacity). Two scenes blend key by key:
 *
 *   - a key present in both morphs (its numbers interpolate)  -> identity survives the change;
 *   - a key present on one side only fades (and may slide)    -> identity does not survive.
 *
 * So the identity rule of DESIGN_LANGUAGE §05.2 ("animate only where a data dimension survives")
 * is decided by how a view names its keys, not by per-view animation code: a row keeps its key
 * across states and moves; a relabeled axis tick gets a new key and crossfades.
 *
 * Blends compose (a blend is itself a scene), so an interrupted transition starts from exactly
 * what is on screen.
 */

export type Props = Record<string, number>;

export interface Scene {
  items: Map<string, Props>;
  /** Bulk numeric channels (e.g. 800 point positions), blended element-wise. */
  arrays: Map<string, Float32Array>;
}

export function emptyScene(): Scene {
  return { items: new Map(), arrays: new Map() };
}

export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;

function lerpProps(a: Props, b: Props, p: number): Props {
  const out: Props = {};
  for (const k in a) out[k] = lerp(a[k]!, b[k] ?? a[k]!, p);
  for (const k in b) if (!(k in a)) out[k] = b[k]!;
  return out;
}

/**
 * Leaving: fade out; when the item asks, drift up by `slide` px, and (for `grow`) give its width
 * back as it goes, so a suffix being replaced shrinks while the new one grows.
 */
function exitProps(a: Props, p: number): Props {
  const out = { ...a };
  // `swap`: labels hand over rather than overlap — out in the first half, in during the second.
  out.o = (a.o ?? 1) * (a.swap ? Math.max(0, 1 - 2 * p) : 1 - p);
  if (a.slide) out.dy = (a.dy ?? 0) - a.slide * p;
  if (a.grow && a.w !== undefined) out.w = a.w * (1 - p);
  return out;
}

/** Arriving: fade in; when asked, rise into place from `slide` px below and grow to its width. */
function enterProps(b: Props, p: number): Props {
  const out = { ...b };
  out.o = (b.o ?? 1) * (b.swap ? Math.max(0, 2 * p - 1) : p);
  if (b.slide) out.dy = (b.dy ?? 0) + b.slide * (1 - p);
  if (b.grow && b.w !== undefined) out.w = b.w * p;
  return out;
}

export function blend(a: Scene, b: Scene, p: number): Scene {
  if (p <= 0) return a;
  if (p >= 1) return b;
  const items = new Map<string, Props>();
  for (const [k, pa] of a.items) {
    const pb = b.items.get(k);
    items.set(k, pb ? lerpProps(pa, pb, p) : exitProps(pa, p));
  }
  for (const [k, pb] of b.items) if (!a.items.has(k)) items.set(k, enterProps(pb, p));
  const arrays = new Map<string, Float32Array>();
  for (const [k, va] of a.arrays) {
    const vb = b.arrays.get(k);
    if (!vb || vb.length !== va.length) {
      arrays.set(k, p < 0.5 ? va : (vb ?? va));
      continue;
    }
    const out = new Float32Array(va.length);
    for (let i = 0; i < va.length; i++) out[i] = va[i]! + (vb[i]! - va[i]!) * p;
    arrays.set(k, out);
  }
  for (const [k, vb] of b.arrays) if (!a.arrays.has(k)) arrays.set(k, vb);
  return { items, arrays };
}

/** Map a 0..1 progress into a sub-window, so a cause can lead and its effect follow. */
export function windowed(p: number, [a, b]: readonly [number, number]): number {
  if (p <= a) return 0;
  if (p >= b) return 1;
  return (p - a) / (b - a);
}

/** Every key any of the scenes uses, in first-seen order — what a view must render. */
export function unionKeys(scenes: Scene[]): string[] {
  const seen = new Set<string>();
  for (const s of scenes) for (const k of s.items.keys()) seen.add(k);
  return [...seen];
}

/** A view's scene per state, computed once per state (the states are fixed data). */
export function cached(of: (state: string) => Scene): (state: string) => Scene {
  const memo = new Map<string, Scene>();
  return (state) => {
    let sc = memo.get(state);
    if (!sc) {
      sc = of(state);
      memo.set(state, sc);
    }
    return sc;
  };
}

/** A small builder so views read declaratively. */
export class SceneBuilder {
  readonly scene: Scene = emptyScene();
  set(key: string, props: Props): this {
    this.scene.items.set(key, props);
    return this;
  }
  array(key: string, values: Float32Array): this {
    this.scene.arrays.set(key, values);
    return this;
  }
}
