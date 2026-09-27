export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Evenly spaced indices into [0, n): a stable subsample for a sparkline. */
export function every(n: number, take: number): number[] {
  if (take >= n) return Array.from({ length: n }, (_, i) => i);
  const step = n / take;
  return Array.from({ length: take }, (_, i) => Math.floor(i * step));
}
