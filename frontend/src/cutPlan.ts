export type CutRange = { start: number; end: number; ids: string[] };
export type CutOrder = "source" | "selection";
/**
 * Turns chosen candidates into timeline ranges. Source order sorts by time and
 * merges ranges that overlap or touch (within 50 ms) so footage is not repeated;
 * selection order keeps every pick as its own clip in the order chosen and only
 * counts the overlaps so the UI can warn about repeated footage.
 */
export function planCut(
  chosen: { id: string; start: number; end: number }[],
  order: CutOrder,
) {
  const ordered =
    order === "source"
      ? [...chosen].sort((a, b) => a.start - b.start || a.end - b.end)
      : chosen;
  const ranges: CutRange[] = [];
  let overlaps = 0;
  for (const c of ordered) {
    const last = ranges[ranges.length - 1];
    if (order === "source" && last && c.start <= last.end + 0.05) {
      last.end = Math.max(last.end, c.end);
      last.ids.push(c.id);
      overlaps++;
      continue;
    }
    if (
      order === "selection" &&
      ranges.some((r) => c.start < r.end && c.end > r.start)
    )
      overlaps++;
    ranges.push({ start: c.start, end: c.end, ids: [c.id] });
  }
  return {
    ranges,
    overlaps,
    duration: ranges.reduce((total, r) => total + (r.end - r.start), 0),
  };
}
