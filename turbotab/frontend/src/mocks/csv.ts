/** A small delimited-text reader so a dropped file becomes a real mock table. */
import type { Dtype, Scalar } from "../api/schema";
import type { MockColumn, MockDataset } from "./datasets";

function splitLine(line: string, sep: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === sep) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

const DATE = /^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$/;
const BOOL = new Set(["true", "false", "yes", "no"]);

function infer(raw: string[]): { dtype: Dtype; physical: string; values: Scalar[] } {
  const present = raw.filter((v) => v !== "" && v.toLowerCase() !== "na");
  const values: Scalar[] = raw.map((v) => (v === "" || v.toLowerCase() === "na" ? null : v));
  if (present.length === 0) return { dtype: "text", physical: "VARCHAR", values };
  if (present.every((v) => v.trim() !== "" && Number.isFinite(Number(v)))) {
    const nums = values.map((v) => (v === null ? null : Number(v)));
    const integer = present.every((v) => /^-?\d+$/.test(v.trim()));
    return integer
      ? { dtype: "integer", physical: "BIGINT", values: nums }
      : { dtype: "numeric", physical: "DOUBLE", values: nums };
  }
  if (present.every((v) => BOOL.has(v.toLowerCase()))) {
    const bools = values.map((v) =>
      v === null ? null : ["true", "yes"].includes(String(v).toLowerCase()),
    );
    return { dtype: "boolean", physical: "BOOLEAN", values: bools };
  }
  if (present.every((v) => DATE.test(v))) return { dtype: "datetime", physical: "DATE", values };
  const unique = new Set(present).size;
  const categorical = unique <= Math.max(20, present.length * 0.5);
  return { dtype: categorical ? "categorical" : "text", physical: "VARCHAR", values };
}

export function parseDelimited(name: string, text: string): MockDataset {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  const sep = name.endsWith(".tsv") || (lines[0] ?? "").includes("\t") ? "\t" : ",";
  const header = splitLine(lines[0] ?? "", sep);
  const rows = lines.slice(1).map((l) => splitLine(l, sep));
  const columns: MockColumn[] = header.map((h, j) => {
    const { dtype, physical, values } = infer(rows.map((r) => r[j] ?? ""));
    return { name: h || `column_${j + 1}`, dtype, physical_type: physical, values };
  });
  return { name, columns, nRows: rows.length, sourceBytes: text.length };
}
