import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const FIXTURE = resolve(dirname(fileURLToPath(import.meta.url)), "../../core/tests/fixtures/nhanes.csv.gz");

/**
 * The real NHANES export the real-server journeys open: $E2E_NHANES, else the tracked fixture
 * decompressed once under the export's own name, _tt_tmp_nhanes.csv (the cache
 * turbotab/core/tests/stage_harness.py's nhanes_fixture writes), so uploads keep that name.
 */
export function nhanesExport(): string {
  if (process.env.E2E_NHANES) return process.env.E2E_NHANES;
  const gz = readFileSync(FIXTURE);
  const key = createHash("sha256").update(gz).digest("hex").slice(0, 16);
  const out = resolve(tmpdir(), `turbotab-nhanes-${key}`, "_tt_tmp_nhanes.csv");
  if (!existsSync(out)) {
    mkdirSync(dirname(out), { recursive: true });
    const part = `${out}.${process.pid}.part`;
    writeFileSync(part, gunzipSync(gz));
    renameSync(part, out);
  }
  return out;
}
