/**
 * The export's zip, read in the browser (zip.ts): every file of the zip GET …/export serves, named
 * as the zip stores it and byte for byte as Python's zipfile reads it (fixtures/capture.py writes
 * each file's SHA-256 from the served bytes), and bytes that are not a zip refused.
 */
import estimate from "./fixtures/bundle-estimate.json";
import predict from "./fixtures/bundle-predict.json";
import { readZip } from "./zip";

const bytesOf = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

describe("readZip", () => {
  it.each([
    ["prediction", predict],
    ["estimate", estimate],
  ])("reads every file of the %s bundle as Python's zipfile reads it", async (_, captured) => {
    const files = await readZip(bytesOf(captured.zip));
    expect([...files.keys()].sort()).toEqual(Object.keys(captured.truth.files).sort());
    for (const [name, sha] of Object.entries(captured.truth.files)) {
      expect(await sha256(files.get(name)!), name).toBe(sha);
    }
  });

  it("refuses bytes that are not a zip", async () => {
    await expect(readZip(new TextEncoder().encode("Not a zip archive at all, only words."))).rejects.toThrow(/not a zip/);
  });

  it("refuses a zip whose end is cut off", async () => {
    const whole = bytesOf(predict.zip);
    await expect(readZip(whole.slice(0, whole.length - 40))).rejects.toThrow(/not a zip/);
  });
});
