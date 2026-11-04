/**
 * The export's zip, read in the browser (zip.ts): every file of the zip GET …/export serves, named
 * as the zip stores it and byte for byte as Python's zipfile reads it (fixtures/capture.py writes
 * each file's SHA-256 from the served bytes), and bytes that are not a zip refused, as are an
 * entry not stored deflated and an entry whose inflated size is not its header's; and whether
 * this browser can inflate the "deflate-raw" its entries are stored in (canReadZip).
 */
import estimate from "./fixtures/bundle-estimate.json";
import predict from "./fixtures/bundle-predict.json";
import { canReadZip, readZip } from "./zip";

const bytesOf = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

/** A zip Python's zipfile writes with one entry stored, not deflated (ZIP_STORED, method 0):
 *
 *     buf = io.BytesIO()
 *     with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
 *         z.writestr(zipfile.ZipInfo("README.md", date_time=(1980, 1, 1, 0, 0, 0)), "Stored, not deflated.\n")
 *     base64.b64encode(buf.getvalue())
 */
const STORED =
  "UEsDBBQAAAAAAAAAIQAjckiQFgAAABYAAAAJAAAAUkVBRE1FLm1kU3RvcmVkLCBub3QgZGVmbGF0ZWQuClBLAQIUAxQAAAAAAAAAIQAjckiQFgAAABYAAAAJAAAAAAAAAAAAAACAAQAAAABSRUFETUUubWRQSwUGAAAAAAEAAQA3AAAAPQAAAAAA";

/** Where a zip's central directory header for `name` starts (APPNOTE.TXT §4.3.12: the signature
 *  0x02014b50, the name's length at +28, the name at +46). */
function centralHeaderOf(bytes: Uint8Array<ArrayBuffer>, name: string): number {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const encoded = new TextEncoder().encode(name);
  for (let at = 0; at + 46 <= bytes.length; at++) {
    if (view.getUint32(at, true) !== 0x02014b50 || view.getUint16(at + 28, true) !== encoded.length) continue;
    if (bytes.subarray(at + 46, at + 46 + encoded.length).every((b, i) => b === encoded[i])) return at;
  }
  throw new Error(`no central directory header for ${name}`);
}

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

  it("refuses an entry stored without deflating, naming its method, rather than guess at it", async () => {
    await expect(readZip(bytesOf(STORED))).rejects.toThrow("README.md is stored with compression method 0, which is not read here");
  });

  it("refuses an entry that inflates to another size than its central directory header records, naming both", async () => {
    // The served bundle, with one entry's uncompressed size (central directory header +24) one more
    // than the size Python's zipfile wrote there.
    const bytes = bytesOf(predict.zip);
    const name = "turbotab-export/README.md";
    const at = centralHeaderOf(bytes, name);
    const view = new DataView(bytes.buffer);
    const size = view.getUint32(at + 24, true);
    view.setUint32(at + 24, size + 1, true);
    await expect(readZip(bytes)).rejects.toThrow(`${name} inflates to ${size} bytes, not the ${size + 1} its header records`);
  });
});

describe("canReadZip", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("holds where DecompressionStream inflates \"deflate-raw\", as it does here", () => {
    expect(canReadZip()).toBe(true);
  });

  it("does not where there is no DecompressionStream", () => {
    vi.stubGlobal("DecompressionStream", undefined);
    expect(canReadZip()).toBe(false);
  });

  it("does not where DecompressionStream inflates other formats but refuses \"deflate-raw\"", () => {
    const Real = DecompressionStream;
    vi.stubGlobal(
      "DecompressionStream",
      class {
        constructor(format: CompressionFormat) {
          if (format === "deflate-raw") throw new TypeError("Unsupported compression format: 'deflate-raw'");
          return new Real(format);
        }
      },
    );
    expect(() => new DecompressionStream("deflate")).not.toThrow();
    expect(canReadZip()).toBe(false);
  });
});
