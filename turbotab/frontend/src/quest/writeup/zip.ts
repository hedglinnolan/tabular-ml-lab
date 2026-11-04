/**
 * The export's zip, read in the browser: the end of central directory record, each central
 * directory header and the local header it points to (PKWARE's APPNOTE.TXT, §4.3), each entry
 * inflated by the browser's own DecompressionStream ("deflate-raw"). The bundle writes every entry
 * deflated (turbotab/core/export/bundle.py `zip_bytes`), so an entry stored any other way, or
 * whose inflated size is not the size its header records, is refused rather than guessed at; so
 * are bytes that are not a zip. No Zip64, no encryption: the bundle uses neither.
 */

const END = 0x06054b50; // end of central directory record
const CENTRAL = 0x02014b50; // central directory file header
const LOCAL = 0x04034b50; // local file header
const DEFLATED = 8;

/** Whether this browser can inflate a zip's entries (DecompressionStream with "deflate-raw"). */
export function canReadZip(): boolean {
  return typeof DecompressionStream === "function";
}

async function inflate(data: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Response(data).body!.pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function notAZip(why: string): Error {
  return new Error(`not a zip archive: ${why}`);
}

/** Every file of the zip, by its name as the zip stores it (folders left out). */
export async function readZip(bytes: Uint8Array<ArrayBuffer>): Promise<Map<string, Uint8Array<ArrayBuffer>>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u16 = (at: number) => view.getUint16(at, true);
  const u32 = (at: number) => view.getUint32(at, true);
  const inside = (at: number, length: number) => at >= 0 && at + length <= bytes.length;
  // The record is 22 bytes, followed by a comment of at most 65,535.
  let end = -1;
  for (let at = bytes.length - 22; at >= 0 && at >= bytes.length - 22 - 0xffff; at--) {
    if (u32(at) === END) {
      end = at;
      break;
    }
  }
  if (end < 0) throw notAZip("no end of central directory record");
  const count = u16(end + 10);
  let at = u32(end + 16);
  const utf8 = new TextDecoder("utf-8", { fatal: true });
  const out = new Map<string, Uint8Array<ArrayBuffer>>();
  for (let n = 0; n < count; n++) {
    if (!inside(at, 46) || u32(at) !== CENTRAL) throw notAZip(`no central directory header for entry ${n + 1} of ${count}`);
    const method = u16(at + 10);
    const packed = u32(at + 20);
    const size = u32(at + 24);
    const nameLength = u16(at + 28);
    const local = u32(at + 42);
    if (!inside(at + 46, nameLength)) throw notAZip(`entry ${n + 1}'s name runs past the end`);
    const name = utf8.decode(bytes.subarray(at + 46, at + 46 + nameLength));
    at += 46 + nameLength + u16(at + 30) + u16(at + 32);
    if (name.endsWith("/")) continue;
    if (!inside(local, 30) || u32(local) !== LOCAL) throw notAZip(`no local header for ${name}`);
    const start = local + 30 + u16(local + 26) + u16(local + 28);
    if (!inside(start, packed)) throw notAZip(`${name} runs past the end`);
    if (method !== DEFLATED) throw new Error(`${name} is stored with compression method ${method}, which is not read here`);
    const data = await inflate(bytes.slice(start, start + packed));
    if (data.length !== size) throw new Error(`${name} inflates to ${data.length} bytes, not the ${size} its header records`);
    out.set(name, data);
  }
  return out;
}
