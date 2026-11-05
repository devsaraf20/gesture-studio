// A tiny OSC (Open Sound Control) encoder/decoder - enough for Gesture Studio.
// OSC message = address ("/gesture/peace") + type tags (",f") + arguments, each padded to 4 bytes.
// Numbers are always sent as 32-bit floats ('f'): TouchDesigner and Unreal both read them easily.

function pad4(buf) {
  const extra = (4 - (buf.length % 4)) % 4;
  return Buffer.concat([buf, Buffer.alloc(extra)]);
}

function oscString(s) {
  return pad4(Buffer.from(`${s}\0`, 'utf8'));
}

export function encodeOsc(address, args = []) {
  if (typeof address !== 'string' || !address.startsWith('/')) throw new Error(`bad OSC address: ${address}`);
  let tags = ',';
  const parts = [];
  for (const a of args) {
    if (typeof a === 'string') {
      tags += 's';
      parts.push(oscString(a));
    } else {
      const n = Number(a); // true -> 1, false -> 0
      if (!Number.isFinite(n)) continue;
      tags += 'f';
      const b = Buffer.alloc(4);
      b.writeFloatBE(n);
      parts.push(b);
    }
  }
  return Buffer.concat([oscString(address), oscString(tags), ...parts]);
}

function readString(buf, offset) {
  const end = buf.indexOf(0, offset);
  const s = buf.toString('utf8', offset, end);
  return [s, end + 1 + ((4 - ((end + 1) % 4)) % 4)];
}

export function decodeOsc(buf) {
  let [address, off] = readString(buf, 0);
  if (address === '#bundle') return { address, args: [] }; // bundles not needed here
  let tags;
  [tags, off] = readString(buf, off);
  const args = [];
  for (const t of tags.slice(1)) {
    if (t === 'f') { args.push(buf.readFloatBE(off)); off += 4; }
    else if (t === 'i') { args.push(buf.readInt32BE(off)); off += 4; }
    else if (t === 's') { let s; [s, off] = readString(buf, off); args.push(s); }
    else if (t === 'T') args.push(true);
    else if (t === 'F') args.push(false);
  }
  return { address, args };
}
