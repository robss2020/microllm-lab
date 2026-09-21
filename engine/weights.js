/** Packed PetitGPT web weight format (PGW1). */

export const DTYPE_CODE = { 0: "f32", 1: "f16", 2: "bf16", 3: "q8", 4: "q4" };

export function parsePgw(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== "PGW1") throw new Error(`bad magic ${magic}`);
  const kind = DTYPE_CODE[view.getUint8(4)] || "f32";
  const archCode = view.getUint8(5);
  const arch = archCode === 1 ? "gpt2" : "llama";
  const cfg = {
    arch,
    vocabSize: view.getUint32(8, true),
    nLayers: view.getUint32(12, true),
    dModel: view.getUint32(16, true),
    nHeads: view.getUint32(20, true),
    nKvHeads: view.getUint32(24, true),
    dFf: view.getUint32(28, true),
    maxSeqLen: view.getUint32(32, true),
    qGroup: view.getUint32(36, true) || 32,
    ropeTheta: view.getFloat64(56, true) || 10000,
    ropePct: view.getFloat64(64, true) || 1,
  };
  const tableOff = view.getUint32(40, true);
  const tableLen = view.getUint32(44, true);
  const payloadOff = view.getUint32(48, true);
  const table = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, tableOff, tableLen)));
  const tensors = Object.create(null);
  for (const e of table) tensors[e.name] = e;
  if (cfg.arch === "gpt2" && !tensors["blocks.0.ln1"] && tensors["blocks.0.norm1"]) {
    cfg.arch = "llama";
  }
  return { kind, cfg, tensors, buffer, payloadOff };
}

export function f32View(bundle, name) {
  const t = bundle.tensors[name];
  if (!t) throw new Error(`missing tensor ${name}`);
  if (t.storage !== "f32") throw new Error(`${name} is ${t.storage}, not f32`);
  return new Float32Array(bundle.buffer, bundle.payloadOff + t.offset, t.nbytes / 4);
}

export function u16View(bundle, name) {
  const t = bundle.tensors[name];
  return new Uint16Array(bundle.buffer, bundle.payloadOff + t.offset, t.nbytes / 2);
}

export function i8View(bundle, name) {
  const t = bundle.tensors[name];
  return new Int8Array(bundle.buffer, bundle.payloadOff + t.offset, t.nbytes);
}

export function u8View(bundle, name) {
  const t = bundle.tensors[name];
  return new Uint8Array(bundle.buffer, bundle.payloadOff + t.offset, t.nbytes);
}

export function scaleView(bundle, name) {
  const t = bundle.tensors[name];
  return new Float32Array(bundle.buffer, bundle.payloadOff + t.scaleOffset, t.scaleNbytes / 4);
}

export function bytesToMB(n) {
  return n / (1024 * 1024);
}

/** Dequantize a 2D weight into a Float32Array (used for tests / small tensors). */
export function dequantizeToF32(bundle, name) {
  const t = bundle.tensors[name];
  const [rows, cols] = t.shape.length === 1 ? [1, t.shape[0]] : t.shape;
  const out = new Float32Array(rows * cols);
  if (t.storage === "f32") {
    out.set(f32View(bundle, name));
    return out;
  }
  if (t.storage === "f16") {
    const u = u16View(bundle, name);
    for (let i = 0; i < u.length; i++) out[i] = f16ToF32(u[i]);
    return out;
  }
  if (t.storage === "bf16") {
    const u = u16View(bundle, name);
    const conv = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < u.length; i++) {
      conv.setUint32(0, u[i] << 16, true);
      out[i] = conv.getFloat32(0, true);
    }
    return out;
  }
  if (t.storage === "q8") {
    const q = i8View(bundle, name);
    const s = scaleView(bundle, name);
    for (let r = 0; r < rows; r++) {
      const sc = s[r];
      const off = r * cols;
      for (let c = 0; c < cols; c++) out[off + c] = q[off + c] * sc;
    }
    return out;
  }
  if (t.storage === "q4") {
    const packed = u8View(bundle, name);
    const s = scaleView(bundle, name);
    const group = t.group || 32;
    const ng = cols / group;
    for (let r = 0; r < rows; r++) {
      for (let g = 0; g < ng; g++) {
        const sc = s[r * ng + g];
        const base = r * (cols / 2) + g * (group / 2);
        const dst = r * cols + g * group;
        for (let k = 0; k < group; k += 2) {
          const b = packed[base + k / 2];
          out[dst + k] = ((b & 15) - 8) * sc;
          out[dst + k + 1] = ((b >> 4) - 8) * sc;
        }
      }
    }
    return out;
  }
  throw new Error(t.storage);
}

const _dv = new DataView(new ArrayBuffer(4));
function _rawF16ToF32(h) {
  const s = (h & 0x8000) << 16;
  let e = (h >> 10) & 0x1f;
  let f = h & 0x3ff;
  let bits;
  if (e === 0) {
    if (f === 0) bits = s;
    else {
      e = 1;
      while ((f & 0x400) === 0) {
        f <<= 1;
        e -= 1;
      }
      f &= 0x3ff;
      bits = s | ((e + 127 - 15) << 23) | (f << 13);
    }
  } else if (e === 31) {
    bits = s | 0x7f800000 | (f << 13);
  } else {
    bits = s | ((e + 127 - 15) << 23) | (f << 13);
  }
  _dv.setUint32(0, bits, true);
  return _dv.getFloat32(0, true);
}

const F16_LUT = new Float32Array(65536);
for (let i = 0; i < 65536; i++) {
  F16_LUT[i] = _rawF16ToF32(i);
}

export function f16ToF32(h) {
  return F16_LUT[h & 0xffff];
}

export function f32ToF16Bits(val) {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, val, true);
  const x = dv.getUint32(0, true);
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const frac = x & 0x7fffff;
  if (exp === 255) return sign | 0x7c00 | (frac ? 0x200 : 0);
  let e = exp - 127 + 15;
  if (e <= 0) {
    if (e < -10) return sign;
    const m = (frac | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  if (e >= 31) return sign | 0x7c00;
  return sign | (e << 10) | ((frac + 0x1000) >> 13);
}
