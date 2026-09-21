/**
 * WebGPU decode for all dtypes.
 *
 * NVIDIA / fused=1: one workgroup walks the transformer; decode uses prompt-lookup
 * speculation. Apple/Safari: multi-workgroup GEMV in webgpu-metal.js, chained decode.
 * Q4/Q8/F16/BF16 stay packed on the GPU; F32 uses f32 GEMV.
 */
import { dequantizeToF32, f32View, u8View, u16View, scaleView } from "./weights.js";
import { SPECIAL } from "./tokenizer.js";
import { initMetal, encodeMetalTokens, METAL_MAX_TOK, CHAIN_TOK } from "./webgpu-metal.js";
import { fusedGpt2Wgsl } from "./fused-gpt2.js";

const MAX_IDS = 256;
const SPEC_K = 2;
const NGRAM = 3;

function packedKind(kind) {
  return kind === "q4" || kind === "q8" || kind === "f16" || kind === "bf16";
}
function scaledKind(kind) {
  return kind === "q4" || kind === "q8";
}

function isMetalGpu(info) {
  const s = `${info?.vendor || ""} ${info?.name || ""} ${info?.architecture || ""}`.toLowerCase();
  if (/apple|metal/.test(s)) return true;
  const ua = globalThis.navigator?.userAgent || "";
  return /Mac|iPhone|iPad/.test(ua);
}

function wantMetalSplit() {
  try {
    const p = new URLSearchParams(globalThis.location?.search || "");
    if (p.get("fused") === "1" || p.get("metal") === "0") return false;
  } catch {
    /* */
  }
  return true;
}

function gemvSource(kind) {
  if (kind === "q4") {
    return `
fn q4nibs(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(word & 15u),
    f32((word >> 4u) & 15u),
    f32((word >> 8u) & 15u),
    f32((word >> 12u) & 15u)
  ) - vec4(8.0);
}
fn q4dot_xn(row: u32, cols: u32, packOff: u32, scaleOff: u32) -> f32 {
  let ng = cols / 32u;
  let rowU = cols / 8u;
  var acc = 0.0;
  for (var g = 0u; g < ng; g++) {
    let sc = SC[scaleOff + row * ng + g];
    let base = packOff + row * rowU + g * 4u;
    let xb = g * 32u;
    var s = 0.0;
    for (var w = 0u; w < 4u; w++) {
      let word = PACK[base + w];
      let i = xb + w * 8u;
      let lo = q4nibs(word);
      let hi = q4nibs(word >> 16u);
      s += dot(lo, vec4(xn[i], xn[i + 1u], xn[i + 2u], xn[i + 3u]));
      s += dot(hi, vec4(xn[i + 4u], xn[i + 5u], xn[i + 6u], xn[i + 7u]));
    }
    acc += s * sc;
  }
  return acc;
}
fn q4dot_src(row: u32, cols: u32, packOff: u32, scaleOff: u32, srcOff: u32) -> f32 {
  let ng = cols / 32u;
  let rowU = cols / 8u;
  var acc = 0.0;
  for (var g = 0u; g < ng; g++) {
    let sc = SC[scaleOff + row * ng + g];
    let base = packOff + row * rowU + g * 4u;
    let xb = g * 32u;
    var s = 0.0;
    for (var w = 0u; w < 4u; w++) {
      let word = PACK[base + w];
      let i = xb + w * 8u;
      let lo = q4nibs(word);
      let hi = q4nibs(word >> 16u);
      s += dot(lo, vec4(SCR[srcOff + i], SCR[srcOff + i + 1u], SCR[srcOff + i + 2u], SCR[srcOff + i + 3u]));
      s += dot(hi, vec4(SCR[srcOff + i + 4u], SCR[srcOff + i + 5u], SCR[srcOff + i + 6u], SCR[srcOff + i + 7u]));
    }
    acc += s * sc;
  }
  return acc;
}
fn gemv_xn(lid: u32, rows: u32, packOff: u32, scaleOff: u32, dstOff: u32) {
  var row = lid;
  while (row < rows) {
    SCR[dstOff + row] = q4dot_xn(row, D, packOff, scaleOff);
    row += WG;
  }
  workgroupBarrier();
}
fn rms_to_xn(lid: u32, normOff: u32) {
  var ss = 0.0;
  var i = lid;
  while (i < D) { let v = SCR[X_OFF + i]; ss += v * v; i += WG; }
  red[lid] = ss;
  let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-6);
  i = lid;
  while (i < D) { xn[i] = SCR[X_OFF + i] * inv * SC[normOff + i]; i += WG; }
  workgroupBarrier();
}
fn embed_row(lid: u32, tok: u32, packOff: u32, scaleOff: u32) {
  var i = lid;
  while (i < D) {
    let ng = D / 32u;
    let rowU = D / 8u;
    let g = i / 32u;
    let sc = SC[scaleOff + tok * ng + g];
    let word = PACK[packOff + tok * rowU + i / 8u];
    let nib = (word >> ((i % 8u) * 4u)) & 15u;
    SCR[X_OFF + i] = (f32(nib) - 8.0) * sc;
    i += WG;
  }
  workgroupBarrier();
}
fn lm_head_row(lid: u32, packOff: u32, scaleOff: u32) {
  var row = lid;
  while (row < VOCAB) { SCR[LOG_OFF + row] = q4dot_xn(row, D, packOff, scaleOff); row += WG; }
  workgroupBarrier();
}
fn w2_acc(lid: u32, packOff: u32, scaleOff: u32) {
  var row = lid;
  while (row < D) {
    SCR[X_OFF + row] = SCR[X_OFF + row] + q4dot_src(row, FF, packOff, scaleOff, FF1_OFF);
    row += WG;
  }
  workgroupBarrier();
}`;
  }
  if (kind === "q8") {
    return `
fn i8_of(word: u32) -> f32 {
  let b = word & 255u;
  return select(f32(b), f32(b) - 256.0, b >= 128u);
}
fn q8dot_xn(row: u32, cols: u32, packOff: u32, scaleOff: u32) -> f32 {
  let sc = SC[scaleOff + row];
  let rowU = cols / 4u;
  var s = 0.0;
  for (var w = 0u; w < rowU; w++) {
    var word = PACK[packOff + row * rowU + w];
    let xb = w * 4u;
    s += i8_of(word) * xn[xb];
    s += i8_of(word >> 8u) * xn[xb + 1u];
    s += i8_of(word >> 16u) * xn[xb + 2u];
    s += i8_of(word >> 24u) * xn[xb + 3u];
  }
  return s * sc;
}
fn q8dot_src(row: u32, cols: u32, packOff: u32, scaleOff: u32, srcOff: u32) -> f32 {
  let sc = SC[scaleOff + row];
  let rowU = cols / 4u;
  var s = 0.0;
  for (var w = 0u; w < rowU; w++) {
    var word = PACK[packOff + row * rowU + w];
    let xb = w * 4u;
    s += i8_of(word) * SCR[srcOff + xb];
    s += i8_of(word >> 8u) * SCR[srcOff + xb + 1u];
    s += i8_of(word >> 16u) * SCR[srcOff + xb + 2u];
    s += i8_of(word >> 24u) * SCR[srcOff + xb + 3u];
  }
  return s * sc;
}
fn gemv_xn(lid: u32, rows: u32, packOff: u32, scaleOff: u32, dstOff: u32) {
  var row = lid;
  while (row < rows) { SCR[dstOff + row] = q8dot_xn(row, D, packOff, scaleOff); row += WG; }
  workgroupBarrier();
}
fn rms_to_xn(lid: u32, normOff: u32) {
  var ss = 0.0;
  var i = lid;
  while (i < D) { let v = SCR[X_OFF + i]; ss += v * v; i += WG; }
  red[lid] = ss;
  let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-6);
  i = lid;
  while (i < D) { xn[i] = SCR[X_OFF + i] * inv * SC[normOff + i]; i += WG; }
  workgroupBarrier();
}
fn embed_row(lid: u32, tok: u32, packOff: u32, scaleOff: u32) {
  let sc = SC[scaleOff + tok];
  let rowU = D / 4u;
  var i = lid;
  while (i < D) {
    let word = PACK[packOff + tok * rowU + i / 4u];
    let sh = (i % 4u) * 8u;
    SCR[X_OFF + i] = i8_of(word >> sh) * sc;
    i += WG;
  }
  workgroupBarrier();
}
fn lm_head_row(lid: u32, packOff: u32, scaleOff: u32) {
  var row = lid;
  while (row < VOCAB) { SCR[LOG_OFF + row] = q8dot_xn(row, D, packOff, scaleOff); row += WG; }
  workgroupBarrier();
}
fn w2_acc(lid: u32, packOff: u32, scaleOff: u32) {
  var row = lid;
  while (row < D) {
    SCR[X_OFF + row] = SCR[X_OFF + row] + q8dot_src(row, FF, packOff, scaleOff, FF1_OFF);
    row += WG;
  }
  workgroupBarrier();
}`;
  }
  if (kind === "f16" || kind === "bf16") {
    const load = kind === "f16"
      ? `fn load_w(packOff: u32, idx: u32) -> f32 {
  let word = PACK[packOff + idx / 2u];
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (idx & 1u) == 1u);
}`
      : `fn load_w(packOff: u32, idx: u32) -> f32 {
  let word = PACK[packOff + idx / 2u];
  let h = select(word & 0xffffu, word >> 16u, (idx & 1u) == 1u);
  return bitcast<f32>(h << 16u);
}`;
    return `
${load}
fn gemv_xn(lid: u32, rows: u32, packOff: u32, _s: u32, dstOff: u32) {
  var row = lid;
  while (row < rows) {
    var s = 0.0;
    let base = row * D;
    for (var k = 0u; k < D; k += 4u) {
      s += load_w(packOff, base + k) * xn[k];
      s += load_w(packOff, base + k + 1u) * xn[k + 1u];
      s += load_w(packOff, base + k + 2u) * xn[k + 2u];
      s += load_w(packOff, base + k + 3u) * xn[k + 3u];
    }
    SCR[dstOff + row] = s;
    row += WG;
  }
  workgroupBarrier();
}
fn rms_to_xn(lid: u32, normOff: u32) {
  var ss = 0.0;
  var i = lid;
  while (i < D) { let v = SCR[X_OFF + i]; ss += v * v; i += WG; }
  red[lid] = ss;
  let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-6);
  i = lid;
  while (i < D) { xn[i] = SCR[X_OFF + i] * inv * SC[normOff + i]; i += WG; }
  workgroupBarrier();
}
fn embed_row(lid: u32, tok: u32, packOff: u32, _s: u32) {
  var i = lid;
  while (i < D) { SCR[X_OFF + i] = load_w(packOff, tok * D + i); i += WG; }
  workgroupBarrier();
}
fn lm_head_row(lid: u32, packOff: u32, _s: u32) {
  var row = lid;
  while (row < VOCAB) {
    var s = 0.0;
    let base = row * D;
    for (var k = 0u; k < D; k += 4u) {
      s += load_w(packOff, base + k) * xn[k];
      s += load_w(packOff, base + k + 1u) * xn[k + 1u];
      s += load_w(packOff, base + k + 2u) * xn[k + 2u];
      s += load_w(packOff, base + k + 3u) * xn[k + 3u];
    }
    SCR[LOG_OFF + row] = s;
    row += WG;
  }
  workgroupBarrier();
}
fn w2_acc(lid: u32, packOff: u32, _s: u32) {
  var row = lid;
  while (row < D) {
    var s = 0.0;
    let base = row * FF;
    for (var k = 0u; k < FF; k += 4u) {
      s += load_w(packOff, base + k) * SCR[FF1_OFF + k];
      s += load_w(packOff, base + k + 1u) * SCR[FF1_OFF + k + 1u];
      s += load_w(packOff, base + k + 2u) * SCR[FF1_OFF + k + 2u];
      s += load_w(packOff, base + k + 3u) * SCR[FF1_OFF + k + 3u];
    }
    SCR[X_OFF + row] = SCR[X_OFF + row] + s;
    row += WG;
  }
  workgroupBarrier();
}`;
  }
  // f32
  return `
fn gemv_xn(lid: u32, rows: u32, wOff: u32, dstOff: u32) {
  var row = lid;
  while (row < rows) {
    var s = 0.0;
    let base = wOff + row * D;
    for (var k = 0u; k < D; k += 4u) {
      s += W[base + k] * xn[k] + W[base + k + 1u] * xn[k + 1u] + W[base + k + 2u] * xn[k + 2u] + W[base + k + 3u] * xn[k + 3u];
    }
    SCR[dstOff + row] = s;
    row += WG;
  }
  workgroupBarrier();
}
fn rms_to_xn(lid: u32, normOff: u32) {
  var ss = 0.0;
  var i = lid;
  while (i < D) { let v = SCR[X_OFF + i]; ss += v * v; i += WG; }
  red[lid] = ss;
  let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-6);
  i = lid;
  while (i < D) { xn[i] = SCR[X_OFF + i] * inv * W[normOff + i]; i += WG; }
  workgroupBarrier();
}
fn embed_row(lid: u32, tok: u32, lmOff: u32, _s: u32) {
  var i = lid;
  while (i < D) { SCR[X_OFF + i] = W[lmOff + tok * D + i]; i += WG; }
  workgroupBarrier();
}
fn lm_head_row(lid: u32, lmOff: u32, _s: u32) {
  var row = lid;
  while (row < VOCAB) {
    var s = 0.0;
    let wbase = lmOff + row * D;
    for (var k = 0u; k < D; k += 4u) {
      s += W[wbase + k] * xn[k] + W[wbase + k + 1u] * xn[k + 1u] + W[wbase + k + 2u] * xn[k + 2u] + W[wbase + k + 3u] * xn[k + 3u];
    }
    SCR[LOG_OFF + row] = s;
    row += WG;
  }
  workgroupBarrier();
}
fn w2_acc(lid: u32, wOff: u32, _s: u32) {
  var row = lid;
  while (row < D) {
    var s = 0.0;
    let wbase = wOff + row * FF;
    for (var k = 0u; k < FF; k += 4u) {
      s += W[wbase + k] * SCR[FF1_OFF + k] + W[wbase + k + 1u] * SCR[FF1_OFF + k + 1u] + W[wbase + k + 2u] * SCR[FF1_OFF + k + 2u] + W[wbase + k + 3u] * SCR[FF1_OFF + k + 3u];
    }
    SCR[X_OFF + row] = SCR[X_OFF + row] + s;
    row += WG;
  }
  workgroupBarrier();
}`;
}

function layerCallsFor(kind) {
  if (scaledKind(kind)) {
    return {
      qkv: `gemv_xn(lid, QKV, OFF[base + 0u], OFF[base + 1u], QKV_OFF);`,
      proj: `gemv_xn(lid, D, OFF[base + 2u], OFF[base + 3u], ATT_OFF);`,
      n1: `rms_to_xn(lid, OFF[base + 10u]);`,
      n2: `rms_to_xn(lid, OFF[base + 11u]);`,
      w1: `gemv_xn(lid, FF, OFF[base + 4u], OFF[base + 5u], FF1_OFF);`,
      w3: `gemv_xn(lid, FF, OFF[base + 6u], OFF[base + 7u], FF3_OFF);`,
      w2: `w2_acc(lid, OFF[base + 8u], OFF[base + 9u]);`,
      stride: 12,
      lm: `OFF[NL * 12u]`,
      lmS: `OFF[NL * 12u + 1u]`,
      nf: `OFF[NL * 12u + 2u]`,
    };
  }
  if (kind === "f16" || kind === "bf16") {
    return {
      qkv: `gemv_xn(lid, QKV, OFF[base + 0u], 0u, QKV_OFF);`,
      proj: `gemv_xn(lid, D, OFF[base + 1u], 0u, ATT_OFF);`,
      n1: `rms_to_xn(lid, OFF[base + 5u]);`,
      n2: `rms_to_xn(lid, OFF[base + 6u]);`,
      w1: `gemv_xn(lid, FF, OFF[base + 2u], 0u, FF1_OFF);`,
      w3: `gemv_xn(lid, FF, OFF[base + 3u], 0u, FF3_OFF);`,
      w2: `w2_acc(lid, OFF[base + 4u], 0u);`,
      stride: 7,
      lm: `OFF[NL * 7u]`,
      lmS: `0u`,
      nf: `OFF[NL * 7u + 1u]`,
    };
  }
  return {
    qkv: `gemv_xn(lid, QKV, OFF[base + 0u], QKV_OFF);`,
    proj: `gemv_xn(lid, D, OFF[base + 1u], ATT_OFF);`,
    n1: `rms_to_xn(lid, OFF[base + 5u]);`,
    n2: `rms_to_xn(lid, OFF[base + 6u]);`,
    w1: `gemv_xn(lid, FF, OFF[base + 2u], FF1_OFF);`,
    w3: `gemv_xn(lid, FF, OFF[base + 3u], FF3_OFF);`,
    w2: `w2_acc(lid, OFF[base + 4u], 0u);`,
    stride: 7,
    lm: `OFF[NL * 7u]`,
    lmS: `0u`,
    nf: `OFF[NL * 7u + 1u]`,
  };
}

function fusedWgsl(cfg, kind) {
  const D = cfg.dModel;
  const FF = cfg.dFf;
  const HD = cfg.dModel / cfg.nHeads;
  const NH = cfg.nHeads;
  const NKV = cfg.nKvHeads;
  const QKV = D + 2 * NKV * HD;
  const NL = cfg.nLayers;
  const VOCAB = cfg.vocabSize;
  const MS = cfg.maxSeqLen;
  const RD = Math.floor(HD * (cfg.ropePct || 1));
  const packed = packedKind(kind);
  const wDecl = packed
    ? `@group(0) @binding(0) var<storage, read> PACK: array<u32>;
@group(0) @binding(7) var<storage, read> SC: array<f32>;`
    : `@group(0) @binding(0) var<storage, read> W: array<f32>;`;
  const layerCalls = layerCallsFor(kind);
  return `
const D: u32 = ${D}u;
const FF: u32 = ${FF}u;
const HD: u32 = ${HD}u;
const NH: u32 = ${NH}u;
const NKV: u32 = ${NKV}u;
const QKV: u32 = ${QKV}u;
const NL: u32 = ${NL}u;
const VOCAB: u32 = ${VOCAB}u;
const MS: u32 = ${MS}u;
const RD: u32 = ${RD}u;
const WG: u32 = 256u;
const X_OFF: u32 = 0u;
const XN_OFF: u32 = D;
const QKV_OFF: u32 = XN_OFF + D;
const ATT_OFF: u32 = QKV_OFF + QKV;
const FF1_OFF: u32 = ATT_OFF + D;
const FF3_OFF: u32 = FF1_OFF + FF;
const LOG_OFF: u32 = FF3_OFF + FF;
const LAYER: u32 = ${layerCalls.stride}u;

struct Params {
  pos0: u32,
  n_tok: u32,
  do_logits: u32,
  logits_each: u32,
  ids: array<vec4<u32>, 64>,
};

${wDecl}
@group(0) @binding(1) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(2) var<storage, read_write> KV: array<f32>;
@group(0) @binding(3) var<storage, read> ROPE: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(5) var<storage, read> OFF: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> xn: array<f32, ${D}>;
var<workgroup> red: array<f32, 256>;
var<workgroup> redi: array<u32, 256>;

fn kv_elems() -> u32 { return NKV * MS * HD; }
fn k_off(li: u32, h: u32, t: u32, d: u32) -> u32 {
  return li * kv_elems() + (h * MS + t) * HD + d;
}
fn v_off(li: u32, h: u32, t: u32, d: u32) -> u32 {
  return NL * kv_elems() + k_off(0u, h, t, d) + li * kv_elems();
}
fn cos_at(pos: u32, d: u32) -> f32 { return ROPE[pos * RD + d]; }
fn sin_at(pos: u32, d: u32) -> f32 { return ROPE[MS * RD + pos * RD + d]; }
fn id_at(i: u32) -> u32 {
  let v = params.ids[i / 4u];
  switch i % 4u {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return v.z; }
    default: { return v.w; }
  }
}
fn reduce_sum(lid: u32) -> f32 {
  workgroupBarrier();
  if (lid < 128u) { red[lid] += red[lid + 128u]; }
  workgroupBarrier();
  if (lid < 64u) { red[lid] += red[lid + 64u]; }
  workgroupBarrier();
  if (lid < 32u) { red[lid] += red[lid + 32u]; }
  workgroupBarrier();
  if (lid < 16u) { red[lid] += red[lid + 16u]; }
  workgroupBarrier();
  if (lid < 8u) { red[lid] += red[lid + 8u]; }
  workgroupBarrier();
  if (lid < 4u) { red[lid] += red[lid + 4u]; }
  workgroupBarrier();
  if (lid < 2u) { red[lid] += red[lid + 2u]; }
  workgroupBarrier();
  return red[0] + red[1];
}
fn reduce_max(lid: u32) -> f32 {
  workgroupBarrier();
  if (lid < 128u) { red[lid] = max(red[lid], red[lid + 128u]); }
  workgroupBarrier();
  if (lid < 64u) { red[lid] = max(red[lid], red[lid + 64u]); }
  workgroupBarrier();
  if (lid < 32u) { red[lid] = max(red[lid], red[lid + 32u]); }
  workgroupBarrier();
  if (lid < 16u) { red[lid] = max(red[lid], red[lid + 16u]); }
  workgroupBarrier();
  if (lid < 8u) { red[lid] = max(red[lid], red[lid + 8u]); }
  workgroupBarrier();
  if (lid < 4u) { red[lid] = max(red[lid], red[lid + 4u]); }
  workgroupBarrier();
  if (lid < 2u) { red[lid] = max(red[lid], red[lid + 2u]); }
  workgroupBarrier();
  return max(red[0], red[1]);
}
fn rope_pair(v: f32, mate: f32, pos: u32, d: u32) -> f32 {
  let half = RD / 2u;
  let rh = select(-mate, mate, d >= half);
  return v * cos_at(pos, d) + rh * sin_at(pos, d);
}
${gemvSource(kind)}

fn argmax_to(lid: u32, slot: u32) {
  var bi = 0u;
  var bv = -1e30;
  var i = lid;
  while (i < VOCAB) {
    let v = SCR[LOG_OFF + i];
    if (v > bv) { bv = v; bi = i; }
    i += WG;
  }
  red[lid] = bv;
  redi[lid] = bi;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      if (red[lid + stride] > red[lid]) {
        red[lid] = red[lid + stride];
        redi[lid] = redi[lid + stride];
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid == 0u) { OUT[slot] = redi[0]; }
  workgroupBarrier();
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  let lmOff = ${layerCalls.lm};
  let lmS = ${layerCalls.lmS};
  let nfOff = ${layerCalls.nf};
  let nTok = params.n_tok;
  var row = 0u;
  var i = 0u;
  var e = 0u;

  for (var ti = 0u; ti < nTok; ti++) {
    let pos = params.pos0 + ti;
    let tok = id_at(ti);
    embed_row(lid, tok, lmOff, lmS);

    for (var li = 0u; li < NL; li++) {
      let base = li * LAYER;
      ${layerCalls.n1}
      ${layerCalls.qkv}

      let half = RD / 2u;
      e = lid;
      while (e < NKV * HD) {
        let h = e / HD;
        let d = e % HD;
        let mateD = select(d + half, d - half, d >= half);
        let k0 = SCR[QKV_OFF + D + e];
        let mate = SCR[QKV_OFF + D + h * HD + mateD];
        KV[k_off(li, h, pos, d)] = rope_pair(k0, mate, pos, d);
        KV[v_off(li, h, pos, d)] = SCR[QKV_OFF + D + NKV * HD + e];
        e += WG;
      }
      workgroupBarrier();

      e = lid;
      while (e < NH * HD) {
        let h = e / HD;
        let d = e % HD;
        let mateD = select(d + half, d - half, d >= half);
        let q0 = SCR[QKV_OFF + e];
        let mate = SCR[QKV_OFF + h * HD + mateD];
        SCR[ATT_OFF + e] = rope_pair(q0, mate, pos, d);
        e += WG;
      }
      workgroupBarrier();

      let seq = pos + 1u;
      let scale = inverseSqrt(f32(HD));
      let rep = NH / NKV;
      var sidx = lid;
      while (sidx < NH * seq) {
        let h = sidx / seq;
        let t = sidx % seq;
        let kvh = h / rep;
        var dot = 0.0;
        let qbase = ATT_OFF + h * HD;
        for (var d = 0u; d < HD; d++) {
          dot += SCR[qbase + d] * KV[k_off(li, kvh, t, d)];
        }
        SCR[LOG_OFF + sidx] = dot * scale;
        sidx += WG;
      }
      workgroupBarrier();

      for (var h = 0u; h < NH; h++) {
        let soff = LOG_OFF + h * seq;
        var mx = -1e30;
        var t = lid;
        while (t < seq) { mx = max(mx, SCR[soff + t]); t += WG; }
        red[lid] = mx;
        mx = reduce_max(lid);
        var sm = 0.0;
        t = lid;
        while (t < seq) {
          let ex = exp(SCR[soff + t] - mx);
          SCR[soff + t] = ex;
          sm += ex;
          t += WG;
        }
        red[lid] = sm;
        let inv = 1.0 / (reduce_sum(lid) + 1e-20);
        t = lid;
        while (t < seq) { SCR[soff + t] = SCR[soff + t] * inv; t += WG; }
        workgroupBarrier();
      }

      e = lid;
      while (e < NH * HD) {
        let h = e / HD;
        let d = e % HD;
        let kvh = h / rep;
        let soff = LOG_OFF + h * seq;
        var acc = 0.0;
        for (var t = 0u; t < seq; t++) { acc += SCR[soff + t] * KV[v_off(li, kvh, t, d)]; }
        SCR[ATT_OFF + e] = acc;
        e += WG;
      }
      workgroupBarrier();

      row = lid;
      while (row < D) { xn[row] = SCR[ATT_OFF + row]; row += WG; }
      workgroupBarrier();
      ${layerCalls.proj}
      i = lid;
      while (i < D) { SCR[X_OFF + i] = SCR[X_OFF + i] + SCR[ATT_OFF + i]; i += WG; }
      workgroupBarrier();

      ${layerCalls.n2}
      ${layerCalls.w1}
      ${layerCalls.w3}
      i = lid;
      while (i < FF) {
        let v = SCR[FF1_OFF + i];
        SCR[FF1_OFF + i] = (v / (1.0 + exp(-v))) * SCR[FF3_OFF + i];
        i += WG;
      }
      workgroupBarrier();
      ${layerCalls.w2}
    }

    let last = ti == nTok - 1u;
    let want = (params.logits_each == 1u) || (last && params.do_logits == 1u);
    if (want) {
      rms_to_xn(lid, nfOff);
      lm_head_row(lid, lmOff, lmS);
      argmax_to(lid, select(0u, ti, params.logits_each == 1u));
    }
  }
}
`;
}

async function pipe(device, code) {
  const module = device.createShaderModule({ code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errs = (info.messages || []).filter((m) => m.type === "error");
    if (errs.length) {
      throw new Error(
        errs.map((m) => `WGSL:${m.lineNum}:${m.linePos} ${m.message}`).join(" | ").slice(0, 800),
      );
    }
  }
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
}

function buf(device, size, usage, label) {
  return device.createBuffer({ size: Math.max(Math.ceil(size / 16) * 16, 16), usage, label });
}

export async function tryWebGpu() {
  if (!globalThis.navigator?.gpu) return { ok: false, reason: "WebGPU API not present" };
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch {
    adapter = await navigator.gpu.requestAdapter();
  }
  if (!adapter) return { ok: false, reason: "No GPU adapter" };
  const requiredLimits = {};
  try {
    requiredLimits.maxBufferSize = Math.min(adapter.limits.maxBufferSize, 2147483648);
    requiredLimits.maxStorageBufferBindingSize = Math.min(
      adapter.limits.maxStorageBufferBindingSize,
      2147483648,
    );
    requiredLimits.maxUniformBufferBindingSize = Math.min(
      adapter.limits.maxUniformBufferBindingSize || 65536,
      65536,
    );
  } catch {
    /* */
  }
  let device;
  try {
    device = await adapter.requestDevice({ requiredLimits });
  } catch (e1) {
    try {
      device = await adapter.requestDevice();
    } catch (e2) {
      return { ok: false, reason: String(e2 || e1) };
    }
  }
  const info = adapter.info || {};
  const features = [];
  try {
    for (const f of adapter.features) features.push(f);
  } catch {
    /* */
  }
  return {
    ok: true,
    adapter,
    device,
    name: info.device || info.description || "WebGPU",
    vendor: info.vendor || "unknown",
    architecture: info.architecture || "",
    isFallbackAdapter: !!info.isFallbackAdapter,
    features,
    limits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      minUniformBufferOffsetAlignment: adapter.limits.minUniformBufferOffsetAlignment,
    },
  };
}

function buildRope(maxSeq, headDim, theta, pct) {
  let ropeDim = Math.floor(headDim * pct);
  ropeDim -= ropeDim % 2;
  const half = ropeDim / 2;
  const cos = new Float32Array(maxSeq * ropeDim);
  const sin = new Float32Array(maxSeq * ropeDim);
  for (let t = 0; t < maxSeq; t++) {
    for (let i = 0; i < half; i++) {
      const freq = t / theta ** ((2 * i) / ropeDim);
      const c = Math.cos(freq);
      const s = Math.sin(freq);
      cos[t * ropeDim + i] = c;
      cos[t * ropeDim + half + i] = c;
      sin[t * ropeDim + i] = s;
      sin[t * ropeDim + half + i] = s;
    }
  }
  return { cos, sin, ropeDim };
}

function packU8ToU32(bytes) {
  const pad = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
  pad.set(bytes);
  return new Uint32Array(pad.buffer);
}

function packU16ToU32(u16) {
  const out = new Uint32Array(Math.ceil(u16.length / 2));
  for (let i = 0; i < u16.length; i += 2) {
    out[i >> 1] = u16[i] | ((i + 1 < u16.length ? u16[i + 1] : 0) << 16);
  }
  return out;
}

/** Prompt-lookup draft: continuation only from the prompt, n-gram match. */
function lookupDraft(prompt, k, n) {
  if (prompt.length < n + 1) return [];
  const suf = prompt.slice(-n);
  const limit = prompt.length - n;
  for (let i = limit - 1; i >= 0; i--) {
    let ok = true;
    for (let j = 0; j < n; j++) {
      if (prompt[i + j] !== suf[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      const d = [];
      for (let t = 0; t < k && i + n + t < prompt.length; t++) d.push(prompt[i + n + t]);
      return d;
    }
  }
  return [];
}

export class GpuPetitGPT {
  constructor(cpuModel, gpuInfo) {
    this.cpu = cpuModel;
    this.cfg = cpuModel.cfg;
    this.kind = cpuModel.kind;
    this.bundle = cpuModel.bundle;
    this.info = gpuInfo;
    this.device = gpuInfo.device;
    this.bytesAllocated = 0;
    this.cacheLen = 0;
    this.shaderKind = cpuModel.kind;
    this.mode = `fused-${cpuModel.kind}-pld`;
    this.useMetalSplit = false;
    this.metalError = null;
  }

  _alloc(size, usage, label) {
    const b = buf(this.device, size, usage, label);
    this.bytesAllocated += Math.max(Math.ceil(size / 16) * 16, 16);
    return b;
  }

  async init() {
    const d = this.device;
    const c = this.cfg;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const hd = c.dModel / c.nHeads;
    const kind = this.kind;
    const metal = isMetalGpu(this.info);
    this.shaderKind = kind;
    const mat = (li, name) => `blocks.${li}.${name}`;

    if (scaledKind(kind)) {
      let packU32 = 0;
      let scF = 0;
      const packChunks = [];
      const scChunks = [];
      const offsets = [];
      const pushMat = (name) => {
        const t = this.bundle.tensors[name];
        if (t && t.storage === "f16") {
          const u32 = packU16ToU32(u16View(this.bundle, name));
          offsets.push(packU32, scF);
          packChunks.push(u32);
          packU32 += u32.length;
          return;
        }
        const packed = u8View(this.bundle, name);
        const scales = scaleView(this.bundle, name);
        const u32 = packU8ToU32(packed);
        offsets.push(packU32, scF);
        packChunks.push(u32);
        scChunks.push(new Float32Array(scales));
        packU32 += u32.length;
        scF += scales.length;
      };
      const pushNorm = (name) => {
        const data = new Float32Array(f32View(this.bundle, name));
        offsets.push(scF);
        scChunks.push(data);
        scF += data.length;
      };
      if (c.arch === "gpt2") {
        for (let li = 0; li < c.nLayers; li++) {
          pushMat(mat(li, "attn.qkv"));
          pushNorm(mat(li, "attn.qkv_bias"));
          pushMat(mat(li, "attn.proj"));
          pushNorm(mat(li, "attn.proj_bias"));
          pushMat(mat(li, "mlp.fc"));
          pushNorm(mat(li, "mlp.fc_bias"));
          pushMat(mat(li, "mlp.proj"));
          pushNorm(mat(li, "mlp.proj_bias"));
          pushNorm(mat(li, "ln1"));
          pushNorm(mat(li, "ln2"));
        }
        pushMat("lm_head");
        pushNorm("ln_f");
        pushNorm("wpe");
      } else {
      for (let li = 0; li < c.nLayers; li++) {
        pushMat(mat(li, "attn.qkv"));
        pushMat(mat(li, "attn.proj"));
        pushMat(mat(li, "mlp.w1"));
        pushMat(mat(li, "mlp.w3"));
        pushMat(mat(li, "mlp.w2"));
        pushNorm(mat(li, "norm1"));
        pushNorm(mat(li, "norm2"));
      }
      pushMat("lm_head");
      pushNorm("norm_f");
      }
      this.W = this._alloc(packU32 * 4, storage, "PACK");
      this.SC = this._alloc(scF * 4, storage, "SC");
      let po = 0;
      for (const ch of packChunks) {
        d.queue.writeBuffer(this.W, po, ch);
        po += ch.byteLength;
      }
      let so = 0;
      for (const ch of scChunks) {
        d.queue.writeBuffer(this.SC, so, ch);
        so += ch.byteLength;
      }
      this._offHost = new Uint32Array(offsets);
    } else if (kind === "f16" || kind === "bf16") {
      const packChunks = [];
      const scChunks = [];
      const offsets = [];
      let packU32 = 0;
      let scF = 0;
      const pushMat = (name) => {
        const u32 = packU16ToU32(u16View(this.bundle, name));
        offsets.push(packU32);
        packChunks.push(u32);
        packU32 += u32.length;
      };
      const pushNorm = (name) => {
        const data = new Float32Array(f32View(this.bundle, name));
        offsets.push(scF);
        scChunks.push(data);
        scF += data.length;
      };
      for (let li = 0; li < c.nLayers; li++) {
        pushMat(mat(li, "attn.qkv"));
        pushMat(mat(li, "attn.proj"));
        pushMat(mat(li, "mlp.w1"));
        pushMat(mat(li, "mlp.w3"));
        pushMat(mat(li, "mlp.w2"));
        pushNorm(mat(li, "norm1"));
        pushNorm(mat(li, "norm2"));
      }
      pushMat("lm_head");
      pushNorm("norm_f");
      this.W = this._alloc(packU32 * 4, storage, "PACK");
      this.SC = this._alloc(scF * 4, storage, "SC");
      let po = 0;
      for (const ch of packChunks) {
        d.queue.writeBuffer(this.W, po, ch);
        po += ch.byteLength;
      }
      let so = 0;
      for (const ch of scChunks) {
        d.queue.writeBuffer(this.SC, so, ch);
        so += ch.byteLength;
      }
      this._offHost = new Uint32Array(offsets);
    } else {
      const order = [];
      for (let li = 0; li < c.nLayers; li++) {
        order.push(
          mat(li, "attn.qkv"),
          mat(li, "attn.proj"),
          mat(li, "mlp.w1"),
          mat(li, "mlp.w3"),
          mat(li, "mlp.w2"),
          mat(li, "norm1"),
          mat(li, "norm2"),
        );
      }
      order.push("lm_head", "norm_f");
      const off = new Uint32Array(order.length);
      let floats = 0;
      for (let i = 0; i < order.length; i++) {
        const t = this.bundle.tensors[order[i]];
        off[i] = floats;
        floats += t.shape.reduce((a, b) => a * b, 1);
      }
      this.W = this._alloc(floats * 4, storage, "W");
      for (let i = 0; i < order.length; i++) {
        const t = this.bundle.tensors[order[i]];
        const data = t.storage === "f32" ? f32View(this.bundle, order[i]) : dequantizeToF32(this.bundle, order[i]);
        d.queue.writeBuffer(this.W, off[i] * 4, data);
      }
      this._offHost = off;
      this.SC = null;
    }

    const logOff = c.dModel * 2 + (c.dModel + 2 * c.nKvHeads * hd) + c.dModel + c.dFf * 2;
    this.SCR = this._alloc((logOff + c.vocabSize + 64) * 4, storage, "SCR");
    this.KV = this._alloc(2 * c.nLayers * c.nKvHeads * c.maxSeqLen * hd * 4, storage, "KV");
    const rope = buildRope(c.maxSeqLen, hd, c.ropeTheta, c.ropePct);
    if (c.arch === "gpt2") {
      rope.cos.fill(1);
      rope.sin.fill(0);
    }
    const ropePacked = new Float32Array(rope.cos.length + rope.sin.length);
    ropePacked.set(rope.cos, 0);
    ropePacked.set(rope.sin, rope.cos.length);
    this.ROPE = this._alloc(ropePacked.byteLength, storage, "ROPE");
    d.queue.writeBuffer(this.ROPE, 0, ropePacked);
    this.OUT = this._alloc(MAX_IDS * 4, storage, "OUT");
    this.outStage = this._alloc(MAX_IDS * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "outStage");
    this.OFF = this._alloc(this._offHost.byteLength, storage, "OFF");
    d.queue.writeBuffer(this.OFF, 0, this._offHost);
    this.uMeta = this._alloc(16 + MAX_IDS * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "params");

    if (metal && wantMetalSplit()) {
      try {
        await initMetal(this);
      } catch (e) {
        this.metalError = String(e?.message || e);
        console.warn("[webgpu] metal split failed, fused", e);
        this.useMetalSplit = false;
      }
    }
    if (!this.useMetalSplit) {
      this.pipeline = await pipe(d, c.arch === "gpt2" ? fusedGpt2Wgsl(c) : fusedWgsl(c, kind));
      const entries = [
        { binding: 0, resource: { buffer: this.W } },
        { binding: 1, resource: { buffer: this.SCR } },
        { binding: 2, resource: { buffer: this.KV } },
        { binding: 3, resource: { buffer: this.ROPE } },
        { binding: 4, resource: { buffer: this.OUT } },
        { binding: 5, resource: { buffer: this.OFF } },
        { binding: 6, resource: { buffer: this.uMeta } },
      ];
      if (this.SC) entries.push({ binding: 7, resource: { buffer: this.SC } });
      this.bg = d.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries });
    }

    this.cacheLen = 0;
    this.ready = true;
    this.cpu = null;
    this.bundle = null;
    this.device.addEventListener("uncapturederror", (ev) => {
      console.warn("[webgpu]", ev.error?.message || ev.error);
    });
    // Warm the pipeline so the first real prompt is not paying shader compile.
    try {
      await this._run(0, [0], { doLogits: true, logitsEach: false });
    } catch {
      /* warmup is best-effort */
    }
    this.cacheLen = 0;
    return this;
  }

  _writeParams(pos0, ids, { doLogits, logitsEach }) {
    const n = Math.min(ids.length, MAX_IDS);
    const u = new Uint32Array(4 + MAX_IDS);
    u[0] = pos0;
    u[1] = n;
    u[2] = doLogits ? 1 : 0;
    u[3] = logitsEach ? 1 : 0;
    for (let i = 0; i < n; i++) u[4 + i] = ids[i];
    this.device.queue.writeBuffer(this.uMeta, 0, u);
    return n;
  }

  async _run(pos0, ids, { doLogits, logitsEach }) {
    if (this.useMetalSplit) return this._runMetal(pos0, ids, { doLogits, logitsEach });
    const n = this._writeParams(pos0, ids, { doLogits, logitsEach });
    const bytes = logitsEach ? n * 4 : 4;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    if (doLogits || logitsEach) enc.copyBufferToBuffer(this.OUT, 0, this.outStage, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    if (!doLogits && !logitsEach) return [];
    await this.outStage.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(this.outStage.getMappedRange().slice(0, bytes));
    const out = Array.from(v);
    this.outStage.unmap();
    return out;
  }

  async _runMetal(pos0, ids, { doLogits, logitsEach }) {
    const n = Math.min(ids.length, METAL_MAX_TOK);
    const bytes = logitsEach ? n * 4 : 4;
    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encodeMetalTokens(this, pass, pos0, ids.slice(0, n), { doLogits, logitsEach });
    pass.end();
    if (doLogits || logitsEach) enc.copyBufferToBuffer(this.OUT, 0, this.outStage, 0, bytes);
    const t1 = performance.now();
    this.device.queue.submit([enc.finish()]);
    if (!doLogits && !logitsEach) {
      this.lastTiming = { n, encodeMs: t1 - t0, gpuMs: 0 };
      return [];
    }
    await this.outStage.mapAsync(GPUMapMode.READ);
    const t2 = performance.now();
    const v = new Uint32Array(this.outStage.getMappedRange().slice(0, bytes));
    const out = Array.from(v);
    this.outStage.unmap();
    this.lastTiming = { n, encodeMs: t1 - t0, gpuMs: t2 - t1 };
    return out;
  }

  async generate(ids, { maxNewTokens = 64, eosId = SPECIAL.EOS, onToken = null } = {}) {
    this.cacheLen = 0;
    const t0 = performance.now();
    const prompt = ids.slice();
    const accepted = [];
    let specHits = 0;
    let specTries = 0;

    const CHUNK = this.useMetalSplit ? METAL_MAX_TOK : MAX_IDS;
    for (let i = 0; i < prompt.length; ) {
      const lastChunk = i + CHUNK >= prompt.length;
      const chunk = prompt.slice(i, Math.min(i + CHUNK, prompt.length));
      const r = await this._run(i, chunk, { doLogits: lastChunk, logitsEach: false });
      i += chunk.length;
      if (lastChunk) accepted.push(r[0]);
    }
    this.cacheLen = prompt.length;
    const ttft = performance.now() - t0;
    if (onToken) await onToken(accepted[0], { phase: "prefill", ms: ttft });
    if (accepted[0] === eosId) {
      return {
        generatedIds: accepted,
        stopReason: "eos",
        ttftMs: ttft,
        totalMs: performance.now() - t0,
        mode: this.mode,
        specHits,
        specTries,
      };
    }

    if (this.useMetalSplit) {
      // Chat / ignoreEos: one chained pass (high tok/s). Suite: 1-token steps so we
      // stop at EOS instead of paying for maxNew unused forwards.
      const chainAll = !!onToken || eosId < 0;
      while (accepted.length < maxNewTokens) {
        const x = accepted[accepted.length - 1];
        const pos = this.cacheLen;
        const n = chainAll ? Math.min(METAL_MAX_TOK, maxNewTokens - accepted.length) : 1;
        const runIds = new Array(n);
        runIds[0] = x;
        for (let i = 1; i < n; i++) runIds[i] = CHAIN_TOK;
        const r = await this._run(pos, runIds, { doLogits: true, logitsEach: n > 1 });
        let stop = false;
        for (let i = 0; i < n; i++) {
          const tok = r[i];
          accepted.push(tok);
          this.cacheLen = pos + i + 1;
          if (onToken) await onToken(tok, { phase: i ? "chain" : "decode" });
          if (tok === eosId) {
            stop = true;
            break;
          }
        }
        if (stop) break;
      }
      return {
        generatedIds: accepted,
        stopReason: accepted[accepted.length - 1] === eosId ? "eos" : "max_new_tokens",
        ttftMs: ttft,
        totalMs: performance.now() - t0,
        mode: this.mode,
        specHits,
        specTries,
        timing: this.lastTiming,
      };
    }

    const seq = prompt.concat(accepted);
    while (accepted.length < maxNewTokens) {
      const x = accepted[accepted.length - 1];
      const pos = this.cacheLen;
      const draft = lookupDraft(seq, SPEC_K, NGRAM);
      specTries++;
      const runIds = [x, ...draft];
      const logitsEach = draft.length > 0;
      const r = await this._run(pos, runIds, { doLogits: true, logitsEach });
      const a0 = r[0];
      accepted.push(a0);
      seq.push(a0);
      this.cacheLen = pos + 1;
      if (onToken) await onToken(a0, { phase: "decode" });
      if (a0 === eosId) break;
      if (draft.length && logitsEach) {
        for (let k = 0; k < draft.length; k++) {
          if (r[k] !== draft[k]) break;
          if (accepted.length >= maxNewTokens) break;
          const nxt = r[k + 1];
          if (nxt === undefined) break;
          specHits++;
          accepted.push(nxt);
          seq.push(nxt);
          this.cacheLen = pos + 2 + k;
          if (onToken) await onToken(nxt, { phase: "spec" });
          if (nxt === eosId) break;
        }
      }
      if (accepted[accepted.length - 1] === eosId) break;
    }

    return {
      generatedIds: accepted,
      stopReason: accepted[accepted.length - 1] === eosId ? "eos" : "max_new_tokens",
      ttftMs: ttft,
      totalMs: performance.now() - t0,
      mode: this.mode,
      specHits,
      specTries,
      timing: this.lastTiming,
    };
  }

  destroy() {
    try {
      this.device.destroy();
    } catch {
      /* */
    }
  }
}

export function gpuMemoryEstimate(cfg, kind) {
  const diskPer = kind === "f32" ? 4 : kind === "q8" ? 1 : kind === "q4" ? 0.5 : 2;
  const gpuPer = kind === "q4" ? 0.55 : kind === "q8" ? 1.05 : kind === "f32" ? 4 : 2.05;
  const nParams = cfg.vocabSize * cfg.dModel + cfg.nLayers * (
    cfg.dModel * 2 + (cfg.dModel + 2 * cfg.nKvHeads * (cfg.dModel / cfg.nHeads)) * cfg.dModel
    + cfg.dModel * cfg.dModel + cfg.dFf * cfg.dModel * 3
  );
  const weightsDisk = nParams * diskPer;
  const weightsGpu = nParams * gpuPer;
  const kv = cfg.nLayers * cfg.nKvHeads * cfg.maxSeqLen * (cfg.dModel / cfg.nHeads) * 2 * 4;
  return { weightsDisk, weightsGpu, kv, total: weightsGpu + kv };
}
