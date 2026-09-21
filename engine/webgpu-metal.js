/**
 * Metal / Safari decoder.
 *
 * Bindings are declared before any helper that reads them (Safari WGSL is strict).
 * GEMV uses many workgroups: cooperative K-reduction for small matrices, one-thread-
 * per-row for the vocab head. Activations sit in threadgroup memory. RMS and SiLU
 * are fused into the GEMV that consumes them. Decode can chain tokens inside one
 * compute pass by reading the previous argmax from OUT.
 */

export const GEMV_WG = 256;
export const ROW_TILE = 8;
export const K_THREADS = 32;
export const JOB_STRIDE = 256;
export const METAL_MAX_TOK = 64;
export const FLAG_RMS = 2;
export const FLAG_SILU = 4;
export const FLAG_LN = 8;
export const FLAG_BIAS = 16;
export const FLAG_GELU = 32;
export const CHAIN_TOK = 0xffffffff;

const COMPUTE = GPUShaderStage.COMPUTE;

function consts(cfg) {
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
  return { D, FF, HD, NH, NKV, QKV, NL, VOCAB, MS, RD };
}

function constWgsl(c) {
  return `
const D: u32 = ${c.D}u;
const FF: u32 = ${c.FF}u;
const HD: u32 = ${c.HD}u;
const NH: u32 = ${c.NH}u;
const NKV: u32 = ${c.NKV}u;
const QKV: u32 = ${c.QKV}u;
const NL: u32 = ${c.NL}u;
const VOCAB: u32 = ${c.VOCAB}u;
const MS: u32 = ${c.MS}u;
const RD: u32 = ${c.RD}u;
const X_OFF: u32 = 0u;
const XN_OFF: u32 = D;
const QKV_OFF: u32 = XN_OFF + D;
const ATT_OFF: u32 = QKV_OFF + QKV;
const FF1_OFF: u32 = ATT_OFF + D;
const FF3_OFF: u32 = FF1_OFF + FF;
const LOG_OFF: u32 = FF3_OFF + FF;
const FLAG_RMS: u32 = 2u;
const FLAG_SILU: u32 = 4u;
const FLAG_LN: u32 = 8u;
const FLAG_BIAS: u32 = 16u;
const FLAG_GELU: u32 = 32u;
const CHAIN: u32 = 0xFFFFFFFFu;
const ROW_TILE: u32 = ${ROW_TILE}u;
const K_THREADS: u32 = ${K_THREADS}u;
const WG: u32 = ${GEMV_WG}u;
`;
}

function structsWgsl() {
  return `
struct Job {
  rows: u32, cols: u32, packOff: u32, scaleOff: u32,
  srcOff: u32, dstOff: u32, addX: u32, flags: u32,
  normOff: u32, biasOff: u32, _p1: u32, _p2: u32,
};
struct Step { pos: u32, token: u32, seq: u32, slot: u32, };
`;
}

function packDecl(kind) {
  if (kind === "f32") return `@group(0) @binding(0) var<storage, read> W: array<f32>;`;
  return `@group(0) @binding(0) var<storage, read> PACK: array<u32>;`;
}

function dotFns(kind) {
  if (kind === "q4") {
    return `
fn q4nibs(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(word & 15u), f32((word >> 4u) & 15u),
    f32((word >> 8u) & 15u), f32((word >> 12u) & 15u)
  ) - vec4<f32>(8.0);
}
fn q4group(row: u32, g: u32, cols: u32, packOff: u32, scaleOff: u32) -> f32 {
  let ng = cols / 32u;
  let rowU = cols / 8u;
  let sc = SC[scaleOff + row * ng + g];
  let base = packOff + row * rowU + g * 4u;
  let xb = g * 32u;
  var s = 0.0;
  for (var w = 0u; w < 4u; w++) {
    let word = PACK[base + w];
    let i = xb + w * 8u;
    let lo = q4nibs(word);
    let hi = q4nibs(word >> 16u);
    s += dot(lo, vec4<f32>(src[i], src[i + 1u], src[i + 2u], src[i + 3u]));
    s += dot(hi, vec4<f32>(src[i + 4u], src[i + 5u], src[i + 6u], src[i + 7u]));
  }
  return s * sc;
}
fn dot_full(row: u32, cols: u32, packOff: u32, scaleOff: u32) -> f32 {
  let ng = cols / 32u;
  var acc = 0.0;
  for (var g = 0u; g < ng; g++) { acc += q4group(row, g, cols, packOff, scaleOff); }
  return acc;
}
fn dot_slice(row: u32, cols: u32, packOff: u32, scaleOff: u32, kid: u32, stride: u32) -> f32 {
  let ng = cols / 32u;
  let rowU = cols / 8u;
  var acc = 0.0;
  var w = kid;
  while (w < rowU) {
    let word = PACK[packOff + row * rowU + w];
    let g = w / 4u;
    let sc = SC[scaleOff + row * ng + g];
    let xb = w * 8u;
    let lo = q4nibs(word);
    let hi = q4nibs(word >> 16u);
    acc += sc * (dot(lo, vec4<f32>(src[xb], src[xb + 1u], src[xb + 2u], src[xb + 3u]))
              + dot(hi, vec4<f32>(src[xb + 4u], src[xb + 5u], src[xb + 6u], src[xb + 7u])));
    w += stride;
  }
  return acc;
}
fn scale_acc(acc: f32, _row: u32, _scaleOff: u32) -> f32 { return acc; }
`;
  }
  if (kind === "q8") {
    return `
fn i8_of(word: u32) -> f32 {
  let b = word & 255u;
  return select(f32(b), f32(b) - 256.0, b >= 128u);
}
fn q8_word(word: u32, xb: u32) -> f32 {
  return i8_of(word) * src[xb]
       + i8_of(word >> 8u) * src[xb + 1u]
       + i8_of(word >> 16u) * src[xb + 2u]
       + i8_of(word >> 24u) * src[xb + 3u];
}
fn dot_full(row: u32, cols: u32, packOff: u32, _s: u32) -> f32 {
  let rowU = cols / 4u;
  var s = 0.0;
  for (var w = 0u; w < rowU; w++) {
    s += q8_word(PACK[packOff + row * rowU + w], w * 4u);
  }
  return s;
}
fn dot_slice(row: u32, cols: u32, packOff: u32, _s: u32, kid: u32, stride: u32) -> f32 {
  let rowU = cols / 4u;
  var s = 0.0;
  var w = kid;
  while (w < rowU) {
    s += q8_word(PACK[packOff + row * rowU + w], w * 4u);
    w += stride;
  }
  return s;
}
fn scale_acc(acc: f32, row: u32, scaleOff: u32) -> f32 { return acc * SC[scaleOff + row]; }
`;
  }
  if (kind === "f16" || kind === "bf16") {
    const pair =
      kind === "f16"
        ? `fn packed_pair(packOff: u32, row: u32, cols: u32, w: u32) -> f32 {
  let word = PACK[packOff + (row * cols) / 2u + w];
  let pair = unpack2x16float(word);
  let xb = w * 2u;
  return pair.x * src[xb] + pair.y * src[xb + 1u];
}`
        : `fn packed_pair(packOff: u32, row: u32, cols: u32, w: u32) -> f32 {
  let word = PACK[packOff + (row * cols) / 2u + w];
  let xb = w * 2u;
  let lo = bitcast<f32>((word & 0xffffu) << 16u);
  let hi = bitcast<f32>((word >> 16u) << 16u);
  return lo * src[xb] + hi * src[xb + 1u];
}`;
    return `
${pair}
fn dot_full(row: u32, cols: u32, packOff: u32, _s: u32) -> f32 {
  var s = 0.0;
  let nW = cols / 2u;
  for (var w = 0u; w < nW; w++) { s += packed_pair(packOff, row, cols, w); }
  return s;
}
fn dot_slice(row: u32, cols: u32, packOff: u32, _s: u32, kid: u32, stride: u32) -> f32 {
  var s = 0.0;
  let nW = cols / 2u;
  var w = kid;
  while (w < nW) { s += packed_pair(packOff, row, cols, w); w += stride; }
  return s;
}
fn scale_acc(acc: f32, _row: u32, _s: u32) -> f32 { return acc; }
`;
  }
  return `
fn dot_full(row: u32, cols: u32, packOff: u32, _s: u32) -> f32 {
  var s = 0.0;
  let base = packOff + row * cols;
  var k = 0u;
  while (k + 3u < cols) {
    s += W[base + k] * src[k] + W[base + k + 1u] * src[k + 1u]
       + W[base + k + 2u] * src[k + 2u] + W[base + k + 3u] * src[k + 3u];
    k += 4u;
  }
  while (k < cols) { s += W[base + k] * src[k]; k++; }
  return s;
}
fn dot_slice(row: u32, cols: u32, packOff: u32, _s: u32, kid: u32, stride: u32) -> f32 {
  var s = 0.0;
  let base = packOff + row * cols;
  var w = kid;
  let nV = cols / 4u;
  while (w < nV) {
    let k = w * 4u;
    s += W[base + k] * src[k] + W[base + k + 1u] * src[k + 1u]
       + W[base + k + 2u] * src[k + 2u] + W[base + k + 3u] * src[k + 3u];
    w += stride;
  }
  return s;
}
fn scale_acc(acc: f32, _row: u32, _s: u32) -> f32 { return acc; }
`;
}

function gammaLoad(kind) {
  return kind === "f32" ? `W[job.normOff + i]` : `SC[job.normOff + i]`;
}

function embedBody(kind) {
  if (kind === "q4") {
    return `
  let ng = D / 32u;
  let rowU = D / 8u;
  let g = i / 32u;
  let sc = SC[job.scaleOff + tok * ng + g];
  let word = PACK[job.packOff + tok * rowU + i / 8u];
  let nib = (word >> ((i % 8u) * 4u)) & 15u;
  SCR[X_OFF + i] = (f32(nib) - 8.0) * sc;`;
  }
  if (kind === "q8") {
    return `
  let sc = SC[job.scaleOff + tok];
  let rowU = D / 4u;
  let word = PACK[job.packOff + tok * rowU + i / 4u];
  let sh = (i % 4u) * 8u;
  let b = (word >> sh) & 255u;
  let q = select(f32(b), f32(b) - 256.0, b >= 128u);
  SCR[X_OFF + i] = q * sc;`;
  }
  if (kind === "f16") {
    return `
  let word = PACK[job.packOff + (tok * D + i) / 2u];
  let pair = unpack2x16float(word);
  SCR[X_OFF + i] = select(pair.x, pair.y, (i & 1u) == 1u);`;
  }
  if (kind === "bf16") {
    return `
  let word = PACK[job.packOff + (tok * D + i) / 2u];
  let h = select(word & 0xffffu, word >> 16u, (i & 1u) == 1u);
  SCR[X_OFF + i] = bitcast<f32>(h << 16u);`;
  }
  return `
  SCR[X_OFF + i] = W[job.packOff + tok * D + i];`;
}

function metalWgsl(cfg, kind) {
  const c = consts(cfg);
  const C = constWgsl(c);
  const S = structsWgsl();
  const P = packDecl(kind);
  const gamma = gammaLoad(kind);

  const gemv = `
${C}
${S}
${P}
@group(0) @binding(1) var<storage, read> SC: array<f32>;
@group(0) @binding(2) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(3) var<uniform> job: Job;

var<workgroup> src: array<f32, ${Math.max(c.D, c.FF)}>;
var<workgroup> red: array<f32, 256>;

${dotFns(kind)}

fn reduce_sum_wg(lid: u32) {
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
}

fn prepare_src(lid: u32) {
  if ((job.flags & FLAG_RMS) == FLAG_RMS) {
    var ss = 0.0;
    var i = lid;
    while (i < D) {
      let v = SCR[X_OFF + i];
      ss += v * v;
      i += WG;
    }
    red[lid] = ss;
    reduce_sum_wg(lid);
    let inv = inverseSqrt((red[0] + red[1]) / f32(D) + 1e-6);
    i = lid;
    while (i < D) {
      src[i] = SCR[X_OFF + i] * inv * (${gamma});
      i += WG;
    }
    workgroupBarrier();
    return;
  }
  if ((job.flags & FLAG_LN) == FLAG_LN) {
    var sm = 0.0;
    var i = lid;
    while (i < D) { sm += SCR[X_OFF + i]; i += WG; }
    red[lid] = sm;
    reduce_sum_wg(lid);
    let mean = (red[0] + red[1]) / f32(D);
    // All threads must read the mean before red is reused for variance.
    workgroupBarrier();
    var sv = 0.0;
    i = lid;
    while (i < D) {
      let del = SCR[X_OFF + i] - mean;
      sv += del * del;
      i += WG;
    }
    red[lid] = sv;
    reduce_sum_wg(lid);
    let inv = inverseSqrt((red[0] + red[1]) / f32(D) + 1e-5);
    workgroupBarrier();
    i = lid;
    while (i < D) {
      src[i] = (SCR[X_OFF + i] - mean) * inv * SC[job.normOff + i] + SC[job.normOff + D + i];
      i += WG;
    }
    workgroupBarrier();
    return;
  }
  if ((job.flags & FLAG_SILU) == FLAG_SILU) {
    var i = lid;
    while (i < FF) {
      let v = SCR[FF1_OFF + i];
      src[i] = (v / (1.0 + exp(-v))) * SCR[FF3_OFF + i];
      i += WG;
    }
    workgroupBarrier();
    return;
  }
  if ((job.flags & FLAG_GELU) == FLAG_GELU) {
    var i = lid;
    while (i < job.cols) {
      let v = clamp(SCR[job.srcOff + i], -20.0, 20.0);
      let z = clamp(0.7978845834732056 * (v + 0.044715 * v * v * v), -8.0, 8.0);
      src[i] = 0.5 * v * (1.0 + tanh(z));
      i += WG;
    }
    workgroupBarrier();
    return;
  }
  var i = lid;
  while (i < job.cols) {
    src[i] = SCR[job.srcOff + i];
    i += WG;
  }
  workgroupBarrier();
}

fn write_row(row: u32, acc: f32) {
  var v = scale_acc(acc, row, job.scaleOff);
  if ((job.flags & FLAG_BIAS) == FLAG_BIAS) { v = v + SC[job.biasOff + row]; }
  if (job.addX == 1u) {
    SCR[X_OFF + row] = SCR[X_OFF + row] + v;
  } else {
    SCR[job.dstOff + row] = v;
  }
}

@compute @workgroup_size(${GEMV_WG})
fn coop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  prepare_src(lid);
  let row = wid.x * ROW_TILE + lid / K_THREADS;
  let kid = lid % K_THREADS;
  var acc = 0.0;
  if (row < job.rows) {
    acc = dot_slice(row, job.cols, job.packOff, job.scaleOff, kid, K_THREADS);
  }
  red[lid] = acc;
  workgroupBarrier();
  if (kid < 16u) { red[lid] += red[lid + 16u]; }
  workgroupBarrier();
  if (kid < 8u) { red[lid] += red[lid + 8u]; }
  workgroupBarrier();
  if (kid < 4u) { red[lid] += red[lid + 4u]; }
  workgroupBarrier();
  if (kid < 2u) { red[lid] += red[lid + 2u]; }
  workgroupBarrier();
  if (kid == 0u && row < job.rows) {
    write_row(row, red[lid] + red[lid + 1u]);
  }
}

@compute @workgroup_size(${GEMV_WG})
fn fat(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  prepare_src(lid);
  let row = gid.x;
  if (row >= job.rows) { return; }
  write_row(row, dot_full(row, job.cols, job.packOff, job.scaleOff));
}
`;

  const embed = `
${C}
${S}
${P}
@group(0) @binding(1) var<storage, read> SC: array<f32>;
@group(0) @binding(2) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<uniform> job: Job;
@group(0) @binding(5) var<storage, read_write> OUT: array<u32>;

@compute @workgroup_size(${GEMV_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  var tok = step.token;
  if (tok == CHAIN) { tok = OUT[step.slot - 1u]; }
  ${embedBody(kind)}
}

@compute @workgroup_size(${GEMV_WG})
fn gpt2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  var tok = step.token;
  if (tok == CHAIN) { tok = OUT[step.slot - 1u]; }
  ${embedBody(kind)}
  SCR[X_OFF + i] = SCR[X_OFF + i] + SC[job.normOff + step.pos * D + i];
}
`;

  const attn = `
${C}
${S}
@group(0) @binding(0) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(1) var<storage, read_write> KV: array<f32>;
@group(0) @binding(2) var<storage, read> ROPE: array<f32>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<uniform> job: Job;
var<workgroup> red: array<f32, 256>;

fn kv_elems() -> u32 { return NKV * MS * HD; }
fn k_off(li: u32, h: u32, t: u32, d: u32) -> u32 {
  return li * kv_elems() + (h * MS + t) * HD + d;
}
fn v_off(li: u32, h: u32, t: u32, d: u32) -> u32 {
  return NL * kv_elems() + k_off(0u, h, t, d) + li * kv_elems();
}
fn cos_at(pos: u32, d: u32) -> f32 { return ROPE[pos * RD + d]; }
fn sin_at(pos: u32, d: u32) -> f32 { return ROPE[MS * RD + pos * RD + d]; }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  let pos = step.pos;
  let li = job.rows;
  let half = RD / 2u;

  var e = lid;
  while (e < NKV * HD) {
    let h = e / HD;
    let d = e % HD;
    let mateD = select(d + half, d - half, d >= half);
    let k0 = SCR[QKV_OFF + D + e];
    let mate = SCR[QKV_OFF + D + h * HD + mateD];
    let rh = select(-mate, mate, d >= half);
    KV[k_off(li, h, pos, d)] = k0 * cos_at(pos, d) + rh * sin_at(pos, d);
    KV[v_off(li, h, pos, d)] = SCR[QKV_OFF + D + NKV * HD + e];
    e += 256u;
  }
  workgroupBarrier();

  e = lid;
  while (e < NH * HD) {
    let h = e / HD;
    let d = e % HD;
    let mateD = select(d + half, d - half, d >= half);
    let q0 = SCR[QKV_OFF + e];
    let mate = SCR[QKV_OFF + h * HD + mateD];
    let rh = select(-mate, mate, d >= half);
    SCR[ATT_OFF + e] = q0 * cos_at(pos, d) + rh * sin_at(pos, d);
    e += 256u;
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
    var dotv = 0.0;
    let qbase = ATT_OFF + h * HD;
    for (var d = 0u; d < HD; d++) {
      dotv += SCR[qbase + d] * KV[k_off(li, kvh, t, d)];
    }
    SCR[LOG_OFF + sidx] = clamp(dotv * scale, -80.0, 80.0);
    sidx += 256u;
  }
  workgroupBarrier();

  for (var h = 0u; h < NH; h++) {
    let soff = LOG_OFF + h * seq;
    var mx = -1e30;
    var t = lid;
    while (t < seq) { mx = max(mx, SCR[soff + t]); t += 256u; }
    red[lid] = mx;
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
    mx = max(red[0], red[1]);
    var sm = 0.0;
    t = lid;
    while (t < seq) {
      let ex = exp(SCR[soff + t] - mx);
      SCR[soff + t] = ex;
      sm += ex;
      t += 256u;
    }
    red[lid] = sm;
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
    let inv = 1.0 / (red[0] + red[1] + 1e-20);
    t = lid;
    while (t < seq) { SCR[soff + t] = SCR[soff + t] * inv; t += 256u; }
    workgroupBarrier();
  }

  e = lid;
  while (e < NH * HD) {
    let h = e / HD;
    let d = e % HD;
    let kvh = h / (NH / NKV);
    let soff = LOG_OFF + h * seq;
    var acc = 0.0;
    for (var t = 0u; t < seq; t++) { acc += SCR[soff + t] * KV[v_off(li, kvh, t, d)]; }
    SCR[ATT_OFF + e] = acc;
    e += 256u;
  }
}
`;

  const argmax = `
${C}
${S}
@group(0) @binding(0) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(2) var<uniform> step: Step;
var<workgroup> red: array<f32, 256>;
var<workgroup> redi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  var bi = 0u;
  var bv = -1e30;
  var i = lid;
  while (i < VOCAB) {
    let v = SCR[LOG_OFF + i];
    if (v > bv) { bv = v; bi = i; }
    i += 256u;
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
  if (lid == 0u) { OUT[step.slot] = redi[0]; }
}
`;

  return { gemv, embed, attn, argmax };
}

async function makePipe(device, code, entryPoint, layout, label) {
  const module = device.createShaderModule({ code, label });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errs = (info.messages || []).filter((m) => m.type === "error");
    if (errs.length) {
      throw new Error(
        `${label} ${errs.map((m) => `WGSL:${m.lineNum}:${m.linePos} ${m.message}`).join(" | ")}`.slice(0, 1200),
      );
    }
  }
  return device.createComputePipeline({
    label,
    layout,
    compute: { module, entryPoint },
  });
}

function putJob(view, index, fields) {
  const o = (index * JOB_STRIDE) / 4;
  view[o] = fields.rows;
  view[o + 1] = fields.cols;
  view[o + 2] = fields.packOff;
  view[o + 3] = fields.scaleOff || 0;
  view[o + 4] = fields.srcOff || 0;
  view[o + 5] = fields.dstOff || 0;
  view[o + 6] = fields.addX || 0;
  view[o + 7] = fields.flags || 0;
  view[o + 8] = fields.normOff || 0;
  view[o + 9] = fields.biasOff || 0;
}

function layerStride(kind) {
  return kind === "q4" || kind === "q8" ? 12 : 7;
}

function layerOff(kind, off, li) {
  const b = li * layerStride(kind);
  if (kind === "q4" || kind === "q8") {
    return {
      qkvP: off[b],
      qkvS: off[b + 1],
      projP: off[b + 2],
      projS: off[b + 3],
      w1P: off[b + 4],
      w1S: off[b + 5],
      w3P: off[b + 6],
      w3S: off[b + 7],
      w2P: off[b + 8],
      w2S: off[b + 9],
      n1: off[b + 10],
      n2: off[b + 11],
    };
  }
  return {
    qkvP: off[b],
    qkvS: 0,
    projP: off[b + 1],
    projS: 0,
    w1P: off[b + 2],
    w1S: 0,
    w3P: off[b + 3],
    w3S: 0,
    w2P: off[b + 4],
    w2S: 0,
    n1: off[b + 5],
    n2: off[b + 6],
  };
}

function headOff(kind, off, nl) {
  const s = layerStride(kind);
  if (kind === "q4" || kind === "q8") {
    return { lmP: off[nl * s], lmS: off[nl * s + 1], nf: off[nl * s + 2] };
  }
  return { lmP: off[nl * s], lmS: 0, nf: off[nl * s + 1] };
}

export async function initMetal(engine) {
  const d = engine.device;
  const c = engine.cfg;
  const kind = engine.kind;
  const dim = consts(c);
  const D = dim.D;
  const FF = dim.FF;
  const QKV = dim.QKV;
  const shaders = metalWgsl(c, kind);

  if (!engine.SC) {
    engine.SC = engine._alloc(256, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "SC-dummy");
  }

  const bglGemv = d.createBindGroupLayout({
    label: "gemv",
    entries: [
      { binding: 0, visibility: COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
    ],
  });
  const bglEmbed = d.createBindGroupLayout({
    label: "embed",
    entries: [
      { binding: 0, visibility: COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
      { binding: 4, visibility: COMPUTE, buffer: { type: "uniform", minBindingSize: 256 } },
      { binding: 5, visibility: COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const bglAttn = d.createBindGroupLayout({
    label: "attn",
    entries: [
      { binding: 0, visibility: COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
      { binding: 4, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
    ],
  });
  const bglArgmax = d.createBindGroupLayout({
    label: "argmax",
    entries: [
      { binding: 0, visibility: COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 256 } },
    ],
  });

  const layoutGemv = d.createPipelineLayout({ bindGroupLayouts: [bglGemv] });
  const layoutEmbed = d.createPipelineLayout({ bindGroupLayouts: [bglEmbed] });
  const layoutAttn = d.createPipelineLayout({ bindGroupLayouts: [bglAttn] });
  const layoutArgmax = d.createPipelineLayout({ bindGroupLayouts: [bglArgmax] });

  engine.pGemvCoop = await makePipe(d, shaders.gemv, "coop", layoutGemv, "gemv-coop");
  engine.pGemvFat = await makePipe(d, shaders.gemv, "fat", layoutGemv, "gemv-fat");
  engine.pEmbed = await makePipe(
    d,
    shaders.embed,
    engine.cfg.arch === "gpt2" ? "gpt2" : "main",
    layoutEmbed,
    "embed",
  );
  engine.pAttn = await makePipe(d, shaders.attn, "main", layoutAttn, "attn");
  engine.pArgmax = await makePipe(d, shaders.argmax, "main", layoutArgmax, "argmax");

  const nJobs = c.nLayers * 8 + 8;
  engine.jobBuf = engine._alloc(nJobs * JOB_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "jobs");
  engine.stepBuf = engine._alloc(METAL_MAX_TOK * JOB_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "steps");
  const jobs = new Uint32Array((nJobs * JOB_STRIDE) / 4);
  const off = engine._offHost;
  let ji = 0;
  const addJob = (fields) => {
    const id = ji++;
    putJob(jobs, id, fields);
    engine._jobFields = engine._jobFields || [];
    engine._jobFields[id] = { id, ...fields };
    return id;
  };

  const ATT = 2 * D + QKV;
  const FF1 = ATT + D;
  const LOG = FF1 + FF + FF;
  engine.arch = c.arch || "llama";

  if (c.arch === "gpt2") {
    const g2off = (li) => {
      const b = li * 14;
      return {
        qkvP: off[b],
        qkvS: off[b + 1],
        qkvB: off[b + 2],
        projP: off[b + 3],
        projS: off[b + 4],
        projB: off[b + 5],
        fcP: off[b + 6],
        fcS: off[b + 7],
        fcB: off[b + 8],
        mpP: off[b + 9],
        mpS: off[b + 10],
        mpB: off[b + 11],
        ln1: off[b + 12],
        ln2: off[b + 13],
      };
    };
    const hb = c.nLayers * 14;
    const lmP = off[hb];
    const lmS = off[hb + 1];
    const lnf = off[hb + 2];
    const wpe = off[hb + 3];
    engine.jEmbed = addJob({
      rows: D,
      cols: D,
      packOff: lmP,
      scaleOff: lmS,
      normOff: wpe,
    });
    engine.jLm = addJob({
      rows: c.vocabSize,
      cols: D,
      packOff: lmP,
      scaleOff: lmS,
      dstOff: LOG,
      flags: FLAG_LN,
      normOff: lnf,
    });
    engine.layerJobs = [];
    for (let li = 0; li < c.nLayers; li++) {
      const L = g2off(li);
      engine.layerJobs.push({
        qkv: addJob({
          rows: QKV,
          cols: D,
          packOff: L.qkvP,
          scaleOff: L.qkvS,
          dstOff: 2 * D,
          flags: FLAG_LN | FLAG_BIAS,
          normOff: L.ln1,
          biasOff: L.qkvB,
        }),
        attn: addJob({ rows: li }),
        proj: addJob({
          rows: D,
          cols: D,
          packOff: L.projP,
          scaleOff: L.projS,
          srcOff: ATT,
          addX: 1,
          flags: FLAG_BIAS,
          biasOff: L.projB,
        }),
        fc: addJob({
          rows: FF,
          cols: D,
          packOff: L.fcP,
          scaleOff: L.fcS,
          dstOff: FF1,
          flags: FLAG_LN | FLAG_BIAS,
          normOff: L.ln2,
          biasOff: L.fcB,
        }),
        w2: addJob({
          rows: D,
          cols: FF,
          packOff: L.mpP,
          scaleOff: L.mpS,
          srcOff: FF1,
          addX: 1,
          flags: FLAG_GELU | FLAG_BIAS,
          biasOff: L.mpB,
        }),
      });
    }
  } else {
  const head = headOff(kind, off, c.nLayers);

  engine.jEmbed = addJob({
    rows: D,
    cols: D,
    packOff: head.lmP,
    scaleOff: head.lmS,
    srcOff: 0,
    dstOff: 0,
    addX: 0,
  });
  engine.jLm = addJob({
    rows: c.vocabSize,
    cols: D,
    packOff: head.lmP,
    scaleOff: head.lmS,
    srcOff: 0,
    dstOff: LOG,
    addX: 0,
    flags: FLAG_RMS,
    normOff: head.nf,
  });
  engine.layerJobs = [];
  for (let li = 0; li < c.nLayers; li++) {
    const L = layerOff(kind, off, li);
    engine.layerJobs.push({
      qkv: addJob({
        rows: QKV,
        cols: D,
        packOff: L.qkvP,
        scaleOff: L.qkvS,
        srcOff: 0,
        dstOff: 2 * D,
        addX: 0,
        flags: FLAG_RMS,
        normOff: L.n1,
      }),
      attn: addJob({ rows: li, cols: 0, packOff: 0, scaleOff: 0, srcOff: 0, dstOff: 0, addX: 0 }),
      proj: addJob({
        rows: D,
        cols: D,
        packOff: L.projP,
        scaleOff: L.projS,
        srcOff: ATT,
        dstOff: 0,
        addX: 1,
      }),
      w13: addJob({
        rows: 2 * FF,
        cols: D,
        packOff: L.w1P,
        scaleOff: L.w1S,
        srcOff: 0,
        dstOff: FF1,
        addX: 0,
        flags: FLAG_RMS,
        normOff: L.n2,
      }),
      w2: addJob({
        rows: D,
        cols: FF,
        packOff: L.w2P,
        scaleOff: L.w2S,
        srcOff: FF1,
        dstOff: 0,
        addX: 1,
        flags: FLAG_SILU,
      }),
    });
  }
  }
  d.queue.writeBuffer(engine.jobBuf, 0, jobs);

  engine.bgGemv = d.createBindGroup({
    layout: bglGemv,
    entries: [
      { binding: 0, resource: { buffer: engine.W } },
      { binding: 1, resource: { buffer: engine.SC } },
      { binding: 2, resource: { buffer: engine.SCR } },
      { binding: 3, resource: { buffer: engine.jobBuf, size: JOB_STRIDE } },
    ],
  });
  engine.bgEmbed = d.createBindGroup({
    layout: bglEmbed,
    entries: [
      { binding: 0, resource: { buffer: engine.W } },
      { binding: 1, resource: { buffer: engine.SC } },
      { binding: 2, resource: { buffer: engine.SCR } },
      { binding: 3, resource: { buffer: engine.stepBuf, size: JOB_STRIDE } },
      { binding: 4, resource: { buffer: engine.jobBuf, offset: engine.jEmbed * JOB_STRIDE, size: JOB_STRIDE } },
      { binding: 5, resource: { buffer: engine.OUT } },
    ],
  });
  engine.bgAttn = d.createBindGroup({
    layout: bglAttn,
    entries: [
      { binding: 0, resource: { buffer: engine.SCR } },
      { binding: 1, resource: { buffer: engine.KV } },
      { binding: 2, resource: { buffer: engine.ROPE } },
      { binding: 3, resource: { buffer: engine.stepBuf, size: JOB_STRIDE } },
      { binding: 4, resource: { buffer: engine.jobBuf, size: JOB_STRIDE } },
    ],
  });
  engine.bgArgmax = d.createBindGroup({
    layout: bglArgmax,
    entries: [
      { binding: 0, resource: { buffer: engine.SCR } },
      { binding: 1, resource: { buffer: engine.OUT } },
      { binding: 2, resource: { buffer: engine.stepBuf, size: JOB_STRIDE } },
    ],
  });

  engine.useMetalSplit = true;
  engine.mode = `metal-${kind}-mwg`;
  engine.metalError = null;
}

function dispatchGemv(pass, engine, jobId, rows) {
  const fat = rows >= 4096;
  pass.setPipeline(fat ? engine.pGemvFat : engine.pGemvCoop);
  pass.setBindGroup(0, engine.bgGemv, [jobId * JOB_STRIDE]);
  if (fat) pass.dispatchWorkgroups(Math.ceil(rows / GEMV_WG));
  else pass.dispatchWorkgroups(Math.ceil(rows / ROW_TILE));
}

export function encodeMetalTokens(engine, pass, pos0, ids, { doLogits, logitsEach }) {
  const c = engine.cfg;
  const D = c.dModel;
  const QKV = D + 2 * c.nKvHeads * (D / c.nHeads);
  const n = Math.min(ids.length, METAL_MAX_TOK);
  const steps = new Uint32Array((n * JOB_STRIDE) / 4);
  for (let i = 0; i < n; i++) {
    const o = (i * JOB_STRIDE) / 4;
    steps[o] = pos0 + i;
    steps[o + 1] = ids[i] >>> 0;
    steps[o + 2] = pos0 + i + 1;
    steps[o + 3] = logitsEach ? i : 0;
  }
  engine.device.queue.writeBuffer(engine.stepBuf, 0, steps);

  for (let ti = 0; ti < n; ti++) {
    pass.setPipeline(engine.pEmbed);
    pass.setBindGroup(0, engine.bgEmbed, [ti * JOB_STRIDE]);
    pass.dispatchWorkgroups(Math.ceil(D / GEMV_WG));

    const nL = engine.debugMaxLayers != null ? engine.debugMaxLayers : c.nLayers;
    for (let li = 0; li < nL; li++) {
      const J = engine.layerJobs[li];
      if (!engine.debugNoAttn) {
        dispatchGemv(pass, engine, J.qkv, QKV);
        pass.setPipeline(engine.pAttn);
        pass.setBindGroup(0, engine.bgAttn, [ti * JOB_STRIDE, J.attn * JOB_STRIDE]);
        pass.dispatchWorkgroups(1);
        dispatchGemv(pass, engine, J.proj, D);
      }
      const mlpTo = engine.debugMlpTo;
      if (!engine.debugNoMlp && (mlpTo == null || li < mlpTo)) {
        if (J.fc != null) dispatchGemv(pass, engine, J.fc, c.dFf);
        else dispatchGemv(pass, engine, J.w13, 2 * c.dFf);
        dispatchGemv(pass, engine, J.w2, D);
      }
    }

    const last = ti === n - 1;
    const want = logitsEach || (last && doLogits);
    if (want) {
      dispatchGemv(pass, engine, engine.jLm, c.vocabSize);
      pass.setPipeline(engine.pArgmax);
      pass.setBindGroup(0, engine.bgArgmax, [ti * JOB_STRIDE]);
      pass.dispatchWorkgroups(1);
    }
  }
  return n;
}

export { initMetal as initMetalQ4 };
