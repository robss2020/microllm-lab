/** 1-WG fused GPT-2 / nanoGPT decoder (LayerNorm, GELU, learned WPE, MHA). Q4. */

export function fusedGpt2Wgsl(cfg) {
  const D = cfg.dModel;
  const FF = cfg.dFf;
  const HD = cfg.dModel / cfg.nHeads;
  const NH = cfg.nHeads;
  const QKV = 3 * D;
  const NL = cfg.nLayers;
  const VOCAB = cfg.vocabSize;
  const MS = cfg.maxSeqLen;
  return `
const D: u32 = ${D}u;
const FF: u32 = ${FF}u;
const HD: u32 = ${HD}u;
const NH: u32 = ${NH}u;
const QKV: u32 = ${QKV}u;
const NL: u32 = ${NL}u;
const VOCAB: u32 = ${VOCAB}u;
const MS: u32 = ${MS}u;
const WG: u32 = 256u;
const X_OFF: u32 = 0u;
const QKV_OFF: u32 = D;
const ATT_OFF: u32 = QKV_OFF + QKV;
const FF1_OFF: u32 = ATT_OFF + D;
const LOG_OFF: u32 = FF1_OFF + FF;
const LAYER: u32 = 14u;

struct Params { pos0: u32, n_tok: u32, do_logits: u32, logits_each: u32, ids: array<vec4<u32>, 64>, };

@group(0) @binding(0) var<storage, read> PACK: array<u32>;
@group(0) @binding(1) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(2) var<storage, read_write> KV: array<f32>;
@group(0) @binding(3) var<storage, read> ROPE: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(5) var<storage, read> OFF: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> SC: array<f32>;

var<workgroup> xn: array<f32, ${Math.max(D, FF)}>;
var<workgroup> red: array<f32, 256>;
var<workgroup> redi: array<u32, 256>;

fn kv_elems() -> u32 { return NH * MS * HD; }
fn k_off(li: u32, h: u32, t: u32, d: u32) -> u32 { return li * kv_elems() + (h * MS + t) * HD + d; }
fn v_off(li: u32, h: u32, t: u32, d: u32) -> u32 { return NL * kv_elems() + k_off(0u, h, t, d) + li * kv_elems(); }
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
fn q4nibs(word: u32) -> vec4<f32> {
  return vec4<f32>(f32(word & 15u), f32((word >> 4u) & 15u), f32((word >> 8u) & 15u), f32((word >> 12u) & 15u)) - vec4<f32>(8.0);
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
      s += dot(lo, vec4<f32>(xn[i], xn[i+1u], xn[i+2u], xn[i+3u]));
      s += dot(hi, vec4<f32>(xn[i+4u], xn[i+5u], xn[i+6u], xn[i+7u]));
    }
    acc += s * sc;
  }
  return acc;
}
fn load_w(packOff: u32, idx: u32) -> f32 {
  let word = PACK[packOff + idx / 8u];
  let nib = (word >> ((idx % 8u) * 4u)) & 15u;
  return f32(nib) - 8.0;
}
fn gelu(v: f32) -> f32 {
  let x = clamp(v, -20.0, 20.0);
  let z = clamp(0.7978845834732056 * (x + 0.044715 * x * x * x), -8.0, 8.0);
  return 0.5 * x * (1.0 + tanh(z));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  // Retain ROPE storage binding in pipeline layout for bind group compatibility
  if (ROPE[0] > 1e30) { return; }
  let nTok = params.n_tok;
  let lmP = OFF[NL * LAYER];
  let lmS = OFF[NL * LAYER + 1u];
  let lnf = OFF[NL * LAYER + 2u];
  let wpe = OFF[NL * LAYER + 3u];
  let ngD = D / 32u;
  let rowU = D / 8u;

  for (var ti = 0u; ti < nTok; ti++) {
    var tok = id_at(ti);
    if (tok == 0xFFFFFFFFu) { tok = OUT[ti - 1u]; }
    let pos = params.pos0 + ti;
    var i = lid;
    while (i < D) {
      let sc = SC[lmS + tok * ngD + i / 32u];
      SCR[X_OFF + i] = load_w(lmP, tok * D + i) * sc + SC[wpe + pos * D + i];
      i += WG;
    }
    workgroupBarrier();

    for (var li = 0u; li < NL; li++) {
      let b = li * LAYER;
      let qkvP = OFF[b]; let qkvS = OFF[b+1u]; let qkvB = OFF[b+2u];
      let projP = OFF[b+3u]; let projS = OFF[b+4u]; let projB = OFF[b+5u];
      let fcP = OFF[b+6u]; let fcS = OFF[b+7u]; let fcB = OFF[b+8u];
      let mpP = OFF[b+9u]; let mpS = OFF[b+10u]; let mpB = OFF[b+11u];
      let ln1 = OFF[b+12u]; let ln2 = OFF[b+13u];

      var ss = 0.0;
      i = lid;
      while (i < D) { ss += SCR[X_OFF + i]; i += WG; }
      red[lid] = ss;
      let mean = reduce_sum(lid) / f32(D);
      workgroupBarrier();
      var sv = 0.0;
      i = lid;
      while (i < D) { let del = SCR[X_OFF + i] - mean; sv += del * del; i += WG; }
      red[lid] = sv;
      let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-5);
      workgroupBarrier();
      i = lid;
      while (i < D) { xn[i] = (SCR[X_OFF + i] - mean) * inv * SC[ln1 + i] + SC[ln1 + D + i]; i += WG; }
      workgroupBarrier();

      var row = lid;
      while (row < QKV) {
        SCR[QKV_OFF + row] = q4dot_xn(row, D, qkvP, qkvS) + SC[qkvB + row];
        row += WG;
      }
      workgroupBarrier();

      var e = lid;
      while (e < NH * HD) {
        let h = e / HD; let d = e % HD;
        KV[k_off(li, h, pos, d)] = SCR[QKV_OFF + D + e];
        KV[v_off(li, h, pos, d)] = SCR[QKV_OFF + D + NH * HD + e];
        SCR[ATT_OFF + e] = SCR[QKV_OFF + e];
        e += WG;
      }
      workgroupBarrier();

      let seq = pos + 1u;
      let scale = inverseSqrt(f32(HD));
      var sidx = lid;
      while (sidx < NH * seq) {
        let h = sidx / seq; let t = sidx % seq;
        var dotv = 0.0;
        let qbase = ATT_OFF + h * HD;
        for (var d = 0u; d < HD; d++) { dotv += SCR[qbase + d] * KV[k_off(li, h, t, d)]; }
        SCR[LOG_OFF + sidx] = clamp(dotv * scale, -80.0, 80.0);
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
        workgroupBarrier();
        var sm = 0.0;
        t = lid;
        while (t < seq) {
          let ex = exp(SCR[soff + t] - mx);
          SCR[soff + t] = ex;
          sm += ex;
          t += WG;
        }
        red[lid] = sm;
        let invs = 1.0 / (reduce_sum(lid) + 1e-20);
        workgroupBarrier();
        t = lid;
        while (t < seq) { SCR[soff + t] = SCR[soff + t] * invs; t += WG; }
        workgroupBarrier();
      }
      e = lid;
      while (e < NH * HD) {
        let h = e / HD; let d = e % HD;
        let soff = LOG_OFF + h * seq;
        var acc = 0.0;
        for (var t = 0u; t < seq; t++) { acc += SCR[soff + t] * KV[v_off(li, h, t, d)]; }
        xn[e] = acc;
        e += WG;
      }
      workgroupBarrier();
      row = lid;
      while (row < D) {
        SCR[X_OFF + row] = SCR[X_OFF + row] + q4dot_xn(row, D, projP, projS) + SC[projB + row];
        row += WG;
      }
      workgroupBarrier();

      ss = 0.0;
      i = lid;
      while (i < D) { ss += SCR[X_OFF + i]; i += WG; }
      red[lid] = ss;
      let mean2 = reduce_sum(lid) / f32(D);
      workgroupBarrier();
      sv = 0.0;
      i = lid;
      while (i < D) { let del2 = SCR[X_OFF + i] - mean2; sv += del2 * del2; i += WG; }
      red[lid] = sv;
      let inv2 = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-5);
      workgroupBarrier();
      i = lid;
      while (i < D) { xn[i] = (SCR[X_OFF + i] - mean2) * inv2 * SC[ln2 + i] + SC[ln2 + D + i]; i += WG; }
      workgroupBarrier();
      row = lid;
      while (row < FF) {
        SCR[FF1_OFF + row] = gelu(q4dot_xn(row, D, fcP, fcS) + SC[fcB + row]);
        row += WG;
      }
      workgroupBarrier();
      i = lid;
      while (i < FF) { xn[i] = SCR[FF1_OFF + i]; i += WG; }
      workgroupBarrier();
      row = lid;
      while (row < D) {
        var s = 0.0;
        let ng = FF / 32u;
        let rU = FF / 8u;
        for (var g = 0u; g < ng; g++) {
          let sc = SC[mpS + row * ng + g];
          let base = mpP + row * rU + g * 4u;
          let xb = g * 32u;
          var p = 0.0;
          for (var w = 0u; w < 4u; w++) {
            let word = PACK[base + w];
            let ii = xb + w * 8u;
            p += dot(q4nibs(word), vec4<f32>(xn[ii], xn[ii+1u], xn[ii+2u], xn[ii+3u]));
            p += dot(q4nibs(word >> 16u), vec4<f32>(xn[ii+4u], xn[ii+5u], xn[ii+6u], xn[ii+7u]));
          }
          s += p * sc;
        }
        SCR[X_OFF + row] = SCR[X_OFF + row] + s + SC[mpB + row];
        row += WG;
      }
      workgroupBarrier();
    }

    let last = ti == nTok - 1u;
    let want = (params.logits_each == 1u) || (last && params.do_logits == 1u);
    if (want) {
      var ss = 0.0;
      i = lid;
      while (i < D) { ss += SCR[X_OFF + i]; i += WG; }
      red[lid] = ss;
      let mean = reduce_sum(lid) / f32(D);
      workgroupBarrier();
      var sv = 0.0;
      i = lid;
      while (i < D) { let del = SCR[X_OFF + i] - mean; sv += del * del; i += WG; }
      red[lid] = sv;
      let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-5);
      workgroupBarrier();
      i = lid;
      while (i < D) { xn[i] = (SCR[X_OFF + i] - mean) * inv * SC[lnf + i] + SC[lnf + D + i]; i += WG; }
      workgroupBarrier();
      var row = lid;
      while (row < VOCAB) {
        SCR[LOG_OFF + row] = q4dot_xn(row, D, lmP, lmS);
        row += WG;
      }
      workgroupBarrier();
      var bi = 0u;
      var bv = -1e30;
      i = lid;
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
        if (lid < stride && red[lid + stride] > red[lid]) {
          red[lid] = red[lid + stride];
          redi[lid] = redi[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
      }
      if (lid == 0u) { OUT[select(0u, ti, params.logits_each == 1u)] = redi[0]; }
    }
  }
}
`;
}
