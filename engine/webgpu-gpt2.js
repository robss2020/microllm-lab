/**
 * Dedicated WebGPU engine for GPT-2 (124M nanoGPT architecture).
 *
 * Implements:
 * - Genuine Q4 group-32 quantization for all 12 transformer layers & lm_head
 * - SmoothQuant channel scaling to eliminate activation outlier noise on lm_head
 * - Learned position embeddings (wpe) + token embeddings (wte / lm_head)
 * - LayerNorm with learnable scale and bias
 * - Multi-Head Attention (MHA) with full QKV projection, scaled dot-product attention
 * - MLP with 4x expansion, GELU activation, projection
 * - 3-gram loop blocking and repetition penalty on generated tokens for cohesive output
 */

import { f32View, u8View, scaleView } from "./weights.js";

function packU8ToU32(u8) {
  const pad = new Uint8Array(Math.ceil(u8.length / 4) * 4);
  pad.set(u8);
  return new Uint32Array(pad.buffer);
}

const WGSL_GPT2 = `
const D: u32 = 768u;
const NL: u32 = 12u;
const NH: u32 = 12u;
const HD: u32 = 64u;
const FF: u32 = 3072u;
const MS: u32 = 1024u;
const VOCAB: u32 = 50257u;

const X_OFF: u32 = 0u;
const XN_OFF: u32 = 768u;
const QKV_OFF: u32 = 1536u;
const ATT_OFF: u32 = 3840u;
const FF1_OFF: u32 = 4608u;
const SCORES_OFF: u32 = 7680u; // 12 * 1024 = 12288 floats
const LOGITS_OFF: u32 = 20000u; // 50257 floats

struct StepParams {
  pos: u32,
  token: u32,
  li_start: u32,
  li_count: u32,
};

@group(0) @binding(0) var<storage, read> PACK: array<u32>;
@group(0) @binding(1) var<storage, read> SC: array<f32>;
@group(0) @binding(2) var<storage, read_write> KV: array<f32>;
@group(0) @binding(3) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(4) var<storage, read> OFF: array<u32>;
@group(0) @binding(5) var<uniform> step: StepParams;
@group(0) @binding(8) var<storage, read_write> OUT: array<u32>;

var<workgroup> src: array<f32, 3072>;
var<workgroup> red: array<f32, 256>;

fn k_off(li: u32, t: u32, e: u32) -> u32 { return (li * MS + t) * D + e; }
fn v_off(li: u32, t: u32, e: u32) -> u32 { return NL * MS * D + (li * MS + t) * D + e; }

fn q4nibs(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(word & 15u),
    f32((word >> 4u) & 15u),
    f32((word >> 8u) & 15u),
    f32((word >> 12u) & 15u)
  ) - vec4<f32>(8.0);
}

fn q4dot_src(row: u32, cols: u32, packOff: u32, scaleOff: u32) -> f32 {
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
      s += dot(lo, vec4<f32>(src[i], src[i + 1u], src[i + 2u], src[i + 3u]));
      s += dot(hi, vec4<f32>(src[i + 4u], src[i + 5u], src[i + 6u], src[i + 7u]));
    }
    acc += s * sc;
  }
  return acc;
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

@compute @workgroup_size(256)
fn step_main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  let pos = step.pos;
  var tok = step.token;
  if (tok == 0xffffffffu && step.li_start > 0u) {
    tok = OUT[step.li_start - 1u];
  }
  let lmP = OFF[0];
  let lmS = OFF[1];
  let wpe = OFF[2];
  let lnf = OFF[3];
  let scFac = OFF[4];

  // 1. Embedding: wte is stored in lm_head (scaled by scale_factors)
  // We de-scale by scale_factors[i] to recover the true token embedding
  var i = lid;
  while (i < D) {
    let ng = D / 32u;
    let rowU = D / 8u;
    let g = i / 32u;
    let sc = SC[lmS + tok * ng + g];
    let word = PACK[lmP + tok * rowU + i / 8u];
    let nib = (word >> ((i % 8u) * 4u)) & 15u;
    let emb = ((f32(nib) - 8.0) * sc) / SC[scFac + i];
    SCR[X_OFF + i] = emb + SC[wpe + pos * D + i];
    i += 256u;
  }
  workgroupBarrier();

  // 2. Transformer layers (12 layers, each with stride 14 in OFF)
  for (var li = 0u; li < NL; li++) {
    let b = 5u + li * 14u;
    let qkvP = OFF[b + 0u];
    let qkvS = OFF[b + 1u];
    let qkvB = OFF[b + 2u];
    let projP = OFF[b + 3u];
    let projS = OFF[b + 4u];
    let projB = OFF[b + 5u];
    let fcP = OFF[b + 6u];
    let fcS = OFF[b + 7u];
    let fcB = OFF[b + 8u];
    let mpP = OFF[b + 9u];
    let mpS = OFF[b + 10u];
    let mpB = OFF[b + 11u];
    let ln1 = OFF[b + 12u];
    let ln2 = OFF[b + 13u];

    // LayerNorm 1
    var ss = 0.0;
    i = lid;
    while (i < D) { ss += SCR[X_OFF + i]; i += 256u; }
    red[lid] = ss;
    let mean = reduce_sum(lid) / f32(D);
    workgroupBarrier();
    var sv = 0.0;
    i = lid;
    while (i < D) { let del = SCR[X_OFF + i] - mean; sv += del * del; i += 256u; }
    red[lid] = sv;
    let inv = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-5);
    workgroupBarrier();
    i = lid;
    while (i < D) {
      src[i] = (SCR[X_OFF + i] - mean) * inv * SC[ln1 + i] + SC[ln1 + D + i];
      i += 256u;
    }
    workgroupBarrier();

    // QKV GEMV (Q4 group-32)
    var row = lid;
    while (row < 3u * D) {
      SCR[QKV_OFF + row] = q4dot_src(row, D, qkvP, qkvS) + SC[qkvB + row];
      row += 256u;
    }
    workgroupBarrier();

    // Store K, V and Q
    var e = lid;
    while (e < D) {
      SCR[ATT_OFF + e] = SCR[QKV_OFF + e];
      KV[k_off(li, pos, e)] = SCR[QKV_OFF + D + e];
      KV[v_off(li, pos, e)] = SCR[QKV_OFF + 2u * D + e];
      e += 256u;
    }
    workgroupBarrier();

    // Multi-head Attention
    let seq = pos + 1u;
    let scale = 0.125; // 1.0 / sqrt(64.0)

    // Compute scores Q * K
    var sidx = lid;
    while (sidx < NH * seq) {
      let h = sidx / seq;
      let t = sidx % seq;
      var dotv = 0.0;
      let qbase = ATT_OFF + h * HD;
      let kbase = (li * MS + t) * D + h * HD;
      for (var d = 0u; d < HD; d++) {
        dotv += SCR[qbase + d] * KV[kbase + d];
      }
      SCR[SCORES_OFF + sidx] = dotv * scale;
      sidx += 256u;
    }
    workgroupBarrier();

    // Softmax per head
    for (var h = 0u; h < NH; h++) {
      let base = SCORES_OFF + h * seq;
      var mx = -1e30;
      var t = lid;
      while (t < seq) { mx = max(mx, SCR[base + t]); t += 256u; }
      red[lid] = mx;
      workgroupBarrier();
      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { red[lid] = max(red[lid], red[lid + s]); }
        workgroupBarrier();
      }
      let hmax = red[0];
      workgroupBarrier();

      var sm = 0.0;
      t = lid;
      while (t < seq) {
        let ex = exp(SCR[base + t] - hmax);
        SCR[base + t] = ex;
        sm += ex;
        t += 256u;
      }
      red[lid] = sm;
      let sum = reduce_sum(lid);
      let inv_sum = 1.0 / (sum + 1e-20);
      workgroupBarrier();

      t = lid;
      while (t < seq) {
        SCR[base + t] = SCR[base + t] * inv_sum;
        t += 256u;
      }
      workgroupBarrier();
    }

    // Weighted sum of V into workgroup src
    e = lid;
    while (e < D) {
      let h = e / HD;
      let d = e % HD;
      let base = SCORES_OFF + h * seq;
      var acc = 0.0;
      for (var t = 0u; t < seq; t++) {
        acc += SCR[base + t] * KV[v_off(li, t, h * HD + d)];
      }
      src[e] = acc;
      e += 256u;
    }
    workgroupBarrier();

    // Attention proj GEMV (Q4 group-32) + residual add
    row = lid;
    while (row < D) {
      SCR[X_OFF + row] += q4dot_src(row, D, projP, projS) + SC[projB + row];
      row += 256u;
    }
    workgroupBarrier();

    // LayerNorm 2
    ss = 0.0;
    i = lid;
    while (i < D) { ss += SCR[X_OFF + i]; i += 256u; }
    red[lid] = ss;
    let mean2 = reduce_sum(lid) / f32(D);
    workgroupBarrier();
    sv = 0.0;
    i = lid;
    while (i < D) { let del2 = SCR[X_OFF + i] - mean2; sv += del2 * del2; i += 256u; }
    red[lid] = sv;
    let inv2 = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-5);
    workgroupBarrier();
    i = lid;
    while (i < D) {
      src[i] = (SCR[X_OFF + i] - mean2) * inv2 * SC[ln2 + i] + SC[ln2 + D + i];
      i += 256u;
    }
    workgroupBarrier();

    // FC GEMV (Q4 group-32) + GELU
    row = lid;
    while (row < FF) {
      let v = q4dot_src(row, D, fcP, fcS) + SC[fcB + row];
      let z = clamp(0.79788456 * (v + 0.044715 * v * v * v), -8.0, 8.0);
      SCR[FF1_OFF + row] = 0.5 * v * (1.0 + tanh(z));
      row += 256u;
    }
    workgroupBarrier();

    // Copy FF1 to workgroup src for MLP proj GEMV
    row = lid;
    while (row < FF) {
      src[row] = SCR[FF1_OFF + row];
      row += 256u;
    }
    workgroupBarrier();

    // MLP Proj GEMV (Q4 group-32) + residual add
    row = lid;
    while (row < D) {
      SCR[X_OFF + row] += q4dot_src(row, FF, mpP, mpS) + SC[mpB + row];
      row += 256u;
    }
    workgroupBarrier();
  }

  // 3. Final LayerNorm (with folded 1/scale_factors)
  var ss_f = 0.0;
  i = lid;
  while (i < D) { ss_f += SCR[X_OFF + i]; i += 256u; }
  red[lid] = ss_f;
  let mean_f = reduce_sum(lid) / f32(D);
  workgroupBarrier();
  var sv_f = 0.0;
  i = lid;
  while (i < D) { let del_f = SCR[X_OFF + i] - mean_f; sv_f += del_f * del_f; i += 256u; }
  red[lid] = sv_f;
  let inv_f = inverseSqrt(reduce_sum(lid) / f32(D) + 1e-5);
  workgroupBarrier();
  i = lid;
  while (i < D) {
    SCR[XN_OFF + i] = (SCR[X_OFF + i] - mean_f) * inv_f * SC[lnf + i] + SC[lnf + D + i];
    i += 256u;
  }
}

@compute @workgroup_size(256)
fn lm_head_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= VOCAB) { return; }
  let lmP = OFF[0];
  let lmS = OFF[1];
  let ng = D / 32u;
  let rowU = D / 8u;
  var acc = 0.0;
  for (var g = 0u; g < ng; g++) {
    let sc = SC[lmS + row * ng + g];
    let base = lmP + row * rowU + g * 4u;
    let xb = g * 32u;
    var s = 0.0;
    for (var w = 0u; w < 4u; w++) {
      let word = PACK[base + w];
      let i = xb + w * 8u;
      let lo = q4nibs(word);
      let hi = q4nibs(word >> 16u);
      s += dot(lo, vec4<f32>(SCR[XN_OFF + i], SCR[XN_OFF + i + 1u], SCR[XN_OFF + i + 2u], SCR[XN_OFF + i + 3u]));
      s += dot(hi, vec4<f32>(SCR[XN_OFF + i + 4u], SCR[XN_OFF + i + 5u], SCR[XN_OFF + i + 6u], SCR[XN_OFF + i + 7u]));
    }
    acc += s * sc;
  }
  SCR[LOGITS_OFF + row] = acc;
}

struct ArgmaxParams {
  slot: u32,
  hist_len: u32,
  no_repeat_ngram: u32,
  penalty: f32,
};

@group(0) @binding(6) var<uniform> argmax_params: ArgmaxParams;
@group(0) @binding(7) var<storage, read_write> HIST: array<u32>;

var<workgroup> redi: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax_main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  var bi = 0u;
  var bv = -1e30;
  var i = lid;

  let n_past = argmax_params.hist_len;
  let block_ngram = argmax_params.no_repeat_ngram;
  var t1 = 0u;
  var t2 = 0u;
  if (block_ngram == 3u && n_past >= 2u) {
    t1 = HIST[n_past - 2u];
    t2 = HIST[n_past - 1u];
  }

  while (i < VOCAB) {
    var v = SCR[LOGITS_OFF + i];

    // N-gram blocking: prevent repeating any 3-gram
    if (block_ngram == 3u && n_past >= 2u) {
      for (var h = 0u; h < n_past - 2u; h++) {
        if (HIST[h] == t1 && HIST[h + 1u] == t2 && HIST[h + 2u] == i) {
          v = -1e9;
          break;
        }
      }
    }

    // Repetition penalty on recently generated tokens (window 32)
    if (argmax_params.penalty > 1.0 && v > -1e8) {
      let win = select(0u, n_past - 32u, n_past > 32u);
      for (var h = win; h < n_past; h++) {
        if (HIST[h] == i) {
          if (v > 0.0) { v = v / argmax_params.penalty; }
          else { v = v * argmax_params.penalty; }
          break;
        }
      }
    }

    if (v > bv) { bv = v; bi = i; }
    i += 256u;
  }
  red[lid] = bv;
  redi[lid] = bi;
  workgroupBarrier();

  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      if (red[lid + s] > red[lid]) {
        red[lid] = red[lid + s];
        redi[lid] = redi[lid + s];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    OUT[argmax_params.slot] = redi[0];
    HIST[n_past] = redi[0];
  }
}
`;

export class GpuGpt2Engine {
  constructor(cpuModel, gpuInfo) {
    this.cpu = cpuModel;
    this.cfg = cpuModel.cfg;
    this.bundle = cpuModel.bundle;
    this.info = gpuInfo;
    this.device = gpuInfo.device;
    this.mode = "q4-gpt2";
    this.bytesAllocated = 0;
    this.cacheLen = 0;
    this.ready = false;
  }

  async init() {
    const d = this.device;
    const bundle = this.bundle;

    const packChunks = [];
    const scChunks = [];
    const offsets = [];
    let packU32 = 0;
    let scF = 0;

    const pushQ4 = (name) => {
      const packed = u8View(bundle, name);
      const scales = scaleView(bundle, name);
      const u32 = packU8ToU32(packed);
      offsets.push(packU32, scF);
      packChunks.push(u32);
      scChunks.push(new Float32Array(scales));
      packU32 += u32.length;
      scF += scales.length;
    };

    const pushF32 = (name) => {
      const f32 = new Float32Array(f32View(bundle, name));
      offsets.push(scF);
      scChunks.push(f32);
      scF += f32.length;
    };

    // OFF[0], OFF[1]: lm_head packOff, scaleOff
    pushQ4("lm_head");
    // OFF[2]: wpe
    pushF32("wpe");
    // OFF[3]: ln_f
    pushF32("ln_f");
    // OFF[4]: scale_factors
    pushF32("scale_factors");

    // Transformer layers: 14 offsets per layer
    for (let li = 0; li < 12; li++) {
      pushQ4(`blocks.${li}.attn.qkv`);        // +0, +1
      pushF32(`blocks.${li}.attn.qkv_bias`);   // +2
      pushQ4(`blocks.${li}.attn.proj`);       // +3, +4
      pushF32(`blocks.${li}.attn.proj_bias`);  // +5
      pushQ4(`blocks.${li}.mlp.fc`);          // +6, +7
      pushF32(`blocks.${li}.mlp.fc_bias`);     // +8
      pushQ4(`blocks.${li}.mlp.proj`);        // +9, +10
      pushF32(`blocks.${li}.mlp.proj_bias`);   // +11
      pushF32(`blocks.${li}.ln1`);            // +12
      pushF32(`blocks.${li}.ln2`);            // +13
    }

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.bW = d.createBuffer({ size: packU32 * 4, usage: storage, label: "GPT2_W" });
    this.bSC = d.createBuffer({ size: scF * 4, usage: storage, label: "GPT2_SC" });
    this.bOFF = d.createBuffer({ size: offsets.length * 4, usage: storage, label: "GPT2_OFF" });
    this.bKV = d.createBuffer({ size: 2 * 12 * 1024 * 768 * 4, usage: storage, label: "GPT2_KV" });
    this.bSCR = d.createBuffer({ size: (20000 + 50257 + 256) * 4, usage: storage, label: "GPT2_SCR" });
    this.bOUT = d.createBuffer({ size: 1024 * 4, usage: storage | GPUBufferUsage.COPY_SRC, label: "GPT2_OUT" });
    this.bHIST = d.createBuffer({ size: 2048 * 4, usage: storage | GPUBufferUsage.COPY_DST, label: "GPT2_HIST" });
    const MAX_STEPS = 64;
    this.bStep = d.createBuffer({ size: MAX_STEPS * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "GPT2_Step" });
    this.bArgmax = d.createBuffer({ size: MAX_STEPS * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "GPT2_Argmax" });
    this.bStage = d.createBuffer({ size: 1024 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "GPT2_Stage" });

    this.bytesAllocated =
      packU32 * 4 +
      scF * 4 +
      offsets.length * 4 +
      2 * 12 * 1024 * 768 * 4 +
      (20000 + 50257 + 256) * 4 +
      1024 * 4 * 2;

    let po = 0;
    for (const ch of packChunks) {
      d.queue.writeBuffer(this.bW, po, ch);
      po += ch.byteLength;
    }
    let so = 0;
    for (const ch of scChunks) {
      d.queue.writeBuffer(this.bSC, so, ch);
      so += ch.byteLength;
    }
    d.queue.writeBuffer(this.bOFF, 0, new Uint32Array(offsets));

    const mod = d.createShaderModule({ code: WGSL_GPT2, label: "GPT2_Module" });
    this.bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.pStep = d.createComputePipeline({ layout, compute: { module: mod, entryPoint: "step_main" } });
    this.pLmHead = d.createComputePipeline({ layout, compute: { module: mod, entryPoint: "lm_head_main" } });
    this.pArgmax = d.createComputePipeline({ layout, compute: { module: mod, entryPoint: "argmax_main" } });

    this.bg = d.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.bW } },
        { binding: 1, resource: { buffer: this.bSC } },
        { binding: 2, resource: { buffer: this.bKV } },
        { binding: 3, resource: { buffer: this.bSCR } },
        { binding: 4, resource: { buffer: this.bOFF } },
        { binding: 5, resource: { buffer: this.bStep, size: 256 } },
        { binding: 6, resource: { buffer: this.bArgmax, size: 256 } },
        { binding: 7, resource: { buffer: this.bHIST } },
        { binding: 8, resource: { buffer: this.bOUT } },
      ],
    });

    this.ready = true;
    this.cpu = null;
    this.bundle = null;
    return this;
  }

  async generate(ids, { maxNewTokens = 64, eosId = 50256, onToken = null } = {}) {
    const d = this.device;
    const t0 = performance.now();

    // Initialize history buffer with prompt tokens for context
    d.queue.writeBuffer(this.bHIST, 0, new Uint32Array(ids));

    // 1. Prefill
    for (let i = 0; i < ids.length; i++) {
      const isLast = i === ids.length - 1;
      d.queue.writeBuffer(this.bStep, 0, new Uint32Array([i, ids[i], 0, 12]));

      const enc = d.createCommandEncoder();
      const pass1 = enc.beginComputePass();
      pass1.setBindGroup(0, this.bg, [0, 0]);
      pass1.setPipeline(this.pStep);
      pass1.dispatchWorkgroups(1);
      pass1.end();

      if (isLast) {
        const pass2 = enc.beginComputePass();
        pass2.setBindGroup(0, this.bg, [0, 0]);
        pass2.setPipeline(this.pLmHead);
        pass2.dispatchWorkgroups(Math.ceil(50257 / 256));
        pass2.end();

        // For first generated token: no penalty on prompt tokens (penalty = 0.0, block = 0)
        d.queue.writeBuffer(this.bArgmax, 0, new Uint32Array([0, 0, 0, 0]));
        const pass3 = enc.beginComputePass();
        pass3.setBindGroup(0, this.bg, [0, 0]);
        pass3.setPipeline(this.pArgmax);
        pass3.dispatchWorkgroups(1);
        pass3.end();

        enc.copyBufferToBuffer(this.bOUT, 0, this.bStage, 0, 4);
      }
      d.queue.submit([enc.finish()]);
    }

    // Read first generated token
    await this.bStage.mapAsync(GPUMapMode.READ);
    let nextTok = new Uint32Array(this.bStage.getMappedRange().slice(0, 4))[0];
    this.bStage.unmap();

    const ttft = performance.now() - t0;
    const accepted = [nextTok];
    if (onToken) await onToken(nextTok, { phase: "prefill", ms: ttft });
    if (nextTok === eosId) {
      return {
        generatedIds: accepted,
        stopReason: "eos",
        ttftMs: ttft,
        totalMs: performance.now() - t0,
        mode: this.mode,
      };
    }

    // Reset history to start only with generated tokens for 3-gram blocking / penalty
    let histLen = 1;
    d.queue.writeBuffer(this.bHIST, 0, new Uint32Array([nextTok]));

    // 2. Autoregressive decode loop in chunks of up to 32 tokens
    let stepIdx = 1;
    while (stepIdx < maxNewTokens) {
      const remaining = maxNewTokens - stepIdx;
      const chunkSize = Math.min(32, remaining);
      if (ids.length + stepIdx - 1 + chunkSize > 1024) break;

      const stepData = new Uint32Array((chunkSize * 256) / 4);
      const argData = new Uint32Array((chunkSize * 256) / 4);
      for (let s = 0; s < chunkSize; s++) {
        const curStep = stepIdx + s;
        const pos = ids.length + curStep - 1;
        const so = (s * 256) / 4;
        stepData[so + 0] = pos;
        stepData[so + 1] = s === 0 ? nextTok : 0xffffffff;
        stepData[so + 2] = curStep; // li_start = curStep
        stepData[so + 3] = 12;

        argData[so + 0] = curStep;
        argData[so + 1] = histLen + s;
        argData[so + 2] = 3; // 3-gram repetition blocking
        new Float32Array(argData.buffer)[so + 3] = 1.15; // 1.15 repetition penalty on recent generated tokens
      }
      d.queue.writeBuffer(this.bStep, 0, stepData);
      d.queue.writeBuffer(this.bArgmax, 0, argData);

      const enc = d.createCommandEncoder();
      for (let s = 0; s < chunkSize; s++) {
        const off = s * 256;
        const pass1 = enc.beginComputePass();
        pass1.setBindGroup(0, this.bg, [off, off]);
        pass1.setPipeline(this.pStep);
        pass1.dispatchWorkgroups(1);
        pass1.end();

        const pass2 = enc.beginComputePass();
        pass2.setBindGroup(0, this.bg, [off, off]);
        pass2.setPipeline(this.pLmHead);
        pass2.dispatchWorkgroups(Math.ceil(50257 / 256));
        pass2.end();

        const pass3 = enc.beginComputePass();
        pass3.setBindGroup(0, this.bg, [off, off]);
        pass3.setPipeline(this.pArgmax);
        pass3.dispatchWorkgroups(1);
        pass3.end();
      }

      enc.copyBufferToBuffer(this.bOUT, stepIdx * 4, this.bStage, 0, chunkSize * 4);
      d.queue.submit([enc.finish()]);

      await this.bStage.mapAsync(GPUMapMode.READ);
      const chunkTokens = Array.from(new Uint32Array(this.bStage.getMappedRange().slice(0, chunkSize * 4)));
      this.bStage.unmap();

      let hitEos = false;
      for (let s = 0; s < chunkSize; s++) {
        const tok = chunkTokens[s];
        accepted.push(tok);
        nextTok = tok;
        histLen++;
        if (onToken) await onToken(tok, { phase: "decode" });
        if (tok === eosId) {
          hitEos = true;
          break;
        }
      }
      if (hitEos) break;
      stepIdx += chunkSize;
    }

    return {
      generatedIds: accepted,
      stopReason: accepted.at(-1) === eosId ? "eos" : "max_new_tokens",
      ttftMs: ttft,
      totalMs: performance.now() - t0,
      mode: this.mode,
    };
  }

  destroy() {
    try {
      this.bW?.destroy();
      this.bSC?.destroy();
      this.bOFF?.destroy();
      this.bKV?.destroy();
      this.bSCR?.destroy();
      this.bOUT?.destroy();
      this.bHIST?.destroy();
      this.bStep?.destroy();
      this.bArgmax?.destroy();
      this.bStage?.destroy();
    } catch {
      /* */
    }
  }
}

export async function createGpuGpt2Engine(weights, cfg) {
  const engine = new GpuGpt2Engine(weights, weights.gpuInfo);
  return await engine.init();
}
