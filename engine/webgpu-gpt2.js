/**
 * Dedicated WebGPU engine for GPT-2 (124M nanoGPT architecture).
 *
 * Implements:
 * - Learned position embeddings (wpe) + token embeddings (wte / lm_head)
 * - LayerNorm with learnable scale and bias
 * - Multi-Head Attention (MHA) with full QKV projection, scaled dot-product attention
 * - MLP with 4x expansion, GELU activation, projection
 * - F16 linear weights for PyTorch-exact numeric precision
 * - 3-gram loop blocking and repetition penalty on generated tokens for cohesive output
 */

import { f32View, u16View } from "./weights.js";

function packU16ToU32(u16) {
  const pad = new Uint16Array(Math.ceil(u16.length / 2) * 2);
  pad.set(u16);
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

var<workgroup> src: array<f32, 3072>;
var<workgroup> red: array<f32, 256>;

fn k_off(li: u32, t: u32, e: u32) -> u32 { return (li * MS + t) * D + e; }
fn v_off(li: u32, t: u32, e: u32) -> u32 { return NL * MS * D + (li * MS + t) * D + e; }

fn f16dot(row: u32, cols: u32, packOff: u32) -> f32 {
  var s = 0.0;
  let nW = cols / 2u;
  let base = packOff + row * nW;
  for (var w = 0u; w < nW; w++) {
    let word = PACK[base + w];
    let pair = unpack2x16float(word);
    let xb = w * 2u;
    s += pair.x * src[xb] + pair.y * src[xb + 1u];
  }
  return s;
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
  let tok = step.token;
  let lmP = OFF[0];
  let wpe = OFF[1];
  let lnf = OFF[2];

  // 1. Embedding
  var i = lid;
  while (i < D) {
    let word = PACK[lmP + (tok * D + i) / 2u];
    let pair = unpack2x16float(word);
    let emb = select(pair.x, pair.y, (i & 1u) == 1u);
    SCR[X_OFF + i] = emb + SC[wpe + pos * D + i];
    i += 256u;
  }
  workgroupBarrier();

  // 2. Transformer layers
  for (var li = 0u; li < NL; li++) {
    let b = 3u + li * 10u;
    let qkvP = OFF[b];
    let qkvB = OFF[b + 1u];
    let projP = OFF[b + 2u];
    let projB = OFF[b + 3u];
    let fcP = OFF[b + 4u];
    let fcB = OFF[b + 5u];
    let mpP = OFF[b + 6u];
    let mpB = OFF[b + 7u];
    let ln1 = OFF[b + 8u];
    let ln2 = OFF[b + 9u];

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

    // QKV GEMV
    var row = lid;
    while (row < 3u * D) {
      SCR[QKV_OFF + row] = f16dot(row, D, qkvP) + SC[qkvB + row];
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

    // Weighted sum of V
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

    // Attention proj GEMV + residual add
    row = lid;
    while (row < D) {
      SCR[X_OFF + row] += f16dot(row, D, projP) + SC[projB + row];
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

    // FC GEMV + GELU
    row = lid;
    while (row < FF) {
      let v = f16dot(row, D, fcP) + SC[fcB + row];
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

    // MLP Proj GEMV + residual add
    row = lid;
    while (row < D) {
      SCR[X_OFF + row] += f16dot(row, FF, mpP) + SC[mpB + row];
      row += 256u;
    }
    workgroupBarrier();
  }

  // 3. Final LayerNorm
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
  var s = 0.0;
  let nW = D / 2u;
  let base = lmP + row * nW;
  for (var w = 0u; w < nW; w++) {
    let word = PACK[base + w];
    let pair = unpack2x16float(word);
    let xb = w * 2u;
    s += pair.x * SCR[XN_OFF + xb] + pair.y * SCR[XN_OFF + xb + 1u];
  }
  SCR[LOGITS_OFF + row] = s;
}

struct ArgmaxParams {
  slot: u32,
  hist_len: u32,
  no_repeat_ngram: u32,
  penalty: f32,
};

@group(0) @binding(6) var<uniform> argmax_params: ArgmaxParams;
@group(0) @binding(7) var<storage, read_write> HIST: array<u32>;
@group(0) @binding(8) var<storage, read_write> OUT: array<u32>;

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
    this.mode = "f16-gpt2";
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

    const pushF16 = (name) => {
      const u32 = packU16ToU32(u16View(bundle, name));
      offsets.push(packU32);
      packChunks.push(u32);
      packU32 += u32.length;
    };
    const pushF32 = (name) => {
      const f32 = new Float32Array(f32View(bundle, name));
      offsets.push(scF);
      scChunks.push(f32);
      scF += f32.length;
    };

    pushF16("lm_head");
    pushF32("wpe");
    pushF32("ln_f");

    for (let li = 0; li < 12; li++) {
      pushF16(`blocks.${li}.attn.qkv`);
      pushF32(`blocks.${li}.attn.qkv_bias`);
      pushF16(`blocks.${li}.attn.proj`);
      pushF32(`blocks.${li}.attn.proj_bias`);
      pushF16(`blocks.${li}.mlp.fc`);
      pushF32(`blocks.${li}.mlp.fc_bias`);
      pushF16(`blocks.${li}.mlp.proj`);
      pushF32(`blocks.${li}.mlp.proj_bias`);
      pushF32(`blocks.${li}.ln1`);
      pushF32(`blocks.${li}.ln2`);
    }

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.bW = d.createBuffer({ size: packU32 * 4, usage: storage, label: "GPT2_W" });
    this.bSC = d.createBuffer({ size: scF * 4, usage: storage, label: "GPT2_SC" });
    this.bOFF = d.createBuffer({ size: offsets.length * 4, usage: storage, label: "GPT2_OFF" });
    this.bKV = d.createBuffer({ size: 2 * 12 * 1024 * 768 * 4, usage: storage, label: "GPT2_KV" });
    this.bSCR = d.createBuffer({ size: (20000 + 50257 + 256) * 4, usage: storage, label: "GPT2_SCR" });
    this.bOUT = d.createBuffer({ size: 1024 * 4, usage: storage | GPUBufferUsage.COPY_SRC, label: "GPT2_OUT" });
    this.bHIST = d.createBuffer({ size: 2048 * 4, usage: storage | GPUBufferUsage.COPY_DST, label: "GPT2_HIST" });
    this.bStep = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "GPT2_Step" });
    this.bArgmax = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "GPT2_Argmax" });
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
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
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
        { binding: 5, resource: { buffer: this.bStep } },
        { binding: 6, resource: { buffer: this.bArgmax } },
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
      pass1.setBindGroup(0, this.bg);
      pass1.setPipeline(this.pStep);
      pass1.dispatchWorkgroups(1);
      pass1.end();

      if (isLast) {
        const pass2 = enc.beginComputePass();
        pass2.setBindGroup(0, this.bg);
        pass2.setPipeline(this.pLmHead);
        pass2.dispatchWorkgroups(Math.ceil(50257 / 256));
        pass2.end();

        // For first generated token: no penalty on prompt tokens (penalty = 0.0, block = 0)
        d.queue.writeBuffer(this.bArgmax, 0, new Uint32Array([0, 0, 0, 0]));
        const pass3 = enc.beginComputePass();
        pass3.setBindGroup(0, this.bg);
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

    // 2. Autoregressive decode loop
    for (let stepIdx = 1; stepIdx < maxNewTokens; stepIdx++) {
      const pos = ids.length + stepIdx - 1;
      if (pos >= 1024) break;

      d.queue.writeBuffer(this.bStep, 0, new Uint32Array([pos, nextTok, 0, 12]));
      const uArg = new Uint32Array(4);
      uArg[0] = stepIdx;
      uArg[1] = histLen;
      uArg[2] = 3; // 3-gram repetition blocking
      new Float32Array(uArg.buffer)[3] = 1.15; // 1.15 repetition penalty on recent generated tokens
      d.queue.writeBuffer(this.bArgmax, 0, uArg);

      const enc = d.createCommandEncoder();
      const pass1 = enc.beginComputePass();
      pass1.setBindGroup(0, this.bg);
      pass1.setPipeline(this.pStep);
      pass1.dispatchWorkgroups(1);
      pass1.end();

      const pass2 = enc.beginComputePass();
      pass2.setBindGroup(0, this.bg);
      pass2.setPipeline(this.pLmHead);
      pass2.dispatchWorkgroups(Math.ceil(50257 / 256));
      pass2.end();

      const pass3 = enc.beginComputePass();
      pass3.setBindGroup(0, this.bg);
      pass3.setPipeline(this.pArgmax);
      pass3.dispatchWorkgroups(1);
      pass3.end();

      enc.copyBufferToBuffer(this.bOUT, stepIdx * 4, this.bStage, 0, 4);
      d.queue.submit([enc.finish()]);

      await this.bStage.mapAsync(GPUMapMode.READ);
      nextTok = new Uint32Array(this.bStage.getMappedRange().slice(0, 4))[0];
      this.bStage.unmap();

      accepted.push(nextTok);
      histLen++;
      if (onToken) await onToken(nextTok, { phase: "decode" });
      if (nextTok === eosId) break;
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
