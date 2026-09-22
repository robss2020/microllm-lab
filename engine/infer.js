/**
 * PetitGPT CPU inference. Matches the native PyTorch graph:
 * pre-norm RMSNorm, fused QKV GQA, Llama-style RoPE, SwiGLU, tied embeddings.
 */
import { SPECIAL } from "./tokenizer.js";
import { dequantizeToF32, f16ToF32, f32View, i8View, scaleView, u8View, u16View } from "./weights.js";

const EPS = 1e-6;

function gemvF32(m, n, w, wOff, x, y) {
  for (let i = 0; i < m; i++) {
    let s = 0;
    const row = wOff + i * n;
    let k = 0;
    const lim = n - 7;
    for (; k < lim; k += 8) {
      s += w[row + k] * x[k];
      s += w[row + k + 1] * x[k + 1];
      s += w[row + k + 2] * x[k + 2];
      s += w[row + k + 3] * x[k + 3];
      s += w[row + k + 4] * x[k + 4];
      s += w[row + k + 5] * x[k + 5];
      s += w[row + k + 6] * x[k + 6];
      s += w[row + k + 7] * x[k + 7];
    }
    for (; k < n; k++) s += w[row + k] * x[k];
    y[i] = s;
  }
}

function gemmF32(t, n, m, a, aOff, w, wOff, y) {
  // y[t, m] = a[t, n] @ W[m, n]^T   (W is [m,n] row-major)
  for (let i = 0; i < t; i++) {
    const xi = aOff + i * n;
    const yi = i * m;
    for (let j = 0; j < m; j++) {
      let s = 0;
      const row = wOff + j * n;
      let k = 0;
      const lim = n - 7;
      for (; k < lim; k += 8) {
        s += w[row + k] * a[xi + k];
        s += w[row + k + 1] * a[xi + k + 1];
        s += w[row + k + 2] * a[xi + k + 2];
        s += w[row + k + 3] * a[xi + k + 3];
        s += w[row + k + 4] * a[xi + k + 4];
        s += w[row + k + 5] * a[xi + k + 5];
        s += w[row + k + 6] * a[xi + k + 6];
        s += w[row + k + 7] * a[xi + k + 7];
      }
      for (; k < n; k++) s += w[row + k] * a[xi + k];
      y[yi + j] = s;
    }
  }
}

function gemvF16(m, n, u16, wOff, x, y) {
  for (let i = 0; i < m; i++) {
    let s = 0;
    const row = wOff + i * n;
    for (let k = 0; k < n; k++) s += f16ToF32(u16[row + k]) * x[k];
    y[i] = s;
  }
}

function gemvBf16(m, n, u16, wOff, x, y) {
  const conv = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < m; i++) {
    let s = 0;
    const row = wOff + i * n;
    for (let k = 0; k < n; k++) {
      conv.setUint32(0, u16[row + k] << 16, true);
      s += conv.getFloat32(0, true) * x[k];
    }
    y[i] = s;
  }
}

function gemvQ8(m, n, q, wOff, scale, scaleOff, x, y) {
  for (let i = 0; i < m; i++) {
    let s = 0;
    const row = wOff + i * n;
    for (let k = 0; k < n; k++) s += q[row + k] * x[k];
    y[i] = s * scale[scaleOff + i];
  }
}

function gemvQ4(m, n, packed, wOffBytes, scale, scaleOff, group, x, y) {
  const ng = n / group;
  const packedRow = n / 2;
  for (let i = 0; i < m; i++) {
    let acc = 0;
    const row = wOffBytes + i * packedRow;
    const so = scaleOff + i * ng;
    for (let g = 0; g < ng; g++) {
      const sc = scale[so + g];
      const base = row + g * (group / 2);
      let s = 0;
      for (let k = 0; k < group; k += 2) {
        const b = packed[base + k / 2];
        s += ((b & 15) - 8) * x[g * group + k];
        s += ((b >> 4) - 8) * x[g * group + k + 1];
      }
      acc += s * sc;
    }
    y[i] = acc;
  }
}

function gemmGeneric(model, name, t, x, y) {
  // x is [t, n], y is [t, m]
  if (t === 1) {
    model.gemv(name, x, y);
    return;
  }
  const tmp = new Float32Array(y.length / t);
  const n = x.length / t;
  const m = y.length / t;
  for (let i = 0; i < t; i++) {
    const xi = x.subarray(i * n, (i + 1) * n);
    model.gemv(name, xi, tmp);
    y.set(tmp, i * m);
  }
}

function rmsNorm(x, w, out, dim) {
  const tokens = x.length / dim;
  for (let t = 0; t < tokens; t++) {
    const off = t * dim;
    let ss = 0;
    for (let i = 0; i < dim; i++) ss += x[off + i] * x[off + i];
    const inv = 1 / Math.sqrt(ss / dim + EPS);
    for (let i = 0; i < dim; i++) out[off + i] = x[off + i] * inv * w[i];
  }
}

const LN_EPS = 1e-5;

function layerNorm(x, wb, out, dim) {
  const tokens = x.length / dim;
  for (let t = 0; t < tokens; t++) {
    const off = t * dim;
    let mean = 0;
    for (let i = 0; i < dim; i++) mean += x[off + i];
    mean /= dim;
    let v = 0;
    for (let i = 0; i < dim; i++) {
      const del = x[off + i] - mean;
      v += del * del;
    }
    const inv = 1 / Math.sqrt(v / dim + LN_EPS);
    for (let i = 0; i < dim; i++) {
      out[off + i] = (x[off + i] - mean) * inv * wb[i] + wb[dim + i];
    }
  }
}

function geluNewInPlace(a, n) {
  for (let i = 0; i < n; i++) {
    const v = a[i];
    a[i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
  }
}

function addBias(y, b, n) {
  if (!b) return;
  for (let i = 0; i < n; i++) y[i] += b[i];
}

function siluMul(a, b, n) {
  for (let i = 0; i < n; i++) {
    const v = a[i];
    a[i] = (v / (1 + Math.exp(-v))) * b[i];
  }
}

function buildRope(maxSeq, headDim, theta, pct) {
  let ropeDim = Math.floor(headDim * pct);
  ropeDim -= ropeDim % 2;
  const half = ropeDim / 2;
  const inv = new Float32Array(half);
  for (let i = 0; i < half; i++) inv[i] = 1 / theta ** ((2 * i) / ropeDim);
  const cos = new Float32Array(maxSeq * ropeDim);
  const sin = new Float32Array(maxSeq * ropeDim);
  for (let t = 0; t < maxSeq; t++) {
    for (let i = 0; i < half; i++) {
      const freq = t * inv[i];
      const c = Math.cos(freq);
      const s = Math.sin(freq);
      // cat([freqs, freqs]) layout
      cos[t * ropeDim + i] = c;
      cos[t * ropeDim + half + i] = c;
      sin[t * ropeDim + i] = s;
      sin[t * ropeDim + half + i] = s;
    }
  }
  return { cos, sin, ropeDim };
}

function applyRope(qOrK, nHeads, t, headDim, rope, offset) {
  const { cos, sin, ropeDim } = rope;
  const half = ropeDim / 2;
  for (let h = 0; h < nHeads; h++) {
    for (let ti = 0; ti < t; ti++) {
      const pos = offset + ti;
      const base = (h * t + ti) * headDim;
      const rbase = pos * ropeDim;
      // rotate_half on first ropeDim components
      const rot = new Float32Array(ropeDim);
      for (let i = 0; i < half; i++) rot[i] = -qOrK[base + half + i];
      for (let i = 0; i < half; i++) rot[half + i] = qOrK[base + i];
      for (let i = 0; i < ropeDim; i++) {
        qOrK[base + i] = qOrK[base + i] * cos[rbase + i] + rot[i] * sin[rbase + i];
      }
    }
  }
}

function softmaxInPlace(x, n) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > max) max = x[i];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[i] - max);
    x[i] = e;
    s += e;
  }
  const inv = 1 / s;
  for (let i = 0; i < n; i++) x[i] *= inv;
}

function argmax(logits) {
  let bi = 0;
  let bv = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bv) {
      bv = logits[i];
      bi = i;
    }
  }
  return bi;
}

function argmaxGpt2(logits, history = [], { noRepeatNgram = 3, penalty = 1.15 } = {}) {
  const n_past = history.length;
  let t1 = -1, t2 = -1;
  if (noRepeatNgram === 3 && n_past >= 2) {
    t1 = history[n_past - 2];
    t2 = history[n_past - 1];
  }
  let bestIdx = 0;
  let bestVal = -Infinity;
  const recentWin = Math.max(0, n_past - 32);

  for (let i = 0; i < logits.length; i++) {
    let v = logits[i];
    if (noRepeatNgram === 3 && n_past >= 2) {
      for (let h = 0; h < n_past - 2; h++) {
        if (history[h] === t1 && history[h + 1] === t2 && history[h + 2] === i) {
          v = -1e9;
          break;
        }
      }
    }
    if (penalty > 1.0 && v > -1e8) {
      for (let h = recentWin; h < n_past; h++) {
        if (history[h] === i) {
          v = v > 0 ? v / penalty : v * penalty;
          break;
        }
      }
    }
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export class PetitGPT {
  constructor(bundle) {
    this.bundle = bundle;
    this.cfg = bundle.cfg;
    this.kind = bundle.kind;
    const c = bundle.cfg;
    this.headDim = c.dModel / c.nHeads;
    this.kvDim = c.nKvHeads * this.headDim;
    this.rep = c.nHeads / c.nKvHeads;
    this.isGpt2 = c.arch === "gpt2";
    this.rope = this.isGpt2
      ? { cos: new Float32Array(0), sin: new Float32Array(0), ropeDim: 0 }
      : buildRope(c.maxSeqLen, this.headDim, c.ropeTheta, c.ropePct);
    this.norms = Object.create(null);
    this.weightCache = Object.create(null);
    for (const name of Object.keys(bundle.tensors)) {
      if (bundle.tensors[name].storage === "f32" && bundle.tensors[name].shape.length === 1) {
        this.norms[name] = f32View(bundle, name);
      }
    }
    this._bindGemv();
    this.resetCache();
  }

  _bindGemv() {
    const b = this.bundle;
    const kind = b.kind;
    this._w = Object.create(null);
    for (const [name, t] of Object.entries(b.tensors)) {
      if (t.shape.length !== 2) continue;
      if (t.storage === "f32") {
        this._w[name] = { mode: "f32", data: f32View(b, name) };
      } else if (t.storage === "f16") {
        this._w[name] = { mode: "f16", data: u16View(b, name) };
      } else if (t.storage === "bf16") {
        this._w[name] = { mode: "bf16", data: u16View(b, name) };
      } else if (t.storage === "q8") {
        this._w[name] = { mode: "q8", data: i8View(b, name), scale: scaleView(b, name) };
      } else if (t.storage === "q4") {
        this._w[name] = {
          mode: "q4",
          data: u8View(b, name),
          scale: scaleView(b, name),
          group: t.group || 32,
        };
      }
    }
  }

  gemv(name, x, y) {
    const w = this._w[name];
    const [m, n] = this.bundle.tensors[name].shape;
    if (w.mode === "f32") gemvF32(m, n, w.data, 0, x, y);
    else if (w.mode === "f16") gemvF16(m, n, w.data, 0, x, y);
    else if (w.mode === "bf16") gemvBf16(m, n, w.data, 0, x, y);
    else if (w.mode === "q8") gemvQ8(m, n, w.data, 0, w.scale, 0, x, y);
    else if (w.mode === "q4") gemvQ4(m, n, w.data, 0, w.scale, 0, w.group, x, y);
  }

  resetCache() {
    const { nLayers, nKvHeads, maxSeqLen } = this.cfg;
    const hd = this.headDim;
    this.kCache = [];
    this.vCache = [];
    for (let i = 0; i < nLayers; i++) {
      this.kCache.push(new Float32Array(nKvHeads * maxSeqLen * hd));
      this.vCache.push(new Float32Array(nKvHeads * maxSeqLen * hd));
    }
    this.cacheLen = 0;
  }

  embed(ids, out, pos0 = 0) {
    const { dModel } = this.cfg;
    const w = this._w.lm_head;
    for (let t = 0; t < ids.length; t++) {
      const id = ids[t];
      const dst = t * dModel;
      if (w.mode === "f32") {
        out.set(w.data.subarray(id * dModel, id * dModel + dModel), dst);
      } else if (w.mode === "f16") {
        for (let i = 0; i < dModel; i++) out[dst + i] = f16ToF32(w.data[id * dModel + i]);
      } else if (w.mode === "bf16") {
        const conv = new DataView(new ArrayBuffer(4));
        for (let i = 0; i < dModel; i++) {
          conv.setUint32(0, w.data[id * dModel + i] << 16, true);
          out[dst + i] = conv.getFloat32(0, true);
        }
      } else {
        // dequant one row
        const row = new Float32Array(dModel);
        this._embedRowQuant(id, row);
        out.set(row, dst);
      }
    }
    if (this.isGpt2) {
      const wpe = this._w.wpe.data;
      for (let t = 0; t < ids.length; t++) {
        const dst = t * dModel;
        const src = (pos0 + t) * dModel;
        for (let i = 0; i < dModel; i++) out[dst + i] += wpe[src + i];
      }
    }
  }

  _embedRowQuant(id, row) {
    const w = this._w.lm_head;
    const n = this.cfg.dModel;
    if (w.mode === "q8") {
      const sc = w.scale[id];
      const off = id * n;
      for (let i = 0; i < n; i++) row[i] = w.data[off + i] * sc;
    } else if (w.mode === "q4") {
      const group = w.group;
      const ng = n / group;
      const packedRow = n / 2;
      for (let g = 0; g < ng; g++) {
        const sc = w.scale[id * ng + g];
        const base = id * packedRow + g * (group / 2);
        for (let k = 0; k < group; k += 2) {
          const b = w.data[base + k / 2];
          row[g * group + k] = ((b & 15) - 8) * sc;
          row[g * group + k + 1] = ((b >> 4) - 8) * sc;
        }
      }
    }
  }

  attention(layer, x, t, pastLen) {
    const { nHeads, nKvHeads, dModel } = this.cfg;
    const hd = this.headDim;
    const kvDim = this.kvDim;
    const qkv = this._scratch.qkv;
    gemmGeneric(this, `blocks.${layer}.attn.qkv`, t, x, qkv);
    if (this.isGpt2) {
      const bias = this.norms[`blocks.${layer}.attn.qkv_bias`];
      const width = dModel + 2 * kvDim;
      for (let ti = 0; ti < t; ti++) addBias(qkv.subarray(ti * width, (ti + 1) * width), bias, width);
    }
    const q = this._scratch.q;
    const kNew = this._scratch.k;
    const vNew = this._scratch.v;
    for (let ti = 0; ti < t; ti++) {
      const src = ti * (dModel + 2 * kvDim);
      // store q as [nHeads, T, hd]
      for (let h = 0; h < nHeads; h++) {
        for (let d = 0; d < hd; d++) q[(h * t + ti) * hd + d] = qkv[src + h * hd + d];
      }
      for (let h = 0; h < nKvHeads; h++) {
        for (let d = 0; d < hd; d++) {
          kNew[(h * t + ti) * hd + d] = qkv[src + dModel + h * hd + d];
          vNew[(h * t + ti) * hd + d] = qkv[src + dModel + kvDim + h * hd + d];
        }
      }
    }
    if (!this.isGpt2) {
      applyRope(q, nHeads, t, hd, this.rope, pastLen);
      applyRope(kNew, nKvHeads, t, hd, this.rope, pastLen);
    }

    const kC = this.kCache[layer];
    const vC = this.vCache[layer];
    const seq = pastLen + t;
    for (let h = 0; h < nKvHeads; h++) {
      for (let ti = 0; ti < t; ti++) {
        const dst = (h * this.cfg.maxSeqLen + pastLen + ti) * hd;
        const src = (h * t + ti) * hd;
        kC.set(kNew.subarray(src, src + hd), dst);
        vC.set(vNew.subarray(src, src + hd), dst);
      }
    }

    const scale = 1 / Math.sqrt(hd);
    const y = this._scratch.attn;
    const scores = this._scratch.scores;
    const yHead = this._scratch.yhead;
    for (let h = 0; h < nHeads; h++) {
      const kvh = Math.floor(h / this.rep);
      for (let qi = 0; qi < t; qi++) {
        const qpos = pastLen + qi;
        const qoff = (h * t + qi) * hd;
        for (let kj = 0; kj < seq; kj++) {
          if (kj > qpos) {
            scores[kj] = -1e9;
            continue;
          }
          const koff = (kvh * this.cfg.maxSeqLen + kj) * hd;
          let dot = 0;
          for (let d = 0; d < hd; d++) dot += q[qoff + d] * kC[koff + d];
          scores[kj] = dot * scale;
        }
        softmaxInPlace(scores, seq);
        for (let d = 0; d < hd; d++) yHead[d] = 0;
        for (let kj = 0; kj < seq; kj++) {
          const a = scores[kj];
          if (a === 0) continue;
          const voff = (kvh * this.cfg.maxSeqLen + kj) * hd;
          for (let d = 0; d < hd; d++) yHead[d] += a * vC[voff + d];
        }
        y.set(yHead, (qi * nHeads + h) * hd);
      }
    }
    // y is [T, nHeads, hd] = [T, dModel]
    const projOut = this._scratch.proj;
    gemmGeneric(this, `blocks.${layer}.attn.proj`, t, y, projOut);
    if (this.isGpt2) {
      const bias = this.norms[`blocks.${layer}.attn.proj_bias`];
      for (let ti = 0; ti < t; ti++) addBias(projOut.subarray(ti * dModel, (ti + 1) * dModel), bias, dModel);
    }
    return projOut;
  }

  ffn(layer, x, t) {
    const { dFf, dModel } = this.cfg;
    const n = t * dFf;
    const u = this._scratch.ff1;
    const down = this._scratch.ff2;
    if (this.isGpt2) {
      gemmGeneric(this, `blocks.${layer}.mlp.fc`, t, x, u);
      const fcB = this.norms[`blocks.${layer}.mlp.fc_bias`];
      for (let ti = 0; ti < t; ti++) addBias(u.subarray(ti * dFf, (ti + 1) * dFf), fcB, dFf);
      geluNewInPlace(u, n);
      gemmGeneric(this, `blocks.${layer}.mlp.proj`, t, u, down);
      const pB = this.norms[`blocks.${layer}.mlp.proj_bias`];
      for (let ti = 0; ti < t; ti++) addBias(down.subarray(ti * dModel, (ti + 1) * dModel), pB, dModel);
      return down;
    }
    const v = this._scratch.ff3;
    gemmGeneric(this, `blocks.${layer}.mlp.w1`, t, x, u);
    gemmGeneric(this, `blocks.${layer}.mlp.w3`, t, x, v);
    siluMul(u, v, n);
    gemmGeneric(this, `blocks.${layer}.mlp.w2`, t, u, down);
    return down;
  }

  ensureScratch(t) {
    const { dModel, dFf, nHeads, maxSeqLen } = this.cfg;
    const kvDim = this.kvDim;
    const need = !this._scratch || this._scratch.t < t;
    if (!need) return;
    this._scratch = {
      t,
      x: new Float32Array(t * dModel),
      xn: new Float32Array(t * dModel),
      qkv: new Float32Array(t * (dModel + 2 * kvDim)),
      q: new Float32Array(nHeads * t * this.headDim),
      k: new Float32Array(this.cfg.nKvHeads * t * this.headDim),
      v: new Float32Array(this.cfg.nKvHeads * t * this.headDim),
      attn: new Float32Array(t * dModel),
      proj: new Float32Array(t * dModel),
      ff1: new Float32Array(t * dFf),
      ff3: new Float32Array(t * dFf),
      ff2: new Float32Array(t * dModel),
      scores: new Float32Array(maxSeqLen),
      yhead: new Float32Array(this.headDim),
      logits: new Float32Array(this.cfg.vocabSize),
    };
  }

  forwardPrompt(ids) {
    const t = ids.length;
    const { dModel, nLayers } = this.cfg;
    this.ensureScratch(t);
    const x = this._scratch.x;
    const xn = this._scratch.xn;
    this.embed(ids, x, 0);
    const pastLen = 0;
    for (let li = 0; li < nLayers; li++) {
      if (this.isGpt2) layerNorm(x, this.norms[`blocks.${li}.ln1`], xn, dModel);
      else rmsNorm(x, this.norms[`blocks.${li}.norm1`], xn, dModel);
      const att = this.attention(li, xn, t, pastLen);
      for (let i = 0; i < t * dModel; i++) x[i] += att[i];
      if (this.isGpt2) layerNorm(x, this.norms[`blocks.${li}.ln2`], xn, dModel);
      else rmsNorm(x, this.norms[`blocks.${li}.norm2`], xn, dModel);
      const ff = this.ffn(li, xn, t);
      for (let i = 0; i < t * dModel; i++) x[i] += ff[i];
    }
    this.cacheLen = t;
    if (this.isGpt2) layerNorm(x, this.norms.ln_f, xn, dModel);
    else rmsNorm(x, this.norms.norm_f, xn, dModel);
    const last = xn.subarray((t - 1) * dModel, t * dModel);
    this.gemv("lm_head", last, this._scratch.logits);
    return this._scratch.logits;
  }

  forwardDecode(tokenId) {
    const { dModel, nLayers } = this.cfg;
    this.ensureScratch(1);
    const x = this._scratch.x;
    const xn = this._scratch.xn;
    this.embed([tokenId], x, this.cacheLen);
    const pastLen = this.cacheLen;
    for (let li = 0; li < nLayers; li++) {
      if (this.isGpt2) layerNorm(x, this.norms[`blocks.${li}.ln1`], xn, dModel);
      else rmsNorm(x, this.norms[`blocks.${li}.norm1`], xn, dModel);
      const att = this.attention(li, xn, 1, pastLen);
      for (let i = 0; i < dModel; i++) x[i] += att[i];
      if (this.isGpt2) layerNorm(x, this.norms[`blocks.${li}.ln2`], xn, dModel);
      else rmsNorm(x, this.norms[`blocks.${li}.norm2`], xn, dModel);
      const ff = this.ffn(li, xn, 1);
      for (let i = 0; i < dModel; i++) x[i] += ff[i];
    }
    this.cacheLen = pastLen + 1;
    if (this.isGpt2) layerNorm(x, this.norms.ln_f, xn, dModel);
    else rmsNorm(x, this.norms.norm_f, xn, dModel);
    this.gemv("lm_head", xn, this._scratch.logits);
    return this._scratch.logits;
  }

  async generate(ids, { maxNewTokens = 64, eosId = SPECIAL.EOS, onToken = null } = {}) {
    this.resetCache();
    const t0 = performance.now();
    const logits = this.forwardPrompt(ids);
    let ttft = performance.now() - t0;
    const out = [];
    let next = this.isGpt2 ? argmaxGpt2(logits, []) : argmax(logits);
    out.push(next);
    if (onToken) await onToken(next, { phase: "prefill", ms: ttft });
    if (next === eosId) {
      return {
        generatedIds: out,
        stopReason: "eos",
        ttftMs: ttft,
        totalMs: performance.now() - t0,
      };
    }
    for (let i = 1; i < maxNewTokens; i++) {
      if (ids.length + out.length >= this.cfg.maxSeqLen) break;
      const stepLogits = this.forwardDecode(next);
      next = this.isGpt2 ? argmaxGpt2(stepLogits, out) : argmax(stepLogits);
      out.push(next);
      if (onToken) await onToken(next, { phase: "decode" });
      if (next === eosId) {
        return {
          generatedIds: out,
          stopReason: "eos",
          ttftMs: ttft,
          totalMs: performance.now() - t0,
        };
      }
    }
    return {
      generatedIds: out,
      stopReason: "max_new_tokens",
      ttftMs: ttft,
      totalMs: performance.now() - t0,
    };
  }
}

export { argmax, dequantizeToF32 };
