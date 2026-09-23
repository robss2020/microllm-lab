import { loadTokenizerFromJson, encodeMessages, SPECIAL } from "./tokenizer.js";
import { parsePgw } from "./weights.js";
import { PetitGPT } from "./infer.js";
import { initWasm, wasmAvailable, gemvF32Wasm } from "./wasm.js";
import { tryWebGpu, GpuPetitGPT } from "./webgpu.js";

let tokenizer = null;
let cpu = null;
let gpu = null;
let backend = "none";
let currentModelId = null;
let activeCard = {};
let wasmOk = false;
let gpuInfo = null;

function post(msg) {
  self.postMessage(msg);
}

function attachWasmGemv(model) {
  if (!wasmAvailable()) return false;
  const orig = model.gemv.bind(model);
  model.gemv = (name, x, y) => {
    const w = model._w[name];
    if (w?.mode === "f32") {
      const [m, n] = model.bundle.tensors[name].shape;
      if (gemvF32Wasm(m, n, w.data, x, y)) return;
    }
    orig(name, x, y);
  };
  return true;
}

async function loadModel(msg) {
  const modelId = msg.modelId || msg.dtype || msg.kind || "petitgpt";
  const preferGpu = msg.preferGpu !== false;
  const buffer = msg.buffer;
  activeCard = msg.card || {};
  currentModelId = modelId;

  if (msg.tokenizerSpec) {
    tokenizer = loadTokenizerFromJson(msg.tokenizerSpec);
  } else if (!tokenizer) {
    const spec = await fetch("../tokenizer.json").then((r) => r.json());
    tokenizer = loadTokenizerFromJson(spec);
  }

  if (gpu) {
    try {
      gpu.destroy();
    } catch {
      /* */
    }
    gpu = null;
  }
  cpu = null;
  backend = "none";

  const bundle = parsePgw(buffer);
  cpu = new PetitGPT(bundle);
  let usedWasm = false;
  try {
    if (!wasmOk) {
      await initWasm();
      wasmOk = true;
    }
    usedWasm = attachWasmGemv(cpu);
  } catch (e) {
    wasmOk = false;
    post({ type: "log", level: "warn", message: "WASM init failed: " + e.message });
  }

  let usedGpu = false;
  if (preferGpu) {
    try {
      const g = await tryWebGpu();
      if (g.ok) {
        gpuInfo = g;
        const engine = new GpuPetitGPT(cpu, g);
        try {
          await engine.init();
        } catch (err) {
          try {
            engine.destroy();
          } catch {
            /* */
          }
          throw err;
        }
        gpu = engine;
        usedGpu = true;
        backend = "webgpu";
        cpu.bundle = null;
        cpu._w = null;
        cpu.kCache = [];
        cpu.vCache = [];
        cpu._scratch = null;
        post({
          type: "log",
          message:
            "GPU path: weights + KV on adapter, one submit/token, vocab matvec only after prefill, argmax u32 readback.",
        });
      } else {
        post({ type: "log", level: "warn", message: "WebGPU unavailable: " + g.reason });
      }
    } catch (e) {
      post({ type: "log", level: "warn", message: "WebGPU init failed: " + e.message });
    }
  }

  if (!usedGpu) backend = usedWasm ? "wasm" : "cpu";

  const heap = globalThis.performance?.memory
    ? {
        jsHeap: performance.memory.usedJSHeapSize,
        jsHeapLimit: performance.memory.jsHeapSizeLimit,
      }
    : {};

  post({
    type: "loaded",
    modelId,
    dtype: modelId,
    backend,
    wasm: wasmOk,
    gpu: usedGpu
      ? {
          name: gpuInfo.name,
          vendor: gpuInfo.vendor,
          bytesAllocated: gpu.bytesAllocated,
          limits: gpuInfo.limits,
        }
      : null,
    cfg: cpu.cfg,
    ...heap,
  });
}

async function generate(prompt, maxNew, opts = {}) {
  if (!cpu && !gpu) throw new Error("no model loaded");
  const ids = encodeMessages(tokenizer, [{ role: "user", content: prompt }], activeCard);
  const engine = gpu || cpu;
  const t0 = performance.now();
  const eosId = opts.ignoreEos ? -1 : (activeCard.eosId ?? SPECIAL.EOS);
  const result = await engine.generate(ids, {
    maxNewTokens: maxNew,
    eosId,
    onToken: async (id, meta) => {
      const piece = tokenizer.decode([id], true);
      post({ type: "token", id, piece, ...meta });
    },
  });
  const text = tokenizer.decode(
    result.generatedIds.at(-1) === eosId ? result.generatedIds.slice(0, -1) : result.generatedIds,
    true
  );
  const raw = tokenizer.decode(result.generatedIds);
  const totalMs = performance.now() - t0;
  const n = result.generatedIds.length || 1;
  post({
    type: "done",
    text,
    raw,
    generatedIds: result.generatedIds,
    promptIds: ids,
    stopReason: result.stopReason,
    ttftMs: result.ttftMs,
    totalMs,
    tokPerS: n / (totalMs / 1000),
    backend,
    dtype: currentModelId,
  });
  return { text, raw, ...result, totalMs, backend, dtype: currentModelId };
}

self.onmessage = async (ev) => {
  const m = ev.data || {};
  try {
    if (m.cmd === "load") {
      await loadModel(m);
    } else if (m.cmd === "generate") {
      await generate(m.prompt, m.maxNewTokens || 64, m.opts || {});
    } else if (m.cmd === "unload") {
      if (gpu) gpu.destroy();
      gpu = null;
      cpu = null;
      backend = "none";
      post({ type: "unloaded" });
    } else if (m.cmd === "ping") {
      post({ type: "pong" });
    }
  } catch (e) {
    post({ type: "error", message: e.message || String(e), stack: e.stack });
  }
};
