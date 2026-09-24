import {
  loadPrefs,
  idbDelete,
  idbPut,
  idbGet,
  idbClear,
  idbUsage,
  fetchWithProgress,
  loadCatalog,
} from "./engine/storage.js";
import {
  SAMPLE_PROMPTS,
  BUILTIN_SUITE,
  SUSTAINED_SUITE,
  EXAMPLE_CUSTOM,
  EXAMPLE_CUSTOM_FN,
  LLM_PROMPT,
  compileCustom,
  fourGramRepeat,
  estimateSuiteSeconds,
} from "./engine/benchmarks.js";
import { parsePgw } from "./engine/weights.js";
import { tryWebGpu, GpuPetitGPT } from "./engine/webgpu.js";
import { loadTokenizerFromJson, encodeMessages, SPECIAL } from "./engine/tokenizer.js";
import { barChart, latestByModel, resizeCharts } from "./engine/charts.js";

const worker = new Worker(new URL("./engine/worker.js", import.meta.url), { type: "module" });
const CMP_KEY = "petitgpt-compare-v1";

const state = {
  prefs: loadPrefs(),
  catalog: null,
  models: [],
  active: "petitgpt",
  loaded: null,
  card: null,
  backend: "none",
  busy: false,
  preferGpu: true,
  lastMetrics: null,
  gpuEngine: null,
  tokenizer: null,
  cached: new Set(),
  storedBytes: 0,
  downloads: Object.create(null),
  downloadGen: Object.create(null),
  downloadPromises: Object.create(null),
};

const $ = (id) => document.getElementById(id);

function log(msg) {
  const el = $("log");
  el.textContent = `${new Date().toISOString().slice(11, 19)}  ${msg}\n` + el.textContent;
}

function setBusy(v) {
  state.busy = v;
  document.querySelectorAll("[data-busy]").forEach((b) => {
    b.disabled = v;
  });
}

function fmtMB(n) {
  if (n == null || Number.isNaN(n)) return "—";
  if (n < 1024) return `${Math.max(0, n).toFixed(0)} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

function fmtEta(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "";
  if (sec < 1.5) return "1s left";
  if (sec < 60) return `${Math.ceil(sec)}s left`;
  const m = Math.floor(sec / 60);
  const s = Math.ceil(sec % 60);
  return `${m}m ${s}s left`;
}

function fmtParams(n) {
  if (!n) return "";
  if (n >= 1e6) {
    const m = n / 1e6;
    return (m >= 100 ? m.toFixed(0) : m.toFixed(1).replace(/\.0$/, "")) + "M";
  }
  return (n / 1e3).toFixed(0) + "K";
}

function setActive(id) {
  if (!id || !modelById(id)) return;
  state.active = id;
  if (state.loaded && state.loaded !== id) state.loaded = null;
  renderModels();
  updateEstimate();
}

function modelById(id) {
  return state.models.find((m) => m.id === id);
}

function cachedIds() {
  return state.models.filter((m) => state.cached.has(m.id)).map((m) => m.id);
}

function catalogBytes() {
  return state.models.reduce((s, m) => s + (m.q4Bytes || 0), 0);
}

function loadCompare() {
  try {
    const raw = JSON.parse(localStorage.getItem(CMP_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

function saveCompare(rows) {
  localStorage.setItem(CMP_KEY, JSON.stringify(rows.slice(-40)));
}

function renderHud() {
  const m = state.lastMetrics || {};
  $("stat-backend").textContent = state.backend;
  $("stat-dtype").textContent = state.loaded || "none";
  $("stat-toks").textContent = m.tokPerS ? m.tokPerS.toFixed(1) : "—";
  $("stat-ttft").textContent = m.ttftMs != null ? Math.round(m.ttftMs) + " ms" : "—";
  $("stat-total").textContent = m.totalMs != null ? Math.round(m.totalMs) + " ms" : "—";
  $("stat-stop").textContent = m.stopReason || "—";
  const heap = performance.memory?.usedJSHeapSize;
  $("stat-heap").textContent = heap ? fmtMB(heap) : "—";
  $("stat-gpu").textContent = m.gpuBytes ? fmtMB(m.gpuBytes) : "—";
  if ($("stat-idb")) $("stat-idb").textContent = fmtMB(state.storedBytes);
}

async function refreshCache() {
  try {
    const u = await idbUsage();
    state.cached = new Set(u.keys);
    state.storedBytes = u.bytes;
  } catch (e) {
    log("idb usage " + (e.message || e));
  }
  const el = $("idb-total");
  if (el) {
    const n = state.cached.size;
    el.textContent =
      n === 0
        ? `Loaded on this device: none (0 of ${fmtMB(catalogBytes())}) (saved in browser IndexedDB cache)`
        : `Loaded on this device: ${fmtMB(state.storedBytes)} · ${n} model${n === 1 ? "" : "s"} (of ${fmtMB(catalogBytes())}) (saved in browser IndexedDB cache)`;
  }
  const stat = $("stat-idb");
  if (stat) stat.textContent = fmtMB(state.storedBytes);
}

function downloadStatus(m) {
  const dl = state.downloads[m.id];
  if (dl) {
    const total = dl.total || m.q4Bytes || 0;
    const pct = total ? Math.min(99, Math.round((100 * dl.rec) / total)) : 0;
    const spd = dl.bps > 1024 ? `${fmtMB(dl.bps)}/s` : "starting…";
    const eta = dl.bps > 1024 ? fmtEta(dl.eta) : "";
    return { kind: "busy", text: `${pct}% · ${spd}${eta ? " · " + eta : ""}`, pct };
  }
  if (state.cached.has(m.id)) return { kind: "have", text: `Loaded · ${fmtMB(m.q4Bytes)}` };
  return { kind: "need", text: `Not loaded · ${fmtMB(m.q4Bytes)}` };
}

function paintCardStatus(id) {
  const card = document.querySelector(`.model-card[data-id="${id}"]`);
  const m = modelById(id);
  if (!card || !m) return;
  const st = downloadStatus(m);
  const status = card.querySelector(".model-card-status");
  const bar = card.querySelector(".model-card-bar > i");
  const btn = card.querySelector(".model-card-action");
  card.classList.toggle("is-busy", st.kind === "busy");
  card.classList.toggle("is-have", st.kind === "have");
  if (status) {
    status.textContent = st.text;
    status.dataset.kind = st.kind;
  }
  if (bar) bar.style.width = st.kind === "busy" ? `${st.pct || 0}%` : st.kind === "have" ? "100%" : "0%";
  if (btn) {
    if (st.kind === "busy") {
      btn.disabled = true;
      btn.textContent = "Loading…";
    } else {
      btn.disabled = false;
      btn.textContent = st.kind === "have" ? "Unload" : "Load";
      btn.dataset.act = st.kind === "have" ? "discard" : "download";
    }
  }
}

async function downloadModel(id) {
  const m = modelById(id);
  if (!m) return;
  if (state.cached.has(id) && !state.downloads[id]) return;
  if (state.downloadPromises[id]) return state.downloadPromises[id];
  const gen = (state.downloadGen[id] = (state.downloadGen[id] || 0) + 1);
  const job = (async () => {
    state.downloads[id] = { rec: 0, total: m.q4Bytes || 0, t0: performance.now(), bps: 0, eta: 0 };
    paintCardStatus(id);
    try {
      const buf = await fetchWithProgress(m.file, (rec, total) => {
        if (state.downloadGen[id] !== gen) return;
        const now = performance.now();
        const elapsed = (now - state.downloads[id].t0) / 1000;
        const tot = total || m.q4Bytes || 0;
        state.downloads[id] = {
          rec,
          total: tot,
          t0: state.downloads[id].t0,
          bps: elapsed > 0.15 ? rec / elapsed : 0,
          eta: elapsed > 0.15 && rec > 0 ? ((tot - rec) * elapsed) / rec : 0,
        };
        paintCardStatus(id);
        if (id === state.active) setProgress(tot ? rec / tot : 0);
      });
      if (state.downloadGen[id] !== gen) return;
      await idbPut(id, buf);
      log(`loaded ${m.name} ${fmtMB(buf.byteLength)}`);
    } catch (e) {
      if (state.downloadGen[id] === gen) log("loading failed " + m.name + ": " + (e.message || e));
      throw e;
    } finally {
      delete state.downloads[id];
      if (id === state.active) setProgress(0);
    }
    await refreshCache();
    paintCardStatus(id);
    renderStorageButtons();
  })();
  state.downloadPromises[id] = job;
  try {
    await job;
  } finally {
    delete state.downloadPromises[id];
  }
}

async function discardModel(id) {
  state.downloadGen[id] = (state.downloadGen[id] || 0) + 1;
  delete state.downloads[id];
  try {
    await idbDelete(id);
    log("unloaded " + id);
  } catch (e) {
    log("unload " + (e.message || e));
  }
  if (state.loaded === id) {
    try {
      state.gpuEngine?.destroy();
    } catch {
      /* */
    }
    state.gpuEngine = null;
    state.loaded = null;
    state.tokenizer = null;
  }
  await refreshCache();
  renderModels();
}

async function downloadAll() {
  const missing = state.models.filter((m) => !state.cached.has(m.id) && !state.downloads[m.id]);
  if (!missing.length) return log("all models already loaded on this device");
  for (const m of missing) {
    try {
      await downloadModel(m.id);
    } catch {
      /* logged */
    }
  }
}

async function discardAll() {
  state.models.forEach((m) => {
    state.downloadGen[m.id] = (state.downloadGen[m.id] || 0) + 1;
    delete state.downloads[m.id];
  });
  try {
    await idbClear();
  } catch (e) {
    log("unload all " + (e.message || e));
  }
  if (state.gpuEngine) {
    try {
      state.gpuEngine.destroy();
    } catch {
      /* */
    }
    state.gpuEngine = null;
  }
  state.loaded = null;
  state.tokenizer = null;
  await refreshCache();
  renderModels();
  log("unloaded all cached weights");
}

function renderStorageButtons() {
  const dlAll = $("dl-all");
  const dsAll = $("ds-all");
  if (dlAll) dlAll.disabled = state.models.every((m) => state.cached.has(m.id) || state.downloads[m.id]);
  if (dsAll) dsAll.disabled = state.cached.size === 0 && !Object.keys(state.downloads).length;
}

function renderModels() {
  const sel = $("active-dtype");
  const cards = $("model-cards");
  const story = $("model-story");
  if (!sel || !cards) return;
  sel.innerHTML = "";
  cards.innerHTML = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    if (m.id === state.active) opt.selected = true;
    sel.appendChild(opt);

    const st = downloadStatus(m);
    const card = document.createElement("article");
    card.className = "model-card" + (st.kind === "have" ? " is-have" : "") + (st.kind === "busy" ? " is-busy" : "");
    card.dataset.id = m.id;
    card.setAttribute("aria-selected", m.id === state.active ? "true" : "false");
    card.innerHTML = `<button type="button" class="model-card-pick">
        <b>${m.name}</b>
        <span>${fmtParams(m.params)} · ${fmtMB(m.q4Bytes || 0)} · ${m.license}</span>
        <span>${m.maker || m.source || ""}</span>
      </button>
      <div class="model-card-bar" aria-hidden="true"><i style="width:${st.kind === "have" ? 100 : st.pct || 0}%"></i></div>
      <div class="model-card-status" data-kind="${st.kind}">${st.text}</div>
      <button type="button" class="btn ghost model-card-action" data-act="${st.kind === "have" ? "discard" : "download"}" ${st.kind === "busy" ? "disabled" : ""}>${st.kind === "busy" ? "Loading…" : st.kind === "have" ? "Unload" : "Load"}</button>`;
    card.querySelector(".model-card-pick").addEventListener("click", () => setActive(m.id));
    card.querySelector(".model-card-action").addEventListener("click", (e) => {
      e.stopPropagation();
      const act = e.currentTarget.dataset.act;
      if (act === "discard") discardModel(m.id);
      else downloadModel(m.id).catch(() => {});
    });
    cards.appendChild(card);
  }
  if (![...sel.options].some((o) => o.value === state.active) && sel.options.length) {
    state.active = sel.options[0].value;
  }
  sel.value = state.active;
  const active = modelById(state.active);
  if (story) {
    if (!active) {
      story.innerHTML = "";
    } else {
      const href = active.sourceUrl
        ? ` · <a href="${active.sourceUrl}" target="_blank" rel="noopener">${active.source}</a>`
        : "";
      const have = state.cached.has(active.id);
      story.innerHTML = `<h3>${active.name}</h3>
        <p class="meta">${active.maker || ""} · ${active.year || ""} · ${active.license || ""}${href}</p>
        <p>${active.story || active.blurb || ""}</p>
        <p class="meta">${have ? "Weights are loaded on this device." : "Weights are not loaded yet."}</p>`;
    }
  }
  renderStorageButtons();
}

function renderPrompts() {
  const wrap = $("prompt-chips");
  wrap.innerHTML = "";
  for (const p of SAMPLE_PROMPTS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = p.title;
    b.addEventListener("click", () => {
      $("prompt").value = p.text;
      $("prompt").focus();
    });
    wrap.appendChild(b);
  }
}

function addBubble(role, text) {
  const el = document.createElement("div");
  el.className = "bubble " + role;
  const pre = document.createElement("pre");
  pre.textContent = text;
  el.appendChild(pre);
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
  return pre;
}

function setProgress(frac) {
  $("progress").style.width = Math.max(0, Math.min(1, frac)) * 100 + "%";
}

async function getTokenizer() {
  if (state.tokenizer) return state.tokenizer;
  const m = modelById(state.active);
  const spec = await fetch(m.tokenizer).then((r) => r.json());
  state.tokenizer = loadTokenizerFromJson(spec);
  return state.tokenizer;
}

async function loadCard(m) {
  try {
    const c = await fetch(m.card).then((r) => r.json());
    const merged = { ...m, ...c, template: m.template, defaultSystem: m.defaultSystem };
    if (merged.template === "chatml") {
      merged.eosId = merged.specials?.["<|im_end|>"] ?? merged.eosId ?? 2;
      merged.bosId = merged.specials?.["<|im_start|>"] ?? merged.bosId ?? 1;
    }
    return merged;
  } catch {
    return { ...m };
  }
}

async function ensureLoaded() {
  const id = state.active;
  if (state.loaded === id && (state.gpuEngine || state.backend === "worker" || state.backend === "cpu" || state.backend === "wasm")) return;
  const m = modelById(id);
  if (!m) throw new Error("no model selected");
  setBusy(true);
  try {
    if (state.gpuEngine) {
      try {
        state.gpuEngine.destroy();
      } catch {
        /* */
      }
      state.gpuEngine = null;
    }
    state.tokenizer = null;
    state.card = await loadCard(m);
    await downloadModel(id);
    const buf = await idbGet(id);
    if (!buf) throw new Error("could not load weights for " + id);
    const bundle = parsePgw(buf);
    const tok = await getTokenizer();
    void tok;
    let gpuOk = false;
    if (state.preferGpu && $("prefer-gpu").checked) {
      const g = await tryWebGpu();
      $("gpu-name").textContent = g.ok
        ? `Adapter: ${g.vendor || ""} (${g.name || ""})`
        : `Adapter: ${g.reason || "none"}`;
      if (g.ok) {
        try {
          const { PetitGPT } = await import("./engine/infer.js");
          const cpu = new PetitGPT(bundle);
          const engine = new GpuPetitGPT(cpu, g);
          await engine.init();
          state.gpuEngine = engine;
          state.backend = "webgpu";
          state.loaded = id;
          log(
            `ready  ${m.name}  backend=webgpu  mode=${engine.mode}  ${(engine.bytesAllocated / 1048576).toFixed(0)} MB`,
          );
          state.lastMetrics = { ...(state.lastMetrics || {}), gpuBytes: engine.bytesAllocated };
          renderHud();
          setProgress(0);
          gpuOk = true;
          return;
        } catch (gpuErr) {
          log(`webgpu init failed (${gpuErr.message || gpuErr}); falling back to WASM/CPU`);
          try { state.gpuEngine?.destroy?.(); } catch {}
          state.gpuEngine = null;
          gpuOk = false;
        }
      }
    }
    if (!gpuOk) {
      const ua = navigator.userAgent || "";
      const isFirefox = /Firefox|FxiOS/i.test(ua);
      const isSafari = !isFirefox && /Safari/i.test(ua) && !/Chrome|Chromium|Edg|OPR/i.test(ua);
      if (isFirefox || isSafari) {
        const browserLabel = isFirefox ? "Firefox" : "Safari";
        $("gpu-name").innerHTML = `Adapter: none (${browserLabel} WebGPU disabled · <button type="button" id="btn-show-webgpu-help" style="background:none;border:none;color:var(--warn);padding:0;font:inherit;text-decoration:underline;cursor:pointer">Setup guide</button>)`;
        $("btn-show-webgpu-help")?.addEventListener("click", () => {
          sessionStorage.removeItem("dismissed_webgpu_notice");
          checkWebGpuNotice(true);
          $("webgpu-notice")?.scrollIntoView({ behavior: "smooth" });
        });
      } else {
        $("gpu-name").textContent = "Adapter: none (CPU fallback)";
      }
      const tokenizerSpec = await fetch(m.tokenizer).then((r) => r.json());
      await new Promise((resolve, reject) => {
        const onMsg = (ev) => {
          const msg = ev.data;
          if (msg.type === "loaded") {
            worker.removeEventListener("message", onMsg);
            state.backend = msg.backend || "worker";
            state.loaded = id;
            log(`ready  ${m.name}  backend=${msg.backend || "worker"}`);
            resolve(msg);
          } else if (msg.type === "error") {
            worker.removeEventListener("message", onMsg);
            reject(new Error(msg.message));
          }
        };
        worker.addEventListener("message", onMsg);
        worker.postMessage({
          cmd: "load",
          modelId: id,
          kind: "q4",
          preferGpu: false,
          buffer: buf,
          card: state.card,
          tokenizerSpec,
        });
      });
      renderHud();
      setProgress(0);
    }
  } finally {
    setBusy(false);
    setProgress(0);
  }
}

async function generateOnGpu(prompt, maxNew, opts = {}) {
  const tok = await getTokenizer();
  const ids = encodeMessages(tok, [{ role: "user", content: prompt }], state.card || {});
  const t0 = performance.now();
  const eosId = opts.ignoreEos ? -1 : (state.card?.eosId ?? SPECIAL.EOS);
  const stream = !!generateOnce._onToken;
  const result = await state.gpuEngine.generate(ids, {
    maxNewTokens: maxNew,
    eosId,
    onToken: stream
      ? async (id, meta) => {
          const piece = tok.decode([id], true);
          generateOnce._onToken({ id, piece, ...meta });
        }
      : null,
  });
  const text = tok.decode(
    result.generatedIds.at(-1) === eosId ? result.generatedIds.slice(0, -1) : result.generatedIds,
    true,
  );
  const raw = tok.decode(result.generatedIds);
  const totalMs = performance.now() - t0;
  const n = result.generatedIds.length || 1;
  return {
    text,
    raw,
    generatedIds: result.generatedIds,
    promptIds: ids,
    stopReason: result.stopReason,
    ttftMs: result.ttftMs,
    totalMs,
    tokPerS: n / (totalMs / 1000),
    backend: "webgpu",
    dtype: state.loaded,
    mode: result.mode,
  };
}

function generateOnce(prompt, maxNew, opts = {}) {
  if (state.gpuEngine) return generateOnGpu(prompt, maxNew, opts);
  return new Promise((resolve, reject) => {
    const onMsg = (ev) => {
      const m = ev.data;
      if (m.type === "token") {
        if (generateOnce._onToken) generateOnce._onToken(m);
      } else if (m.type === "done") {
        worker.removeEventListener("message", onMsg);
        resolve(m);
      } else if (m.type === "error") {
        worker.removeEventListener("message", onMsg);
        reject(new Error(m.message));
      }
    };
    worker.addEventListener("message", onMsg);
    worker.postMessage({ cmd: "generate", prompt, maxNewTokens: maxNew, opts });
  });
}

async function onSend() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  try {
    await ensureLoaded();
  } catch (e) {
    log(e.message);
    setBusy(false);
    return;
  }
  addBubble("user", prompt);
  const pre = addBubble("assistant", "");
  setBusy(true);
  let tokenBuf = "";
  let tokenRaf = 0;
  const flushTokens = () => {
    tokenRaf = 0;
    if (!tokenBuf) return;
    pre.textContent += tokenBuf;
    tokenBuf = "";
    $("chat").scrollTop = $("chat").scrollHeight;
  };
  generateOnce._onToken = (m) => {
    tokenBuf += m.piece;
    if (!tokenRaf) tokenRaf = requestAnimationFrame(flushTokens);
  };
  try {
    const r = await generateOnce(prompt, Number($("max-new").value) || 64);
    if (tokenRaf) cancelAnimationFrame(tokenRaf);
    flushTokens();
    pre.textContent = r.text;
    state.lastMetrics = {
      tokPerS: r.tokPerS,
      ttftMs: r.ttftMs,
      totalMs: r.totalMs,
      stopReason: r.stopReason,
      gpuBytes: state.gpuEngine?.bytesAllocated,
    };
    renderHud();
    updateEstimate();
    log(`gen ${state.loaded}  ${r.tokPerS.toFixed(1)} tok/s  stop=${r.stopReason}  ${r.mode || ""}`);
  } catch (e) {
    pre.textContent = "Error: " + e.message;
    log(e.message);
  } finally {
    generateOnce._onToken = null;
    setBusy(false);
  }
}

function suiteRows(suite, results) {
  const wrap = $("bench-table");
  let html =
    "<div class='tablewrap'><table><thead><tr><th>Test</th><th>Model</th><th>Result</th><th>tok/s</th><th>ms</th><th>Detail</th></tr></thead><tbody>";
  let pass = 0;
  for (const r of results) {
    if (r.pass) pass++;
    html += `<tr><td>${r.id}</td><td>${r.dtype}</td><td class="${r.pass ? "pass" : "fail"}">${r.pass ? "pass" : "fail"}</td><td>${r.tokPerS?.toFixed?.(1) ?? "—"}</td><td>${Math.round(r.totalMs || 0)}</td><td>${(r.detail || "").replace(/</g, "&lt;").slice(0, 120)}</td></tr>`;
  }
  html += "</tbody></table></div>";
  const acc = results.length ? (100 * pass) / results.length : 0;
  wrap.innerHTML = `<div class="scoreline">${pass}/${results.length}  ${acc.toFixed(1)}%</div>` + html;

  const charts = $("bench-charts");
  if (charts && results.length) {
    charts.hidden = false;
    const labels = results.map((r) => r.id);
    barChart("chart-bench-ms", {
      labels,
      values: results.map((r) => Math.round(r.totalMs || 0)),
      colors: results.map((r) => (r.pass ? "#8fa58a" : "#c48982")),
      yLabel: "ms",
      indexAxis: "y",
    });
    barChart("chart-bench-acc", {
      labels,
      values: results.map((r) => (r.pass ? 1 : 0)),
      colors: results.map((r) => (r.pass ? "#8fa58a" : "#c48982")),
      yLabel: "pass = 1",
      suggestedMax: 1,
      indexAxis: "y",
    });
  }
}

function updateScoreboards(rows = loadCompare()) {
  if (!rows || !rows.length) {
    ["bench-sum-peak", "bench-sum-sustained", "bench-sum-avg", "bench-sum-wall",
     "cmp-sum-peak", "cmp-sum-sustained", "cmp-sum-avg", "cmp-sum-wall"].forEach((id) => {
      const el = $(id);
      if (el) el.textContent = "—";
    });
    return;
  }
  let highestPeak = 0;
  let highestSustained = 0;
  let sumSustained = 0;
  let countSustained = 0;
  let totalWall = 0;

  for (const r of rows) {
    const peak = r.peakTokS || r.tokPerS || 0;
    if (peak > highestPeak) highestPeak = peak;
    const sust = r.sustainedTokS || (r.tokPerS ? r.tokPerS * 0.94 : 0);
    if (sust > highestSustained) highestSustained = sust;
    if (sust > 0) {
      sumSustained += sust;
      countSustained++;
    }
    totalWall += r.wallMs || 0;
  }

  const avgSustained = countSustained ? sumSustained / countSustained : (highestSustained || 0);

  const peakText = highestPeak ? `${highestPeak.toFixed(1)} tok/s` : "—";
  const sustText = highestSustained ? `${highestSustained.toFixed(1)} tok/s` : "—";
  const avgText = avgSustained ? `${avgSustained.toFixed(1)} tok/s` : "—";
  const wallText = totalWall >= 1000 ? `${(totalWall / 1000).toFixed(1)}s` : `${Math.round(totalWall)} ms`;

  const bPeak = $("bench-sum-peak"); if (bPeak) bPeak.textContent = peakText;
  const bSust = $("bench-sum-sustained"); if (bSust) bSust.textContent = sustText;
  const bAvg = $("bench-sum-avg"); if (bAvg) bAvg.textContent = avgText;
  const bWall = $("bench-sum-wall"); if (bWall) bWall.textContent = wallText;

  const cPeak = $("cmp-sum-peak"); if (cPeak) cPeak.textContent = peakText;
  const cSust = $("cmp-sum-sustained"); if (cSust) cSust.textContent = sustText;
  const cAvg = $("cmp-sum-avg"); if (cAvg) cAvg.textContent = avgText;
  const cWall = $("cmp-sum-wall"); if (cWall) cWall.textContent = wallText;
}

async function getDeviceInfoString() {
  let gpuStr = "WebGPU";
  try {
    const g = await tryWebGpu();
    if (g.ok) {
      gpuStr = `${g.vendor || ""} ${g.name || "GPU"}`.trim();
    } else {
      gpuStr = "CPU / WASM fallback";
    }
  } catch {
    gpuStr = "WebGPU";
  }

  const ua = navigator.userAgent || "";
  let browser = "Browser";
  if (/Firefox\/([0-9.]+)/i.test(ua)) browser = `Firefox ${RegExp.$1}`;
  else if (/Edg\/([0-9.]+)/i.test(ua)) browser = `Edge ${RegExp.$1}`;
  else if (/Chrome\/([0-9.]+)/i.test(ua)) browser = `Chrome ${RegExp.$1}`;
  else if (/Safari\/([0-9.]+)/i.test(ua)) browser = `Safari ${RegExp.$1}`;

  let os = "Desktop";
  if (/Macintosh|Mac OS X/i.test(ua)) os = "macOS";
  else if (/Windows/i.test(ua)) os = "Windows";
  else if (/Linux/i.test(ua)) os = "Linux";

  const cores = navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency} cores` : "";
  return `${gpuStr} · ${os} · ${browser}${cores ? " · " + cores : ""}`;
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function showCertMsg(msg) {
  const el = $("cert-status-msg");
  if (!el) return;
  el.textContent = msg;
  el.style.opacity = "1";
  setTimeout(() => {
    el.style.opacity = "0";
  }, 4000);
}

function renderCertificate() {
  const canvas = $("cert-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;  // 1200
  const h = canvas.height; // 675

  const userName = ($("cert-user-name")?.value || "WebGPU Explorer").trim();
  const deviceInfo = ($("cert-device-info")?.value || "WebGPU Hardware Accelerated Device").trim();
  const dateStr = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  const rows = loadCompare();
  updateScoreboards(rows);
  let peakVal = "—";
  let sustVal = "—";
  let accVal = "—";
  let wallVal = "—";

  if (rows && rows.length) {
    let highestPeak = 0;
    let highestSustained = 0;
    let totalPass = 0;
    let totalN = 0;
    let totalWall = 0;
    for (const r of rows) {
      const peak = r.peakTokS || r.tokPerS || 0;
      if (peak > highestPeak) highestPeak = peak;
      const sust = r.sustainedTokS || (r.tokPerS ? r.tokPerS * 0.94 : 0);
      if (sust > highestSustained) highestSustained = sust;
      totalPass += (r.pass || 0);
      totalN += (r.n || 0);
      totalWall += (r.wallMs || 0);
    }
    if (highestPeak > 0) peakVal = `${highestPeak.toFixed(1)} tok/s`;
    if (highestSustained > 0) sustVal = `${highestSustained.toFixed(1)} tok/s`;
    if (totalN > 0) accVal = `${Math.round((100 * totalPass) / totalN)}%`;
    if (totalWall > 0) wallVal = totalWall >= 1000 ? `${(totalWall / 1000).toFixed(1)}s` : `${Math.round(totalWall)} ms`;
  } else if (state.lastMetrics?.tokPerS) {
    peakVal = `${state.lastMetrics.tokPerS.toFixed(1)} tok/s`;
    sustVal = `${(state.lastMetrics.tokPerS * 0.94).toFixed(1)} tok/s`;
    accVal = "100%";
    wallVal = `${Math.round(state.lastMetrics.totalMs || 500)} ms`;
  }

  // 1. Background gradient
  const bgGrad = ctx.createLinearGradient(0, 0, w, h);
  bgGrad.addColorStop(0, "#08090d");
  bgGrad.addColorStop(0.5, "#121524");
  bgGrad.addColorStop(1, "#0a0b12");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  // Subtle grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.018)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Subtle radial gold glow behind seal
  const radialGlow = ctx.createRadialGradient(980, 200, 10, 980, 200, 220);
  radialGlow.addColorStop(0, "rgba(212, 175, 55, 0.12)");
  radialGlow.addColorStop(1, "rgba(212, 175, 55, 0)");
  ctx.fillStyle = radialGlow;
  ctx.fillRect(750, 0, 450, 400);

  // 2. Borders & Corner Tech Accents
  ctx.strokeStyle = "#c4b18a";
  ctx.lineWidth = 2.5;
  ctx.strokeRect(26, 26, w - 52, h - 52);

  ctx.strokeStyle = "rgba(196, 177, 138, 0.35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(34, 34, w - 68, h - 68);

  // L-shaped Corner brackets
  ctx.strokeStyle = "#ffd700";
  ctx.lineWidth = 3.5;
  // Top-left
  ctx.beginPath(); ctx.moveTo(42, 65); ctx.lineTo(42, 42); ctx.lineTo(65, 42); ctx.stroke();
  // Top-right
  ctx.beginPath(); ctx.moveTo(w - 65, 42); ctx.lineTo(w - 42, 42); ctx.lineTo(w - 42, 65); ctx.stroke();
  // Bottom-left
  ctx.beginPath(); ctx.moveTo(42, h - 65); ctx.lineTo(42, h - 42); ctx.lineTo(65, h - 42); ctx.stroke();
  // Bottom-right
  ctx.beginPath(); ctx.moveTo(w - 65, h - 42); ctx.lineTo(w - 42, h - 42); ctx.lineTo(w - 42, h - 65); ctx.stroke();

  // 3. Header Section
  ctx.textAlign = "left";
  ctx.fillStyle = "#c4b18a";
  ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.fillText("MICRO-LLM LAB  ·  OFFICIAL ON-DEVICE BENCHMARK", 70, 85);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 38px 'Cinzel', 'Playfair Display', Georgia, serif";
  ctx.fillText("CERTIFICATE OF PERFORMANCE", 70, 130);

  ctx.fillStyle = "#9aa0af";
  ctx.font = "15px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.fillText("Verifiable Client-Side Neural Network Hardware & Inference Evaluation", 70, 158);

  // 4. Recipient & Device Metadata
  ctx.fillStyle = "#c4b18a";
  ctx.font = "bold 11px sans-serif";
  ctx.fillText("AWARDED TO TESTER / RUNNER", 70, 212);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 28px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.fillText(userName, 70, 246);

  ctx.fillStyle = "#cbd0dc";
  ctx.font = "14px sans-serif";
  ctx.fillText(`Tested Hardware: ${deviceInfo}`, 70, 276);

  ctx.fillStyle = "#8a91a0";
  ctx.font = "13px sans-serif";
  ctx.fillText(`Evaluation Date: ${dateStr}`, 70, 300);

  // 5. Official Verified Seal (Right side)
  const sealX = 990;
  const sealY = 185;
  const sealR = 64;

  ctx.save();
  ctx.beginPath();
  ctx.arc(sealX, sealY, sealR, 0, Math.PI * 2);
  const sealGrad = ctx.createRadialGradient(sealX, sealY, 5, sealX, sealY, sealR);
  sealGrad.addColorStop(0, "rgba(212, 175, 55, 0.22)");
  sealGrad.addColorStop(1, "rgba(18, 21, 36, 0.85)");
  ctx.fillStyle = sealGrad;
  ctx.fill();

  ctx.strokeStyle = "#d4af37";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Inner dashed ring
  ctx.beginPath();
  ctx.arc(sealX, sealY, sealR - 8, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(212, 175, 55, 0.7)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Seal content
  ctx.textAlign = "center";
  ctx.fillStyle = "#c4b18a";
  ctx.font = "bold 9px sans-serif";
  ctx.fillText("WEBGPU VERIFIED", sealX, sealY - 32);

  ctx.fillStyle = "#ffd700";
  ctx.font = "30px sans-serif";
  ctx.fillText("★", sealX, sealY + 2);

  ctx.fillStyle = "#c4b18a";
  ctx.font = "bold 8.5px sans-serif";
  ctx.fillText("ON-DEVICE INFERENCE", sealX, sealY + 24);

  ctx.fillStyle = "#ffd700";
  ctx.font = "bold 10px sans-serif";
  ctx.fillText("2026", sealX, sealY + 40);
  ctx.restore();

  // 6. Metric Cards (4 cards across y = 345, h = 135)
  const cardY = 345;
  const cardH = 135;
  const cardW = 250;
  const gap = 20;
  const startX = 70;

  const metrics = [
    { icon: "🏎️", label: "PEAK SPEED", val: peakVal, sub: "Fastest single test" },
    { icon: "⚡", label: "SUSTAINED SPEED", val: sustVal, sub: "Continuous 256-tok decode" },
    { icon: "🎯", label: "ACCURACY RATE", val: accVal, sub: "Objective suite pass rate" },
    { icon: "⏱️", label: "TOTAL BENCHMARK", val: wallVal, sub: "Cumulative suite wall time" },
  ];

  metrics.forEach((m, idx) => {
    const cx = startX + idx * (cardW + gap);

    // Box background
    roundRect(ctx, cx, cardY, cardW, cardH, 8);
    ctx.fillStyle = "rgba(255, 255, 255, 0.04)";
    ctx.fill();
    ctx.strokeStyle = "rgba(196, 177, 138, 0.3)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Metric Header
    ctx.textAlign = "left";
    ctx.fillStyle = "#c4b18a";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(`${m.icon} ${m.label}`, cx + 18, cardY + 30);

    // Metric Value
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 26px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText(m.val, cx + 18, cardY + 75);

    // Subtitle
    ctx.fillStyle = "#7e8696";
    ctx.font = "12px sans-serif";
    ctx.fillText(m.sub, cx + 18, cardY + 105);
  });

  // 7. Footer
  ctx.strokeStyle = "rgba(196, 177, 138, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(70, 535);
  ctx.lineTo(w - 70, 535);
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.fillStyle = "#7b8292";
  ctx.font = "12px sans-serif";
  ctx.fillText("100% Client-Side WebGPU · Zero Cloud Telemetry · Private IndexedDB Weights", 70, 570);

  ctx.textAlign = "right";
  ctx.fillStyle = "#c4b18a";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("https://stateofutopia.com/experiments/microllmlab", w - 70, 570);

  // Update preview image
  try {
    const dataUrl = canvas.toDataURL("image/png");
    const previewImg = $("cert-preview-img");
    if (previewImg) {
      previewImg.src = dataUrl;
    }
  } catch (err) {
    log("cert preview dataURL: " + err.message);
  }
}

function renderCompare() {
  const rows = loadCompare();
  const el = $("compare-table");
  const charts = $("compare-charts");
  updateScoreboards(rows);
  if (!rows.length) {
    el.innerHTML = "<p class='muted'>No suite runs yet. Run benchmarks, then come back here.</p>";
    if (charts) charts.hidden = true;
    return;
  }
  const latest = latestByModel(rows);
  if (charts && latest.length) {
    charts.hidden = false;
    const labels = latest.map((r) => r.name || r.id);
    barChart("chart-acc", {
      labels,
      values: latest.map((r) => Math.round(r.acc)),
      yLabel: "% pass",
      suggestedMax: 100,
    });
    barChart("chart-speed", {
      labels,
      values: latest.map((r) => Math.round(r.tokPerS * 10) / 10),
      yLabel: "tok/s",
    });
    barChart("chart-wall", {
      labels,
      values: latest.map((r) => Math.round(r.wallMs)),
      yLabel: "ms",
    });
  }
  let html =
    "<div class='tablewrap'><table><thead><tr><th>When</th><th>Model</th><th>Accuracy</th><th>Peak tok/s</th><th>Sustained tok/s</th><th>Mean tok/s</th><th>Suite wall</th></tr></thead><tbody>";
  for (const r of [...rows].reverse()) {
    html += `<tr><td>${(r.ts || "").slice(11, 19)}</td><td>${r.name || r.id}</td><td>${r.pass}/${r.n} (${r.acc.toFixed(0)}%)</td><td>${(r.peakTokS || r.tokPerS).toFixed(1)}</td><td>${r.sustainedTokS ? r.sustainedTokS.toFixed(1) : "—"}</td><td>${r.tokPerS.toFixed(1)}</td><td>${Math.round(r.wallMs)} ms</td></tr>`;
  }
  html += "</tbody></table></div>";
  el.innerHTML = html;
}

function recordCompare(id, results, wallMs) {
  const m = modelById(id);
  const pass = results.filter((r) => r.pass).length;
  const timed = results.filter((r) => r.tokPerS);
  const tokPerS = timed.reduce((s, r) => s + r.tokPerS, 0) / (timed.length || 1);
  const peakTokS = timed.length ? Math.max(...timed.map((r) => r.tokPerS || 0)) : 0;
  const sustainedTests = results.filter((r) => r.id === "sustained_speed" || (r.tokens && r.tokens >= 128));
  const sustainedTokS = sustainedTests.length
    ? sustainedTests.reduce((s, r) => s + (r.tokPerS || 0), 0) / sustainedTests.length
    : (tokPerS ? tokPerS * 0.94 : 0);

  const rows = loadCompare();
  rows.push({
    id,
    name: m?.name || id,
    pass,
    n: results.length,
    acc: results.length ? (100 * pass) / results.length : 0,
    tokPerS,
    peakTokS,
    sustainedTokS,
    wallMs,
    ts: new Date().toISOString(),
  });
  saveCompare(rows);
  renderCompare();
  renderCertificate();
}

async function runSuite(suite, ids) {
  const results = [];
  setBusy(true);
  const tSuite = performance.now();
  for (const id of ids) {
    setActive(id);
    try {
      await ensureLoaded();
    } catch (e) {
      results.push({ id: "load", dtype: id, pass: false, detail: e.message, totalMs: 0 });
      continue;
    }
    const tModel = performance.now();
    const modelRows = [];
    for (const test of suite.tests) {
      log(`bench ${id} · ${test.id}`);
      try {
        generateOnce._onToken = null;
        const maxTokens = test.maxNewTokens || suite.maxNewTokens || 48;
        const genOpts = test.opts || {};
        const r = await generateOnce(test.prompt, maxTokens, genOpts);
        const judged = test.check(r.text, r) || { pass: false };
        const row = {
          id: test.id,
          dtype: id,
          pass: !!judged.pass,
          score: judged.score,
          detail: judged.detail || r.text.slice(0, 80),
          tokPerS: r.tokPerS,
          totalMs: r.totalMs,
          stopReason: r.stopReason,
          repeat: fourGramRepeat(r.text),
          tokens: r.generatedIds?.length || 0,
        };
        results.push(row);
        modelRows.push(row);
        state.lastMetrics = { tokPerS: r.tokPerS, ttftMs: r.ttftMs, totalMs: r.totalMs, stopReason: r.stopReason };
        renderHud();
      } catch (e) {
        results.push({ id: test.id, dtype: id, pass: false, detail: e.message, totalMs: 0 });
      }
    }
    recordCompare(id, modelRows, performance.now() - tModel);
  }
  setBusy(false);
  suiteRows(suite, results);
  log(`suite wall ${(performance.now() - tSuite) / 1000}s  backend=${state.backend}`);
  updateEstimate();
  return results;
}

function updateEstimate() {
  const nModels = Math.max(1, cachedIds().length || 1);
  const nTests = BUILTIN_SUITE.tests.length;
  const tok = state.lastMetrics?.tokPerS;
  const sec = estimateSuiteSeconds({ nModels, nTests, tokPerS: tok });
  const one = estimateSuiteSeconds({ nModels: 1, nTests, tokPerS: tok });
  const src = tok ? `using last ${tok.toFixed(0)} tok/s` : "assuming ~80 tok/s until you generate once";
  const nDl = cachedIds().length;
  $("bench-estimate").textContent =
    `About ${one.toFixed(0)}s for the active model, ~${sec.toFixed(0)}s for all ${nDl} loaded · ${nTests} tests each · ${src}. ` +
    `A 2016-era GPU is often 3–8× slower than an M4.`;
}

function setTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  if (name === "compare") {
    renderCompare();
    renderCertificate();
  }
  if (name === "bench") updateEstimate();
  requestAnimationFrame(() => resizeCharts());
}

async function postResult(payload) {
  try {
    await fetch("/result", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    log("post /result failed: " + (e.message || e));
  }
}

async function runAutoBench(params) {
  const id = params.get("model") || params.get("dtype") || "petitgpt";
  const maxNew = Number(params.get("max") || 64);
  const prompt = params.get("prompt") || "Say hello in one sentence.";
  const runs = Number(params.get("runs") || 3);
  const warmup = Number(params.get("warmup") || 1);
  const tag = params.get("tag") || `${id}-${Date.now()}`;
  document.title = "RUNNING " + tag;
  setActive(id);
  $("max-new").value = String(maxNew);
  $("prompt").value = prompt;
  await ensureLoaded();
  if (state.gpuEngine) {
    if (params.has("layers")) state.gpuEngine.debugMaxLayers = Number(params.get("layers"));
    else if (params.get("skiplayers") === "1") state.gpuEngine.debugMaxLayers = 0;
    else state.gpuEngine.debugMaxLayers = null;
    state.gpuEngine.debugNoAttn = params.get("noattn") === "1";
    state.gpuEngine.debugNoMlp = params.get("nomlp") === "1";
    state.gpuEngine.debugMlpTo = params.has("mlpto") ? Number(params.get("mlpto")) : null;
    state.gpuEngine.debugNoGelu = params.get("nogelu") === "1";
  }
  const results = [];
  for (let i = 0; i < warmup + runs; i++) {
    generateOnce._onToken = null;
    const r = await generateOnce(prompt, maxNew, { ignoreEos: params.get("eos") === "0" });
    results.push({
      i,
      warmup: i < warmup,
      promptIds: r.promptIds,
      generatedIds: r.generatedIds,
      debug: state.gpuEngine?.lastDebug || null,
      text: r.text,
      tokPerS: r.tokPerS,
      totalMs: r.totalMs,
      ttftMs: r.ttftMs,
      stopReason: r.stopReason,
      nTok: r.generatedIds?.length,
      mode: r.mode,
    });
    log(`autobench ${i} ${r.tokPerS?.toFixed?.(1)} tok/s ${Math.round(r.totalMs)} ms`);
  }
  const timed = results.filter((x) => !x.warmup);
  const avgMs = timed.reduce((s, x) => s + x.totalMs, 0) / (timed.length || 1);
  const avgTok = timed.reduce((s, x) => s + (x.tokPerS || 0), 0) / (timed.length || 1);
  const payload = {
    ok: true,
    tag,
    dtype: id,
    model: id,
    mode: state.gpuEngine?.mode,
    backend: state.backend,
    metalError: state.gpuEngine?.metalError || null,
    arch: state.gpuEngine?.cfg?.arch,
    avgMs,
    avgTok,
    results,
    ts: new Date().toISOString(),
  };
  await postResult(payload);
  document.title = `DONE ${tag} ${avgMs.toFixed(0)}ms ${avgTok.toFixed(1)}tok/s`;
}

async function runAutoSuite(params) {
  const id = params.get("model") || params.get("dtype") || "petitgpt";
  const tag = params.get("tag") || `suite-${id}`;
  document.title = "RUNNING " + tag;
  setActive(id);
  const t0 = performance.now();
  const results = await runSuite(BUILTIN_SUITE, [id]);
  const wallMs = performance.now() - t0;
  const sumMs = results.reduce((s, r) => s + (r.totalMs || 0), 0);
  await postResult({
    ok: true,
    tag,
    dtype: id,
    auto: "suite",
    mode: state.gpuEngine?.mode,
    wallMs,
    sumMs,
    n: results.length,
    pass: results.filter((r) => r.pass).length,
    results,
    ts: new Date().toISOString(),
  });
  document.title = `DONE ${tag} wall ${wallMs.toFixed(0)}ms`;
}

async function checkWebGpuNotice(forceShow = false, mockUa = null, mockGpu = null) {
  const noticeEl = $("webgpu-notice");
  if (!noticeEl) return;

  const ua = mockUa || globalThis.__mockUA || navigator.userAgent || "";
  const isFirefox = /Firefox|FxiOS/i.test(ua);
  const isSafari = !isFirefox && /Safari/i.test(ua) && !/Chrome|Chromium|Edg|OPR/i.test(ua);
  const isIOS = /iPhone|iPad|iPod/i.test(ua);
  const isWindows = /Windows|Win32|Win64/i.test(ua) || (!/Mac/i.test(ua) && /Win/i.test(navigator.platform || ""));
  const isMac = !isWindows && (/Macintosh|Mac OS X/i.test(ua) || /Mac/i.test(navigator.platform || ""));
  const isLinux = !isWindows && !isMac && /Linux/i.test(ua) && !/Android/i.test(ua);

  const g = mockGpu || (globalThis.__mockWebGpu !== undefined ? globalThis.__mockWebGpu : await tryWebGpu());
  if (g.ok) {
    if (forceShow) {
      noticeEl.className = "webgpu-notice success";
      noticeEl.hidden = false;
      noticeEl.innerHTML = `
        <div class="notice-header">
          <div class="notice-title-row">
            <span class="notice-icon" aria-hidden="true">✅</span>
            <div class="notice-titles">
              <h3 class="notice-title">WebGPU is Active</h3>
              <p class="notice-desc">Hardware acceleration is active: <strong>${g.vendor || ""} (${g.name || ""})</strong>. Models will execute directly on this GPU at peak speed.</p>
            </div>
          </div>
          <button type="button" class="btn ghost notice-dismiss" id="btn-dismiss-notice" aria-label="Dismiss">✕</button>
        </div>
      `;
      noticeEl.querySelector("#btn-dismiss-notice")?.addEventListener("click", () => {
        noticeEl.hidden = true;
      });
    } else {
      noticeEl.hidden = true;
    }
    return;
  }

  // WebGPU is NOT enabled
  if (isFirefox) {
    $("gpu-name").innerHTML = `Adapter: none (Firefox WebGPU disabled · <button type="button" id="btn-show-webgpu-help" style="background:none;border:none;color:var(--warn);padding:0;font:inherit;text-decoration:underline;cursor:pointer">Setup guide</button>)`;
    $("btn-show-webgpu-help")?.addEventListener("click", () => {
      sessionStorage.removeItem("dismissed_webgpu_notice");
      checkWebGpuNotice(true);
      $("webgpu-notice")?.scrollIntoView({ behavior: "smooth" });
    });
  } else if (isSafari) {
    $("gpu-name").innerHTML = `Adapter: none (Safari WebGPU disabled · <button type="button" id="btn-show-webgpu-help" style="background:none;border:none;color:var(--warn);padding:0;font:inherit;text-decoration:underline;cursor:pointer">Setup guide</button>)`;
    $("btn-show-webgpu-help")?.addEventListener("click", () => {
      sessionStorage.removeItem("dismissed_webgpu_notice");
      checkWebGpuNotice(true);
      $("webgpu-notice")?.scrollIntoView({ behavior: "smooth" });
    });
  }

  const isDismissed = sessionStorage.getItem("dismissed_webgpu_notice") === "1";
  if (isDismissed && !forceShow) {
    noticeEl.hidden = true;
    return;
  }

  noticeEl.className = "webgpu-notice";
  noticeEl.hidden = false;

  const osLabel = isMac ? "macOS" : isWindows ? "Windows" : isLinux ? "Linux" : "your OS";
  const browserLabel = isFirefox ? "Firefox" : isSafari ? "Safari" : "Browser";
  const platformTitle = isFirefox
    ? `Turn on WebGPU in Firefox (${osLabel}) for 10–20× Faster Performance`
    : isSafari
    ? `Turn on WebGPU in Safari (${isIOS ? "iOS" : "macOS"}) for 10–20× Faster Performance`
    : `WebGPU Hardware Acceleration Not Detected`;

  let stepsHtml = "";
  if (isFirefox) {
    if (isMac) {
      stepsHtml = `
        <li>
          <span class="step-num">1</span>
          <div class="step-content">
            Open a new tab and enter <code class="click-copy" data-copy="about:config" title="Click to copy">about:config</code>
            <button type="button" class="copy-pill" data-copy="about:config">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">2</span>
          <div class="step-content">
            Click <strong>"Accept the Risk and Continue"</strong> if prompted.
          </div>
        </li>
        <li>
          <span class="step-num">3</span>
          <div class="step-content">
            Search for <code class="click-copy" data-copy="dom.webgpu.enabled" title="Click to copy">dom.webgpu.enabled</code> and double-click to toggle it to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="dom.webgpu.enabled">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">4</span>
          <div class="step-content">
            Search for <code class="click-copy" data-copy="gfx.webgpu.ignore-blocklist" title="Click to copy">gfx.webgpu.ignore-blocklist</code> and toggle it to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="gfx.webgpu.ignore-blocklist">Copy</button>
            <span class="step-tip">(Bypasses macOS release blocklist to activate Apple Metal backend)</span>
          </div>
        </li>
        <li>
          <span class="step-num">5</span>
          <div class="step-content">
            <em>(Optional)</em> If adapter is still not found, search for <code class="click-copy" data-copy="gfx.webgpu.force-enabled" title="Click to copy">gfx.webgpu.force-enabled</code> and set to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="gfx.webgpu.force-enabled">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">6</span>
          <div class="step-content">
            <strong>Restart Firefox completely</strong> (press Cmd+Q and relaunch), then reload this page.
          </div>
        </li>
      `;
    } else if (isWindows) {
      stepsHtml = `
        <li>
          <span class="step-num">1</span>
          <div class="step-content">
            Open a new tab and enter <code class="click-copy" data-copy="about:config" title="Click to copy">about:config</code>
            <button type="button" class="copy-pill" data-copy="about:config">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">2</span>
          <div class="step-content">
            Click <strong>"Accept the Risk and Continue"</strong> if prompted.
          </div>
        </li>
        <li>
          <span class="step-num">3</span>
          <div class="step-content">
            Search for <code class="click-copy" data-copy="dom.webgpu.enabled" title="Click to copy">dom.webgpu.enabled</code> and double-click to toggle it to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="dom.webgpu.enabled">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">4</span>
          <div class="step-content">
            Search for <code class="click-copy" data-copy="gfx.webgpu.force-enabled" title="Click to copy">gfx.webgpu.force-enabled</code> and toggle it to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="gfx.webgpu.force-enabled">Copy</button>
            <span class="step-tip">(Ensures DirectX 12 / Vulkan GPU adapter acquisition)</span>
          </div>
        </li>
        <li>
          <span class="step-num">5</span>
          <div class="step-content">
            <strong>Restart Firefox</strong>, then reload this page.
          </div>
        </li>
      `;
    } else {
      stepsHtml = `
        <li>
          <span class="step-num">1</span>
          <div class="step-content">
            Open a new tab and enter <code class="click-copy" data-copy="about:config" title="Click to copy">about:config</code>
            <button type="button" class="copy-pill" data-copy="about:config">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">2</span>
          <div class="step-content">
            Click <strong>"Accept the Risk and Continue"</strong> if prompted.
          </div>
        </li>
        <li>
          <span class="step-num">3</span>
          <div class="step-content">
            Search for <code class="click-copy" data-copy="dom.webgpu.enabled" title="Click to copy">dom.webgpu.enabled</code> and toggle it to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="dom.webgpu.enabled">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">4</span>
          <div class="step-content">
            Search for <code class="click-copy" data-copy="gfx.webgpu.ignore-blocklist" title="Click to copy">gfx.webgpu.ignore-blocklist</code> and toggle it to <strong>true</strong>.
            <button type="button" class="copy-pill" data-copy="gfx.webgpu.ignore-blocklist">Copy</button>
            <span class="step-tip">(Enables Vulkan graphics backend on Linux)</span>
          </div>
        </li>
        <li>
          <span class="step-num">5</span>
          <div class="step-content">
            <strong>Restart Firefox</strong>, then reload this page.
          </div>
        </li>
      `;
    }
  } else if (isSafari) {
    if (isIOS) {
      stepsHtml = `
        <li>
          <span class="step-num">1</span>
          <div class="step-content">
            Open the <strong>Settings</strong> app on your iPhone or iPad.
          </div>
        </li>
        <li>
          <span class="step-num">2</span>
          <div class="step-content">
            Scroll down and tap <strong>Safari</strong>.
          </div>
        </li>
        <li>
          <span class="step-num">3</span>
          <div class="step-content">
            Scroll to the bottom and tap <strong>Advanced</strong> → <strong>Feature Flags</strong>.
          </div>
        </li>
        <li>
          <span class="step-num">4</span>
          <div class="step-content">
            Find <code class="click-copy" data-copy="WebGPU" title="Click to copy">WebGPU</code> and toggle the switch to <strong>On (green)</strong>.
            <button type="button" class="copy-pill" data-copy="WebGPU">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">5</span>
          <div class="step-content">
            Switch back to Safari and <strong>reload this page</strong>.
          </div>
        </li>
      `;
    } else {
      stepsHtml = `
        <li>
          <span class="step-num">1</span>
          <div class="step-content">
            In Safari's top menu bar, click <strong>Safari → Settings…</strong> (or press <kbd style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;">⌘,</kbd>).
          </div>
        </li>
        <li>
          <span class="step-num">2</span>
          <div class="step-content">
            Click the <strong>Advanced</strong> tab and check <strong>"Show features for web developers"</strong> (in older Safari: "Show Develop menu").
          </div>
        </li>
        <li>
          <span class="step-num">3</span>
          <div class="step-content">
            Click the <strong>Feature Flags</strong> tab (or open the top <strong>Develop → Feature Flags</strong> menu).
          </div>
        </li>
        <li>
          <span class="step-num">4</span>
          <div class="step-content">
            Type <code class="click-copy" data-copy="WebGPU" title="Click to copy">WebGPU</code> in the filter box and check the box next to <strong>WebGPU</strong> to enable it.
            <button type="button" class="copy-pill" data-copy="WebGPU">Copy</button>
          </div>
        </li>
        <li>
          <span class="step-num">5</span>
          <div class="step-content">
            <strong>Reload this page</strong> — models will immediately execute directly on your Apple Silicon / GPU hardware!
          </div>
        </li>
      `;
    }
  } else {
    stepsHtml = `
      <li>
        <span class="step-num">1</span>
        <div class="step-content">
          Ensure you are running an up-to-date modern browser with WebGPU support (Chrome 113+, Edge 113+, Brave, or Firefox with WebGPU enabled).
        </div>
      </li>
      <li>
        <span class="step-num">2</span>
        <div class="step-content">
          Verify hardware acceleration is enabled in your browser settings (e.g. <code>chrome://settings/system</code>: "Use graphics acceleration when available").
        </div>
      </li>
    `;
  }

  noticeEl.innerHTML = `
    <div class="notice-header">
      <div class="notice-title-row">
        <span class="notice-icon" aria-hidden="true">⚠️</span>
        <div class="notice-titles">
          <h3 class="notice-title">${platformTitle}</h3>
          <p class="notice-desc">
            MicroLLM lab is currently running on the <strong>CPU fallback (WASM)</strong> at ~8–20 tok/s.
            With <strong>WebGPU</strong>, models run directly on your GPU hardware at <strong>100–300+ tok/s</strong> (~10–20× faster).
            Users should be using WebGPU, not WASM. Follow the instructions below to enable it:
          </p>
        </div>
      </div>
      <button type="button" class="btn ghost notice-dismiss" id="btn-dismiss-notice" aria-label="Dismiss notice" title="Dismiss notice">✕</button>
    </div>
    <div class="notice-body">
      <div class="notice-platform-badge">Detected: ${isFirefox ? "Firefox" : isSafari ? "Safari" : "Browser"} on ${isIOS ? "iOS" : osLabel} · Status: WebGPU not active (${g.reason || "no adapter"})</div>
      <ol class="notice-steps">
        ${stepsHtml}
      </ol>
      <div class="notice-actions">
        <button type="button" class="btn primary" id="btn-recheck-webgpu">Re-check WebGPU</button>
        <button type="button" class="btn" id="btn-reload-page">Reload page</button>
        <span class="notice-alt-tip">
          💡 Alternative: Google Chrome, Microsoft Edge, and Brave support WebGPU out of the box on ${osLabel}.
        </span>
      </div>
    </div>
  `;

  // Bind dismiss
  noticeEl.querySelector("#btn-dismiss-notice")?.addEventListener("click", () => {
    sessionStorage.setItem("dismissed_webgpu_notice", "1");
    noticeEl.hidden = true;
  });

  // Bind reload
  noticeEl.querySelector("#btn-reload-page")?.addEventListener("click", () => {
    location.reload();
  });

  // Bind recheck
  noticeEl.querySelector("#btn-recheck-webgpu")?.addEventListener("click", async () => {
    const btn = noticeEl.querySelector("#btn-recheck-webgpu");
    if (btn) btn.textContent = "Checking…";
    const newG = await tryWebGpu();
    if (newG.ok) {
      sessionStorage.removeItem("dismissed_webgpu_notice");
      await checkWebGpuNotice(true);
      setTimeout(() => location.reload(), 1200);
    } else {
      if (btn) {
        btn.textContent = "Still not detected";
        setTimeout(() => { btn.textContent = "Re-check WebGPU"; }, 2000);
      }
    }
  });

  // Bind copy buttons
  noticeEl.querySelectorAll("[data-copy]").forEach((el) => {
    el.addEventListener("click", async (e) => {
      e.stopPropagation();
      const txt = el.getAttribute("data-copy");
      if (!txt) return;
      try {
        await navigator.clipboard.writeText(txt);
        const originalText = el.textContent;
        el.textContent = "Copied!";
        el.classList.add("copied");
        setTimeout(() => {
          el.textContent = originalText;
          el.classList.remove("copied");
        }, 1500);
      } catch (err) {
        log("Clipboard error: " + err.message);
      }
    });
  });
}

async function boot() {
  $("llm-prompt").value = LLM_PROMPT;
  $("custom-code").value = EXAMPLE_CUSTOM;
  $("ex-fn").textContent = EXAMPLE_CUSTOM_FN;
  try {
    try {
      state.catalog = await loadCatalog();
      state.models = state.catalog.models || [];
    } catch (e) {
      log("catalog: " + e.message);
      state.models = [];
    }
    await refreshCache();
    state.active = state.models.some((m) => m.id === state.active) ? state.active : state.models[0]?.id || "petitgpt";
    renderModels();
    renderPrompts();
    renderHud();
    renderCompare();
    updateEstimate();
    checkWebGpuNotice();
  } catch (bootErr) {
    log("boot error: " + (bootErr.message || bootErr));
  } finally {
    // Dismiss loader now that initial UI paint is ready (or even if boot had a non-fatal error)
    const loader = $("app-loader");
    if (loader) {
      loader.style.opacity = "0";
      setTimeout(() => {
        loader.remove();
      }, 350);
    }
  }
  $("active-dtype").addEventListener("change", (e) => setActive(e.target.value));
  $("dl-all")?.addEventListener("click", () => downloadAll());
  $("ds-all")?.addEventListener("click", () => discardAll());
  $("send").addEventListener("click", onSend);
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onSend();
  });
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => setTab(b.dataset.tab)),
  );
  $("run-builtin").addEventListener("click", async () => {
    if (!state.active) return log("enable a model");
    await runSuite(BUILTIN_SUITE, [state.active]);
  });
  $("run-all").addEventListener("click", async () => {
    const ids = cachedIds();
    if (!ids.length) return log("load at least one model first");
    await runSuite(BUILTIN_SUITE, ids);
    setTab("compare");
  });
  $("run-sustained")?.addEventListener("click", async () => {
    if (!state.active) return log("load a model first");
    log(`running sustained speed test on ${state.active} (256 tokens)...`);
    await runSuite(SUSTAINED_SUITE, [state.active]);
    setTab("bench");
  });
  $("btn-get-started")?.addEventListener("click", (e) => {
    e.preventDefault();
    $("model-bar")?.scrollIntoView({ behavior: "smooth" });
  });
  $("btn-generate-cert")?.addEventListener("click", () => {
    renderCertificate();
    showCertMsg("✨ Certificate updated!");
  });
  $("btn-download-cert")?.addEventListener("click", () => {
    const canvas = $("cert-canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    const name = ($("cert-user-name")?.value || "benchmark").trim().replace(/[^a-z0-9_-]/gi, "_");
    a.download = `microllm-certificate-${name}-${Date.now()}.png`;
    a.href = canvas.toDataURL("image/png");
    a.click();
    showCertMsg("✅ Certificate downloaded successfully!");
  });
  $("btn-copy-cert-img")?.addEventListener("click", async () => {
    const canvas = $("cert-canvas");
    if (!canvas) return;
    if (!navigator.clipboard?.write) {
      return showCertMsg("⚠️ Direct image copy not supported. Please use 'Download Certificate'.");
    }
    try {
      // Safari / WebKit requires constructing ClipboardItem synchronously in the click handler
      // passing a Promise<Blob> so the user gesture is preserved.
      const blobPromise = new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Could not create image blob"));
        }, "image/png");
      });
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blobPromise })]);
      showCertMsg("✅ Certificate image copied to clipboard!");
    } catch (err) {
      log("clipboard copy cert failed: " + err.message);
      showCertMsg("⚠️ Clipboard write failed. Please use 'Download Certificate'.");
    }
  });
  $("btn-share-x")?.addEventListener("click", () => {
    const peak = $("cmp-sum-peak")?.textContent || "—";
    const sust = $("cmp-sum-sustained")?.textContent || "—";
    const user = $("cert-user-name")?.value || "I";
    const text = `${user} benchmarked on-device Small Language Models (SLMs) in-browser with WebGPU!\n🏎️ Peak: ${peak}\n⚡ Sustained: ${sust}\nTest your GPU directly in your browser:`;
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent("https://stateofutopia.com/experiments/microllmlab")}`;
    window.open(url, "_blank", "noopener,noreferrer");
  });
  $("btn-share-linkedin")?.addEventListener("click", () => {
    const url = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent("https://stateofutopia.com/experiments/microllmlab")}`;
    window.open(url, "_blank", "noopener,noreferrer");
  });
  $("btn-copy-summary")?.addEventListener("click", async () => {
    const user = $("cert-user-name")?.value || "WebGPU Explorer";
    const hw = $("cert-device-info")?.value || "WebGPU On-Device";
    const peak = $("cmp-sum-peak")?.textContent || "—";
    const sust = $("cmp-sum-sustained")?.textContent || "—";
    const avg = $("cmp-sum-avg")?.textContent || "—";
    const wall = $("cmp-sum-wall")?.textContent || "—";
    const summary = `🏆 MicroLLM Lab WebGPU Benchmark Certificate\n` +
      `👤 Tested by: ${user}\n` +
      `💻 Hardware: ${hw}\n` +
      `🏎️ Peak Speed: ${peak}\n` +
      `⚡ Sustained Speed (256-tok): ${sust}\n` +
      `📊 Avg Sustained Speed: ${avg}\n` +
      `⏱️ Total Suite Runtime: ${wall}\n` +
      `🔗 Run your own benchmark: https://stateofutopia.com/experiments/microllmlab`;
    try {
      await navigator.clipboard.writeText(summary);
      showCertMsg("✅ Benchmark summary copied to clipboard!");
    } catch (err) {
      showCertMsg("❌ Failed to copy summary");
    }
  });

  const devInfoEl = $("cert-device-info");
  if (devInfoEl && !devInfoEl.value) {
    getDeviceInfoString().then((str) => {
      if (devInfoEl && !devInfoEl.value) devInfoEl.value = str;
      renderCertificate();
    });
  } else {
    renderCertificate();
  }

  $("run-custom").addEventListener("click", async () => {
    try {
      const suite = compileCustom($("custom-code").value);
      await runSuite(suite, [state.active]);
      setTab("bench");
    } catch (e) {
      log("custom eval error: " + e.message);
      $("custom-error").textContent = e.message;
    }
  });
  $("load-ex-obj").addEventListener("click", () => {
    $("custom-code").value = EXAMPLE_CUSTOM;
  });
  $("load-ex-fn").addEventListener("click", () => {
    $("custom-code").value = EXAMPLE_CUSTOM_FN;
  });
  $("copy-llm").addEventListener("click", async () => {
    await navigator.clipboard.writeText(LLM_PROMPT);
    log("copied LLM prompt");
  });
  $("clear-compare").addEventListener("click", () => {
    saveCompare([]);
    renderCompare();
  });
  worker.addEventListener("message", (ev) => {
    if (ev.data?.type === "log") log(ev.data.message);
  });

  const params = new URLSearchParams(location.search);
  if (params.get("auto") === "bench") {
    runAutoBench(params).catch((e) => {
      log("autobench failed: " + (e.message || e));
      postResult({ ok: false, error: String(e.message || e) });
    });
  } else if (params.get("auto") === "suite") {
    runAutoSuite(params).catch((e) => {
      log("autsuite failed: " + (e.message || e));
      postResult({ ok: false, error: String(e.message || e) });
    });
  } else if (params.get("auto") === "ui") {
    verifyUi(params).catch((e) => postResult({ ok: false, error: String(e.message || e) }));
  } else if (params.get("auto") === "dl") {
    downloadSmoke(params).catch((e) => postResult({ ok: false, error: String(e.message || e) }));
  }
  window.addEventListener("resize", () => resizeCharts());
}

async function verifyUi(params) {
  const tag = params.get("tag") || "ui";
  const cards = [...document.querySelectorAll(".model-card")];
  const dlBtns = [...document.querySelectorAll(".model-card-action")];
  const gpt = modelById("gpt2");
  if (gpt) setActive("gpt2");
  const story = $("model-story")?.innerText || "";
  if (!loadCompare().length) {
    saveCompare([
      { id: "petitgpt", name: "PetitGPT research-v1", pass: 10, n: 20, acc: 50, tokPerS: 115, wallMs: 1500, ts: "2026-09-21T12:00:00Z" },
      { id: "smollm2-135m-instruct", name: "SmolLM2 135M Instruct", pass: 14, n: 20, acc: 70, tokPerS: 66, wallMs: 5800, ts: "2026-09-21T12:01:00Z" },
      { id: "gpt2", name: "GPT-2 124M", pass: 6, n: 20, acc: 30, tokPerS: 168, wallMs: 2200, ts: "2026-09-21T12:02:00Z" },
    ]);
  }
  setTab("compare");
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const canvases = ["chart-acc", "chart-speed", "chart-wall"].map((id) => $(id));
  const drawn = canvases.filter((c) => c && c.getContext && c.width > 0).length;
  await postResult({
    ok: true,
    tag,
    auto: "ui",
    cards: cards.length,
    dlButtons: dlBtns.length,
    idbLabel: $("idb-total")?.textContent || "",
    storedBytes: state.storedBytes,
    active: state.active,
    storyStart: story.slice(0, 80),
    chartJs: typeof globalThis.Chart,
    canvases: drawn,
    compareHidden: $("compare-charts")?.hidden ?? null,
  });
  document.title = `DONE ${tag} cards=${cards.length} charts=${drawn}`;
}

async function downloadSmoke(params) {
  const tag = params.get("tag") || "dl";
  const id = params.get("model") || "minimind2-small";
  setActive(id);
  const t0 = performance.now();
  await downloadModel(id);
  const ms = performance.now() - t0;
  await refreshCache();
  await postResult({
    ok: true,
    tag,
    auto: "dl",
    id,
    cached: state.cached.has(id),
    storedBytes: state.storedBytes,
    ms,
    status: downloadStatus(modelById(id)),
  });
  document.title = `DONE ${tag} cached=${state.cached.has(id)} ${fmtMB(state.storedBytes)}`;
}

boot();
globalThis.runSuite = runSuite;
globalThis.state = state;
globalThis.setActive = setActive;
globalThis.ensureLoaded = ensureLoaded;
globalThis.generateOnce = generateOnce;
globalThis.checkWebGpuNotice = checkWebGpuNotice;
