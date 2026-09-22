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
        ? `Stored on this device: none (0 of ${fmtMB(catalogBytes())})`
        : `Stored on this device: ${fmtMB(state.storedBytes)} · ${n} model${n === 1 ? "" : "s"} (of ${fmtMB(catalogBytes())})`;
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
  if (state.cached.has(m.id)) return { kind: "have", text: `Downloaded · ${fmtMB(m.q4Bytes)}` };
  return { kind: "need", text: `Not downloaded · ${fmtMB(m.q4Bytes)}` };
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
      btn.textContent = "Downloading…";
    } else {
      btn.disabled = false;
      btn.textContent = st.kind === "have" ? "Discard" : "Download";
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
      log(`downloaded ${m.name} ${fmtMB(buf.byteLength)}`);
    } catch (e) {
      if (state.downloadGen[id] === gen) log("download failed " + m.name + ": " + (e.message || e));
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
    log("discarded " + id);
  } catch (e) {
    log("discard " + (e.message || e));
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
  if (!missing.length) return log("all models already on this device");
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
    log("discard all " + (e.message || e));
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
  log("discarded all cached weights");
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
      <button type="button" class="btn ghost model-card-action" data-act="${st.kind === "have" ? "discard" : "download"}" ${st.kind === "busy" ? "disabled" : ""}>${st.kind === "busy" ? "Downloading…" : st.kind === "have" ? "Discard" : "Download"}</button>`;
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
        <p class="meta">${have ? "Weights are on this device." : "Weights are not downloaded yet."}</p>`;
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
  if (state.loaded === id && state.gpuEngine) return;
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
    if (state.preferGpu && $("prefer-gpu").checked) {
      const g = await tryWebGpu();
      $("gpu-name").textContent = g.ok
        ? `Adapter: ${g.vendor || ""} (${g.name || ""})`
        : `Adapter: ${g.reason || "none"}`;
      if (g.ok) {
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
        return;
      }
    }
    worker.postMessage({ cmd: "load", kind: "q4", preferGpu: false, buffer: buf });
    state.backend = "worker";
    state.loaded = id;
    log("ready  worker fallback  " + m.name);
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
    worker.postMessage({ cmd: "generate", prompt, maxNewTokens: maxNew });
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

function renderCompare() {
  const rows = loadCompare();
  const el = $("compare-table");
  const charts = $("compare-charts");
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
    "<div class='tablewrap'><table><thead><tr><th>When</th><th>Model</th><th>Accuracy</th><th>Mean tok/s</th><th>Suite wall</th></tr></thead><tbody>";
  for (const r of [...rows].reverse()) {
    html += `<tr><td>${(r.ts || "").slice(11, 19)}</td><td>${r.name || r.id}</td><td>${r.pass}/${r.n} (${r.acc.toFixed(0)}%)</td><td>${r.tokPerS.toFixed(1)}</td><td>${Math.round(r.wallMs)} ms</td></tr>`;
  }
  html += "</tbody></table></div>";
  el.innerHTML = html;
}

function recordCompare(id, results, wallMs) {
  const m = modelById(id);
  const pass = results.filter((r) => r.pass).length;
  const timed = results.filter((r) => r.tokPerS);
  const tokPerS = timed.reduce((s, r) => s + r.tokPerS, 0) / (timed.length || 1);
  const rows = loadCompare();
  rows.push({
    id,
    name: m?.name || id,
    pass,
    n: results.length,
    acc: results.length ? (100 * pass) / results.length : 0,
    tokPerS,
    wallMs,
    ts: new Date().toISOString(),
  });
  saveCompare(rows);
  renderCompare();
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
        const r = await generateOnce(test.prompt, suite.maxNewTokens || 48);
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
    `About ${one.toFixed(0)}s for the active model, ~${sec.toFixed(0)}s for all ${nDl} downloaded · ${nTests} tests each · ${src}. ` +
    `A 2016-era GPU is often 3–8× slower than an M4.`;
}

function setTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  if (name === "compare") renderCompare();
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

async function boot() {
  $("llm-prompt").value = LLM_PROMPT;
  $("custom-code").value = EXAMPLE_CUSTOM;
  $("ex-fn").textContent = EXAMPLE_CUSTOM_FN;
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
    if (!ids.length) return log("download at least one model first");
    await runSuite(BUILTIN_SUITE, ids);
    setTab("compare");
  });
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
