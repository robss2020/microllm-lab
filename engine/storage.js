/** IndexedDB cache for weight files. Unchecked dtypes are deleted. */

const DB_NAME = "petitgpt-webgpu";
const STORE = "weights";
const PREF_KEY = "petitgpt-enabled-dtypes";

export const DTYPES = [
  { id: "q4", file: "weights/petitgpt.q4.bin", label: "Q4", bytes: 78_000_000, note: "Grouped int4 (g=32)" },
];

let _catalog = null;
export async function loadCatalog() {
  if (_catalog) return _catalog;
  const r = await fetch("./models/catalog.json");
  if (!r.ok) throw new Error("catalog " + r.status);
  _catalog = await r.json();
  return _catalog;
}

export function catalogModels(cat) {
  return (cat || _catalog)?.models || [];
}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 2);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (db.objectStoreNames.contains(STORE)) db.deleteObjectStore(STORE);
      db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function valueBytes(v) {
  if (!v) return 0;
  if (typeof v.byteLength === "number") return v.byteLength;
  if (typeof v.size === "number") return v.size;
  return 0;
}

export function loadPrefs() {
  try {
    const raw = JSON.parse(localStorage.getItem(PREF_KEY) || "null");
    if (raw && typeof raw === "object") {
      if (raw.models) return raw;
      return { models: {} };
    }
  } catch {
    /* */
  }
  return { models: {} };
}

export function savePrefs(p) {
  localStorage.setItem(PREF_KEY, JSON.stringify(p));
}

export async function idbHas(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, "readonly").objectStore(STORE).getKey(id);
    req.onsuccess = () => resolve(req.result === id);
    req.onerror = () => reject(req.error);
  });
}

export async function idbGet(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, "readonly").objectStore(STORE).get(id);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function idbPut(id, buf) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, "readwrite").objectStore(STORE).put(buf, id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function idbDelete(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, "readwrite").objectStore(STORE).delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function idbClear() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, "readwrite").objectStore(STORE).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Sum ArrayBuffer sizes currently in the weights store. */
export async function idbUsage() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const keys = [];
    let bytes = 0;
    const req = db.transaction(STORE, "readonly").objectStore(STORE).openCursor();
    req.onsuccess = (ev) => {
      const cur = ev.target.result;
      if (cur) {
        keys.push(String(cur.key));
        bytes += valueBytes(cur.value);
        cur.continue();
      } else {
        resolve({ keys, bytes });
      }
    };
    req.onerror = () => reject(req.error);
  });
}

export async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body || !res.body.getReader) {
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let rec = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    rec += value.byteLength;
    onProgress?.(rec, total);
  }
  const out = new Uint8Array(rec);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out.buffer;
}

export async function loadWeightBuffer(dtype, { onProgress, cache = true, file } = {}) {
  const meta = file ? { id: dtype, file } : DTYPES.find((d) => d.id === dtype);
  if (!meta) throw new Error(dtype);
  if (cache) {
    const hit = await idbGet(dtype);
    if (hit) {
      onProgress?.(hit.byteLength, hit.byteLength);
      return hit;
    }
  }
  const buf = await fetchWithProgress(meta.file, onProgress);
  if (cache) await idbPut(dtype, buf);
  return buf;
}
