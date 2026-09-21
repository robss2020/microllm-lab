/** WASM SIMD GEMV accelerator. Weights stay in JS; rows are tiled into WASM memory. */

let exports_ = null;
let memory = null;
let heap = null;
let heapU8 = null;

export async function initWasm(url = new URL("./kernels.wasm", import.meta.url)) {
  const bytes = await fetch(url).then((r) => r.arrayBuffer());
  const result = await WebAssembly.instantiate(bytes, {});
  exports_ = result.instance.exports;
  memory = exports_.memory;
  growHeap();
  return { ok: true, version: exports_.kernel_version() };
}

function growHeap() {
  heap = new Float32Array(memory.buffer);
  heapU8 = new Uint8Array(memory.buffer);
}

function ensure(bytes) {
  const need = (bytes + 65535) >> 16;
  const have = memory.buffer.byteLength >> 16;
  if (need > have) memory.grow(need - have);
  growHeap();
}

const ALIGN = 16;
function align(n) {
  return (n + ALIGN - 1) & ~(ALIGN - 1);
}

export function wasmAvailable() {
  return !!exports_;
}

export function gemvF32Wasm(m, n, w, x, y) {
  if (!exports_) return false;
  const xBytes = align(n * 4);
  const yBytes = align(m * 4);
  const tile = Math.min(m, 96);
  const wBytes = align(tile * n * 4);
  ensure(xBytes + yBytes + wBytes + 64);
  const xPtr = 0;
  const yPtr = xBytes;
  const wPtr = xBytes + yBytes;
  heap.set(x.subarray(0, n), xPtr / 4);
  for (let i = 0; i < m; i += tile) {
    const rows = Math.min(tile, m - i);
    heapU8.set(
      new Uint8Array(w.buffer, w.byteOffset + i * n * 4, rows * n * 4),
      wPtr,
    );
    exports_.gemv_f32(rows, n, wPtr, xPtr, yPtr);
    y.set(heap.subarray(yPtr / 4, yPtr / 4 + rows), i);
  }
  return true;
}
