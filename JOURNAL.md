# PetitGPT Metal/Safari optimization journal

Machine: Mac mini M4, 10-core GPU, 24 GB unified LPDDR5X (~120 GB/s), Safari 26.6, Metal 4.

Windows reference from the brief (GTX 1060 6GB + i7-3770):

| dtype | Windows ms | brief Mac ms |
| ----- | ---------- | ------------ |
| q4    | 4028       | 4450         |
| q8    | 6680       | 5050         |
| bf16  | 7149       | 5915         |
| f16   | 6860       | 5859         |
| f32   | 11836      | 9807         |

Fair methodology used here (unless noted): Safari on `127.0.0.1`, prompt *Write a long story about a brave mouse…*, chat template, **64 new tokens**, EOS ignored so every dtype does the same amount of work, 1 warmup + 2 timed runs.

## Pipeline

124.6M params, 30 layers, d=576, GQA 9/3, d_ff=1536, vocab=32000.

Per token: ~125M MACs and a full weight read (Q4 ~62 MB). At 120 GB/s that is ~0.5 ms of Q4 traffic, so a well-fed GPU should be tens–hundreds of tok/s. Anything around 15 tok/s is occupancy/latency, not bandwidth.

Two decoder shapes we started with:

1. **Fused 1-WG** (NVIDIA path): one workgroup of 256 walks the whole transformer. One of ten M4 GPU cores.
2. **Old Metal Q4 split**: one thread per output row. LM head launches 125 WGs (fine). Layer GEMVs are 576–1536 rows → **3–6 WGs**. 85% of MACs ran almost idle. Also: no threadgroup cache of the activation vector, ~10 dispatches/layer, `mapAsync` every token.

Safari WGSL does not allow helpers to mention `PACK` before the binding is declared. The old metal shaders did exactly that (`WGSL:43:18 unresolved identifier 'PACK'`), so **the previous metal-q4 path never ran on this Safari** — it silently fell back to fused 1-WG.

## Log

### Baseline (fused 1-WG, this Mac)

Story prompt, 64 tokens:

| dtype | ms   | tok/s |
| ----- | ---- | ----- |
| q4    | 1220 | 52.5  |
| q8    | 1396 | 45.8  |
| f16   | 1622 | 39.5  |
| f32   | 2691 | 23.8  |

Hello (EOS at 10 tokens) was ~243 ms / 41 tok/s — not comparable across dtypes because they hit EOS at different lengths. Later benches force 64 tokens.

Output of fused q4 on the story prompt is coherent and stable across runs.

### v1 — new metal path

- Bindings declared first (Safari WGSL).
- Cooperative GEMV: WG=256, 8 rows × 32 threads (one SIMD group per row on Apple). 576-row matrices → 72 WGs instead of 3.
- Fat GEMV for vocab (1 thread/row, 125 WGs).
- Source vector in threadgroup memory (576 or 1536 floats).
- RMS fused into the GEMV that consumes xn; SiLU fused into w2; w1+w3 as one 3072×576 GEMV (they are contiguous in PACK and in scratch).
- RoPE KV fused into attention.
- Dynamic uniform offsets (one bind group per pipeline, not thousands).
- GPU-chained decode: embed reads previous argmax from `OUT` inside one compute pass. 63 decode steps, one `mapAsync`.

Result q4: **516 ms, 124 tok/s**. Text **identical** to fused. GPU wait ~400 ms, JS encode ~1 ms.

### v2 — coalesced Q4

Old coop Q4 walked **groups** (32 elements). 18 groups and 32 threads left 14 threads idle, and consecutive threads jumped 16 bytes.

New: walk **packed words** (8 nibbles). 72 words / 32 threads, coalesced u32 loads, every thread busy.

q4: **381 ms, 168 tok/s**. Text still matches fused. GPU wait 295 ms.

WG=128 / ROW_TILE=4 (v3) was a **regression** (474 ms). Reverted to 256/8.

### v4 — packed-word Q8 / F16 / BF16, vec4 F32

Same idea as v2 for the other dtypes. Q8 427→391 ms (text match). F16/BF16/F32 within noise. Keep v4.

### Correctness

Greedy text of metal vs fused, same dtype, 64-token story: **exact match** for q4, q8, f16, f32.

Prompts:

- “Say hello in one sentence.” → `Hello! How can I help you today?`
- “What is the capital of France?” q4 → `Paris is the capital of France.` f16/f32 → `The capital of France is Paris.`
- “What is 2 + 2?” → rambling (the 125M checkpoint is weak at arithmetic; still grammatical)

### Things that did not help

- WG=128, 4 rows/WG: −24% vs 256/8.
- Packed-word F16/BF16: flat (already unpack2x16float).
- F32 vec4-strided: ~2%, inside noise.
- Prompt-lookup speculation is the fused path; metal chaining already removes per-token round-trips. Story prompts do not n-gram-hit enough to beat a full 64-token GPU chain.
- Persistent device-wide atomic barrier: not attempted. Deadlock risk if Metal does not keep every workgroup resident.

### Benchmarks tab vs chat tok/s

Chat ~160 tok/s is long decode. The suite is 14 short EOS-stopped prompts. Metal was chaining all 48 `maxNew` tokens on the GPU, so suite wall stayed ~4 s (10% vs Windows 4028 ms). Fused 1-WG suite on this Mac: **4621 ms** (matches the brief Mac 4450).

Batched prefill + strided TMAX scratch + exponential decode batches **regressed chat tok/s**. Reverted those kernels to the compact T=1 GEMV.

Kept: suite (no `onToken`) decodes one token at a time and stops at EOS; chat still chains a full remaining batch.

Rechecked in Safari: story **384 ms / 166 tok/s**; hello **82 ms**; Q4 suite wall **1541 ms**.

### Plateau

Q4 GPU time is ~4.7 ms/token vs ~0.5 ms of raw Q4 weight traffic (about 10× off peak bandwidth). Remaining cost is dequant ALUs, ~150 serial dispatches/token (Metal launch), and the serial layer DAG. Further GEMV micro-opts bounced. Dispatch fusion would need a device barrier, which WebGPU does not have.

## Final 64-token numbers (Safari, this Mac)

| dtype | fused 1-WG ms | metal ms | metal tok/s | vs fused | vs brief Mac | vs brief Win |
| ----- | ------------- | -------- | ----------- | -------- | ------------ | ------------ |
| q4    | 1220          | 383      | 167         | 3.2×     | 11.6×        | 10.5×        |
| q8    | 1396          | 391      | 164         | 3.6×     | 12.9×        | 17.1×        |
| f16   | 1622          | 467      | 137         | 3.5×     | 12.5×        | 14.7×        |
| bf16  | —             | 471      | 136         | —        | 12.6×        | 15.2×        |
| f32   | 2691          | 624      | 102         | 4.3×     | 15.7×        | 19.0×        |

Brief wall times are a different (unknown) workload; ratios assume they are comparable. The apples-to-apples figure on this Mac is **metal vs fused**.
