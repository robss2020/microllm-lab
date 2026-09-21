# PetitGPT Safari / Metal speedup

Mac mini M4 (10-core GPU, 24 GB, Safari 26.6). Goal: make the WebGPU decoder fast on this machine, especially Q4, without breaking greedy decode.

## What was actually slow

The NVIDIA-style decoder is **one workgroup of 256 threads** for the whole transformer. On a 10-core M4 GPU that leaves most of the chip idle.

A previous “Metal Q4 multi-workgroup” path existed on disk, but Safari’s WGSL compiler rejected it (`unresolved identifier 'PACK'` — helpers ran before the binding). The app caught that and **silently used the 1-WG fused kernel**. So the brief’s Mac Q4 number was not this kernel, and not this Safari.

Q4 is memory-light (~62 MB/token). At 120 GB/s the weight traffic is ~0.5 ms. The 1-WG fused kernel was doing 64 tokens in **1220 ms (52 tok/s)** — occupancy bound, not bandwidth bound.

## What we changed

All of this is the Safari/Metal path (`engine/webgpu-metal.js`). Windows/NVIDIA still uses the fused 1-WG kernel.

1. **Many workgroups per GEMV.** Small matrices (576–1536 rows) use 8 rows × 32 threads per workgroup so a 576-row multiply launches 72 workgroups instead of 3. The vocab head stays 1 thread/row (125 workgroups). Apple SIMD groups are 32 wide; that mapping matches.
2. **Threadgroup cache of the activation.** Every row of a GEMV needs the same 576- or 1536-vector. It is loaded once per workgroup instead of once per row.
3. **Fewer kernels.** RMS is computed into that shared vector inside the GEMV. SiLU is applied while loading w2’s source. w1 and w3 are one 3072×576 multiply (they already sit back-to-back in memory). RoPE-into-KV is inside attention. Dynamic uniform offsets so we do not create thousands of bind groups.
4. **One GPU round-trip per decode batch.** Embed for token t+1 reads the argmax written for token t, in the same compute pass. 63 decode steps, one `mapAsync`. JS command encoding is ~1 ms; the GPU owns the rest.
5. **Coalesced packed loads.** Q4/Q8 walk packed `u32` words so consecutive threads hit consecutive addresses. That was the second big Q4 jump (516 ms → 383 ms) and a solid Q8 jump (427 → 391 ms).

Greedy text is **token-identical** to the fused kernel for q4, q8, f16, and f32 on a 64-token story. Chat checks: hello and “capital of France” stay coherent.

## Numbers

Same prompt, 64 new tokens, EOS disabled, Safari, 1 warmup + 2 runs:

| dtype | fused 1-WG | metal (final) | tok/s | vs fused on this Mac |
| ----- | ---------- | ------------- | ----- | -------------------- |
| q4    | 1220 ms    | **383 ms**    | 167   | **3.2×**             |
| q8    | 1396 ms    | **391 ms**    | 164   | **3.6×**             |
| f16   | 1622 ms    | **467 ms**    | 137   | **3.5×**             |
| bf16  | —          | **471 ms**    | 136   | —                    |
| f32   | 2691 ms    | **624 ms**    | 102   | **4.3×**             |

Against the brief’s Windows GTX 1060 Q4 (4028 ms) that is about **10×** if the workloads are similar; against the brief’s Mac Q4 (4450 ms) about **12×**. Treat those last two as indicative: the brief did not record prompt length or EOS.

Milestones: fused baseline 52 tok/s → occupancy GEMV + fusion + chained decode 124 tok/s → coalesced Q4 **168 tok/s**. WG=128 was a regression and was reverted.

## Why the Benchmarks tab was not 10×

Chat at ~160 tok/s is a **long decode** (dozens of new tokens, one prefill). The built-in suite is **14 independent short chats** (`maxNew=48`, most hit EOS after a handful of tokens).

The first Metal decoder chained the entire `maxNew` on the GPU before looking at EOS. Hello (10 tokens) still paid for 47 transformer steps. Windows/fused **stop at EOS**, so the suite wall (~4 s Q4) was 14 × (prefill + ~10–20 tokens). We were doing 14 × (prefill + 48 tokens) with a faster kernel — about **10% faster**, which matches what you saw.

A later batched-prefill / strided-scratch experiment **regressed chat tok/s** (extra GPU round-trips on every decode, heavier layout). That path was reverted.

What stayed:

1. The compact multi-workgroup GEMV (the ~160–170 tok/s chat path).
2. **Stop at EOS in the suite** (1-token decode steps when not streaming). Chat still chains the remaining tokens in one pass.

Built-in suite on this Safari, Q4:

| path | suite wall | vs Windows 4028 ms |
| ----- | ---------- | ------------------ |
| fused 1-WG | 4621 ms | ~same as the brief Mac 4450 |
| metal, chain-through-EOS | ~4 s | ~10% faster |
| metal, compact GEMV + EOS stop | **1541 ms** | **2.6×** |

Long story decode **384 ms / 166 tok/s**. Hello **82 ms** (10 tokens, EOS). Chat streaming uses the chained path, so tok/s stays in the 160s.

## What we stopped doing

Further GEMV tweaks (smaller workgroups, F16 packed-word, F32 vec4) were flat or slower. Q4 still only uses ~10% of peak memory bandwidth; the rest is dequant work plus ~150 serial dispatches per token. Collapsing those dispatches needs a GPU-wide barrier, which WebGPU does not provide. That is the plateau.

How to reproduce: `python3 tools/serve.py`, then open  
`http://127.0.0.1:8080/?auto=bench&dtype=q4&max=64&eos=0&runs=2&warmup=1`.
