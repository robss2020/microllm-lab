MicroLLM on-device lab
======================

Static port of yangqi0/petitgpt (research-v1, alpha075, 124,635,456 params).

Harness: Apple/Safari uses a multi-workgroup Metal decoder for every dtype
(cooperative GEMV, threadgroup-cached activations, fused RMS/SiLU, chained
decode). NVIDIA keeps the fused 1-WG kernel. Add ?fused=1 to force 1-WG on Mac.

Serve this folder over HTTP (any static server). Opening index.html as file://
will fail because ES modules, workers, and fetch require an origin.

    python3 -m http.server 8000

Weights
-------
weights/petitgpt.f32.bin   native FP32 (parity-checked vs PyTorch greedy)
weights/petitgpt.f16.bin   IEEE fp16
weights/petitgpt.bf16.bin  bfloat16
weights/petitgpt.q8.bin    per-row int8
weights/petitgpt.q4.bin    grouped int4 (group 32)

Check only the sizes you want. Unchecked files are not fetched; cached copies
are deleted from IndexedDB.

Backends
--------
1. WebGPU (preferred) — main thread (Safari / Apple Silicon included).
   On Metal: many workgroups per GEMV, native q4/q8/f16/bf16/f32, one compute
   pass per decode batch. On NVIDIA: fused 1-WG decoder.
2. WASM SIMD GEMV (kernels.wasm) + JS transformer
3. Pure JS (the FP32 path was matched token-for-token against the official
   native checkpoint on "Say hello in one sentence.")

The official repo is CUDA-only. This port reimplements the graph:
pre-norm RMSNorm, fused QKV GQA 9/3, Llama-style RoPE, SwiGLU, tied embeddings,
greedy argmax, chat template [BOS] <|user|> … <|assistant|>.

License
-------
Model, tokenizer, and this port: Apache-2.0 (upstream).
Do not use for high-stakes answers — the 124.6M checkpoint is a small language-model research project.
