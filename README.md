# MicroLLM lab

**Run a 135M-class language model in the tab you already have open.**

No API key. No Python. No CUDA install. WebGPU talks to the GPU in Safari, Chrome, or Edge; weights stay in this origin’s IndexedDB. Q4 is small enough that a Mac mini from this year answers in well over a hundred tokens per second, and a decade-old GTX 1060 still keeps up.

**Try it live:** [stateofutopia.com/experiments/microllmlab](https://stateofutopia.com/experiments/microllmlab)

**Download everything:** [current.zip (591 MB)](https://stateofutopia.com/experiments/microllmlab/current.zip)

---

## Why would anyone bother?

Cloud models are better. That is not the point.

- **Privacy is the default.** The prompt never leaves the machine. That matters for drafts, medical notes you should not paste into a chatbot, and anything you would not put in a vendor log.
- **Latency is a round trip you do not pay.** First token is local. There is no queue, no cold start, no “capacity exceeded.”
- **Cost is zero after the download.** A 80 MB Q4 file is cheaper than a single busy afternoon of API calls, and it still works on a plane.
- **The models are honest about being small.** A 135M network will fail arithmetic and invent facts. This lab *measures* that, instead of hiding it behind a chat skin.

If you are shipping an on-device feature, this is a way to feel the quality/speed tradeoff in the same browser your users have.

## What’s in the box

Q4 (group-32) conversions of redistributable checkpoints, plus Metal and fused-NVIDIA WebGPU decoders (Llama-style GQA/SwiGLU and GPT-2/nanoGPT LayerNorm+GELU). Pick a model at the top of the lab; the Compare tab charts speed and accuracy from suite runs in *this* browser.

| Model | Params | Q4 | License | Who |
| --- | ---: | ---: | --- | --- |
| PetitGPT research-v1 | 124.6M | 74 MB | Apache-2.0 | yangqi0 |
| SmolLM2 135M Instruct | 134.5M | 80 MB | Apache-2.0 | Hugging Face Smol Models Research |
| SmolLM 135M Instruct | 134.5M | 80 MB | Apache-2.0 | Hugging Face Smol Models Research |
| L20-Edu 135M | 134.5M | 80 MB | Apache-2.0 | AliceYin |
| SmolLM2 360M Instruct | 362M | 216 MB | Apache-2.0 | Hugging Face Smol Models Research |
| MiniMind2 104M | 104M | 62 MB | Apache-2.0 | jingyaogong |
| MiniMind2 Small 26M | 26M | 15 MB | Apache-2.0 | jingyaogong |
| GPT-2 124M | 124M | 77 MB | MIT | OpenAI (Radford et al.) |

### PetitGPT research-v1

GitHub user **[yangqi0](https://github.com/yangqi0/petitgpt)** trained PetitGPT as a small language-model research project on a single RTX 4090 with roughly 13 billion pretraining token positions: a 124.6M Llama-style decoder (pre-norm RMSNorm, grouped-query attention 9/3, RoPE, SwiGLU, tied embeddings), Apache-2.0. This lab is a WebGPU port of that architecture and checkpoint — not a replacement of that work. It is usually the fastest model here and the reference baseline the others are measured against.

### SmolLM2 135M Instruct

Second-generation small-model line from **Hugging Face’s Smol Models Research** (Loubna Ben Allal and colleagues), 2025, Apache-2.0. Pretrained on about 2 trillion tokens, then instruction-tuned. Same 30×576 GQA shape as PetitGPT, so it is a fair speed-vs-accuracy comparison: slower (larger vocab, ChatML prefix), usually much better on the objective suite.

### SmolLM 135M Instruct

The **2024** first-generation SmolLM from the same Hugging Face group, pretrained on a Cosmopedia / FineWeb-style mix of about 600 billion tokens, then instruction-tuned. Same size class as SmolLM2. Useful if you want to see what a year of data and recipe changes bought.

### L20-Edu 135M

**AliceYin** trained this 135M Llama-style model on a single NVIDIA L20 with about 13 billion tokens (Apache-2.0). An educational, small-budget counterpart to the Hugging Face 135M runs: same shape, far less pretraining compute. Expect weaker answers; the point is an auditable “what one GPU can do” checkpoint.

### SmolLM2 360M Instruct

The 360M instruct sibling in Hugging Face’s SmolLM2 family (2025, Apache-2.0): 32 layers, 960-d, still Q4 in the browser. Slower to load and to decode; usually sharper on the suite.

### MiniMind2 104M

**jingyaogong** built MiniMind as a from-scratch teaching project: train a tiny LLM on one consumer GPU, with open code aimed at Chinese learners. The Hugging Face MiniMind2 export (Apache-2.0) is Llama-style (16×768, GQA 8/2, 6.4k vocab, ChatML). Stronger in Chinese than English. Native training code has extras this decoder does not need; we ship the Llama export.

### MiniMind2 Small 26M

The 26M teaching checkpoint from the same MiniMind line (8 layers × 512-d). Fastest to download and to run. Expect short, shaky English; better luck with Chinese.

### GPT-2 124M (nanoGPT-shaped)

**OpenAI**, February 2019 (Alec Radford et al.), later re-released MIT as `openai-community/gpt2`. Twelve layers, LayerNorm, GELU, learned absolute positions, 50k BPE — the architecture Andrej Karpathy’s **nanoGPT** trains. It is a completion model, not chat-tuned. Greedy decode often loops; that is the checkpoint, not a decoder bug.

## Charts

The **Benchmarks** tab plots per-test wall time and pass/fail after a suite. **Compare** keeps the latest suite per model and charts accuracy (%), mean tok/s, and suite wall time. Charts use [Chart.js](https://www.chartjs.org/) (MIT), vendored at `vendor/chart.umd.min.js` so the zip works offline.

## So what did we measure?

On an Apple M4 (Safari, Metal Q4), greedy decode:

| | PetitGPT | SmolLM2 135M Instruct |
| --- | ---: | ---: |
| “Say hello…” | 87 ms · 115 tok/s | 227 ms · 66 tok/s |
| Objective suite | 10/14 on the original short set | **14/20** on the expanded set (including `1+1=2`, `2+2=4`, Berlin, copy-from-context) |

The instruct checkpoint is slower (larger vocab, ChatML prefix) and *noticeably* more accurate on the dumb tests a 135M model might still pass. That is the comparison this lab exists to make.

Suite wall time on this M4 is about **1.5 s** for PetitGPT and about **6 s** for SmolLM2-135M-Instruct (20 tests, stop at EOS). The UI estimates duration from your last tok/s before you click **Run**.

## Credit

PetitGPT — the architecture, the research checkpoint, and the reason this lab exists — is **[yangqi0/petitgpt](https://github.com/yangqi0/petitgpt)** (Apache-2.0). This repository is a WebGPU port, a Q4 packer, and a multi-model harness. It is not a substitute for that work.

SmolLM / SmolLM2: Hugging Face Smol Models Research, Apache-2.0.  
L20-Edu-135M: [AliceYin/l20-edu-135m](https://huggingface.co/AliceYin/l20-edu-135m), Apache-2.0.  
MiniMind2: [jingyaogong/MiniMind2](https://huggingface.co/jingyaogong/MiniMind2), Apache-2.0.  
GPT-2 124M: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), MIT.

## Run it yourself

Do not open `index.html` as `file://`. Modules, workers, and `fetch` need an origin.

```bash
python3 -m http.server 
# open http://127.0.0.1:8000/
```

Pick a model at the top of the page. Nothing is downloaded until you tap **Download** on a tile, **Download all**, or Generate (which fetches the active model). Each tile shows whether weights are on this device; while a fetch is running it shows speed and time left. **Discard** / **Discard all** drop IndexedDB copies. The bar reports total stored size. Then chat or run the suite; **Compare** charts speed and accuracy.

Convert another Llama-style Hugging Face checkpoint (Q4 only):

```bash
python3 -m venv .venv && .venv/bin/pip install torch transformers safetensors
.venv/bin/python tools/convert_hf_to_pgw.py HuggingFaceTB/SmolLM2-135M-Instruct models/my-model
```

GPT-2 / nanoGPT-shaped:

```bash
.venv/bin/python tools/convert_gpt2_to_pgw.py openai-community/gpt2 models/gpt2
```

Then add a row to `models/catalog.json`.

## Limits

These models will hallucinate. Do not use them for high-stakes answers. Context is capped at 2048 tokens in this pack so the KV cache fits in a laptop GPU. Greedy decode only.

## License

Lab code: Apache-2.0. Each weight file keeps its upstream license (Apache-2.0 except GPT-2, which is MIT). Chart.js is MIT. See `NOTICE`.
