# MicroLLM lab — conversion and harness report

## What shipped

A static WebGPU app that chats with and benchmarks **Q4** packs of several
small LMs in the browser (Llama-style and GPT-2/nanoGPT). Live:
https://stateofutopia.com/experiments/microllmlab

## Models converted

All Q4 group-32, PGW1 container. Llama-style context capped at 2048; GPT-2 at 1024.

| id | upstream | params | Q4 bytes | template |
| --- | --- | ---: | ---: | --- |
| petitgpt | yangqi0/petitgpt | 124,635,456 | 78,046,720 | role tokens |
| smollm2-135m-instruct | HuggingFaceTB/SmolLM2-135M-Instruct | 134,515,008 | 84,221,440 | ChatML |
| smollm-135m-instruct | HuggingFaceTB/SmolLM-135M-Instruct | 134,515,008 | 84,221,440 | ChatML |
| l20-edu-135m | AliceYin/l20-edu-135m | 134,515,008 | 84,221,440 | ChatML |
| smollm2-360m-instruct | HuggingFaceTB/SmolLM2-360M-Instruct | 361,821,120 | 226,382,256 | ChatML |
| minimind2 | jingyaogong/MiniMind2 | 104,030,976 | 65,121,632 | ChatML |
| minimind2-small | jingyaogong/MiniMind2-Small | 25,829,888 | 16,181,568 | ChatML |
| gpt2 | openai-community/gpt2 | 124,439,808 | 80,854,688 | completion |

PGW header byte 5 selects the decoder: 0 = Llama (RMSNorm, RoPE, SwiGLU, GQA),
1 = GPT-2 (LayerNorm, learned WPE, GELU, MHA, biases). Metal and fused-NVIDIA
paths exist for both. GPT-2 greedy next-token on
`Hello, I'm a language model,` matches Hugging Face (id 407, ` not`).

## Decoder

Safari/Metal path: cooperative multi-workgroup GEMV, threadgroup-cached
activations, fused RMS/SiLU, chained decode for chat. Suite decode is
one token at a time so EOS stops the GPU. Config (d_model, heads, layers,
vocab, rope θ) comes from the PGW header, so 576-d and 960-d models share
the same kernels.

## Safari checks (M4)

- PetitGPT “hello”: 87 ms, 115 tok/s, `Hello! How can I help you today?`
- SmolLM2-135M-Instruct “hello”: 227 ms, 66 tok/s, coherent greeting, EOS
- SmolLM2-360M-Instruct loaded and ran (72 tok/s on a 48-token sample)
- SmolLM2-135M-Instruct objective suite: **14/20** pass, wall ~5.8 s
  (gets `1+1=2`, `2+2=4`, Berlin, ORANGE-42, rusty nails)
- PetitGPT remains the latency baseline; instruct models win accuracy

## Files people actually need

`index.html`, `app.js`, `engine/*`, `models/catalog.json`, each
`models/<id>/{model.q4.bin,tokenizer.json,card.json}`. Serve over HTTP.
