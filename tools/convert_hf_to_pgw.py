#!/usr/bin/env python3
"""Convert a Llama-style HuggingFace causal LM to PGW1 Q4 (group 32)."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

GROUP = 32


def quant_q4(mat: np.ndarray, group: int = GROUP):
    rows, cols = mat.shape
    if cols % group:
        raise ValueError(f"cols {cols} not divisible by group {group}")
    ng = cols // group
    scales = np.empty((rows, ng), np.float32)
    packed = np.empty((rows, cols // 2), np.uint8)
    for r in range(rows):
        row = mat[r]
        for g in range(ng):
            chunk = row[g * group : (g + 1) * group]
            amax = float(np.max(np.abs(chunk)))
            sc = amax / 8.0 if amax > 0 else 1.0
            scales[r, g] = sc
            q = np.clip(np.rint(chunk / sc) + 8.0, 0, 15).astype(np.uint8)
            packed[r, g * (group // 2) : (g + 1) * (group // 2)] = q[0::2] | (q[1::2] << 4)
    return packed, scales


def add_tensor(entries, blobs, name, arr, storage):
    if storage == "q4":
        packed, scales = quant_q4(arr)
        scale_off = blobs.tell()
        blobs.write(scales.tobytes())
        data_off = blobs.tell()
        blobs.write(packed.tobytes())
        entries.append(
            {
                "name": name,
                "shape": [int(arr.shape[0]), int(arr.shape[1])],
                "kind": "q4",
                "storage": "q4",
                "group": GROUP,
                "scaleOffset": scale_off,
                "scaleNbytes": scales.nbytes,
                "offset": data_off,
                "nbytes": packed.nbytes,
            }
        )
    elif storage == "f32":
        data = np.ascontiguousarray(arr, dtype=np.float32)
        off = blobs.tell()
        blobs.write(data.tobytes())
        entries.append(
            {
                "name": name,
                "shape": list(data.shape),
                "kind": "q4",
                "storage": "f32",
                "offset": off,
                "nbytes": data.nbytes,
            }
        )
    else:
        raise ValueError(storage)


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def convert(hf_id: str, out_dir: Path, max_seq: int = 2048):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {hf_id}", flush=True)
    cfg = AutoConfig.from_pretrained(hf_id)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval()
    m = model.model if hasattr(model, "model") else model

    n_layers = int(cfg.num_hidden_layers)
    d_model = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    d_ff = int(cfg.intermediate_size)
    vocab = int(cfg.vocab_size)
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)
    hd = d_model // n_heads
    if d_model % n_heads:
        raise SystemExit(f"d_model {d_model} not divisible by n_heads {n_heads}")

    from io import BytesIO

    blobs = BytesIO()
    entries = []

    lm = model.lm_head.weight if hasattr(model, "lm_head") else m.embed_tokens.weight
    add_tensor(entries, blobs, "lm_head", to_np(lm), "q4")

    layers = m.layers
    for i, layer in enumerate(layers):
        add_tensor(entries, blobs, f"blocks.{i}.norm1", to_np(layer.input_layernorm.weight), "f32")
        q = to_np(layer.self_attn.q_proj.weight)
        k = to_np(layer.self_attn.k_proj.weight)
        v = to_np(layer.self_attn.v_proj.weight)
        qkv = np.concatenate([q, k, v], axis=0)
        add_tensor(entries, blobs, f"blocks.{i}.attn.qkv", qkv, "q4")
        add_tensor(entries, blobs, f"blocks.{i}.attn.proj", to_np(layer.self_attn.o_proj.weight), "q4")
        add_tensor(entries, blobs, f"blocks.{i}.norm2", to_np(layer.post_attention_layernorm.weight), "f32")
        add_tensor(entries, blobs, f"blocks.{i}.mlp.w1", to_np(layer.mlp.gate_proj.weight), "q4")
        add_tensor(entries, blobs, f"blocks.{i}.mlp.w3", to_np(layer.mlp.up_proj.weight), "q4")
        add_tensor(entries, blobs, f"blocks.{i}.mlp.w2", to_np(layer.mlp.down_proj.weight), "q4")
        print(f"  layer {i+1}/{n_layers}", flush=True)

    add_tensor(entries, blobs, "norm_f", to_np(m.norm.weight), "f32")
    payload = blobs.getvalue()

    table = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    table_off = 256
    pad = (16 - ((table_off + len(table)) % 16)) % 16
    payload_off = table_off + len(table) + pad

    header = bytearray(256)
    header[0:4] = b"PGW1"
    header[4] = 4
    header[5] = 0  # llama-style
    struct.pack_into("<8I", header, 8, vocab, n_layers, d_model, n_heads, n_kv, d_ff, max_seq, GROUP)
    struct.pack_into("<III", header, 40, table_off, len(table), payload_off)
    struct.pack_into("<d", header, 56, rope_theta)
    struct.pack_into("<d", header, 64, 1.0)

    bin_path = out_dir / "model.q4.bin"
    with bin_path.open("wb") as f:
        f.write(header)
        f.write(table)
        f.write(b"\x00" * pad)
        f.write(payload)

    tok.save_pretrained(out_dir)
    # Keep tokenizer.json; drop pytorch leftovers if the tokenizer writer added them
    for extra in ("tokenizer_config.json", "special_tokens_map.json"):
        p = out_dir / extra
        if p.exists():
            pass

    tcfg = {}
    tcfg_path = out_dir / "tokenizer_config.json"
    if tcfg_path.exists():
        tcfg = json.loads(tcfg_path.read_text())

    specials = {}
    for t in getattr(tok, "all_special_tokens", []) or []:
        tid = tok.convert_tokens_to_ids(t)
        specials[t] = int(tid) if tid is not None else None

    card = {
        "hf": hf_id,
        "vocabSize": vocab,
        "nLayers": n_layers,
        "dModel": d_model,
        "nHeads": n_heads,
        "nKvHeads": n_kv,
        "dFf": d_ff,
        "maxSeqLen": max_seq,
        "ropeTheta": rope_theta,
        "headDim": hd,
        "params": int(sum(p.numel() for p in model.parameters())),
        "q4Bytes": bin_path.stat().st_size,
        "bosId": int(getattr(tok, "bos_token_id", None) or getattr(cfg, "bos_token_id", 0) or 0),
        "eosId": int(getattr(tok, "eos_token_id", None) or getattr(cfg, "eos_token_id", 0) or 0),
        "padId": int(getattr(tok, "pad_token_id", None) or getattr(cfg, "pad_token_id", 0) or 0),
        "unkId": int(getattr(tok, "unk_token_id", None) or 0),
        "bosToken": tok.bos_token,
        "eosToken": tok.eos_token,
        "chatTemplate": tcfg.get("chat_template"),
        "specials": specials,
        "qGroup": GROUP,
    }
    (out_dir / "card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB) params={card['params']}", flush=True)
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_id")
    ap.add_argument("out_dir")
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()
    convert(args.hf_id, Path(args.out_dir), args.max_seq)


if __name__ == "__main__":
    main()
