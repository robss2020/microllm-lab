#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 (nanoGPT-shaped) to PGW1 Q4. Arch byte = 1."""
from __future__ import annotations

import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

GROUP = 32
ARCH_GPT2 = 1


def quant_q4(mat: np.ndarray, group: int = GROUP):
    rows, cols = mat.shape
    if cols % group:
        raise ValueError(f"cols {cols} not divisible by {group}")
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


def add_q4(entries, blobs, name, arr):
    packed, scales = quant_q4(np.ascontiguousarray(arr, np.float32))
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


def add_f32(entries, blobs, name, arr):
    data = np.ascontiguousarray(arr, np.float32)
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


def conv1d(w):
    # HF Conv1D weight is [in, out]; we want [out, in] row-major GEMV.
    return w.detach().float().cpu().numpy().T


def bias(t):
    return t.detach().float().cpu().numpy()


def convert(hf_id: str, out_dir: Path, max_seq: int = 1024):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading gpt2 {hf_id}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(hf_id)
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    model.eval()
    cfg = model.config
    n_layers = int(cfg.n_layer)
    d = int(cfg.n_embd)
    nh = int(cfg.n_head)
    vocab = int(cfg.vocab_size)
    n_ctx = min(int(cfg.n_positions), max_seq)
    d_ff = 4 * d
    tr = model.transformer

    blobs = BytesIO()
    entries = []
    add_q4(entries, blobs, "lm_head", tr.wte.weight.detach().float().cpu().numpy())
    add_f32(entries, blobs, "wpe", tr.wpe.weight.detach().float().cpu().numpy()[:n_ctx])
    ln_f = np.concatenate([bias(tr.ln_f.weight), bias(tr.ln_f.bias)])
    add_f32(entries, blobs, "ln_f", ln_f)

    for i, block in enumerate(tr.h):
        ln1 = np.concatenate([bias(block.ln_1.weight), bias(block.ln_1.bias)])
        add_f32(entries, blobs, f"blocks.{i}.ln1", ln1)
        add_q4(entries, blobs, f"blocks.{i}.attn.qkv", conv1d(block.attn.c_attn.weight))
        add_f32(entries, blobs, f"blocks.{i}.attn.qkv_bias", bias(block.attn.c_attn.bias))
        add_q4(entries, blobs, f"blocks.{i}.attn.proj", conv1d(block.attn.c_proj.weight))
        add_f32(entries, blobs, f"blocks.{i}.attn.proj_bias", bias(block.attn.c_proj.bias))
        ln2 = np.concatenate([bias(block.ln_2.weight), bias(block.ln_2.bias)])
        add_f32(entries, blobs, f"blocks.{i}.ln2", ln2)
        add_q4(entries, blobs, f"blocks.{i}.mlp.fc", conv1d(block.mlp.c_fc.weight))
        add_f32(entries, blobs, f"blocks.{i}.mlp.fc_bias", bias(block.mlp.c_fc.bias))
        add_q4(entries, blobs, f"blocks.{i}.mlp.proj", conv1d(block.mlp.c_proj.weight))
        add_f32(entries, blobs, f"blocks.{i}.mlp.proj_bias", bias(block.mlp.c_proj.bias))
        print(f"  layer {i+1}/{n_layers}", flush=True)

    payload = blobs.getvalue()
    table = json.dumps(entries, separators=(",", ":")).encode()
    table_off = 256
    pad = (16 - ((table_off + len(table)) % 16)) % 16
    payload_off = table_off + len(table) + pad
    header = bytearray(256)
    header[0:4] = b"PGW1"
    header[4] = 4
    header[5] = ARCH_GPT2
    struct.pack_into("<8I", header, 8, vocab, n_layers, d, nh, nh, d_ff, n_ctx, GROUP)
    struct.pack_into("<III", header, 40, table_off, len(table), payload_off)
    struct.pack_into("<d", header, 56, 10000.0)
    struct.pack_into("<d", header, 64, 1.0)

    bin_path = out_dir / "model.q4.bin"
    with bin_path.open("wb") as f:
        f.write(header)
        f.write(table)
        f.write(b"\x00" * pad)
        f.write(payload)
    tok.save_pretrained(out_dir)
    for extra in list(out_dir.iterdir()):
        if extra.name not in ("model.q4.bin", "tokenizer.json", "card.json", "tokenizer_config.json"):
            if extra.suffix in (".txt", ".model") or extra.name in ("merges.txt", "vocab.json", "special_tokens_map.json"):
                continue
    card = {
        "hf": hf_id,
        "arch": "gpt2",
        "vocabSize": vocab,
        "nLayers": n_layers,
        "dModel": d,
        "nHeads": nh,
        "nKvHeads": nh,
        "dFf": d_ff,
        "maxSeqLen": n_ctx,
        "params": int(sum(p.numel() for p in model.parameters())),
        "q4Bytes": bin_path.stat().st_size,
        "bosId": int(cfg.bos_token_id),
        "eosId": int(cfg.eos_token_id),
        "template": "completion",
        "qGroup": GROUP,
    }
    (out_dir / "card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {bin_path} ({bin_path.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("hf_id", nargs="?", default="openai-community/gpt2")
    ap.add_argument("out_dir", nargs="?", default="models/gpt2")
    args = ap.parse_args()
    convert(args.hf_id, Path(args.out_dir))
