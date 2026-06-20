# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 26934ea3

"""
================================================================
从 qwen2.5 GGUF 提取 tokenizer + 解量化 embedding 权重。
纯 Python + numpy，一次转换缓存到磁盘。

Q4_K 解量化参考 llama.cpp ggml-quants.c:
  超块: 256 元素, 144 字节
  value = d * (q4 * scale - dmin)

输出:
  data/qwen_embed_f32.npy    float32 [vocab, 3584] (~2.1 GB)
  data/qwen_tokenizer.json   {tokens, merges, bos_id, eos_id}

用法: python scripts/extract_qwen_embed.py
================================================================
"""
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ── GGUF 解析 ────────────────────────────────────────────

GGUF_MAGIC = b'GGUF'
GGUF_VALUE_TYPES = {
    0: 'uint8', 1: 'int8', 2: 'uint16', 3: 'int16',
    4: 'uint32', 5: 'int32', 6: 'float32', 7: 'bool',
    8: 'string', 9: 'array', 10: 'uint64', 11: 'int64', 12: 'float64',
}

GGUF_PATH = "D:/ollama_models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EMBED_PATH = os.path.join(OUT_DIR, "qwen_embed_f32.npy")
TOKENIZER_PATH = os.path.join(OUT_DIR, "qwen_tokenizer.json")


def parse_gguf(path):
    """解析 GGUF 文件，返回 (metadata dict, tensor_info list, tensor_data_offset)。"""
    f = open(path, 'rb')

    magic = f.read(4)
    assert magic == GGUF_MAGIC, f"Bad magic: {magic}"
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} metadata entries")

    metadata = {}
    for _ in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
        value_type = struct.unpack('<I', f.read(4))[0]

        if value_type == 4:  # uint32
            metadata[key] = struct.unpack('<I', f.read(4))[0]
        elif value_type == 5:  # int32
            metadata[key] = struct.unpack('<i', f.read(4))[0]
        elif value_type == 6:  # float32
            metadata[key] = struct.unpack('<f', f.read(4))[0]
        elif value_type == 7:  # bool
            metadata[key] = f.read(1)[0] != 0
        elif value_type == 8:  # string
            slen = struct.unpack('<Q', f.read(8))[0]
            metadata[key] = f.read(slen).decode('utf-8', errors='replace')
        elif value_type == 10:  # uint64
            metadata[key] = struct.unpack('<Q', f.read(8))[0]
        elif value_type == 12:  # float64
            metadata[key] = struct.unpack('<d', f.read(8))[0]
        elif value_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]

            if arr_type == 8:  # string array — 仅对 tokenizer tokens 读取
                if key == 'tokenizer.ggml.tokens':
                    print(f"  Reading {arr_len:,} tokens...")
                    tokens = []
                    for i in range(arr_len):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        tokens.append(f.read(slen).decode('utf-8', errors='replace'))
                        if i % 50000 == 0 and i > 0:
                            print(f"    {i:,}/{arr_len:,}")
                    metadata[key] = tokens
                    print(f"    Done: {len(tokens):,} tokens")
                elif key == 'tokenizer.ggml.token_type':
                    types = []
                    for i in range(arr_len):
                        types.append(struct.unpack('<I', f.read(4))[0])
                    metadata[key] = types
                elif key == 'tokenizer.ggml.merges':
                    merges = []
                    for i in range(arr_len):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        merges.append(f.read(slen).decode('utf-8', errors='replace'))
                    metadata[key] = merges
                else:
                    # skip unknown string arrays
                    for i in range(arr_len):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.seek(slen, 1)
            elif arr_type in (4, 5):  # int32/uint32 array
                elem_size = 4
                f.seek(arr_len * elem_size, 1)
            else:
                # skip unknown array type
                elem_size = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(arr_type, 1)
                f.seek(arr_len * elem_size, 1)
        else:
            # unknown type, skip roughly
            pass

    # 读取 tensor info
    tensors = []
    for _ in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8', errors='replace')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack('<Q', f.read(8))[0])
        tensor_type = struct.unpack('<I', f.read(4))[0]
        tensor_offset = struct.unpack('<Q', f.read(8))[0]
        tensors.append({
            'name': name, 'dims': dims, 'type': tensor_type,
            'offset': tensor_offset,
        })

        if name == 'token_embd.weight':
            print(f"  Found: {name} shape={dims} offset={tensor_offset}")

    tensor_data_offset = f.tell()
    f.close()

    return metadata, tensors, tensor_data_offset


# ── Q4_K 解量化 ─────────────────────────────────────────

# Q4_K 超块: 256 元素, 144 字节
# bytes 0-1: d (fp16), bytes 2-3: dmin (fp16)
# bytes 4-15: 12 字节 → 16 × 6-bit scales
# bytes 16-143: 128 字节 → 256 × 4-bit values
# value[i] = d * (scale[i//16] * q4_value[i] - dmin)

Q4K_BLOCK_SIZE = 256
Q4K_BLOCK_BYTES = 144

# 预计算 fp16 lookup table
FP16_TABLE = {}
for h in range(65536):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        val = mant / 1024.0
    elif exp == 31:
        val = float('nan') if mant else float('inf')
    else:
        val = (1 + mant / 1024.0)
    val *= 2.0 ** (exp - 15) if exp != 0 else 2.0 ** -14
    FP16_TABLE[h] = -val if sign else val


def dequant_q4k_row(raw_row: np.ndarray, vocab_size: int, out_row: np.ndarray):
    """解量化一行 embedding (vocab_size 元素)。

    raw_row: uint8 array of shape [(vocab_size // 256) * 144]
    out_row: float32 array of shape [vocab_size]
    """
    n_blocks = vocab_size // 256

    # 提取所有 block 的 bytes
    # d: 每个 block 的前 2 字节 (fp16)
    raw_d = raw_row[0::144].astype(np.uint16) | (raw_row[1::144].astype(np.uint16) << 8)  # [n_blocks]
    raw_dmin = raw_row[2::144].astype(np.uint16) | (raw_row[3::144].astype(np.uint16) << 8)  # [n_blocks]

    # 用预计算表转 float32
    d_vals = np.array([FP16_TABLE[int(x)] for x in raw_d], dtype=np.float32)
    dmin_vals = np.array([FP16_TABLE[int(x)] for x in raw_dmin], dtype=np.float32)

    # scales: 字节 4-15, 12 字节 → 16 × 6-bit
    # scale[i] = 6 bits at position i*6 in the 12-byte window
    scales_raw = raw_row.reshape(n_blocks, 144)[:, 4:16]  # [n_blocks, 12]

    # 提取 16 个 6-bit scales
    all_scales = np.zeros((n_blocks, 16), dtype=np.float32)
    for sub in range(16):
        byte_offset = (sub * 6) // 8
        bit_offset = (sub * 6) % 8
        # 6 bits spanning potentially 2 bytes
        if bit_offset <= 2:
            all_scales[:, sub] = (scales_raw[:, byte_offset] >> bit_offset) & 0x3F
        else:
            low_part = scales_raw[:, byte_offset] >> bit_offset
            high_part = (scales_raw[:, byte_offset + 1] & ((1 << (bit_offset - 2)) - 1)) << (8 - bit_offset)
            all_scales[:, sub] = (low_part | high_part) & 0x3F

    # qs: 128 字节 → 256 × 4-bit
    qs = raw_row.reshape(n_blocks, 144)[:, 16:144]  # [n_blocks, 128]

    # 解量化所有 block 并行
    for sub in range(16):
        sub_scale = dmin_vals + d_vals * all_scales[:, sub]  # [n_blocks]
        base = sub * 16  # 16 elements per sub-block (256/16=16)

        for j in range(8):  # 16 个 4-bit 值占 8 字节
            b = qs[:, base // 2 + j]  # [n_blocks], 2x4-bit packed
            q4_low = (b & 0x0F).astype(np.float32)
            q4_high = (b >> 4).astype(np.float32)

            col_low = sub * 16 + j * 2
            col_high = sub * 16 + j * 2 + 1

            # 广播: sub_scale shape [n_blocks], q4 shape [n_blocks]
            out_row[col_low::256] = sub_scale * q4_low - dmin_vals * d_vals
            out_row[col_high::256] = sub_scale * q4_high - dmin_vals * d_vals


# ── 主流程 ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Extract qwen2.5 GGUF: tokenizer + embedding")
    print("=" * 60)

    # Parse
    print("\n[1/3] Parsing GGUF...")
    t0 = time.time()
    metadata, tensors, data_start = parse_gguf(GGUF_PATH)
    print(f"  Parsed in {time.time()-t0:.1f}s")

    # Tokenizer
    print("\n[2/3] Extracting tokenizer...")
    tokens = metadata.get('tokenizer.ggml.tokens', [])
    merges = metadata.get('tokenizer.ggml.merges', [])
    bos_id = metadata.get('tokenizer.ggml.bos_token_id')
    eos_id = metadata.get('tokenizer.ggml.eos_token_id')

    tokenizer_data = {
        'tokens': tokens,
        'merges': merges,
        'vocab_size': len(tokens),
        'bos_token_id': bos_id,
        'eos_token_id': eos_id,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)
    print(f"  {len(tokens)} tokens, {len(merges)} merges")
    print(f"  Saved to {TOKENIZER_PATH}")

    # Embedding
    print("\n[3/3] Dequantizing embedding weights...")
    emb_tensor = None
    for t in tensors:
        if t['name'] == 'token_embd.weight':
            emb_tensor = t
            break

    if emb_tensor is None:
        print("ERROR: token_embd.weight not found!")
        return

    hidden_dim, vocab_size = emb_tensor['dims']
    emb_offset = emb_tensor['offset'] + data_start  # 需要加上 tensor data 起始位置

    print(f"  Hidden dim: {hidden_dim}, Vocab size: {vocab_size}")
    print(f"  Reading raw data from offset {emb_offset}...")

    # mmap 读取 embedding 区域
    raw_embed = np.memmap(
        GGUF_PATH, dtype=np.uint8, mode='r',
        offset=emb_offset,
        shape=(hidden_dim * (vocab_size // 256) * Q4K_BLOCK_BYTES,)
    )

    blocks_per_row = vocab_size // 256
    row_bytes = blocks_per_row * Q4K_BLOCK_BYTES

    # 输出: [vocab, dim]（按列排列，方便 pool 时索引）
    output = np.zeros((vocab_size, hidden_dim), dtype=np.float32)

    print(f"  Dequantizing {hidden_dim} rows...")
    t0 = time.time()

    for row in range(hidden_dim):
        start = row * row_bytes
        end = start + row_bytes
        raw_row = raw_embed[start:end]  # shape [row_bytes]

        dequant_q4k_row(raw_row, vocab_size, output[:, row])

        if row % 500 == 0 or row == hidden_dim - 1:
            elapsed = time.time() - t0
            pct = (row + 1) / hidden_dim * 100
            eta = elapsed / (row + 1) * (hidden_dim - row - 1) if row > 0 else 0
            print(f"  {row+1}/{hidden_dim} rows ({pct:.0f}%) | {elapsed:.0f}s ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # 保存
    print(f"\n  Saving to {EMBED_PATH}...")
    np.save(EMBED_PATH, output)
    size_gb = os.path.getsize(EMBED_PATH) / 1024 / 1024 / 1024
    print(f"  Saved: {size_gb:.2f} GB")

    print("\nDone!")


if __name__ == "__main__":
    main()
