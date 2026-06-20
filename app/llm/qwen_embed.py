# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: e75fec52

"""
qwen2.5 自制嵌入模型 — 从 GGUF 提取的 token embedding 表独立运行。

纯 Python + numpy，不依赖 Ollama。
词表 152K × 3584 维，BPE tokenizer + mean pooling。
比 Ollama /api/embeddings 快 40x+（查表 vs 加载完整模型）。

用法:
    from app.llm.qwen_embed import QwenEmbedder
    embedder = QwenEmbedder()
    vec = embedder.embed("Rust borrow checker怎么理解")  # np.ndarray [3584]

与 bge-m3 接口兼容，可替换 local_embed() 中的 Ollama 调用。
"""

import json
import os
from pathlib import Path

import numpy as np

# ── 路径 ────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_EMBED_PATH = _DATA_DIR / "qwen_embed_f32.npy"
_TOKENIZER_PATH = _DATA_DIR / "qwen_tokenizer.json"


# ── BPE Tokenizer ───────────────────────────────────────

class QwenBPETokenizer:
    """qwen2.5 BPE tokenizer 的纯 Python 实现。

    基于 GGUF 中提取的 vocab tokens + merges。
    使用 GPT-2 标准的 byte-level BPE + bytes_to_unicode 编码。
    """

    def __init__(self, tokens: list[str], merges: list[str]):
        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.token_to_id = {t: i for i, t in enumerate(tokens)}

        # 解析 merges (格式: "token1 token2")
        self.merges = {}
        self._merge_ranks = {}
        for rank, merge_str in enumerate(merges):
            parts = merge_str.split(" ")
            if len(parts) == 2:
                merged = merge_str.replace(" ", "")
                pair = (parts[0], parts[1])
                self.merges[pair] = merged
                self._merge_ranks[pair] = rank

        # GPT-2 bytes_to_unicode: byte → printable Unicode char
        self._byte_encoder = self._build_bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

    @staticmethod
    def _build_bytes_to_unicode() -> dict[int, str]:
        """GPT-2 标准 bytes_to_unicode 映射。

        将 256 个字节映射到 Unicode 字符，避免空白/控制字符干扰 BPE。
        Space (0x20) → chr(288) = 'Ġ'。
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def encode(self, text: str) -> list[int]:
        """BPE encode text → token IDs.

        GPT-2 byte-level BPE:
        1. UTF-8 bytes → bytes_to_unicode() → Unicode string
        2. BPE 合并
        3. 每个 token → ID
        """
        if not text:
            return []

        # Step 1: UTF-8 bytes → unicode chars via GPT-2 encoding
        utf8_bytes = text.encode("utf-8")
        unicode_chars = [self._byte_encoder[b] for b in utf8_bytes]

        # Step 2: BPE 合并
        tokens_list = list(unicode_chars)

        changed = True
        while changed and len(tokens_list) > 1:
            changed = False
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens_list) - 1):
                pair = (tokens_list[i], tokens_list[i + 1])
                if pair in self._merge_ranks:
                    rank = self._merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i

            if best_idx >= 0:
                pair = (tokens_list[best_idx], tokens_list[best_idx + 1])
                merged = self.merges[pair]
                tokens_list = (
                    tokens_list[:best_idx]
                    + [merged]
                    + tokens_list[best_idx + 2:]
                )
                changed = True

        # Step 3: tokens → IDs
        ids = []
        for token in tokens_list:
            tid = self.token_to_id.get(token)
            if tid is not None:
                ids.append(tid)

        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]


# ── Embedder ────────────────────────────────────────────

class QwenEmbedder:
    """qwen2.5 轻量嵌入模型。

    加载预提取的 float32 embedding 表 + tokenizer。
    embed() 返回 3584 维 mean-pooled 向量。
    """

    def __init__(self):
        self._load()

    def _load(self):
        if not _EMBED_PATH.exists():
            raise FileNotFoundError(
                f"Embedding table not found: {_EMBED_PATH}\n"
                f"Run: python scripts/extract_qwen_embed.py"
            )
        if not _TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {_TOKENIZER_PATH}\n"
                f"Run: python scripts/extract_qwen_embed.py"
            )

        self.embed_table = np.load(_EMBED_PATH)  # [vocab, 3584]

        with open(_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        self.tokenizer = QwenBPETokenizer(
            tokens=tokenizer_data["tokens"],
            merges=tokenizer_data.get("merges", []),
        )
        self.dim = self.embed_table.shape[1]

    def embed(self, text: str) -> np.ndarray:
        """单条文本 → embedding 向量 [dim]."""
        ids = self.tokenizer.encode(text)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = self.embed_table[ids]  # [n_tokens, dim]
        return vecs.mean(axis=0).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量文本 → embedding 矩阵 [batch, dim]."""
        all_ids = self.tokenizer.encode_batch(texts)
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, ids in enumerate(all_ids):
            if ids:
                result[i] = self.embed_table[ids].mean(axis=0)
        return result


# ── 全局单例 ────────────────────────────────────────────

_embedder: QwenEmbedder | None = None


def get_qwen_embedder() -> QwenEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = QwenEmbedder()
    return _embedder


def qwen_embed(text: str) -> np.ndarray:
    return get_qwen_embedder().embed(text)


def qwen_embed_batch(texts: list[str]) -> np.ndarray:
    return get_qwen_embedder().embed_batch(texts)


# ── 与现有 embed.py 接口兼容 ────────────────────────────

async def qwen_embed_async(texts: list[str]) -> list[list[float]]:
    """异步接口，与 local_embed() 签名兼容。

    返回: list[list[float]]，每个内层 list 长度为 dim
    """
    emb = get_qwen_embedder()
    result = emb.embed_batch(texts)
    return result.tolist()
