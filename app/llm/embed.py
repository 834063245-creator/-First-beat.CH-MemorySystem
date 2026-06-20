# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c3d4e5f6

"""Embedding 统一入口 — qwen2.5 独立嵌入模型（纯 Python+numpy）。

v3: 从 bge-m3 (Ollama HTTP) 切换到 qwen_embed — 查表 351x 加速，零 HTTP。
    去掉 Ollama HTTP 客户端、请求合并器、n-gram 近似缓存。
    保留请求级缓存 + LRU 全局缓存。

向量维度: 3584（qwen2.5 embedding 层）。
"""
import logging
import threading
from typing import List, Optional

import numpy as np

from app.llm.qwen_embed import get_qwen_embedder

logger = logging.getLogger(__name__)

# qwen2.5 embedding 维度（与 settings.EMBED_MODELS["qwen_embed"]["dimension"] 保持一致）
QWEN_EMBED_DIM = 3584

# 全局缓存（跨请求，LRU）
_embed_cache = {}
_embed_cache_lock = threading.Lock()
_EMBED_CACHE_MAX = 1024

# 请求级缓存（同请求内绝不重复 embed，线程本地，无锁）
_request_cache: threading.local = threading.local()


def clear_request_cache():
    """清空当前线程的请求级 embedding 缓存。

    每个请求开始时调用一次，防止线程池线程复用导致缓存膨胀。
    """
    _request_cache.__dict__.clear()


def _get_request_cache() -> dict:
    """获取当前线程的请求级缓存（无锁）。"""
    d = getattr(_request_cache, 'data', None)
    if d is None:
        d = {}
        _request_cache.data = d
    return d


# ── qwen_embed 后端（纯 Python + numpy，无 HTTP）─────────────────

def _embed_via_qwen(text: str) -> list[float] | None:
    """qwen_embed 单条嵌入，返回归一化向量。"""
    text = text.strip()[:2000]
    if not text:
        return None
    try:
        embedder = get_qwen_embedder()
        vec = embedder.embed(text)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
    except Exception as e:
        logger.warning("qwen_embed 失败: %s", e)
        return None


def _embed_via_qwen_batch(texts: list[str]) -> list[list[float] | None]:
    """qwen_embed 批量嵌入。"""
    results: list[list[float] | None] = [None] * len(texts)
    clean_texts: list[str] = []
    index_map: list[int] = []
    for i, t in enumerate(texts):
        t = (t or "").strip()[:2000]
        if t:
            clean_texts.append(t)
            index_map.append(i)

    if not clean_texts:
        return results

    try:
        embedder = get_qwen_embedder()
        emb_matrix = embedder.embed_batch(clean_texts)  # [batch, dim]
        for pos, orig_idx in enumerate(index_map):
            vec = emb_matrix[pos]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results[orig_idx] = vec.tolist()
    except Exception as e:
        logger.warning("qwen_embed batch 失败 (%d 条): %s", len(clean_texts), e)
        # 回退逐条
        for pos, orig_idx in enumerate(index_map):
            results[orig_idx] = _embed_via_qwen(clean_texts[pos])

    return results


# ═══════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════

def local_embed(text: str) -> list[float] | None:
    """单条文本嵌入，返回 3584 维归一化向量。失败返回 None。

    缓存策略（两级）：
      请求级缓存（无锁，0ms） → 全局 LRU 缓存 → qwen_embed 直接计算
    """
    # 第一级：请求级缓存（线程本地，无锁，最快）
    req_cache = _get_request_cache()
    if text in req_cache:
        return req_cache[text]

    # 第二级：全局精确缓存（LRU）
    with _embed_cache_lock:
        if text in _embed_cache:
            val = _embed_cache.pop(text)
            _embed_cache[text] = val
            req_cache[text] = val  # 回填请求缓存
            return val

    # 第三级：qwen_embed 直接计算（纯 numpy，微秒级）
    result = _embed_via_qwen(text)

    if result:
        req_cache[text] = result  # 写请求缓存
        with _embed_cache_lock:
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                _embed_cache.pop(next(iter(_embed_cache)))
            _embed_cache[text] = result
    return result


def local_embed_batch(texts: list[str]) -> list[list[float] | None]:
    """批量嵌入 — 先查缓存，未命中的一次 qwen_embed batch。

    返回的向量做归一化 + 写两级缓存。
    """
    if not texts:
        return []
    if len(texts) == 1:
        return [local_embed(texts[0])]

    req_cache = _get_request_cache()
    results: list[list[float] | None] = [None] * len(texts)
    to_embed: list[int] = []
    to_embed_texts: list[str] = []

    for i, t in enumerate(texts):
        t = (t or "").strip()[:2000]
        if not t:
            results[i] = None
            continue
        # 先查请求缓存
        if t in req_cache:
            results[i] = req_cache[t]
            continue
        # 再查全局缓存
        with _embed_cache_lock:
            if t in _embed_cache:
                results[i] = _embed_cache[t]
                req_cache[t] = _embed_cache[t]  # 回填请求缓存
                continue
        to_embed.append(i)
        to_embed_texts.append(t)

    if not to_embed_texts:
        return results

    raw_embs = _embed_via_qwen_batch(to_embed_texts)

    for pos, idx in enumerate(to_embed):
        emb = raw_embs[pos] if pos < len(raw_embs) else None
        if emb:
            results[idx] = emb
            req_cache[to_embed_texts[pos]] = emb  # 写请求缓存
            with _embed_cache_lock:
                if len(_embed_cache) >= _EMBED_CACHE_MAX:
                    _embed_cache.pop(next(iter(_embed_cache)))
                _embed_cache[to_embed_texts[pos]] = emb
        else:
            results[idx] = None

    return results


async def local_embed_async(text: str) -> list[float] | None:
    """异步版 — qwen_embed 是纯 CPU numpy 运算，直接同步调用即可。"""
    return local_embed(text)
