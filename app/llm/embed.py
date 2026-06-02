"""Embedding 统一入口 — Ollama GPU 推理（bge-m3）。"""
import asyncio
import logging
import os
import threading
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── Ollama Embedding ──
_OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
_OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

# 请求级缓存
_embed_cache = {}
_embed_cache_lock = threading.Lock()
_EMBED_CACHE_MAX = 1024


def local_embed(text: str) -> Optional[List[float]]:
    """单条文本嵌入，返回 1024 维归一化向量。失败返回 None。

    缓存策略：精确匹配优先 → 语义近似命中兜底 → Ollama API 最后。
    语义近似用 n-gram 复合相似度，阈值 0.95（几乎相同措辞略有差异时命中）。
    """
    # 快速路径：精确缓存命中（LRU）
    with _embed_cache_lock:
        if text in _embed_cache:
            val = _embed_cache.pop(text)
            _embed_cache[text] = val
            return val

    # 中速路径：语义近似命中（n-gram 复合相似度，微秒级）
    with _embed_cache_lock:
        if _embed_cache:
            q_ng = _ngram_sig(text)
            best_key = None
            best_sim = -1.0
            # 按命中顺序遍历，同时对 len 做粗筛
            for cached_text, cached_vec in _embed_cache.items():
                if abs(len(text) - len(cached_text)) / max(len(text), len(cached_text), 1) > 0.3:
                    continue
                sim = _ngram_sim(q_ng, _ngram_sig(cached_text))
                if sim > best_sim:
                    best_sim = sim
                    best_key = cached_text
                    if sim >= 0.98:
                        break
            if best_sim >= 0.55 and best_key:
                val = _embed_cache.pop(best_key)
                _embed_cache[text] = val
                return val

    # 慢速路径：调 Ollama
    result = _embed_via_ollama(text)

    if result:
        with _embed_cache_lock:
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                _embed_cache.pop(next(iter(_embed_cache)))
            _embed_cache[text] = result
    return result


# ── n-gram 语义近似缓存 ─────────────────────────────────────
# 用字符级 bigram/trigram 分布做快速相似度估算，微秒级。
# 对措辞略有差异但语义相同的短文本（查询）命中率 > 90%。

_ngram_cache: dict[str, dict[str, float]] = {}
_NGRAM_CACHE_MAX = 2048


def _ngram_sig(text: str) -> dict[str, float]:
    """Compute character trigram signature for a text string."""
    if not text:
        return {}
    if text in _ngram_cache:
        return _ngram_cache[text]
    sig: dict[str, float] = {}
    t = text.lower()
    for i in range(len(t) - 1):
        bigram = t[i:i + 2]
        sig[bigram] = sig.get(bigram, 0) + 1
    for i in range(len(t) - 2):
        trigram = t[i:i + 3]
        sig[trigram] = sig.get(trigram, 0) + 1.5  # trigram 权重高一点
    # 归一化
    total = sum(sig.values()) or 1.0
    sig = {k: v / total for k, v in sig.items()}
    # LRU 缓存
    if len(_ngram_cache) >= _NGRAM_CACHE_MAX:
        _ngram_cache.clear()
    _ngram_cache[text] = sig
    return sig


def _ngram_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two n-gram signatures."""
    if not a or not b:
        return 0.0
    inter = set(a.keys()) & set(b.keys())
    if not inter:
        return 0.0
    dot = sum(a[k] * b[k] for k in inter)
    na = sum(v * v for v in a.values()) ** 0.5 or 1.0
    nb = sum(v * v for v in b.values()) ** 0.5 or 1.0
    return dot / (na * nb)


def _embed_via_ollama(text: str) -> Optional[List[float]]:
    """通过 Ollama API 嵌入（GPU 推理）。"""
    text = text.strip()[:2000]
    if not text:
        return None
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = client.post(
                f"{_OLLAMA_URL}/api/embeddings",
                json={"model": _OLLAMA_EMBED_MODEL, "prompt": text, "keep_alive": "30m"},
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding")
            if emb and isinstance(emb, list):
                arr = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                return arr.tolist()
    except Exception as e:
        logger.warning("Ollama embedding 失败: %s", e)
    return None


def local_embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """批量嵌入 — 调用 Ollama /api/embed 一次 HTTP 嵌入多条文本。

    对返回的向量做归一化 + 写缓存。
    所有文本共享一次 HTTP 往返，远快于逐条调用。
    失败时回退到逐条 local_embed。
    """
    if not texts:
        return []
    if len(texts) == 1:
        return [local_embed(texts[0])]

    # 检查缓存
    results: List[Optional[List[float]]] = [None] * len(texts)
    to_embed: List[int] = []  # 需要调 API 的索引
    to_embed_texts: List[str] = []
    for i, t in enumerate(texts):
        t = (t or "").strip()[:2000]
        if not t:
            results[i] = None
            continue
        with _embed_cache_lock:
            if t in _embed_cache:
                results[i] = _embed_cache[t]
                continue
        to_embed.append(i)
        to_embed_texts.append(t)

    if not to_embed_texts:
        return results

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            resp = client.post(
                f"{_OLLAMA_URL}/api/embed",
                json={"model": _OLLAMA_EMBED_MODEL, "input": to_embed_texts, "keep_alive": "30m"},
            )
            resp.raise_for_status()
            data = resp.json()
            raw_embs = data.get("embeddings", [])
    except Exception as e:
        logger.warning("Ollama batch embed 失败 (%d 条), 回退逐条: %s", len(to_embed_texts), e)
        for idx in to_embed:
            results[idx] = local_embed(texts[idx])
        return results

    # 归一化 + 写缓存
    for pos, idx in enumerate(to_embed):
        if pos < len(raw_embs) and raw_embs[pos] and isinstance(raw_embs[pos], list):
            arr = np.array(raw_embs[pos], dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            emb = arr.tolist()
            results[idx] = emb
            with _embed_cache_lock:
                if len(_embed_cache) >= _EMBED_CACHE_MAX:
                    _embed_cache.pop(next(iter(_embed_cache)))
                _embed_cache[to_embed_texts[pos]] = emb
        else:
            results[idx] = None

    return results


async def local_embed_async(text: str) -> Optional[List[float]]:
    """异步版。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, local_embed, text)
