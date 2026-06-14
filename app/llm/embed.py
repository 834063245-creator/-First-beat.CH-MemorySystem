"""Embedding 统一入口 — Ollama GPU 推理（bge-m3）。

v2: 请求合并器 — 短时间内的多次 local_embed 调用自动合并为一次 batch HTTP，
    将 N 次 Ollama HTTP 往返压缩为 1 次，大幅降低排队延迟。
"""
import logging
import os
import threading
import time as _time
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── Ollama Embedding ──
_OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
_OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

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


# ═══════════════════════════════════════════════════════════════
# 请求合并器（Coalescer）
# ═══════════════════════════════════════════════════════════════
# 当 local_embed 缓存未命中时，不立即发单独的 HTTP 请求，
# 而是入队等待一个极短的合并窗口（15ms）。
# 同一窗口内的所有请求会被打包成一次 batch HTTP 发送到 Ollama。
#
# 效果：10 个并发的 local_embed → 1 次 Ollama /api/embed 调用
#       而不是 10 次独立的 /api/embeddings 调用

_COALESCE_WINDOW = 0.015        # 合并窗口 15ms
_COALESCE_TIMEOUT = 30.0         # 单个请求最大等待时间
_COALESCE_MAX_BATCH = 64         # 单批上限

_coalesce_lock = threading.Lock()
_coalesce_pending: dict[int, tuple[str, threading.Event, list]] = {}  # id → (text, event, [result])
_coalesce_counter: int = 0
_coalesce_timer: threading.Timer | None = None


def _drain_coalesce():
    """取出所有待处理请求，批量发送到 Ollama。

    直接调 _embed_via_ollama_batch 绕过 local_embed_batch，
    避免单条时 local_embed_batch → local_embed → coalescer 的死循环。
    调用方 local_embed 负责写缓存。
    """
    with _coalesce_lock:
        if not _coalesce_pending:
            return
        batch = list(_coalesce_pending.items())
        _coalesce_pending.clear()

    if not batch:
        return

    texts = [t for _, (t, _, _) in batch]

    # 直接走 Ollama batch API，不经过 local_embed_batch（避免循环调用）
    embeddings = _embed_via_ollama_batch(texts)

    for i, (rid, (_, event, holder)) in enumerate(batch):
        holder.append(embeddings[i] if i < len(embeddings) else None)
        event.set()


def _on_timer_drain():
    """定时器回调：窗口到期，执行排空。"""
    global _coalesce_timer
    _coalesce_timer = None
    _drain_coalesce()


def _schedule_drain():
    """确保排空定时器已启动（幂等）。"""
    global _coalesce_timer
    if _coalesce_timer is not None:
        return
    _coalesce_timer = threading.Timer(_COALESCE_WINDOW, _on_timer_drain)
    _coalesce_timer.daemon = True
    _coalesce_timer.start()


def _coalesced_embed(text: str) -> list[float] | None:
    """将单条 embed 请求入队，等待合并窗口后批量发送。"""
    global _coalesce_counter

    event = threading.Event()
    holder: list = []   # 用 list 承载返回值（闭包可变引用）

    with _coalesce_lock:
        _coalesce_counter += 1
        rid = _coalesce_counter
        _coalesce_pending[rid] = (text, event, holder)
        # 达到批量上限立即排空
        if len(_coalesce_pending) >= _COALESCE_MAX_BATCH:
            _drain_coalesce()
        else:
            _schedule_drain()

    # 等待结果
    if not event.wait(timeout=_COALESCE_TIMEOUT):
        # 超时兜底：直接调 Ollama
        logger.debug("embed coalescer 超时，回退到直接调用")
        return _embed_via_ollama(text)

    return holder[0] if holder else None


# ═══════════════════════════════════════════════════════════════
# n-gram 语义近似缓存
# ═══════════════════════════════════════════════════════════════

_ngram_cache: dict[str, dict[str, float]] = {}
_ngram_cache_lock = threading.Lock()
_NGRAM_CACHE_MAX = 2048


def _ngram_sig(text: str) -> dict[str, float]:
    """Compute character trigram signature for a text string."""
    if not text:
        return {}
    with _ngram_cache_lock:
        if text in _ngram_cache:
            return _ngram_cache[text]
    sig: dict[str, float] = {}
    t = text.lower()
    for i in range(len(t) - 1):
        bigram = t[i:i + 2]
        sig[bigram] = sig.get(bigram, 0) + 1
    for i in range(len(t) - 2):
        trigram = t[i:i + 3]
        sig[trigram] = sig.get(trigram, 0) + 1.5
    total = sum(sig.values()) or 1.0
    sig = {k: v / total for k, v in sig.items()}
    with _ngram_cache_lock:
        if len(_ngram_cache) >= _NGRAM_CACHE_MAX:
            _ngram_cache.pop(next(iter(_ngram_cache)))
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


# ── httpx 客户端单例 ──

_embed_client: httpx.Client | None = None
_embed_client_lock = threading.Lock()


def _get_embed_client() -> httpx.Client:
    """获取或创建模块级 httpx.Client 单例（连接池复用）。"""
    global _embed_client
    if _embed_client is None:
        with _embed_client_lock:
            if _embed_client is None:
                _embed_client = httpx.Client(
                    timeout=httpx.Timeout(30.0, connect=5.0),
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                )
    return _embed_client


def _embed_via_ollama(text: str) -> list[float] | None:
    """通过 Ollama API 嵌入（GPU 推理）— 单条路径，仅作兜底。"""
    text = text.strip()[:2000]
    if not text:
        return None
    try:
        client = _get_embed_client()
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


def _embed_via_ollama_batch(texts: list[str]) -> list[list[float] | None]:
    """批量嵌入 — 一次 HTTP 调用嵌入多条文本（coalescer 的核心路径）。

    调用 Ollama /api/embed，微批次 16 条，返回归一化向量。
    """
    results: list[list[float] | None] = [None] * len(texts)
    clean_texts: list[str] = []
    index_map: list[int] = []  # clean_texts pos → original pos
    for i, t in enumerate(texts):
        t = (t or "").strip()[:2000]
        if t:
            clean_texts.append(t)
            index_map.append(i)

    if not clean_texts:
        return results

    _MICRO_BATCH = 16
    for mb_start in range(0, len(clean_texts), _MICRO_BATCH):
        mb_end = min(mb_start + _MICRO_BATCH, len(clean_texts))
        mb_texts = clean_texts[mb_start:mb_end]
        try:
            client = _get_embed_client()
            resp = client.post(
                f"{_OLLAMA_URL}/api/embed",
                json={"model": _OLLAMA_EMBED_MODEL, "input": mb_texts, "keep_alive": "30m"},
            )
            resp.raise_for_status()
            data = resp.json()
            raw_embs = data.get("embeddings", [])
        except Exception as e:
            logger.warning("Ollama batch embed 失败 (%d 条): %s", len(mb_texts), e)
            # 回退逐条
            for pos in range(mb_start, mb_end):
                orig_idx = index_map[pos]
                results[orig_idx] = _embed_via_ollama(clean_texts[pos])
            continue

        for pos in range(mb_start, mb_end):
            rel = pos - mb_start
            orig_idx = index_map[pos]
            if rel < len(raw_embs) and raw_embs[rel] and isinstance(raw_embs[rel], list):
                arr = np.array(raw_embs[rel], dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                results[orig_idx] = arr.tolist()

    return results


# ═══════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════

def local_embed(text: str) -> list[float] | None:
    """单条文本嵌入，返回 1024 维归一化向量。失败返回 None。

    缓存策略（三级）：
      请求级缓存（无锁，0ms） → 全局 LRU 缓存 → n-gram 近似 → 合并器 → Ollama
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

    # 中速路径：语义近似命中（n-gram 复合相似度，微秒级）
    # v2: 长度过滤 + 早停优化，避免扫描全部 1024 条缓存
    with _embed_cache_lock:
        if _embed_cache:
            q_ng = _ngram_sig(text)
            best_key = None
            best_sim = -1.0
            scanned = 0
            _SCAN_LIMIT = 200       # 早停：扫描 200 条仍无可用匹配则放弃
            _STRONG_MATCH = 0.98    # 强匹配：立即停止
            for cached_text in _embed_cache:
                if abs(len(text) - len(cached_text)) / max(len(text), len(cached_text), 1) > 0.3:
                    continue
                sim = _ngram_sim(q_ng, _ngram_sig(cached_text))
                if sim > best_sim:
                    best_sim = sim
                    best_key = cached_text
                    if sim >= _STRONG_MATCH:
                        break
                scanned += 1
                # 早停：已扫描足够条数但未找到可用的近似匹配
                if scanned >= _SCAN_LIMIT and best_sim < 0.55:
                    break
            if best_sim >= 0.55 and best_key:
                val = _embed_cache.pop(best_key)
                _embed_cache[text] = val
                req_cache[text] = val  # 回填请求缓存
                return val

    # 第四级：通过合并器批量发 Ollama
    result = _coalesced_embed(text)

    if result:
        req_cache[text] = result  # 写请求缓存
        with _embed_cache_lock:
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                _embed_cache.pop(next(iter(_embed_cache)))
            _embed_cache[text] = result
    return result


def local_embed_batch(texts: list[str]) -> list[list[float] | None]:
    """批量嵌入 — 先查请求缓存 → 全局缓存，未命中的通过一次 Ollama HTTP 嵌入。

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

    raw_embs = _embed_via_ollama_batch(to_embed_texts)

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
    """异步版。"""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, local_embed, text)
