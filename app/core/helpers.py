"""共享辅助函数 — 从 backend/main.py 迁移至此。

包含：计时包装、trace/debug 构建、情绪反转加载、JSONL 缓存。
"""
import json
import logging
import os
import time

from app.config.settings import DATA_DIR
from app.core import bottleneck

logger = logging.getLogger(__name__)


# ── 计时包装 ──────────────────────────────────────────────────

async def timed(name: str, coro):
    """计时包装，自动记录耗时并上报 bottleneck。"""
    t0 = time.perf_counter()
    r = await coro
    _ms = (time.perf_counter() - t0) * 1000
    logger.info("[耗时] %s: %.0fms", name, _ms)
    bottleneck.record(name, _ms)
    return r


# ── 溯源 trace 构建 ──────────────────────────────────────────

def build_trace(memories: list) -> list[dict]:
    """从检索结果中提取 trace 数据，响应式传递给前端。"""
    trace = []
    for m in memories:
        meta = m.get("metadata", {})
        raw_tags = meta.get("tags", "")
        if isinstance(raw_tags, str):
            tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
        else:
            tags_list = list(raw_tags) if raw_tags else []
        trace.append({
            "id": m["id"],
            "summary": meta.get("summary", ""),
            "timestamp": meta.get("timestamp", 0),
            "source": m.get("source", ""),
            "display_source": m.get("display_source", ""),
            "hit_count": meta.get("hit_count", 0),
            "tags": tags_list,
        })
    return trace


# ── Debug info 构建 ───────────────────────────────────────────

def build_debug_info(memories: list, personalities: list, timeline_recent: list,
                     prompt: str | None = None) -> dict:
    """构建调试信息，debug=True 时附加到响应中。"""
    debug_memories = []
    for m in memories:
        meta = m.get("metadata", {})
        debug_memories.append({
            "id": m["id"],
            "summary": meta.get("summary", ""),
            "semantic_score": m.get("semantic_score"),
            "hit_count": meta.get("hit_count", 0),
            "reason": m.get("reason", "unknown"),
            "timestamp": meta.get("timestamp", 0),
        })

    debug_personalities = []
    for p in personalities:
        if isinstance(p, str):
            debug_personalities.append({"content": p, "hit_count": 0})
        elif isinstance(p, dict):
            debug_personalities.append({
                "content": p.get("content", ""),
                "hit_count": p.get("hit_count", 0),
            })

    result = {
        "retrieved_memories": debug_memories,
        "personalities": debug_personalities,
        "timeline_recent": timeline_recent,
    }
    if prompt is not None:
        result["prompt"] = prompt
    return result


# ── JSONL 文件缓存 ────────────────────────────────────────────

_jsonl_cache: dict[str, tuple[float, object]] = {}
_JSONL_CACHE_TTL = 30


def _load_jsonl_cached(path: str, parser: callable) -> object:
    """带缓存的 JSONL 读取，30 秒 TTL + 文件 mtime 变化时自动刷新。"""
    key = path
    now = time.time()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    cached = _jsonl_cache.get(key)
    if cached is not None:
        cache_time, cache_mtime, cache_value = cached
        if now - cache_time < _JSONL_CACHE_TTL and cache_mtime == mtime:
            return cache_value
    # 缓存未命中或过期，重新读取
    value = parser()
    _jsonl_cache[key] = (time.time(), mtime, value)
    return value


# ── 情绪反转加载 ──────────────────────────────────────────────

def load_recent_reversals(data_dir: str = DATA_DIR) -> list[dict]:
    """加载最近的情绪反转事件，供 prompt 注入。"""
    path = os.path.join(data_dir, "emotional_reversals.jsonl")
    def _parse():
        results = []
        try:
            if not os.path.exists(path):
                return results
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        results.append(rec)
                    except json.JSONDecodeError:
                        continue
            return results[-5:]
        except Exception as exc:
            logger.debug("加载情绪反转日志失败: %s", exc)
            return results
    return _load_jsonl_cached(path, _parse)
