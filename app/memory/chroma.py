"""记忆层 — ChromaDB 存储/检索 + 上下文包裹管理."""
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import chromadb

from app.config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBED_MODELS,
    DEFAULT_EMBED_MODEL,
    EMBED_BACKFILL_MARKER,
)

logger = logging.getLogger(__name__)


# ===================================================================
# ChromaDB — 向量存储 & 检索
# ===================================================================

def _build_items(result: dict) -> list[dict]:
    """ChromaDB get() 原始结果 → 统一 item 格式。"""
    docs = result.get("documents") or [None] * len(result["ids"])
    items = []
    for i, mid in enumerate(result["ids"]):
        meta = result["metadatas"][i] if result.get("metadatas") else {}
        items.append({
            "id": mid,
            "document": docs[i] if i < len(docs) and docs[i] else None,
            "metadata": dict(meta) if meta else {},
        })
    return items


class ChromaService:
    """ChromaDB 记忆存储与检索（读写分离，避免并发冲突）。"""

    # 情绪淡化参数（替代原 REM 命名）
    DESENSITIZATION_CHECK_INTERVAL = 50   # 每50次 increment_hit_count 触发一次检查
    EMOTION_DECAY_DAYS = 3               # 3天未提及则淡化
    EMOTION_DECREMENT = 1                # 每次减1

    def __init__(self, persist_dir: str | None = None, *,
                 collection_name: str | None = None):
        chroma_path = persist_dir or CHROMA_PERSIST_DIR
        # ChromaDB PersistentClient 内部有连接池，线程安全，无需读写分离
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._lock = threading.Lock()
        # 情绪淡化计数器
        self._desensitization_counter = 0
        # Embedding 缓存（attention 位移因子用）
        self._emb_cache: dict[str, list] = {}
        self._emb_cache_lock = threading.Lock()
        # list_all 全局缓存（后台线程共享，减少 SQLite 全量读取）
        self._list_all_cache: list[dict] | None = None
        self._list_all_cache_time: float = 0.0
        self._list_all_cache_lock = threading.Lock()
        self._list_all_cache_ttl: float = 300  # 默认 5 分钟
        # stats() 运行计数器，避免全量 metadata 遍历
        self._total_hits: int = 0
        self._earliest_ts: float | None = None
        self._latest_ts: float | None = None

        coll_name = collection_name or EMBED_MODELS[DEFAULT_EMBED_MODEL]["collection"]
        self._collection = self._client.get_or_create_collection(
            name=coll_name, embedding_function=None,
        )

    # ------------------------------------------------------------------
    # 记忆写入
    # ------------------------------------------------------------------

    def add_memory(
        self,
        user_message: str,
        ai_message: str,
        summary: str,
        tags: list[str],
        embedding: list[float],
        *,
        model_id: str = DEFAULT_EMBED_MODEL,
        entities: list[dict] | None = None,
        date_tag: str | None = None,
        time_features: dict | None = None,
        source: str = "user",
    ) -> str:
        """写入一轮对话到 ChromaDB。

        entities：命名实体列表。
        date_tag：日期标签"YYYY-MM-DD"。
        time_features：预计算时间特征字典（year, month, day, week, day_of_week, quarter, season, year_month）。
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        document = f"用户：{user_message}\nAI：{ai_message}"

        # 初始热度：情绪强或已有高情感 → hot，否则 warm
        _initial_heat = "hot" if (
            (time_features and time_features.get("emotional_intensity", 0) or 0) >= 2
        ) else "warm"
        meta = {
            "user_message": user_message,
            "ai_message": ai_message,
            "summary": summary,
            "tags": ",".join(tags),
            "timestamp": timestamp,
            "hit_count": 0,
            "heat": _initial_heat,
            "embed_model": DEFAULT_EMBED_MODEL,
            "stale": False,
            "archived": False,
            "superseded_by": "",
            "storage_complete": False,
            "source": source,
        }
        if entities:
            meta["entities"] = json.dumps(entities, ensure_ascii=False)
        if date_tag:
            meta["date_tag"] = date_tag
        if time_features:
            meta.update(time_features)

        self._collection.add(
            ids=[memory_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[meta],
        )
        logger.info("记忆写入完成 id=%s source=%s summary=%s", memory_id[:8], source, summary[:60])
        # 更新统计计数器
        if self._earliest_ts is None or timestamp < self._earliest_ts:
            self._earliest_ts = timestamp
        if self._latest_ts is None or timestamp > self._latest_ts:
            self._latest_ts = timestamp
        self._invalidate_list_all_cache()
        return memory_id

    def mark_storage_complete(self, memory_id: str):
        """标记记忆入库完成。"""
        with self._lock:
            self._collection.update(
                ids=[memory_id],
                metadatas=[{"storage_complete": True}],
            )

    def update_memory(
        self,
        memory_id: str,
        summary: str,
        tags: list[str],
        embedding: list[float],
    ):
        """纠正记忆：只更新 metadata（summary/tags）和 embedding，不修改 documents。

        用于引用溯源面板的「纠正」操作。
        """
        meta = {
            "summary": summary,
            "tags": ",".join(tags),
        }
        self._collection.update(
            ids=[memory_id],
            metadatas=[meta],
            embeddings=[embedding],
        )
        logger.info("记忆纠正成功 id=%s summary=%s tags=%s", memory_id[:8], summary, tags)

    def count(self) -> int:
        """返回 ChromaDB 中的记忆总数。"""
        return self._collection.count()

    # ------------------------------------------------------------------
    # 命中计数
    # ------------------------------------------------------------------

    def increment_hit_count(self, memory_id: str, delta: int = 1):
        """命中计数 +delta，同时记录 last_hit_time（threading.Lock 保证原子性）。"""
        with self._lock:
            result = self._collection.get(ids=[memory_id], include=["metadatas"])
            if not result["ids"]:
                return
            meta = dict(result["metadatas"][0])
            meta["hit_count"] = meta.get("hit_count", 0) + delta
            meta["last_hit_time"] = time.time()
            # 热升级：hit_count ≥ 3 或 情绪强度 ≥ 2 → hot
            if (meta.get("hit_count", 0) >= 3
                    or (meta.get("emotional_intensity", 0) or 0) >= 2
                    or meta.get("emotion_valence_bin", "") in ("positive", "negative")):
                meta["heat"] = "hot"
            self._collection.update(ids=[memory_id], metadatas=[meta])
            # 更新统计计数器
            self._total_hits += delta

        # 情绪淡化独立触发：每 50 次命中检查一次，3 天未提及则淡化
        self._desensitization_counter += 1
        if self._desensitization_counter >= self.DESENSITIZATION_CHECK_INTERVAL:
            self._desensitization_counter = 0
            self._apply_emotional_desensitization()

    def batch_increment_hit_count(self, ids_and_deltas: list[tuple[str, int]]):
        """批量命中计数：一次锁、一次 get、一次 update，替代逐条调用。

        用于检索管线——200 条命中 = 1 次 ChromaDB 往返，而非 200 次。
        保留完整的 hit_count 递增、last_hit_time 更新、热升级、情感淡化计数。
        """
        if not ids_and_deltas:
            return

        with self._lock:
            all_ids = [mid for mid, _ in ids_and_deltas]
            result = self._collection.get(ids=all_ids, include=["metadatas"])
            if not result["ids"]:
                return

            # 建立 id → metadata 索引
            meta_map: dict[str, dict] = {}
            for i, mid in enumerate(result["ids"]):
                meta_map[mid] = dict(result["metadatas"][i])

            # 按 delta 聚合（同一 id 可能被多条检索路径命中）
            aggregated: dict[str, int] = {}
            for mid, delta in ids_and_deltas:
                aggregated[mid] = aggregated.get(mid, 0) + delta

            now = time.time()
            update_ids = []
            update_metas = []

            for mid, total_delta in aggregated.items():
                meta = meta_map.get(mid)
                if meta is None:
                    continue
                meta["hit_count"] = meta.get("hit_count", 0) + total_delta
                meta["last_hit_time"] = now
                # 热升级：hit_count ≥ 3 或 情绪强度 ≥ 2 → hot
                if (meta.get("hit_count", 0) >= 3
                        or (meta.get("emotional_intensity", 0) or 0) >= 2
                        or meta.get("emotion_valence_bin", "") in ("positive", "negative")):
                    meta["heat"] = "hot"
                update_ids.append(mid)
                update_metas.append(meta)

            if update_ids:
                self._collection.update(ids=update_ids, metadatas=update_metas)
            # 更新统计计数器
            self._total_hits += sum(aggregated.values())

        # 情绪淡化计数：按去重前的调用次数递增（与旧逐条行为一致）
        self._desensitization_counter += len(ids_and_deltas)
        if self._desensitization_counter >= self.DESENSITIZATION_CHECK_INTERVAL:
            self._desensitization_counter = 0
            self._apply_emotional_desensitization()

    # ------------------------------------------------------------------
    # 情绪淡化
    # ------------------------------------------------------------------

    def _apply_emotional_desensitization(self):
        """情绪淡化：扫描所有 emotional_intensity>=1 的记忆，
        超过 EMOTION_DECAY_DAYS 天未被提及则 emotional_intensity 减 1，
        事实保留，情绪标签淡化。

        查询移出锁外，仅 update 时持有锁，避免阻塞前台写入。
        """
        # 查询阶段：不持锁
        try:
            result = self._collection.get(
                where={"emotional_intensity": {"$gte": 1}},
                include=["metadatas"],
            )
        except Exception as exc:
            logger.warning("情绪淡化查询失败: %s", exc)
            return

        if not result["ids"]:
            return

        now = datetime.now()
        cutoff = now - timedelta(days=self.EMOTION_DECAY_DAYS)
        updates: list[tuple[str, int]] = []  # (memory_id, new_ei)

        for memory_id, meta in zip(result["ids"], result["metadatas"]):
            meta = dict(meta)
            ei = meta.get("emotional_intensity", 0)
            if ei <= 0:
                continue

            # 获取最后命中时间，兜底用创建时间
            last_hit = meta.get("last_hit_time")
            if last_hit is not None:
                try:
                    last_dt = datetime.fromtimestamp(
                        last_hit if isinstance(last_hit, (int, float))
                        else float(last_hit)
                    )
                except (ValueError, TypeError, OSError):
                    last_dt = None
            else:
                last_dt = None

            # 无 last_hit_time 时，用原始时间戳兜底
            if last_dt is None:
                ts = meta.get("timestamp")
                if ts:
                    try:
                        last_dt = datetime.fromtimestamp(
                            ts if isinstance(ts, (int, float))
                            else float(ts)
                        )
                    except (ValueError, TypeError, OSError):
                        continue
                else:
                    continue

            if last_dt < cutoff:
                new_ei = max(0, ei - self.EMOTION_DECREMENT)
                updates.append((memory_id, new_ei))

        if not updates:
            return

        # 写入阶段：持锁批量 update（一次调用替代 N+1 逐条更新）
        with self._lock:
            if updates:
                batch_ids = [mid for mid, _ in updates]
                batch_metas = [{"emotional_intensity": new_ei} for _, new_ei in updates]
                self._collection.update(ids=batch_ids, metadatas=batch_metas)

        # 日志（锁外）
        for memory_id, new_ei in updates:
            mid_short = memory_id[:8]
            if new_ei == 0:
                logger.info("情绪淡化归零 id=%s emotional_intensity→0", mid_short)
            else:
                logger.info("情绪淡化 id=%s emotional_intensity→%d", mid_short, new_ei)

    # ------------------------------------------------------------------
    # 事实时序：取代标记
    # ------------------------------------------------------------------

    def supersede_memory(self, old_id: str, new_id: str, reason: str = ""):
        """标记旧记忆被新记忆取代（事实冲突/更新）。

        stale=True, superseded_by=new_id, 附加理由和时间。
        """
        from datetime import datetime as _dt
        with self._lock:
            self._collection.update(
                ids=[old_id],
                metadatas=[{
                    "stale": True,
                    "superseded_by": new_id,
                    "supersede_reason": reason,
                    "superseded_at": _dt.now().isoformat(),
                }],
            )
        self._invalidate_list_all_cache()
        logger.info(
            "事实取代: %s → %s reason=%s",
            old_id[:8], new_id[:8], reason[:60] if reason else "-",
        )

    def get_memories_by_timerange(
        self, since_ts: float = 0, until_ts: float | None = None, limit: int = 200,
    ) -> list[dict]:
        """按时间范围获取记忆列表（含 metadata），用于巩固分析。

        since_ts: 起始时间戳（含）
        until_ts: 结束时间戳（含），None 表示不设上限
        """
        all_items = self.list_all()
        result = []
        for m in all_items:
            ts = (m.get("metadata") or {}).get("timestamp", 0)
            if ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            result.append(m)
            if len(result) >= limit:
                break
        return result

    # ------------------------------------------------------------------
    # 记忆管理（列表 / 详情 / 删除 / 统计）
    # ------------------------------------------------------------------

    def list_memories(self, page: int = 1, per_page: int = 20,
                      sort: str = "time", order: str = "desc",
                      tag: str = "", date_from: float = 0, date_to: float = 0) -> dict:
        """分页返回记忆列表，支持筛选和排序。

        使用缓存的全量数据 + Python 过滤/排序/分页。
        list_all_cached() 有 5 分钟 TTL，避免每次请求全量扫描 ChromaDB。
        """
        all_result = self.list_all_cached()  # 缓存全量数据 + Python 侧筛选/分页

        items = []
        for mem in all_result:
            mid = mem.get("id", "")
            meta = mem.get("metadata") or {}
            # 标签筛选（逗号分隔，OR 逻辑）
            if tag:
                filter_tags = [t.strip() for t in tag.split(",") if t.strip()]
                mem_tags = [t.strip() for t in (meta.get("tags", "").split(",") if meta.get("tags") else []) if t.strip()]
                if not any(ft in mem_tags for ft in filter_tags):
                    continue
            # 时间范围筛选
            ts = meta.get("timestamp", 0)
            if date_from and ts < date_from:
                continue
            if date_to and ts > date_to:
                continue
            items.append({
                "id": mid,
                "timestamp": ts,
                "summary": meta.get("summary", ""),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "hit_count": meta.get("hit_count", 0),
            })
        # 排序
        rev = order != "asc"
        if sort == "hit_count":
            items.sort(key=lambda x: x["hit_count"], reverse=rev)
        else:
            items.sort(key=lambda x: x["timestamp"], reverse=rev)
        # 分页
        total = len(items)
        offset = (page - 1) * per_page
        items = items[offset:offset + per_page]
        return {"items": items, "total": total, "page": page, "per_page": per_page}

    def get_memory_detail(self, memory_id: str) -> dict | None:
        """单条记忆详情，含原始对话、元信息、上下文。"""
        result = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None

        meta = result["metadatas"][0]
        detail = {
            "id": memory_id,
            "document": result["documents"][0],
            "user_message": meta.get("user_message", ""),
            "ai_message": meta.get("ai_message", ""),
            "summary": meta.get("summary", ""),
            "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
            "timestamp": meta.get("timestamp", 0),
            "hit_count": meta.get("hit_count", 0),
        }
        return detail

    def delete_memory(self, memory_id: str):
        """从 ChromaDB 删除记忆。"""
        self._collection.delete(ids=[memory_id])
        self._emb_cache.pop(memory_id, None)
        self._invalidate_list_all_cache()
        logger.info("ChromaDB 删除成功 id=%s", memory_id[:8])

    def archive_topic_cluster(self, tag: str, memory_ids: list[str]):
        """将某个话题簇的记忆标记为归档。不删除，不参与日常检索。"""
        with self._lock:
            for mid in memory_ids:
                try:
                    self._collection.update(
                        ids=[mid],
                        metadatas=[{"archived": True}],
                    )
                except Exception as exc:
                    logger.warning("归档失败 id=%s: %s", mid[:8], exc)
                    continue
        self._invalidate_list_all_cache()
        logger.info("归档: tag=%s, %d 条", tag, len(memory_ids))

    def stats(self) -> dict:
        """记忆统计：使用运行计数器，零 ChromaDB I/O。"""
        total = self._collection.count()
        if total == 0:
            return {"total": 0, "total_hits": 0, "earliest": None, "latest": None}

        # 兜底：计数器未初始化（旧数据升级），一次性全量遍历回填
        if self._total_hits == 0 and total > 0:
            result = self._collection.get(include=["metadatas"])
            for meta in result["metadatas"]:
                self._total_hits += meta.get("hit_count", 0)
                ts = meta.get("timestamp", 0)
                if self._earliest_ts is None or ts < self._earliest_ts:
                    self._earliest_ts = ts
                if self._latest_ts is None or ts > self._latest_ts:
                    self._latest_ts = ts

        return {
            "total": total,
            "total_hits": self._total_hits,
            "earliest": self._earliest_ts,
            "latest": self._latest_ts,
        }

    # ------------------------------------------------------------------
    # Embedding 模型版本化 — 回填
    # ------------------------------------------------------------------
    def backfill_embed_model(self) -> int:
        """为现有缺乏 embed_model 字段的记忆补充默认值。

        返回补充的条数。标记文件存在时跳过，避免重复扫描。
        """
        if os.path.exists(EMBED_BACKFILL_MARKER):
            logger.info("Embedding 模型回填已执行过，跳过")
            return 0

        coll = self._collection
        result = coll.get(include=["metadatas"])
        if not result["ids"]:
            # 空库，写入标记文件即可
            self._write_backfill_marker()
            return 0

        backfilled = 0
        for i, mid in enumerate(result["ids"]):
            meta = dict(result["metadatas"][i])
            if "embed_model" not in meta:
                meta["embed_model"] = DEFAULT_EMBED_MODEL
                coll.update(ids=[mid], metadatas=[meta])
                backfilled += 1

        self._write_backfill_marker()
        if backfilled:
            logger.info("回填 embed_model 字段: %d 条", backfilled)
        return backfilled

    @staticmethod
    def _write_backfill_marker():
        parent = os.path.dirname(EMBED_BACKFILL_MARKER)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(EMBED_BACKFILL_MARKER, "w") as f:
            f.write("done")
        logger.info("回填标记文件已写入: %s", EMBED_BACKFILL_MARKER)

    # ------------------------------------------------------------------
    # 实验 / 管理用 — 全量遍历
    # ------------------------------------------------------------------
    def list_all(self) -> list[dict]:
        """返回全部记忆列表（不含 embedding），用于实验。"""
        coll = self._collection
        result = coll.get(include=["metadatas"])
        return _build_items(result)

    def list_since(self, since_ts: float, limit: int = 500) -> list[dict]:
        """按时间过滤：返回 timestamp >= since_ts 的记忆，Server 端过滤。"""
        try:
            result = self._collection.get(
                where={"timestamp": {"$gte": since_ts}},
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            logger.warning("list_since ChromaDB 过滤失败: %s", exc)
            return []
        items = _build_items(result)
        return items[:limit]

    def query_by_emotion(
        self, valence_range: tuple[float, float], limit: int = 20,
    ) -> list[dict]:
        """按情绪 valence 范围检索记忆 (Python 侧过滤, AI 记忆库专用)。

        Args:
            valence_range: (min_valence, max_valence)，含边界
            limit: 最多返回条数

        Returns:
            按 valence 接近度排序的记忆列表
        """
        all_items = self.list_all()
        lo, hi = valence_range
        candidates = []
        for mem in all_items:
            meta = mem.get("metadata") or {}
            mv = meta.get("emotion_valence")
            if mv is None:
                continue
            try:
                mv = float(mv)
            except (ValueError, TypeError):
                continue
            if lo <= mv <= hi:
                candidates.append(mem)
        # 按与范围中心的距离排序
        center = (lo + hi) / 2
        candidates.sort(key=lambda m: abs(
            float((m.get("metadata") or {}).get("emotion_valence", 0)) - center
        ))
        return candidates[:limit]

    def list_before(self, before_ts: float, limit: int = 500) -> list[dict]:
        """按时间过滤：返回 timestamp < before_ts 的记忆，Server 端过滤。"""
        try:
            result = self._collection.get(
                where={"timestamp": {"$lt": before_ts}},
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            logger.warning("list_before ChromaDB 过滤失败: %s", exc)
            return []
        items = _build_items(result)
        return items[:limit]

    def list_all_paginated(self, batch_size: int = 500) -> list[dict]:
        """分页获取全部记忆，避免一次性加载过大。"""
        offset = 0
        all_items = []
        while True:
            result = self._collection.get(
                offset=offset, limit=batch_size,
                include=["metadatas", "documents"],
            )
            if not result["ids"]:
                break
            items = _build_items(result)
            all_items.extend(items)
            if len(result["ids"]) < batch_size:
                break
            offset += batch_size
        return all_items

    def list_all_cached(self, ttl: float | None = None) -> list[dict]:
        """返回全部记忆列表（带缓存）。后台线程专用，减少 SQLite 全量读取。

        ttl: 缓存有效期（秒），默认 300（5 分钟）。
        写入操作（add/delete/supersede/archive）自动失效。
        """
        ttl = ttl if ttl is not None else self._list_all_cache_ttl
        now = time.time()
        with self._list_all_cache_lock:
            if (self._list_all_cache is not None
                    and now - self._list_all_cache_time < ttl):
                return self._list_all_cache
        # 缓存未命中，走原始 list_all
        result = self.list_all()
        with self._list_all_cache_lock:
            self._list_all_cache = result
            self._list_all_cache_time = now
        return result

    def _invalidate_list_all_cache(self):
        """写入操作后失效 list_all 缓存。"""
        lock = getattr(self, '_list_all_cache_lock', None)
        if lock is None:
            return  # __init__ 未执行（如测试直接用 __new__），安全跳过
        with lock:
            self._list_all_cache = None
            self._list_all_cache_time = 0.0

    # ------------------------------------------------------------------
    # Embedding 缓存（attention 位移因子用）
    # ------------------------------------------------------------------
    _EMB_CACHE_MAX = 10000  # 10K 条封顶，超出 memory 上限的旧数据不缓存

    def _build_embedding_cache(self):
        """从 ChromaDB 批量读取已有记忆的 embedding 到内存缓存。

        使用 collection.get(ids=batch, include=["embeddings"]) 分批读取，
        绕过全量 get(include=["embeddings"]) 的已知 ChromaDB bug。
        每批 500 条，封顶 _EMB_CACHE_MAX 条。
        """
        try:
            all_ids = self._collection.get(include=[])["ids"]
            if not all_ids:
                return
            # UUID v4 尾部近似时间序，取最新
            capped = all_ids[-self._EMB_CACHE_MAX:] if len(all_ids) > self._EMB_CACHE_MAX else all_ids
            batch_size = 500
            total = 0
            for i in range(0, len(capped), batch_size):
                batch = capped[i:i + batch_size]
                result = self._collection.get(ids=batch, include=["embeddings"])
                emb_list = result.get("embeddings") or []
                id_list = result.get("ids") or []
                with self._emb_cache_lock:
                    for j, mid in enumerate(id_list):
                        if j < len(emb_list) and emb_list[j] is not None:
                            self._emb_cache[mid] = emb_list[j]
                            total += 1
            logger.info("embedding 缓存构建完成: %d/%d 条", total, len(capped))
        except Exception as exc:
            with self._emb_cache_lock:
                self._emb_cache = {}
            logger.warning("embedding 缓存构建失败，回退空缓存: %s", exc)

    def _get_embedding_cached(self, memory_id: str) -> list | None:
        """获取缓存的 embedding。缓存由 _build_embedding_cache 预填充，
        新记忆在 store-time 由 context.py 写入。"""
        return self._emb_cache.get(memory_id)

    def close(self):
        """释放 ChromaDB PersistentClient 资源（带锁保护，防止双线程竞态）。"""
        with self._lock:
            self._client = None
            self._collection = None
        with self._emb_cache_lock:
            self._emb_cache = {}

