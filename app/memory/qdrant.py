"""记忆层 — Qdrant 存储/检索 + 上下文包裹管理 (替代 chroma.py).

Phase 2: 杀全量模式 — ChromaDB→Qdrant API 翻译层 + _translate_filter + MatchText。
"""
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from qdrant_client import QdrantClient, models

from app.config.settings import (
    DEFAULT_EMBED_MODEL,
    EMBED_MODELS,
    EMBED_BACKFILL_MARKER,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_ON_DISK,
    QDRANT_HNSW_M,
    QDRANT_HNSW_EF_CONSTRUCT,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Qdrant — 向量存储 & 检索
# ===================================================================

def _build_items_from_points(points: list) -> list[dict]:
    """Qdrant scroll/retrieve 结果 → 统一 item 格式（兼容 ChromaService._build_items）。"""
    items = []
    for pt in points:
        payload = pt.payload or {}
        items.append({
            "id": pt.id,
            "document": payload.get("document"),
            "metadata": dict(payload),
        })
    return items


# ===================================================================
# ChromaDB where → Qdrant Filter 翻译层 (Phase 2)
# ===================================================================

def _build_condition(key: str, value) -> models.FieldCondition | models.Filter:
    """构建单个字段条件。

    注意: $and / $or 由上层 _translate_filter 处理，不进入本函数。
    """
    if not isinstance(value, dict):
        # 简单等值: {"heat": "hot"}
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    for op, val in value.items():
        if op == "$gte":
            return models.FieldCondition(key=key, range=models.Range(gte=val))
        elif op == "$lte":
            return models.FieldCondition(key=key, range=models.Range(lte=val))
        elif op == "$gt":
            return models.FieldCondition(key=key, range=models.Range(gt=val))
        elif op == "$lt":
            return models.FieldCondition(key=key, range=models.Range(lt=val))
        elif op == "$eq":
            return models.FieldCondition(key=key, match=models.MatchValue(value=val))
        elif op == "$ne":
            return models.Filter(must_not=[
                models.FieldCondition(key=key, match=models.MatchValue(value=val))
            ])
        elif op == "$in":
            return models.FieldCondition(key=key, match=models.MatchAny(any=val))
        elif op == "$contains":
            # ChromaDB $contains → Qdrant MatchText (需 text index on field)
            return models.FieldCondition(key=key, match=models.MatchText(text=str(val)))

    raise ValueError(f"Unsupported operator in: {value}")


def _translate_filter(chroma_where: dict) -> models.Filter:
    """ChromaDB where dict → Qdrant Filter.

    支持的运算符: $gte, $lte, $gt, $lt, $eq, $ne, $in, $contains, $and, $or
    """
    if not chroma_where:
        return models.Filter()

    conditions = []
    for key, value in chroma_where.items():
        if key == "$and":
            # $and: [{"key": {"$gte": v}}, {"key": {"$lte": v}}]
            sub_conditions = []
            for sub_clause in value:
                if not isinstance(sub_clause, dict):
                    continue
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            if sub_conditions:
                conditions.append(models.Filter(must=sub_conditions))
        elif key == "$or":
            # $or: [{"key1": {"$eq": v1}}, {"key2": {"$eq": v2}}]
            sub_conditions = []
            for sub_clause in value:
                if not isinstance(sub_clause, dict):
                    continue
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            if sub_conditions:
                conditions.append(models.Filter(should=sub_conditions))
        else:
            conditions.append(_build_condition(key, value))

    if not conditions:
        return models.Filter()
    if len(conditions) == 1 and isinstance(conditions[0], models.Filter):
        return conditions[0]
    # 多个条件 → must 包裹 (AND 语义)
    return models.Filter(must=[
        c if isinstance(c, models.Filter) else models.Filter(must=[c])
        for c in conditions
    ])


# ===================================================================
# Qdrant → ChromaDB Collection API 兼容适配器 (Phase 2)
# ===================================================================

class _QdrantCollectionCompat:
    """使 QdrantService._collection 表现如 chromadb Collection。

    所有 ChromaDB collection API (query/get/update/count) 翻译为 Qdrant API。
    pipeline.py / context.py / dispatch.py 通过此适配器无需改动即可使用 Qdrant。
    """

    def __init__(self, service: 'QdrantService'):
        self._svc = service

    def query(self, query_embeddings, n_results, where=None, include=None):
        """ChromaDB col.query() → Qdrant search()。

        返回格式（兼容 ChromaDB）:
          {"ids": [[id1,id2,...]], "documents": [["doc1","doc2",...]],
           "metadatas": [[{...},{...},...]], "distances": [[0.1,0.2,...]]}
        """
        qf = _translate_filter(where) if where else None
        include_docs = include is None or "documents" in include
        include_meta = include is None or "metadatas" in include

        ids_list, docs_list, metas_list, dists_list = [], [], [], []
        for q_emb in query_embeddings:
            try:
                results = self._svc._client.search(
                    collection_name=self._svc._collection_name,
                    query_vector=q_emb,
                    query_filter=qf,
                    limit=n_results,
                    with_payload=include_meta,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Qdrant search 失败: %s", exc)
                results = []

            batch_ids, batch_docs, batch_metas, batch_dists = [], [], [], []
            for pt in results:
                batch_ids.append(pt.id)
                payload = pt.payload or {}
                batch_docs.append(payload.get("document", "") if include_docs else "")
                batch_metas.append(dict(payload) if include_meta else {})
                # Qdrant search returns score (cosine similarity), ChromaDB returns distance (1 - similarity)
                batch_dists.append(1.0 - pt.score)

            ids_list.append(batch_ids)
            docs_list.append(batch_docs)
            metas_list.append(batch_metas)
            dists_list.append(batch_dists)

        return {
            "ids": ids_list,
            "documents": docs_list,
            "metadatas": metas_list,
            "distances": dists_list,
        }

    def get(self, ids=None, where=None, include=None, limit=None):
        """ChromaDB col.get() → Qdrant retrieve()/scroll()。

        返回格式（兼容 ChromaDB）:
          {"ids": [...], "documents": [...], "metadatas": [...]}
        """
        include_docs = include is None or "documents" in include
        include_meta = include is None or "metadatas" in include

        if ids:
            # 按 ID 精确检索
            id_list = ids if isinstance(ids, list) else list(ids)
            try:
                pts = self._svc._client.retrieve(
                    collection_name=self._svc._collection_name,
                    ids=id_list,
                    with_payload=include_meta,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Qdrant retrieve 失败: %s", exc)
                pts = []

            result_ids, result_docs, result_metas = [], [], []
            # 保持输入 ID 顺序
            id_to_pt = {pt.id: pt for pt in pts}
            for mid in id_list:
                pt = id_to_pt.get(mid)
                if pt is None:
                    continue
                result_ids.append(mid)
                payload = pt.payload or {}
                result_docs.append(payload.get("document", "") if include_docs else "")
                result_metas.append(dict(payload) if include_meta else {})
            return {"ids": result_ids, "documents": result_docs, "metadatas": result_metas}

        else:
            # 按 filter 检索
            qf = _translate_filter(where) if where else None
            _limit = limit or 50000
            try:
                pts, _ = self._svc._client.scroll(
                    collection_name=self._svc._collection_name,
                    scroll_filter=qf,
                    with_payload=True,
                    limit=_limit,
                )
            except Exception as exc:
                logger.warning("Qdrant scroll 失败: %s", exc)
                pts = []

            result_ids, result_docs, result_metas = [], [], []
            for pt in pts:
                result_ids.append(pt.id)
                payload = pt.payload or {}
                result_docs.append(payload.get("document", "") if include_docs else "")
                result_metas.append(dict(payload) if include_meta else {})
            return {"ids": result_ids, "documents": result_docs, "metadatas": result_metas}

    def update(self, ids, metadatas=None, embeddings=None):
        """ChromaDB col.update() → Qdrant set_payload()/update_vectors()。"""
        if not ids:
            return
        id_list = ids if isinstance(ids, list) else [ids]
        with self._svc._lock:
            if metadatas:
                for i, mid in enumerate(id_list):
                    if i < len(metadatas) and metadatas[i]:
                        try:
                            self._svc._client.set_payload(
                                collection_name=self._svc._collection_name,
                                payload=dict(metadatas[i]),
                                points=[mid],
                            )
                        except Exception as exc:
                            logger.warning("Qdrant set_payload 失败 id=%s: %s", mid[:8], exc)
            if embeddings:
                for i, mid in enumerate(id_list):
                    if i < len(embeddings) and embeddings[i]:
                        try:
                            self._svc._client.update_vectors(
                                collection_name=self._svc._collection_name,
                                points=[models.PointVectors(id=mid, vector=embeddings[i])],
                            )
                        except Exception as exc:
                            logger.warning("Qdrant update_vectors 失败 id=%s: %s", mid[:8], exc)

    def count(self):
        return self._svc._client.count(collection_name=self._svc._collection_name).count


class QdrantService:
    """Qdrant 记忆存储与检索（API 完全兼容 ChromaService）。"""

    # 情绪淡化参数
    DESENSITIZATION_CHECK_INTERVAL = 50
    EMOTION_DECAY_DAYS = 3
    EMOTION_DECREMENT = 1

    def __init__(self, persist_dir: str | None = None, *,
                 collection_name: str | None = None):
        # Qdrant 客户端：优先用 URL 连接服务器，否则本地模式
        if QDRANT_URL and not _is_default_local_url():
            self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
            logger.info("Qdrant 服务端模式: %s", QDRANT_URL)
        else:
            # 本地模式：用 persist_dir 作为存储路径
            local_path = persist_dir or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "qdrant_local",
            )
            os.makedirs(local_path, exist_ok=True)
            self._client = QdrantClient(path=local_path)
            logger.info("Qdrant 本地模式: %s", local_path)

        self._lock = threading.Lock()
        self._desensitization_counter = 0
        # Embedding 缓存
        self._emb_cache: dict[str, list] = {}
        self._emb_cache_lock = threading.Lock()
        # list_all 缓存
        self._list_all_cache: list[dict] | None = None
        self._list_all_cache_time: float = 0.0
        self._list_all_cache_lock = threading.Lock()
        self._list_all_cache_ttl: float = 300
        # stats() 运行计数器
        self._total_hits: int = 0
        self._earliest_ts: float | None = None
        self._latest_ts: float | None = None

        coll_name = collection_name or EMBED_MODELS[DEFAULT_EMBED_MODEL]["collection"]

        # 确保 collection 存在
        existing = {c.name for c in self._client.get_collections().collections}
        if coll_name not in existing:
            self._client.create_collection(
                collection_name=coll_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    on_disk=QDRANT_ON_DISK,
                    hnsw_config=models.HnswConfigDiff(
                        m=QDRANT_HNSW_M,
                        ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                    ) if not _is_default_local_url() else None,
                ),
            )
            logger.info("创建 Qdrant collection: %s", coll_name)

        self._collection_name = coll_name
        # Phase 2: ChromaDB → Qdrant API 兼容适配器
        self._collection = _QdrantCollectionCompat(self)

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
        """写入一轮对话到 Qdrant。"""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        document = f"用户：{user_message}\nAI：{ai_message}"

        _initial_heat = "hot" if (
            (time_features and time_features.get("emotional_intensity", 0) or 0) >= 2
        ) else "warm"

        payload = {
            "user_message": user_message,
            "ai_message": ai_message,
            "document": document,
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
        # entities 保持原生 list[dict]，不 JSON 序列化
        if entities:
            payload["entities"] = entities
        if date_tag:
            payload["date_tag"] = date_tag
        if time_features:
            payload.update(time_features)

        self._client.upsert(
            collection_name=self._collection_name,
            points=[models.PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload,
            )],
        )
        logger.info("记忆写入完成 id=%s source=%s summary=%s", memory_id[:8], source, summary[:60])
        if self._earliest_ts is None or timestamp < self._earliest_ts:
            self._earliest_ts = timestamp
        if self._latest_ts is None or timestamp > self._latest_ts:
            self._latest_ts = timestamp
        self._invalidate_list_all_cache()
        return memory_id

    def mark_storage_complete(self, memory_id: str):
        """标记记忆入库完成。"""
        with self._lock:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={"storage_complete": True},
                points=[memory_id],
            )

    def update_memory(
        self,
        memory_id: str,
        summary: str,
        tags: list[str],
        embedding: list[float],
    ):
        """纠正记忆：更新 summary/tags 和 embedding。"""
        self._client.set_payload(
            collection_name=self._collection_name,
            payload={"summary": summary, "tags": ",".join(tags)},
            points=[memory_id],
        )
        self._client.update_vectors(
            collection_name=self._collection_name,
            points=[models.PointVectors(id=memory_id, vector=embedding)],
        )
        logger.info("记忆纠正成功 id=%s summary=%s tags=%s", memory_id[:8], summary, tags)

    def count(self) -> int:
        """返回记忆总数。"""
        return self._client.count(collection_name=self._collection_name).count

    # ------------------------------------------------------------------
    # 命中计数
    # ------------------------------------------------------------------

    def increment_hit_count(self, memory_id: str, delta: int = 1):
        """命中计数 +delta，记录 last_hit_time。"""
        with self._lock:
            pts = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[memory_id],
                with_payload=True,
            )
            if not pts:
                return
            payload = dict(pts[0].payload or {})
            payload["hit_count"] = payload.get("hit_count", 0) + delta
            payload["last_hit_time"] = time.time()
            if (payload.get("hit_count", 0) >= 3
                    or (payload.get("emotional_intensity", 0) or 0) >= 2
                    or payload.get("emotion_valence_bin", "") in ("positive", "negative")):
                payload["heat"] = "hot"
            self._client.overwrite_payload(
                collection_name=self._collection_name,
                payload=payload,
                points=[memory_id],
            )
            self._total_hits += delta

        self._desensitization_counter += 1
        if self._desensitization_counter >= self.DESENSITIZATION_CHECK_INTERVAL:
            self._desensitization_counter = 0
            self._apply_emotional_desensitization()

    def batch_increment_hit_count(self, ids_and_deltas: list[tuple[str, int]]):
        """批量命中计数：一次检索 + N 次 set_payload（合并器锁保护）。"""
        if not ids_and_deltas:
            return

        with self._lock:
            all_ids = [mid for mid, _ in ids_and_deltas]
            pts = self._client.retrieve(
                collection_name=self._collection_name,
                ids=all_ids,
                with_payload=True,
            )
            if not pts:
                return

            payload_map: dict[str, dict] = {}
            for pt in pts:
                payload_map[pt.id] = dict(pt.payload or {})

            aggregated: dict[str, int] = {}
            for mid, delta in ids_and_deltas:
                aggregated[mid] = aggregated.get(mid, 0) + delta

            now = time.time()
            for mid, total_delta in aggregated.items():
                payload = payload_map.get(mid)
                if payload is None:
                    continue
                payload["hit_count"] = payload.get("hit_count", 0) + total_delta
                payload["last_hit_time"] = now
                if (payload.get("hit_count", 0) >= 3
                        or (payload.get("emotional_intensity", 0) or 0) >= 2
                        or payload.get("emotion_valence_bin", "") in ("positive", "negative")):
                    payload["heat"] = "hot"
                self._client.overwrite_payload(
                    collection_name=self._collection_name,
                    payload=payload,
                    points=[mid],
                )
            self._total_hits += sum(aggregated.values())

        self._desensitization_counter += len(ids_and_deltas)
        if self._desensitization_counter >= self.DESENSITIZATION_CHECK_INTERVAL:
            self._desensitization_counter = 0
            self._apply_emotional_desensitization()

    # ------------------------------------------------------------------
    # 情绪淡化
    # ------------------------------------------------------------------

    def _apply_emotional_desensitization(self):
        """情绪淡化：扫描 emotional_intensity>=1 的记忆，超期未命中则减 1。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="emotional_intensity",
                        range=models.Range(gte=1),
                    ),
                ]),
                with_payload=True,
                limit=10000,
            )
        except Exception as exc:
            logger.warning("情绪淡化查询失败: %s", exc)
            return

        if not pts:
            return

        now = datetime.now()
        cutoff = now - timedelta(days=self.EMOTION_DECAY_DAYS)
        updates: list[tuple[str, dict]] = []

        for pt in pts:
            payload = dict(pt.payload or {})
            ei = payload.get("emotional_intensity", 0)
            if ei <= 0:
                continue

            last_hit = payload.get("last_hit_time")
            if last_hit is not None:
                try:
                    last_dt = datetime.fromtimestamp(float(last_hit))
                except (ValueError, TypeError, OSError):
                    last_dt = None
            else:
                last_dt = None

            if last_dt is None:
                ts = payload.get("timestamp")
                if ts:
                    try:
                        last_dt = datetime.fromtimestamp(float(ts))
                    except (ValueError, TypeError, OSError):
                        continue
                else:
                    continue

            if last_dt < cutoff:
                new_ei = max(0, ei - self.EMOTION_DECREMENT)
                updates.append((pt.id, {"emotional_intensity": new_ei}))

        if not updates:
            return

        with self._lock:
            for mid, payload_update in updates:
                self._client.set_payload(
                    collection_name=self._collection_name,
                    payload=payload_update,
                    points=[mid],
                )

        for memory_id, payload_update in updates:
            mid_short = memory_id[:8]
            new_ei = payload_update["emotional_intensity"]
            if new_ei == 0:
                logger.info("情绪淡化归零 id=%s emotional_intensity→0", mid_short)
            else:
                logger.info("情绪淡化 id=%s emotional_intensity→%d", mid_short, new_ei)

    # ------------------------------------------------------------------
    # 事实时序：取代标记
    # ------------------------------------------------------------------

    def supersede_memory(self, old_id: str, new_id: str, reason: str = ""):
        """标记旧记忆被新记忆取代。"""
        from datetime import datetime as _dt
        with self._lock:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={
                    "stale": True,
                    "superseded_by": new_id,
                    "supersede_reason": reason,
                    "superseded_at": _dt.now().isoformat(),
                },
                points=[old_id],
            )
        self._invalidate_list_all_cache()
        logger.info(
            "事实取代: %s → %s reason=%s",
            old_id[:8], new_id[:8], reason[:60] if reason else "-",
        )

    def get_memories_by_timerange(
        self, since_ts: float = 0, until_ts: float | None = None, limit: int = 200,
    ) -> list[dict]:
        """按时间范围获取记忆列表。"""
        must = [models.FieldCondition(
            key="timestamp", range=models.Range(gte=since_ts),
        )]
        if until_ts is not None:
            must.append(models.FieldCondition(
                key="timestamp", range=models.Range(lte=until_ts),
            ))
        pts, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=models.Filter(must=must),
            with_payload=True,
            limit=limit,
        )
        return _build_items_from_points(pts)

    # ------------------------------------------------------------------
    # 记忆管理（列表 / 详情 / 删除 / 统计）
    # ------------------------------------------------------------------

    def list_memories(self, page: int = 1, per_page: int = 20,
                      sort: str = "time", order: str = "desc",
                      tag: str = "", date_from: float = 0, date_to: float = 0) -> dict:
        """分页返回记忆列表。"""
        all_result = self.list_all_cached()

        items = []
        for mem in all_result:
            meta = mem.get("metadata") or {}
            if tag:
                filter_tags = [t.strip() for t in tag.split(",") if t.strip()]
                mem_tags_str = meta.get("tags", "")
                mem_tags = [t.strip() for t in mem_tags_str.split(",") if t.strip()] if mem_tags_str else []
                if not any(ft in mem_tags for ft in filter_tags):
                    continue
            ts = meta.get("timestamp", 0)
            if date_from and ts < date_from:
                continue
            if date_to and ts > date_to:
                continue
            items.append({
                "id": mem.get("id", ""),
                "timestamp": ts,
                "summary": meta.get("summary", ""),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "hit_count": meta.get("hit_count", 0),
            })
        rev = order != "asc"
        if sort == "hit_count":
            items.sort(key=lambda x: x["hit_count"], reverse=rev)
        else:
            items.sort(key=lambda x: x["timestamp"], reverse=rev)
        total = len(items)
        offset = (page - 1) * per_page
        items = items[offset:offset + per_page]
        return {"items": items, "total": total, "page": page, "per_page": per_page}

    def get_memory_detail(self, memory_id: str) -> dict | None:
        """单条记忆详情。"""
        pts = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[memory_id],
            with_payload=True,
        )
        if not pts:
            return None
        payload = pts[0].payload or {}
        return {
            "id": memory_id,
            "document": payload.get("document", ""),
            "user_message": payload.get("user_message", ""),
            "ai_message": payload.get("ai_message", ""),
            "summary": payload.get("summary", ""),
            "tags": payload.get("tags", "").split(",") if payload.get("tags") else [],
            "timestamp": payload.get("timestamp", 0),
            "hit_count": payload.get("hit_count", 0),
        }

    def delete_memory(self, memory_id: str):
        """删除记忆。"""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[memory_id],
        )
        self._emb_cache.pop(memory_id, None)
        self._invalidate_list_all_cache()
        logger.info("Qdrant 删除成功 id=%s", memory_id[:8])

    def archive_topic_cluster(self, tag: str, memory_ids: list[str]):
        """将话题簇标记为归档。"""
        with self._lock:
            for mid in memory_ids:
                try:
                    self._client.set_payload(
                        collection_name=self._collection_name,
                        payload={"archived": True},
                        points=[mid],
                    )
                except Exception as exc:
                    logger.warning("归档失败 id=%s: %s", mid[:8], exc)
                    continue
        self._invalidate_list_all_cache()
        logger.info("归档: tag=%s, %d 条", tag, len(memory_ids))

    def stats(self) -> dict:
        """记忆统计：使用运行计数器。"""
        total = self._client.count(collection_name=self._collection_name).count
        if total == 0:
            return {"total": 0, "total_hits": 0, "earliest": None, "latest": None}

        if self._total_hits == 0 and total > 0:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=["hit_count", "timestamp"],
                limit=10000,
            )
            for pt in pts:
                p = pt.payload or {}
                self._total_hits += p.get("hit_count", 0)
                ts = p.get("timestamp", 0)
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
        """为缺乏 embed_model 字段的记忆补充默认值。"""
        if os.path.exists(EMBED_BACKFILL_MARKER):
            logger.info("Embedding 模型回填已执行过，跳过")
            return 0

        pts, _ = self._client.scroll(
            collection_name=self._collection_name,
            with_payload=["embed_model"],
            limit=10000,
        )
        if not pts:
            self._write_backfill_marker()
            return 0

        backfilled = 0
        for pt in pts:
            payload = pt.payload or {}
            if "embed_model" not in payload:
                self._client.set_payload(
                    collection_name=self._collection_name,
                    payload={"embed_model": DEFAULT_EMBED_MODEL},
                    points=[pt.id],
                )
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
        """返回全部记忆列表（不含 embedding）。"""
        pts, _ = self._client.scroll(
            collection_name=self._collection_name,
            with_payload=True,
            limit=100000,
        )
        return _build_items_from_points(pts)

    def list_since(self, since_ts: float, limit: int = 500) -> list[dict]:
        """按时间过滤：返回 timestamp >= since_ts 的记忆。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="timestamp", range=models.Range(gte=since_ts),
                    ),
                ]),
                with_payload=True,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("list_since Qdrant 过滤失败: %s", exc)
            return []
        return _build_items_from_points(pts)

    def query_by_emotion(
        self, valence_range: tuple[float, float], limit: int = 20,
    ) -> list[dict]:
        """按情绪 valence 范围检索记忆。"""
        lo, hi = valence_range
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="emotion_valence", range=models.Range(gte=lo, lte=hi),
                    ),
                ]),
                with_payload=True,
                limit=limit * 5,  # 多取一些用于排序
            )
        except Exception:
            # Qdrant 本地模式可能不支持 range filter，回退到全量 + Python 过滤
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=True,
                limit=10000,
            )

        candidates = []
        for pt in pts:
            p = pt.payload or {}
            mv = p.get("emotion_valence")
            if mv is None:
                continue
            try:
                mv = float(mv)
            except (ValueError, TypeError):
                continue
            if lo <= mv <= hi:
                candidates.append({"id": pt.id, "document": p.get("document"), "metadata": dict(p)})

        center = (lo + hi) / 2
        candidates.sort(key=lambda m: abs(
            float((m.get("metadata") or {}).get("emotion_valence", 0)) - center
        ))
        return candidates[:limit]

    def list_before(self, before_ts: float, limit: int = 500) -> list[dict]:
        """按时间过滤：返回 timestamp < before_ts 的记忆。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="timestamp", range=models.Range(lt=before_ts),
                    ),
                ]),
                with_payload=True,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("list_before Qdrant 过滤失败: %s", exc)
            return []
        return _build_items_from_points(pts)

    def list_all_paginated(self, batch_size: int = 500) -> list[dict]:
        """分页获取全部记忆。"""
        all_items = []
        offset = None
        while True:
            pts, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=True,
                limit=batch_size,
                offset=offset,
            )
            if not pts:
                break
            all_items.extend(_build_items_from_points(pts))
            if next_offset is None or len(pts) < batch_size:
                break
            offset = next_offset
        return all_items

    def list_all_cached(self, ttl: float | None = None) -> list[dict]:
        """返回全部记忆列表（带缓存）。"""
        ttl = ttl if ttl is not None else self._list_all_cache_ttl
        now = time.time()
        with self._list_all_cache_lock:
            if (self._list_all_cache is not None
                    and now - self._list_all_cache_time < ttl):
                return self._list_all_cache
        result = self.list_all()
        with self._list_all_cache_lock:
            self._list_all_cache = result
            self._list_all_cache_time = now
        return result

    def _invalidate_list_all_cache(self):
        """写入操作后失效 list_all 缓存。"""
        lock = getattr(self, '_list_all_cache_lock', None)
        if lock is None:
            return
        with lock:
            self._list_all_cache = None
            self._list_all_cache_time = 0.0

    # ------------------------------------------------------------------
    # Embedding 缓存（attention 位移因子用）
    # ------------------------------------------------------------------
    _EMB_CACHE_MAX = 10000

    def _build_embedding_cache(self):
        """从 Qdrant 批量读取已有记忆的 embedding 到内存缓存。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=["last_hit_time"],
                with_vectors=True,
                limit=self._EMB_CACHE_MAX,
            )
            with self._emb_cache_lock:
                for pt in pts:
                    if pt.vector is not None:
                        self._emb_cache[pt.id] = pt.vector
            logger.info("embedding 缓存构建完成: %d/%d 条", len(self._emb_cache), len(pts))
        except Exception as exc:
            with self._emb_cache_lock:
                self._emb_cache = {}
            logger.warning("embedding 缓存构建失败，回退空缓存: %s", exc)

    def _get_embedding_cached(self, memory_id: str) -> list | None:
        """获取缓存的 embedding。"""
        return self._emb_cache.get(memory_id)

    # ------------------------------------------------------------------
    # ChromaDB Collection API 兼容 (Phase 2: delegate to _collection adapter)
    # ------------------------------------------------------------------

    def get(self, ids=None, where=None, include=None, limit=None):
        """ChromaDB col.get() 兼容 — delegate to _collection adapter。"""
        return self._collection.get(ids=ids, where=where, include=include, limit=limit)

    def query(self, query_embeddings, n_results, where=None, include=None):
        """ChromaDB col.query() 兼容 — delegate to _collection adapter。"""
        return self._collection.query(
            query_embeddings=query_embeddings, n_results=n_results,
            where=where, include=include,
        )

    def close(self):
        """释放 Qdrant 客户端资源。"""
        with self._lock:
            self._client.close() if hasattr(self._client, 'close') else None
        with self._emb_cache_lock:
            self._emb_cache = {}

def _is_default_local_url() -> bool:
    """判断 QDRANT_URL 是否是默认本地地址（未显式配置远程服务器）。"""
    return (QDRANT_URL in ("http://localhost:6333", "http://127.0.0.1:6333")
            and os.getenv("QDRANT_URL") is None)
