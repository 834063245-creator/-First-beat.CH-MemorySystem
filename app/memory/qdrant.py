"""记忆层 — Qdrant 存储/检索 + 上下文包裹管理（唯一向量存储后端）。

Phase 4: 百万级硬骨头 — 量化 + payload 索引 + embedding 缓存 LRU。
Phase 5: ChromaDB 已移除，Qdrant 为唯一后端。
"""
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
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
    QDRANT_QUANTIZATION,
    QDRANT_QUANTIZATION_QUANTILE,
    QDRANT_EMB_CACHE_MAX,
    QDRANT_EMB_CACHE_BATCH,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 本地 Payload 索引 (Phase 4 — 本地模式补偿)
# ===================================================================

def _is_local_client(client: QdrantClient) -> bool:
    """检测是否是 Qdrant 本地模式（无服务端索引）。"""
    from qdrant_client.local.qdrant_local import QdrantLocal
    return isinstance(client._client, QdrantLocal)


class _LocalPayloadIndex:
    """本地模式的内存 payload 索引，补偿 Qdrant 本地引擎无服务端索引的限制。

    设计:
      - keyword/boolean 字段: dict[value, set[point_id]] → MatchValue/MatchAny O(1)
      - float/int 字段: sorted list[(value, point_id)] → Range O(log n)
      - 增量维护: add/remove 实时更新，无需重建
      - 内存: 10K 条 × 8 字段 ≈ 5-10MB

    线程安全: 外部调用方自行加锁（QdrantService._lock 已保护写入路径）。
    """

    def __init__(self):
        # keyword/boolean 索引: field -> value -> set[point_id]
        self._kw: dict[str, dict[str, set[str]]] = {}
        # float/int 索引: field -> list[(value, point_id)] (sorted)
        self._num: dict[str, list[tuple[float, str]]] = {}
        self._num_dirty: set[str] = set()  # 标记需要重排的字段
        # 全量 ID 集合 (用于 must_not)
        self._all_ids: set[str] = set()

    # ── 构建 ──

    def build(self, points: list):
        """从已有 points 构建全量索引。"""
        self._kw.clear()
        self._num.clear()
        self._all_ids.clear()
        for pt in points:
            self._index_point(pt.id, pt.payload or {})
        self._sort_all_dirty()

    def _index_point(self, pid: str, payload: dict):
        """索引单条 point 的 payload。"""
        self._all_ids.add(pid)
        for field, val in payload.items():
            if val is None:
                continue
            if isinstance(val, bool):
                k = str(val).lower()
                self._kw.setdefault(field, {}).setdefault(k, set()).add(pid)
            elif isinstance(val, str):
                self._kw.setdefault(field, {}).setdefault(val, set()).add(pid)
            elif isinstance(val, (int, float)):
                store = self._num.setdefault(field, [])
                store.append((float(val), pid))
                self._num_dirty.add(field)

    # ── 增量维护 ──

    def add(self, pid: str, payload: dict):
        """添加/全量更新一条 point 的索引。"""
        self._remove_point(pid)
        self._index_point(pid, payload)

    def update(self, pid: str, partial_payload: dict):
        """部分更新 — 仅重新索引 payload 中出现的字段，其他字段保持不变。"""
        for field in partial_payload:
            self._remove_field(pid, field)
        self._index_point(pid, partial_payload)

    def remove(self, pid: str):
        """增量删除一条 point。"""
        self._remove_point(pid)
        self._all_ids.discard(pid)

    def _remove_point(self, pid: str):
        """从所有索引中移除一个 point。"""
        self._all_ids.discard(pid)
        for field, val_idx in self._kw.items():
            for val, pids in list(val_idx.items()):
                pids.discard(pid)
                if not pids:
                    del val_idx[val]
        for field in list(self._num.keys()):
            self._num[field] = [(v, p) for v, p in self._num.get(field, []) if p != pid]
            self._num_dirty.add(field)

    def _remove_field(self, pid: str, field: str):
        """仅移除 point 在某个字段上的索引（用于部分更新）。"""
        if field in self._kw:
            for val, pids in list(self._kw[field].items()):
                pids.discard(pid)
                if not pids:
                    del self._kw[field][val]
        if field in self._num:
            self._num[field] = [(v, p) for v, p in self._num[field] if p != pid]
            self._num_dirty.add(field)

    def _sort_all_dirty(self):
        """重排所有脏的数值索引。"""
        for field in list(self._num_dirty):
            self._num[field].sort(key=lambda x: x[0])
        self._num_dirty.clear()

    def resolve(self, qdrant_filter: models.Filter | None) -> set[str] | None:
        """将 Qdrant Filter 解析为匹配的 point ID 集合。

        返回 None 表示「无法用索引解析，请回退到暴力扫描」。
        返回空 set 表示「索引确定没有匹配结果」。
        """
        if qdrant_filter is None:
            return None
        self._sort_all_dirty()
        return self._resolve_filter(qdrant_filter)

    def _resolve_filter(self, f: models.Filter) -> set[str] | None:
        result: set[str] | None = None

        if hasattr(f, 'must') and f.must:
            ids: set[str] | None = None
            for cond in f.must:
                sub = self._resolve_condition(cond)
                if sub is None:
                    return None
                if ids is None:
                    ids = set(sub)
                else:
                    ids &= sub
                if not ids:
                    return set()
            result = ids if ids is not None else set(self._all_ids)

        if hasattr(f, 'should') and f.should:
            union: set[str] = set()
            for cond in f.should:
                sub = self._resolve_condition(cond)
                if sub is None:
                    return None
                union |= sub
            if result is None:
                result = union
            else:
                result &= union

        if hasattr(f, 'must_not') and f.must_not:
            for cond in f.must_not:
                sub = self._resolve_condition(cond)
                if sub is None:
                    return None
                if result is None:
                    result = set(self._all_ids)
                result -= sub

        return result if result is not None else set(self._all_ids)

    def _resolve_condition(self, cond) -> set[str] | None:
        if hasattr(cond, 'key') and hasattr(cond, 'match'):
            return self._resolve_match(cond.key, cond.match)
        if hasattr(cond, 'key') and hasattr(cond, 'range'):
            return self._resolve_range(cond.key, cond.range)
        if hasattr(cond, 'must') or hasattr(cond, 'should') or hasattr(cond, 'must_not'):
            return self._resolve_filter(cond)
        return None

    def _resolve_match(self, key: str, match) -> set[str] | None:
        if hasattr(match, 'value'):
            val = match.value
            lookup = str(val).lower() if isinstance(val, bool) else val
            kw_idx = self._kw.get(key, {})
            return kw_idx.get(lookup, set())
        elif hasattr(match, 'any'):
            kw_idx = self._kw.get(key, {})
            result: set[str] = set()
            for v in match.any:
                result |= kw_idx.get(v, set())
            return result
        elif hasattr(match, 'except_'):
            return None
        elif hasattr(match, 'text'):
            return None
        return None

    def _resolve_range(self, key: str, r) -> set[str] | None:
        store = self._num.get(key)
        if not store:
            return set()
        lo = r.gt if hasattr(r, 'gt') and r.gt is not None else \
             (r.gte if hasattr(r, 'gte') and r.gte is not None else None)
        hi = r.lt if hasattr(r, 'lt') and r.lt is not None else \
             (r.lte if hasattr(r, 'lte') and r.lte is not None else None)
        if lo is None and hi is None:
            return None

        import bisect
        if lo is not None:
            lo_idx = bisect.bisect_right(store, (lo - 1e-10, ""), key=lambda x: x[0])
        else:
            lo_idx = 0
        if hi is not None:
            hi_idx = bisect.bisect_left(store, (hi + 1e-10, ""), key=lambda x: x[0])
        else:
            hi_idx = len(store)

        return {pid for _, pid in store[lo_idx:hi_idx]}

    def stats(self) -> dict:
        return {
            "kw_fields": len(self._kw),
            "num_fields": len(self._num),
            "total_ids": len(self._all_ids),
            "approx_memory_bytes": (
                sum(len(v) * 80 for idx in self._kw.values() for v in idx.values())
                + sum(len(s) * 24 for s in self._num.values())
            ),
        }


# ===================================================================
# 量化 & 索引配置 (Phase 4)
# ===================================================================

def _build_quantization_config() -> models.QuantizationConfig | None:
    """构建 Qdrant 量化配置。Phase 4: scalar_int8 将 4GB→1GB 向量存储。"""
    if not QDRANT_QUANTIZATION or _is_default_local_url():
        return None
    if QDRANT_QUANTIZATION == "scalar_int8":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=QDRANT_QUANTIZATION_QUANTILE,
                always_ram=True,
            ),
        )
    logger.warning("未知量化类型: %s，跳过量化配置", QDRANT_QUANTIZATION)
    return None


# ===================================================================
# Qdrant — 向量存储 & 检索
# ===================================================================

def _build_items_from_points(points: list) -> list[dict]:
    """Qdrant scroll/retrieve 结果 → 统一 item 格式。"""
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
# 旧 ChromaDB where 格式 → Qdrant Filter 翻译层 (Phase 2 遗留兼容)
# ===================================================================

def _build_condition(key: str, value) -> models.FieldCondition | models.Filter:
    """构建单个字段条件。"""
    if not isinstance(value, dict):
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
            return models.FieldCondition(key=key, match=models.MatchText(text=str(val)))

    raise ValueError(f"Unsupported operator in: {value}")


def _translate_filter(chroma_where: dict) -> models.Filter:
    """旧 ChromaDB where dict 格式 → Qdrant Filter。

    支持的运算符: $gte, $lte, $gt, $lt, $eq, $ne, $in, $contains, $and, $or
    """
    if not chroma_where:
        return models.Filter()

    conditions = []
    for key, value in chroma_where.items():
        if key == "$and":
            sub_conditions = []
            for sub_clause in value:
                if not isinstance(sub_clause, dict):
                    continue
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            if sub_conditions:
                conditions.append(models.Filter(must=sub_conditions))
        elif key == "$or":
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
    return models.Filter(must=[
        c if isinstance(c, models.Filter) else models.Filter(must=[c])
        for c in conditions
    ])


# ===================================================================
# 集合操作适配器 — 统一的 collection API
# ===================================================================

class _QdrantCollectionCompat:
    """QdrantService._collection 暴露的集合操作接口。

    提供 query/get/update/count 等集合级 API（翻译为底层 Qdrant 调用），
    供 pipeline.py / context.py / dispatch.py 统一使用。
    """

    def __init__(self, service: 'QdrantService'):
        self._svc = service

    def query(self, query_embeddings, n_results, where=None, include=None):
        """旧 ChromaDB col.query() → Qdrant search()（本地索引加速）。

        返回格式（兼容 ChromaDB 旧代码）:
          {"ids": [[id1,id2,...]], "documents": [["doc1","doc2",...]],
           "metadatas": [[{...},{...},...]], "distances": [[0.1,0.2,...]]}
        """
        qf = _translate_filter(where) if where else None
        include_docs = include is None or "documents" in include
        include_meta = include is None or "metadatas" in include

        ids_list, docs_list, metas_list, dists_list = [], [], [], []
        for q_emb in query_embeddings:
            try:
                results = self._svc._search_with_index(
                    query_vector=q_emb,
                    query_filter=qf,
                    limit=n_results,
                    with_payload=include_meta,
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
        """旧 ChromaDB col.get() → Qdrant retrieve()/scroll()。

        返回格式（兼容 ChromaDB 旧代码）:
          {"ids": [...], "documents": [...], "metadatas": [...]}
        """
        include_docs = include is None or "documents" in include
        include_meta = include is None or "metadatas" in include

        if ids:
            id_list = ids if isinstance(ids, list) else [ids]
            pts = self._svc._client.retrieve(
                collection_name=self._svc._collection_name,
                ids=id_list,
                with_payload=include_meta,
            )
        else:
            qf = _translate_filter(where) if where else None
            pts, _ = self._svc._scroll_with_index(
                scroll_filter=qf,
                with_payload=include_meta,
                limit=limit or 1000,
            )

        result_ids, docs, metas = [], [], []
        for pt in pts:
            result_ids.append(pt.id)
            payload = pt.payload or {}
            docs.append(payload.get("document", "") if include_docs else "")
            metas.append(dict(payload) if include_meta else {})

        return {
            "ids": result_ids,
            "documents": docs,
            "metadatas": metas,
        }

    def update(self, ids, metadatas=None, documents=None, embeddings=None):
        """旧 ChromaDB col.update() 格式 → Qdrant set_payload() + update_vectors()。"""
        id_list = [ids] if isinstance(ids, str) else ids
        for mid in id_list:
            payload_updates = {}
            if metadatas is not None:
                idx = id_list.index(mid) if isinstance(metadatas, list) else 0
                meta = metadatas[idx] if isinstance(metadatas, list) else metadatas
                if isinstance(meta, dict):
                    payload_updates.update(meta)
            if documents is not None:
                idx = id_list.index(mid) if isinstance(documents, list) else 0
                doc = documents[idx] if isinstance(documents, list) else documents
                payload_updates["document"] = doc
            if payload_updates:
                self._svc._client.set_payload(
                    collection_name=self._svc._collection_name,
                    payload=payload_updates,
                    points=[mid],
                )
            if embeddings is not None:
                idx = id_list.index(mid) if isinstance(embeddings, list) else 0
                emb = embeddings[idx] if isinstance(embeddings, list) else embeddings
                self._svc._client.update_vectors(
                    collection_name=self._svc._collection_name,
                    points=[models.PointVectors(id=mid, vector=emb)],
                )
        self._svc._invalidate_list_all_cache()

    def count(self):
        return self._svc._client.count(
            collection_name=self._svc._collection_name).count


class QdrantService:
    """Qdrant 记忆存储与检索 — 唯一向量存储后端。"""

    DESENSITIZATION_CHECK_INTERVAL = 50
    DESENSITIZATION_DECAY_DAYS = 7
    DESENSITIZATION_DECREMENT = 1

    LIST_ALL_CACHE_TTL = 5.0

    def __init__(self, persist_dir: str = None, collection_name: str = "memories",
                 url: str = None, api_key: str = None):
        self._collection_name = collection_name
        self._lock = threading.Lock()
        self._emb_cache_lock = threading.Lock()
        self._emb_cache: OrderedDict[str, list] = OrderedDict()
        self._total_hits = 0
        self._earliest_ts = None
        self._latest_ts = None
        self._desensitization_counter = 0

        # 后端选择优先级：
        #   1. 显式 url / 环境 QDRANT_URL（http(s):// → 服务器；:memory: → 内存）
        #   2. persist_dir → 本地嵌入式文件模式
        # Windows 路径含盘符冒号（C:\...），因此用 scheme 前缀判断，而非 ":" in url。
        target = url or QDRANT_URL or persist_dir
        api_key = api_key or QDRANT_API_KEY

        if target == ":memory:":
            self._client = QdrantClient(location=":memory:")
        elif target and (target.startswith("http://") or target.startswith("https://")):
            self._client = QdrantClient(url=target, api_key=api_key) if api_key else \
                          QdrantClient(url=target)
        else:
            # 本地嵌入式文件模式
            local_path = target or persist_dir
            os.makedirs(local_path, exist_ok=True)
            self._client = QdrantClient(path=local_path)

        self._ensure_collection()
        self._collection = _QdrantCollectionCompat(self)
        self._local_index: _LocalPayloadIndex | None = None
        if _is_local_client(self._client):
            self._local_index = _LocalPayloadIndex()

        self._list_all_cache_lock = threading.Lock()
        self._list_all_cache = None
        self._list_all_cache_time = 0.0
        self._list_all_cache_ttl = self.LIST_ALL_CACHE_TTL

        self._emb_cache_build_done = False

    def _ensure_collection(self):
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            quant_cfg = _build_quantization_config()
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    on_disk=QDRANT_ON_DISK,
                    hnsw_config=models.HnswConfigDiff(
                        m=QDRANT_HNSW_M,
                        ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                    ),
                ),
                quantization_config=quant_cfg,
            )
            logger.info("创建 Qdrant collection: %s (quantization=%s)",
                       self._collection_name, QDRANT_QUANTIZATION or "none")
            # 本地嵌入式模式下服务端 payload 索引无效（由 _LocalPayloadIndex 补偿），跳过避免告警
            if _is_local_client(self._client):
                return
            for field, stype in [
                ("timestamp", models.PayloadSchemaType.FLOAT),
                ("hit_count", models.PayloadSchemaType.INTEGER),
                ("tags", models.PayloadSchemaType.KEYWORD),
                ("stale", models.PayloadSchemaType.BOOL),
                ("archived", models.PayloadSchemaType.BOOL),
                ("emotion_valence", models.PayloadSchemaType.FLOAT),
                ("embed_model", models.PayloadSchemaType.KEYWORD),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field,
                        field_schema=stype,
                    )
                except Exception:
                    pass

    def _search_with_index(self, query_vector, query_filter, limit, with_payload):
        """Phase 4: 优先本地索引，失败回退 Qdrant API。

        qdrant-client 1.10+ 用 query_points 替代 search；返回 .points（ScoredPoint 列表）。
        """
        if self._local_index is not None:
            prefilter = self._local_index.resolve(query_filter)
            if prefilter is not None and not prefilter:
                return []
            if prefilter is not None:
                return self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=min(limit, len(prefilter)),
                    with_payload=with_payload,
                ).points
        return self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        ).points

    def _scroll_with_index(self, scroll_filter, with_payload, limit, offset=None):
        """Phase 4: 优先本地索引，失败回退 Qdrant API。"""
        if self._local_index is not None:
            prefilter = self._local_index.resolve(scroll_filter)
            if prefilter is not None and not prefilter:
                return [], None
            if prefilter is not None:
                return self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=scroll_filter,
                    with_payload=with_payload,
                    limit=min(limit, len(prefilter)),
                    offset=offset,
                )
        return self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add(self, document: str, metadata: dict = None, embedding: list = None,
            id: str = None) -> str:
        """添加记忆。"""
        mid = id or str(uuid.uuid4())
        payload = dict(metadata or {})
        payload["document"] = document
        payload.setdefault("timestamp", time.time())
        payload.setdefault("hit_count", 0)
        payload.setdefault("embed_model", DEFAULT_EMBED_MODEL)

        vec = embedding if embedding is not None else [0.0] * 1024

        with self._lock:
            self._client.upsert(
                collection_name=self._collection_name,
                points=[models.PointStruct(id=mid, vector=vec, payload=payload)],
            )
            if self._local_index is not None:
                self._local_index.add(mid, payload)
        self._emb_cache_put(mid, vec)
        self._invalidate_list_all_cache()
        return mid

    def add_memory(self, user_message: str, ai_message: str, summary: str,
                   tags=None, embedding: list = None, *,
                   entities=None, date_tag: str = None, time_features: dict = None,
                   source: str = "user", model_id: str = DEFAULT_EMBED_MODEL,
                   metadata: dict = None, id: str = None, **extra) -> str:
        """添加完整记忆（红线 2 metadata schema）。

        tags 接受 list 或逗号字符串；entities 接受 list[dict] 或 JSON 字符串。
        time_features / extra 中的键平铺进 payload。
        """
        mid = id or str(uuid.uuid4())

        if isinstance(tags, (list, tuple)):
            tags_str = ",".join(str(t) for t in tags)
        else:
            tags_str = tags or ""

        if entities is None:
            entities_str = ""
        elif isinstance(entities, str):
            entities_str = entities
        else:
            entities_str = json.dumps(entities, ensure_ascii=False)

        payload = dict(metadata or {})
        payload.update({
            "user_message": user_message,
            "ai_message": ai_message,
            "summary": summary,
            "tags": tags_str,
            "entities": entities_str,
            "document": f"[{tags_str}] {summary}" if tags_str else summary,
            "timestamp": time.time(),
            "hit_count": 0,
            "heat": 0,
            "embed_model": model_id,
            "stale": False,
            "archived": False,
            "superseded_by": "",
            "storage_complete": True,
            "source": source,
            "date_tag": date_tag or "",
        })
        if isinstance(time_features, dict):
            for k, v in time_features.items():
                payload.setdefault(k, v)
        for k, v in extra.items():
            payload.setdefault(k, v)

        vec = embedding if embedding is not None else [0.0] * 1024

        with self._lock:
            self._client.upsert(
                collection_name=self._collection_name,
                points=[models.PointStruct(id=mid, vector=vec, payload=payload)],
            )
            if self._local_index is not None:
                self._local_index.add(mid, payload)
        self._emb_cache_put(mid, vec)
        self._invalidate_list_all_cache()
        return mid

    def count(self) -> int:
        """记忆总数。"""
        return self._collection.count()

    def clear_all(self):
        """清空所有记忆（别名 clear）。"""
        self.clear()

    def update_entity_co_counts(self, memory_id: str, entities) -> None:
        """入库时预计算实体共现对，存入该记录的 payload.entity_co_counts。

        无外部读取方（CoOccurrenceStore 才是权威共现源），失败可忽略。
        """
        try:
            texts = []
            for ent in (entities or []):
                if isinstance(ent, dict):
                    t = ent.get("text", "")
                else:
                    t = str(ent)
                if t:
                    texts.append(t)
            texts = sorted(set(texts))
            pairs = {}
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    pairs[f"{texts[i]}|{texts[j]}"] = 1
            if not pairs:
                return
            payload = {"entity_co_counts": json.dumps(pairs, ensure_ascii=False)}
            with self._lock:
                self._client.set_payload(
                    collection_name=self._collection_name,
                    payload=payload,
                    points=[memory_id],
                )
        except Exception:
            pass

    def _local_index_build(self):
        """从全量 points 重建本地 payload 索引（启动预热用）。"""
        if self._local_index is None:
            return
        try:
            points = []
            offset = None
            while True:
                batch, offset = self._client.scroll(
                    collection_name=self._collection_name,
                    with_payload=True,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                points.extend(batch)
                if offset is None:
                    break
            with self._lock:
                self._local_index.build(points)
        except Exception as exc:
            logger.warning("本地 payload 索引构建失败: %s", exc)

    def update_memory(self, memory_id: str, **kwargs):
        """更新记忆的指定字段。"""
        payload = {}
        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        if not payload:
            return

        with self._lock:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload=payload,
                points=[memory_id],
            )
            if self._local_index is not None:
                self._local_index.update(memory_id, payload)
        self._invalidate_list_all_cache()

    def update(self, memory_id: str, document: str = None, metadata: dict = None,
               embedding: list = None):
        """通用更新接口。"""
        if document:
            self.update_memory(memory_id, document=document)
        if metadata:
            self.update_memory(memory_id, **metadata)
        if embedding:
            self._client.update_vectors(
                collection_name=self._collection_name,
                points=[models.PointVectors(id=memory_id, vector=embedding)],
            )
            self._emb_cache_put(memory_id, embedding)

    def increment_hit_count(self, memory_id: str):
        """递增命中计数 + 更新 last_hit_time。"""
        now = time.time()
        with self._lock:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={
                    "hit_count": models.PayloadSelectorInclude(include=["hit_count"]),
                    "last_hit_time": now,
                },
                points=[memory_id],
            )
        try:
            cur = (self._client.retrieve(
                collection_name=self._collection_name,
                ids=[memory_id],
                with_payload=["hit_count"],
            ) or [None])[0]
            if cur and cur.payload:
                new_count = cur.payload.get("hit_count", 0) + 1
                self._client.set_payload(
                    collection_name=self._collection_name,
                    payload={"hit_count": new_count, "last_hit_time": now},
                    points=[memory_id],
                )
        except Exception:
            pass

    def batch_increment_hit_count(self, memory_ids: list[str]):
        """批量递增命中计数。"""
        now = time.time()
        for mid in memory_ids:
            try:
                cur = (self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=[mid],
                    with_payload=["hit_count"],
                ) or [None])[0]
                if cur and cur.payload:
                    new_count = cur.payload.get("hit_count", 0) + 1
                    self._client.set_payload(
                        collection_name=self._collection_name,
                        payload={"hit_count": new_count, "last_hit_time": now},
                        points=[mid],
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def resolve(self, chroma_where: dict) -> list[dict]:
        """旧 ChromaDB where 条件 → Qdrant scroll。"""
        qf = _translate_filter(chroma_where)
        pts, _ = self._scroll_with_index(
            scroll_filter=qf,
            with_payload=True,
            limit=5000,
        )
        return _build_items_from_points(pts)

    def query(self, query_embedding, n_results: int = 10, where=None) -> list[dict]:
        """向量检索。"""
        qf = _translate_filter(where) if where else None
        results = self._search_with_index(
            query_vector=query_embedding,
            query_filter=qf,
            limit=n_results,
            with_payload=True,
        )
        items = []
        for pt in results:
            payload = pt.payload or {}
            items.append({
                "id": pt.id,
                "document": payload.get("document", ""),
                "metadata": dict(payload),
                "distance": 1.0 - pt.score,
            })
        return items

    def get_related(self, memory_id: str, _data=None) -> list[tuple[str, int]]:
        """占位：关联检索由 CoOccurrenceStore 处理。"""
        return []

    def get_memory_ids(self, tag: str = "", limit: int = 100) -> list[str]:
        """按 tag 获取记忆 ID 列表。"""
        if not tag:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=False,
                limit=limit,
            )
        else:
            pts, _ = self._scroll_with_index(
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="tags", match=models.MatchValue(value=tag),
                    ),
                ]),
                with_payload=False,
                limit=limit,
            )
        return [pt.id for pt in pts]

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def remove(self, memory_id: str):
        """删除记忆。"""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[memory_id],
        )
        self._emb_cache.pop(memory_id, None)
        if self._local_index is not None:
            self._local_index.remove(memory_id)
        self._invalidate_list_all_cache()

    def remove_memory(self, memory_id: str):
        """删除记忆（别名）。"""
        self.remove(memory_id)

    def clear(self):
        """清空所有记忆。

        本地嵌入式 Qdrant 下 delete_collection+重建不会真正清空持久化分段，
        因此按 ID 批量删除，更可靠。
        """
        with self._lock:
            try:
                all_ids = []
                offset = None
                while True:
                    batch, offset = self._client.scroll(
                        collection_name=self._collection_name,
                        with_payload=False, with_vectors=False,
                        limit=1000, offset=offset,
                    )
                    all_ids.extend([p.id for p in batch])
                    if offset is None:
                        break
                if all_ids:
                    self._client.delete(
                        collection_name=self._collection_name,
                        points_selector=models.PointIdsList(points=all_ids),
                    )
            except Exception:
                # 回退：删除并重建 collection
                try:
                    self._client.delete_collection(self._collection_name)
                except Exception:
                    pass
                self._ensure_collection()
        with self._emb_cache_lock:
            self._emb_cache = OrderedDict()
        self._emb_cache_build_done = False
        if self._local_index is not None:
            self._local_index = _LocalPayloadIndex()
        self._invalidate_list_all_cache()

    def mark_storage_complete(self, memory_id: str = None):
        """标记某条记录入库完成（storage_complete=True，红线 2）。"""
        self._emb_cache_build_done = True
        if not memory_id:
            return
        try:
            with self._lock:
                self._client.set_payload(
                    collection_name=self._collection_name,
                    payload={"storage_complete": True},
                    points=[memory_id],
                )
                if self._local_index is not None:
                    self._local_index.update(memory_id, {"storage_complete": True})
        except Exception:
            pass

    def _apply_emotional_desensitization(self):
        """情绪淡化：周期性衰减情绪强度。"""
        with self._lock:
            self._desensitization_counter += 1
            if self._desensitization_counter < self.DESENSITIZATION_CHECK_INTERVAL:
                return
            self._desensitization_counter = 0

        cutoff = time.time() - self.DESENSITIZATION_DECAY_DAYS * 86400
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="timestamp", range=models.Range(lt=cutoff),
                    ),
                ]),
                with_payload=["emotion_valence"],
                limit=5000,
            )
        except Exception:
            return

        for pt in pts:
            p = pt.payload or {}
            ev = p.get("emotion_valence")
            if ev is None:
                continue
            try:
                ev = float(ev)
            except (ValueError, TypeError):
                continue
            new_ev = max(0.0, ev - self.DESENSITIZATION_DECREMENT * 0.1)
            try:
                self._client.set_payload(
                    collection_name=self._collection_name,
                    payload={"emotion_valence": new_ev},
                    points=[pt.id],
                )
            except Exception:
                pass

    def _prune(self, max_memories: int = 100000):
        """超限裁剪。"""
        try:
            total = self._client.count(collection_name=self._collection_name).count
        except Exception:
            return
        if total <= max_memories:
            return
        to_remove = total - max_memories
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=["timestamp"],
                order_by=models.OrderBy(
                    key="timestamp", direction=models.Direction.ASC,
                ),
                limit=to_remove,
            )
            if pts:
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=[pt.id for pt in pts],
                )
        except Exception:
            pass

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
        if self._local_index is not None:
            self._local_index.update(old_id, {
                "stale": True, "superseded_by": new_id,
                "supersede_reason": reason, "superseded_at": _dt.now().isoformat(),
            })
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
        pts, _ = self._scroll_with_index(
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
        if self._local_index is not None:
            self._local_index.remove(memory_id)
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
            pts, _ = self._scroll_with_index(
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
                limit=limit * 5,
            )
        except Exception:
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
            pts, _ = self._scroll_with_index(
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
    # Embedding 缓存（attention 位移因子用）— Phase 4: LRU 20K
    # ------------------------------------------------------------------

    def _build_embedding_cache(self):
        """按 last_hit_time DESC 分批 scroll 最近 N 条记忆的 embedding 到 LRU 缓存。"""
        loaded = 0
        try:
            offset = None
            while loaded < QDRANT_EMB_CACHE_MAX:
                pts, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    with_payload=["last_hit_time"],
                    with_vectors=True,
                    limit=min(QDRANT_EMB_CACHE_BATCH, QDRANT_EMB_CACHE_MAX - loaded),
                    offset=offset,
                )
                if not pts:
                    break
                with self._emb_cache_lock:
                    for pt in pts:
                        if pt.vector is not None:
                            self._emb_cache[pt.id] = pt.vector
                            self._emb_cache.move_to_end(pt.id)
                loaded += len(pts)
                if next_offset is None or len(pts) < QDRANT_EMB_CACHE_BATCH:
                    break
                offset = next_offset
            logger.info("embedding 缓存构建完成: %d 条 (max=%d)", len(self._emb_cache), QDRANT_EMB_CACHE_MAX)
        except Exception as exc:
            with self._emb_cache_lock:
                self._emb_cache = OrderedDict()
            logger.warning("embedding 缓存构建失败，回退空缓存: %s", exc)

    def _get_embedding_cached(self, memory_id: str) -> list | None:
        """获取缓存的 embedding，LRU 更新访问时间。"""
        with self._emb_cache_lock:
            emb = self._emb_cache.get(memory_id)
            if emb is not None:
                self._emb_cache.move_to_end(memory_id)
            return emb

    def _emb_cache_put(self, memory_id: str, embedding: list):
        """写入 embedding 缓存（LRU 淘汰）。"""
        with self._emb_cache_lock:
            if memory_id in self._emb_cache:
                self._emb_cache.move_to_end(memory_id)
                self._emb_cache[memory_id] = embedding
                return
            while len(self._emb_cache) >= QDRANT_EMB_CACHE_MAX:
                self._emb_cache.popitem(last=False)
            self._emb_cache[memory_id] = embedding

    # ------------------------------------------------------------------
    # 旧 ChromaDB Collection API 兼容层 (Phase 2: delegate to _collection adapter)
    # ------------------------------------------------------------------

    def get(self, ids=None, where=None, include=None, limit=None):
        """旧 ChromaDB col.get() 兼容 — delegate to _collection adapter。"""
        return self._collection.get(ids=ids, where=where, include=include, limit=limit)

    def query(self, query_embeddings, n_results, where=None, include=None):
        """旧 ChromaDB col.query() 兼容 — delegate to _collection adapter。"""
        return self._collection.query(
            query_embeddings=query_embeddings, n_results=n_results,
            where=where, include=include,
        )

    @property
    def client(self) -> QdrantClient:
        """暴露 Qdrant 客户端，供 CoOccurrenceStore/HyperEdgeStore 复用。"""
        return self._client

    def close(self):
        """释放 Qdrant 客户端资源。"""
        with self._lock:
            self._client.close() if hasattr(self._client, 'close') else None
        with self._emb_cache_lock:
            self._emb_cache = OrderedDict()

def _is_default_local_url() -> bool:
    """判断 QDRANT_URL 是否是默认本地地址（未显式配置远程服务器）。"""
    return (QDRANT_URL in ("http://localhost:6333", "http://127.0.0.1:6333")
            and os.getenv("QDRANT_URL") is None)

# CoOccurrenceStore 和 HyperEdgeStore 已拆分至独立文件：
#   from app.memory.qdrant_cooccur import CoOccurrenceStore
#   from app.memory.qdrant_hyperedge import HyperEdgeStore
