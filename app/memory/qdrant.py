"""记忆层 — Qdrant 存储/检索 + 上下文包裹管理 (替代 chroma.py).

Phase 4: 百万级硬骨头 — 量化 + payload 索引 + embedding 缓存 LRU。
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
        # 先移除这些字段的旧值
        for field in partial_payload:
            self._remove_field(pid, field)
        # 再索引新值
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
        # keyword 索引
        if field in self._kw:
            for val, pids in list(self._kw[field].items()):
                pids.discard(pid)
                if not pids:
                    del self._kw[field][val]
        # 数值索引
        if field in self._num:
            self._num[field] = [(v, p) for v, p in self._num[field] if p != pid]
            self._num_dirty.add(field)

    def _sort_all_dirty(self):
        """重排所有脏的数值索引。"""
        for field in list(self._num_dirty):
            self._num[field].sort(key=lambda x: x[0])
        self._num_dirty.clear()

    # ── 查询 ──

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
        """递归解析 Filter 对象。"""
        result: set[str] | None = None

        # must → 交集
        if hasattr(f, 'must') and f.must:
            ids: set[str] | None = None
            for cond in f.must:
                sub = self._resolve_condition(cond)
                if sub is None:
                    return None  # 无法解析的条件 → 回退
                if ids is None:
                    ids = set(sub)
                else:
                    ids &= sub
                if not ids:
                    return set()
            result = ids if ids is not None else set(self._all_ids)

        # should → 并集
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

        # must_not → 差集
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
        """解析单个条件对象。"""
        # FieldCondition
        if hasattr(cond, 'key') and hasattr(cond, 'match'):
            return self._resolve_match(cond.key, cond.match)
        if hasattr(cond, 'key') and hasattr(cond, 'range'):
            return self._resolve_range(cond.key, cond.range)
        # 嵌套 Filter
        if hasattr(cond, 'must') or hasattr(cond, 'should') or hasattr(cond, 'must_not'):
            return self._resolve_filter(cond)
        return None

    def _resolve_match(self, key: str, match) -> set[str] | None:
        """解析 MatchValue / MatchAny / MatchExcept / MatchText。"""
        if hasattr(match, 'value'):
            # MatchValue
            val = match.value
            lookup = str(val).lower() if isinstance(val, bool) else val
            kw_idx = self._kw.get(key, {})
            return kw_idx.get(lookup, set())
        elif hasattr(match, 'any'):
            # MatchAny
            kw_idx = self._kw.get(key, {})
            result: set[str] = set()
            for v in match.any:
                result |= kw_idx.get(v, set())
            return result
        elif hasattr(match, 'except_'):
            # MatchExcept — 暂不支持，回退
            return None
        elif hasattr(match, 'text'):
            # MatchText — 文本分词匹配，本地索引不支持
            return None
        return None

    def _resolve_range(self, key: str, r) -> set[str] | None:
        """解析 Range (gte/lte/gt/lt) 查询。"""
        store = self._num.get(key)
        if not store:
            return set()
        lo = r.gt if hasattr(r, 'gt') and r.gt is not None else \
             (r.gte if hasattr(r, 'gte') and r.gte is not None else None)
        hi = r.lt if hasattr(r, 'lt') and r.lt is not None else \
             (r.lte if hasattr(r, 'lte') and r.lte is not None else None)
        if lo is None and hi is None:
            return None

        # 二分查找边界
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
        """ChromaDB col.query() → Qdrant search() (本地索引加速)。

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
                # Phase 4: 本地索引加速
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
            # 按 filter 检索 — Phase 4: 本地索引加速
            qf = _translate_filter(where) if where else None
            _limit = limit or 50000
            try:
                pts, _ = self._svc._scroll_with_index(
                    scroll_filter=qf,
                    limit=_limit,
                    with_payload=True,
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
        # Phase 4: Embedding 缓存 — OrderedDict LRU, 上限 20K
        self._emb_cache: OrderedDict[str, list] = OrderedDict()
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
            # Phase 4: 量化配置 — scalar_int8 将 4GB→1GB
            quant_cfg = _build_quantization_config()
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
                quantization_config=quant_cfg,
            )
            logger.info("创建 Qdrant collection: %s (quantization=%s)", coll_name, QDRANT_QUANTIZATION or "none")
            # Phase 4: 创建 payload 索引 (idempotent, 幂等)
            self._create_payload_indexes(coll_name)

        self._collection_name = coll_name
        # Phase 2: ChromaDB → Qdrant API 兼容适配器
        self._collection = _QdrantCollectionCompat(self)

        # Phase 4: 本地模式补偿 — Python 侧内存 payload 索引
        self._local_index: _LocalPayloadIndex | None = None
        self._local_index_ready = False
        if _is_local_client(self._client):
            self._local_index = _LocalPayloadIndex()
            logger.info("Qdrant 本地模式已启用 — Python 侧 payload 索引补偿")

    # ------------------------------------------------------------------
    # 本地索引辅助 (Phase 4)
    # ------------------------------------------------------------------

    def _local_index_build(self):
        """构建本地 payload 索引（启动后异步调用）。"""
        if self._local_index is None:
            return
        try:
            all_pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=True,
                with_vectors=False,
                limit=100000,
            )
            self._local_index.build(all_pts)
            self._local_index_ready = True
            logger.info("本地 payload 索引构建完成: %s", self._local_index.stats())
        except Exception as exc:
            logger.warning("本地索引构建失败: %s", exc)

    def _scroll_with_index(self, scroll_filter=None, limit: int = 100,
                           with_payload: bool = True, with_vectors: bool = False,
                           order_by=None, offset=None) -> tuple[list, str | None]:
        """scroll() 的索引加速版。

        当本地索引就绪且 filter 可解析时，先走索引缩小候选集。
        否则回退到普通 scroll。
        """
        # 检查是否能用索引
        use_index = (
            self._local_index is not None
            and self._local_index_ready
            and scroll_filter is not None
            and order_by is None  # 排序时不能用索引预筛选（影响全局排序）
            and offset is None    # 偏移时不能用索引（需要全量结果计算 offset）
        )

        if use_index:
            matching = self._local_index.resolve(scroll_filter)
            if matching is not None:
                # 索引命中 → 只 retrieve 匹配的点
                if not matching:
                    return [], None  # 无匹配
                # 取 limit 个 ID，直接 retrieve
                target_ids = list(matching)[:limit]
                try:
                    pts = self._client.retrieve(
                        collection_name=self._collection_name,
                        ids=target_ids,
                        with_payload=with_payload,
                        with_vectors=with_vectors,
                    )
                    return pts, None  # next_offset = None，索引模式不支持分页
                except Exception:
                    pass  # 回退到普通 scroll

        # 回退：普通 scroll
        try:
            return self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                order_by=order_by,
                offset=offset,
            )
        except Exception:
            return [], None

    def _search_with_index(self, query_vector, query_filter=None, limit: int = 50,
                           with_payload: bool = True, with_vectors: bool = False):
        """search() 的索引加速版。

        先用本地索引缩小候选集，再在候选集中做向量搜索。
        如果索引不适用，回退到普通 search。
        """
        # 检查是否能用索引
        use_index = (
            self._local_index is not None
            and self._local_index_ready
            and query_filter is not None
        )

        if use_index:
            matching = self._local_index.resolve(query_filter)
            if matching is not None and len(matching) < 5000:
                # 候选集不太大 → 用索引缩小范围
                if not matching:
                    return []
                # Qdrant search 不支持 ID 列表预选，策略是：
                # 1. 先用 search 正常搜（limit 放宽一点补偿精度损失）
                # 2. 再用索引过滤掉不匹配的结果
                try:
                    results = self._client.search(
                        collection_name=self._collection_name,
                        query_vector=query_vector,
                        limit=min(limit * 3, 200),
                        with_payload=with_payload,
                        with_vectors=with_vectors,
                    )
                    # 用索引过滤结果
                    filtered = [r for r in results if r.id in matching][:limit]
                    return filtered
                except Exception:
                    pass
            # 候选集太大或无法解析 → 回退

        # 回退：普通 search
        try:
            return self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Payload 索引 (Phase 4 — §4.5 清单)
    # ------------------------------------------------------------------

    def _create_payload_indexes(self, coll_name: str):
        """为 collection 创建所有高频过滤字段的 payload 索引。幂等——重复创建不报错。"""
        # §4.5 必须建索引的字段
        _indexes = [
            # keyword / enum 类型
            ("heat", models.PayloadSchemaType.KEYWORD),
            ("emotion_valence_bin", models.PayloadSchemaType.KEYWORD),
            ("date_tag", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
            # float (range queries)
            ("timestamp", models.PayloadSchemaType.FLOAT),
            ("last_hit_time", models.PayloadSchemaType.FLOAT),
            # integer
            ("emotional_intensity", models.PayloadSchemaType.INTEGER),
            # bool
            ("stale", models.PayloadSchemaType.BOOL),
            ("archived", models.PayloadSchemaType.BOOL),
            # text (MatchText on document)
            ("document", models.PayloadSchemaType.TEXT),
            # 按需: 嵌套 entity keyword
            ("entities[].text", models.PayloadSchemaType.KEYWORD),
            ("entities[].type", models.PayloadSchemaType.KEYWORD),
            # 按需: year_month
            ("year_month", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in _indexes:
            try:
                self._client.create_payload_index(
                    collection_name=coll_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception as exc:
                # 索引已存在或字段类型不兼容——跳过
                logger.debug("payload 索引 %s/%s: %s", coll_name, field_name, exc)
        logger.info("payload 索引创建完成: %s (%d 字段)", coll_name, len(_indexes))

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
        # 确定性 ID：基于对话内容哈希，避免队列积压恢复时重复入库
        _id_seed = f"{user_message}||{ai_message}||{timestamp if timestamp else ''}"
        memory_id = str(uuid.uuid5(uuid.NAMESPACE_OID, _id_seed))
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
        # Phase 4: 维护本地索引 (持锁，防止并发 query 看到不一致的索引)
        with self._lock:
            if self._local_index is not None:
                self._local_index.add(memory_id, payload)
        if self._earliest_ts is None or timestamp < self._earliest_ts:
            self._earliest_ts = timestamp
        if self._latest_ts is None or timestamp > self._latest_ts:
            self._latest_ts = timestamp
        self._invalidate_list_all_cache()
        return memory_id

    def update_entity_co_counts(self, memory_id: str, entities: list[dict]):
        """入库后更新该条记忆涉及实体的全局共现计数（替代 SQLite entity_pair）。

        策略: 从 payload entity_co_counts 字段做增量写——
        从 Qdrant 查询匹配的记忆，聚合其已有 entity_co_counts，写入当前记忆。
        """
        entity_names = [e.get("text", "") for e in entities if e.get("text")]
        if not entity_names:
            return

        # 从 Qdrant 查询这些 entity 的当前计数
        try:
            pts, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(should=[
                    models.FieldCondition(
                        key="entities[].text", match=models.MatchValue(value=en),
                    ) for en in entity_names
                ]),
                with_payload=["entity_co_counts"],
                limit=50,
            )
        except Exception:
            # scroll with nested field filter 可能不支持，回退到全量采样
            try:
                pts, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    with_payload=["entity_co_counts"],
                    limit=100,
                )
            except Exception:
                return

        # 聚合
        aggregated: dict[str, int] = {}
        for pt in pts:
            prev = (pt.payload or {}).get("entity_co_counts", {})
            if isinstance(prev, dict):
                for en, cnt in prev.items():
                    aggregated[en] = aggregated.get(en, 0) + cnt

        # 写入当前记忆的 payload
        try:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={"entity_co_counts": aggregated},
                points=[memory_id],
            )
        except Exception as exc:
            logger.debug("entity_co_counts 更新失败 id=%s: %s", memory_id[:8], exc)

    def get_entity_co_counts(self, memory_ids: list[str]) -> dict[str, int]:
        """从 payload 直接读 entity_co_counts。"""
        try:
            pts = self._client.retrieve(
                collection_name=self._collection_name,
                ids=memory_ids,
                with_payload=["entity_co_counts"],
            )
        except Exception:
            return {}
        merged: dict[str, int] = {}
        for pt in pts:
            co = (pt.payload or {}).get("entity_co_counts", {})
            if isinstance(co, dict):
                for en, cnt in co.items():
                    merged[en] = merged.get(en, 0) + cnt
        return dict(sorted(merged.items(), key=lambda x: -x[1]))

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
            # Phase 4: 维护本地索引
            if self._local_index is not None:
                self._local_index.add(memory_id, payload)

        should_desensitize = False
        with self._lock:
            self._desensitization_counter += 1
            if self._desensitization_counter >= self.DESENSITIZATION_CHECK_INTERVAL:
                self._desensitization_counter = 0
                should_desensitize = True
        if should_desensitize:
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
            # Phase 4: 维护本地索引（批量）
            if self._local_index is not None:
                for mid, total_delta in aggregated.items():
                    payload = payload_map.get(mid)
                    if payload is not None:
                        self._local_index.add(mid, payload)

        should_desensitize = False
        with self._lock:
            self._desensitization_counter += len(ids_and_deltas)
            if self._desensitization_counter >= self.DESENSITIZATION_CHECK_INTERVAL:
                self._desensitization_counter = 0
                should_desensitize = True
        if should_desensitize:
            self._apply_emotional_desensitization()

    # ------------------------------------------------------------------
    # 情绪淡化
    # ------------------------------------------------------------------

    def _apply_emotional_desensitization(self):
        """情绪淡化：扫描 emotional_intensity>=1 的记忆，超期未命中则减 1。"""
        try:
            pts, _ = self._scroll_with_index(
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
        # Phase 4: 维护本地索引（部分更新）
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
        # Phase 4: 维护本地索引
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
        """按 last_hit_time DESC 分批 scroll 最近 N 条记忆的 embedding 到 LRU 缓存。

        Phase 4: 上限 20K, 按 last_hit_time 降序（自然偏向活跃记忆）,
        分批 scroll 避免单次大量传输, 启动时不阻塞（由 storage_executor.submit 调用）。
        """
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
                            # LRU: 新加载的放到末尾
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
            # LRU 淘汰: 超过上限时移除最旧条目
            while len(self._emb_cache) >= QDRANT_EMB_CACHE_MAX:
                self._emb_cache.popitem(last=False)
            self._emb_cache[memory_id] = embedding

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


# ===================================================================
# CoOccurrenceStore — 独立 Qdrant collection 替代 SQLite cooccur.py
# ===================================================================

class CoOccurrenceStore:
    """记忆对共现次数，独立 Qdrant collection。

    提供与 CoOccurrenceTracker 完全相同的公开 API，
    底层从 SQLite 迁移到 Qdrant。

    每条 point = 一对记忆的共现计数。
    """

    EXTEND_TOP_K = 3
    CO_WITH_LIMIT = 10
    LTD_CHECK_INTERVAL = 20
    LTD_DECAY_DAYS = 7
    LTD_DECREMENT = 1

    def __init__(self, client: QdrantClient, collection_name: str,
                 embed_getter=None):
        self._client = client
        self._coll = collection_name
        self._embed_getter = embed_getter  # callable(memory_id) → list[float] | None
        self._ltd_lock = threading.Lock()
        self._ltd_counter = 0
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self._client.get_collections().collections}
        if self._coll not in existing:
            quant_cfg = _build_quantization_config()
            self._client.create_collection(
                collection_name=self._coll,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    on_disk=QDRANT_ON_DISK,
                ),
                quantization_config=quant_cfg,
            )
            logger.info("创建 Qdrant collection: %s (co_occurrence, quantization=%s)",
                       self._coll, QDRANT_QUANTIZATION or "none")
            # Phase 4: payload 索引 — id_a, id_b keyword, count integer
            for field, stype in [
                ("id_a", models.PayloadSchemaType.KEYWORD),
                ("id_b", models.PayloadSchemaType.KEYWORD),
                ("count", models.PayloadSchemaType.INTEGER),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self._coll,
                        field_name=field,
                        field_schema=stype,
                    )
                except Exception:
                    pass

    # ── 向后兼容方法 ──

    def _invalidate_cache(self):
        """向后兼容：Qdrant 无内存缓存，no-op。"""
        pass

    def _load(self) -> dict:
        """向后兼容：返回旧格式 dict。实际不使用。"""
        return {}

    # ── 写入 ──

    def record(self, memory_ids: list[str]):
        """同轮出现的记忆对，count += 1。合并器锁保护。"""
        if len(memory_ids) < 2:
            return

        now = time.time()
        # 生成所有 pair
        pairs: list[tuple[str, str]] = []
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                a, b = sorted([memory_ids[i], memory_ids[j]])
                pairs.append((a, b))

        if not pairs:
            return

        # 用 UUID v5 生成确定性 point ID
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, f"{a}||{b}"))
                     for a, b in pairs]

        # 批量检索现有点
        existing_map: dict[str, int] = {}
        try:
            existing = self._client.retrieve(
                collection_name=self._coll,
                ids=point_ids,
                with_payload=["count"],
            )
            for pt in existing:
                existing_map[pt.id] = (pt.payload or {}).get("count", 0)
        except Exception as exc:
            logger.debug("CoOccurrence retrieve 失败: %s", exc)

        # 批量 upsert
        points = []
        for (a, b), pid in zip(pairs, point_ids):
            new_count = existing_map.get(pid, 0) + 1
            emb = None
            if self._embed_getter:
                emb = self._embed_getter(a)  # id_a 的 embedding
            if emb is None:
                emb = [0.0] * 1024
            points.append(models.PointStruct(
                id=pid,
                vector=emb,
                payload={"id_a": a, "id_b": b, "count": new_count, "last_time": now},
            ))

        try:
            self._client.upsert(collection_name=self._coll, points=points)
        except Exception as exc:
            logger.warning("CoOccurrence upsert 失败: %s", exc)

        self._maybe_cleanup()

    def _maybe_cleanup(self):
        """超限裁剪：按 count 升序删最旧的。"""
        try:
            total = self._client.count(collection_name=self._coll).count
        except Exception:
            return
        from app.config.settings import CO_OCCURRENCE_MAX_PAIRS, CO_OCCURRENCE_CLEANUP_RATIO
        if total < CO_OCCURRENCE_MAX_PAIRS:
            return
        to_remove = max(1, int(total * CO_OCCURRENCE_CLEANUP_RATIO))
        try:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                with_payload=["count"],
                order_by=models.OrderBy(key="count", direction=models.Direction.ASC),
                limit=to_remove,
            )
            if pts:
                self._client.delete(
                    collection_name=self._coll,
                    points_selector=[pt.id for pt in pts],
                )
        except Exception as exc:
            logger.debug("CoOccurrence cleanup 失败: %s", exc)

    # ── 查询 ──

    def get_related(self, memory_id: str, data=None) -> list[tuple[str, int]]:
        """返回与 memory_id 共现频率最高的 TOP_K 个 partner。"""
        pts = self._scroll_by_memory(memory_id, self.EXTEND_TOP_K * 3)
        pairs = []
        for pt in pts:
            p = pt.payload or {}
            partner = p["id_b"] if p["id_a"] == memory_id else p["id_a"]
            pairs.append((partner, p.get("count", 0)))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:self.EXTEND_TOP_K]

    def get_co_counts(self, memory_ids: list[str]) -> dict[str, int]:
        """返回每个 memory_id 的共现度数。"""
        counts = {mid: 0 for mid in memory_ids}
        if not memory_ids:
            return counts
        for mid in memory_ids:
            pts = self._scroll_by_memory(mid, 500)
            counts[mid] = len(pts)
        return counts

    def get_co_count(self, memory_id: str) -> int:
        pts = self._scroll_by_memory(memory_id, 500)
        return len(pts)

    def get_co_with(self, memory_id: str) -> list[dict]:
        """返回 {id, count} 列表，最多 CO_WITH_LIMIT 条。"""
        pts = self._scroll_by_memory(memory_id, self.CO_WITH_LIMIT * 2)
        pairs = []
        for pt in pts:
            p = pt.payload or {}
            partner = p["id_b"] if p["id_a"] == memory_id else p["id_a"]
            pairs.append({"id": partner, "count": p.get("count", 0)})
        pairs.sort(key=lambda x: -x["count"])
        return pairs[:self.CO_WITH_LIMIT]

    def _scroll_by_memory(self, memory_id: str, limit: int) -> list:
        """按 id_a 或 id_b 检索。Qdrant 不支持 OR，分两次 scroll。"""
        all_pts = []
        for field in ["id_a", "id_b"]:
            try:
                pts, _ = self._client.scroll(
                    collection_name=self._coll,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key=field, match=models.MatchValue(value=memory_id),
                        ),
                    ]),
                    with_payload=True,
                    limit=limit,
                )
                all_pts.extend(pts)
            except Exception:
                pass
        # 按 count 降序排序
        all_pts.sort(key=lambda pt: (pt.payload or {}).get("count", 0), reverse=True)
        return all_pts[:limit]

    def query(self, memory_ids: list[str]) -> list[dict]:
        """批量查询共现，返回 {id, count} 列表。带 LTD 周期衰减。"""
        memory_set = set(memory_ids)
        all_pts = []
        # 分两次 scroll（id_a / id_b）
        for field in ["id_a", "id_b"]:
            try:
                pts, _ = self._client.scroll(
                    collection_name=self._coll,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key=field, match=models.MatchAny(any=memory_ids),
                        ),
                    ]),
                    with_payload=["id_a", "id_b", "count"],
                    limit=5000,
                )
                all_pts.extend(pts)
            except Exception:
                pass

        seen = set(memory_ids)
        partners: dict[str, int] = {}
        for pt in all_pts:
            p = pt.payload or {}
            partner = p["id_b"] if p["id_a"] in memory_set else p["id_a"]
            if partner not in seen:
                partners[partner] = partners.get(partner, 0) + p.get("count", 0)

        results = sorted(
            [{"id": k, "count": v} for k, v in partners.items()],
            key=lambda x: -x["count"],
        )

        # LTD：周期性衰减
        with self._ltd_lock:
            self._ltd_counter += 1
            if self._ltd_counter >= self.LTD_CHECK_INTERVAL:
                self._ltd_counter = 0
                self._apply_ltd()

        return results

    def _apply_ltd(self):
        """扫描共现条目，超过 LTD_DECAY_DAYS 未同时出现减 1，归零删除。"""
        cutoff = time.time() - self.LTD_DECAY_DAYS * 86400
        try:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="last_time", range=models.Range(lt=cutoff),
                    ),
                ]),
                with_payload=["count"],
                limit=10000,
            )
        except Exception:
            return

        if not pts:
            return

        updates, deletes = [], []
        for pt in pts:
            cnt = (pt.payload or {}).get("count", 0) - self.LTD_DECREMENT
            if cnt <= 0:
                deletes.append(pt.id)
            else:
                updates.append((pt.id, cnt))

        for pt_id, new_cnt in updates:
            try:
                self._client.set_payload(
                    collection_name=self._coll,
                    payload={"count": new_cnt},
                    points=[pt_id],
                )
            except Exception:
                pass
        if deletes:
            try:
                self._client.delete(
                    collection_name=self._coll,
                    points_selector=deletes,
                )
            except Exception:
                pass

    # ── 维护 ──

    def remove(self, memory_id: str):
        """删除包含该 memory_id 的所有共现点。"""
        try:
            pts = self._scroll_by_memory(memory_id, 10000)
            if pts:
                self._client.delete(
                    collection_name=self._coll,
                    points_selector=[pt.id for pt in pts],
                )
        except Exception as exc:
            logger.warning("CoOccurrence remove 失败 id=%s: %s", memory_id[:8], exc)

    def clear(self):
        """清空所有共现记录。"""
        try:
            self._client.delete_collection(self._coll)
            self._ensure_collection()
        except Exception as exc:
            logger.warning("CoOccurrence clear 失败: %s", exc)

    def export_for_symmetry(self, limit: int = 10000) -> dict[str, dict[str, int]]:
        """导出为对称性分析兼容格式：{entity: {related_entity: count}}。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                with_payload=["id_a", "id_b", "count"],
                order_by=models.OrderBy(key="count", direction=models.Direction.DESC),
                limit=limit,
            )
        except Exception:
            return {}

        data: dict[str, dict[str, int]] = {}
        for pt in pts:
            p = pt.payload or {}
            a, b, cnt = p.get("id_a", ""), p.get("id_b", ""), p.get("count", 0)
            if a not in data:
                data[a] = {}
            data[a][b] = cnt
            if b not in data:
                data[b] = {}
            data[b][a] = cnt
        return data


# ===================================================================
# HyperEdgeStore — 独立 Qdrant collection 替代 SQLite hyperedge.py
# ===================================================================

class HyperEdgeStore:
    """超边索引，独立 Qdrant collection。

    提供与 HyperEdgeIndex 完全相同的公开 API，
    底层从 SQLite 三表迁移到 Qdrant 单 collection。

    每条 point = 一个超边（一组实体 + 关联记忆ID）。
    """

    EXPAND_TOP_K = 10
    MAX_EDGES = 10000
    SCROLL_LIMIT_PER_ENTITY = 500

    def __init__(self, client: QdrantClient, collection_name: str,
                 embed_batch_fn=None):
        self._client = client
        self._coll = collection_name
        self._embed_batch_fn = embed_batch_fn  # callable(list[str]) → list[list[float]]
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self._client.get_collections().collections}
        if self._coll not in existing:
            quant_cfg = _build_quantization_config()
            self._client.create_collection(
                collection_name=self._coll,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    on_disk=QDRANT_ON_DISK,
                ),
                quantization_config=quant_cfg,
            )
            logger.info("创建 Qdrant collection: %s (hyper_edges, quantization=%s)",
                       self._coll, QDRANT_QUANTIZATION or "none")
            # Phase 4: payload 索引 — entities keyword, created_at float, edge_size integer
            for field, stype in [
                ("entities", models.PayloadSchemaType.KEYWORD),
                ("created_at", models.PayloadSchemaType.KEYWORD),
                ("edge_size", models.PayloadSchemaType.INTEGER),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self._coll,
                        field_name=field,
                        field_schema=stype,
                    )
                except Exception:
                    pass

    # ── 向后兼容 ──

    def _load(self):
        """向后兼容：返回旧格式 list[dict]。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                with_payload=["entities", "memory_ids"],
                limit=self.MAX_EDGES + 100,
            )
        except Exception:
            return []
        result = []
        for pt in pts:
            p = pt.payload or {}
            result.append({
                "entities": p.get("entities", []),
                "memory_ids": p.get("memory_ids", []),
            })
        return result

    # ── 写入 ──

    def record(self, entities: list[str], memory_id: str):
        """记录一组实体在同一段对话中共现。"""
        entities = sorted(set(e for e in entities
                            if isinstance(e, str) and len(e) >= 2))
        if len(entities) < 2:
            return

        avg_vec = self._compute_avg_embedding(entities) if self._embed_batch_fn else [0.0] * 1024
        point_id = str(uuid.uuid4())
        try:
            self._client.upsert(
                collection_name=self._coll,
                points=[models.PointStruct(
                    id=point_id,
                    vector=avg_vec,
                    payload={
                        "entities": entities,
                        "memory_ids": [memory_id],
                        "created_at": datetime.utcnow().isoformat(),
                        "edge_size": len(entities),
                    },
                )],
            )
        except Exception as exc:
            logger.warning("HyperEdge upsert 失败: %s", exc)
            return

        # 超边数超限 → 裁剪
        try:
            total = self._client.count(collection_name=self._coll).count
        except Exception:
            total = 0
        if total > self.MAX_EDGES:
            self._prune()

    # ── 查询 ──

    def expand(self, entity_names: list[str], top_k: int = None) -> dict[str, int]:
        """展开实体 → 返回 {related_entity: total_weight}。"""
        if top_k is None:
            top_k = self.EXPAND_TOP_K
        if not entity_names:
            return {}

        input_set = set(entity_names)
        scores: dict[str, int] = {}
        for ename in entity_names:
            try:
                pts, _ = self._client.scroll(
                    collection_name=self._coll,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="entities", match=models.MatchValue(value=ename),
                        ),
                    ]),
                    with_payload=["entities"],
                    limit=self.SCROLL_LIMIT_PER_ENTITY,
                )
            except Exception:
                continue
            for pt in pts:
                try:
                    edge_entities = set(pt.payload.get("entities", []))
                except (TypeError, KeyError):
                    continue
                for e in edge_entities - input_set:
                    scores[e] = scores.get(e, 0) + 1

        return dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])

    def get_memory_ids(self, entity_names: list[str],
                       max_memories: int = 50) -> list[str]:
        """给定实体名，收集所有关联超边的记忆 ID，按出现次数降序。"""
        if not entity_names:
            return []

        scored: dict[str, int] = {}
        for ename in entity_names:
            try:
                pts, _ = self._client.scroll(
                    collection_name=self._coll,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="entities", match=models.MatchValue(value=ename),
                        ),
                    ]),
                    with_payload=["memory_ids"],
                    limit=self.SCROLL_LIMIT_PER_ENTITY,
                )
            except Exception:
                continue
            for pt in pts:
                try:
                    mids = pt.payload.get("memory_ids", [])
                except (TypeError, KeyError):
                    mids = []
                for mid in mids:
                    scored[mid] = scored.get(mid, 0) + 1

        return [mid for mid, _ in
                sorted(scored.items(), key=lambda x: -x[1])[:max_memories]]

    def cluster_key(self, entities: list[str],
                    existing_groups: list[set[str]],
                    min_overlap: int = 2) -> int | None:
        """纯内存集合运算——不调 Qdrant。"""
        if not entities:
            return None
        entity_set = set(entities)
        best_idx, best_overlap = None, 0
        for i, group in enumerate(existing_groups):
            overlap = len(entity_set & group)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        return best_idx if best_overlap >= min_overlap else None

    def cluster_entities(self, entities: list[str],
                         min_overlap: int = 2) -> set[str]:
        """通过超边扩展实体集合。"""
        if not entities:
            return set()

        input_set = set(entities)
        result = set(entities)

        for ename in entities:
            try:
                pts, _ = self._client.scroll(
                    collection_name=self._coll,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="entities", match=models.MatchValue(value=ename),
                        ),
                    ]),
                    with_payload=["entities"],
                    limit=200,
                )
            except Exception:
                continue
            for pt in pts:
                try:
                    edge_entities = set(pt.payload.get("entities", []))
                except (TypeError, KeyError):
                    continue
                if len(input_set & edge_entities) >= min_overlap:
                    result |= edge_entities

        return result

    # ── 维护 ──

    def remove_memory(self, memory_id: str):
        """删除记忆时同步清理超边。"""
        try:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                with_payload=["memory_ids"],
                limit=self.MAX_EDGES + 100,
            )
        except Exception:
            return

        updates, deletes = [], []
        for pt in pts:
            try:
                mids = list(pt.payload.get("memory_ids", []))
            except (TypeError, KeyError):
                mids = []
            if memory_id not in mids:
                continue
            mids.remove(memory_id)
            if mids:
                updates.append((pt.id, mids))
            else:
                deletes.append(pt.id)

        for pt_id, new_mids in updates:
            try:
                self._client.set_payload(
                    collection_name=self._coll,
                    payload={"memory_ids": new_mids},
                    points=[pt_id],
                )
            except Exception:
                pass
        if deletes:
            try:
                self._client.delete(
                    collection_name=self._coll,
                    points_selector=deletes,
                )
            except Exception:
                pass

    def _prune(self):
        """裁剪最老的超边，保留最近一半。"""
        keep = self.MAX_EDGES // 2
        try:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                with_payload=["created_at"],
                order_by=models.OrderBy(
                    key="created_at", direction=models.Direction.ASC,
                ),
                limit=self.MAX_EDGES,
            )
        except Exception:
            return
        if len(pts) <= keep:
            return

        to_delete = [pt.id for pt in pts[:len(pts) - keep]]
        try:
            self._client.delete(
                collection_name=self._coll,
                points_selector=to_delete,
            )
        except Exception:
            pass
        logger.info("超边索引裁剪: %d → %d", len(pts), len(to_delete))

    def clear(self):
        """清空所有超边。"""
        try:
            self._client.delete_collection(self._coll)
            self._ensure_collection()
        except Exception as exc:
            logger.warning("HyperEdge clear 失败: %s", exc)

    def stats(self) -> dict:
        try:
            total = self._client.count(collection_name=self._coll).count
        except Exception:
            total = 0
        return {"total_hyperedges": total}

    # ── 辅助 ──

    def _compute_avg_embedding(self, entities: list[str]) -> list[float]:
        """计算实体名称列表的平均 bge-m3 embedding。"""
        if not self._embed_batch_fn:
            return [0.0] * 1024
        try:
            embs = self._embed_batch_fn(entities)
        except Exception:
            return [0.0] * 1024
        valid = [e for e in embs if e is not None]
        if not valid:
            return [0.0] * 1024
        n = len(valid)
        return [sum(dim) / n for dim in zip(*valid)]
