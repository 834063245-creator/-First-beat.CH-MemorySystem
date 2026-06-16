"""超边索引存储 — 独立 Qdrant collection 替代 SQLite hyperedge.py。

Phase 3 从 app/memory/qdrant.py 拆分。
"""
import logging
import uuid
from datetime import datetime
from qdrant_client import QdrantClient, models

from app.config.settings import QDRANT_ON_DISK, QDRANT_QUANTIZATION
from app.memory.qdrant import _build_quantization_config

logger = logging.getLogger(__name__)


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
