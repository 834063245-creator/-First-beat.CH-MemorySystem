# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 1848c19e

"""记忆对共现存储 — 独立 Qdrant collection 替代 SQLite cooccur.py。

Phase 3 从 app/memory/qdrant.py 拆分。
"""
import logging
import time
import uuid
from datetime import datetime
from qdrant_client import QdrantClient, models

from app.config.settings import QDRANT_ON_DISK, QDRANT_QUANTIZATION
from app.memory.qdrant import _build_quantization_config

logger = logging.getLogger(__name__)


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
        self._ltd_lock = __import__('threading').Lock()
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
