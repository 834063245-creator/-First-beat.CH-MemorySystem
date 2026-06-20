# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: cf6864da

"""本地 Payload 索引 — 补偿 Qdrant 本地引擎无服务端索引的限制。

从 qdrant.py 中提取，消除模块内循环依赖。
"""
import bisect
import logging

from qdrant_client import models

logger = logging.getLogger(__name__)


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
