# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 22dc33f7

"""时间模式索引 — 发现话题的时间规律，替代硬编码日期匹配。

索引结构：(时间粒度, 话题标签) → {出现次数, 最后出现时间}
查询产出：当前时间活跃的话题模式 → 动态优先级

更新时机：DMN浅巩固时增量维护（每4h扫新记忆）
数据来源：入库时预计算的 time_features（month/day_of_week/season/time_period）
"""

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime

from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)


class TemporalPatternIndex:
    """时间模式索引。发现"每年X月都聊Y"、"每个周一都聊Z"这类规律。"""

    MIN_OBSERVATIONS = 2      # 最少出现次数
    MAX_ENTRIES_PER_GRAN = 500  # 每个粒度最多保留数

    GRANULARITIES = ["month", "day_of_week", "season", "period"]

    def __init__(self, data_dir: str):
        self._path = os.path.join(data_dir, "temporal_patterns.json")
        self._lock = threading.Lock()
        # index[granularity][tag][bucket_str] = {"count": int, "last_seen": float}
        self._index: dict[str, dict] = {g: {} for g in self.GRANULARITIES}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                    for g in self.GRANULARITIES:
                        raw = data.get(g, {})
                        # 确保所有值都是 dict（防格式损坏）
                        self._index[g] = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self):
        with self._lock:
            atomic_write(self._path, self._index)

    def update(self, memories: list[dict]):
        """从一批记忆增量更新时间模式索引。每4h由DMN浅巩固调用。"""
        now = datetime.now()
        # 聚合：每个粒度 × 标签 × 桶 → 增量
        updates: dict[str, dict] = {g: defaultdict(lambda: defaultdict(int))
                                     for g in self.GRANULARITIES}

        for m in memories:
            meta = m.get("metadata") or {}
            tags_str = meta.get("tags", "") or ""
            tags = [t.strip() for t in tags_str.split(",") if len(t.strip()) >= 2]
            if not tags:
                continue

            for tag in tags:
                month = meta.get("month")
                if month and 1 <= month <= 12:
                    updates["month"][tag][str(month)] += 1
                dow = meta.get("day_of_week")
                if dow is not None and 0 <= dow <= 6:
                    updates["day_of_week"][tag][str(dow)] += 1
                season = meta.get("season")
                if season and 1 <= season <= 4:
                    updates["season"][tag][str(season)] += 1
                period = meta.get("time_period")
                if period:
                    updates["period"][tag][str(period)] += 1

        with self._lock:
            for gran in self.GRANULARITIES:
                for tag, buckets in updates[gran].items():
                    if tag not in self._index[gran]:
                        self._index[gran][tag] = {}
                    existing = self._index[gran][tag]
                    for bucket, count in buckets.items():
                        if bucket not in existing:
                            existing[bucket] = {"count": 0, "last_seen": 0}
                        existing[bucket]["count"] += count
                        existing[bucket]["last_seen"] = now.timestamp()
            self._prune()
        self._save()

    def query(self, now: datetime | None = None) -> list[tuple[str, int, str]]:
        """返回当前时间活跃的话题模式。

        返回: [(话题标签, 动态优先级, 粒度名), ...]
        优先级已按强度排序，用于冲动源决定说什么、以多高优先级说。
        """
        if now is None:
            now = datetime.now()
        results: list[tuple[str, int, str]] = []
        now_ts = now.timestamp()

        for gran in self.GRANULARITIES:
            curr = self._current_bucket(gran, now)
            if curr is None:
                continue
            curr_str = str(curr)
            for tag, buckets in self._index[gran].items():
                bd = buckets.get(curr_str)
                if not bd or bd["count"] < self.MIN_OBSERVATIONS:
                    continue
                # 动态优先级：出现越多、越近期，优先级越高
                count = bd["count"]
                recency = 1.0
                last = bd.get("last_seen", 0)
                if last and now_ts - last < 86400 * 14:  # 14天内活跃的加成
                    recency = 1.2
                priority = int(min(count / self.MIN_OBSERVATIONS * 10 * recency, 50))
                if priority >= 10:
                    results.append((tag, priority, gran))

        results.sort(key=lambda x: -x[1])
        return results[:8]

    def prune_memory(self, tag: str):
        """删除某个标签的所有模式记录（用于冲突消解后清理）。"""
        with self._lock:
            for gran in self.GRANULARITIES:
                self._index[gran].pop(tag, None)
        self._save()

    @staticmethod
    def _current_bucket(gran: str, now: datetime):
        if gran == "month":
            return now.month
        elif gran == "day_of_week":
            return now.weekday()
        elif gran == "season":
            return (now.month % 12 + 3) // 3
        elif gran == "period":
            h = now.hour
            if h < 6:
                return "深夜"
            elif h < 9:
                return "早晨"
            elif h < 12:
                return "上午"
            elif h < 14:
                return "中午"
            elif h < 17:
                return "下午"
            elif h < 21:
                return "傍晚"
            else:
                return "晚上"
        return None

    def _prune(self):
        for gran in self.GRANULARITIES:
            idx = self._index[gran]
            total = sum(len(b) for b in idx.values())
            if total <= self.MAX_ENTRIES_PER_GRAN:
                continue
            scored = [(tag, sum(b.get("count", 0) for b in buckets.values()))
                      for tag, buckets in idx.items()]
            scored.sort(key=lambda x: -x[1])
            keep = {t for t, _ in scored[:self.MAX_ENTRIES_PER_GRAN]}
            self._index[gran] = {t: b for t, b in idx.items() if t in keep}
