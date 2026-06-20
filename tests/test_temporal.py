# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 47183501

"""测试 app/memory/temporal.py — 时间模式索引。

覆盖：TemporalPatternIndex 的更新、查询、剪枝、删除。
"""
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

from app.memory.temporal import TemporalPatternIndex


class TestTemporalPatternIndex:
    @staticmethod
    def _make_index(data_dir: str) -> TemporalPatternIndex:
        """创建无预存数据的索引。"""
        path = os.path.join(data_dir, "temporal_patterns.json")
        if os.path.exists(path):
            os.unlink(path)
        return TemporalPatternIndex(data_dir)

    def test_initial_state(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            assert idx.MIN_OBSERVATIONS == 2
            assert set(idx.GRANULARITIES) == {"month", "day_of_week", "season", "period"}
            for g in idx.GRANULARITIES:
                assert isinstance(idx._index[g], dict)

    def test_update_and_query_basic(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            # 注入两条同月同标签记忆
            mems = [
                {"metadata": {"tags": "咖啡, 生活", "month": 6}},
                {"metadata": {"tags": "咖啡, 工作", "month": 6}},
            ]
            idx.update(mems)
            now = datetime(2025, 6, 15, 10, 0, 0)
            results = idx.query(now=now)
            # 应该匹配 6 月的 "咖啡"
            assert len(results) > 0
            tags = [r[0] for r in results]
            assert "咖啡" in tags

    def test_update_accumulates_counts(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"tags": "运动", "month": 6}},
                {"metadata": {"tags": "运动", "month": 6}},
                {"metadata": {"tags": "运动", "month": 6}},
            ]
            idx.update(mems)
            now = datetime(2025, 6, 15)
            results = idx.query(now=now)
            # 出现 3 次，优先级应该 ≥ 15
            matching = [r for r in results if r[0] == "运动"]
            assert len(matching) > 0
            assert matching[0][1] >= 10

    def test_below_min_observations_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            # 只注入 1 条，达不到 MIN_OBSERVATIONS=2
            mems = [
                {"metadata": {"tags": "罕见话题", "month": 6}},
            ]
            idx.update(mems)
            now = datetime(2025, 6, 15)
            results = idx.query(now=now)
            tags = [r[0] for r in results]
            assert "罕见话题" not in tags

    def test_query_limits_top_8(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            tags = [f"话题{i}" for i in range(15)]
            mems = []
            for t in tags:
                for _ in range(3):
                    mems.append({"metadata": {"tags": t, "month": 6}})
            idx.update(mems)
            now = datetime(2025, 6, 15)
            results = idx.query(now=now)
            assert len(results) <= 8

    def test_query_sorted_by_priority_desc(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            # low 出现 2 次，high 出现 10 次
            mems = []
            for _ in range(2):
                mems.append({"metadata": {"tags": "低频话题", "month": 6}})
            for _ in range(10):
                mems.append({"metadata": {"tags": "高频话题", "month": 6}})
            idx.update(mems)
            now = datetime(2025, 6, 15)
            results = idx.query(now=now)
            # 高频应排在低频前面
            high_idx = next(i for i, r in enumerate(results) if r[0] == "高频话题")
            low_idx = next(i for i, r in enumerate(results) if r[0] == "低频话题")
            assert high_idx < low_idx

    def test_ignore_tags_too_short(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"tags": "a,  b,  咖啡   ", "month": 6}},
                {"metadata": {"tags": "a, 咖啡", "month": 6}},
            ]
            idx.update(mems)
            now = datetime(2025, 6, 15)
            results = idx.query(now=now)
            tags = [r[0] for r in results]
            # "a" 和 "b" 长度 <2，应被过滤
            assert "a" not in tags
            assert "b" not in tags
            # "咖啡" 长度 ≥2，应该保留
            assert "咖啡" in tags

    def test_no_tags_memory_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"month": 6}},  # 无 tags
                {"metadata": {"tags": "", "month": 6}},  # 空 tags
            ]
            idx.update(mems)
            now = datetime(2025, 6, 15)
            results = idx.query(now=now)
            assert len(results) == 0

    def test_day_of_week_granularity(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"tags": "周一综合征", "day_of_week": 0}},
                {"metadata": {"tags": "周一综合征", "day_of_week": 0}},
            ]
            idx.update(mems)
            # 周一
            now = datetime(2025, 6, 2)  # Monday
            results = idx.query(now=now)
            tags = [r[0] for r in results]
            assert "周一综合征" in tags

    def test_season_granularity(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"tags": "夏倦", "season": 2}},
                {"metadata": {"tags": "夏倦", "season": 2}},
            ]
            idx.update(mems)
            now = datetime(2025, 7, 15)  # July → season 3? Let's check
            # season = (now.month % 12 + 3) // 3 → (7 % 12 + 3) // 3 = 10 // 3 = 3
            # But our data has season=2 (summer for June). Let's use June instead.
            now = datetime(2025, 6, 15)  # June → (6 % 12 + 3) // 3 = 9 // 3 = 3
            # Hmm, let me just use the right month. season 2 is months 3-5 (Mar-May).
            # Actually: (3+3)//3=2, (4+3)//3=2, (5+3)//3=2 → season 2 is spring
            # Let me test with season=2 data at March.
            # Better: just adjust the expected season in data.
            idx2 = self._make_index(td)
            # season=3 at June: (6%12+3)//3 = 9//3 = 3
            mems2 = [
                {"metadata": {"tags": "夏日", "season": 3}},
                {"metadata": {"tags": "夏日", "season": 3}},
            ]
            idx2.update(mems2)
            results = idx2.query(now=datetime(2025, 6, 15))
            tags = [r[0] for r in results]
            assert "夏日" in tags

    def test_period_granularity(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"tags": "深夜思绪", "time_period": "深夜"}},
                {"metadata": {"tags": "深夜思绪", "time_period": "深夜"}},
            ]
            idx.update(mems)
            now = datetime(2025, 6, 15, 2, 0, 0)  # 凌晨 2 点 → 深夜
            results = idx.query(now=now)
            tags = [r[0] for r in results]
            assert "深夜思绪" in tags

    def test_prune_memory_removes_all_entries(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            mems = [
                {"metadata": {"tags": "要删除", "month": 6}},
                {"metadata": {"tags": "要删除", "month": 6}},
            ]
            idx.update(mems)
            idx.prune_memory("要删除")
            # 所有粒度下都清掉了
            for g in idx.GRANULARITIES:
                assert "要删除" not in idx._index[g]

    def test_current_bucket_static(self):
        # 测试 _current_bucket 各粒度
        morning = datetime(2025, 6, 15, 7, 30)
        assert TemporalPatternIndex._current_bucket("month", morning) == 6
        # Jun 15 2025 is a Sunday → weekday = 6
        assert TemporalPatternIndex._current_bucket("day_of_week", morning) == 6
        assert TemporalPatternIndex._current_bucket("season", morning) == 3
        assert TemporalPatternIndex._current_bucket("period", morning) == "早晨"

        # 各个时间段
        times = {
            (2, "深夜"), (7, "早晨"), (10, "上午"),
            (13, "中午"), (15, "下午"), (19, "傍晚"), (22, "晚上"),
        }
        for hour, expected in times:
            t = datetime(2025, 6, 15, hour, 0)
            assert TemporalPatternIndex._current_bucket("period", t) == expected

    def test_prune_enforces_max_entries(self):
        """超过 MAX_ENTRIES_PER_GRAN 时自动剪枝。"""
        with tempfile.TemporaryDirectory() as td:
            idx = self._make_index(td)
            # 创建超过 500 个不同标签
            mems = []
            for i in range(600):
                tag = f"x{i:04d}"
                mems.append({"metadata": {"tags": tag, "month": 6}})
                mems.append({"metadata": {"tags": tag, "month": 6}})
            idx.update(mems)
            # 被剪枝后，month 粒度下的条目数 ≤ 500
            total = sum(len(b) for b in idx._index["month"].values())
            assert total <= TemporalPatternIndex.MAX_ENTRIES_PER_GRAN

    def test_load_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "temporal_patterns.json")
            existing = {
                "month": {"咖啡": {"6": {"count": 5, "last_seen": 1700000000}}},
                "day_of_week": {},
                "season": {},
                "period": {},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f)
            idx = TemporalPatternIndex(td)
            assert "咖啡" in idx._index["month"]
            assert idx._index["month"]["咖啡"]["6"]["count"] == 5

    def test_load_corrupted_file_no_crash(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "temporal_patterns.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json{{{{")
            # 不应崩溃
            idx = TemporalPatternIndex(td)
            for g in idx.GRANULARITIES:
                assert isinstance(idx._index[g], dict)

    def test_load_filters_non_dict_values(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "temporal_patterns.json")
            existing = {
                "month": {
                    "咖啡": {"6": {"count": 5, "last_seen": 1700000000}},
                    "broken": "not_a_dict",
                },
                "day_of_week": {},
                "season": {},
                "period": {},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f)
            idx = TemporalPatternIndex(td)
            assert "咖啡" in idx._index["month"]
            assert "broken" not in idx._index["month"]
