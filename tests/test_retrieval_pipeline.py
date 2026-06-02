"""检索管线回归测试 — 组件级验证。

注入已知测试记忆 → 验证各检索通路正确返回预期结果。
"""
import sys
sys.path.insert(0, ".")

import os
import tempfile

import pytest


# ============================================================
# 倒排索引测试
# ============================================================

class TestInvertedIndex:
    """验证 InvertedIndex 的构建、查询和更新。"""

    @pytest.fixture
    def idx(self):
        from inverted_index import InvertedIndex
        return InvertedIndex()

    def test_empty_query(self, idx):
        result = idx.query(["不存在的关键词"], min_match=1)
        assert result == []

    def test_basic_query(self, idx):
        idx.build([
            ("mem1", "用户想学 Rust 编程"),
            ("mem2", "用户喜欢猫咪"),
        ])
        result = idx.query(["Rust", "编程"], min_match=2)
        assert "mem1" in result
        assert "mem2" not in result

    def test_min_match_threshold(self, idx):
        idx.build([
            ("mem1", "Rust 编程 性能"),
            ("mem2", "Python 容易学"),
            ("mem3", "Python 编程"),
        ])
        # "编程" 单独查询 → 应返回 mem1 和 mem3
        result = idx.query(["编程"], min_match=1)
        assert "mem1" in result
        assert "mem3" in result

    def test_get_exact(self, idx):
        idx.build([
            ("mem1", "Rust 编程"),
            ("mem2", "Rust 系统"),
        ])
        result = idx.get_exact("Rust")
        assert "mem1" in result
        assert "mem2" in result

    def test_add_incremental(self, idx):
        idx.build([("mem1", "Rust 编程")])
        idx.add("mem2", "Python 编程")
        result = idx.query(["编程"], min_match=1)
        assert "mem1" in result
        assert "mem2" in result

    def test_remove(self, idx):
        idx.build([
            ("mem1", "Rust 编程"),
            ("mem2", "Python 编程"),
        ])
        idx.remove("mem1")
        result = idx.query(["编程"], min_match=1)
        assert "mem1" not in result
        assert "mem2" in result

    def test_and_fallback_to_or(self, idx):
        """AND 结果 <3 条时退化为 OR+匹配数排序。"""
        idx.build([
            ("mem1", "Rust 编程 系统"),
            ("mem2", "Python 编程"),
            ("mem3", "Rust 异步"),
        ])
        # "Rust" AND "编程" → AND 结果只有 mem1 < 3
        # OR 退化: mem1(2匹配), mem2(1), mem3(1) → mem1 排最前
        result = idx.query(["Rust", "编程"], min_match=1)
        assert result[0] == "mem1"
        assert "mem2" in result


# ============================================================
# 共现跟踪器测试
# ============================================================

class TestCoOccurrenceTracker:
    """验证 CoOccurrenceTracker 的基本功能。"""

    @pytest.fixture
    def tracker(self):
        from retrieval import CoOccurrenceTracker
        tmpf = tempfile.mktemp(suffix=".json")
        ct = CoOccurrenceTracker(file_path=tmpf)
        yield ct
        try:
            os.unlink(tmpf)
        except OSError:
            pass

    def test_record_and_query(self, tracker):
        tracker.record(["mem1", "mem2"])
        tracker.record(["mem1", "mem3"])
        tracker.record(["mem1", "mem2"])
        result = tracker.query(["mem1"])
        assert len(result) >= 1
        # mem2 应该比 mem3 高（2次 vs 1次共现）
        mem2_entry = next((r for r in result if r["id"] == "mem2"), None)
        mem3_entry = next((r for r in result if r["id"] == "mem3"), None)
        assert mem2_entry is not None
        assert mem3_entry is not None
        assert mem2_entry["count"] > mem3_entry["count"]

    def test_empty_query(self, tracker):
        result = tracker.query(["nonexistent"])
        assert result == []

    def test_remove_orphan(self, tracker):
        tracker.record(["mem1", "mem2"])
        tracker.remove("mem1")
        result = tracker.query(["mem2"])
        mem1_entries = [r for r in result if r["id"] == "mem1"]
        assert len(mem1_entries) == 0


# ============================================================
# 时间节律函数测试
# ============================================================

class TestTimeRhythmFunctions:
    """验证时间节律 4 窗函数。"""

    def test_recency_score(self):
        from app.background.distill import _recency_score
        now = 1717000000.0
        recent = now - 3 * 86400
        assert _recency_score(recent, now) == 1.0
        old = now - 31 * 86400
        assert _recency_score(old, now) == 0.0
        mid = now - 15 * 86400
        score = _recency_score(mid, now)
        assert 0.0 < score < 1.0
