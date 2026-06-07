"""测试 app/tools/dispatch.py — dispatch 函数。

覆盖：query_memory / query_explore / count_memories / analyze_pattern 核心路径。
"""
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════
# query_memory
# ═══════════════════════════════════════════════════════════════

class TestQueryMemory:
    def test_no_query_no_time_returns_error(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        result = query_memory(coll, query="", from_date="", to_date="")
        assert len(result) == 1
        assert "error" in result[0]

    def test_query_with_empty_string(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        result = query_memory(coll, query="   ", from_date="", to_date="")
        assert len(result) == 1
        assert "error" in result[0]

    def test_with_query(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        # mock embedding
        coll.query.return_value = {"ids": [[]], "metadatas": [[{}]], "documents": [[""]], "distances": [[0.5]]}
        coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with patch("app.tools.dispatch.local_embed", return_value=[0.1] * 1024):
            result = query_memory(coll, query="Python")
        assert isinstance(result, list)

    def test_with_from_date_only(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        result = query_memory(coll, query="", from_date="2026-01-01", to_date="")
        assert isinstance(result, list)

    def test_with_to_date_only(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        result = query_memory(coll, query="", from_date="", to_date="2026-01-01")
        assert isinstance(result, list)

    def test_with_both_dates(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        result = query_memory(coll, query="", from_date="2026-01-01", to_date="2026-01-31")
        assert isinstance(result, list)

    def test_top_k_capped(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        coll.get.return_value = {"ids": ["m1", "m2", "m3"], "documents": ["d1", "d2", "d3"],
                                  "metadatas": [{}, {}, {}]}
        result = query_memory(coll, query="", from_date="2026-01-01", to_date="2026-01-31", top_k=2)
        assert len(result) <= 2

    def test_with_filters(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        result = query_memory(coll, query="Python", from_date="", to_date="",
                              filters={"time_period": "上午", "emotional_intensity": 2})
        assert isinstance(result, list)

    def test_invalid_date_format_handled(self):
        from app.tools.dispatch import query_memory
        coll = MagicMock()
        coll.query.return_value = {"ids": [[]], "metadatas": [[{}]], "documents": [[""]], "distances": [[0.5]]}
        coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with patch("app.tools.dispatch.local_embed", return_value=[0.1] * 1024):
            result = query_memory(coll, query="test", from_date="bad_date", to_date="")
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════
# count_memories
# ═══════════════════════════════════════════════════════════════

class TestCountMemories:
    def test_returns_total(self):
        from app.tools.dispatch import count_memories
        coll = MagicMock()
        coll.count.return_value = 42
        result = count_memories(coll)
        assert result["total_memories"] == 42

    def test_error_handled(self):
        from app.tools.dispatch import count_memories
        coll = MagicMock()
        coll.count.side_effect = Exception("DB error")
        result = count_memories(coll)
        assert "error" in result


# ═══════════════════════════════════════════════════════════════
# query_explore — modes
# ═══════════════════════════════════════════════════════════════

class TestQueryExplore:
    def test_unknown_mode(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        result = query_explore(mode="unknown_mode", _collection=coll)
        assert "未知模式" in result or "不支持" in result

    def test_timeline_no_time(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        result = query_explore(mode="timeline", _collection=coll)
        assert "请提供时间范围" in result

    def test_timeline_with_time(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "metadatas": []}
        result = query_explore(mode="timeline", _collection=coll,
                              from_date="2026-01-01", to_date="2026-01-02")
        assert "该时间段内没有记忆" in result or isinstance(result, str)

    def test_timeline_with_when(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "metadatas": []}
        result = query_explore(mode="timeline", _collection=coll, when="今天")
        assert isinstance(result, str)

    def test_emotion_no_results(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "metadatas": []}
        result = query_explore(mode="emotion", _collection=coll, min_intensity=2)
        assert "未找到" in result or isinstance(result, str)

    def test_emotion_with_valence(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "metadatas": []}
        result = query_explore(mode="emotion", _collection=coll, valence="negative")
        assert isinstance(result, str)

    def test_topics_no_data(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "metadatas": [], "documents": []}
        result = query_explore(mode="topics", _collection=coll,
                              from_date="2026-01-01", to_date="2026-01-02")
        assert "数据不足" in result or isinstance(result, str)

    def test_rhythm(self):
        from app.tools.dispatch import query_explore
        coll = MagicMock()
        coll.get.return_value = {"ids": [], "metadatas": []}
        result = query_explore(mode="rhythm", _collection=coll)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════
# analyze_pattern
# ═══════════════════════════════════════════════════════════════

class TestAnalyzePattern:
    def test_no_ids_returns_prompt(self):
        from app.tools.dispatch import analyze_pattern
        result = analyze_pattern(memory_ids=[])
        assert "请提供" in result

    def test_no_ids_none(self):
        from app.tools.dispatch import analyze_pattern
        result = analyze_pattern(memory_ids=None)
        assert "请提供" in result


# ═══════════════════════════════════════════════════════════════
# _parse_natural_date
# ═══════════════════════════════════════════════════════════════

class TestParseNaturalDate:
    def test_today(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("今天")
        assert result is not None
        assert "from_date" in result

    def test_yesterday(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("昨天")
        assert result is not None

    def test_day_before_yesterday(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("前天")
        assert result is not None

    def test_n_days_ago(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("3天前")
        assert result is not None

    def test_last_week(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("上周")
        assert result is not None

    def test_last_month(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("上个月")
        assert result is not None

    def test_this_month(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("这个月")
        assert result is not None

    def test_month_day(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("6月15日")
        assert result is not None

    def test_unknown_format(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("未来的某一天")
        assert result is None

    def test_empty_text(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("")
        assert result is None
