# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: ac238dbc

"""测试 app/tools/dispatch.py 纯函数 — 提行覆盖。

覆盖：_parse_natural_date 全部分支 / count_memories / _get_memory_collection。
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestParseNaturalDate:
    def test_empty_text(self):
        from app.tools.dispatch import _parse_natural_date
        assert _parse_natural_date("") is None

    def test_today(self):
        from app.tools.dispatch import _parse_natural_date
        now = datetime.now()
        result = _parse_natural_date("今天")
        assert result is not None
        assert result["from_date"] == now.strftime("%Y-%m-%d")

    def test_yesterday(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("昨天")
        assert result is not None

    def test_day_before(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("前天")
        assert result is not None

    def test_n_days_ago_digit(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("3天前")
        assert result is not None

    def test_n_days_ago_chinese(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("三天前")
        assert result is not None

    def test_last_week(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("上周")
        assert result is not None
        assert "from_date" in result

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

    def test_month_only(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("3月")
        assert result is not None

    def test_unrecognized_text(self):
        from app.tools.dispatch import _parse_natural_date
        result = _parse_natural_date("乱七八糟的文本")
        assert result is None


class TestCountMemories:
    def test_counts(self):
        from app.tools.dispatch import count_memories
        col = MagicMock()
        col.count.return_value = 42
        result = count_memories(col)
        assert isinstance(result, dict)
        assert result.get("total") == 42 or "error" not in str(result)

    def test_error_handled(self):
        from app.tools.dispatch import count_memories
        col = MagicMock()
        col.count.side_effect = Exception("down")
        result = count_memories(col)
        assert isinstance(result, dict)


class TestGetMemoryCollection:
    def test_returns_collection(self, tmp_path):
        from app.tools.dispatch import _get_memory_collection
        result = _get_memory_collection(str(tmp_path))
        assert result is not None


class TestQueryMemory:
    def test_empty_query(self):
        from app.tools.dispatch import query_memory
        col = MagicMock()
        col.count.return_value = 0
        result = query_memory(col, query="")
        assert isinstance(result, list) or isinstance(result, dict)

    def test_no_results(self):
        from app.tools.dispatch import query_memory
        col = MagicMock()
        col.count.return_value = 5
        col.query.return_value = {"ids": [[]], "documents": [[]],
                                  "metadatas": [[]], "distances": [[]]}
        with patch("app.llm.embed.local_embed", return_value=[0.1] * 3584):
            result = query_memory(col, query="稀有查询词", from_date="", to_date="")
            assert isinstance(result, list) or isinstance(result, dict)
