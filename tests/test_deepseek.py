# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 367af850

"""测试 app/llm/deepseek.py 纯函数。

覆盖：now_hint / _relative_time / _confidence_label / _timeline_to_messages
      / parse_dsml_tool_calls / strip_dsml / _build_impulses。
"""
import json
import re
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest


# ═══════════════════════════════════════════════════════
# now_hint
# ═══════════════════════════════════════════════════════

class TestNowHint:
    def test_returns_formatted_time(self):
        from app.llm.deepseek import now_hint
        result = now_hint()
        assert "当前时间" in result
        assert "星期" in result or "星期" in result

    def test_contains_year_month_day(self):
        from app.llm.deepseek import now_hint
        result = now_hint()
        now = datetime.now()
        assert str(now.year) in result
        assert str(now.month) in result


# ═══════════════════════════════════════════════════════
# _relative_time
# ═══════════════════════════════════════════════════════

class TestRelativeTime:
    def test_just_now(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 30
        result = LLMClient._relative_time(ts)
        assert "刚刚" in result or "秒" in result or "分钟前" in result

    def test_minutes_ago(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 180  # 3 分钟前
        result = LLMClient._relative_time(ts)
        assert "分钟前" in result

    def test_hours_ago(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 7200  # 2 小时前
        result = LLMClient._relative_time(ts)
        assert "小时前" in result

    def test_days_ago(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 172800  # 2 天前
        result = LLMClient._relative_time(ts)
        assert "天前" in result

    def test_weeks_ago(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 86400 * 10  # 10 天前
        result = LLMClient._relative_time(ts)
        assert "周前" in result

    def test_months_ago(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 86400 * 40  # 40 天前
        result = LLMClient._relative_time(ts)
        assert "个月前" in result

    def test_years_ago(self):
        from app.llm.deepseek import LLMClient
        ts = time.time() - 86400 * 400  # >1 年前
        result = LLMClient._relative_time(ts)
        assert "年前" in result


# ═══════════════════════════════════════════════════════
# _confidence_label
# ═══════════════════════════════════════════════════════

class TestConfidenceLabel:
    def test_high_via_hit_count(self):
        from app.llm.deepseek import LLMClient
        assert LLMClient._confidence_label({
            "metadata": {"hit_count": 50},
            "source": "semantic",
        }) == "高"

    def test_medium_via_hit_count(self):
        from app.llm.deepseek import LLMClient
        assert LLMClient._confidence_label({
            "metadata": {"hit_count": 10},
            "source": "semantic",
        }) == "中"

    def test_low_by_default(self):
        from app.llm.deepseek import LLMClient
        assert LLMClient._confidence_label({
            "metadata": {"hit_count": 3},
            "source": "semantic",
        }) == "低"

    def test_indirect_source_is_low(self):
        from app.llm.deepseek import LLMClient
        assert LLMClient._confidence_label({
            "metadata": {"hit_count": 100},
            "source": "co_occurrence",
        }) == "低"
        assert LLMClient._confidence_label({
            "metadata": {"hit_count": 100},
            "source": "time_triggered",
        }) == "低"

    def test_low_similarity_overrides(self):
        from app.llm.deepseek import LLMClient
        # distance=0.8 → sim=0.2 → <0.3 → 低
        assert LLMClient._confidence_label({
            "metadata": {"hit_count": 50},
            "distance": 0.8,
            "source": "semantic",
        }) == "低"


# ═══════════════════════════════════════════════════════
# _timeline_to_messages
# ═══════════════════════════════════════════════════════

class TestTimelineToMessages:
    def test_converts_user_ai_pair(self):
        from app.llm.deepseek import LLMClient
        timeline = [{
            "user_message": "你好",
            "llm_reply": "你好呀",
            "timestamp": 1700000000,
        }]
        msgs = LLMClient._timeline_to_messages(timeline)
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles
        user_msg = [m for m in msgs if m["role"] == "user"][0]
        assert "你好" in user_msg["content"]

    def test_inner_monologue(self):
        from app.llm.deepseek import LLMClient
        timeline = [{
            "user_message": "[内心独白]",
            "llm_reply": "用户好像有点不开心",
        }]
        msgs = LLMClient._timeline_to_messages(timeline)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert "[内心独白]" in msgs[0]["content"]

    def test_empty_user_skipped(self):
        from app.llm.deepseek import LLMClient
        timeline = [{
            "user_message": "",
            "llm_reply": "回复",
        }]
        msgs = LLMClient._timeline_to_messages(timeline)
        # 只有 assistant 消息
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"


# ═══════════════════════════════════════════════════════
# parse_dsml_tool_calls
# ═══════════════════════════════════════════════════════

class TestParseDsml:
    @patch("app.llm.deepseek.LLM_MODEL", "deepseek-v4-flash")
    def test_parses_dsml_format(self):
        from app.llm.deepseek import parse_dsml_tool_calls
        text = '<|DSML|tool_calls|name|search_web|params|' \
               '<|DSML|parameter name|query|>测试</|DSML|parameter>' \
               '<|DSML|parameter name|count|>5</|DSML|parameter>' \
               '|>\n</|DSML|tool_calls|>'
        calls = parse_dsml_tool_calls(text)
        assert len(calls) >= 1
        assert calls[0]["function"]["name"] == "search_web"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "测试"

    @patch("app.llm.deepseek.LLM_MODEL", "deepseek-v4-flash")
    def test_no_dsml_returns_empty(self):
        from app.llm.deepseek import parse_dsml_tool_calls
        text = "这是普通文本，没有 DSML 标记"
        calls = parse_dsml_tool_calls(text)
        assert calls == []

    @patch("app.llm.deepseek.LLM_MODEL", "deepseek-v4-flash")
    def test_empty_text(self):
        from app.llm.deepseek import parse_dsml_tool_calls
        assert parse_dsml_tool_calls("") == []


# ═══════════════════════════════════════════════════════
# strip_dsml
# ═══════════════════════════════════════════════════════

class TestStripDsml:
    @patch("app.llm.deepseek.LLM_MODEL", "deepseek-v4-flash")
    def test_removes_dsml_tags(self):
        from app.llm.deepseek import strip_dsml
        text = '正常文本 <|DSML|tool_calls|name|test|params||> </|DSML|tool_calls|> 后续'
        result = strip_dsml(text)
        assert "<|DSML" not in result
        assert "正常文本" in result
        assert "后续" in result

    @patch("app.llm.deepseek.LLM_MODEL", "deepseek-v4-flash")
    def test_plain_text_unchanged(self):
        from app.llm.deepseek import strip_dsml
        text = "这是纯文本没有标记"
        result = strip_dsml(text)
        assert result == text

    @patch("app.llm.deepseek.LLM_MODEL", "deepseek-v4-flash")
    def test_strips_unclosed_fragments(self):
        from app.llm.deepseek import strip_dsml
        text = "前面 <|DSML|tool_calls 未闭合的片段"
        result = strip_dsml(text)
        assert "<|DSML" not in result
        assert "前面" in result


# ═══════════════════════════════════════════════════════
# _build_impulses
# ═══════════════════════════════════════════════════════

class TestBuildImpulses:
    def test_empty_impulses(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.impulses = []
        result = LLMClient._build_impulses(mock_state)
        assert result == ""

    def test_recall_impulse(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_impulse = MagicMock()
        mock_impulse.intent = "recall"
        mock_impulse.target_concept = "咖啡"
        mock_state.impulses = [mock_impulse]
        result = LLMClient._build_impulses(mock_state)
        assert "咖啡" in result

    def test_check_impulse(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_impulse = MagicMock()
        mock_impulse.intent = "check"
        mock_impulse.target_concept = "天气"
        mock_state.impulses = [mock_impulse]
        result = LLMClient._build_impulses(mock_state)
        assert "天气" in result

    def test_unknown_intent_fallback(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_impulse = MagicMock()
        mock_impulse.intent = "unknown_type"
        mock_impulse.target_concept = "随便什么"
        mock_state.impulses = [mock_impulse]
        result = LLMClient._build_impulses(mock_state)
        assert "随便什么" in result
        assert "念头浮动" in result


# ═══════════════════════════════════════════════════════
# _build_memories_for_tool
# ═══════════════════════════════════════════════════════

class TestBuildMemoriesForTool:
    def test_empty_memories(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.woven_context = None
        mock_state.memories = []
        mock_state.stale_context = []
        result = LLMClient._build_memories_for_tool(mock_state)
        items = json.loads(result)
        assert items == []

    def test_formats_memory_dict(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.woven_context = None
        mock_state.memories = [{
            "id": "mem_001",
            "summary": "用户喜欢咖啡",
            "document": "用户说他喜欢喝咖啡",
            "metadata": {
                "timestamp": 1700000000,
                "hit_count": 3,
                "stale": False,
            },
            "display_source": "语义检索",
            "score": 0.85,
        }]
        mock_state.stale_context = []
        result = LLMClient._build_memories_for_tool(mock_state)
        items = json.loads(result)
        assert len(items) == 1
        assert items[0]["id"] == "mem_001"
        assert items[0]["summary"] == "用户喜欢咖啡"
        assert items[0]["relevance"] == 0.85

    def test_no_timestamp_handled(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.woven_context = None
        mock_state.memories = [{
            "id": "mem_002",
            "summary": "测试",
            "metadata": {},
            "display_source": "关键词",
            "score": 0.5,
        }]
        mock_state.stale_context = []
        result = LLMClient._build_memories_for_tool(mock_state)
        items = json.loads(result)
        assert items[0]["time"] == ""
        assert items[0]["relative_time"] == ""


# ═══════════════════════════════════════════════════════
# load_system_prompt
# ═══════════════════════════════════════════════════════

class TestLoadSystemPrompt:
    def test_returns_string(self):
        from app.llm.deepseek import load_system_prompt
        result = load_system_prompt()
        assert isinstance(result, str)

    @patch("app.llm.deepseek._PROMPT_PATH", "/nonexistent/path/prompt.txt")
    def test_missing_file_returns_empty(self):
        from app.llm.deepseek import load_system_prompt
        result = load_system_prompt()
        assert result == ""


# ═══════════════════════════════════════════════════════
# LLMClient 构造
# ═══════════════════════════════════════════════════════

class TestLLMClientConstruction:
    def test_creates_without_crash(self):
        from app.llm.deepseek import LLMClient
        client = LLMClient()
        assert client.model is not None

    def test_set_pattern_discovery(self):
        from app.llm.deepseek import LLMClient
        client = LLMClient()
        mock_pd = MagicMock()
        client.set_pattern_discovery(mock_pd)
        assert client._pattern_discovery is mock_pd
