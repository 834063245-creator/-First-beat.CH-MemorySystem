# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 5ac5280f

"""测试 app/core/helpers.py — 共享辅助函数。

覆盖：build_trace / build_debug_info / _load_jsonl_cached / load_recent_reversals。
"""
import json
import os
import tempfile
import time
import pytest
from app.core.helpers import (
    build_trace,
    build_debug_info,
    _load_jsonl_cached,
    load_recent_reversals,
)


class TestBuildTrace:
    def test_extracts_id_and_summary(self):
        mems = [{
            "id": "mem_001",
            "metadata": {
                "summary": "用户喜欢喝咖啡",
                "timestamp": 1700000000,
                "hit_count": 5,
                "tags": "咖啡, 生活",
            },
            "source": "semantic",
            "display_source": "语义检索",
        }]
        trace = build_trace(mems)
        assert len(trace) == 1
        assert trace[0]["id"] == "mem_001"
        assert trace[0]["summary"] == "用户喜欢喝咖啡"
        assert trace[0]["hit_count"] == 5
        assert trace[0]["tags"] == ["咖啡", "生活"]

    def test_tags_list_passthrough(self):
        """tags 如已是 list，直接使用。"""
        mems = [{
            "id": "mem_001",
            "metadata": {"tags": ["Rust", "编程"], "summary": "学习Rust"},
            "source": "keyword",
            "display_source": "关键词匹配",
        }]
        trace = build_trace(mems)
        assert trace[0]["tags"] == ["Rust", "编程"]

    def test_empty_tags(self):
        mems = [{
            "id": "mem_001",
            "metadata": {"tags": "", "summary": ""},
            "source": "",
            "display_source": "",
        }]
        trace = build_trace(mems)
        assert trace[0]["tags"] == []

    def test_empty_memories(self):
        assert build_trace([]) == []


class TestBuildDebugInfo:
    def test_basic_structure(self):
        mems = [{
            "id": "mem_001",
            "metadata": {"summary": "测试", "hit_count": 3, "timestamp": 1700000000},
            "semantic_score": 0.85,
            "reason": "语义检索",
        }]
        p_notes = ["用户喜欢编程", {"content": "偏好安静", "hit_count": 2}]
        timeline = [{"user": "你好", "ai": "你好呀"}]
        result = build_debug_info(mems, p_notes, timeline)
        assert "retrieved_memories" in result
        assert "personalities" in result
        assert "timeline_recent" in result
        assert len(result["retrieved_memories"]) == 1
        assert result["retrieved_memories"][0]["semantic_score"] == 0.85

    def test_with_prompt(self):
        result = build_debug_info([], [], [], prompt="测试prompt")
        assert result["prompt"] == "测试prompt"

    def test_without_prompt(self):
        result = build_debug_info([], [], [])
        assert "prompt" not in result

    def test_personality_string(self):
        p_notes = ["标签1", "标签2"]
        result = build_debug_info([], p_notes, [])
        assert len(result["personalities"]) == 2
        assert result["personalities"][0]["content"] == "标签1"


class TestLoadJsonlCached:
    def test_caches_result(self):
        """30s TTL 内相同 mtime 直接返回缓存。"""
        import app.core.helpers as h
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"a": 1}) + "\n")

            calls = [0]
            def parse():
                calls[0] += 1
                with open(path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]

            # 清除模块级缓存避免测试间污染
            with h._jsonl_cache_lock:
                h._jsonl_cache.pop(path, None)

            result1 = _load_jsonl_cached(path, parse)
            assert calls[0] == 1
            result2 = _load_jsonl_cached(path, parse)
            assert calls[0] == 1  # 缓存命中，未重新调用 parse

    def test_refreshes_after_mtime_change(self):
        import time
        import app.core.helpers as h
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ver": 1}) + "\n")

            calls = [0]
            def parse():
                calls[0] += 1
                with open(path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]

            with h._jsonl_cache_lock:
                h._jsonl_cache.pop(path, None)

            _load_jsonl_cached(path, parse)
            assert calls[0] == 1
            # 修改文件（mtime 变化）
            time.sleep(0.05)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ver": 2}) + "\n")
            _load_jsonl_cached(path, parse)
            assert calls[0] == 2  # mtime 变化 → 重新读取


class TestLoadRecentReversals:
    def test_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            result = load_recent_reversals(data_dir=td)
            assert result == []
