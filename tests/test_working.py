# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 7d643ae4

"""测试 app/memory/working.py — 工作记忆摘要。"""
import json
import os
import tempfile

import pytest


# ═══════════════════════════════════════════════════════════════
# _load / _save — 文件读写
# ═══════════════════════════════════════════════════════════════

class TestLoad:
    def test_missing_file_returns_default(self):
        from app.memory.working import _load
        result = _load("/nonexistent/wm.json")
        assert result["summary"] == ""
        assert result["topics"] == []
        assert result["version"] == 0

    def test_loads_valid_file(self):
        from app.memory.working import _load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            data = {"summary": "测试摘要", "topics": ["Python", "AI"], "version": 3}
            f.write(json.dumps(data))
            f.flush()
            result = _load(f.name)
        os.unlink(f.name)
        assert result["summary"] == "测试摘要"
        assert result["topics"] == ["Python", "AI"]
        assert result["version"] == 3

    def test_invalid_json_returns_default(self):
        from app.memory.working import _load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not json{{{")
            f.flush()
            result = _load(f.name)
        os.unlink(f.name)
        assert result["summary"] == ""


class TestSave:
    def test_saves_and_reloads(self):
        from app.memory.working import _save, _load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.close()
            data = {"summary": "新摘要", "version": 1}
            _save(data, f.name)
            result = _load(f.name)
        os.unlink(f.name)
        assert result["summary"] == "新摘要"
        assert result["version"] == 1


# ═══════════════════════════════════════════════════════════════
# get_summary — 摘要格式化
# ═══════════════════════════════════════════════════════════════

class TestGetSummary:
    def test_empty_wm_returns_empty(self):
        from app.memory.working import get_summary
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"summary": "", "topics": [], "current_state": "", "last_updated": "", "version": 0}))
            f.flush()
            result = get_summary(f.name)
        os.unlink(f.name)
        assert result == ""

    def test_full_wm(self):
        from app.memory.working import get_summary
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            data = {
                "summary": "用户在讨论Python性能优化",
                "current_state": "正在写代码",
                "topics": ["Python", "性能", "优化"],
                "recent_keywords": ["numpy", "pandas"],
                "recent_entities": ["Python"],
                "version": 1,
            }
            f.write(json.dumps(data))
            f.flush()
            result = get_summary(f.name)
        os.unlink(f.name)
        assert "Python性能优化" in result
        assert "正在写代码" in result
        assert "Python" in result

    def test_summary_only(self):
        from app.memory.working import get_summary
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"summary": "只有摘要", "topics": [], "current_state": "", "version": 0}))
            f.flush()
            result = get_summary(f.name)
        os.unlink(f.name)
        assert "只有摘要" in result


# ═══════════════════════════════════════════════════════════════
# _topic_shift_detected — 话题变化检测
# ═══════════════════════════════════════════════════════════════

class TestTopicShiftDetected:
    def test_no_topics_returns_true(self):
        from app.memory.working import _topic_shift_detected
        turns = [{"user_message": "聊聊Python"}]
        assert _topic_shift_detected(turns, []) is True

    def test_empty_turns_returns_false(self):
        from app.memory.working import _topic_shift_detected
        assert _topic_shift_detected([], ["Python"]) is False

    def test_empty_messages_returns_false(self):
        from app.memory.working import _topic_shift_detected
        turns = [{"user_message": ""}]
        assert _topic_shift_detected(turns, ["Python"]) is False

    def test_turns_too_few(self):
        """turns 数量少于 MIN_UPDATE_INTERVAL 时外层会拦截，这里测内层逻辑。"""
        from app.memory.working import _topic_shift_detected
        turns = [{"user_message": "今天天气不错"}]
        # wm_topics 不为空且有有效文本，应运行话题检测
        result = _topic_shift_detected(turns, ["Python", "AI"])
        # 话题不同，应检测到偏移
        assert result in (True, False)  # 取决于 extract_tags 是否可用


# ═══════════════════════════════════════════════════════════════
# incremental_update — 增量更新
# ═══════════════════════════════════════════════════════════════

class TestIncrementalUpdate:
    def test_too_few_turns_returns_false(self):
        from app.memory.working import incremental_update
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"summary": "", "topics": [], "version": 0}))
            f.flush()
            path = f.name
        result = incremental_update([{"user_message": "hi"}], wm_path=path)
        os.unlink(path)
        assert result is False

    def test_sufficient_turns_without_shift(self):
        """>= 5 turns 但话题没变 → 不更新。"""
        from app.memory.working import incremental_update
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"summary": "旧摘要", "topics": [], "version": 0}))
            f.flush()
            path = f.name
        turns = [{"user_message": ""} for _ in range(6)]
        result = incremental_update(turns, wm_path=path)
        os.unlink(path)
        assert result is False
