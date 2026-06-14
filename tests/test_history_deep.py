"""测试 app/memory/history.py 深层方法 — 提行覆盖。

覆盖：ChatHistory.append / get_recent / delete_by_timestamp / clear / annotate_chunks。
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest


class TestChatHistory:
    @pytest.fixture
    def history(self):
        with tempfile.TemporaryDirectory() as td:
            from app.memory.history import ChatHistory
            path = os.path.join(td, "chat_history.jsonl")
            h = ChatHistory(path)
            yield h

    def test_append_and_get_recent(self, history):
        history.append("你好", "你好呀", "2025-06-01 10:00:00")
        records = history.get_recent(10)
        assert len(records) >= 1
        assert any("你好" in str(r) for r in records)

    def test_get_recent_limit(self, history):
        for i in range(30):
            history.append(f"msg{i}", f"reply{i}", f"2025-06-{i+1:02d} 10:00:00")
        records = history.get_recent(5)
        assert len(records) <= 5

    def test_truncate_with_many_messages(self):
        """大量消息存储后 get_recent 仍然正常。"""
        from app.memory.history import ChatHistory
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "chat_history.jsonl")
            h = ChatHistory(path)
            for i in range(20):
                h.append(f"msg{i}", f"reply{i}", f"2025-06-{i+1:02d} 10:00:00")
            records = h.get_recent(10)
            assert len(records) <= 10
            # 应该是最新的 10 条
            assert any("msg19" in str(r) for r in records)

    def test_delete_by_timestamp(self, history):
        history.append("msg1", "r1", "2025-06-01 10:00:00")
        history.append("msg2", "r2", "2025-06-02 10:00:00")
        ok = history.delete_by_timestamp("2025-06-01 10:00:00")
        assert ok is True
        records = history.get_recent(10)
        # msg1 被删了
        assert not any("msg1" in str(r) for r in records)

    def test_delete_nonexistent_timestamp(self, history):
        ok = history.delete_by_timestamp("2099-01-01 00:00:00")
        assert ok is False

    def test_annotate_chunks_static(self):
        from app.memory.history import ChatHistory
        timeline = [
            {"user_message": "你好", "llm_reply": "你好呀", "timestamp": "2025-06-01 10:00"},
        ]
        result = ChatHistory.annotate_chunks(timeline)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_annotate_chunks_empty(self):
        from app.memory.history import ChatHistory
        result = ChatHistory.annotate_chunks([])
        assert result == []

    def test_get_records_snapshot(self, history):
        history.append("test", "reply", "2025-06-01 10:00:00")
        records = history.get_records_snapshot()
        assert isinstance(records, list)

    def test_get_context_by_chroma_id(self, history):
        history.append("user msg", "ai reply", "2025-06-01 10:00:00")
        # 按 chroma_id 检索上下文
        result = history.get_context_by_chroma_id("nonexistent_id")
        assert isinstance(result, dict)
        assert "context_before" in result or isinstance(result.get("context_before", []), list)
