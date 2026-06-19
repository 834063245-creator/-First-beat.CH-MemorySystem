"""Tests for query_explore and related tool functions."""
import uuid
import pytest
from datetime import datetime
from app.memory.qdrant import QdrantService
from app.tools.dispatch import query_explore, analyze_pattern


def _make_collection(tmp_path):
    """Create an isolated test Qdrant memory store (Qdrant 要求 UUID 点 ID)。"""
    svc = QdrantService(persist_dir=str(tmp_path), collection_name="memories")
    now = datetime.now().timestamp()
    for i in range(5):
        svc.add(
            document=f"用户：这是第{i}条测试消息\nAI：回复{i}",
            metadata={
                "timestamp": now - (5 - i) * 86400,
                "summary": f"测试记忆第{i}条",
                "tags": "test,测试,AI",
                "hit_count": i * 10,
                "emotional_intensity": min(i, 3),
                "emotion_valence": "positive" if i % 2 == 0 else "negative",
            },
            embedding=[0.1] * 1024,
            id=str(uuid.uuid4()),
        )
    return svc


class TestQueryExplore:
    def test_emotion_mode(self, tmp_path):
        coll = _make_collection(tmp_path)
        result = query_explore("emotion", _collection=coll, min_intensity=1, top_k=3)
        assert isinstance(result, str) and len(result) > 0

    def test_timeline_mode(self, tmp_path):
        coll = _make_collection(tmp_path)
        result = query_explore("timeline", _collection=coll,
                               from_date="2020-01-01", to_date="2030-12-31")
        assert isinstance(result, str) and len(result) > 0

    def test_co_occurrence_mode_empty(self, tmp_path):
        coll = _make_collection(tmp_path)
        result = query_explore("co_occurrence", _collection=coll, memory_id="test_0")
        assert "未找到" in result

    def test_rhythm_mode(self, tmp_path):
        coll = _make_collection(tmp_path)
        result = query_explore("rhythm", _collection=coll)
        assert isinstance(result, str)

    def test_topics_mode(self, tmp_path):
        coll = _make_collection(tmp_path)
        result = query_explore("topics", _collection=coll,
                               from_date="2020-01-01", to_date="2030-12-31")
        assert isinstance(result, str)

    def test_invalid_mode(self, tmp_path):
        result = query_explore("invalid_mode")
        assert "未知模式" in result


class TestAnalyzePattern:
    def test_empty_ids(self):
        result = analyze_pattern(memory_ids=[])
        assert "请提供" in result

    def test_invalid_ids(self):
        result = analyze_pattern(memory_ids=["nonexistent_id"])
        assert "未找到" in result or "获取" in result
