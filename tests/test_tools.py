"""Tests for query_explore and related tool functions."""
import pytest
import chromadb
from datetime import datetime
from app.tools.dispatch import query_explore, analyze_pattern


def _make_collection(tmp_path):
    """Create an isolated test ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    coll = client.get_or_create_collection("memories", embedding_function=None)
    now = datetime.now().timestamp()
    ids = [f"test_{i}" for i in range(5)]
    docs = [f"用户：这是第{i}条测试消息\nAI：回复{i}" for i in range(5)]
    metas = [{
        "timestamp": now - (5 - i) * 86400,
        "summary": f"测试记忆第{i}条",
        "tags": "test,测试,AI",
        "hit_count": i * 10,
        "emotional_intensity": min(i, 3),
        "emotion_valence": "positive" if i % 2 == 0 else "negative",
    } for i in range(5)]
    embs = [[0.1] * 1024 for _ in range(5)]
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return coll


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
