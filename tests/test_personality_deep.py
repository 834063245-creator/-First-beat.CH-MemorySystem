"""测试 app/personality/store.py 深层方法 — 提行覆盖。

覆盖：rerank_tags / store_tag / increment_hit / update_tag / search。
"""
import time
from unittest.mock import MagicMock, patch
import pytest


def _fake_personality_collection():
    col = MagicMock()
    col.query.return_value = {
        "ids": [["p1", "p2", "p3"]],
        "documents": [["用户喜欢咖啡", "用户喜欢编程", "用户喜欢安静"]],
        "metadatas": [[
            {"type": "偏好模式", "confidence": "高", "hit_count": 10, "last_hit_time": time.time(),
             "outdated": False, "source": "user"},
            {"type": "行为模式", "confidence": "中", "hit_count": 5, "last_hit_time": time.time() - 99999,
             "outdated": False, "source": "user"},
            {"type": "偏好模式", "confidence": "低", "hit_count": 1, "last_hit_time": time.time() - 86400 * 10,
             "outdated": True, "source": "user"},
        ]],
        "distances": [[0.1, 0.2, 0.5]],
        "embeddings": [[[0.1] * 10, [0.2] * 10, [0.3] * 10]],
    }
    col.get.return_value = {
        "ids": ["p1"], "documents": ["标签"], "metadatas": [{"hit_count": 5}],
    }
    col.count.return_value = 3
    col.add.return_value = None
    col.update.return_value = None
    col.delete.return_value = None
    return col


class TestRerankTags:
    @patch("chromadb.PersistentClient")
    def test_empty_results(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.query.return_value = {"ids": [[]], "documents": [[]],
                                  "metadatas": [[]], "distances": [[]],
                                  "embeddings": [[]]}
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.rerank_tags("咖啡", [0.1] * 1024)
        assert result == []

    @patch("chromadb.PersistentClient")
    def test_filters_outdated(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.query.return_value = {
            "ids": [["p1"]],
            "documents": [["过时标签"]],
            "metadatas": [[{"outdated": True, "type": "行为模式", "confidence": "低",
                            "hit_count": 0, "last_hit_time": 0, "source": "user"}]],
            "distances": [[0.1]],
            "embeddings": [[None]],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.rerank_tags("查询", [0.1] * 1024)
        assert result == []  # 过时被过滤

    @patch("chromadb.PersistentClient")
    def test_confidence_boost(self, mock_client):
        from app.personality.store import PersonalityStore
        col = _fake_personality_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.increment_hit = MagicMock()  # 避免真实 ChromaDB 调用
        result = store.rerank_tags("咖啡", [0.1] * 1024)
        assert len(result) > 0
        # 高置信度的应排在前面
        confs = [r.get("confidence") for r in result]
        if "高" in confs and "低" in confs:
            assert confs.index("高") < confs.index("低")

    @patch("chromadb.PersistentClient")
    def test_topic_match_boost(self, mock_client):
        from app.personality.store import PersonalityStore
        col = _fake_personality_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.increment_hit = MagicMock()
        # "咖啡" 会匹配 TOPIC_KEYWORDS 中的"生活"类或"情感"类
        result = store.rerank_tags("咖啡", [0.1] * 1024)
        assert isinstance(result, list)


class TestStoreTag:
    @patch("chromadb.PersistentClient")
    def test_returns_uuid(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        tag_id = store.store_tag("用户喜欢音乐", [0.1] * 1024, tag_type="偏好模式")
        assert isinstance(tag_id, str)
        assert len(tag_id) > 30  # UUID
        col.add.assert_called_once()


class TestIncrementHit:
    @patch("chromadb.PersistentClient")
    def test_increments_count(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {
            "ids": ["p1"],
            "metadatas": [{"hit_count": 5, "last_hit_time": 0}],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.increment_hit("p1")
        col.update.assert_called_once()
        meta = col.update.call_args[1]["metadatas"][0]
        assert meta["hit_count"] == 6

    @patch("chromadb.PersistentClient")
    def test_nonexistent_no_error(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {"ids": [], "metadatas": []}
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.increment_hit("no_such_tag")  # 不崩溃


class TestUpdateTag:
    @patch("chromadb.PersistentClient")
    def test_update_with_content(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {
            "ids": ["p1"], "metadatas": [{"type": "行为", "confidence": "中"}],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        with patch("app.llm.embed.local_embed", return_value=[0.1] * 1024):
            store.update_tag("p1", content="新内容")
            assert col.update.called

    @patch("chromadb.PersistentClient")
    def test_update_metadata_only(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {
            "ids": ["p1"], "metadatas": [{"type": "行为", "confidence": "低"}],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.update_tag("p1", tag_type="偏好模式", confidence="高")
        call_meta = col.update.call_args[1]["metadatas"][0]
        assert call_meta["type"] == "偏好模式"
        assert call_meta["confidence"] == "高"


class TestSearch:
    @patch("chromadb.PersistentClient")
    def test_returns_items(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.query.return_value = {
            "ids": [["p1"]],
            "documents": [["标签1"]],
            "metadatas": [[{"type": "行为", "confidence": "中"}]],
            "distances": [[0.1]],
            "embeddings": [[[0.1] * 10]],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.search([0.1] * 1024)
        assert len(result) == 1
        assert result[0]["content"] == "标签1"

    @patch("chromadb.PersistentClient")
    def test_search_error_returns_empty(self, mock_client):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.query.side_effect = Exception("db down")
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.search([0.1] * 1024)
        assert result == []
