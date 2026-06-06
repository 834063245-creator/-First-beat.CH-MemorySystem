"""测试 app/personality/behavior.py — BehaviorStore 行为模式存储。

覆盖：构造 / store / search / count / list_all。mock ChromaDB 客户端。
"""
import pytest
from unittest.mock import MagicMock, patch


class TestBehaviorStore:
    @patch("chromadb.PersistentClient")
    @patch("app.personality.behavior.local_embed", return_value=[0.1] * 1024)
    def test_store_returns_id(self, mock_embed, mock_chroma_cls):
        from app.personality.behavior import BehaviorStore
        col = MagicMock()
        mock_chroma_cls.return_value.get_or_create_collection.return_value = col
        store = BehaviorStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.store("用户每天凌晨2点写代码", confidence="高")
        assert isinstance(result, str)
        assert len(result) > 0
        # 验证 add 被调用
        col.add.assert_called_once()
        call_args = col.add.call_args
        assert "ids" in call_args[1]
        assert "documents" in call_args[1]
        assert "embeddings" in call_args[1]
        assert call_args[1]["metadatas"][0]["confidence"] == "高"
        assert call_args[1]["metadatas"][0]["hit_count"] == 0

    @patch("chromadb.PersistentClient")
    def test_search_returns_items(self, mock_chroma_cls):
        from app.personality.behavior import BehaviorStore
        col = MagicMock()
        col.query.return_value = {
            "ids": [["b1", "b2"]],
            "documents": [["模式1", "模式2"]],
            "metadatas": [[{"confidence": "高", "hit_count": 3},
                           {"confidence": "中", "hit_count": 1}]],
            "distances": [[0.1, 0.3]],
        }
        store = BehaviorStore(persist_dir="/tmp/fake")
        store._collection = col
        results = store.search([0.1] * 1024, top_k=2)
        assert len(results) == 2
        assert results[0]["content"] == "模式1"
        assert results[0]["confidence"] == "高"
        assert results[0]["hit_count"] == 3

    @patch("chromadb.PersistentClient")
    def test_count(self, mock_chroma_cls):
        from app.personality.behavior import BehaviorStore
        col = MagicMock()
        col.count.return_value = 7
        store = BehaviorStore(persist_dir="/tmp/fake")
        store._collection = col
        assert store.count() == 7

    @patch("chromadb.PersistentClient")
    def test_list_all(self, mock_chroma_cls):
        from app.personality.behavior import BehaviorStore
        col = MagicMock()
        col.get.return_value = {
            "ids": ["b1", "b2"],
            "documents": ["行为A", "行为B"],
            "metadatas": [{"confidence": "高"}, {"confidence": "低"}],
        }
        store = BehaviorStore(persist_dir="/tmp/fake")
        store._collection = col
        items = store.list_all()
        assert len(items) == 2
        assert items[0]["id"] == "b1"
        assert items[1]["content"] == "行为B"

    @patch("chromadb.PersistentClient")
    def test_search_empty_results(self, mock_chroma_cls):
        from app.personality.behavior import BehaviorStore
        col = MagicMock()
        col.query.return_value = {"ids": [[]], "documents": [[]],
                                  "metadatas": [[]], "distances": [[]]}
        store = BehaviorStore(persist_dir="/tmp/fake")
        store._collection = col
        results = store.search([0.1] * 1024)
        assert results == []
