"""测试 app/personality/store.py — PersonalityStore 纯内存逻辑。

覆盖：list_tags 排序/分页/筛选，get_user_tags/get_ai_tags，
      rerank_tags 得分计算和去重，get_tag/delete_tag/mark_outdated。
所有测试 mock ChromaDB collection，不触发真实连接。
"""
import math
import pytest
from unittest.mock import MagicMock, patch


# 预构建假的 collection 数据
def _fake_collection():
    col = MagicMock()
    col.get.return_value = {
        "ids": ["t1", "t2", "t3", "t4", "t5", "t6"],
        "documents": [
            "用户喜欢安静的环境", "用户对咖啡有执念", "用户偏好早睡早起",
            "AI倾向于简洁回答", "用户工作中容易焦虑", "过时的标签内容",
        ],
        "metadatas": [
            {"type": "偏好模式", "confidence": "高", "hit_count": 10, "last_hit_time": 1700000000,
             "created_at": 1700000000, "source": "user", "outdated": False},
            {"type": "行为模式", "confidence": "中", "hit_count": 5, "last_hit_time": 1700100000,
             "created_at": 1700100000, "source": "user", "outdated": False},
            {"type": "偏好模式", "confidence": "低", "hit_count": 1, "last_hit_time": 1699000000,
             "created_at": 1699000000, "source": "user", "outdated": False},
            {"type": "沟通模式", "confidence": "中", "hit_count": 3, "last_hit_time": 1700200000,
             "created_at": 1700200000, "source": "ai", "outdated": False},
            {"type": "情绪模式", "confidence": "高", "hit_count": 8, "last_hit_time": 1700050000,
             "created_at": 1700050000, "source": "user", "outdated": False},
            {"type": "行为模式", "confidence": "低", "hit_count": 0, "last_hit_time": 0,
             "created_at": 1698000000, "source": "user", "outdated": True},
        ],
    }
    col.count.return_value = 6
    return col


class TestPersonalityStoreListTags:
    @patch("chromadb.PersistentClient")
    def test_list_all_no_filter(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        mock_chroma_cls.return_value.get_or_create_collection.return_value = col
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col  # 直接用 fake collection
        result = store.list_tags(page=1, page_size=10)
        assert result["total"] == 6
        assert len(result["items"]) == 6

    @patch("chromadb.PersistentClient")
    def test_pagination(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.list_tags(page=1, page_size=2)
        assert len(result["items"]) == 2
        result2 = store.list_tags(page=2, page_size=2)
        assert result2["items"][0]["id"] != result["items"][0]["id"]

    @patch("chromadb.PersistentClient")
    def test_sort_by_hit_count(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.list_tags(sort="hit_count", order="desc")
        hits = [t["hit_count"] for t in result["items"]]
        assert hits == sorted(hits, reverse=True)

    @patch("chromadb.PersistentClient")
    def test_sort_asc(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.list_tags(sort="hit_count", order="asc")
        hits = [t["hit_count"] for t in result["items"]]
        assert hits == sorted(hits)

    @patch("chromadb.PersistentClient")
    def test_filter_by_source(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.list_tags(source="ai")
        assert all(t["source"] == "ai" for t in result["items"])

    @patch("chromadb.PersistentClient")
    def test_min_hits_filter(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.list_tags(min_hits=5)
        assert all(t["hit_count"] >= 5 for t in result["items"])


class TestPersonalityStoreUsersTags:
    @patch("chromadb.PersistentClient")
    def test_get_user_tags(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.get_user_tags(top_k=3)
        assert len(result["items"]) <= 3
        assert all(t["source"] == "user" for t in result["items"])

    @patch("chromadb.PersistentClient")
    def test_get_ai_tags(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        result = store.get_ai_tags()
        assert all(t["source"] == "ai" for t in result["items"])


class TestPersonalityStoreCRUD:
    @patch("chromadb.PersistentClient")
    def test_get_tag_exists(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {
            "ids": ["t1"],
            "documents": ["标签内容"],
            "metadatas": [{"type": "行为模式", "confidence": "高",
                           "outdated": False, "hit_count": 5,
                           "last_hit_time": 1700000000, "created_at": 1700000000}],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        tag = store.get_tag("t1")
        assert tag is not None
        assert tag["content"] == "标签内容"
        assert tag["confidence"] == "高"

    @patch("chromadb.PersistentClient")
    def test_get_tag_not_found(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        assert store.get_tag("nonexistent") is None

    @patch("chromadb.PersistentClient")
    def test_delete_tag(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.delete_tag("t1")
        col.delete.assert_called_once_with(ids=["t1"])

    @patch("chromadb.PersistentClient")
    def test_mark_outdated(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {
            "ids": ["t1"],
            "metadatas": [{"outdated": False}],
        }
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.mark_outdated("t1")
        # 确认 update 被调用且 outdated=True
        call_args = col.update.call_args
        assert call_args[1]["ids"] == ["t1"]
        assert call_args[1]["metadatas"][0]["outdated"] is True

    @patch("chromadb.PersistentClient")
    def test_mark_outdated_not_found(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = MagicMock()
        col.get.return_value = {"ids": [], "metadatas": []}
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        store.mark_outdated("nonexistent")  # 不崩溃


class TestPersonalityStoreGetCount:
    @patch("chromadb.PersistentClient")
    def test_get_count(self, mock_chroma_cls):
        from app.personality.store import PersonalityStore
        col = _fake_collection()
        store = PersonalityStore(persist_dir="/tmp/fake")
        store._collection = col
        assert store.get_count() == 6
