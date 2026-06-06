"""测试 app/memory/chroma.py — ChromaService 核心方法 mock。

覆盖：add_memory / update_memory / list_all / stats / supersede_memory / list_memories
"""
import pytest
from unittest.mock import MagicMock, patch


def _fake_chroma_collection():
    col = MagicMock()
    col.count.return_value = 5
    col.get.return_value = {
        "ids": ["m1", "m2", "m3"],
        "documents": ["doc1", "doc2", "doc3"],
        "metadatas": [
            {"summary": "记忆1", "tags": "测试, 咖啡", "timestamp": 1700000000,
             "hit_count": 10, "source": "chat", "stale": False,
             "emotion_valence_bin": "positive", "emotional_intensity": 2,
             "heat": "warm"},
            {"summary": "记忆2", "tags": "工作, 项目", "timestamp": 1700100000,
             "hit_count": 3, "source": "benchmark", "stale": False, "heat": "hot"},
            {"summary": "旧记忆", "tags": "过时", "timestamp": 1600000000,
             "hit_count": 0, "source": "chat", "stale": True, "heat": "cool"},
        ],
    }
    return col


class TestChromaService:
    @patch("chromadb.PersistentClient")
    def test_init_creates_collection(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="test")
        assert svc._collection is col

    @patch("chromadb.PersistentClient")
    def test_list_all(self, mock_client):
        from app.memory.chroma import ChromaService
        col = _fake_chroma_collection()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        mems = svc.list_all()
        assert len(mems) == 3

    @patch("chromadb.PersistentClient")
    def test_count(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        col.count.return_value = 42
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        assert svc.count() == 42

    @patch("chromadb.PersistentClient")
    def test_stats(self, mock_client):
        from app.memory.chroma import ChromaService
        col = _fake_chroma_collection()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        stats = svc.stats()
        assert isinstance(stats, dict)
        assert "total" in stats

    @patch("chromadb.PersistentClient")
    def test_list_memories(self, mock_client):
        from app.memory.chroma import ChromaService
        col = _fake_chroma_collection()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        result = svc.list_memories(page=1, per_page=10)
        assert result["total"] == 3

    @patch("chromadb.PersistentClient")
    def test_list_memories_with_filter(self, mock_client):
        from app.memory.chroma import ChromaService
        col = _fake_chroma_collection()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        result = svc.list_memories(tag="咖啡")
        assert isinstance(result, dict)

    @patch("chromadb.PersistentClient")
    def test_get_memory_detail(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        col.get.return_value = {
            "ids": ["m1"],
            "documents": ["完整文档"],
            "metadatas": [{"summary": "测试", "tags": "tag1, tag2", "timestamp": 1700000000,
                           "hit_count": 5, "source": "chat", "emotion_valence_bin": "positive"}],
        }
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        detail = svc.get_memory_detail("m1")
        assert detail is not None
        assert detail["summary"] == "测试"

    @patch("chromadb.PersistentClient")
    def test_get_memory_detail_not_found(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        assert svc.get_memory_detail("no_exist") is None

    @patch("chromadb.PersistentClient")
    def test_delete_memory(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        svc.delete_memory("m1")
        col.delete.assert_called_with(ids=["m1"])

    @patch("chromadb.PersistentClient")
    def test_supersede_memory(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        svc.supersede_memory("old_id", "new_id", "测试原因")
        col.update.assert_called()
        meta = col.update.call_args[1]["metadatas"][0]
        assert meta["stale"] is True
        assert meta["superseded_by"] == "new_id"

    @patch("chromadb.PersistentClient")
    def test_increment_hit_count(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        col.get.return_value = {
            "ids": ["m1"],
            "metadatas": [{"hit_count": 5}],
        }
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        svc.increment_hit_count("m1")
        assert col.update.called

    @patch("chromadb.PersistentClient")
    def test_increment_hit_nonexistent(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        col.get.return_value = {"ids": [], "metadatas": []}
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        svc.increment_hit_count("no_exist")  # 不崩溃

    @patch("chromadb.PersistentClient")
    def test_update_memory(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        svc.update_memory("m1", summary="新摘要", tags=["新标签"], embedding=[0.1] * 1024)
        assert col.update.called

    @patch("chromadb.PersistentClient")
    def test_archive_topic_cluster(self, mock_client):
        from app.memory.chroma import ChromaService
        col = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = col
        svc = ChromaService(persist_dir="/tmp/fake", collection_name="t")
        svc._collection = col
        svc.archive_topic_cluster("测试话题", ["m1", "m2"])
        assert col.update.called
