"""行为模式独立存储。行为模式 = 用户的使用习惯（时间规律、话题序列、固定流程）。"""
import uuid
import logging
import threading
from datetime import datetime

import chromadb
from app.llm.embed import local_embed

logger = logging.getLogger(__name__)


class BehaviorStore:
    """行为模式存储，独立 collection，轻量封装。"""

    def __init__(self, persist_dir: str, collection_name: str = "behavior_patterns"):
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name, embedding_function=None)
        self._write_client = chromadb.PersistentClient(path=persist_dir)
        self._write_collection = self._write_client.get_or_create_collection(collection_name, embedding_function=None)

    def store(self, content: str, confidence: str = "中") -> str:
        """存储一条行为模式。"""
        pattern_id = str(uuid.uuid4())
        embedding = local_embed(content)
        now = datetime.now().isoformat()
        with self._lock:
            self._write_collection.add(
                ids=[pattern_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"confidence": confidence, "created_at": now, "hit_count": 0}],
            )
        return pattern_id

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """语义搜索行为模式。"""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for i, pid in enumerate(results.get("ids", [[]])[0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            items.append({
                "id": pid,
                "content": results["documents"][0][i] if results.get("documents") else "",
                "confidence": meta.get("confidence", "中"),
                "hit_count": meta.get("hit_count", 0),
            })
        return items

    def count(self) -> int:
        with self._lock:
            return self._write_collection.count()

    def list_all(self) -> list[dict]:
        data = self._collection.get(include=["documents", "metadatas"])
        items = []
        for i, pid in enumerate(data.get("ids", [])):
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            items.append({
                "id": pid,
                "content": data["documents"][i] if data.get("documents") else "",
                "confidence": meta.get("confidence", "中"),
            })
        return items
