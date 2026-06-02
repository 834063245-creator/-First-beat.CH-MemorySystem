"""人格库 — PersonalityStore，独立 ChromaDB collection。

使用 bge 做 embedding（与记忆库同一模型），独立 client 实例。
collection 名由 config.PERSONALITY_COLLECTION 控制。

注意：rerank_tags 在检索管线中被调用（多线程并发），
increment_hit 内部依赖 ChromaDB 的 get+update 原子性。
ChromaDB 客户端自身有锁，当前无需额外锁。
"""
import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime
from typing import Optional

import chromadb
import numpy as np

from app.config.settings import PERSONALITY_CHROMA_DIR, PERSONALITY_COLLECTION, PERSONALITY_DEDUP_THRESHOLD
from metadata import extract_topics

logger = logging.getLogger(__name__)


class PersonalityStore:
    """人格标签存储，独立 collection，线程安全。"""

    def __init__(self, persist_dir: Optional[str] = None):
        chroma_path = persist_dir or PERSONALITY_CHROMA_DIR
        self._read_client = chromadb.PersistentClient(path=chroma_path)
        self._write_client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._read_client.get_or_create_collection(
            name=PERSONALITY_COLLECTION, embedding_function=None,
        )
        self._write_coll = self._write_client.get_or_create_collection(
            name=PERSONALITY_COLLECTION, embedding_function=None,
        )
        self._lock = threading.Lock()

    def store_tag(
        self, content: str, embedding: list[float], tag_type: str = "行为模式",
        confidence: str = "低", hit_count: int = 0, source: str = "user",
    ) -> str:
        """存储一条人格标签。"""
        tag_id = str(uuid.uuid4())
        now = datetime.now().timestamp()
        self._write_coll.add(
            ids=[tag_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[{
                "type": tag_type,
                "confidence": confidence,
                "hit_count": hit_count,
                "last_hit_time": now if hit_count > 0 else 0,
                "created_at": now,
                "outdated": False,
                "source": source,
            }],
        )
        return tag_id

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """语义搜索，返回带相似度和 embedding 的结果。"""
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as exc:
            logger.warning("人格库搜索失败: %s", exc)
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        embeddings_list = results.get("embeddings", [None])[0] if results.get("embeddings") is not None else [None]

        items = []
        for i, tag_id in enumerate(results["ids"][0]):
            meta = dict(results["metadatas"][0][i]) if results["metadatas"] else {}
            items.append({
                "id": tag_id,
                "content": results["documents"][0][i] if results["documents"] else "",
                "metadata": meta,
                "distance": results["distances"][0][i] if results["distances"] else 0,
                "embedding": embeddings_list[i] if embeddings_list is not None and i < len(embeddings_list) else None,
            })
        return items[:top_k]

    def increment_hit(self, tag_id: str):
        """命中计数 +1，更新 last_hit_time。"""
        with self._lock:
            result = self._write_coll.get(ids=[tag_id], include=["metadatas"])
            if not result["ids"]:
                return
            meta = dict(result["metadatas"][0])
            meta["hit_count"] = meta.get("hit_count", 0) + 1
            meta["last_hit_time"] = datetime.now().timestamp()
            self._write_coll.update(ids=[tag_id], metadatas=[meta])

    def get_tag(self, tag_id: str) -> Optional[dict]:
        """单条人格标签详情。"""
        result = self._collection.get(ids=[tag_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return {
            "id": tag_id,
            "content": result["documents"][0],
            "type": meta.get("type", "行为模式"),
            "confidence": meta.get("confidence", "低"),
            "outdated": meta.get("outdated", False),
            "hit_count": meta.get("hit_count", 0),
            "last_hit_time": meta.get("last_hit_time", 0),
            "created_at": meta.get("created_at", 0),
        }

    def update_tag(self, tag_id, content=None, tag_type=None, confidence=None):
        """更新标签内容或置信度。"""
        with self._lock:
            result = self._write_coll.get(ids=[tag_id], include=["documents", "metadatas"])
            if not result["ids"]:
                return
            meta = dict(result["metadatas"][0])
            if content:
                from app.llm.embed import local_embed
                embedding = local_embed(content)
                meta["content"] = content
                self._write_coll.update(ids=[tag_id], documents=[content], embeddings=[embedding], metadatas=[meta])
            else:
                if tag_type:
                    meta["type"] = tag_type
                if confidence:
                    meta["confidence"] = confidence
                self._write_coll.update(ids=[tag_id], metadatas=[meta])

    def mark_outdated(self, tag_id):
        """标记画像为过时。"""
        result = self._write_coll.get(ids=[tag_id], include=["metadatas"])
        if not result["ids"]:
            return
        meta = dict(result["metadatas"][0])
        meta["outdated"] = True
        self._write_coll.update(ids=[tag_id], metadatas=[meta])

    def list_tags(self, page: int = 1, page_size: int = 20,
                  sort: str = "created_at", order: str = "desc",
                  min_hits: int = 0, source: Optional[str] = None) -> dict:
        """分页列表，支持排序和筛选。source 不为 None 时过滤。"""
        result = self._collection.get(include=["documents", "metadatas"])
        items = []
        for i, tag_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            hc = meta.get("hit_count", 0)
            if min_hits > 0 and hc < min_hits:
                continue
            if source is not None and meta.get("source", "user") != source:
                continue
            items.append({
                "id": tag_id,
                "content": result["documents"][i],
                "type": meta.get("type", "行为模式"),
                "confidence": meta.get("confidence", "低"),
                "source": meta.get("source", "user"),
                "hit_count": hc,
                "last_hit_time": meta.get("last_hit_time", 0),
                "created_at": meta.get("created_at", 0),
            })
        # 排序
        rev = order != "asc"
        if sort == "hit_count":
            items.sort(key=lambda x: x["hit_count"], reverse=rev)
        elif sort == "last_hit_time":
            items.sort(key=lambda x: x["last_hit_time"], reverse=rev)
        else:
            items.sort(key=lambda x: x["created_at"], reverse=rev)
        # 分页
        total = len(items)
        offset = (page - 1) * page_size
        items = items[offset:offset + page_size]
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def get_user_tags(self, top_k=5):
        """取用户人格标签（用于【我对你的了解】）。"""
        return self.list_tags(page=1, page_size=top_k, source="user")

    def get_ai_tags(self, top_k=5):
        """取 AI 自我模型标签（用于【我自己的表达习惯】）。"""
        return self.list_tags(page=1, page_size=top_k, source="ai")

    def delete_tag(self, tag_id: str):
        """删除人格标签。"""
        self._write_coll.delete(ids=[tag_id])

    def get_count(self) -> int:
        """人格标签总数。"""
        return self._collection.count()

    def rerank_tags(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """人格标签重排序：时间窗口优先 + 话题匹配 + confidence 加成 + 多样性去重 + 选中即命中。

        search(top_k=15) → 7天内命中排序 → 话题加分 → confidence加成 → 去重 → increment_hit → 返回含元数据。
        """
        results = self.search(query_embedding, top_k=15)
        if not results:
            return []

        # 过滤掉过时标签
        results = [r for r in results if not (r.get("metadata") or {}).get("outdated", False)]

        now = datetime.now().timestamp()
        for r in results:
            meta = r.get("metadata") or {}
            last_hit = meta.get("last_hit_time", 0) or 0
            hc = meta.get("hit_count", 0) or 0
            recency_days = (now - last_hit) / 86400 if last_hit > 0 else 999
            r["_score"] = hc if recency_days <= 7 else 0

        # confidence 加成
        for r in results:
            meta = r.get("metadata") or {}
            conf = meta.get("confidence", "低")
            if conf == "高":
                r["_score"] += 1.0
            elif conf == "中":
                r["_score"] += 0.3

        # 话题匹配加分
        query_topics = extract_topics(query)
        if query_topics:
            for r in results:
                tag_topics = extract_topics(r.get("content", ""))
                if any(t in query_topics for t in tag_topics):
                    r["_score"] += 0.5

        # 按得分降序
        results.sort(key=lambda x: x["_score"], reverse=True)

        # 多样性去重（cosine > 0.85 丢弃）
        selected = []
        for r in results:
            emb = r.get("embedding")
            duplicate = False
            if emb is not None:
                emb_a = np.asarray(emb).flatten()
                for s in selected:
                    s_emb = s.get("embedding")
                    if s_emb is None:
                        continue
                    s_a = np.asarray(s_emb).flatten()
                    sim = float(np.dot(emb_a, s_a) / (np.linalg.norm(emb_a) * np.linalg.norm(s_a) + 1e-10))
                    if sim >= PERSONALITY_DEDUP_THRESHOLD:
                        duplicate = True
                        break
            if not duplicate:
                selected.append(r)
                if len(selected) >= top_k:
                    break

        # 选中即命中 + 返回含元数据
        result_list = []
        for r in selected[:top_k]:
            try:
                self.increment_hit(r["id"])
            except Exception:
                pass
            meta = r.get("metadata") or {}
            result_list.append({
                "content": r.get("content", ""),
                "type": meta.get("type", "行为模式"),
                "confidence": meta.get("confidence", "低"),
                "source": meta.get("source", "user"),
            })
        return result_list
