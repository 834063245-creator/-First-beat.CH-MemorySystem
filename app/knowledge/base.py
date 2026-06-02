#!/usr/bin/env python3
"""LlamaIndex 知识库模块 — 封装导入、检索、列表、清理等接口。

替换手写的 knowledge_importer.py + knowledge_retrieval.py。
所有 public 方法独立 try-except，失败返回空结果不抛异常。
"""

import json
import logging
import os
import time
from datetime import datetime

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.llm.embed import local_embed

logger = logging.getLogger(__name__)

# 文件状态追踪路径（由实例化时传入）


# ─── 自定义 Embedding ───

class ChuhenEmbedding(BaseEmbedding):
    """使用 local_embed() 作为后端的 Embedding 实现。

    local_embed 走 Ollama GPU（bge-m3），
    失败时返回 1024 维零向量兜底，不阻塞链路。
    """

    def _get_text_embedding(self, text: str) -> list[float]:
        try:
            emb = local_embed(text)
            if emb is None:
                return [0.0] * 1024
            return emb
        except Exception:
            return [0.0] * 1024

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


# ─── 知识库主类 ───

class KnowledgeBase:
    """知识库封装。

    基于 LlamaIndex + ChromaDB，提供文档导入、语义检索、条目列表、孤立清理等功能。
    现有 ChromaDB collection 中的旧条目（缺少 LlamaIndex node_info 等字段）兼容保留，
    不删除不迁移，向量检索依然可命中。
    """

    def __init__(self, chroma_dir: str, collection_name: str = "knowledge", *,
                 state_path: str):
        self._state_path = state_path
        self._chroma_dir = chroma_dir
        self._collection_name = collection_name
        self._embed_model = ChuhenEmbedding()
        self._chroma_collection = None
        self._vector_store = None
        self._storage_context = None
        self._index = None
        self._retriever = None

        try:
            self._db = chromadb.PersistentClient(path=chroma_dir)
            try:
                self._chroma_collection = self._db.get_collection(collection_name)
            except Exception:
                self._chroma_collection = self._db.create_collection(collection_name)
            self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
            self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store,
                embed_model=self._embed_model,
            )
        except Exception as e:
            logger.error("KnowledgeBase 初始化失败: %s", e)

    # ── 文件状态追踪 ──

    @staticmethod
    def _load_state(state_path: str) -> dict:
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    @staticmethod
    def _save_state(state: dict, state_path: str):
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _file_fingerprint(path: str) -> tuple:
        try:
            stat = os.stat(path)
            return (stat.st_mtime, stat.st_size)
        except OSError:
            return (0, 0)

    # ── 导入 ──

    def import_file(self, file_path: str, force: bool = False) -> list[str]:
        """导入单个文件到知识库。

        参数:
            file_path: 文件路径
            force: 为 True 时先删除该文件已有条目再导入

        返回:
            导入条目的 ID 列表
        """
        try:
            abspath = os.path.abspath(file_path)
            if not os.path.exists(abspath):
                logger.error("文件不存在: %s", abspath)
                return []

            # 文件状态检查（跳过未变化文件）
            fingerprint = self._file_fingerprint(abspath)
            if not force:
                state = self._load_state(self._state_path)
                prev = state.get(abspath)
                if prev and prev["mtime"] == fingerprint[0] and prev["size"] == fingerprint[1]:
                    logger.debug("跳过(未变化): %s", os.path.basename(abspath))
                    return []

            # 读取文件内容
            content = self._read_file_content(abspath)
            if not content or not content.strip():
                logger.warning("空文件: %s", abspath)
                return []

            # force 模式：先删除旧数据
            if force:
                self._delete_file_data(abspath)

            # 分块
            nodes = self._chunk_content(content, abspath)
            if not nodes:
                return []

            # 写入 ChromaDB
            if self._index is None:
                if self._vector_store is None:
                    logger.error("KnowledgeBase 未正确初始化")
                    return []
                self._index = VectorStoreIndex.from_vector_store(
                    self._vector_store,
                    embed_model=self._embed_model,
                )

            self._index.insert_nodes(nodes)
            node_ids = [node.node_id for node in nodes]
            logger.info("导入 %s: %d 个条目", os.path.basename(abspath), len(node_ids))

            # 更新文件状态
            try:
                state = self._load_state(self._state_path)
                state[abspath] = {"mtime": fingerprint[0], "size": fingerprint[1]}
                self._save_state(state, self._state_path)
            except Exception as e:
                logger.warning("文件状态保存失败: %s", e)

            return node_ids

        except Exception as e:
            logger.error("import_file 失败 (%s): %s", file_path, e)
            return []

    def _read_file_content(self, file_path: str) -> str:
        """读取文件内容，按扩展名选择读取方式。"""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                from llama_index.readers.file import PDFReader
                docs = PDFReader().load_data(file_path=file_path)
                return "\n".join(d.text for d in docs)
            elif ext == ".docx":
                from llama_index.readers.file import DocxReader
                docs = DocxReader().load_data(file_path=file_path)
                return "\n".join(d.text for d in docs)
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        except Exception as e:
            logger.warning("专用读取器失败 (%s)，回退纯文本: %s", ext, e)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    def _chunk_content(self, content: str, source_file: str) -> list[TextNode]:
        """将文本分块并添加 metadata，返回 TextNode 列表。"""
        timestamp = datetime.fromtimestamp(os.path.getmtime(source_file)).isoformat()
        created_at = datetime.now().isoformat()

        try:
            splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)
            doc = Document(text=content)
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception as e:
            logger.warning("SentenceSplitter 失败，使用简易分块: %s", e)
            nodes = self._fallback_chunk(content)

        for i, node in enumerate(nodes):
            node.metadata["source_file"] = os.path.abspath(source_file)
            node.metadata["chunk_index"] = i + 1
            node.metadata["type"] = "chunk"
            node.metadata["timestamp"] = timestamp
            node.metadata["created_at"] = created_at
            node.metadata["summary"] = node.text[:100]

        return nodes

    @staticmethod
    def _fallback_chunk(content: str) -> list[TextNode]:
        """SentenceSplitter 不可用时的简易分块方案：按行聚合，每块约 800 字。"""
        lines = content.split("\n")
        nodes = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 > 800 and current_chunk:
                nodes.append(TextNode(text=current_chunk.strip()))
                current_chunk = line
            else:
                current_chunk += line + "\n"
        if current_chunk.strip():
            nodes.append(TextNode(text=current_chunk.strip()))
        return nodes

    def _delete_file_data(self, source_file: str):
        """删除指定源文件的所有知识库条目。"""
        if self._chroma_collection is None:
            return
        try:
            results = self._chroma_collection.get(where={"source_file": os.path.abspath(source_file)})
            if results["ids"]:
                self._chroma_collection.delete(ids=results["ids"])
                logger.info("删除旧数据: %d 条", len(results["ids"]))
        except Exception as e:
            logger.warning("删除旧数据失败: %s", e)

    # ── 检索 ──

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索知识库。

        参数:
            query: 查询文本
            top_k: 返回 top-k 条

        返回:
            [{id, document, metadata, source, semantic_score}, ...]
        """
        try:
            if self._index is None:
                logger.warning("KnowledgeBase 未初始化，返回空结果")
                return []

            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
            )
            nodes = retriever.retrieve(query)

            results = []
            for n in nodes:
                results.append({
                    "id": n.node.node_id,
                    "document": n.node.text,
                    "metadata": dict(n.node.metadata) if n.node.metadata else {},
                    "source": "knowledge",
                    "semantic_score": float(n.score) if n.score is not None else 0.0,
                })
            return results

        except Exception as e:
            logger.warning("知识库检索失败: %s", e)
            return []

    # ── 列表 ──

    def list_entries(self, page: int = 1, per_page: int = 20) -> dict:
        """返回知识库条目列表（分页，直接查 ChromaDB，不走 LlamaIndex）。

        返回:
            {items: [{id, summary, type, source_file, chapter, created_at}], total, page, per_page}
        """
        try:
            if self._chroma_collection is None:
                return {"items": [], "total": 0, "page": page, "per_page": per_page}

            all_data = self._chroma_collection.get(include=["metadatas", "documents"])
            total = len(all_data["ids"])
            start = (page - 1) * per_page
            end = min(start + per_page, total)

            items = []
            for i in range(start, end):
                meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                doc_text = all_data["documents"][i] if all_data.get("documents") else ""
                items.append({
                    "id": all_data["ids"][i],
                    "summary": meta.get("summary", doc_text[:80]) if doc_text else "",
                    "type": meta.get("type", "chunk"),
                    "source_file": meta.get("source_file", ""),
                    "chapter": meta.get("chapter", ""),
                    "created_at": meta.get("created_at", ""),
                })

            return {"items": items, "total": total, "page": page, "per_page": per_page}

        except Exception as e:
            logger.error("list_entries 失败: %s", e)
            return {"items": [], "total": 0, "page": page, "per_page": per_page}

    # ── 清理 ──

    def clean_orphans(self) -> int:
        """删除源文件已不存在的孤立条目。返回删除数量。"""
        try:
            if self._chroma_collection is None:
                return 0

            all_data = self._chroma_collection.get(include=["metadatas"])
            deleted = 0
            for i, mid in enumerate(all_data["ids"]):
                src = all_data["metadatas"][i].get("source_file", "")
                if src and not os.path.exists(src):
                    try:
                        self._chroma_collection.delete(ids=[mid])
                        deleted += 1
                    except Exception:
                        pass
            logger.info("清理孤立条目: %d 条", deleted)
            return deleted

        except Exception as e:
            logger.error("clean_orphans 失败: %s", e)
            return 0

    # ── 详情 ──

    def get_detail(self, entry_id: str) -> dict | None:
        """获取单条知识库条目的完整详情。"""
        try:
            if self._chroma_collection is None:
                return None
            result = self._chroma_collection.get(
                ids=[entry_id],
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                return None
            meta = dict(result["metadatas"][0]) if result.get("metadatas") else {}
            doc_text = result["documents"][0] if result.get("documents") else ""
            tags_raw = meta.get("tags", "")
            entities_raw = meta.get("entities", "")
            return {
                "id": result["ids"][0],
                "document": doc_text,
                "metadata": meta,
                "source_file": meta.get("source_file", ""),
                "type": meta.get("type", "chunk"),
                "chunk_index": meta.get("chunk_index", 1),
                "chapter": meta.get("chapter", ""),
                "timestamp": meta.get("timestamp", ""),
                "created_at": meta.get("created_at", ""),
                "tags": tags_raw.split(",") if tags_raw else [],
                "entities": entities_raw,
            }
        except Exception as e:
            logger.error("get_detail 失败 (%s): %s", entry_id, e)
            return None

    # ── 删除 ──

    def delete_entry(self, entry_id: str) -> bool:
        """删除单条知识库条目。"""
        try:
            if self._chroma_collection is None:
                return False
            self._chroma_collection.delete(ids=[entry_id])
            logger.info("删除知识库条目: %s", entry_id)
            return True
        except Exception as e:
            logger.error("delete_entry 失败 (%s): %s", entry_id, e)
            return False

    # ── 访问底层 collection ──

    def get_collection(self):
        """返回底层 ChromaDB collection 对象，供 main.py 直接访问。"""
        return self._chroma_collection
