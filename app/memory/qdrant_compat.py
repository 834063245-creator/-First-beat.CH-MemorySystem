# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 44da844b

"""旧 ChromaDB Collection API 兼容层 — Qdrant 翻译适配器。

从 qdrant.py 中提取，消除模块内循环依赖。
QdrantService 通过 self._collection 暴露此适配器给 pipeline/context/dispatch 使用。
"""
import logging

from qdrant_client import models

from app.memory.qdrant_filter import _translate_filter

logger = logging.getLogger(__name__)


def _build_items_from_points(points: list) -> list[dict]:
    """Qdrant scroll/retrieve 结果 → 统一 item 格式。"""
    items = []
    for pt in points:
        payload = pt.payload or {}
        items.append({
            "id": pt.id,
            "document": payload.get("document"),
            "metadata": dict(payload),
        })
    return items


class _QdrantCollectionCompat:
    """QdrantService._collection 暴露的集合操作接口。

    提供 query/get/update/count 等集合级 API（翻译为底层 Qdrant 调用），
    供 pipeline.py / context.py / dispatch.py 统一使用。
    """

    def __init__(self, service: 'QdrantService'):
        self._svc = service

    def query(self, query_embeddings, n_results, where=None, include=None):
        """旧 ChromaDB col.query() → Qdrant search()（本地索引加速）。

        返回格式（兼容 ChromaDB 旧代码）:
          {"ids": [[id1,id2,...]], "documents": [["doc1","doc2",...]],
           "metadatas": [[{...},{...},...]], "distances": [[0.1,0.2,...]]}
        """
        qf = _translate_filter(where) if where else None
        include_docs = include is None or "documents" in include
        include_meta = include is None or "metadatas" in include

        ids_list, docs_list, metas_list, dists_list = [], [], [], []
        for q_emb in query_embeddings:
            try:
                results = self._svc._search_with_index(
                    query_vector=q_emb,
                    query_filter=qf,
                    limit=n_results,
                    with_payload=include_meta,
                )
            except Exception as exc:
                logger.warning("Qdrant search 失败: %s", exc)
                results = []

            batch_ids, batch_docs, batch_metas, batch_dists = [], [], [], []
            for pt in results:
                batch_ids.append(pt.id)
                payload = pt.payload or {}
                batch_docs.append(payload.get("document", "") if include_docs else "")
                batch_metas.append(dict(payload) if include_meta else {})
                batch_dists.append(1.0 - pt.score)

            ids_list.append(batch_ids)
            docs_list.append(batch_docs)
            metas_list.append(batch_metas)
            dists_list.append(batch_dists)

        return {
            "ids": ids_list,
            "documents": docs_list,
            "metadatas": metas_list,
            "distances": dists_list,
        }

    def get(self, ids=None, where=None, include=None, limit=None):
        """旧 ChromaDB col.get() → Qdrant retrieve()/scroll()。

        返回格式（兼容 ChromaDB 旧代码）:
          {"ids": [...], "documents": [...], "metadatas": [...]}
        """
        include_docs = include is None or "documents" in include
        include_meta = include is None or "metadatas" in include

        if ids:
            id_list = ids if isinstance(ids, list) else [ids]
            pts = self._svc._client.retrieve(
                collection_name=self._svc._collection_name,
                ids=id_list,
                with_payload=include_meta,
            )
        else:
            qf = _translate_filter(where) if where else None
            pts, _ = self._svc._scroll_with_index(
                scroll_filter=qf,
                with_payload=include_meta,
                limit=limit or 1000,
            )

        result_ids, docs, metas = [], [], []
        for pt in pts:
            result_ids.append(pt.id)
            payload = pt.payload or {}
            docs.append(payload.get("document", "") if include_docs else "")
            metas.append(dict(payload) if include_meta else {})

        return {
            "ids": result_ids,
            "documents": docs,
            "metadatas": metas,
        }

    def update(self, ids, metadatas=None, documents=None, embeddings=None):
        """旧 ChromaDB col.update() 格式 → Qdrant set_payload() + update_vectors()。"""
        id_list = [ids] if isinstance(ids, str) else ids
        for mid in id_list:
            payload_updates = {}
            if metadatas is not None:
                idx = id_list.index(mid) if isinstance(metadatas, list) else 0
                meta = metadatas[idx] if isinstance(metadatas, list) else metadatas
                if isinstance(meta, dict):
                    payload_updates.update(meta)
            if documents is not None:
                idx = id_list.index(mid) if isinstance(documents, list) else 0
                doc = documents[idx] if isinstance(documents, list) else documents
                payload_updates["document"] = doc
            if payload_updates:
                self._svc._client.set_payload(
                    collection_name=self._svc._collection_name,
                    payload=payload_updates,
                    points=[mid],
                )
            if embeddings is not None:
                idx = id_list.index(mid) if isinstance(embeddings, list) else 0
                emb = embeddings[idx] if isinstance(embeddings, list) else embeddings
                self._svc._client.update_vectors(
                    collection_name=self._svc._collection_name,
                    points=[models.PointVectors(id=mid, vector=emb)],
                )
        self._svc._invalidate_list_all_cache()

    def count(self):
        return self._svc._client.count(
            collection_name=self._svc._collection_name).count
