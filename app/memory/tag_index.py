# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 6cf8e118

"""标签嵌入索引 — 用 bge-m3 embedding + cosine 相似度找近邻标签。

替代 TopicTree 分支扩展：不需要等共现数据累积，从第一个标签起就能做最近邻扩展。
每次浅巩固时增量更新——新标签嵌一次入库，后续检索时 cosine 查 top-K。

用法
----
    idx = TagEmbeddingIndex(data_dir)
    idx.update(["记忆", "编码", "人格"])          # 嵌入新标签
    idx.nearest(["记忆"], top_k=5)               → ["编码", "人格", "存储", ...]
    idx.nearest(["记忆", "编码"], top_k=5)       → 合并后去重取 top-K
"""

import json
import logging
import os
import threading
from typing import Optional

import numpy as np

from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)


class TagEmbeddingIndex:
    """标签 → bge-m3 embedding → cosine 最近邻查询。"""

    MAX_NEAREST = 10
    CACHE_FILE = "tag_embeddings.json"

    def __init__(self, data_dir: str, embed_fn: callable | None = None):
        self._path = os.path.join(data_dir, self.CACHE_FILE)
        self._lock = threading.Lock()
        # tag → embedding (list[float], 1024-dim)
        self._embeddings: dict[str, list[float]] = {}
        self._embed_fn = embed_fn  # lazy set via set_embed_fn() if needed
        self._load()

    def set_embed_fn(self, fn: callable):
        """注入嵌入函数（避免循环导入）。"""
        self._embed_fn = fn

    # ── 公开接口 ──────────────────────────────────────────────

    def update(self, tags: list[str]):
        """增量嵌入新标签。已存在的跳过（不重复嵌）。

        嵌入函数应支持批量输入 (list[str] → list[Optional[list[float]]])。
        """
        if not self._embed_fn:
            logger.warning("TagEmbeddingIndex 未注入 embed_fn，跳过更新")
            return

        new_tags = [t for t in tags if t not in self._embeddings]
        if not new_tags:
            return

        logger.debug("嵌入 %d 个新标签...", len(new_tags))
        embeds = self._embed_fn(new_tags)  # local_embed_batch: list[str] → list[Optional[list[float]]]

        with self._lock:
            for tag, emb in zip(new_tags, embeds):
                if emb is not None and len(emb) > 0:
                    self._embeddings[tag] = list(emb)
            self._save()

    def nearest(self, tags: list[str], top_k: int = 5) -> list[str]:
        """返回与给定标签集最相似的 top_k 个标签（排除自身）。

        多个 query tag 时：对每个找最近邻，合并后按平均相似度排序取 top_k。
        """
        if not tags:
            return []

        with self._lock:
            available = {
                t: np.array(e, dtype=np.float32)
                for t, e in self._embeddings.items()
                if t not in tags  # 排除 query 自身
            }
            if not available:
                return []

            tag_vecs = np.stack(list(available.values()))
            tag_names = list(available.keys())

            # 对每个 query tag 计算 cosine 相似度，取平均
            scores = np.zeros(len(tag_names), dtype=np.float32)
            for qt in tags:
                qv = self._embeddings.get(qt)
                if qv is None:
                    continue
                qv_arr = np.array(qv, dtype=np.float32)
                q_norm = np.linalg.norm(qv_arr)
                if q_norm == 0:
                    continue
                # batch cosine: (tag_vecs · qv_arr) / (norms * q_norm)
                tag_norms = np.linalg.norm(tag_vecs, axis=1)
                tag_norms = np.where(tag_norms == 0, 1.0, tag_norms)
                sims = np.dot(tag_vecs, qv_arr) / (tag_norms * q_norm)
                # NaN guard
                sims = np.nan_to_num(sims, nan=0.0)
                scores += sims

            # 取 top_k
            order = np.argsort(scores)[::-1]
            result = []
            for idx in order:
                if scores[idx] <= 0:
                    continue
                result.append(tag_names[idx])
                if len(result) >= top_k:
                    break
            return result

    def size(self) -> int:
        with self._lock:
            return len(self._embeddings)

    # ── 持久化 ────────────────────────────────────────────────

    def _save(self):
        try:
            atomic_write(self._path, {
                "version": 1,
                "embeddings": self._embeddings,
            })
        except Exception:
            logger.exception("标签嵌入索引写入失败")

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self._embeddings = data.get("embeddings", {})
        except Exception:
            self._embeddings = {}
