"""话题亲和图 — 标签级关联网络，替代记忆ID级共现。

数据来自对话中标签的先后出现关系，不依赖检索管线的命中记录。
更新时机：DMN浅巩固时增量维护。
查询用途：检索时扩展相关话题。
"""

import json
import logging
import os
import threading
from collections import defaultdict

from app.tools.atomic import atomic_write


logger = logging.getLogger(__name__)


class TopicAffinity:
    """标签亲和图。记录哪些标签经常在同一段对话中出现。"""

    MIN_AFFINITY = 2        # 最少共现次数
    MAX_ENTRIES = 5000      # 矩阵上限

    def __init__(self, data_dir: str):
        self._path = os.path.join(data_dir, "topic_affinity.json")
        self._lock = threading.Lock()
        self._matrix: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    self._matrix = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._matrix = {}

    def _save(self):
        with self._lock:
            atomic_write(self._path, self._matrix)

    def update(self, tags: list[str]):
        """从同一段对话的一组标签增量更新亲和图。"""
        if len(tags) < 2:
            return
        with self._lock:
            for i, t1 in enumerate(tags):
                if t1 not in self._matrix:
                    self._matrix[t1] = {}
                for t2 in tags[i + 1:]:
                    if t2 == t1:
                        continue
                    self._matrix[t1][t2] = self._matrix[t1].get(t2, 0) + 1
                    if t2 not in self._matrix:
                        self._matrix[t2] = {}
                    self._matrix[t2][t1] = self._matrix[t2].get(t1, 0) + 1
            if len(self._matrix) > self.MAX_ENTRIES:
                self._prune()
        self._save()

    def expand(self, tags: list[str], top_k: int = 5) -> list[tuple[str, int]]:
        """给定一组标签，返回亲和最强的其他标签。"""
        scores: dict[str, int] = defaultdict(int)
        with self._lock:
            for t in tags:
                related = self._matrix.get(t, {})
                for rt, cnt in related.items():
                    if rt not in tags:
                        scores[rt] += cnt
        sorted_tags = sorted(scores.items(), key=lambda x: -x[1])
        return [(t, s) for t, s in sorted_tags if s >= self.MIN_AFFINITY][:top_k]

    def get_related_tags(self, tag: str, top_k: int = 5) -> list[tuple[str, int]]:
        """单标签查询。"""
        with self._lock:
            rels = self._matrix.get(tag, {})
            sorted_rels = sorted(rels.items(), key=lambda x: -x[1])
            return [(t, c) for t, c in sorted_rels if c >= self.MIN_AFFINITY][:top_k]

    def _prune(self):
        scored = [(t, sum(rels.values())) for t, rels in self._matrix.items()]
        scored.sort(key=lambda x: -x[1])
        keep = {t for t, _ in scored[:self.MAX_ENTRIES // 2]}
        self._matrix = {t: rels for t, rels in self._matrix.items() if t in keep}
