"""词→记忆ID映射。启动时从所有记忆的summary构建，支持增量更新。

线程安全（threading.Lock）。
删除时同步清理。
"""

import logging
import threading

import jieba

logger = logging.getLogger(__name__)


class InvertedIndex:
    """词→记忆ID映射。启动时从所有记忆的summary构建，支持增量更新。

    线程安全（threading.Lock）。
    删除时同步清理。
    """

    def __init__(self):
        self._index: dict[str, set[str]] = {}
        self._exact_entities: dict[str, set[str]] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def _tokenize(self, text: str) -> list[str]:
        """jieba分词，滤掉长度<2的词。"""
        words = jieba.cut(text)
        return [w.strip() for w in words if len(w.strip()) >= 2]

    def build(self, summaries: list[tuple[str, str]]):
        """从 [(memory_id, summary), ...] 构建索引。"""
        new_index: dict[str, set[str]] = {}
        new_exact: dict[str, set[str]] = {}
        for memory_id, summary in summaries:
            words = self._tokenize(summary)
            for w in words:
                if w not in new_index:
                    new_index[w] = set()
                new_index[w].add(memory_id)
                if w not in new_exact:
                    new_exact[w] = set()
                new_exact[w].add(memory_id)
        with self._lock:
            self._index = new_index
            self._exact_entities = new_exact

    def query(self, keywords: list[str], min_match: int = 2) -> list[str]:
        """返回包含 ≥min_match 个关键词的记忆ID列表。

        先取交集（AND），如果结果<3条则退化为OR + 按匹配数排序。
        """
        if not keywords:
            return []
        with self._lock:
            key_sets = []
            for kw in keywords:
                if kw in self._index:
                    key_sets.append(self._index[kw])
            if not key_sets:
                return []

            and_result = set.intersection(*key_sets) if len(key_sets) > 1 else key_sets[0]

            if len(and_result) >= 3:
                return list(and_result)

            # OR退化: 收集所有匹配ID，按匹配数排序
            counts: dict[str, int] = {}
            for s in key_sets:
                for mid in s:
                    counts[mid] = counts.get(mid, 0) + 1
            result = [mid for mid, cnt in counts.items() if cnt >= min_match]
            result.sort(key=lambda x: counts[x], reverse=True)
            return result

    def get_exact(self, word: str) -> set[str]:
        """精确匹配：返回包含该词的记忆ID（用于实体名精确命中）。"""
        with self._lock:
            result = self._index.get(word, set())
            extra = self._exact_entities.get(word, set())
            if extra:
                result = result | extra
            return result

    def add(self, memory_id: str, summary: str):
        """增量更新：将一条新记忆的分词结果加入索引。"""
        words = self._tokenize(summary)
        with self._lock:
            for w in words:
                if w not in self._index:
                    self._index[w] = set()
                self._index[w].add(memory_id)
                if w not in self._exact_entities:
                    self._exact_entities[w] = set()
                self._exact_entities[w].add(memory_id)

    # ── Tag 倒排索引（独立于 summary 索引，用于替代 $contains 扫描） ──

    def build_tags(self, tag_entries: list[tuple[str, str]]):
        """从 [(memory_id, tags_str), ...] 构建标签索引。"""
        new_idx: dict[str, set[str]] = {}
        for mid, tags_str in tag_entries:
            if not tags_str:
                continue
            for tag in tags_str.split(","):
                tag = tag.strip()
                if len(tag) >= 2:
                    if tag not in new_idx:
                        new_idx[tag] = set()
                    new_idx[tag].add(mid)
        self._tag_index = new_idx

    def query_tags(self, tags: list[str]) -> set[str]:
        """标签精确匹配：返回包含 ≥1 个标签的记忆 ID。"""
        if not tags:
            return set()
        result: set[str] = set()
        for t in tags:
            ids = self._tag_index.get(t, set())
            result.update(ids)
        return result

    def remove(self, memory_id: str):
        """删除记忆时同步清理。"""
        with self._lock:
            for w in list(self._index.keys()):
                self._index[w].discard(memory_id)
                if not self._index[w]:
                    del self._index[w]
            for w in list(self._exact_entities.keys()):
                self._exact_entities[w].discard(memory_id)
                if not self._exact_entities[w]:
                    del self._exact_entities[w]
