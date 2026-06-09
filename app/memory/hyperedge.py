"""超边索引 — 轻量超图，保留实体在多实体上下文中的完整共现关系。

区别于 EntityPairTracker（只记录两两共现），超边索引维护"哪些实体在同一段
对话中同时出现"的完整上下文。用于 weave_context 的故事线聚合，替代
entities[:3] 硬截断。

数据格式：
{
    "entity_index": {
        "Alice": {
            "co_entities": {"Bob": 3, "Charlie": 1},
            "hyper_edges": [0, 1],           # 超边 ID 列表
            "memory_ids": ["id1", "id2"]
        }
    },
    "hyper_edges": [
        {"entities": ["Alice", "Bob", "PostgreSQL"], "memory_ids": ["id1"]},
        ...
    ]
}

线程安全，纯本地文件存储。
"""

import json
import logging
import os
import threading
from collections import defaultdict

from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)


class HyperEdgeIndex:
    """超边索引。记录、查询，线程安全。"""

    EXPAND_TOP_K = 10       # 展开时每实体最多返回的共现实体数
    MAX_EDGES = 10000       # 超边总数上限（触发裁剪）

    def __init__(self, file_path: str):
        self._file = file_path
        self._lock = threading.Lock()
        self._entity_index: dict[str, dict] = {}
        self._hyper_edges: list[dict] = []
        self._next_edge_id = 0
        self._ensure_file()
        self._load()

    def _ensure_file(self):
        parent = os.path.dirname(self._file)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self._file):
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump({"entity_index": {}, "hyper_edges": []}, f)

    def _load(self):
        with self._lock:
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entity_index = data.get("entity_index", {})
                self._hyper_edges = data.get("hyper_edges", [])
                self._next_edge_id = len(self._hyper_edges)
            except (json.JSONDecodeError, OSError):
                self._entity_index = {}
                self._hyper_edges = []
                self._next_edge_id = 0

    def _save_nolock(self):
        """Caller must hold self._lock."""
        from app.tools.atomic import atomic_write
        atomic_write(self._file, {
            "entity_index": self._entity_index,
            "hyper_edges": self._hyper_edges,
        })

    # ─── 公开 API ───────────────────────────────────────────────

    def record(self, entities: list[str], memory_id: str):
        """记录一组实体在同一段对话中共现。

        为这组实体创建一条超边，更新 entity_index。
        """
        entities = list(set(e for e in entities if isinstance(e, str) and len(e) >= 2))
        if len(entities) < 2:
            return

        with self._lock:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
            edge = {
                "entities": sorted(entities),
                "memory_ids": [memory_id],
            }
            self._hyper_edges.append(edge)

            # 更新 entity_index
            for i, e1 in enumerate(entities):
                if e1 not in self._entity_index:
                    self._entity_index[e1] = {
                        "co_entities": {},
                        "hyper_edges": [],
                        "memory_ids": [],
                    }
                idx = self._entity_index[e1]
                idx["hyper_edges"].append(edge_id)
                if memory_id not in idx["memory_ids"]:
                    idx["memory_ids"].append(memory_id)

                for e2 in entities[i + 1:]:
                    idx["co_entities"][e2] = idx["co_entities"].get(e2, 0) + 1

            # 同样更新 e2 对 e1 的共现
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if e2 not in self._entity_index:
                        self._entity_index[e2] = {
                            "co_entities": {},
                            "hyper_edges": [],
                            "memory_ids": [],
                        }
                    idx2 = self._entity_index[e2]
                    idx2["co_entities"][e1] = idx2["co_entities"].get(e1, 0) + 1
                    idx2["hyper_edges"].append(edge_id)
                    if memory_id not in idx2["memory_ids"]:
                        idx2["memory_ids"].append(memory_id)

            # 裁剪
            if len(self._hyper_edges) > self.MAX_EDGES:
                self._prune()

            self._save_nolock()

    def expand(self, entity_names: list[str], top_k: int = None) -> dict[str, int]:
        """给定一批实体名，展开超边，返回 {related_entity: total_weight}。

        权重 = 共现次数，只返回不在输入列表中的实体。
        """
        if top_k is None:
            top_k = self.EXPAND_TOP_K
        if not entity_names:
            return {}

        with self._lock:
            scores: dict[str, int] = defaultdict(int)
            for ename in entity_names:
                idx = self._entity_index.get(ename, {})
                for co_entity, cnt in idx.get("co_entities", {}).items():
                    if co_entity not in entity_names:
                        scores[co_entity] += cnt

            sorted_items = sorted(scores.items(), key=lambda x: -x[1])
            return dict(sorted_items[:top_k])

    def get_memory_ids(self, entity_names: list[str], max_memories: int = 50) -> list[str]:
        """给定一批实体名，收集所有超边关联的记忆 ID，按出现次数降序。"""
        if not entity_names:
            return []

        with self._lock:
            scored: dict[str, int] = defaultdict(int)
            seen_edges: set[int] = set()
            for ename in entity_names:
                idx = self._entity_index.get(ename, {})
                for edge_id in idx.get("hyper_edges", []):
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)
                    if edge_id < len(self._hyper_edges):
                        for mid in self._hyper_edges[edge_id].get("memory_ids", []):
                            scored[mid] = scored.get(mid, 0) + 1

            sorted_ids = sorted(scored.items(), key=lambda x: -x[1])
            return [mid for mid, _ in sorted_ids[:max_memories]]

    def cluster_key(self, entities: list[str], existing_groups: list[set[str]],
                    min_overlap: int = 2) -> int | None:
        """为给定的实体集合找到最佳匹配的已有分组。

        返回 best_group_index，或 None（没有足够重叠的分组）。
        用于 weave_context 的故事线聚合：替代 entities[:3] 硬截断。

        existing_groups: 已有的实体组列表（每个组是一个 set）。
        min_overlap: 至少共享多少个实体才算同一组。
        """
        if not entities:
            return None
        entity_set = set(entities)
        best_idx = None
        best_overlap = 0
        for i, group in enumerate(existing_groups):
            overlap = len(entity_set & group)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        return best_idx if best_overlap >= min_overlap else None

    def cluster_entities(self, entities: list[str],
                         min_overlap: int = 2) -> set[str]:
        """展开实体集合：通过超边找到所有与当前集合共享 ≥ min_overlap 个实体的
        相关实体，返回扩展后的实体集合。

        用于 weave_context 中为一个记忆生成更丰富的分组 key。
        """
        if not entities:
            return set()

        with self._lock:
            entity_set = set(entities)
            # 通过超边展开：对于每个实体，找到它参与的超边
            # 如果某条超边包含当前实体集合中的 ≥ min_overlap 个实体，
            # 则将该超边的所有实体加入结果
            related_edges: set[int] = set()
            for ename in entities:
                idx = self._entity_index.get(ename, {})
                for edge_id in idx.get("hyper_edges", []):
                    related_edges.add(edge_id)

            result = set(entities)
            for edge_id in related_edges:
                if edge_id < len(self._hyper_edges):
                    edge_entities = set(self._hyper_edges[edge_id]["entities"])
                    if len(entity_set & edge_entities) >= min_overlap:
                        result |= edge_entities

            return result

    def remove_memory(self, memory_id: str):
        """删除记忆时同步清理超边索引。"""
        with self._lock:
            # 从超边中移除 memory_id
            for edge in self._hyper_edges:
                mids = edge.get("memory_ids", [])
                if memory_id in mids:
                    mids.remove(memory_id)

            # 从 entity_index 中移除 memory_id
            for idx in self._entity_index.values():
                mids = idx.get("memory_ids", [])
                if memory_id in mids:
                    mids.remove(memory_id)

            # 清理空的超边和实体索引条目
            self._hyper_edges = [e for e in self._hyper_edges if e.get("memory_ids")]
            empty_entities = [e for e, idx in self._entity_index.items()
                              if not idx.get("memory_ids")]
            for e in empty_entities:
                del self._entity_index[e]

            self._next_edge_id = len(self._hyper_edges)
            self._save_nolock()

    def _prune(self):
        """裁剪最老的超边，保留最近一半。"""
        keep = self.MAX_EDGES // 2
        removed = self._hyper_edges[:len(self._hyper_edges) - keep]
        kept = self._hyper_edges[len(self._hyper_edges) - keep:]

        # 重建 entity_index
        new_index: dict[str, dict] = {}
        for edge_id, edge in enumerate(kept):
            for e in edge["entities"]:
                if e not in new_index:
                    new_index[e] = {"co_entities": {}, "hyper_edges": [], "memory_ids": []}
                new_index[e]["hyper_edges"].append(edge_id)
                for mid in edge.get("memory_ids", []):
                    if mid not in new_index[e]["memory_ids"]:
                        new_index[e]["memory_ids"].append(mid)
            for i, e1 in enumerate(edge["entities"]):
                for e2 in edge["entities"][i + 1:]:
                    new_index[e1]["co_entities"][e2] = new_index[e1]["co_entities"].get(e2, 0) + 1
                    new_index[e2]["co_entities"][e1] = new_index[e2]["co_entities"].get(e1, 0) + 1

        self._hyper_edges = kept
        self._entity_index = new_index
        self._next_edge_id = len(kept)
        logger.info("超边索引裁剪: %d → %d", len(removed) + len(kept), len(kept))

    def stats(self) -> dict:
        """统计信息。"""
        with self._lock:
            return {
                "total_entities": len(self._entity_index),
                "total_hyperedges": len(self._hyper_edges),
            }
