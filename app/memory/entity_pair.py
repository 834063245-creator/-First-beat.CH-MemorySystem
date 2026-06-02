"""实体共现跟踪器 — 记录同一段对话中哪些实体成对出现。

入库时记录实体对，检索时查询给定实体的共现实体及关联记忆。
纯本地文件存储，零 API 依赖，线程安全。

数据格式：
{
    "林琳": {
        "预算":     {"count": 3, "memory_ids": ["id1", "id2"]},
        "项目A":    {"count": 2, "memory_ids": ["id3"]}
    },
    "预算": {
        "林琳":     {"count": 3, "memory_ids": ["id1", "id2"]}
    }
}
"""

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)


class EntityPairTracker:
    """实体共现跟踪器。记录、查询、线程安全。"""

    EXPAND_TOP_K = 5       # 每实体最多返回的共现实体数
    MAX_MEMORIES = 30      # 扩展时最多取回的记忆数

    def __init__(self, file_path: str):
        self._file = file_path
        self._lock = threading.Lock()
        self._cache: dict | None = None
        self._ensure_file()

    def _ensure_file(self):
        parent = os.path.dirname(self._file)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self._file):
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load(self) -> dict:
        with self._lock:
            if self._cache is not None:
                return self._cache
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                    return self._cache
            except (json.JSONDecodeError, OSError):
                self._cache = {}
                return {}

    def _load_nolock(self) -> dict:
        """Read without acquiring lock (caller must hold self._lock)."""
        if self._cache is not None:
            return self._cache
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
                return self._cache
        except (json.JSONDecodeError, OSError):
            self._cache = {}
            return {}

    def _save_nolock(self, data: dict):
        """Write without acquiring lock (caller must hold self._lock)."""
        from app.tools.atomic import atomic_write
        atomic_write(self._file, data)

    def _save(self, data: dict):
        with self._lock:
            from app.tools.atomic import atomic_write
            atomic_write(self._file, data)

    def _invalidate_cache(self):
        with self._lock:
            self._cache = None

    def record(self, entity_a: str, entity_b: str, memory_id: str):
        """记录两个实体在同一段对话中共现，关联到具体记忆 ID。"""
        if not entity_a or not entity_b or entity_a == entity_b:
            return
        with self._lock:
            data = self._load_nolock()
            for a, b in [(entity_a, entity_b), (entity_b, entity_a)]:
                if a not in data:
                    data[a] = {}
                if b not in data[a]:
                    data[a][b] = {"count": 0, "memory_ids": []}
                data[a][b]["count"] += 1
                if memory_id not in data[a][b]["memory_ids"]:
                    data[a][b]["memory_ids"].append(memory_id)
            self._save_nolock(data)

    def expand(self, entity_names: list[str]) -> dict[str, dict]:
        """给定一批实体名，返回 {entity: {related_entity: count, ...}, ...}。

        用于检索阶段判断「用户提到 A，是否常和 B 一起出现」。
        """
        if not entity_names:
            return {}
        data = self._load()
        result: dict[str, dict] = {}
        for ename in entity_names:
            related = data.get(ename, {})
            if related:
                # 按 count 降序，截取 top_k
                sorted_items = sorted(related.items(), key=lambda x: -x[1]["count"])[:self.EXPAND_TOP_K]
                result[ename] = {k: v["count"] for k, v in sorted_items}
        return result

    def get_memory_ids(self, entity_names: list[str]) -> list[str]:
        """给定一批实体名，收集所有相关的记忆 ID，按共现次数降序、去重。"""
        if not entity_names:
            return []
        data = self._load()
        scored: dict[str, int] = {}
        for ename in entity_names:
            related = data.get(ename, {})
            for rel_name, val in related.items():
                for mid in val.get("memory_ids", []):
                    scored[mid] = scored.get(mid, 0) + val["count"]
        # 按总分降序
        sorted_ids = sorted(scored.items(), key=lambda x: -x[1])
        return [mid for mid, _ in sorted_ids[:self.MAX_MEMORIES]]

    def remove_memory(self, memory_id: str):
        """删除记忆时同步清理。"""
        with self._lock:
            data = self._load_nolock()
            changed = False
            for entity in list(data.keys()):
                for rel_entity in list(data[entity].keys()):
                    ids = data[entity][rel_entity].get("memory_ids", [])
                    if memory_id in ids:
                        ids.remove(memory_id)
                        data[entity][rel_entity]["count"] = max(1, data[entity][rel_entity]["count"] - 1)
                        changed = True
                        if not ids:
                            del data[entity][rel_entity]
                if not data[entity]:
                    del data[entity]
                    changed = True
            if changed:
                self._save_nolock(data)
                self._cache = None

    def stats(self) -> dict:
        """统计信息。"""
        data = self._load()
        total_entities = len(data)
        total_pairs = sum(len(v) for v in data.values())
        return {
            "total_entities": total_entities,
            "total_pairs": total_pairs,
        }
