"""超边索引 — 轻量超图，保留实体在多实体上下文中的完整共现关系。

区别于 EntityPairTracker（只记录两两共现），超边索引维护"哪些实体在同一段
对话中同时出现"的完整上下文。用于 weave_context 的故事线聚合，替代
entities[:3] 硬截断。

v3: SQLite 替代 JSONL。三表结构：hyper_edge + entity_index + entity_edge。
    查询走索引，不再全量加载。
"""

import json as _json
import logging
import os
import threading
from collections import defaultdict

from app.core.db import get_db

logger = logging.getLogger(__name__)

# 默认路径
_HYPEREDGE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "hyper_edges.db"
)


class HyperEdgeIndex:
    """超边索引。记录、查询，SQLite 存储，线程安全。"""

    EXPAND_TOP_K = 10       # 展开时每实体最多返回的共现实体数
    MAX_EDGES = 10000       # 超边总数上限（触发裁剪）

    def __init__(self, file_path: str = _HYPEREDGE_FILE):
        self._file = file_path
        self._conn: "sqlite3.Connection | None" = None
        self._init_db()

    def _init_db(self):
        conn = get_db(self._file)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS hyper_edge (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entities TEXT NOT NULL,
                memory_ids TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS entity_index (
                entity TEXT PRIMARY KEY,
                co_entities TEXT NOT NULL DEFAULT '{}',
                memory_ids TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS entity_edge (
                entity TEXT NOT NULL,
                edge_id INTEGER NOT NULL,
                PRIMARY KEY (entity, edge_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ee_entity ON entity_edge(entity);
            CREATE INDEX IF NOT EXISTS idx_ee_edge ON entity_edge(edge_id);
        """)
        conn.commit()
        self._conn = conn
        self._migrate_if_needed()

    def _migrate_if_needed(self):
        """一次性：旧 hyper_edges.json → SQLite。"""
        conn = self._conn
        existing = conn.execute("SELECT COUNT(*) as n FROM hyper_edge").fetchone()["n"]
        if existing > 0:
            return

        json_path = self._file.replace(".db", ".json")
        if not os.path.exists(json_path):
            return

        logger.info("超边索引迁移 JSON → SQLite: %s", json_path)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            return

        edges = data.get("hyper_edges", [])
        entity_index_data = data.get("entity_index", {})

        for edge in edges:
            conn.execute(
                "INSERT INTO hyper_edge(entities, memory_ids) VALUES (?, ?)",
                (
                    _json.dumps(edge.get("entities", []), ensure_ascii=False),
                    _json.dumps(edge.get("memory_ids", []), ensure_ascii=False),
                ),
            )

        for entity, idx in (entity_index_data or {}).items():
            conn.execute(
                "INSERT OR IGNORE INTO entity_index(entity, co_entities, memory_ids) "
                "VALUES (?, ?, ?)",
                (
                    entity,
                    _json.dumps(idx.get("co_entities", {}), ensure_ascii=False),
                    _json.dumps(idx.get("memory_ids", []), ensure_ascii=False),
                ),
            )
            for edge_id in idx.get("hyper_edges", []):
                conn.execute(
                    "INSERT OR IGNORE INTO entity_edge(entity, edge_id) VALUES (?, ?)",
                    (entity, edge_id + 1),  # SQLite AUTOINCREMENT starts at 1, old IDs at 0
                )

        conn.commit()
        logger.info("超边迁移完成: %d 边 / %d 实体 → %s",
                     len(edges), len(entity_index_data), self._file)

    # ═════════════════════════════════════════════════════════
    # 向后兼容
    # ═════════════════════════════════════════════════════════

    def _load(self):
        """向后兼容：返回旧格式 dict，供 _prune 重建使用。"""
        conn = self._conn
        rows = conn.execute(
            "SELECT edge_id, entities, memory_ids FROM hyper_edge ORDER BY edge_id"
        ).fetchall()
        edges = []
        for row in rows:
            try:
                ent = _json.loads(row["entities"])
            except (_json.JSONDecodeError, TypeError):
                ent = []
            try:
                mids = _json.loads(row["memory_ids"])
            except (_json.JSONDecodeError, TypeError):
                mids = []
            edges.append({"entities": ent, "memory_ids": mids})
        return edges

    # ═════════════════════════════════════════════════════════
    # 写入
    # ═════════════════════════════════════════════════════════

    def record(self, entities: list[str], memory_id: str):
        """记录一组实体在同一段对话中共现。"""
        entities = list(set(e for e in entities if isinstance(e, str) and len(e) >= 2))
        if len(entities) < 2:
            return

        conn = self._conn
        sorted_entities = sorted(entities)

        # 插入超边
        cur = conn.execute(
            "INSERT INTO hyper_edge(entities, memory_ids) VALUES (?, ?)",
            (_json.dumps(sorted_entities, ensure_ascii=False),
             _json.dumps([memory_id], ensure_ascii=False)),
        )
        edge_id = cur.lastrowid

        # 更新 entity_index + entity_edge
        for e1 in entities:
            # entity_edge
            conn.execute(
                "INSERT OR IGNORE INTO entity_edge(entity, edge_id) VALUES (?, ?)",
                (e1, edge_id),
            )

            # entity_index: upsert
            conn.execute(
                "INSERT OR IGNORE INTO entity_index(entity, co_entities, memory_ids) "
                "VALUES (?, '{}', '[]')",
                (e1,),
            )
            # 追加 memory_id
            conn.execute(
                "UPDATE entity_index SET memory_ids = "
                "CASE WHEN ? NOT IN (SELECT value FROM json_each(entity_index.memory_ids)) "
                "THEN json_insert(entity_index.memory_ids, '$[#]', ?) "
                "ELSE entity_index.memory_ids END "
                "WHERE entity = ?",
                (memory_id, memory_id, e1),
            )

            # 更新 co_entities：对每个 e2 != e1，count += 1
            for e2 in entities:
                if e2 == e1:
                    continue
                row = conn.execute(
                    "SELECT co_entities FROM entity_index WHERE entity = ?", (e1,)
                ).fetchone()
                if row:
                    try:
                        co = _json.loads(row["co_entities"])
                    except (_json.JSONDecodeError, TypeError):
                        co = {}
                    co[e2] = co.get(e2, 0) + 1
                    conn.execute(
                        "UPDATE entity_index SET co_entities = ? WHERE entity = ?",
                        (_json.dumps(co, ensure_ascii=False), e1),
                    )

        # 裁剪
        total = conn.execute("SELECT COUNT(*) as n FROM hyper_edge").fetchone()["n"]
        if total > self.MAX_EDGES:
            self._prune()

        conn.commit()

    def _prune(self):
        """裁剪最老的超边，保留最近一半，重建 entity_index。"""
        conn = self._conn
        keep = self.MAX_EDGES // 2

        # 删除最老的边
        conn.execute(
            "DELETE FROM hyper_edge WHERE edge_id IN ("
            "  SELECT edge_id FROM hyper_edge ORDER BY edge_id ASC"
            "  LIMIT (SELECT COUNT(*) - ? FROM hyper_edge)"
            ")",
            (keep,),
        )
        # 清理孤儿 entity_edge
        conn.execute(
            "DELETE FROM entity_edge WHERE edge_id NOT IN (SELECT edge_id FROM hyper_edge)"
        )

        # 重建 entity_index
        conn.execute("DELETE FROM entity_index")
        remaining = conn.execute(
            "SELECT edge_id, entities, memory_ids FROM hyper_edge ORDER BY edge_id"
        ).fetchall()

        for row in remaining:
            try:
                edge_entities = _json.loads(row["entities"])
            except (_json.JSONDecodeError, TypeError):
                edge_entities = []
            try:
                edge_mids = _json.loads(row["memory_ids"])
            except (_json.JSONDecodeError, TypeError):
                edge_mids = []

            for e1 in edge_entities:
                conn.execute(
                    "INSERT OR IGNORE INTO entity_index(entity, co_entities, memory_ids) "
                    "VALUES (?, '{}', '[]')",
                    (e1,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO entity_edge(entity, edge_id) VALUES (?, ?)",
                    (e1, row["edge_id"]),
                )
                # 合并 memory_ids
                for mid in edge_mids:
                    conn.execute(
                        "UPDATE entity_index SET memory_ids = "
                        "CASE WHEN ? NOT IN (SELECT value FROM json_each(entity_index.memory_ids)) "
                        "THEN json_insert(entity_index.memory_ids, '$[#]', ?) "
                        "ELSE entity_index.memory_ids END "
                        "WHERE entity = ?",
                        (mid, mid, e1),
                    )
                for e2 in edge_entities:
                    if e2 == e1:
                        continue
                    ei_row = conn.execute(
                        "SELECT co_entities FROM entity_index WHERE entity = ?", (e1,)
                    ).fetchone()
                    if ei_row:
                        try:
                            co = _json.loads(ei_row["co_entities"])
                        except (_json.JSONDecodeError, TypeError):
                            co = {}
                        co[e2] = co.get(e2, 0) + 1
                        conn.execute(
                            "UPDATE entity_index SET co_entities = ? WHERE entity = ?",
                            (_json.dumps(co, ensure_ascii=False), e1),
                        )

        logger.info("超边索引裁剪: → %d", len(remaining))

    # ═════════════════════════════════════════════════════════
    # 查询
    # ═════════════════════════════════════════════════════════

    def expand(self, entity_names: list[str], top_k: int = None) -> dict[str, int]:
        """给定一批实体名，展开超边，返回 {related_entity: total_weight}。"""
        if top_k is None:
            top_k = self.EXPAND_TOP_K
        if not entity_names:
            return {}

        conn = self._conn
        scores: dict[str, int] = defaultdict(int)

        for ename in entity_names:
            row = conn.execute(
                "SELECT co_entities FROM entity_index WHERE entity = ?", (ename,)
            ).fetchone()
            if not row:
                continue
            try:
                co = _json.loads(row["co_entities"])
            except (_json.JSONDecodeError, TypeError):
                continue
            for co_entity, cnt in co.items():
                if co_entity not in entity_names:
                    scores[co_entity] += cnt

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return dict(sorted_items[:top_k])

    def get_memory_ids(self, entity_names: list[str], max_memories: int = 50) -> list[str]:
        """给定一批实体名，收集所有超边关联的记忆 ID，按出现次数降序。"""
        if not entity_names:
            return []

        conn = self._conn
        scored: dict[str, int] = defaultdict(int)

        for ename in entity_names:
            rows = conn.execute(
                "SELECT he.memory_ids FROM entity_edge ee "
                "JOIN hyper_edge he ON ee.edge_id = he.edge_id "
                "WHERE ee.entity = ?",
                (ename,),
            ).fetchall()
            for row in rows:
                try:
                    mids = _json.loads(row["memory_ids"])
                except (_json.JSONDecodeError, TypeError):
                    mids = []
                for mid in mids:
                    scored[mid] = scored.get(mid, 0) + 1

        sorted_ids = sorted(scored.items(), key=lambda x: -x[1])
        return [mid for mid, _ in sorted_ids[:max_memories]]

    def cluster_key(self, entities: list[str], existing_groups: list[set[str]],
                    min_overlap: int = 2) -> int | None:
        """为给定的实体集合找到最佳匹配的已有分组。"""
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
        """展开实体集合：通过超边找到相关实体，返回扩展后的集合。"""
        if not entities:
            return set()

        conn = self._conn
        entity_set = set(entities)
        result = set(entities)

        # 找到所有关联的超边
        edge_ids = set()
        for ename in entities:
            rows = conn.execute(
                "SELECT edge_id FROM entity_edge WHERE entity = ?", (ename,)
            ).fetchall()
            for row in rows:
                edge_ids.add(row["edge_id"])

        # 检查每条超边是否与输入共享 ≥ min_overlap 个实体
        for eid in edge_ids:
            row = conn.execute(
                "SELECT entities FROM hyper_edge WHERE edge_id = ?", (eid,)
            ).fetchone()
            if not row:
                continue
            try:
                edge_entities = set(_json.loads(row["entities"]))
            except (_json.JSONDecodeError, TypeError):
                continue
            if len(entity_set & edge_entities) >= min_overlap:
                result |= edge_entities

        return result

    # ═════════════════════════════════════════════════════════
    # 维护
    # ═════════════════════════════════════════════════════════

    def remove_memory(self, memory_id: str):
        """删除记忆时同步清理超边索引。"""
        conn = self._conn

        # 从 hyper_edge 中移除
        rows = conn.execute("SELECT edge_id, memory_ids FROM hyper_edge").fetchall()
        for row in rows:
            try:
                mids = _json.loads(row["memory_ids"])
            except (_json.JSONDecodeError, TypeError):
                mids = []
            if memory_id in mids:
                mids.remove(memory_id)
                if mids:
                    conn.execute(
                        "UPDATE hyper_edge SET memory_ids = ? WHERE edge_id = ?",
                        (_json.dumps(mids, ensure_ascii=False), row["edge_id"]),
                    )
                else:
                    conn.execute(
                        "DELETE FROM hyper_edge WHERE edge_id = ?", (row["edge_id"],)
                    )

        # 清理空的 entity_edge
        conn.execute(
            "DELETE FROM entity_edge WHERE edge_id NOT IN (SELECT edge_id FROM hyper_edge)"
        )

        # 从 entity_index 中移除 memory_id
        ei_rows = conn.execute("SELECT entity, memory_ids FROM entity_index").fetchall()
        for row in ei_rows:
            try:
                mids = _json.loads(row["memory_ids"])
            except (_json.JSONDecodeError, TypeError):
                mids = []
            if memory_id in mids:
                mids.remove(memory_id)
                conn.execute(
                    "UPDATE entity_index SET memory_ids = ? WHERE entity = ?",
                    (_json.dumps(mids, ensure_ascii=False), row["entity"]),
                )

        # 清理空的 entity_index 行
        conn.execute("DELETE FROM entity_index WHERE memory_ids = '[]'")

        conn.commit()

    def clear(self):
        """清空所有超边记录。"""
        conn = self._conn
        conn.execute("DELETE FROM hyper_edge")
        conn.execute("DELETE FROM entity_index")
        conn.execute("DELETE FROM entity_edge")
        conn.commit()

    def stats(self) -> dict:
        """统计信息。"""
        conn = self._conn
        total_entities = conn.execute("SELECT COUNT(*) as n FROM entity_index").fetchone()["n"]
        total_edges = conn.execute("SELECT COUNT(*) as n FROM hyper_edge").fetchone()["n"]
        return {
            "total_entities": total_entities,
            "total_hyperedges": total_edges,
        }
