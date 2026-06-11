"""实体共现跟踪器 — 记录同一段对话中哪些实体成对出现。

入库时记录实体对，检索时查询给定实体的共现实体及关联记忆。
v3: SQLite 替代 JSONL。查询走索引，不再全量加载。

数据格式：
    entity_a | entity_b | count | memory_ids (JSON array)
    双向存储：record(A, B) → 写两行 (A, B) 和 (B, A)
"""

import json as _json
import logging
import os
import threading

from app.core.db import get_db

logger = logging.getLogger(__name__)

# 默认路径（context.py 通常传显式路径，这里兜底）
_ENTITY_PAIR_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "entity_pairs.db"
)


class EntityPairTracker:
    """实体共现跟踪器。记录、查询，SQLite 存储，线程安全。"""

    EXPAND_TOP_K = 5       # 每实体最多返回的共现实体数
    MAX_MEMORIES = 30      # 扩展时最多取回的记忆数

    def __init__(self, file_path: str = _ENTITY_PAIR_FILE):
        self._file = file_path  # 公开，向后兼容（E2E 测试引用）
        self._conn: "sqlite3.Connection | None" = None
        self._init_db()

    def _init_db(self):
        conn = get_db(self._file)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entity_pair (
                entity_a TEXT NOT NULL,
                entity_b TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                memory_ids TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (entity_a, entity_b)
            );
            CREATE INDEX IF NOT EXISTS idx_ep_a ON entity_pair(entity_a);
            CREATE INDEX IF NOT EXISTS idx_ep_b ON entity_pair(entity_b);
        """)
        conn.commit()
        self._conn = conn
        self._migrate_if_needed()

    def _migrate_if_needed(self):
        """一次性：旧 entity_pairs.json → SQLite。"""
        conn = self._conn
        existing = conn.execute("SELECT COUNT(*) as n FROM entity_pair").fetchone()["n"]
        if existing > 0:
            return

        json_path = self._file.replace(".db", ".json")
        if not os.path.exists(json_path):
            return

        logger.info("实体对迁移 JSON → SQLite: %s", json_path)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            return

        if not data:
            return

        count = 0
        for entity_a, related in data.items():
            for entity_b, val in related.items():
                cnt = val.get("count", 1) if isinstance(val, dict) else int(val)
                mids = val.get("memory_ids", []) if isinstance(val, dict) else []
                conn.execute(
                    "INSERT OR IGNORE INTO entity_pair(entity_a, entity_b, count, memory_ids) "
                    "VALUES (?, ?, ?, ?)",
                    (entity_a, entity_b, cnt, _json.dumps(mids, ensure_ascii=False)),
                )
                count += 1
        conn.commit()
        logger.info("实体对迁移完成: %d 对迁移至 %s", count, self._file)

    # ═════════════════════════════════════════════════════════
    # 向后兼容方法（测试中直接使用）
    # ═════════════════════════════════════════════════════════

    def _load(self) -> dict:
        """向后兼容：将 SQLite 数据重建为旧 JSON dict 格式。"""
        conn = self._conn
        rows = conn.execute("SELECT entity_a, entity_b, count, memory_ids FROM entity_pair").fetchall()
        data: dict[str, dict] = {}
        for row in rows:
            a, b, cnt, mids_json = row["entity_a"], row["entity_b"], row["count"], row["memory_ids"]
            if a not in data:
                data[a] = {}
            try:
                mids = _json.loads(mids_json) if mids_json else []
            except (_json.JSONDecodeError, TypeError):
                mids = []
            data[a][b] = {"count": cnt, "memory_ids": mids}
        return data

    def _invalidate_cache(self):
        """向后兼容：SQLite 无内存缓存，no-op。"""
        pass

    # ═════════════════════════════════════════════════════════
    # 写入
    # ═════════════════════════════════════════════════════════

    def record(self, entity_a: str, entity_b: str, memory_id: str):
        """记录两个实体在同一段对话中共现，关联到具体记忆 ID。"""
        if not entity_a or not entity_b or entity_a == entity_b:
            return
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")  # 显式事务包裹双向写入
        # 双向写入 (A, B) 和 (B, A) — 保持与旧 JSON 格式兼容
        for a, b in [(entity_a, entity_b), (entity_b, entity_a)]:
            conn.execute(
                "INSERT INTO entity_pair(entity_a, entity_b, count, memory_ids) "
                "VALUES (?, ?, 1, ?) "
                "ON CONFLICT(entity_a, entity_b) DO UPDATE SET "
                "count = entity_pair.count + 1, "
                "memory_ids = CASE "
                "  WHEN ? NOT IN (SELECT value FROM json_each(entity_pair.memory_ids)) "
                "  THEN json_insert(entity_pair.memory_ids, '$[#]', ?) "
                "  ELSE entity_pair.memory_ids END",
                (a, b, _json.dumps([memory_id], ensure_ascii=False),
                 memory_id, memory_id),
            )
        conn.commit()

    # ═════════════════════════════════════════════════════════
    # 查询
    # ═════════════════════════════════════════════════════════

    def expand(self, entity_names: list[str]) -> dict[str, dict]:
        """给定一批实体名，返回 {entity: {related_entity: count, ...}, ...}。

        用于检索阶段判断「用户提到 A，是否常和 B 一起出现」。
        """
        if not entity_names:
            return {}
        conn = self._conn
        result: dict[str, dict] = {}
        for ename in entity_names:
            rows = conn.execute(
                "SELECT entity_b, count FROM entity_pair WHERE entity_a = ?"
                " ORDER BY count DESC LIMIT ?",
                (ename, self.EXPAND_TOP_K),
            ).fetchall()
            if rows:
                result[ename] = {row["entity_b"]: row["count"] for row in rows}
        return result

    def get_memory_ids(self, entity_names: list[str]) -> list[str]:
        """给定一批实体名，收集所有相关的记忆 ID，按共现次数降序、去重。

        单次批量查询替代逐 entity 的 N+1 模式。
        """
        if not entity_names:
            return []
        conn = self._conn
        scored: dict[str, int] = {}
        placeholders = ",".join("?" * len(entity_names))
        rows = conn.execute(
            f"SELECT entity_a, memory_ids, count FROM entity_pair WHERE entity_a IN ({placeholders})",
            entity_names,
        ).fetchall()
        for row in rows:
            try:
                mids = _json.loads(row["memory_ids"]) if row["memory_ids"] else []
            except (_json.JSONDecodeError, TypeError):
                mids = []
            for mid in mids:
                scored[mid] = scored.get(mid, 0) + row["count"]
        sorted_ids = sorted(scored.items(), key=lambda x: -x[1])
        return [mid for mid, _ in sorted_ids[:self.MAX_MEMORIES]]

    # ═════════════════════════════════════════════════════════
    # 维护
    # ═════════════════════════════════════════════════════════

    def remove_memory(self, memory_id: str):
        """删除记忆时同步清理。"""
        conn = self._conn
        # 查找包含此 memory_id 的所有行
        rows = conn.execute(
            "SELECT entity_a, entity_b, memory_ids FROM entity_pair"
        ).fetchall()

        for row in rows:
            try:
                mids = _json.loads(row["memory_ids"]) if row["memory_ids"] else []
            except (_json.JSONDecodeError, TypeError):
                mids = []
            if memory_id in mids:
                mids.remove(memory_id)
                if mids:
                    conn.execute(
                        "UPDATE entity_pair SET memory_ids = ?, count = MAX(1, count - 1)"
                        " WHERE entity_a = ? AND entity_b = ?",
                        (_json.dumps(mids, ensure_ascii=False), row["entity_a"], row["entity_b"]),
                    )
                else:
                    conn.execute(
                        "DELETE FROM entity_pair WHERE entity_a = ? AND entity_b = ?",
                        (row["entity_a"], row["entity_b"]),
                    )
        conn.commit()

    def clear(self):
        """清空所有实体对记录（benchmark reset 用）。"""
        conn = self._conn
        conn.execute("DELETE FROM entity_pair")
        conn.commit()

    def stats(self) -> dict:
        """统计信息。"""
        conn = self._conn
        total_rows = conn.execute("SELECT COUNT(*) as n FROM entity_pair").fetchone()["n"]
        total_entities = conn.execute(
            "SELECT COUNT(DISTINCT entity_a) as n FROM entity_pair"
        ).fetchone()["n"]
        return {
            "total_entities": total_entities,
            "total_pairs": total_rows,
        }
