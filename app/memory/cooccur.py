"""共现记录跟踪器 — 只记录记忆之间的共现关系，不做检索决策。

v3: SQLite 替代 JSONL。不再全量加载，查询走索引。
"""

import logging
import threading
from datetime import datetime, timedelta

from app.config.settings import (
    CO_OCCURRENCE_FILE, CO_OCCURRENCE_MAX_PAIRS,
    CO_OCCURRENCE_CLEANUP_RATIO, CO_OCCURRENCE_MIN_COUNT)
from app.core.db import get_db

logger = logging.getLogger(__name__)

_CO_OCCURRENCE_FILE = CO_OCCURRENCE_FILE


class CoOccurrenceTracker:
    """两两共现记录，SQLite 存储，线程安全。"""

    EXTEND_TOP_K = 3
    CO_WITH_LIMIT = 10
    LTD_CHECK_INTERVAL = 20   # 每 query 调用 20 次检查一次
    LTD_DECAY_DAYS = 7         # 超过 7 天未同时出现则衰减
    LTD_DECREMENT = 1          # 每次衰减减 1

    def __init__(self, file_path: str = _CO_OCCURRENCE_FILE):
        self._file_path = file_path
        self._conn: "sqlite3.Connection | None" = None
        self._ltd_lock = threading.Lock()
        self._ltd_counter = 0
        self._init_db()

    def _init_db(self):
        conn = get_db(self._file_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cooccurrence (
                id_a TEXT NOT NULL,
                id_b TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                last_time TEXT NOT NULL,
                PRIMARY KEY (id_a, id_b)
            );
            CREATE INDEX IF NOT EXISTS idx_cooc_a ON cooccurrence(id_a);
            CREATE INDEX IF NOT EXISTS idx_cooc_b ON cooccurrence(id_b);
        """)
        conn.commit()
        self._conn = conn
        self._migrate_if_needed()

    def _migrate_if_needed(self):
        """一次性：旧 co_occurrence.json → SQLite。迁移后不删旧文件，用户自行处理。"""
        conn = self._conn
        existing = conn.execute("SELECT COUNT(*) as n FROM cooccurrence").fetchone()["n"]
        if existing > 0:
            return  # 已有数据，跳过

        import os as _os
        json_path = self._file_path.replace(".db", ".json")
        if not _os.path.exists(json_path):
            return

        logger.info("共现记录迁移 JSON → SQLite: %s", json_path)
        try:
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            return

        if not data:
            return

        count = 0
        for key, val in data.items():
            ids = key.split("|")
            if len(ids) != 2:
                continue
            a, b = sorted(ids)
            cnt = val.get("count", 1) if isinstance(val, dict) else int(val)
            lt = val.get("last_time", "") if isinstance(val, dict) else ""
            conn.execute(
                "INSERT OR IGNORE INTO cooccurrence(id_a, id_b, count, last_time) "
                "VALUES (?, ?, ?, ?)",
                (a, b, cnt, lt),
            )
            count += 1
        conn.commit()
        logger.info("共现迁移完成: %d 对迁移至 %s", count, self._file_path)

    # ═════════════════════════════════════════════════════════
    # 写入
    # ═════════════════════════════════════════════════════════

    def record(self, memory_ids: list[str]):
        if len(memory_ids) < 2:
            return
        conn = self._conn
        now = datetime.now().isoformat()
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                a, b = sorted([memory_ids[i], memory_ids[j]])
                conn.execute(
                    "INSERT INTO cooccurrence(id_a, id_b, count, last_time) "
                    "VALUES (?, ?, 1, ?) "
                    "ON CONFLICT(id_a, id_b) DO UPDATE SET "
                    "count = cooccurrence.count + 1, last_time = excluded.last_time",
                    (a, b, now),
                )
        conn.commit()
        self._maybe_cleanup()

    def _maybe_cleanup(self):
        conn = self._conn
        total = conn.execute("SELECT COUNT(*) as n FROM cooccurrence").fetchone()["n"]
        if total < CO_OCCURRENCE_MAX_PAIRS:
            return
        to_remove = max(1, int(total * CO_OCCURRENCE_CLEANUP_RATIO))
        # 先删 count < MIN_COUNT 的最旧条目
        removed = conn.execute(
            "DELETE FROM cooccurrence WHERE rowid IN ("
            "  SELECT rowid FROM cooccurrence"
            "  WHERE count < ?"
            "  ORDER BY last_time ASC LIMIT ?"
            ")",
            (CO_OCCURRENCE_MIN_COUNT, to_remove),
        ).rowcount
        # 还不够就硬删最旧
        if removed < to_remove:
            conn.execute(
                "DELETE FROM cooccurrence WHERE rowid IN ("
                "  SELECT rowid FROM cooccurrence"
                "  ORDER BY last_time ASC LIMIT ?"
                ")",
                (to_remove - removed,),
            )
        conn.commit()

    # ═════════════════════════════════════════════════════════
    # 查询
    # ═════════════════════════════════════════════════════════

    def get_related(self, memory_id: str, data: dict | None = None) -> list[tuple[str, int]]:
        """返回与 memory_id 共现频率最高的 TOP_K 个 partner。"""
        conn = self._conn
        rows = conn.execute(
            "SELECT id_a, id_b, count FROM cooccurrence WHERE id_a = ? OR id_b = ?"
            " ORDER BY count DESC LIMIT ?",
            (memory_id, memory_id, self.EXTEND_TOP_K * 3),
        ).fetchall()
        pairs = []
        for row in rows:
            partner = row["id_b"] if row["id_a"] == memory_id else row["id_a"]
            pairs.append((partner, row["count"]))
        pairs.sort(key=lambda x: -x[1])
        return pairs[: self.EXTEND_TOP_K]

    def get_co_counts(self, memory_ids: list[str]) -> dict[str, int]:
        """返回每个 memory_id 的共现度数。"""
        counts = {mid: 0 for mid in memory_ids}
        conn = self._conn
        for mid in memory_ids:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM cooccurrence WHERE id_a = ? OR id_b = ?",
                (mid, mid),
            ).fetchone()
            counts[mid] = row["n"]
        return counts

    def get_co_count(self, memory_id: str) -> int:
        conn = self._conn
        row = conn.execute(
            "SELECT COUNT(*) as n FROM cooccurrence WHERE id_a = ? OR id_b = ?",
            (memory_id, memory_id),
        ).fetchone()
        return row["n"]

    def get_co_with(self, memory_id: str) -> list[dict]:
        """返回 {id, count} 列表，最多 CO_WITH_LIMIT 条。"""
        conn = self._conn
        rows = conn.execute(
            "SELECT id_a, id_b, count FROM cooccurrence WHERE id_a = ? OR id_b = ?"
            " ORDER BY count DESC LIMIT ?",
            (memory_id, memory_id, self.CO_WITH_LIMIT * 2),
        ).fetchall()
        pairs = []
        for row in rows:
            partner = row["id_b"] if row["id_a"] == memory_id else row["id_a"]
            pairs.append({"id": partner, "count": row["count"]})
        pairs.sort(key=lambda x: -x["count"])
        return pairs[: self.CO_WITH_LIMIT]

    def query(self, memory_ids: list[str]) -> list[dict]:
        """批量查询共现，返回 {id, count} 列表。带 LTD 周期衰减。"""
        memory_set = set(memory_ids)
        conn = self._conn

        # 构建参数化查询
        placeholders = ",".join("?" * len(memory_ids))
        query_sql = (
            f"SELECT id_a, id_b, count FROM cooccurrence "
            f"WHERE id_a IN ({placeholders}) OR id_b IN ({placeholders})"
        )
        params = list(memory_ids) + list(memory_ids)
        rows = conn.execute(query_sql, params).fetchall()

        seen = set(memory_ids)
        results = []
        for row in rows:
            partner = row["id_b"] if row["id_a"] in memory_set else row["id_a"]
            if partner not in seen:
                seen.add(partner)
                results.append({"id": partner, "count": row["count"]})
        results.sort(key=lambda x: -x["count"])

        # LTD：周期性衰减
        with self._ltd_lock:
            self._ltd_counter += 1
            if self._ltd_counter >= self.LTD_CHECK_INTERVAL:
                self._ltd_counter = 0
                self._apply_ltd()

        return results

    def _apply_ltd(self):
        """扫描共现条目，超过 LTD_DECAY_DAYS 未同时出现减 1，归零删除。"""
        conn = self._conn
        cutoff = (datetime.now() - timedelta(days=self.LTD_DECAY_DAYS)).isoformat()
        conn.execute(
            "UPDATE cooccurrence SET count = count - ?"
            " WHERE last_time < ?",
            (self.LTD_DECREMENT, cutoff),
        )
        conn.execute("DELETE FROM cooccurrence WHERE count <= 0")
        conn.commit()

    # ═════════════════════════════════════════════════════════
    # 维护
    # ═════════════════════════════════════════════════════════

    def remove(self, memory_id: str):
        conn = self._conn
        conn.execute(
            "DELETE FROM cooccurrence WHERE id_a = ? OR id_b = ?",
            (memory_id, memory_id),
        )
        conn.commit()

    def clear(self):
        """清空所有共现记录（benchmark reset 用）。"""
        conn = self._conn
        conn.execute("DELETE FROM cooccurrence")
        conn.commit()

    def export_for_symmetry(self) -> dict[str, dict[str, int]]:
        """导出为对称性分析兼容格式：{entity: {related_entity: count}}。

        将 SQLite 中 (id_a, id_b, count) 的行重建为嵌套 dict，
        供 PersonaSymmetry 消费，替代旧 JSON 文件直接读取。
        """
        conn = self._conn
        rows = conn.execute(
            "SELECT id_a, id_b, count FROM cooccurrence ORDER BY count DESC"
        ).fetchall()
        data: dict[str, dict[str, int]] = {}
        for row in rows:
            a, b, cnt = row["id_a"], row["id_b"], row["count"]
            if a not in data:
                data[a] = {}
            data[a][b] = cnt
            # 也填充反向关系，保持与旧格式兼容
            if b not in data:
                data[b] = {}
            data[b][a] = cnt
        return data
