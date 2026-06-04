"""共现记录跟踪器 — 只记录记忆之间的共现关系，不做检索决策。"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta

from app.config.settings import (
    CO_OCCURRENCE_FILE, CO_OCCURRENCE_MAX_PAIRS, CO_OCCURRENCE_CLEANUP_RATIO, CO_OCCURRENCE_MIN_COUNT)
from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)

_CO_OCCURRENCE_FILE = CO_OCCURRENCE_FILE


class CoOccurrenceTracker:
    """两两共现记录，独立文件存储，线程安全。"""

    EXTEND_TOP_K = 3
    CO_WITH_LIMIT = 10
    LTD_CHECK_INTERVAL = 20   # 每 query 调用 20 次检查一次
    LTD_DECAY_DAYS = 7         # 超过 7 天未同时出现则衰减
    LTD_DECREMENT = 1          # 每次衰减减 1

    def __init__(self, file_path: str = _CO_OCCURRENCE_FILE):
        self._file = file_path
        self._lock = threading.Lock()
        self._ltd_lock = threading.Lock()
        self._cache: dict | None = None
        self._ltd_counter = 0
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
        atomic_write(self._file, data)

    def _save(self, data: dict):
        with self._lock:
            atomic_write(self._file, data)

    def record(self, memory_ids: list[str]):
        if len(memory_ids) < 2:
            return
        with self._lock:
            data = self._load_nolock()
            now = datetime.now().isoformat()
            for i in range(len(memory_ids)):
                for j in range(i + 1, len(memory_ids)):
                    key = "|".join(sorted([memory_ids[i], memory_ids[j]]))
                    if key in data:
                        data[key]["count"] += 1
                        data[key]["last_time"] = now
                    else:
                        data[key] = {"count": 1, "last_time": now}
            self._save_nolock(data)
        self._maybe_cleanup(data)

    def _maybe_cleanup(self, data: dict):
        if len(data) < CO_OCCURRENCE_MAX_PAIRS:
            return
        sorted_pairs = sorted(data.items(), key=lambda x: x[1]["last_time"])
        to_remove_count = max(1, int(len(sorted_pairs) * CO_OCCURRENCE_CLEANUP_RATIO))
        removed = 0
        for key, val in sorted_pairs:
            if removed >= to_remove_count:
                break
            if val["count"] < CO_OCCURRENCE_MIN_COUNT:
                del data[key]
                removed += 1
        if removed == 0:
            for key, val in sorted_pairs[:to_remove_count]:
                del data[key]
                removed += 1
        self._save(data)

    def _invalidate_cache(self):
        with self._lock:
            self._cache = None

    def get_related(self, memory_id: str, data: dict | None = None) -> list[tuple[str, int]]:
        if data is None:
            data = self._load()
        pairs = []
        for key, val in data.items():
            ids = key.split("|")
            if memory_id in ids:
                partner = ids[0] if ids[1] == memory_id else ids[1]
                pairs.append((partner, val["count"]))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:self.EXTEND_TOP_K]

    def get_co_counts(self, memory_ids: list[str]) -> dict[str, int]:
        data = self._load()
        counts = {mid: 0 for mid in memory_ids}
        for key in data:
            ids = key.split("|")
            for mid in ids:
                if mid in counts:
                    counts[mid] += 1
        return counts

    def remove(self, memory_id: str):
        with self._lock:
            data = self._load_nolock()
            keys = [k for k in data if memory_id in k]
            for k in keys:
                del data[k]
            if keys:
                self._save_nolock(data)
                self._cache = None

    def get_co_count(self, memory_id: str) -> int:
        data = self._load()
        return sum(1 for k in data if memory_id in k)

    def get_co_with(self, memory_id: str) -> list[dict]:
        data = self._load()
        pairs = []
        for key, val in data.items():
            ids = key.split("|")
            if memory_id in ids:
                partner = ids[0] if ids[1] == memory_id else ids[1]
                pairs.append({"id": partner, "count": val["count"]})
        pairs.sort(key=lambda x: -x["count"])
        return pairs[:self.CO_WITH_LIMIT]

    def query(self, memory_ids: list[str]) -> list[dict]:
        memory_set = set(memory_ids)
        data = self._load()
        index: dict[str, list[tuple[str, int]]] = {}
        for key, val in data.items():
            a, b = key.split("|")
            cnt = val["count"]
            index.setdefault(a, []).append((b, cnt))
            index.setdefault(b, []).append((a, cnt))
        seen = set()
        results = []
        for mid in memory_ids:
            for partner_id, count in index.get(mid, ()):
                if partner_id not in seen and partner_id not in memory_set:
                    seen.add(partner_id)
                    results.append({"id": partner_id, "count": count})
        results.sort(key=lambda x: -x["count"])
        # LTD：周期性衰减过时关联（锁保护，防止多线程竞态）
        with self._ltd_lock:
            self._ltd_counter += 1
            if self._ltd_counter >= self.LTD_CHECK_INTERVAL:
                self._ltd_counter = 0
                should_decay = True
            else:
                should_decay = False
        if should_decay:
            self._apply_ltd()
        return results

    def _apply_ltd(self):
        """扫描所有共现条目，超过 LTD_DECAY_DAYS 未同时出现的自动减 1，归零删除。"""
        data = self._load()
        if not data:
            return
        now = datetime.now()
        cutoff = now - timedelta(days=self.LTD_DECAY_DAYS)
        changed = False
        for key, val in list(data.items()):
            last = val.get("last_time")
            if not last:
                continue
            try:
                last_dt = datetime.fromisoformat(last) if isinstance(last, str) else datetime.fromtimestamp(last)
            except (ValueError, TypeError):
                continue
            if last_dt < cutoff:
                data[key]["count"] = val["count"] - self.LTD_DECREMENT
                if data[key]["count"] <= 0:
                    del data[key]
                changed = True
        if changed:
            self._save(data)
