"""测试 app/memory/entity_pair.py — 实体共现跟踪器。"""
import os
import tempfile

import pytest


def _make_tracker():
    """创建临时 EntityPairTracker 实例，确保初始状态为空。"""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)  # 删除 mkstemp 创建的空文件，让 _ensure_file 重建
    from app.memory.entity_pair import EntityPairTracker
    t = EntityPairTracker(path)
    t._invalidate_cache()
    t._path = path  # 保存路径用于清理
    return t


def _cleanup(t):
    try:
        os.unlink(t._path)
    except OSError:
        pass


class TestInit:
    def test_creates_file(self):
        t = _make_tracker()
        try:
            assert os.path.exists(t._file)
        finally:
            _cleanup(t)

    def test_initial_state_empty(self):
        t = _make_tracker()
        try:
            s = t.stats()
            assert s["total_entities"] == 0
            assert s["total_pairs"] == 0
        finally:
            _cleanup(t)


class TestRecord:
    def test_records_pair(self):
        t = _make_tracker()
        try:
            t.record("Python", "Django", "mem_1")
            t._invalidate_cache()
            s = t.stats()
            assert s["total_entities"] >= 2
        finally:
            _cleanup(t)

    def test_bidirectional_record(self):
        t = _make_tracker()
        try:
            t.record("A", "B", "m1")
            t._invalidate_cache()
            data = t._load()
            assert "A" in data
            assert "B" in data["A"]
            assert "A" in data["B"]
        finally:
            _cleanup(t)

    def test_same_entity_ignored(self):
        t = _make_tracker()
        try:
            t.record("X", "X", "m1")
            s = t.stats()
            assert s["total_pairs"] == 0
        finally:
            _cleanup(t)

    def test_empty_entity_ignored(self):
        t = _make_tracker()
        try:
            t.record("", "valid", "m1")
            t.record("valid", "", "m1")
            s = t.stats()
            assert s["total_pairs"] == 0
        finally:
            _cleanup(t)

    def test_count_increments(self):
        t = _make_tracker()
        try:
            t.record("Py", "AI", "m1")
            t.record("Py", "AI", "m2")
            t._invalidate_cache()
            data = t._load()
            assert data["Py"]["AI"]["count"] == 2
        finally:
            _cleanup(t)

    def test_multiple_pairs(self):
        t = _make_tracker()
        try:
            pairs = [("A", "B"), ("A", "C"), ("B", "C")]
            for i, (a, b) in enumerate(pairs):
                t.record(a, b, f"mem_{i}")
            t._invalidate_cache()
            s = t.stats()
            assert s["total_entities"] == 3
            assert s["total_pairs"] == 6  # bidirectional
        finally:
            _cleanup(t)


class TestExpand:
    def test_expand_returns_top_k(self):
        t = _make_tracker()
        try:
            t.record("Python", "Django", "m1")
            t.record("Python", "Flask", "m2")
            t.record("Python", "FastAPI", "m3")
            t.record("Python", "Django", "m4")
            t._invalidate_cache()
            result = t.expand(["Python"])
            assert "Python" in result
            related = result["Python"]
            assert related.get("Django", 0) == 2
        finally:
            _cleanup(t)

    def test_expand_empty_input(self):
        t = _make_tracker()
        try:
            result = t.expand([])
            assert result == {}
        finally:
            _cleanup(t)

    def test_expand_unknown_entity(self):
        t = _make_tracker()
        try:
            result = t.expand(["NotExist"])
            assert result == {}
        finally:
            _cleanup(t)


class TestGetMemoryIds:
    def test_returns_related_ids(self):
        t = _make_tracker()
        try:
            t.record("X", "Y", "abc")
            t.record("X", "Z", "def")
            t._invalidate_cache()
            ids = t.get_memory_ids(["X"])
            assert "abc" in ids
            assert "def" in ids
        finally:
            _cleanup(t)

    def test_empty_input(self):
        t = _make_tracker()
        try:
            assert t.get_memory_ids([]) == []
        finally:
            _cleanup(t)

    def test_unknown_entity(self):
        t = _make_tracker()
        try:
            assert t.get_memory_ids(["Nope"]) == []
        finally:
            _cleanup(t)


class TestRemoveMemory:
    def test_removes_memory_id(self):
        t = _make_tracker()
        try:
            t.record("Go", "Rust", "del_me")
            t.record("Go", "Rust", "keep_me")
            t.remove_memory("del_me")
            t._invalidate_cache()
            ids = t.get_memory_ids(["Go"])
            assert "del_me" not in ids
            assert "keep_me" in ids
        finally:
            _cleanup(t)

    def test_removes_empty_pairs(self):
        t = _make_tracker()
        try:
            t.record("Solo", "Only", "only_id")
            t.remove_memory("only_id")
            t._invalidate_cache()
            s = t.stats()
            assert s["total_pairs"] == 0
            assert s["total_entities"] == 0
        finally:
            _cleanup(t)

    def test_remove_nonexistent(self):
        t = _make_tracker()
        try:
            t.record("A", "B", "m1")
            t.remove_memory("nonexistent")
            t._invalidate_cache()
            s = t.stats()
            assert s["total_pairs"] == 2  # bidirectional
        finally:
            _cleanup(t)


class TestStats:
    def test_stats_after_records(self):
        t = _make_tracker()
        try:
            t.record("A", "B", "m1")
            t.record("A", "C", "m2")
            t._invalidate_cache()
            s = t.stats()
            assert s["total_entities"] == 3
            assert s["total_pairs"] == 4  # A-B, B-A, A-C, C-A
        finally:
            _cleanup(t)

    def test_stats_empty(self):
        t = _make_tracker()
        try:
            s = t.stats()
            assert s == {"total_entities": 0, "total_pairs": 0}
        finally:
            _cleanup(t)


class TestCacheInvalidation:
    def test_invalidate_clears_cache(self):
        t = _make_tracker()
        try:
            t._load()  # populates cache
            assert t._cache is not None
            t._invalidate_cache()
            assert t._cache is None
        finally:
            _cleanup(t)
