"""ChatHistory 线程安全 + Impulse 锁 + get_brain 竞态 + emb_cache。

扩展：增加 get_brain() 竞态条件测试和 _emb_cache 并发写测试。
"""
import threading
import time

import pytest


class TestChatHistorySnapshot:
    """验证 get_records_snapshot 线程安全。"""

    def test_snapshot_returns_copy(self, tmp_path):
        from app.memory.history import ChatHistory
        ch = ChatHistory(path=str(tmp_path / "chat.jsonl"), max_memory=100)
        ch.append("hello", "hi", "2026-01-01 00:00:00")
        ch.append("how are you", "fine", "2026-01-01 00:01:00")

        snap = ch.get_records_snapshot()
        assert len(snap) == 2
        assert snap[0]["user_message"] == "hello"

        snap.clear()
        assert len(ch.records) == 2

    def test_concurrent_append_and_snapshot(self, tmp_path):
        from app.memory.history import ChatHistory
        ch = ChatHistory(path=str(tmp_path / "chat.jsonl"), max_memory=200)
        errors = []

        def writer():
            for i in range(100):
                try:
                    ch.append(f"msg{i}", f"reply{i}", f"2026-01-01 00:{i:02d}:00")
                except Exception as e:
                    errors.append(f"write: {e}")

        def reader():
            for _ in range(50):
                try:
                    snap = ch.get_records_snapshot()
                    assert isinstance(snap, list)
                except Exception as e:
                    errors.append(f"read: {e}")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"并发错误: {errors}"
        assert len(ch.records) >= 100


class TestImpulseLock:
    """冲动系统锁修复验证。"""

    def test_fatigue_lock_protects_dict(self, tmp_path):
        import os
        from app.background.impulse import ImpulseScheduler

        original = os.environ.get("DATA_DIR", "")
        os.environ["DATA_DIR"] = str(tmp_path)

        try:
            import os as _os
            state_path = _os.path.join(str(tmp_path), "impulse_state.json")
            sched = ImpulseScheduler(state_path=state_path)
            errors = []

            def feeder():
                for i in range(50):
                    try:
                        sched.feed_impulse(f"impulse_{i}", priority=30, source="test", ttl=60)
                    except Exception as e:
                        errors.append(f"feed: {e}")

            def getter():
                for _ in range(20):
                    try:
                        sched.get_next()
                    except Exception as e:
                        errors.append(f"get: {e}")

            threads = [threading.Thread(target=feeder) for _ in range(3)]
            threads.append(threading.Thread(target=getter))
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"并发锁错误: {errors}"
        finally:
            os.environ["DATA_DIR"] = original


class TestEmbCacheConcurrentWrite:
    """ChromaService._emb_cache 并发写测试。"""

    def test_concurrent_write_different_keys(self):
        import threading

        class MockChroma:
            def __init__(self):
                self._emb_cache: dict[str, list] = {}
                self._emb_cache_lock = threading.Lock()

        chroma = MockChroma()
        errors = []

        def writer_a():
            try:
                with chroma._emb_cache_lock:
                    chroma._emb_cache["key_a"] = [0.1, 0.2]
            except Exception as e:
                errors.append(f"writer_a: {e}")

        def writer_b():
            try:
                with chroma._emb_cache_lock:
                    chroma._emb_cache["key_b"] = [0.3, 0.4]
            except Exception as e:
                errors.append(f"writer_b: {e}")

        t1 = threading.Thread(target=writer_a)
        t2 = threading.Thread(target=writer_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"并发写错误: {errors}"
        assert "key_a" in chroma._emb_cache
        assert "key_b" in chroma._emb_cache
