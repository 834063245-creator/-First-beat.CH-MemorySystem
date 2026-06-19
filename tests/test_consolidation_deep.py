"""测试 app/background/consolidation.py 深层分支 — 提行覆盖。

覆盖：on_idle 分支 / _review_today / _check_conflicts / _assess_archival
      / _generate_topic_notes / _load_notes / get_topic_notes / get_status
"""
import json
import os
import tempfile
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


def _make_engine(tmpdir, **overrides):
    from app.background.consolidation import ConsolidationEngine
    chroma = MagicMock()
    chroma.list_all.return_value = []
    chroma.list_all_cached.side_effect = lambda *a, **kw: chroma.list_all()
    chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
        m for m in chroma.list_all()
        if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
    ][:limit]
    chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
        m for m in chroma.list_all()
        if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
    ][:limit]
    chroma.list_all_paginated.side_effect = lambda *a, **kw: chroma.list_all()
    chroma._collection = MagicMock()
    chroma._emb_cache = {}
    chat_history = MagicMock()
    co_tracker = MagicMock()
    state_path = os.path.join(tmpdir, "dmn_state.json")
    notes_path = os.path.join(tmpdir, "topic_notes.json")
    return ConsolidationEngine(
        chroma, chat_history, co_tracker,
        state_path=state_path, notes_path=notes_path,
        **overrides,
    )


class TestOnIdle:
    def test_level_2_triggered(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            # idle >= 4h 触发 Level 2
            with patch.object(eng, '_review_today', return_value={"summary": "ok"}) as mock_review:
                with patch.object(eng, '_preheat_predictions') as mock_preheat:
                    eng.on_idle(5.0)
                    mock_review.assert_called_once()
                    mock_preheat.assert_called_once()

    def test_level_3_triggered(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            with patch.object(eng, '_consolidate_day', return_value={"summary": "ok"}) as mock_cons:
                with patch.object(eng, '_check_conflicts', return_value=[]) as mock_conf:
                    with patch.object(eng, '_review_today', return_value={"summary": "ok"}):
                        with patch.object(eng, '_preheat_predictions'):
                            eng.on_idle(13.0)
                            mock_cons.assert_called_once()
                            mock_conf.assert_called_once()

    def test_level_3_not_triggered_twice_same_day(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            state = eng._read_state()
            state["level3_triggered_today"] = True
            eng._write_state(state)
            with patch.object(eng, '_consolidate_day') as mock_cons:
                with patch.object(eng, '_review_today', return_value={"summary": "ok"}):
                    with patch.object(eng, '_preheat_predictions'):
                        eng.on_idle(13.0)
                        mock_cons.assert_not_called()

    def test_exceptions_handled(self):
        """内部异常不传播。"""
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            with patch.object(eng, '_review_today', side_effect=RuntimeError("fail")):
                with patch.object(eng, '_preheat_predictions', side_effect=RuntimeError("fail2")):
                    eng.on_idle(5.0)  # 不抛异常


class TestReviewToday:
    def test_empty_memories(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            eng._memory.list_all.return_value = []
            review = eng._review_today()
            assert review["total"] == 0

    def test_with_memories(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            now_ts = datetime.now().timestamp()
            eng._memory.list_all.return_value = [
                {"id": "m1", "metadata": {
                    "timestamp": now_ts, "summary": "学习了Rust编程",
                    "user_message": "我今天学了Rust", "emotional_intensity": 3,
                }},
            ]
            review = eng._review_today()
            assert review["total"] == 1
            assert review["emotional_count"] == 1  # intensity >= 2
            assert review["mood_warning"] is True  # 1/1 > 0.3

    def test_mood_warning_below_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            now_ts = datetime.now().timestamp()
            # 10 条记忆中只有 2 条情绪密集 → 2/10 = 0.2 < 0.3
            mems = []
            for i in range(8):
                mems.append({"id": f"n{i}", "metadata": {"timestamp": now_ts, "summary": "x", "emotional_intensity": 0}})
            mems.append({"id": "e1", "metadata": {"timestamp": now_ts, "summary": "y", "emotional_intensity": 3}})
            mems.append({"id": "e2", "metadata": {"timestamp": now_ts, "summary": "z", "emotional_intensity": 2}})
            eng._memory.list_all.return_value = mems
            review = eng._review_today()
            assert review["total"] == 10
            assert review["mood_warning"] is False  # 2/10 = 0.2 < 0.3


class TestCheckConflicts:
    def test_no_memories(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            conflicts = eng._check_conflicts()
            assert conflicts == []

    def test_no_old_memories(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            now_ts = datetime.now().timestamp()
            eng._memory.list_all.return_value = [
                {"id": "m1", "metadata": {"timestamp": now_ts, "tags": "咖啡, 生活"}},
            ]
            conflicts = eng._check_conflicts()
            assert conflicts == []

    def test_detects_tag_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            now_ts = datetime.now().timestamp()
            old_ts = now_ts - 86400 * 10  # 10 天前
            eng._memory.list_all.return_value = [
                {"id": "m1", "metadata": {"timestamp": now_ts, "tags": "咖啡, 工作", "summary": "喜欢喝咖啡"}},
                {"id": "m2", "metadata": {"timestamp": old_ts, "tags": "咖啡, 零食", "summary": "咖啡戒了"}},
            ]
            conflicts = eng._check_conflicts()
            assert len(conflicts) == 1
            assert "咖啡" in conflicts[0]["shared_tags"]


class TestLoadAndSaveNotes:
    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as td:
            from app.background.consolidation import ConsolidationEngine as CE
            notes = CE._load_notes(os.path.join(td, "nope.json"))
            assert notes == {}

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            from app.background.consolidation import ConsolidationEngine as CE
            p = os.path.join(td, "notes.json")
            CE._save_notes({"tag1": {"val": 1}}, p)
            loaded = CE._load_notes(p)
            assert loaded["tag1"]["val"] == 1

    def test_load_corrupted(self):
        with tempfile.TemporaryDirectory() as td:
            from app.background.consolidation import ConsolidationEngine as CE
            p = os.path.join(td, "notes.json")
            with open(p, "w") as f:
                f.write("not json")
            notes = CE._load_notes(p)
            assert notes == {}


class TestGetTopicNotes:
    def test_empty_tags(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            assert eng.get_topic_notes([]) == []

    def test_matches_tags(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            from app.background.consolidation import ConsolidationEngine as CE
            CE._save_notes(
                {"咖啡": {"tag": "咖啡", "memory_count": 5, "top_keywords": ["咖啡", "饮料"]}},
                eng._notes_path,
            )
            result = eng.get_topic_notes(["咖啡"])
            assert len(result) == 1
            assert result[0]["memory_count"] == 5

    def test_sorted_by_count(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            from app.background.consolidation import ConsolidationEngine as CE
            CE._save_notes(
                {"咖啡": {"tag": "咖啡", "memory_count": 5, "top_keywords": []},
                 "茶": {"tag": "茶", "memory_count": 10, "top_keywords": []}},
                eng._notes_path,
            )
            result = eng.get_topic_notes(["咖啡", "茶"])
            assert result[0]["memory_count"] >= result[1]["memory_count"]


class TestGetStatus:
    def test_returns_status_dict(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            status = eng.get_status()
            assert "last_shallow_consolidation" in status


class TestGetStateUpdate:
    def test_returns_state(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            update = eng.get_state_update()
            assert isinstance(update, dict)


class TestApplyToCognitiveState:
    def test_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            mock_cs = MagicMock()
            eng.apply_to_cognitive_state(mock_cs)  # 不崩溃


class TestAssessArchival:
    def test_empty(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            assert eng._assess_archival() == 0

    def test_small_cluster_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            now_ts = time.time()
            old_ts = now_ts - 86400 * 50  # 超过归档阈值
            eng._memory.list_all.return_value = [
                {"id": "m1", "metadata": {"tags": "罕见话题", "timestamp": old_ts, "last_hit_time": old_ts}},
                {"id": "m2", "metadata": {"tags": "罕见话题", "timestamp": old_ts, "last_hit_time": old_ts}},
            ]
            assert eng._assess_archival() == 0  # < 3 条，跳过


class TestGenerateTopicNotes:
    def test_empty(self):
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine(td)
            assert eng._generate_topic_notes() == 0


class TestLoadSaveState:
    def test_default_state_keys(self):
        from app.background.consolidation import _default_state
        s = _default_state()
        assert "pending_conflicts" in s
        assert "last_shallow_consolidation" in s
        assert "last_deep_consolidation" in s

    def test_load_nonexistent(self):
        from app.background.consolidation import _load_state
        s = _load_state("/nonexistent/path_state.json")
        assert "pending_conflicts" in s

    def test_save_and_load(self):
        from app.background.consolidation import _load_state, _save_state
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "state.json")
            _save_state({"pending_conflicts": [{"a": 1}]}, p)
            s = _load_state(p)
            assert len(s["pending_conflicts"]) == 1

    def test_load_corrupted(self):
        from app.background.consolidation import _load_state
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "state.json")
            with open(p, "w") as f:
                f.write("{{bad json")
            s = _load_state(p)
            assert "pending_conflicts" in s  # 回退默认
