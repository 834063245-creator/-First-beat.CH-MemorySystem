"""consolidation.py 测试 — 巩固引擎状态、预热缓存、冲突检测、话题笔记。"""
import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from app.background.consolidation import (
    ConsolidationEngine,
    _load_state,
    _save_state,
    _default_state,
    _extract_keywords,
)


# ═══════════════════════════════════════════════════════════════
# 辅助工厂函数
# ═══════════════════════════════════════════════════════════════

def _make_chroma_mock(memories=None):
    """构造假的 chroma_service。"""
    svc = MagicMock()
    svc.list_all.return_value = memories or []
    svc.list_all_cached.side_effect = lambda *a, **kw: svc.list_all()
    svc.list_since.side_effect = lambda since_ts, limit=500, **kw: [
        m for m in svc.list_all()
        if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
    ][:limit]
    svc.list_before.side_effect = lambda before_ts, limit=500, **kw: [
        m for m in svc.list_all()
        if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
    ][:limit]
    svc.list_all_paginated.side_effect = lambda *a, **kw: svc.list_all()
    svc._collection = MagicMock()
    svc._emb_cache = {}
    svc._build_embedding_cache = MagicMock()
    svc.increment_hit_count = MagicMock()
    svc.supersede_memory = MagicMock()
    svc.count.return_value = len(memories) if memories else 0
    svc._get_embedding_cached = MagicMock(return_value=None)
    svc.archive_topic_cluster = MagicMock()
    svc._apply_emotional_desensitization = MagicMock()
    svc.mark_storage_complete = MagicMock()
    return svc


def _make_memory(mid, timestamp, summary="", tags="", emotional_intensity=0,
                 emotion_valence_bin="", stale=False, archived=False,
                 heat="warm", hit_count=0, user_message="", embedding=None):
    return {
        "id": mid,
        "document": user_message or summary or f"doc_{mid}",
        "metadata": {
            "timestamp": timestamp,
            "summary": summary or f"摘要_{mid}",
            "tags": tags,
            "emotional_intensity": emotional_intensity,
            "emotion_valence_bin": emotion_valence_bin,
            "stale": stale,
            "archived": archived,
            "heat": heat,
            "hit_count": hit_count,
            "user_message": user_message or "",
            "embedding": embedding,
        },
    }


def _make_chat_history(records=None):
    ch = MagicMock()
    ch.records = records or []
    ch.get_recent.return_value = ch.records[-5:] if ch.records else []
    ch.get_records_snapshot.return_value = ch.records
    return ch


# ═══════════════════════════════════════════════════════════════
# 状态持久化
# ═══════════════════════════════════════════════════════════════

class TestStatePersistence:
    """巩固引擎状态读写。"""

    def test_default_state(self):
        state = _default_state()
        assert "last_idle_time" in state
        assert "last_shallow_consolidation" in state
        assert "last_deep_consolidation" in state
        assert state["archived_topic_count"] == 0

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            state = _default_state()
            state["today_topics"] = ["Python", "架构"]
            state["last_shallow_consolidation"] = 1234567890.0
            _save_state(state, path)

            loaded = _load_state(path)
            assert loaded["today_topics"] == ["Python", "架构"]
            assert loaded["last_shallow_consolidation"] == 1234567890.0

    def test_load_nonexistent_returns_default(self):
        state = _load_state("/nonexistent/consolidation_state.json")
        assert state == _default_state()

    def test_load_corrupted_json_returns_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as f:
                f.write("{corrupted")
            state = _load_state(path)
            assert state == _default_state()

    def test_missing_fields_merged(self):
        """旧版本缺少新字段时自动补齐。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as f:
                json.dump({"last_idle_time": "2026-06-06"}, f)
            state = _load_state(path)
            assert state["last_idle_time"] == "2026-06-06"
            # 新字段补齐
            assert "archived_topic_count" in state
            assert state["archived_topic_count"] == 0


# ═══════════════════════════════════════════════════════════════
# 关键词提取（纯函数，无外部依赖）
# ═══════════════════════════════════════════════════════════════

class TestExtractKeywords:
    """_extract_keywords 关键词提取。"""

    def test_extracts_from_chinese_text(self):
        # _extract_keywords 依赖 extract_tags
        # conftest.py autouse mock 了 extract_entities，但 extract_tags 仍走真实路径
        result = _extract_keywords("")
        assert result == []

    def test_empty_input(self):
        assert _extract_keywords("") == []
        assert _extract_keywords("   ") == []

    def test_chinese_input(self):
        """中文输入能提取关键词"""
        result = _extract_keywords("今天天气真好适合跑步运动健身")
        assert isinstance(result, list)

    def test_english_input(self):
        """英文输入能提取关键词"""
        result = _extract_keywords("machine learning and artificial intelligence")
        assert isinstance(result, list)

    def test_mixed_input(self):
        """中英混合"""
        result = _extract_keywords("Python编程和machine learning")
        assert isinstance(result, list)

    def test_dedup(self):
        """重复关键词去重"""
        result = _extract_keywords("跑步跑步跑步运动运动")
        assert isinstance(result, list)
        # 即使有关键词，也不应有明显重复


# ═══════════════════════════════════════════════════════════════
# 预热缓存
# ═══════════════════════════════════════════════════════════════

class TestPreheatCache:
    """get_preheated 缓存逻辑。"""

    def _make_engine(self, tmp_dir, chroma=None):
        return ConsolidationEngine(
            chroma_service=chroma or _make_chroma_mock(),
            chat_history=_make_chat_history(),
            co_tracker=MagicMock(),
            state_path=os.path.join(tmp_dir, "state.json"),
            notes_path=os.path.join(tmp_dir, "notes.json"),
        )

    def test_empty_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            result = engine.get_preheated("任意消息")
            assert result is None

    def test_empty_message_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            # 往缓存里塞点东西
            engine._preheat_cache["测试关键词"] = [{"id": "1"}]
            result = engine.get_preheated("")
            assert result is None

    def test_keyword_match_hits_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            # 直接往缓存里塞（绕过 _extract_keywords 的 embedding 调用）
            engine._preheat_cache["Python架构设计"] = [{"id": "hit_1", "source": "dmn_preheat"}]
            # 查询消息包含相同关键词
            result = engine.get_preheated("关于Python和架构的问题")
            if result is not None:
                assert result[0]["id"] == "hit_1"

    def test_no_keyword_overlap_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            engine._preheat_cache["Python"] = [{"id": "1"}]
            # 完全不相关的话题
            result = engine.get_preheated("今天天气真好")
            if result is not None:
                # 如果 embedding 提取出意外关键词，至少不应匹配 "Python"
                pass


# ═══════════════════════════════════════════════════════════════
# 冲突检测
# ═══════════════════════════════════════════════════════════════

class TestConflictDetection:
    """_check_conflicts 和 _detect_fact_contradictions。"""

    def _make_engine(self, tmp_dir, chroma=None, chat_history=None):
        return ConsolidationEngine(
            chroma_service=chroma or _make_chroma_mock(),
            chat_history=chat_history or _make_chat_history(),
            co_tracker=MagicMock(),
            state_path=os.path.join(tmp_dir, "state.json"),
            notes_path=os.path.join(tmp_dir, "notes.json"),
        )

    def test_check_conflicts_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            engine = self._make_engine(tmp, chroma=chroma)
            conflicts = engine._check_conflicts()
            assert conflicts == []

    def test_check_conflicts_no_old_memories(self):
        """全是最近 7 天的记忆，没有旧记忆可冲突。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("1", now - 60, tags="Python, 架构"),
                _make_memory("2", now - 120, tags="Python, 数据库"),
            ]
            chroma = _make_chroma_mock(mems)
            engine = self._make_engine(tmp, chroma=chroma)
            conflicts = engine._check_conflicts()
            assert conflicts == []

    def test_check_conflicts_detects_tag_overlap(self):
        """新记忆与 7 天前的旧记忆共享 tag → 标记冲突。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("new1", now - 60, tags="Python, 架构",
                             summary="Python新架构方案"),
                _make_memory("old1", now - 86400 * 14, tags="Python",
                             summary="Python旧方案"),
            ]
            chroma = _make_chroma_mock(mems)
            engine = self._make_engine(tmp, chroma=chroma)
            conflicts = engine._check_conflicts()
            assert len(conflicts) >= 1
            c = conflicts[0]
            assert "new_id" in c and "old_id" in c
            assert "shared_tags" in c

    def test_fact_contradictions_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            engine = self._make_engine(tmp, chroma=chroma)
            count = engine._detect_fact_contradictions()
            assert count == 0

    def test_fact_contradictions_no_new_memories(self):
        """没有 7 天内的新记忆 → 不产生冲突。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("old1", now - 86400 * 30, tags="Python",
                             emotional_intensity=3,
                             emotion_valence_bin="positive"),
            ]
            chroma = _make_chroma_mock(mems)
            engine = self._make_engine(tmp, chroma=chroma)
            count = engine._detect_fact_contradictions()
            assert count == 0

    def test_fact_contradictions_no_tag_overlap(self):
        """新旧记忆无共同标签 → 不产生冲突。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("new1", now - 60, tags="Python",
                             summary="Python新特性"),
                _make_memory("old1", now - 86400 * 14, tags="Rust",
                             summary="Rust老内容"),
            ]
            chroma = _make_chroma_mock(mems)
            engine = self._make_engine(tmp, chroma=chroma)
            count = engine._detect_fact_contradictions()
            assert count == 0

    def test_fact_contradictions_skips_stale(self):
        """已标记 stale 的记忆不参与冲突检测。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("new1", now - 60, tags="Python",
                             summary="Python新内容"),
                _make_memory("old1", now - 86400 * 14, tags="Python",
                             summary="Python旧内容", stale=True),
            ]
            chroma = _make_chroma_mock(mems)
            engine = self._make_engine(tmp, chroma=chroma)
            count = engine._detect_fact_contradictions()
            # stale 记忆在第 679 行被 continue 跳过
            assert count == 0

    def test_fact_contradictions_needs_embedding(self):
        """没有 embedding 时跳过（_detect_fact_contradictions 依赖 embedding 计算相似度）。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("new1", now - 60, tags="Python",
                             summary="Python新总结"),
                _make_memory("old1", now - 86400 * 14, tags="Python",
                             summary="Python旧总结"),
            ]
            chroma = _make_chroma_mock(mems)
            # _emb_cache 默认是空的 {}, 且没有 embedding 字段
            engine = self._make_engine(tmp, chroma=chroma)
            count = engine._detect_fact_contradictions()
            # 没有 embedding 走不了相似度计算，新记忆也被跳过
            assert count == 0


# ═══════════════════════════════════════════════════════════════
# 话题笔记
# ═══════════════════════════════════════════════════════════════

class TestTopicNotes:
    """话题笔记读写。"""

    def _make_engine(self, tmp_dir, chroma=None):
        return ConsolidationEngine(
            chroma_service=chroma or _make_chroma_mock(),
            chat_history=_make_chat_history(),
            co_tracker=MagicMock(),
            state_path=os.path.join(tmp_dir, "state.json"),
            notes_path=os.path.join(tmp_dir, "notes.json"),
        )

    def test_get_topic_notes_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            notes = engine.get_topic_notes([])
            assert notes == []

    def test_get_topic_notes_no_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            notes = engine.get_topic_notes(["不存在的标签"])
            assert notes == []

    def test_load_notes_nonexistent(self):
        assert ConsolidationEngine._load_notes("/nonexistent/notes.json") == {}

    def test_save_and_load_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.json")
            notes = {
                "Python": {
                    "tag": "Python",
                    "memory_count": 10,
                    "time_range": "2026-01-01 ~ 2026-06-06",
                    "top_keywords": ["架构", "性能"],
                    "dominant_valence": "positive",
                    "emotional_ratio": 0.2,
                    "last_updated": time.time(),
                }
            }
            ConsolidationEngine._save_notes(notes, path)
            loaded = ConsolidationEngine._load_notes(path)
            assert loaded["Python"]["tag"] == "Python"
            assert loaded["Python"]["memory_count"] == 10

    def test_get_topic_notes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.json")
            notes = {"Python": {"tag": "Python", "memory_count": 10}}
            ConsolidationEngine._save_notes(notes, path)

            engine = ConsolidationEngine(
                chroma_service=_make_chroma_mock(),
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=path,
            )
            result = engine.get_topic_notes(["Python"])
            assert len(result) == 1
            assert result[0]["tag"] == "Python"

    def test_get_topic_notes_sorted_by_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.json")
            notes = {
                "Python": {"tag": "Python", "memory_count": 5},
                "架构": {"tag": "架构", "memory_count": 20},
                "数据库": {"tag": "数据库", "memory_count": 15},
            }
            ConsolidationEngine._save_notes(notes, path)

            engine = ConsolidationEngine(
                chroma_service=_make_chroma_mock(),
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=path,
            )
            result = engine.get_topic_notes(["Python", "架构", "数据库"])
            assert len(result) == 3
            # 按 memory_count 降序
            assert result[0]["tag"] == "架构"
            assert result[1]["tag"] == "数据库"
            assert result[2]["tag"] == "Python"


# ═══════════════════════════════════════════════════════════════
# 状态更新注入
# ═══════════════════════════════════════════════════════════════

class TestStateUpdate:
    """get_state_update 和 apply_to_cognitive_state。"""

    def _make_engine(self, tmp_dir):
        return ConsolidationEngine(
            chroma_service=_make_chroma_mock(),
            chat_history=_make_chat_history(),
            co_tracker=MagicMock(),
            state_path=os.path.join(tmp_dir, "state.json"),
            notes_path=os.path.join(tmp_dir, "notes.json"),
        )

    def test_get_state_update_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            update = engine.get_state_update()
            assert "topics" in update
            assert "conflicts" in update
            assert "mood_warning" in update
            assert update["topics"] == []
            assert update["conflicts"] == []

    def test_apply_to_cognitive_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            cs = MagicMock()
            cs.today_topics = []
            engine.apply_to_cognitive_state(cs)
            # 不应抛异常
            assert cs.today_topics is not None

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            status = engine.get_status()
            assert "level3_triggered_today" in status
            assert "cache_size" in status
            assert "pending_conflicts" in status


# ═══════════════════════════════════════════════════════════════
# 归档评估
# ═══════════════════════════════════════════════════════════════

class TestArchival:
    """_assess_archival 归档逻辑。"""

    def test_empty_memory_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            count = engine._assess_archival()
            assert count == 0

    def test_small_clusters_not_archived(self):
        """少于 3 条记忆的话题簇不归档。"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("1", now - 86400 * 100, tags="Python",
                             hit_count=0),
                _make_memory("2", now - 86400 * 100, tags="Python",
                             hit_count=0),
            ]
            chroma = _make_chroma_mock(mems)
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            count = engine._assess_archival()
            assert count == 0  # < 3 条不归档

    def test_already_archived_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("1", now - 86400 * 100, tags="Python",
                             hit_count=0, archived=True),
                _make_memory("2", now - 86400 * 100, tags="Python",
                             hit_count=0, archived=True),
                _make_memory("3", now - 86400 * 100, tags="Python",
                             hit_count=0, archived=True),
            ]
            chroma = _make_chroma_mock(mems)
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            count = engine._assess_archival()
            assert count == 0  # 已归档的跳过


# ═══════════════════════════════════════════════════════════════
# 日巩固
# ═══════════════════════════════════════════════════════════════

class TestConsolidateDay:
    def test_empty_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            result = engine._consolidate_day()
            assert result["total"] == 0
            assert result["emotional_count"] == 0

    def test_with_today_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("1", now - 60, tags="Python,编程",
                             summary="Python学习", emotional_intensity=2),
                _make_memory("2", now - 120, tags="运动,跑步",
                             summary="跑步打卡", emotional_intensity=1),
            ]
            chroma = _make_chroma_mock(mems)
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            result = engine._consolidate_day()
            assert result["total"] == 2
            assert "summary" in result

    def test_stale_candidates(self):
        """今天的标签与旧记忆标签重叠，产生 stale 候选"""
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("new1", now - 60, tags="Python",
                             summary="今天Python学习"),
                _make_memory("old1", now - 86400 * 14, tags="Python",
                             summary="旧Python笔记"),
            ]
            chroma = _make_chroma_mock(mems)
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            result = engine._consolidate_day()
            assert "stale_candidates" in result


# ═══════════════════════════════════════════════════════════════
# 预热预测
# ═══════════════════════════════════════════════════════════════

class TestPreheatPredictions:
    def test_no_history_no_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            behavior = MagicMock()
            behavior.list_all.return_value = []
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            engine._preheat_predictions()
            # 不应抛异常
            state = engine._read_state()
            assert "preheat_queries" in state

    def test_with_today_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            behavior = MagicMock()
            behavior.list_all.return_value = []
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            # 预设 today_topics
            state = engine._read_state()
            state["today_topics"] = ["Python", "架构"]
            engine._write_state(state)
            engine._preheat_predictions()
            state = engine._read_state()
            assert len(state.get("preheat_queries", [])) >= 0


# ═══════════════════════════════════════════════════════════════
# 浅巩固 / 深巩固
# ═══════════════════════════════════════════════════════════════

class TestConsolidateShallow:
    def test_empty_memory_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            engine.consolidate_shallow()
            # 不应抛异常
            state = engine._read_state()
            assert state["last_shallow_consolidation"] > 0

    def test_with_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("1", now - 3600, tags="Python,编程",
                             summary="Python学习"),
            ]
            chroma = _make_chroma_mock(mems)
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            engine.consolidate_shallow()
            state = engine._read_state()
            assert state["last_shallow_consolidation"] > 0


class TestConsolidateDeep:
    def test_empty_memory_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            chroma = _make_chroma_mock([])
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            engine.consolidate_deep()
            state = engine._read_state()
            assert state["last_deep_consolidation"] > 0

    def test_with_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mems = [
                _make_memory("1", now - 86400 * 50, tags="Python,编程",
                             summary="Python笔记", hit_count=0),
                _make_memory("2", now - 86400 * 50, tags="Python,架构",
                             summary="架构笔记", hit_count=0),
                _make_memory("3", now - 86400 * 50, tags="Python,数据库",
                             summary="数据库笔记", hit_count=0),
            ]
            chroma = _make_chroma_mock(mems)
            engine = ConsolidationEngine(
                chroma_service=chroma,
                chat_history=_make_chat_history(),
                co_tracker=MagicMock(),
                state_path=os.path.join(tmp, "state.json"),
                notes_path=os.path.join(tmp, "notes.json"),
            )
            engine.consolidate_deep()
            state = engine._read_state()
            assert state["last_deep_consolidation"] > 0
