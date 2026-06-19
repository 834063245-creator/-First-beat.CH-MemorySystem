"""测试 app/portrait/writer.py — 画像写入引擎。

覆盖: 静态方法、提取辅助、数据拉取逻辑（实时层浅层测试）。
深巩固/浅巩固的 LLM 调用路径通过 mock 覆盖。
"""
import json
import os
import tempfile
import time
from datetime import datetime
from collections import Counter
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from app.portrait.writer import PortraitWriter
from app.portrait.manager import PortraitManager
from app.portrait.state import PortraitEntry, EntryStatus


# ═══════════════════════════════════════════════════════════════════
# _extract_emotion_from_text
# ═══════════════════════════════════════════════════════════════════

class TestExtractEmotionFromText:
    def test_extracts_english_emotion(self):
        assert PortraitWriter._extract_emotion_from_text("**情绪**: positive") == "positive"
        assert PortraitWriter._extract_emotion_from_text("**情绪**: negative") == "negative"
        assert PortraitWriter._extract_emotion_from_text("**情绪**: frustrated") == "frustrated"

    def test_extracts_with_parentheses(self):
        assert PortraitWriter._extract_emotion_from_text(
            "**情绪**: positive （待验证）"
        ) == "positive"

    def test_extracts_chinese_mapping(self):
        assert PortraitWriter._extract_emotion_from_text("**情绪**: 低落") == "negative"
        assert PortraitWriter._extract_emotion_from_text("**情绪**: 开心") == "positive"
        assert PortraitWriter._extract_emotion_from_text("**情绪**: 平静") == "neutral"

    def test_no_match_returns_none(self):
        assert PortraitWriter._extract_emotion_from_text("没有情绪标记") is None
        assert PortraitWriter._extract_emotion_from_text("") is None

    def test_partial_chinese_in_emotion(self):
        """匹配中文并映射"""
        result = PortraitWriter._extract_emotion_from_text("**情绪**: 很焦虑")
        assert result == "negative"  # "焦虑" in emotion


# ═══════════════════════════════════════════════════════════════════
# _get_time_period
# ═══════════════════════════════════════════════════════════════════

class TestGetTimePeriod:
    def test_morning(self):
        with patch("app.config.settings.TIME_PERIOD_MAP", {(6, 11): "早上", (12, 17): "下午"}):
            result = PortraitWriter._get_time_period(8)
            assert result == "早上"

    def test_afternoon(self):
        with patch("app.config.settings.TIME_PERIOD_MAP", {(6, 11): "早上", (12, 17): "下午"}):
            result = PortraitWriter._get_time_period(14)
            assert result == "下午"

    def test_default_to_evening(self):
        with patch("app.config.settings.TIME_PERIOD_MAP", {(6, 11): "早上"}):
            result = PortraitWriter._get_time_period(23)
            assert result == "晚上"


# ═══════════════════════════════════════════════════════════════════
# _call_local_llm
# ═══════════════════════════════════════════════════════════════════

class TestCallLocalLLM:
    def test_returns_result_on_success(self):
        with patch("app.llm.local.LocalLLM") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "合成结果"
            mock_llm_cls.return_value = mock_llm

            result = PortraitWriter._call_local_llm("测试 prompt")
            assert result == "合成结果"
            mock_llm.generate.assert_called_once()

    def test_returns_none_on_import_error(self):
        with patch("app.llm.local.LocalLLM", side_effect=ImportError("no module")):
            result = PortraitWriter._call_local_llm("测试 prompt")
            assert result is None

    def test_returns_none_on_generate_exception(self):
        with patch("app.llm.local.LocalLLM") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.generate.side_effect = RuntimeError("LLM down")
            mock_llm_cls.return_value = mock_llm

            result = PortraitWriter._call_local_llm("测试 prompt")
            assert result is None

    def test_returns_none_when_generate_returns_empty(self):
        with patch("app.llm.local.LocalLLM") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = ""
            mock_llm_cls.return_value = mock_llm

            result = PortraitWriter._call_local_llm("测试 prompt")
            assert result is None  # "" is falsy


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — 初始化
# ═══════════════════════════════════════════════════════════════════

class TestPortraitWriterInit:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initial_state(self, writer):
        assert writer._last_shallow_update == 0.0
        assert writer._last_deep_update == 0.0
        assert writer._turns_since_last_deep == 0

    def test_manager_reference(self, writer):
        assert writer._manager is not None
        assert isinstance(writer._manager, PortraitManager)


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — realtime_update_user
# ═══════════════════════════════════════════════════════════════════

class TestRealtimeUpdateUser:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_creates_emotion_entry_first_time(self, writer):
        """首次调用应创建 usr2-001（情绪）和 usr2-002（如有话题）"""
        user_mock = MagicMock()
        user_mock.emotion = "positive"
        user_mock.topics = []

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        writer.realtime_update_user(utterance_mock, rel_mock)

        entry = writer._manager.get_entry("usr2-001")
        assert entry is not None
        assert "情绪" in entry.text
        assert "positive" in entry.text
        assert "待验证" in entry.text or entry.status == EntryStatus.PENDING

    def test_updates_existing_emotion_same(self, writer):
        """相同情绪 → 直接更新（不标记翻转）"""
        # 先创建一个已有条目
        writer._manager.set_entry(
            "usr2-001", "**情绪**: positive",
            status=EntryStatus.ACTIVE,
            last_observed=datetime.now().isoformat(),
        )

        user_mock = MagicMock()
        user_mock.emotion = "positive"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr2-001")
        assert entry is not None
        # 相同情绪 → ACTIVE
        assert entry.status == EntryStatus.ACTIVE

    def test_emotion_flip_detected(self, writer):
        """情绪翻转为 PENDING"""
        writer._manager.set_entry(
            "usr2-001", "**情绪**: positive",
            status=EntryStatus.ACTIVE,
            last_observed=datetime.now().isoformat(),
        )

        user_mock = MagicMock()
        user_mock.emotion = "negative"  # flip!
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr2-001")
        assert entry is not None
        assert entry.status == EntryStatus.PENDING
        assert "情绪翻转" in entry.text or "negative" in entry.text

    def test_creates_focus_entry_with_topics(self, writer):
        """有话题时应创建 usr2-002"""
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = ["Python", "微服务", "Docker"]

        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr2-002")
        assert entry is not None
        assert "关注焦点" in entry.text
        assert "Python" in entry.text

    def test_creates_trust_entry(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = 0.75
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr4-001")
        assert entry is not None
        assert "信任度" in entry.text
        assert "0.75" in entry.text

    def test_creates_closeness_entry(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = 0.60
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr4-002")
        assert entry is not None
        assert "亲密度" in entry.text

    def test_creates_familiarity_entry(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = 0.50
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr4-003")
        assert entry is not None
        assert "熟悉度" in entry.text

    def test_creates_interaction_mode_entry(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = "collaborator"

        writer.realtime_update_user(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("usr4-004")
        assert entry is not None
        assert "互动模式" in entry.text
        assert "collaborator" in entry.text

    def test_skips_none_relationship_fields(self, writer):
        """None 的关系字段不应创建条目"""
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        writer.realtime_update_user(utterance_mock, rel_mock)
        # usr4-001/002/003/004 都不应被创建
        assert writer._manager.get_entry("usr4-001") is None
        assert writer._manager.get_entry("usr4-002") is None
        assert writer._manager.get_entry("usr4-003") is None
        assert writer._manager.get_entry("usr4-004") is None


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — realtime_update_ai
# ═══════════════════════════════════════════════════════════════════

class TestRealtimeUpdateAI:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_creates_ai_emotion_tone_entry(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "positive"
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.interaction_mode = None
        rel_mock.trust = None

        writer.realtime_update_ai(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("ai2-001")
        assert entry is not None
        assert "表达色调" in entry.text
        assert "positive" in entry.text

    def test_creates_ai_relationship_entry_with_trust(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.interaction_mode = None
        rel_mock.trust = 0.80

        writer.realtime_update_ai(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("ai4-001")
        assert entry is not None
        assert "信任" in entry.text

    def test_creates_ai_interaction_stage_entry(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.interaction_mode = "partner"
        rel_mock.trust = None

        writer.realtime_update_ai(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("ai4-003")
        assert entry is not None
        assert "伙伴关系" in entry.text

    def test_unknown_interaction_mode(self, writer):
        """未知互动模式直接使用原值"""
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.interaction_mode = "unknown_mode"
        rel_mock.trust = None

        writer.realtime_update_ai(utterance_mock, rel_mock)
        entry = writer._manager.get_entry("ai4-003")
        assert entry is not None
        assert "unknown_mode" in entry.text


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — realtime_update (入口)
# ═══════════════════════════════════════════════════════════════════

class TestRealtimeUpdate:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_calls_both_user_and_ai(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "positive"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock

        rel_mock = MagicMock()
        rel_mock.trust = 0.70
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = "casual"

        writer.realtime_update(utterance_mock, rel_mock)

        # 用户侧
        assert writer._manager.get_entry("usr2-001") is not None
        assert writer._manager.get_entry("usr4-001") is not None
        # AI 侧
        assert writer._manager.get_entry("ai2-001") is not None
        assert writer._manager.get_entry("ai4-001") is not None

    def test_increments_turns_counter(self, writer):
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock
        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        assert writer._turns_since_last_deep == 0
        writer.realtime_update(utterance_mock, rel_mock)
        assert writer._turns_since_last_deep == 1
        writer.realtime_update(utterance_mock, rel_mock)
        assert writer._turns_since_last_deep == 2

    def test_calls_save(self, writer):
        """实时更新后应调用 manager.save()"""
        user_mock = MagicMock()
        user_mock.emotion = "neutral"
        user_mock.topics = []
        utterance_mock = MagicMock()
        utterance_mock.user = user_mock
        rel_mock = MagicMock()
        rel_mock.trust = None
        rel_mock.closeness = None
        rel_mock.familiarity = None
        rel_mock.interaction_mode = None

        with patch.object(writer._manager, "save") as mock_save:
            writer.realtime_update(utterance_mock, rel_mock)
            mock_save.assert_called_once()

    def test_user_exception_does_not_block_ai(self, writer):
        """用户侧异常不应阻止 AI 侧执行"""
        writer.realtime_update_user = MagicMock(side_effect=RuntimeError("user error"))
        writer.realtime_update_ai = MagicMock()

        writer.realtime_update(MagicMock(), MagicMock())
        writer.realtime_update_ai.assert_called_once()  # AI 侧仍执行


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — _pull_tag_stats
# ═══════════════════════════════════════════════════════════════════

class TestPullTagStats:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_empty_chroma_returns_empty(self, writer):
        chroma_mock = MagicMock()
        chroma_mock.list_all_cached.return_value = []
        result = writer._pull_tag_stats(chroma_mock)
        assert result == {}

    def test_extracts_tag_stats(self, writer):
        now = time.time()
        chroma_mock = MagicMock()
        chroma_mock.list_all_cached.return_value = [
            {
                "id": "mem-1",
                "metadata": {"tags": "Python, 编程", "timestamp": now - 86400},
            },
            {
                "id": "mem-2",
                "metadata": {"tags": "Python, Rust", "timestamp": now - 86400 * 5},
            },
        ]
        result = writer._pull_tag_stats(chroma_mock)
        assert "Python" in result
        assert result["Python"]["count"] == 2
        assert "编程" in result
        assert result["编程"]["count"] == 1
        assert "Rust" in result

    def test_handles_missing_metadata(self, writer):
        chroma_mock = MagicMock()
        chroma_mock.list_all_cached.return_value = [{"id": "mem-1"}]
        result = writer._pull_tag_stats(chroma_mock)
        assert result == {}

    def test_handles_empty_tags(self, writer):
        chroma_mock = MagicMock()
        chroma_mock.list_all_cached.return_value = [
            {"id": "mem-1", "metadata": {"tags": "", "timestamp": time.time()}},
        ]
        result = writer._pull_tag_stats(chroma_mock)
        assert result == {}

    def test_handles_exception_gracefully(self, writer):
        storage_mock = MagicMock()
        storage_mock.list_all_cached.side_effect = RuntimeError("Qdrant down")
        result = writer._pull_tag_stats(storage_mock)
        assert result == {}


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — _pull_emotion_data
# ═══════════════════════════════════════════════════════════════════

class TestPullEmotionData:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_extracts_emotion_triggers(self, writer):
        ctx_mock = MagicMock()
        ctx_mock.memory_service.list_all_cached.return_value = [
            {
                "metadata": {
                    "emotion_valence": 0.5,
                    "emotion_valence_bin": "positive",
                    "emotional_intensity": 3,
                    "tags": "Python, 编程",
                },
            },
            {
                "metadata": {
                    "emotion_valence": -0.5,
                    "emotion_valence_bin": "negative",
                    "emotional_intensity": 4,
                    "tags": "压力, 工作",
                },
            },
        ]
        result = writer._pull_emotion_data(ctx_mock)
        assert "positive_triggers" in result
        assert "negative_triggers" in result
        # Python → positive trigger
        assert ("Python", 1) in result["positive_triggers"]
        # 压力 → negative trigger
        assert ("压力", 1) in result["negative_triggers"]

    def test_handles_exception_gracefully(self, writer):
        ctx_mock = MagicMock()
        ctx_mock.memory_service.list_all_cached.side_effect = RuntimeError("fail")
        result = writer._pull_emotion_data(ctx_mock)
        assert result == {}

    def test_no_emotion_data_returns_empty_dicts(self, writer):
        ctx_mock = MagicMock()
        ctx_mock.memory_service.list_all_cached.return_value = [
            {"metadata": {"tags": "test"}},
        ]
        result = writer._pull_emotion_data(ctx_mock)
        # 所有 valence 为 0 → neutral，不触发任何方向
        assert len(result.get("positive_triggers", [])) == 0
        assert len(result.get("negative_triggers", [])) == 0


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — _pull_temporal
# ═══════════════════════════════════════════════════════════════════

class TestPullTemporal:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_no_temporal_source_returns_empty(self, writer):
        ctx_mock = MagicMock(spec=[])
        result = writer._pull_temporal(ctx_mock)
        assert result == {}

    def test_with_temporal_patterns(self, writer):
        ctx_mock = MagicMock()
        ctx_mock.temporal_pattern_index = MagicMock()
        ctx_mock.temporal_pattern_index.query.return_value = ["pattern1"]
        ctx_mock.mirror_neuron = None

        result = writer._pull_temporal(ctx_mock)
        assert "current_patterns" in result

    def test_with_mirror_neuron_predictions(self, writer):
        ctx_mock = MagicMock()
        ctx_mock.temporal_pattern_index = None
        ctx_mock.mirror_neuron = MagicMock()
        ctx_mock.mirror_neuron._table = {"a": 1}

        result = writer._pull_temporal(ctx_mock)
        assert "predictions" in result


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — _apply_llm_dim_update
# ═══════════════════════════════════════════════════════════════════

class TestApplyLLMDimUpdate:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_applies_single_entry_update(self, writer):
        llm_output = "<!-- entry:usr1-001 -->\n- 用户喜欢钻研新技术"
        writer._apply_llm_dim_update("usr1", llm_output)
        entry = writer._manager.get_entry("usr1-001")
        assert entry is not None
        assert "钻研新技术" in entry.text

    def test_applies_multiple_entries(self, writer):
        llm_output = (
            "<!-- entry:usr5-001 -->\n- Python开发\n"
            "<!-- entry:usr5-002 -->\n- Rust学习\n"
            "<!-- entry:usr5-003 -->\n- 微服务架构"
        )
        writer._apply_llm_dim_update("usr5", llm_output)
        assert writer._manager.get_entry("usr5-001") is not None
        assert writer._manager.get_entry("usr5-002") is not None
        assert writer._manager.get_entry("usr5-003") is not None

    def test_replaces_old_entries(self, writer):
        """LLM 输出应完全替代旧条目"""
        # 先创建旧条目
        writer._manager.set_entry("usr1-001", "旧特征1")
        writer._manager.set_entry("usr1-002", "旧特征2")

        llm_output = "<!-- entry:usr1-003 -->\n- 新特征"
        writer._apply_llm_dim_update("usr1", llm_output)

        # 旧条目应被删除
        assert writer._manager.get_entry("usr1-001") is None
        assert writer._manager.get_entry("usr1-002") is None
        # 新条目应存在
        assert writer._manager.get_entry("usr1-003") is not None

    def test_no_entry_ids_skips_update(self, writer):
        """没有有效 entry ID 的输出应被跳过"""
        writer._manager.set_entry("usr1-001", "原始条目")
        writer._apply_llm_dim_update("usr1", "没有 entry ID 的文本")
        # 原始条目应保留
        assert writer._manager.get_entry("usr1-001") is not None

    def test_strips_backtick_meta_from_llm_output(self, writer):
        llm_output = "<!-- entry:usr1-001 -->\n- 新特征 `高 · 3条证据`"
        writer._apply_llm_dim_update("usr1", llm_output)
        entry = writer._manager.get_entry("usr1-001")
        assert entry is not None
        assert "`" not in entry.text

    def test_strips_list_marker(self, writer):
        llm_output = "<!-- entry:usr1-001 -->\n- 条目内容"
        writer._apply_llm_dim_update("usr1", llm_output)
        entry = writer._manager.get_entry("usr1-001")
        assert entry is not None
        assert not entry.text.startswith("- ")
        assert entry.text == "条目内容"


# ═══════════════════════════════════════════════════════════════════
# PortraitWriter — shallow / deep update (guard clauses)
# ═══════════════════════════════════════════════════════════════════

class TestShallowUpdateGuard:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_too_soon_skips(self, writer):
        """距上次更新不足 PORTRAIT_SHALLOW_HOURS → 跳过"""
        writer._last_shallow_update = time.time()  # 刚更新过
        with patch("app.config.settings.PORTRAIT_SHALLOW_HOURS", 4):
            with patch.object(writer, "_shallow_update_user") as mock_user:
                ctx = MagicMock()
                writer.shallow_update(ctx)
                mock_user.assert_not_called()

    def test_enough_time_passes(self, writer):
        """距上次更新足够久 → 执行浅巩固"""
        writer._last_shallow_update = time.time() - 5 * 3600  # 5小时前
        with patch("app.config.settings.PORTRAIT_SHALLOW_HOURS", 4):
            with patch.object(writer, "_shallow_update_user") as mock_user:
                with patch.object(writer, "_shallow_update_ai") as mock_ai:
                    with patch.object(writer._manager, "apply_state_machine"):
                        with patch.object(writer._manager, "save"):
                            ctx = MagicMock()
                            writer.shallow_update(ctx)
                            mock_user.assert_called_once()
                            mock_ai.assert_called_once()


class TestDeepUpdateGuard:
    @pytest.fixture
    def writer(self):
        tmpdir = tempfile.mkdtemp()
        path = f"{tmpdir}/PORTRAIT.md"
        mgr = PortraitManager(file_path=path)
        w = PortraitWriter(mgr)
        yield w
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_not_enough_turns_skips(self, writer):
        """距上次深巩固不足 PORTRAIT_DEEP_MIN_TURNS → 跳过"""
        writer._turns_since_last_deep = 1
        with patch("app.config.settings.PORTRAIT_DEEP_MIN_TURNS", 20):
            with patch.object(writer, "_deep_update_user_core") as mock_core:
                ctx = MagicMock()
                writer.deep_update(ctx)
                mock_core.assert_not_called()

    def test_not_enough_time_skips(self, writer):
        writer._turns_since_last_deep = 30  # 足够轮数
        writer._last_deep_update = time.time()  # 但时间不够
        with patch("app.config.settings.PORTRAIT_DEEP_MIN_TURNS", 20):
            with patch("app.config.settings.PORTRAIT_DEEP_HOURS", 24):
                with patch.object(writer, "_deep_update_user_core") as mock_core:
                    ctx = MagicMock()
                    writer.deep_update(ctx)
                    mock_core.assert_not_called()

    def test_conditions_met_executes(self, writer):
        writer._turns_since_last_deep = 30
        writer._last_deep_update = time.time() - 25 * 3600
        with patch("app.config.settings.PORTRAIT_DEEP_MIN_TURNS", 20):
            with patch("app.config.settings.PORTRAIT_DEEP_HOURS", 24):
                with patch.object(writer, "_deep_update_user_core") as mock_user:
                    with patch.object(writer, "_deep_update_ai_core") as mock_ai:
                        with patch.object(writer, "_cross_dimension_review"):
                            with patch.object(writer._manager, "apply_state_machine"):
                                with patch.object(writer._manager, "save"):
                                    ctx = MagicMock()
                                    writer.deep_update(ctx)
                                    mock_user.assert_called_once()
                                    mock_ai.assert_called_once()
                                    assert writer._turns_since_last_deep == 0
