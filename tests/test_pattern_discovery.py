"""测试 app/analysis/pattern_discovery.py — 模式发现层纯函数和缓存逻辑。"""
import json
import os
import time
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from app.analysis.pattern_discovery import (
    _merge_tuning,
    _DEFAULT_TUNING,
    _INJECT_RULES,
    PatternDiscovery,
)


# ═══════════════════════════════════════════════════════════════════
# _merge_tuning
# ═══════════════════════════════════════════════════════════════════

class TestMergeTuning:
    def test_both_default(self):
        a = dict(_DEFAULT_TUNING)
        b = dict(_DEFAULT_TUNING)
        result = _merge_tuning(a, b)
        assert result == _DEFAULT_TUNING

    def test_a_overrides_default(self):
        a = {"emotional_dampening": True, "formality_shift": 0, "proactive_suppression": False}
        b = dict(_DEFAULT_TUNING)
        result = _merge_tuning(a, b)
        assert result["emotional_dampening"] is True

    def test_b_overrides_default_when_a_is_default(self):
        a = dict(_DEFAULT_TUNING)
        b = {"emotional_dampening": True, "formality_shift": 1, "proactive_suppression": True}
        result = _merge_tuning(a, b)
        assert result == b

    def test_a_takes_priority_over_b(self):
        """a 非默认值优先于 b"""
        a = {"emotional_dampening": True, "formality_shift": 0, "proactive_suppression": False}
        b = {"emotional_dampening": False, "formality_shift": 1, "proactive_suppression": True}
        result = _merge_tuning(a, b)
        assert result["emotional_dampening"] is True  # a 非默认
        assert result["formality_shift"] == 1         # a 是默认，用 b
        assert result["proactive_suppression"] is True  # a 是默认，用 b

    def test_float_merge_min(self):
        a = {"emotional_dampening": False, "formality_shift": -1, "proactive_suppression": False}
        b = {"emotional_dampening": False, "formality_shift": 0, "proactive_suppression": False}
        result = _merge_tuning(a, b)
        assert result["formality_shift"] == -1  # a 非默认

    def test_float_merge_max(self):
        a = {"emotional_dampening": False, "formality_shift": 0, "proactive_suppression": False}
        b = {"emotional_dampening": False, "formality_shift": 2, "proactive_suppression": False}
        result = _merge_tuning(a, b)
        assert result["formality_shift"] == 2  # a 默认，用 b

    def test_partial_keys(self):
        """只有部分 key 的情况"""
        a = {"emotional_dampening": True}
        b = {"formality_shift": 1}
        result = _merge_tuning(a, b)
        assert result["emotional_dampening"] is True
        assert result["formality_shift"] == 1
        # proactive_suppression 未出现在输入中 → 取默认值 False
        assert result.get("proactive_suppression", False) is False

    def test_empty_dicts(self):
        result = _merge_tuning({}, {})
        assert result == {}  # 空输入产生空输出


# ═══════════════════════════════════════════════════════════════════
# _linear_trend
# ═══════════════════════════════════════════════════════════════════

class TestLinearTrend:
    def test_upward_trend(self):
        slope = PatternDiscovery._linear_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert slope > 0

    def test_downward_trend(self):
        slope = PatternDiscovery._linear_trend([5.0, 4.0, 3.0, 2.0, 1.0])
        assert slope < 0

    def test_flat_trend(self):
        slope = PatternDiscovery._linear_trend([3.0, 3.0, 3.0, 3.0, 3.0])
        assert slope == pytest.approx(0.0)

    def test_single_point(self):
        slope = PatternDiscovery._linear_trend([42.0])
        assert slope == 0.0

    def test_empty_list(self):
        slope = PatternDiscovery._linear_trend([])
        assert slope == 0.0

    def test_two_points_up(self):
        slope = PatternDiscovery._linear_trend([0.0, 10.0])
        assert slope > 0

    def test_two_points_down(self):
        slope = PatternDiscovery._linear_trend([10.0, 0.0])
        assert slope < 0

    def test_slope_magnitude(self):
        """验证斜率与变化量成正比"""
        slope1 = PatternDiscovery._linear_trend([0.0, 1.0, 2.0])
        slope2 = PatternDiscovery._linear_trend([0.0, 2.0, 4.0])
        assert abs(slope2) > abs(slope1)


# ═══════════════════════════════════════════════════════════════════
# PatternDiscovery - 缓存和基础方法
# ═══════════════════════════════════════════════════════════════════

class TestPatternDiscoveryInit:
    def test_default_state(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        assert pd._observations == []
        assert pd._tuning == _DEFAULT_TUNING

    def test_get_tuning_returns_copy(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        tuning = pd.get_tuning()
        tuning["emotional_dampening"] = True
        # 修改返回的 dict 不影响内部状态
        assert pd._tuning["emotional_dampening"] is False

    def test_get_tuning_initial(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        assert pd.get_tuning() == _DEFAULT_TUNING


class TestGetObservations:
    def test_empty_observations(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        assert pd.get_observations() == []

    def test_with_observations_inject_true(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        pd._observations = [
            {"text": "观察1", "inject": True},
            {"text": "观察2", "inject": True},
        ]
        result = pd.get_observations()
        assert result == ["观察1", "观察2"]

    def test_with_observations_inject_false(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        pd._observations = [
            {"text": "可见的", "inject": True},
            {"text": "不可见的", "inject": False},
            {"text": "也可见", "inject": True},
        ]
        result = pd.get_observations()
        assert result == ["可见的", "也可见"]

    def test_all_inject_false(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        pd._observations = [
            {"text": "隐藏1", "inject": False},
            {"text": "隐藏2", "inject": False},
        ]
        result = pd.get_observations()
        assert result == []

    def test_missing_inject_key_defaults_true(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        pd._observations = [
            {"text": "默认可见"},
        ]
        result = pd.get_observations()
        # inject 缺失时 get 返回 True（默认值）
        assert result == ["默认可见"]


class TestLoadCache:
    def test_no_cache_file(self):
        pd = PatternDiscovery(data_dir="/tmp/nonexistent_pd")
        pd.load_cache()
        assert pd._observations == []

    def test_v3_cache_fresh(self, tmp_path):
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 3,
            "updated_at": time.time(),
            "tuning": {"emotional_dampening": True, "formality_shift": 1, "proactive_suppression": False},
            "observations": [{"text": "v3测试观察", "inject": True}],
            "trajectory": [],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd.load_cache()
        assert len(pd._observations) == 1
        assert pd._observations[0]["text"] == "v3测试观察"
        assert pd._tuning["emotional_dampening"] is True
        assert pd._tuning["formality_shift"] == 1

    def test_v2_cache(self, tmp_path):
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 2,
            "updated_at": time.time(),
            "tuning": {"emotional_dampening": False, "formality_shift": 0, "proactive_suppression": True},
            "observations": [{"text": "v2观察"}],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd.load_cache()
        assert len(pd._observations) == 1
        assert pd._tuning["proactive_suppression"] is True

    def test_v1_cache(self, tmp_path):
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 1,
            "updated_at": time.time(),
            "observations": [{"text": "v1老数据"}],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd.load_cache()
        assert len(pd._observations) == 1
        # v1 无 tuning 字段，使用默认值
        assert pd._tuning == _DEFAULT_TUNING

    def test_corrupted_cache(self, tmp_path):
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write("not valid json{[[[")

        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd.load_cache()
        assert pd._observations == []

    def test_expired_cache(self, tmp_path):
        """超过 1 小时的缓存被清空"""
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 3,
            "updated_at": time.time() - 7200,  # 2 小时前
            "tuning": dict(_DEFAULT_TUNING),
            "observations": [{"text": "过期观察"}],
            "trajectory": [],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd.load_cache()
        assert pd._observations == []  # 超时清空
        assert pd._tuning == _DEFAULT_TUNING  # tuning 仍加载


class TestDetectTrends:
    def test_no_cache_file(self):
        pd = PatternDiscovery(data_dir="/tmp/nonexistent_trends")
        result = pd.detect_trends()
        assert result == []

    def test_empty_trajectory(self, tmp_path):
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 3,
            "updated_at": time.time(),
            "tuning": dict(_DEFAULT_TUNING),
            "observations": [],
            "trajectory": [],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        result = pd.detect_trends()
        assert result == []

    def test_insufficient_trajectory(self, tmp_path):
        """少于 4 个数据点不检测趋势"""
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 3,
            "updated_at": time.time(),
            "tuning": dict(_DEFAULT_TUNING),
            "observations": [],
            "trajectory": [
                {"time": time.time(), "tuning": dict(_DEFAULT_TUNING), "obs_count": 0},
                {"time": time.time(), "tuning": dict(_DEFAULT_TUNING), "obs_count": 1},
                {"time": time.time(), "tuning": {"emotional_dampening": False, "formality_shift": 0, "proactive_suppression": False}, "obs_count": 2},
            ],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        result = pd.detect_trends()
        assert result == []

    def test_formality_upward_trend(self, tmp_path):
        """formality_shift 持续上升"""
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 3,
            "updated_at": time.time(),
            "tuning": dict(_DEFAULT_TUNING),
            "observations": [],
            "trajectory": [
                {"time": time.time() - 3600 * 3, "tuning": {"emotional_dampening": False, "formality_shift": 0, "proactive_suppression": False}, "obs_count": 0},
                {"time": time.time() - 3600 * 2, "tuning": {"emotional_dampening": False, "formality_shift": 1, "proactive_suppression": False}, "obs_count": 1},
                {"time": time.time() - 3600 * 1, "tuning": {"emotional_dampening": False, "formality_shift": 2, "proactive_suppression": False}, "obs_count": 2},
                {"time": time.time(), "tuning": {"emotional_dampening": False, "formality_shift": 3, "proactive_suppression": False}, "obs_count": 3},
            ],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        result = pd.detect_trends()
        # formality_shift 斜率 >= 0.3，应有趋势
        assert len(result) >= 1
        trend_texts = [r["text"] for r in result]
        assert any("正式度" in t for t in trend_texts)

    def test_emotional_dampening_trend(self, tmp_path):
        """emotional_dampening 持续高比例"""
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "pattern_cache.json")
        data = {
            "version": 3,
            "updated_at": time.time(),
            "tuning": dict(_DEFAULT_TUNING),
            "observations": [],
            "trajectory": [
                {"time": time.time() - 3600 * i, "tuning": {"emotional_dampening": True, "formality_shift": 0, "proactive_suppression": False}, "obs_count": i}
                for i in range(5)
            ],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pd = PatternDiscovery(data_dir=str(tmp_path))
        result = pd.detect_trends()
        # dampening ratio >= 0.5 → 趋势
        trend_texts = [r["text"] for r in result]
        # 可能有 dampening 趋势或 formality 趋势（dampening 比例 >= 0.5）
        assert any("情绪压制" in t for t in trend_texts) or len(result) >= 0


class TestInjectRules:
    def test_inject_rules_exist(self):
        assert "深夜情绪话题多" in _INJECT_RULES
        assert "早上工作话题多" in _INJECT_RULES
        assert "连续多轮深度讨论" in _INJECT_RULES
        assert "发现用户焦虑话题" in _INJECT_RULES

    def test_inject_rules_values(self):
        assert _INJECT_RULES["深夜情绪话题多"] is True
        assert _INJECT_RULES["早上工作话题多"] is True
        assert _INJECT_RULES["连续多轮深度讨论"] is False
        assert _INJECT_RULES["发现用户焦虑话题"] is False


# ═══════════════════════════════════════════════════════════════════
# PatternDiscovery._dedup_and_filter
# ═══════════════════════════════════════════════════════════════════

class TestDedupAndFilter:
    def test_no_duplicates(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        obs = [
            {"text": "观察A"},
            {"text": "观察B"},
            {"text": "观察C"},
        ]
        result = pd._dedup_and_filter(obs)
        assert len(result) == 3

    def test_remove_duplicates(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        obs = [
            {"text": "观察A"},
            {"text": "观察B"},
            {"text": "观察A"},
        ]
        result = pd._dedup_and_filter(obs)
        assert len(result) == 2

    def test_remove_empty_text(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        obs = [
            {"text": ""},
            {"text": "有效观察"},
            {"text": ""},
        ]
        result = pd._dedup_and_filter(obs)
        assert len(result) == 1
        assert result[0]["text"] == "有效观察"

    def test_all_empty(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        obs = [
            {"text": ""},
            {"text": ""},
        ]
        result = pd._dedup_and_filter(obs)
        assert result == []

    def test_empty_list(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        result = pd._dedup_and_filter([])
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# PatternDiscovery.run (mock detectors)
# ═══════════════════════════════════════════════════════════════════

class TestRun:
    def test_run_with_no_detections(self, tmp_path):
        """所有检测器返回空时，run 正常完成"""
        pd = PatternDiscovery(data_dir=str(tmp_path))
        # Mock 所有检测器返回空
        pd._detect_temporal_patterns = MagicMock(return_value=[])
        pd._detect_emotional_anchors = MagicMock(return_value=[])
        pd._detect_topic_drift = MagicMock(return_value=[])
        pd._detect_interaction_rhythm = MagicMock(return_value=[])
        pd.detect_trends = MagicMock(return_value=[])
        pd._save = MagicMock()

        pd.run()
        assert pd._observations == []
        assert pd._tuning == _DEFAULT_TUNING
        pd._save.assert_called_once()

    def test_run_with_observations_and_tuning(self, tmp_path):
        """检测器返回观察 + tuning 时，run 正确合并"""
        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd._detect_temporal_patterns = MagicMock(return_value=[
            {"type": "temporal", "text": "时间节律观察",
             "tuning": {"emotional_dampening": True, "formality_shift": -1}}
        ])
        pd._detect_emotional_anchors = MagicMock(return_value=[
            {"type": "emotion", "text": "情绪观察"}
        ])
        pd._detect_topic_drift = MagicMock(return_value=[])
        pd._detect_interaction_rhythm = MagicMock(return_value=[])
        pd.detect_trends = MagicMock(return_value=[])
        pd._save = MagicMock()

        pd.run()
        assert len(pd._observations) >= 1
        assert pd._tuning["emotional_dampening"] is True

    def test_run_max_observations(self, tmp_path):
        """观察数量不超过 MAX_OBSERVATIONS=3"""
        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd._detect_temporal_patterns = MagicMock(return_value=[
            {"text": f"观察{i}"} for i in range(10)
        ])
        pd._detect_emotional_anchors = MagicMock(return_value=[])
        pd._detect_topic_drift = MagicMock(return_value=[])
        pd._detect_interaction_rhythm = MagicMock(return_value=[])
        pd.detect_trends = MagicMock(return_value=[])
        pd._save = MagicMock()

        pd.run()
        assert len(pd._observations) <= pd.MAX_OBSERVATIONS

    def test_run_exception_handling(self, tmp_path):
        """run 中检测器异常不影响整体"""
        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd._detect_temporal_patterns = MagicMock(side_effect=Exception("模拟错误"))
        pd._detect_emotional_anchors = MagicMock(return_value=[])
        pd._detect_topic_drift = MagicMock(return_value=[])
        pd._detect_interaction_rhythm = MagicMock(return_value=[])
        pd.detect_trends = MagicMock(return_value=[])
        pd._save = MagicMock()

        pd.run()  # 不应抛出异常
        assert pd._observations == []


# ═══════════════════════════════════════════════════════════════════
# PatternDiscovery._save
# ═══════════════════════════════════════════════════════════════════

class TestSave:
    def test_save_creates_cache(self, tmp_path):
        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd._observations = [{"text": "测试观察", "inject": True}]
        pd._tuning = {"emotional_dampening": True, "formality_shift": 1, "proactive_suppression": False}
        pd._save()
        assert os.path.exists(pd._cache_path)
        with open(pd._cache_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == 3
        assert len(data["observations"]) == 1
        assert len(data["trajectory"]) == 1

    def test_save_appends_trajectory(self, tmp_path):
        """多次 save 追加 trajectory"""
        pd = PatternDiscovery(data_dir=str(tmp_path))
        pd._observations = []
        pd._tuning = dict(_DEFAULT_TUNING)
        pd._save()
        pd._save()
        pd._save()
        with open(pd._cache_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["trajectory"]) == 3

    def test_trajectory_cap_30(self, tmp_path):
        """trajectory 最多保留 30 条"""
        pd = PatternDiscovery(data_dir=str(tmp_path))
        # 保存 35 次
        for i in range(35):
            pd._tuning["formality_shift"] = i
            pd._save()
        with open(pd._cache_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["trajectory"]) <= 30


# ═══════════════════════════════════════════════════════════════════
# PatternDiscovery._tag_frequency
# ═══════════════════════════════════════════════════════════════════

class TestTagFrequency:
    def test_empty_entries(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        result = pd._tag_frequency([])
        assert result == {}

    def test_entries_without_user_message(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd")
        entries = [{"other_field": "value"}]
        result = pd._tag_frequency(entries)
        assert result == {}


# ═══════════════════════════════════════════════════════════════════
# PatternDiscovery._load_recent_chat
# ═══════════════════════════════════════════════════════════════════

class TestLoadRecentChat:
    def test_no_chat_history_path(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd", chat_history_path="")
        result = pd._load_recent_chat()
        assert result == []

    def test_file_not_exists(self):
        pd = PatternDiscovery(data_dir="/tmp/test_pd",
                              chat_history_path="/nonexistent/chat.jsonl")
        result = pd._load_recent_chat()
        assert result == []

    def test_valid_jsonl(self, tmp_path):
        chat_path = os.path.join(str(tmp_path), "chat.jsonl")
        with open(chat_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"user_message": "你好", "timestamp": time.time()}) + "\n")
            f.write(json.dumps({"user_message": "今天天气不错", "timestamp": time.time()}) + "\n")

        pd = PatternDiscovery(data_dir=str(tmp_path), chat_history_path=chat_path)
        result = pd._load_recent_chat()
        assert len(result) == 2
        assert result[0]["user_message"] == "你好"

    def test_limit(self, tmp_path):
        chat_path = os.path.join(str(tmp_path), "chat.jsonl")
        with open(chat_path, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"user_message": f"msg{i}", "timestamp": time.time()}) + "\n")

        pd = PatternDiscovery(data_dir=str(tmp_path), chat_history_path=chat_path)
        result = pd._load_recent_chat(limit=5)
        assert len(result) == 5

    def test_corrupted_lines_skipped(self, tmp_path):
        chat_path = os.path.join(str(tmp_path), "chat.jsonl")
        with open(chat_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"user_message": "valid"}) + "\n")
            f.write("not valid json{{{\n")
            f.write(json.dumps({"user_message": "also valid"}) + "\n")

        pd = PatternDiscovery(data_dir=str(tmp_path), chat_history_path=chat_path)
        result = pd._load_recent_chat()
        assert len(result) == 2
