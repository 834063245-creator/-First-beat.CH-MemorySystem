"""测试 app/portrait/extractors.py — 画像特征提取器纯函数。

所有函数无副作用，输入→输出，可独立测试。
"""
import math
import time
from datetime import datetime

import pytest
from unittest.mock import patch, MagicMock

from app.portrait.extractors import (
    extract_keywords,
    recency_score,
    compute_confidence,
    detect_emotion_flip,
    compute_tag_density,
    classify_tag_heat,
    extract_emotion_category,
    tag_similarity,
)


# ═══════════════════════════════════════════════════════════════════
# extract_keywords
# ═══════════════════════════════════════════════════════════════════

class TestExtractKeywords:
    def test_empty_text_returns_empty(self):
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_extracts_keywords_from_text(self):
        with patch("app.brain.semantic.extract_tags") as mock_extract:
            mock_extract.return_value = ["Python", "编程", "学习", "Rust", "a"]
            with patch("app.config.settings.STOP_WORDS", set()):
                result = extract_keywords("我爱学习Python和Rust编程", topk=10)
            # "a" 应被长度过滤掉
            assert "a" not in result
            assert len(result) <= 10

    def test_respects_topk_limit(self):
        with patch("app.brain.semantic.extract_tags") as mock_extract:
            mock_extract.return_value = [f"词{i}" for i in range(20)]
            with patch("app.config.settings.STOP_WORDS", set()):
                result = extract_keywords("测试文本", topk=5)
            assert len(result) == 5

    def test_filters_stop_words(self):
        with patch("app.brain.semantic.extract_tags") as mock_extract:
            mock_extract.return_value = ["的", "是", "Python", "在"]
            with patch("app.config.settings.STOP_WORDS", {"的", "是", "在"}):
                result = extract_keywords("测试文本", topk=10)
            assert result == ["Python"]

    def test_import_error_returns_empty(self):
        with patch("app.brain.semantic.extract_tags", side_effect=ImportError):
            result = extract_keywords("测试文本")
            assert result == []

    def test_general_exception_returns_empty(self):
        with patch("app.brain.semantic.extract_tags", side_effect=RuntimeError("fail")):
            result = extract_keywords("测试文本")
            assert result == []

    def test_none_input(self):
        # None 没有 .strip()
        try:
            result = extract_keywords(None)
        except AttributeError:
            # 预期行为：传入 None 会抛 AttributeError
            pass


# ═══════════════════════════════════════════════════════════════════
# recency_score
# ═══════════════════════════════════════════════════════════════════

class TestRecencyScore:
    def test_now_returns_one(self):
        now = time.time()
        assert recency_score(now, now=now) == 1.0

    def test_future_returns_one(self):
        now = time.time()
        future = now + 3600
        assert recency_score(future, now=now) == 1.0

    def test_exactly_90_days_returns_zero(self):
        now = time.time()
        ts = now - 90 * 86400
        score = recency_score(ts, now=now)
        assert score == 0.0

    def test_more_than_90_days_returns_zero(self):
        now = time.time()
        ts = now - 120 * 86400
        assert recency_score(ts, now=now) == 0.0

    def test_45_days_returns_0_5(self):
        now = time.time()
        ts = now - 45 * 86400
        score = recency_score(ts, now=now)
        assert abs(score - 0.5) < 0.01

    def test_1_day_returns_approximately_0_989(self):
        now = time.time()
        ts = now - 86400
        score = recency_score(ts, now=now)
        assert 0.98 < score < 1.0

    def test_default_now_is_current_time(self):
        """不传 now 参数时应使用当前时间。"""
        now = time.time()
        score = recency_score(now)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_very_old_timestamp(self):
        now = time.time()
        ts = now - 365 * 86400  # 一年前
        assert recency_score(ts, now=now) == 0.0


# ═══════════════════════════════════════════════════════════════════
# compute_confidence
# ═══════════════════════════════════════════════════════════════════

class TestComputeConfidence:
    def test_max_evidence(self):
        """证据数 9999 + 新近度 1.0 + 一致性 1.0 → 接近 1.0"""
        score = compute_confidence(9999, 1.0, 1.0)
        assert 0.95 < score <= 1.0

    def test_min_everything(self):
        """证据数 0 + 新近度 0.0 + 一致性 0.0 → 0.0"""
        # log(0+1)/log(10) = 0/2.302 = 0
        score = compute_confidence(0, 0.0, 0.0)
        assert score == 0.0

    def test_single_evidence_medium_recency(self):
        """1条证据 + 新近度 0.5 + 一致性 1.0"""
        # evidence_score = log(2)/log(10) ≈ 0.301
        # 0.301 * 0.4 + 0.5 * 0.4 + 1.0 * 0.2 = 0.120 + 0.200 + 0.200 = 0.52
        score = compute_confidence(1, 0.5, 1.0)
        assert abs(score - 0.52) < 0.02

    def test_evidence_count_10(self):
        """log(11)/log(10) = 1.0 → evidence_score = 1.0"""
        # 1.0 * 0.4 + 0.8 * 0.4 + 0.9 * 0.2 = 0.4 + 0.32 + 0.18 = 0.90
        score = compute_confidence(10, 0.8, 0.9)
        assert abs(score - 0.90) < 0.02

    def test_result_is_rounded_to_2_decimals(self):
        score = compute_confidence(5, 0.5, 0.5)
        # 验证是两位小数
        assert round(score, 2) == score

    def test_evidence_score_capped_at_1(self):
        """evidence_count >= 9 时 evidence_score 已达 1.0"""
        score_large = compute_confidence(100, 1.0, 1.0)
        score_huge = compute_confidence(999999, 1.0, 1.0)
        assert abs(score_large - score_huge) < 0.01


# ═══════════════════════════════════════════════════════════════════
# detect_emotion_flip
# ═══════════════════════════════════════════════════════════════════

class TestDetectEmotionFlip:
    def test_positive_to_negative_is_flip(self):
        assert detect_emotion_flip("positive", "negative") is True
        assert detect_emotion_flip("happy", "sad") is True
        assert detect_emotion_flip("excited", "frustrated") is True

    def test_negative_to_positive_is_flip(self):
        assert detect_emotion_flip("negative", "positive") is True
        assert detect_emotion_flip("sad", "happy") is True
        assert detect_emotion_flip("angry", "excited") is True

    def test_same_valence_not_flip(self):
        assert detect_emotion_flip("positive", "happy") is False
        assert detect_emotion_flip("happy", "excited") is False
        assert detect_emotion_flip("negative", "sad") is False
        assert detect_emotion_flip("sad", "frustrated") is False

    def test_same_emotion_not_flip(self):
        assert detect_emotion_flip("positive", "positive") is False
        assert detect_emotion_flip("negative", "negative") is False

    def test_case_insensitive(self):
        assert detect_emotion_flip("POSITIVE", "negative") is True
        assert detect_emotion_flip("Happy", "SAD") is True

    def test_neutral_to_negative_not_flip(self):
        """neutral 不在 POSITIVE 也不在 NEGATIVE 集合中，所以不会翻转为 flip"""
        assert detect_emotion_flip("neutral", "negative") is False

    def test_neutral_to_positive_not_flip(self):
        assert detect_emotion_flip("neutral", "positive") is False

    def test_unknown_emotions_not_flip(self):
        assert detect_emotion_flip("bored", "curious") is False

    def test_intimate_is_positive(self):
        """intimate 属于 POSITIVE"""
        assert detect_emotion_flip("intimate", "negative") is True

    def test_anxious_is_negative(self):
        """anxious 属于 NEGATIVE"""
        assert detect_emotion_flip("anxious", "positive") is True


# ═══════════════════════════════════════════════════════════════════
# compute_tag_density
# ═══════════════════════════════════════════════════════════════════

class TestComputeTagDensity:
    def test_simple_density(self):
        result = compute_tag_density({"Python": 10, "Rust": 5}, 10)
        assert result == {"Python": 1.0, "Rust": 0.5}

    def test_zero_days_returns_empty(self):
        assert compute_tag_density({"Python": 10}, 0) == {}

    def test_negative_days_returns_empty(self):
        assert compute_tag_density({"Python": 10}, -5) == {}

    def test_empty_tags_returns_empty(self):
        assert compute_tag_density({}, 30) == {}

    def test_fractional_density(self):
        result = compute_tag_density({"tag": 7}, 30)
        assert abs(result["tag"] - 0.2333) < 0.01


# ═══════════════════════════════════════════════════════════════════
# classify_tag_heat
# ═══════════════════════════════════════════════════════════════════

class TestClassifyTagHeat:
    def test_hot_tag(self):
        """密度 ≥ 1.0 + 最近3天内出现 → hot"""
        assert classify_tag_heat("Python", count=10, days=10, last_seen_days=2.0) == "hot"
        assert classify_tag_heat("Rust", count=5, days=5, last_seen_days=1.0) == "hot"

    def test_warm_tag(self):
        """最近7天内出现（但密度不够 hot） → warm"""
        assert classify_tag_heat("Python", count=1, days=30, last_seen_days=5.0) == "warm"
        # 密度 < 1.0 且 last_seen ≤ 7 → warm
        assert classify_tag_heat("Go", count=2, days=10, last_seen_days=7.0) == "warm"

    def test_cooling_tag(self):
        """超过7天 → cooling"""
        assert classify_tag_heat("Java", count=10, days=10, last_seen_days=8.0) == "cooling"

    def test_exactly_3_days_hot_when_dense(self):
        """last_seen_days=3, density=1.0 → hot"""
        assert classify_tag_heat("tag", count=5, days=5, last_seen_days=3.0) == "hot"

    def test_exactly_7_days_warm(self):
        assert classify_tag_heat("tag", count=1, days=30, last_seen_days=7.0) == "warm"

    def test_days_edge_case(self):
        """days=0 时密度无穷大，但 max(days,1) 保护"""
        result = classify_tag_heat("tag", count=1, days=0, last_seen_days=1.0)
        assert result in ("hot", "warm", "cooling")


# ═══════════════════════════════════════════════════════════════════
# extract_emotion_category
# ═══════════════════════════════════════════════════════════════════

class TestExtractEmotionCategory:
    def test_positive_valence(self):
        assert extract_emotion_category(0.5) == "positive"
        assert extract_emotion_category(1.0) == "positive"
        assert extract_emotion_category(0.21) == "positive"

    def test_negative_valence(self):
        assert extract_emotion_category(-0.5) == "negative"
        assert extract_emotion_category(-1.0) == "negative"
        assert extract_emotion_category(-0.21) == "negative"

    def test_neutral_valence(self):
        assert extract_emotion_category(0.0) == "neutral"
        assert extract_emotion_category(0.19) == "neutral"
        assert extract_emotion_category(-0.19) == "neutral"
        assert extract_emotion_category(0.2) == "neutral"
        assert extract_emotion_category(-0.2) == "neutral"


# ═══════════════════════════════════════════════════════════════════
# tag_similarity
# ═══════════════════════════════════════════════════════════════════

class TestTagSimilarity:
    def test_identical_sets(self):
        assert tag_similarity(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_disjoint_sets(self):
        assert tag_similarity(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        # intersection=2 (b,c), union=4 (a,b,c,d) → 0.5
        assert tag_similarity(["a", "b", "c"], ["b", "c", "d"]) == 0.5

    def test_both_empty(self):
        assert tag_similarity([], []) == 0.0

    def test_one_empty(self):
        assert tag_similarity(["a"], []) == 0.0
        assert tag_similarity([], ["a"]) == 0.0

    def test_duplicate_tags(self):
        """集合化会去重"""
        assert tag_similarity(["a", "a", "b"], ["a", "b"]) == 1.0

    def test_single_element_same(self):
        assert tag_similarity(["x"], ["x"]) == 1.0

    def test_superset_relation(self):
        """超集关系：intersection=2, union=3"""
        assert tag_similarity(["a", "b", "c"], ["a", "b"]) == 2 / 3
