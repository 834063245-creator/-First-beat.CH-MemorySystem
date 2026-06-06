"""测试 app/brain/ 层纯导出/常量模块的导入完整性和数据完整性。

覆盖：brain/models.py（语义脑外壳）、brain/keywords.py（意图/情绪常量）。
"""
import pytest


class TestBrainModels:
    """语义脑外壳 — 纯 re-export 模块。"""

    def test_all_exports_available(self):
        from app.brain.models import (
            classify_intent,
            analyze_emotion,
            classify_urgency,
            detect_negation,
            extract_tags,
            tokenize,
            extract_entities,
        )
        # 所有导出都必须可调用
        assert callable(classify_intent)
        assert callable(analyze_emotion)
        assert callable(classify_urgency)
        assert callable(detect_negation)
        assert callable(extract_tags)
        assert callable(tokenize)
        assert callable(extract_entities)

    def test_all_list_complete(self):
        from app.brain import models
        expected = [
            "classify_intent", "analyze_emotion", "classify_urgency",
            "detect_negation", "extract_tags", "tokenize", "extract_entities",
        ]
        for name in expected:
            assert name in models.__all__


class TestKeywords:
    """意图/情绪关键词常量完整性。"""

    def test_intent_keywords_structure(self):
        from app.brain.keywords import INTENT_KEYWORDS
        required_intents = {"recall", "emotional_sharing", "conflict",
                            "ask_fact", "request", "meta"}
        assert set(INTENT_KEYWORDS.keys()) == required_intents
        for keywords in INTENT_KEYWORDS.values():
            assert isinstance(keywords, list)
            assert len(keywords) >= 3  # 每个意图至少有 3 个关键词

    def test_emotion_keywords_structure(self):
        from app.brain.keywords import EMOTION_KEYWORDS
        required_emotions = {"intimate", "positive", "negative", "frustrated"}
        assert set(EMOTION_KEYWORDS.keys()) == required_emotions
        for keywords in EMOTION_KEYWORDS.values():
            assert isinstance(keywords, list)
            assert len(keywords) >= 3

    def test_intensifiers_non_empty(self):
        from app.brain.keywords import INTENSIFIERS
        assert isinstance(INTENSIFIERS, set)
        assert len(INTENSIFIERS) >= 5

    def test_emotion_repeat_patterns(self):
        from app.brain.keywords import EMOTION_REPEAT_PATTERN
        assert len(EMOTION_REPEAT_PATTERN) >= 3

    def test_work_keywords_non_empty(self):
        from app.brain.keywords import WORK_KEYWORDS
        assert len(WORK_KEYWORDS) >= 5

    def test_negation_words_non_empty(self):
        from app.brain.keywords import NEGATION_WORDS
        assert len(NEGATION_WORDS) >= 3
