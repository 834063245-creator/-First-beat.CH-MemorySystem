# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 81dd50c9

"""测试 app/brain/export_training_data.py — 意图/情绪分类器。

覆盖：classify_intent / classify_emotion 纯函数。
"""
import pytest
from app.brain.export_training_data import classify_intent, classify_emotion


class TestClassifyIntent:
    def test_conflict_detected(self):
        assert classify_intent("不对，你搞错了") == "conflict"

    def test_emotional_sharing_detected(self):
        assert classify_intent("我今天心情很好") == "emotional_sharing"

    def test_recall_detected(self):
        assert classify_intent("我之前说过的那件事还记得吗") == "recall"

    def test_request_detected(self):
        assert classify_intent("帮我写一段代码") == "request"

    def test_ask_fact_detected(self):
        assert classify_intent("为什么Python这么流行") == "ask_fact"

    def test_meta_detected(self):
        # "你能做什么" 不含 ask_fact 的关键词 "什么"（注意不是同一个词）
        # 实际包含"什么"所以不能用"你叫什么"；用"你是谁"
        assert classify_intent("你是谁") == "meta"

    def test_casual_default(self):
        assert classify_intent("今天天气不错") == "casual"

    def test_empty_text(self):
        assert classify_intent("") == "casual"

    def test_conflict_first(self):
        """冲突关键词优先级最高（第一个检查）。"""
        # 同时有 conflict 和 emotional 关键词，应返回 conflict
        assert classify_intent("不对，我觉得很难过") == "conflict"


class TestClassifyEmotion:
    def test_intimate(self):
        assert classify_emotion("想你，爱你") == "intimate"

    def test_frustrated(self):
        assert classify_emotion("烦死了，无语") == "frustrated"

    def test_negative(self):
        assert classify_emotion("最近压力很大很难过") == "negative"

    def test_positive(self):
        assert classify_emotion("太开心了，好棒") == "positive"

    def test_neutral_default(self):
        assert classify_emotion("开会时间到了各位") == "neutral"

    def test_empty_text(self):
        assert classify_emotion("") == "neutral"

    def test_frustrated_before_negative(self):
        """frustrated 优先级高于 negative。"""
        assert classify_emotion("烦死了很难过") == "frustrated"
