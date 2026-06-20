# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 72d20a4c

"""Tests for emotion analysis — analyze_emotion function."""
from app.analysis.emotion import analyze_emotion


class TestAnalyzeEmotion:
    def test_positive_detection(self):
        assert analyze_emotion("今天好开心啊") == "positive"
        assert analyze_emotion("太棒了，完美") == "positive"
        assert analyze_emotion("期待明天的旅行") == "positive"

    def test_negative_detection(self):
        assert analyze_emotion("气死我了") == "negative"
        assert analyze_emotion("难受，想哭") == "negative"
        assert analyze_emotion("太烦人了") == "negative"

    def test_negative_priority(self):
        assert analyze_emotion("虽然很开心，但是好累") == "negative"

    def test_neutral_detection(self):
        assert analyze_emotion("今天天气不错") == "neutral"
        assert analyze_emotion("我去吃了饭") == "neutral"

    def test_empty_and_punctuation(self):
        assert analyze_emotion("") == "neutral"
        assert analyze_emotion("！！！？？") == "neutral"
