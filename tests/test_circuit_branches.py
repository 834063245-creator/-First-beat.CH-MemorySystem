"""测试 app/core/circuit.py 纯函数 — 提行覆盖。

覆盖：_compute_emotion_intensity 全部路径 / basal_ganglia_gate 全部分支
      / analyze_user_message 空消息降级
"""
from unittest.mock import MagicMock, patch
import pytest


class TestComputeEmotionIntensity:
    def test_exclamation_marks(self):
        from app.core.circuit import _compute_emotion_intensity
        s = _compute_emotion_intensity("好！！！")
        assert s > 0.3  # 3*0.15=0.45

    def test_emoji(self):
        from app.core.circuit import _compute_emotion_intensity
        s = _compute_emotion_intensity("好开心😊😊")
        assert s > 0  # emoji 计数

    def test_intensifiers(self):
        from app.core.circuit import _compute_emotion_intensity
        s = _compute_emotion_intensity("真的非常好")
        assert s > 0.15

    def test_repeat_patterns(self):
        from app.core.circuit import _compute_emotion_intensity
        s = _compute_emotion_intensity("好好好")
        assert s >= 0.2

    def test_long_text_boost(self):
        from app.core.circuit import _compute_emotion_intensity
        long_msg = "我很开心！！！😊好好好" * 10  # > 80 chars + 有叹号有重复
        s = _compute_emotion_intensity(long_msg)
        assert s >= 0.5  # 多种加成叠加

    def test_capped_at_1(self):
        from app.core.circuit import _compute_emotion_intensity
        s = _compute_emotion_intensity("！！！😊😊😊好好好真的非常" * 5)
        assert s <= 1.0


class TestBasalGangliaGate:
    def _make_pfc(self, intent, emotion="neutral"):
        from app.core.state import UserMessageAnalysis
        return UserMessageAnalysis(
            intent=intent, emotion=emotion, urgency=0.0, topics=[],
            raw_text="test", confidence=0.7, emotion_intensity=0.0,
        )

    def test_emotional_sharing_negative(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("emotional_sharing", "negative")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "caring"
        assert g.response_mode == "soothe"

    def test_emotional_sharing_positive(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("emotional_sharing", "positive")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.response_mode == "question_first"

    def test_conflict(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("conflict", "frustrated")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "soft"
        assert g.response_mode == "confirm"
        assert g.formality == 0.5

    def test_recall_neutral(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("recall", "neutral")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "direct"

    def test_recall_intimate(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("recall", "intimate")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "warm"

    def test_ask_fact(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("ask_fact")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "direct"
        assert g.response_mode == "direct_answer"

    def test_casual(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("casual")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "warm"
        assert g.response_mode == "auto"

    def test_work_impulse_suppression(self):
        from app.core.circuit import basal_ganglia_gate
        from app.core.state import ImpulseDirective
        pfc = self._make_pfc("emotional_sharing", "intimate")
        work_impulse = ImpulseDirective(
            intent="check", target_concept="bug 修好了吗",
        )
        g = basal_ganglia_gate(pfc, [], [work_impulse], [])
        # 工作冲动在亲密场景被压制
        assert len(g.impulses_to_show) == 0
        assert any("工作" in r for r in g.suppression_reasons)

    def test_intimacy_computation(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("emotional_sharing", "intimate")
        pfc.emotion_intensity = 0.8
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.intimacy > 0


class TestAnalyzeUserMessageEdgeCases:
    def test_empty_message(self):
        from app.core.circuit import analyze_user_message
        result = analyze_user_message("")
        assert result.intent == "casual"
        assert result.emotion == "neutral"

    @patch("app.llm.embed.local_embed", return_value=[0.0] * 1024)
    def test_whitespace_only(self, mock_embed):
        from app.core.circuit import analyze_user_message
        result = analyze_user_message("   ")
        assert result.intent != ""


class TestBasalGangliaGateMoreEdges:
    def _make_pfc(self, intent, emotion="neutral"):
        from app.core.state import UserMessageAnalysis
        return UserMessageAnalysis(
            intent=intent, emotion=emotion, urgency=0.0, topics=[],
            raw_text="test", confidence=0.7, emotion_intensity=0.0,
        )

    def test_high_urgency_sets_low_intimacy(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("ask_fact")
        pfc.urgency = 0.9
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.intimacy == 0.0

    def test_emotional_sharing_frustrated(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("emotional_sharing", "frustrated")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "caring"

    def test_request_intent(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("request")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.tone == "direct"

    def test_meta_intent(self):
        from app.core.circuit import basal_ganglia_gate
        pfc = self._make_pfc("meta")
        g = basal_ganglia_gate(pfc, [], [], [])
        assert g.response_mode == "direct_answer"
