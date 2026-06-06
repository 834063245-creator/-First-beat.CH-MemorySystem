"""测试 app/core/circuit.py 的 basal_ganglia_gate()。

覆盖：各意图/情绪组合的语气、响应模式、情绪压制、亲密值。
"""
import pytest
from app.core.state import UserMessageAnalysis, GatingDecision, ImpulseDirective
from app.core.circuit import basal_ganglia_gate, analyze_user_message


def _make_pfc(intent="casual", emotion="neutral", urgency=0.0,
              emotion_intensity=0.0, raw_text=""):
    """构造 UserMessageAnalysis 的快捷方式。"""
    return UserMessageAnalysis(
        intent=intent, emotion=emotion, urgency=urgency,
        emotion_intensity=emotion_intensity, raw_text=raw_text,
        topics=[], confidence=0.6,
    )


class TestBasalGangliaGate:

    # ── 意图+情绪 → 语气/模式映射 ──

    @pytest.mark.parametrize("intent,emotion,exp_tone,exp_mode", [
        ("emotional_sharing", "negative",  "caring", "soothe"),
        ("emotional_sharing", "intimate",  "caring", "soothe"),
        ("emotional_sharing", "frustrated","caring", "soothe"),
        ("emotional_sharing", "positive",  "warm",   "question_first"),
        ("emotional_sharing", "neutral",   "warm",   "question_first"),
        ("conflict",    "frustrated",  "soft",  "confirm"),
        ("conflict",    "negative",    "soft",  "confirm"),
        ("conflict",    "neutral",     "soft",  "confirm"),
        ("recall",      "neutral",     "direct","auto"),
        ("recall",      "positive",    "warm",  "auto"),
        ("recall",      "intimate",    "warm",  "auto"),
        ("ask_fact",    "neutral",     "direct","direct_answer"),
        ("request",     "neutral",     "direct","direct_answer"),
        ("meta",        "neutral",     "direct","direct_answer"),
        ("casual",      "neutral",     "warm",  "auto"),
    ])
    def test_tone_and_mode(self, intent, emotion, exp_tone, exp_mode):
        pfc = _make_pfc(intent=intent, emotion=emotion)
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.tone == exp_tone, f"tone: {gate.tone} != {exp_tone}"
        assert gate.response_mode == exp_mode, f"mode: {gate.response_mode} != {exp_mode}"

    # ── 亲密值 ──

    def test_intimate_emotion_sets_high_intimacy(self):
        pfc = _make_pfc(intent="emotional_sharing", emotion="intimate")
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.intimacy >= 0.7

    def test_high_urgency_sets_low_intimacy(self):
        pfc = _make_pfc(intent="casual", emotion="neutral", urgency=0.8)
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.intimacy == 0.0

    def test_high_emotion_intensity_sets_mid_intimacy(self):
        pfc = _make_pfc(intent="casual", emotion="neutral",
                        urgency=0.0, emotion_intensity=0.8)
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.intimacy == 0.5

    def test_default_intimacy(self):
        pfc = _make_pfc(intent="casual", emotion="neutral")
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.intimacy == 0.3

    # ── 正式度 ──

    @pytest.mark.parametrize("intent,emotion,exp_formality", [
        ("emotional_sharing", "negative", 0.1),
        ("conflict",    "frustrated", 0.5),
        ("ask_fact",    "neutral",    0.4),
        ("request",     "neutral",    0.3),
        ("casual",      "neutral",    0.3),
    ])
    def test_formality(self, intent, emotion, exp_formality):
        pfc = _make_pfc(intent=intent, emotion=emotion)
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.formality == exp_formality

    # ── 冲动压制（工作类冲动在亲密/情绪场景下被压制） ──

    def test_work_impulses_suppressed_in_emotional_scene(self):
        pfc = _make_pfc(intent="emotional_sharing", emotion="negative")
        work_impulse = ImpulseDirective(
            intent="share_observation",
            target_concept="用户上次熬夜修bug",
            emotional_tone="neutral",
        )
        gate = basal_ganglia_gate(pfc, [], [work_impulse], [])
        assert len(gate.impulses_to_show) == 0
        assert len(gate.suppression_reasons) >= 1

    def test_non_work_impulses_not_suppressed(self):
        pfc = _make_pfc(intent="ask_fact", emotion="neutral")
        normal_impulse = ImpulseDirective(
            intent="recall",
            target_concept="相关记忆",
            emotional_tone="neutral",
        )
        gate = basal_ganglia_gate(pfc, [], [normal_impulse], [])
        # ask_fact 场景不压制普通冲动
        assert len(gate.impulses_to_show) >= 1

    # ── 端到端：analyze_user_message + basal_ganglia_gate ──

    @pytest.mark.parametrize("msg,exp_tone,exp_mode", [
        ("我昨晚梦到你了", "caring", "soothe"),
        ("不对，你搞错了", "soft", "confirm"),
        ("Python怎么读文件", "direct", "direct_answer"),
        ("你好", "warm", "auto"),
    ])
    def test_end_to_end(self, msg, exp_tone, exp_mode):
        pfc = analyze_user_message(msg)
        gate = basal_ganglia_gate(pfc, [], [], [])
        # Ollama不可用时语义模型降级，放宽断言
        assert gate.tone in (exp_tone, "warm"), f"tone: {gate.tone} not in ({exp_tone}, warm)"
        assert gate.response_mode in (exp_mode, "auto"), f"mode: {gate.response_mode} not in ({exp_mode}, auto)"
