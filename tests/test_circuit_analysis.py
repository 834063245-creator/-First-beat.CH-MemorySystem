"""测试 app/core/circuit.py 的 analyze_user_message()。

覆盖：意图/情绪分析、紧急度、否定检测。
注：_compute_urgency/_keyword_intent/_keyword_emotion 已随 v2.0 重构移除，
    对应逻辑已迁移至 app.brain.semantic（bge-m3 原型匹配）。
"""
import pytest
from app.core.state import UserMessageAnalysis
from app.core.circuit import analyze_user_message


class TestAnalyzeUserMessage:
    """回路①：用户消息分析 — 纯规则路径（brain=None）。"""

    # ── 意图分类 ──

    @pytest.mark.parametrize("msg,exp_intent", [
        ("我好累啊", "emotional_sharing"),
        ("你还记得我之前说的那个bug吗", "recall"),
        ("帮我写个冒泡排序", "request"),
        ("不对，你说错了", "conflict"),
        ("今天天气怎么样", "ask_fact"),
        ("你是谁", "meta"),
        ("嗯", "casual"),
    ])
    def test_intent_classification(self, msg, exp_intent):
        r = analyze_user_message(msg)
        assert r.intent == exp_intent, f"{msg} → intent={r.intent}, expected={exp_intent}"

    # ── 情绪分类 ──

    @pytest.mark.parametrize("msg,exp_emotion", [
        ("我好累啊", "negative"),
        ("哈哈太棒了！！！", "positive"),
        ("烦死了无语", "frustrated"),
        ("嗯", "neutral"),
    ])
    def test_emotion_classification(self, msg, exp_emotion):
        r = analyze_user_message(msg)
        assert r.emotion == exp_emotion, f"{msg} → emotion={r.emotion}, expected={exp_emotion}"

    def test_positive_emotion_intensity_positive(self):
        """感叹号应该标记情绪强度 > 0。"""
        r = analyze_user_message("哈哈太棒了！！！")
        assert r.emotion == "positive"
        assert r.emotion_intensity >= 0.3

    def test_negation_detection(self):
        """"不开心"不应判为 positive。"""
        r = analyze_user_message("我不开心")
        assert r.emotion != "positive"
        # "开心"被否定 → 应该是 neutral（否定词"不"触发）
        assert r.emotion == "neutral"

    # ── 紧急度 ──

    def test_urgency_above_baseline(self):
        r = analyze_user_message("急死我了！")
        assert r.urgency >= 0.3

    def test_urgency_low(self):
        r = analyze_user_message("嗯嗯")
        assert r.urgency < 0.3

    # ── 原始文本保留 ──

    def test_raw_text_preserved(self):
        r = analyze_user_message("  你好世界  ")
        assert r.raw_text == "你好世界"

    # ── 空消息 ──

    def test_empty_message(self):
        r = analyze_user_message("")
        assert r.intent == "casual"
        assert r.emotion == "neutral"

    # ── brain=None 退化路径 ──

    def test_brain_none_confidence(self):
        r = analyze_user_message("你好")
        assert r.confidence >= 0.3

    # ── 话题提取 ──

    def test_topics_extracted(self):
        r = analyze_user_message("帮我查一下数据库的bug")
        assert len(r.topics) >= 1


@pytest.mark.skip(reason="_compute_urgency 已随 v2.0 重构移除，逻辑迁移至 app.brain.semantic")
class TestComputeUrgency:

    @pytest.mark.parametrize("msg,exp_min", [
        ("急！马上帮我！", 0.4),
        ("急死我了", 0.4),
        ("急！！", 0.4),
        ("嗯嗯", 0.0),
        ("", 0.0),
    ])
    def test_urgency_values(self, msg, exp_min):
        pass


@pytest.mark.skip(reason="_keyword_intent 已随 v2.0 重构移除，逻辑迁移至 app.brain.semantic")
class TestKeywordIntent:
    """_keyword_intent 所有分支覆盖 — 已废弃。"""

    @pytest.mark.parametrize("msg,exp_intent", [
        ("不对，你错了", "conflict"),
        ("今天心情很差", "emotional_sharing"),
        ("还记得上次吗", "recall"),
        ("帮我查一下数据", "request"),
        ("什么是Python", "ask_fact"),
        ("你是谁", "meta"),
        ("你好", "casual"),
        ("", "casual"),
        ("随便聊聊", "casual"),
    ])
    def test_all_intents(self, msg, exp_intent):
        pass


@pytest.mark.skip(reason="_keyword_emotion 已随 v2.0 重构移除，逻辑迁移至 app.brain.semantic")
class TestKeywordEmotion:
    """_keyword_emotion 所有分支覆盖 — 已废弃。"""

    @pytest.mark.parametrize("msg,exp_emotion", [
        ("好想你啊", "intimate"),
        ("烦死了无语", "frustrated"),
        ("好难过", "negative"),
        ("太棒了", "positive"),
        ("今天天气不错", "positive"),
        ("", "neutral"),
        ("你好你好", "positive"),
    ])
    def test_all_emotions(self, msg, exp_emotion):
        pass

    def test_emotion_priority(self):
        pass


@pytest.mark.parametrize("msg,exp_intent,exp_emotion", [
    ("我好累啊", "emotional_sharing", "negative"),
    ("你还记得我之前说的那个bug吗", "recall", "neutral"),
    ("帮我写个冒泡排序", "request", "neutral"),
    ("不对，你说错了", "conflict", "neutral"),
    ("今天天气怎么样", "ask_fact", "neutral"),
    ("你是谁", "meta", "neutral"),
    ("嗯", "casual", "neutral"),
    ("哈哈太棒了！！！", "casual", "positive"),
    ("烦死了无语", "emotional_sharing", "frustrated"),
    ("我不开心", "emotional_sharing", "neutral"),  # 否定检测
])
def test_combined_intent_emotion(msg, exp_intent, exp_emotion):
    r = analyze_user_message(msg)
    assert r.intent == exp_intent, f"intent: {r.intent} != {exp_intent}"
    assert r.emotion == exp_emotion, f"emotion: {r.emotion} != {exp_emotion}"
