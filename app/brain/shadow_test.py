"""ChuchenBrain 影子测试 — 对比规则和模型输出。

用法: python -m app.brain.shadow_test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass
from app.brain.models import ChuchenBrain


@dataclass
class _ShadowItem:
    """单次分析的对比结果。"""
    user_message: str
    rule_intent: str
    model_intent: str
    rule_emotion: str
    model_emotion: str
    rule_gate_tone: str
    model_gate_tone: str

    @property
    def intent_match(self) -> bool:
        return self.rule_intent == self.model_intent

    @property
    def emotion_match(self) -> bool:
        return self.rule_emotion == self.model_emotion


TEST_MESSAGES = [
    "最近压力好大，项目快崩了",
    "我今天心情特别好，终于把那个 Bug 修掉了",
    "好累啊，不想动了",
    "我今天被老板骂了一顿",
    "你还记得上次我们聊的那个架构方案吗",
    "之前我们说过的那件事，我现在有新的想法了",
    "不对，你说的不是这样的",
    "你搞错了，根本不是这个意思",
    "帮我看看这段代码为什么报错",
    "能不能帮我写一个数据库迁移脚本",
    "Python 的 GIL 是什么东西",
    "我觉得这个世界有时候挺荒诞的",
    "你知道吗，我今天走在路上突然想起一件事",
    "行吧，就这样吧",
    "太好了，又是一个通宵",
    "算了不说了",
]


def build_shadow(brain: ChuchenBrain, text: str) -> _ShadowItem:
    """对一条消息同时跑规则和模型，返回对比结果。"""
    rule_intent = brain.intent_classifier._rule_classify(text)
    rule_emotion = brain.emotion_analyzer._rule_analyze(text)
    model_intent = brain.classify_intent(text)
    model_emotion = brain.analyze_emotion(text)
    rule_gate = brain.gate_maker._rule_decide(rule_intent, rule_emotion.primary)
    model_gate = brain.gate_maker.decide(model_intent.intent, model_emotion.primary)
    return _ShadowItem(
        user_message=text,
        rule_intent=rule_intent,
        model_intent=model_intent.intent,
        rule_emotion=rule_emotion.primary,
        model_emotion=model_emotion.primary,
        rule_gate_tone=rule_gate.tone,
        model_gate_tone=model_gate.tone,
    )


def main():
    brain = ChuchenBrain()
    brain.load_all()

    print("\n  ChuchenBrain 影子测试 — 规则 vs 模型")
    print("=" * 70)
    results = [build_shadow(brain, m) for m in TEST_MESSAGES]

    for i, shadow in enumerate(results, 1):
        im = "[=]" if shadow.intent_match else "[X]"
        em = "[=]" if shadow.emotion_match else "[X]"
        print(f'  [{i:2d}] "{shadow.user_message}"')
        print(f"       意图 {im}  规则={shadow.rule_intent:<20s}  模型={shadow.model_intent:<20s}")
        print(f"       情绪 {em}  规则={shadow.rule_emotion:<15s}  模型={shadow.model_emotion:<15s}")
        print(f"       门控  规则=tone:{shadow.rule_gate_tone:<8s}  模型=tone:{shadow.model_gate_tone:<8s}")
        print()

    intent_ok = sum(1 for r in results if r.intent_match)
    emotion_ok = sum(1 for r in results if r.emotion_match)
    total = len(results)
    print(f"  意图一致: {intent_ok}/{total} ({100*intent_ok//total}%)")
    print(f"  情绪一致: {emotion_ok}/{total} ({100*emotion_ok//total}%)")


if __name__ == "__main__":
    main()
