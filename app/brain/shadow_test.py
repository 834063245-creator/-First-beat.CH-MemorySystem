"""ChuchenBrain 影子测试 — 用一批消息对比规则和模型输出。

现在模型还没接入，所以规则侧和模型侧一致（都走规则兜底）。
目标：模型接入后，这里立刻就能看到差异。

Usage from 初痕根目录:
    python -m app.brain.shadow_test
"""

import sys
import os

# 确保能从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.brain.models import ChuchenBrain, ShadowResult


TEST_MESSAGES = [
    # ── 情绪分享 ──
    "最近压力好大，项目快崩了",
    "我今天心情特别好，终于把那个 Bug 修掉了",
    "好累啊，不想动了",
    "我今天被老板骂了一顿",

    # ── 回忆 ──
    "你还记得上次我们聊的那个架构方案吗",
    "之前我们说过的那件事，我现在有新的想法了",

    # ── 冲突 ──
    "不对，你说的不是这样的",
    "你搞错了，根本不是这个意思",

    # ── 请求 ──
    "帮我看看这段代码为什么报错",
    "能不能帮我写一个数据库迁移脚本",

    # ── 事实询问 ──
    "Python 的 GIL 是什么东西",

    # ── 边界情况（关键词缺失但语义明确） ──
    "我觉得这个世界有时候挺荒诞的",
    "你知道吗，我今天走在路上突然想起一件事",
    "行吧，就这样吧",
    "太好了，又是一个通宵",          # 讽刺
    "算了不说了",
]


def print_separator(title: str = ""):
    if title:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
    else:
        print(f"{'─'*70}")


def main():
    print("\n" + "=" * 70)
    print("  ChuchenBrain 影子测试 — 规则 vs 模型（当前均为规则兜底）")
    print("=" * 70)
    print()

    brain = ChuchenBrain(model_name="qwen2.5:3b")
    
    # 加载模型
    load_status = brain.load_all()
    print(f"  模型加载状态: {load_status}")
    if any(load_status.values()):
        print(f"  [!!] Ollama 模型已连接！下方将显示规则 vs 模型的真实对比。")
        print(f"       model 列 = qwen2.5:3b 的 Prompt 推理结果")
    else:
        print(f"  当前模式: 纯规则兜底（Ollama 未连接或模型不可用）")
    print()

    # ── 逐个测试 ──
    for i, msg in enumerate(TEST_MESSAGES, 1):
        shadow = brain.shadow_analyze(msg)

        intent_icon = "[=]" if shadow.intent_match else "[X]"
        emotion_icon = "[=]" if shadow.emotion_match else "[X]"
        gate_icon = "[=]" if shadow.rule_gate.tone == shadow.model_gate.tone else "[X]"

        print(f"  [{i:2d}] \"{msg}\"")
        print(f"       意图  {intent_icon}  规则={shadow.rule_intent.intent:<20s}  模型={shadow.model_intent.intent:<20s}  ({shadow.model_intent.source})")
        print(f"       情绪  {emotion_icon}  规则={shadow.rule_emotion.primary:<20s}  模型={shadow.model_emotion.primary:<20s}  ({shadow.model_emotion.source})")
        print(f"       门控  {gate_icon}  规则=tone:{shadow.rule_gate.tone:<8s} form:{shadow.rule_gate.formality} mode:{shadow.rule_gate.response_mode:<18s}")
        print(f"              模型=tone:{shadow.model_gate.tone:<8s} form:{shadow.model_gate.formality} mode:{shadow.model_gate.response_mode}")
        print()

    # ── 统计 ──
    results = [brain.shadow_analyze(m) for m in TEST_MESSAGES]
    intent_match = sum(1 for r in results if r.intent_match)
    emotion_match = sum(1 for r in results if r.emotion_match)
    total = len(results)

    print_separator("统计")
    print(f"  意图一致: {intent_match}/{total}  ({intent_match/total*100:.0f}%)")
    print(f"  情绪一致: {emotion_match}/{total}  ({emotion_match/total*100:.0f}%)")
    print()
    print(f"  [!] 当前全一致是因为模型侧也在用规则兜底。")
    print(f"  接入真实模型后，这里会显示出规则 vs 模型的真实差异。")
    print()

    # ── 逐条列出初痕自己的盲区 ──
    print_separator("初痕规则已知盲区（模型接入后会改善）")

    known_blind_spots = [
        ("最近压力好大，项目快崩了", "casual", "emotional_sharing",
         '关键词"崩溃""压力"都在情绪词表中，但不在intent关键词表中'),
        ("我今天被老板骂了一顿", "casual", "emotional_sharing",
         '关键词"骂"不在任何词表中，纯规则无法识别'),
        ("太好了，又是一个通宵", "casual", "emotional_sharing (负面)",
         '正面词"太好了"触发positive，但整句是反讽/抱怨，规则无法处理'),
        ("行吧，就这样吧", "casual", "emotional_sharing (无奈)",
         '放弃/无奈的语气，没有任何关键词'),
        ("算了不说了", "casual", "emotional_sharing (回避)",
         '回避型表达，规则词表无法匹配'),
    ]

    for msg, current, expected, reason in known_blind_spots:
        print(f"  \"{msg}\"")
        print(f"    当前: {current}  →  期望: {expected}")
        print(f"    原因: {reason}")
        print()

    print(f"  接入小模型后，以上盲区应该全部消失。")
    print()


if __name__ == "__main__":
    main()
