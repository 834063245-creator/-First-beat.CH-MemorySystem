# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: e5f8a96d

"""导出训练数据：从 Qdrant + ChatHistory 提取意图和情绪标注

用法: python -m app.brain.export_training_data
输出: app/brain/training_data.jsonl
"""
import sys
import os
import json
import logging
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.disable(logging.CRITICAL)

# ── 意图分类器（和 circuit.py 一致）────────────────
_INTENT_KEYWORDS = {
    "conflict": ["不对", "不是", "你错了", "别说了", "你搞错了", "乱说",
                 "你没听懂", "不是这样"],
    "emotional_sharing": ["想", "觉得", "感觉", "心情", "难过", "开心", "烦",
                          "累", "困", "疲惫", "焦虑", "担心", "感动", "温暖",
                          "梦到", "失眠", "心疼", "好烦", "好累", "好开心", "好难过"],
    "recall": ["记得", "之前", "上次", "以前", "曾经", "想起", "是不是说过",
               "那时候", "那会儿", "还记得", "记不记得"],
    "request": ["帮我", "请你", "需要你", "帮我查", "帮我找", "帮我写",
                "帮我改", "帮我看看", "能不能帮我"],
    "ask_fact": ["什么", "怎么", "为什么", "如何", "能不能", "请问", "啥", "是不是"],
    "meta": ["你是谁", "你能做什么", "你会什么", "你叫什么", "你有什么功能", "你是吗"],
}

_EMOTION_KEYWORDS = {
    "intimate": ["想你", "爱", "心疼", "抱", "陪", "温暖", "梦到", "亲", "在乎", "爱你", "抱抱"],
    "positive": ["开心", "高兴", "好", "棒", "喜欢", "感动", "幸福", "感谢", "太棒", "太好了", "不错", "厉害"],
    "negative": ["难过", "烦", "累", "焦虑", "担心", "生气", "讨厌", "失望",
                 "痛苦", "崩溃", "孤独", "压力", "郁闷", "烦躁", "骂"],
    "frustrated": ["烦死了", "受不了", "无语", "气死", "崩溃", "不想说了", "够了", "算了吧"],
}

def classify_intent(text):
    for intent in ["conflict", "emotional_sharing", "recall", "request", "ask_fact", "meta"]:
        for kw in _INTENT_KEYWORDS[intent]:
            if kw in text:
                return intent
    return "casual"

def classify_emotion(text):
    for label in ["intimate", "frustrated", "negative", "positive"]:
        for kw in _EMOTION_KEYWORDS[label]:
            if kw in text:
                return label
    return "neutral"

def main():
    # ── 从 ChatHistory JSONL 捞数据 ──
    # 从 __file__ 算出项目根
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    history_paths = [
        os.path.join(project_root, "backend", "data", "chat_history.jsonl"),
        os.path.join(project_root, "instances", "predecessor", "data", "chat_history.jsonl"),
    ]

    records = []
    for hp in history_paths:
        if not os.path.exists(hp):
            continue
        with open(hp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = rec.get("user_message", "")
                if msg and len(msg.strip()) >= 3:
                    records.append(msg.strip())

    # ── 标注 ──
    print(f"从 {len(history_paths)} 个 JSONL 文件找到 {len(records)} 条用户消息")

    intents = defaultdict(int)
    emotions = defaultdict(int)

    with open(os.path.join(os.path.dirname(__file__), "training_data.jsonl"), "w", encoding="utf-8") as out:
        for msg in records:
            intent = classify_intent(msg)
            emotion = classify_emotion(msg)
            intents[intent] += 1
            emotions[emotion] += 1
            out.write(json.dumps({"text": msg, "intent": intent, "emotion": emotion},
                                 ensure_ascii=False) + "\n")

    print("\n意图分布:")
    for k, v in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"  {k:<20s}: {v}")

    print("\n情绪分布:")
    for k, v in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {k:<20s}: {v}")

    print(f"\n训练数据已导出: app/brain/training_data.jsonl ({len(records)} 条)")
    print("下一步: 用这些数据微调 MiniLM-L6-v2 → <10ms 推理")

if __name__ == "__main__":
    main()
