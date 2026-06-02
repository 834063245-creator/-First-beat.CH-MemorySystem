"""DeepSeekLLM 格式器 — 将引擎认知状态转换为 LLM 可消费的结构化上下文。

开源版不包含文本生成。本模块仅负责：
  - 将 UtteranceSpec 格式化为 LLM 执行指令
  - 将记忆/冲动数据序列化为 JSON/文本
  - 对外部 Agent 的 LLM 提供统一的结构化上下文
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DeepSeekLLM:
    """LLM 格式器 — 不做文本生成，只做认知状态的结构化输出。

    对应设计哲学第 5 条："引擎决策 → LLM 执行"。
    本类的职责是把引擎的 UtteranceSpec 翻译成 LLM 能直接
    使用的结构化数据，不涉及任何 API 调用。
    """

    # ── 模式发现引用（由外部注入）──
    _pattern_discovery: Any = None

    def __init__(self, pattern_discovery: Any = None):
        if pattern_discovery is not None:
            DeepSeekLLM._pattern_discovery = pattern_discovery

    @staticmethod
    def _build_execute_directive(utterance_spec: Any) -> dict:
        """从 UtteranceSpec.gate 提取执行指令，给 LLM 渲染回复用。

        返回 dict:
          - tone:        回复语气 (warm / caring / direct / soft / neutral)
          - formality:   正式度 0~1
          - intimacy:    亲密度 0~1
          - response_mode: 回复模式 (auto / question_first / soothe / direct_answer / confirm)
          - user_mood:   用户情绪
          - user_intent: 用户意图
        """
        try:
            gate = utterance_spec.gate
            user = utterance_spec.user

            return {
                "tone": getattr(gate, "tone", "warm"),
                "formality": round(getattr(gate, "formality", 0.3), 3),
                "intimacy": round(getattr(gate, "intimacy", 0.0), 3),
                "response_mode": getattr(gate, "response_mode", "auto"),
                "user_mood": getattr(user, "emotion", "neutral"),
                "user_intent": getattr(user, "intent", "casual"),
            }
        except Exception as e:
            logger.warning("_build_execute_directive 失败: %s", e)
            return {
                "tone": "warm",
                "formality": 0.3,
                "intimacy": 0.0,
                "response_mode": "auto",
                "user_mood": "neutral",
                "user_intent": "casual",
            }

    @staticmethod
    def _build_memories_for_tool(utterance_spec: Any) -> str:
        """将记忆指令序列化为 JSON 字符串，供外部 LLM 消费。

        输出结构：
          [
            {"role": "fact",       "summary": "...", "time_hint": "昨天", "emotional_context": "..."},
            {"role": "reference",  "summary": "...", "time_hint": "3天前"},
            ...
          ]
        """
        memories: list[dict] = []
        try:
            for mem in (utterance_spec.memories or []):
                memories.append(_serialize_memory_directive(mem, role="fact"))
            for mem in (utterance_spec.reference_memories or []):
                memories.append(_serialize_memory_directive(mem, role="reference"))
        except Exception as e:
            logger.warning("_build_memories_for_tool 失败: %s", e)

        return json.dumps(memories, ensure_ascii=False)

    @staticmethod
    def _build_impulses(utterance_spec: Any) -> str:
        """将冲动指令序列化为文本，每行一条冲动。

        输出示例：
          想问问你今天心情怎么样
          最近讨论过"机器学习"这个话题
        """
        lines: list[str] = []
        try:
            for imp in (utterance_spec.impulses or []):
                intent = getattr(imp, "intent", "")
                target = getattr(imp, "target_concept", "")

                if intent and target:
                    if intent == "recall":
                        lines.append(f'最近聊过「{target}」，要不要回顾一下？')
                    elif intent == "check":
                        lines.append(f'关心一下用户关于「{target}」的情况')
                    elif intent == "share_observation":
                        lines.append(f'分享关于「{target}」的观察')
                    elif intent == "emotional_check":
                        lines.append(f'关于「{target}」——用户情绪有变化，注意语气')
                    else:
                        lines.append(target)
                elif target:
                    lines.append(target)
        except Exception as e:
            logger.warning("_build_impulses 失败: %s", e)

        return "\n".join(lines)


def _serialize_memory_directive(mem: Any, *, role: str) -> dict:
    """将一条 MemoryDirective 序列化为 dict。"""
    return {
        "role": role,
        "summary": getattr(mem, "summary", ""),
        "time_hint": getattr(mem, "time_hint", None),
        "emotional_context": getattr(mem, "emotional_context", None),
        "certainty": round(getattr(mem, "certainty", 0.0), 3),
    }
