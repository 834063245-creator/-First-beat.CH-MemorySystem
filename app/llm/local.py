"""本地 LLM — 开源版提供基础的文本摘要功能。

开源版不包含完整的文本生成模型。摘要功能基于以下优先级：
  1. 如果配置了 DeepSeek API，委托给 DeepSeek 生成摘要
  2. 否则使用 jieba 做关键词抽取 + 启发式截断

工作记忆摘要通过此模块更新，实际文本生成由外部 Agent 的 LLM 完成。
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class LocalLLM:
    """本地 LLM 代理。

    开源版的 summarize() 尝试调用 DeepSeek API；不可用则回退到
    关键词提取 + 截断，保证系统不会因文本生成而崩溃。
    """

    def __init__(self, model: str | None = None):
        from app.config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
        self._api_key = LLM_API_KEY
        self._base_url = LLM_BASE_URL
        self._model = model or LLM_MODEL

    def summarize(self, text: str, max_chars: int = 200) -> str:
        """生成文本摘要。

        Args:
            text: 待摘要的 prompt 文本。
            max_chars: 摘要最大字数。

        Returns:
            摘要字符串。
        """
        if not text or not text.strip():
            return ""

        # 尝试调 DeepSeek API 生成摘要
        api_summary = self._try_api_summarize(text, max_chars)
        if api_summary:
            return api_summary

        # 回退：关键词提取 + 截断
        return self._fallback_summarize(text, max_chars)

    def _try_api_summarize(self, text: str, max_chars: int) -> str | None:
        """尝试通过 DeepSeek API 生成摘要，失败返回 None。"""
        if not self._api_key:
            return None

        clean_text = text[:2000]  # 限制 prompt 长度
        system_prompt = f"你是一个专业的摘要助手。请用中文输出摘要，不超过{max_chars}字。只输出摘要本身，不要加任何解释。"

        try:
            with httpx.Client(
                timeout=httpx.Timeout(30.0, connect=5.0)
            ) as client:
                resp = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": clean_text},
                        ],
                        "max_tokens": max_chars * 2,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    summary = choices[0].get("message", {}).get("content", "")
                    return summary.strip()
        except Exception as e:
            logger.warning("DeepSeek 摘要生成失败: %s", e)

        return None

    def _fallback_summarize(self, text: str, max_chars: int) -> str:
        """启发式摘要：提取核心提示词中的关键词 + 截断文本。

        从 prompt 中提取「当前在讨论什么」和「用户的核心关注点」。
        """
        import re

        # 尝试从 prompt 中找「用户：」和「助手：」最后几行的内容
        user_lines = re.findall(r"用户[：:]\s*(.+?)(?=\n|$)", text)
        if user_lines:
            # 取最后 2 条用户消息做摘要主体
            recent = user_lines[-2:]
            summary = "；".join(
                line[:80] for line in recent if len(line) >= 4
            )
            if summary and len(summary) >= 10:
                return summary[:max_chars]

        # 最简回退：取 prompt 的后半段
        clean = text.strip()
        if len(clean) <= max_chars:
            return clean
        return clean[-max_chars:] + "..."
