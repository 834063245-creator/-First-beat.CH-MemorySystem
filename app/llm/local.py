"""本地 LLM — 通过 Ollama qwen2.5:3b 做摘要，零 API 费用。

已存在的 qwen2.5:3b（实体抽取用）被复用为摘要模型，
不需额外加载，不占用额外显存。
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_SUMMARIZE_MODEL = "qwen2.5:3b"
_OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")


class LocalLLM:
    """本地 LLM 代理 — 用 Ollama qwen2.5:3b 做摘要。"""

    def __init__(self, model: str | None = None):
        self._model = model or _SUMMARIZE_MODEL

    def summarize(self, text: str, max_chars: int = 200) -> str:
        """通过本地 Ollama 生成文本摘要，零 API 费用。"""
        if not text or not text.strip():
            return ""

        clean = text.strip()[:1500]
        prompt = (
            f"请用一句话概括以下对话的核心内容（不超过{max_chars}字）：\n\n{clean}"
        )

        try:
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=3.0)) as client:
                resp = client.post(
                    f"{_OLLAMA_URL}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": max_chars},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                summary = (data.get("response") or "").strip()
                if summary:
                    return summary[:max_chars]
        except Exception as e:
            logger.warning("Ollama 摘要失败: %s", e)

        # 回退：关键词截断
        return self._fallback(text, max_chars)

    @staticmethod
    def _fallback(text: str, max_chars: int) -> str:
        import re
        user_lines = re.findall(r"用户[：:]\s*(.+?)(?=\n|$)", text)
        if user_lines:
            recent = user_lines[-2:]
            summary = "；".join(line[:80] for line in recent if len(line) >= 4)
            if summary and len(summary) >= 10:
                return summary[:max_chars]
        clean = text.strip()
        return clean[:max_chars] if len(clean) > max_chars else clean
