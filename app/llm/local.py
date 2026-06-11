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


# ── httpx 客户端单例（模块级复用，避免每次调用重建 TCP+TLS）──
_local_client: httpx.Client | None = None
_local_client_lock = __import__("threading").Lock()


def _get_local_client() -> httpx.Client:
    """获取或创建模块级 httpx.Client 单例（连接池复用）。"""
    global _local_client
    if _local_client is None:
        with _local_client_lock:
            if _local_client is None:
                _local_client = httpx.Client(
                    timeout=httpx.Timeout(15.0, connect=3.0),
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                )
    return _local_client


class LocalLLM:
    """本地 LLM 代理 — 用 Ollama qwen2.5:3b 做摘要 / 通用生成。"""

    def __init__(self, model: str | None = None):
        self._model = model or _SUMMARIZE_MODEL

    def generate(self, prompt: str, max_tokens: int = 1024) -> str | None:
        """通用生成接口 — 发送任意 prompt 到本地 Ollama 模型。

        用于画像合成等需要 LLM 自由生成文本的场景。
        失败返回 None，调用方应优雅降级。
        """
        if not prompt or not prompt.strip():
            return None
        try:
            client = _get_local_client()
            resp = client.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return (data.get("response") or "").strip() or None
        except Exception as e:
            logger.warning("LocalLLM generate 失败: %s", e)
            return None

    def summarize(self, text: str, max_chars: int = 200, *, fast: bool = False) -> str:
        """通过本地 Ollama 生成文本摘要，零 API 费用。

        fast=True 时跳过 LLM，直接用截断回退（benchmark / 极速场景）。
        """
        if not text or not text.strip():
            return ""

        clean = text.strip()[:1500]
        if fast:
            return self._fallback(text, max_chars)

        prompt = (
            f"请用一句话概括以下对话的核心内容（不超过{max_chars}字）：\n\n{clean}"
        )

        try:
            client = _get_local_client()
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
