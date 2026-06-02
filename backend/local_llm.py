"""本地 LLM 已移除 — 开源版不包含文本生成。"""
import logging

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, *args, **kwargs):
        pass

    def summarize(self, text: str) -> str:
        logger.debug("local_llm.summarize 不可用（开源版），返回截断摘要")
        return text[:50] + "..." if len(text) > 50 else text


# 兼容旧函数式调用
def summarize(text: str) -> str:
    logger.debug("local_llm.summarize 不可用（开源版），返回截断摘要")
    return text[:50] + "..." if len(text) > 50 else text

