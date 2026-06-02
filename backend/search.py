"""网页搜索已移除 — 开源版不包含外部 API 调用。"""
import logging

logger = logging.getLogger(__name__)


def search_web(query: str, count: int = 5) -> list:
    logger.debug("search_web 不可用（开源版）")
    return []
