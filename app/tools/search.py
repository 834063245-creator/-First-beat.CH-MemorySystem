# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 950df7d6

"""网页搜索 — 基于博查 Search API（bochaai.com）。"""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")
BOCHA_BASE_URL = "https://api.bochaai.com/v1/ai/search"


async def search_web(query: str, count: int = 5) -> str:
    """调用博查搜索 API，返回格式化搜索结果文本。"""
    if not query or not query.strip():
        return ""

    if not BOCHA_API_KEY:
        logger.warning("search_web 未配置 BOCHA_API_KEY，跳过搜索")
        return ""

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(
                BOCHA_BASE_URL,
                headers={
                    "Authorization": f"Bearer {BOCHA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"query": query.strip(), "count": count},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        pages = data.get("data", {}).get("pages", []) if isinstance(data, dict) else []
        for i, page in enumerate(pages[:count]):
            title = page.get("title", "")
            snippet = page.get("snippet", "")
            url = page.get("url", "")
            if title or snippet:
                results.append(f"[{i + 1}] {title}\n{snippet}\n{url}")

        if not results:
            return f"搜索「{query}」未找到结果。"

        return "\n\n".join(results)

    except httpx.TimeoutException:
        logger.warning("search_web 超时: %s", query[:50])
        return f"搜索「{query}」超时，请稍后重试。"
    except Exception as exc:
        logger.warning("search_web 失败: %s", exc)
        return f"搜索「{query}」失败: {exc}"
