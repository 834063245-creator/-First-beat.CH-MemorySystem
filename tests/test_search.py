# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: ce7539fb

"""测试 app/tools/search.py — 博查网页搜索。

覆盖：search_web 的各种边界条件和错误处理。
"""
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestSearchWeb:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from app.tools.search import search_web
        result = await search_web("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self):
        from app.tools.search import search_web
        result = await search_web("   ")
        assert result == ""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        from app.tools.search import search_web
        with patch("app.tools.search.BOCHA_API_KEY", ""):
            result = await search_web("测试查询")
            assert result == ""

    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "pages": [
                    {"title": "结果1", "snippet": "摘要1", "url": "http://example.com/1"},
                    {"title": "结果2", "snippet": "摘要2", "url": "http://example.com/2"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.tools.search.BOCHA_API_KEY", "test-key"):
            with patch("app.tools.search.httpx.AsyncClient", return_value=mock_client):
                from app.tools.search import search_web
                result = await search_web("测试查询")
                assert "[1]" in result
                assert "结果1" in result
                assert "http://example.com/1" in result

    @pytest.mark.asyncio
    async def test_empty_pages_returns_not_found(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"pages": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.tools.search.BOCHA_API_KEY", "test-key"):
            with patch("app.tools.search.httpx.AsyncClient", return_value=mock_client):
                from app.tools.search import search_web
                result = await search_web("稀有查询")
                assert "未找到结果" in result

    @pytest.mark.asyncio
    async def test_http_error_handled(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with patch("app.tools.search.BOCHA_API_KEY", "test-key"):
            with patch("app.tools.search.httpx.AsyncClient", return_value=mock_client):
                from app.tools.search import search_web
                result = await search_web("查询")
                assert "超时" in result

    @pytest.mark.asyncio
    async def test_generic_exception_handled(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=RuntimeError("unknown error"))

        with patch("app.tools.search.BOCHA_API_KEY", "test-key"):
            with patch("app.tools.search.httpx.AsyncClient", return_value=mock_client):
                from app.tools.search import search_web
                result = await search_web("查询")
                assert "失败" in result
                assert "unknown error" in result
