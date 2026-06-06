"""测试 app/llm/deepseek.py — generate()/generate_stream() HTTP mock。

覆盖：完整消息构建路径 + HTTP 响应解析 + DSML 清理。
"""
import json
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.fixture
def client():
    from app.llm.deepseek import LLMClient
    llm = LLMClient()
    # 用 AsyncMock 替换 _client
    llm._client = AsyncMock()
    return llm


@pytest.fixture
def mock_resp():
    """模拟 DeepSeek API 响应。"""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{
            "message": {
                "content": "你好！今天天气不错。",
                "tool_calls": [],
                "reasoning_content": "",
            }
        }],
        "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 50, "prompt_cache_miss_tokens": 50},
    }
    return resp


class TestGenerateBasic:
    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_simple_response(self, mock_load, client, mock_resp):
        client._client.post = AsyncMock(return_value=mock_resp)
        result = await client.generate("你好", memories=[], personalities=["用户喜欢咖啡"])
        assert "content" in result
        assert client._client.post.called

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_with_cognitive_state(self, mock_load, client, mock_resp):
        from app.core.state import UtteranceSpec, UserMessageAnalysis, GatingDecision
        spec = MagicMock(spec=UtteranceSpec)
        spec.memories = [{"id": "m1", "summary": "记忆", "document": "文档",
                          "metadata": {}, "display_source": "", "score": 0.5}]
        spec.personality_notes = []
        spec.personality_notes_ai = []
        spec.impulses = []
        spec.user = MagicMock()
        spec.user.raw_text = ""
        spec.emotional_reversals = []
        spec.stale_context = []
        spec.woven_context = None
        spec.relationship = None
        spec.gate = MagicMock()
        spec.gate.tone = "warm"
        spec.gate.response_mode = "auto"
        spec.mirror_prediction = None
        client._client.post = AsyncMock(return_value=mock_resp)
        result = await client.generate("你好", cognitive_state=spec)
        assert "content" in result

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_with_tools(self, mock_load, client, mock_resp):
        client._client.post = AsyncMock(return_value=mock_resp)
        tools = [{"type": "function", "function": {"name": "search_web", "parameters": {}}}]
        result = await client.generate("搜索天气", memories=[], tools=tools)
        assert "content" in result
        # 验证 body 中有 tools
        call_args = client._client.post.call_args
        body = call_args[1]["json"]
        assert "tools" in body

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_timeout_raised(self, mock_load, client):
        import httpx
        client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            await client.generate("测试")

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_with_tool_calls_in_response(self, mock_load, client):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "tc1", "type": "function",
                        "function": {"name": "search_web", "arguments": '{"query":"天气"}'}
                    }],
                }
            }],
            "usage": {"prompt_tokens": 50},
        }
        client._client.post = AsyncMock(return_value=resp)
        result = await client.generate("搜索", memories=[])
        assert result["tool_calls"]


class TestGenerateStream:
    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_stream_yields_content(self, mock_load, client):
        """测试流式生成的 token yield 路径。"""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)
        # 模拟 HTTP 流式响应
        mock_resp_obj = MagicMock()
        mock_resp_obj.raise_for_status = MagicMock()
        mock_resp_obj.headers = {"eo-cache-status": "hit"}
        mock_stream.__aenter__.return_value = mock_resp_obj
        # 模拟 stream 的 aiter_lines
        fake_lines = [
            'data: {"choices":[{"delta":{"content":"你"}}]}',
            'data: {"choices":[{"delta":{"content":"好"}}]}',
            'data: [DONE]',
        ]
        async def _fake_aiter_lines():
            for line in fake_lines:
                yield line
        mock_resp_obj.aiter_lines = _fake_aiter_lines
        client._client.stream = MagicMock(return_value=mock_stream)
        tokens = []
        async for tag, token in client.generate_stream("你好"):
            if tag == "content":
                tokens.append(token)
        assert "".join(tokens) == "你好"

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_stream_handles_timeout(self, mock_load, client):
        import httpx
        client._client.stream = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            async for _ in client.generate_stream("test"):
                pass


class TestBuildPromptAdvanced:
    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_with_context_before_after(self, mock_load):
        from app.llm.deepseek import LLMClient
        mem = {
            "id": "m1",
            "document": "main doc",
            "metadata": {"timestamp": 1700000000, "hit_count": 10},
            "source": "semantic",
            "context_before": [{"user": "之前的问题", "ai": "之前的回答"}],
            "context_after": [{"user": "之后的问题", "ai": "之后的回答"}],
        }
        prompt = LLMClient._build_prompt([mem])
        assert "之前的问题" in prompt
        assert "之前的回答" in prompt
        assert "之后的问题" in prompt

    def test_benchmark_mode_adds_rule(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.personality_notes = []
        mock_state.personality_notes_ai = []
        with patch("app.llm.deepseek.BENCHMARK_MODE", True):
            with patch("app.llm.deepseek.load_system_prompt", return_value="SYS"):
                prompt = LLMClient._build_stable_system_prompt(mock_state)
                assert "知识更新冲突解决" in prompt


class TestEdgeCases:
    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    @pytest.mark.asyncio
    async def test_generate_without_api_key(self, mock_load):
        from app.llm.deepseek import LLMClient
        with patch("app.llm.deepseek.LLM_API_KEY", ""):
            llm = LLMClient()
            llm._client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok", "tool_calls": []}}],
                "usage": {},
            }
            llm._client.post = AsyncMock(return_value=mock_resp)
            result = await llm.generate("hi")
            assert "content" in result
