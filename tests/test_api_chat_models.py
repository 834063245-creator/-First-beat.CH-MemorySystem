"""测试 app/api/chat.py 纯端点 + 工具调度函数。

覆盖：/v1/models /chat/stream 空消息 / _handle_tool_call 工具路由。
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.api.app import app
    return TestClient(app)


class TestOpenaiModels:
    def test_returns_model_list(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    def test_model_has_required_fields(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        model = data["data"][0]
        assert "id" in model
        assert model["owned_by"] == "初痕"


class TestChatStreamEmpty:
    def test_empty_message_returns_prompt(self, client):
        resp = client.post("/chat/stream", json={"message": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert "说点什么" in data["response"]


class TestHandleToolCall:
    @pytest.mark.asyncio
    async def test_read_file_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_01",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "/tmp/test.txt"}'}
        }
        with patch("app.api.chat.read_file", return_value="file content"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert len(extra_msgs) == 2
        assert extra_msgs[0]["role"] == "assistant"
        assert extra_msgs[1]["role"] == "tool"
        assert extra_msgs[1]["content"] == "file content"

    @pytest.mark.asyncio
    async def test_list_files_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_02",
            "type": "function",
            "function": {"name": "list_files", "arguments": '{"pattern": "*.py"}'}
        }
        with patch("app.api.chat.list_files", return_value="a.py\nb.py"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert extra_msgs[1]["content"] == "a.py\nb.py"

    @pytest.mark.asyncio
    async def test_write_file_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_03",
            "type": "function",
            "function": {"name": "write_file", "arguments": '{"path": "x.txt", "content": "hello"}'}
        }
        with patch("app.api.chat.write_file", return_value="已写入: x.txt"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert "已写入" in extra_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_grep_files_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_04",
            "type": "function",
            "function": {"name": "grep_files", "arguments": '{"pattern": "def test_"}'}
        }
        with patch("app.api.chat.grep_files", return_value="test.py:1: def test_foo"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert "def test_foo" in extra_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_glob_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_05",
            "type": "function",
            "function": {"name": "glob", "arguments": '{"pattern": "*.txt", "root": "."}'}
        }
        await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert len(extra_msgs) == 2
        # glob 结果可能是 "未匹配到文件" 或实际匹配

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_06",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "x"}'}
        }
        with patch("app.api.chat.read_file", return_value="OK"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx, reasoning_content="思考中...")
        assert extra_msgs[0].get("reasoning_content") == "思考中..."

    @pytest.mark.asyncio
    async def test_search_web_async(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_07",
            "type": "function",
            "function": {"name": "search_web", "arguments": '{"query": "天气"}'}
        }
        with patch("app.api.chat.search_web", new_callable=AsyncMock, return_value="搜索结果"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert "搜索结果" in extra_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_edit_file_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_edit",
            "type": "function",
            "function": {"name": "edit_file", "arguments": '{"path": "test.py", "old_str": "hello", "new_str": "world"}'}
        }
        with patch("app.api.chat.edit_file", return_value="已编辑: test.py"):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert len(extra_msgs) == 2
        assert extra_msgs[1]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_edit_file_exception(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_edit_err",
            "type": "function",
            "function": {"name": "edit_file", "arguments": '{"path": "x", "old_str": "a", "new_str": "b"}'}
        }
        with patch("app.api.chat.edit_file", side_effect=Exception("编辑失败")):
            with pytest.raises(Exception, match="编辑失败"):
                await _handle_tool_call(tc, extra_msgs, mock_ctx)

    @pytest.mark.asyncio
    async def test_bash_dispatched(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_bash",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo hello"}'}
        }
        with patch("subprocess.run", return_value=MagicMock(stdout="hello\n", stderr="")):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert len(extra_msgs) == 2

    @pytest.mark.asyncio
    async def test_bash_timeout(self):
        from app.api.chat import _handle_tool_call
        import subprocess
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_bash_timeout",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "sleep 999"}'}
        }
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sleep", 30)):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert len(extra_msgs) == 2
        assert "超时" in extra_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_bash_exception(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_bash_err",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "rm -rf /"}'}
        }
        with patch("subprocess.run", side_effect=Exception("禁止执行")):
            await _handle_tool_call(tc, extra_msgs, mock_ctx)
        assert any("禁止执行" in m.get("content", "") for m in extra_msgs if m.get("role") == "tool")

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        from app.api.chat import _handle_tool_call
        mock_ctx = MagicMock()
        extra_msgs = []
        tc = {
            "id": "call_unknown",
            "type": "function",
            "function": {"name": "unknown_tool", "arguments": '{}'}
        }
        await _handle_tool_call(tc, extra_msgs, mock_ctx)
        # unknown tool silently skipped, no extra messages
        assert len(extra_msgs) == 0
