# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: a3fde25b

"""Tests for _handle_tool_call — the shared tool dispatch function."""
import pytest


class TestHandleToolCallExists:
    """验证 _handle_tool_call 函数存在且可调用。"""

    def test_function_exists(self):
        from app.api.chat import _handle_tool_call
        assert callable(_handle_tool_call)

    def test_is_async(self):
        from app.api.chat import _handle_tool_call
        import asyncio
        assert asyncio.iscoroutinefunction(_handle_tool_call)


class TestHandleToolCallContract:
    """验证工具调用的基本契约：追加到 extra_msgs。"""

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved(self):
        """传入 reasoning_content 时应出现在 assistant 消息中。"""
        from app.api.chat import _handle_tool_call
        tc = {
            "id": "call_test_003",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"test.txt"}'},
        }
        extra_msgs = []
        await _handle_tool_call(tc, extra_msgs, None,
                                reasoning_content="test reasoning", is_stream=False)
        assert extra_msgs[0].get("reasoning_content") == "test reasoning"

    @pytest.mark.asyncio
    async def test_unknown_tool_no_crash(self):
        """未知工具名不崩溃，静默跳过。"""
        from app.api.chat import _handle_tool_call
        tc = {
            "id": "call_test_004",
            "type": "function",
            "function": {"name": "nonexistent_tool_xyz", "arguments": "{}"},
        }
        extra_msgs = []
        # 不会抛异常
        await _handle_tool_call(tc, extra_msgs, None, is_stream=False)
        # 不应追加任何消息
        assert len(extra_msgs) == 0


class TestHandleToolCallToolNames:
    """验证所有已注册的工具名在分发函数中有对应分支。"""

    def test_all_registered_tools_have_handler(self):
        """检查 V3 注册的工具名列表和 _handle_tool_call 中的分支是否一致。"""
        from app.core.tools import (
            SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
        )
        registered_names = {
            SEARCH_WEB_TOOL["function"]["name"],
            READ_FILE_TOOL["function"]["name"],
            LIST_FILES_TOOL["function"]["name"],
            GREP_FILES_TOOL["function"]["name"],
        }
        import inspect
        from app.api.chat import _handle_tool_call
        source = inspect.getsource(_handle_tool_call)
        # 检查每个注册的工具名是否在函数体中被引用
        for name in registered_names:
            assert name in source, f"工具 {name} 在 _handle_tool_call 中缺少处理分支"
