"""测试 app/api/openai.py — OpenAI Chat Completions API 兼容层。

覆盖：parse_openai_messages / format_openai_chunk / format_openai_response 纯函数。
"""
import json
from app.api.openai import parse_openai_messages, format_openai_chunk, format_openai_response


class TestParseOpenaiMessages:
    def test_extracts_system_prompt(self):
        msgs = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
        sys_prompt, user_msg, history = parse_openai_messages(msgs)
        assert sys_prompt == "你是助手"
        assert user_msg == "你好"

    def test_last_user_message_extracted(self):
        msgs = [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
        ]
        sys_prompt, user_msg, history = parse_openai_messages(msgs)
        assert user_msg == "第二问"
        # history 不含最后一条 user message
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "第一问"
        assert history[1]["role"] == "assistant"

    def test_no_system_prompt(self):
        msgs = [
            {"role": "user", "content": "hello"},
        ]
        sys_prompt, user_msg, history = parse_openai_messages(msgs)
        assert sys_prompt == ""
        assert user_msg == "hello"
        assert history == []

    def test_multiple_system_messages(self):
        """多 system message 时取最后一条。"""
        msgs = [
            {"role": "system", "content": "v1"},
            {"role": "system", "content": "v2"},
            {"role": "user", "content": "hi"},
        ]
        sys_prompt, user_msg, _ = parse_openai_messages(msgs)
        assert sys_prompt == "v2"

    def test_history_excludes_system(self):
        msgs = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        _, _, history = parse_openai_messages(msgs)
        # history 不含 system，不含最后一条 user
        assert len(history) == 2
        assert all(m["role"] != "system" for m in history)


class TestFormatOpenaiChunk:
    def test_produces_sse_format(self):
        chunk = format_openai_chunk("test-model", "hello")
        assert chunk.startswith("data: ")
        data = json.loads(chunk[6:].strip())
        assert data["object"] == "chat.completion.chunk"
        assert "choices" in data
        assert data["choices"][0]["delta"]["content"] == "hello"

    def test_finish_reason(self):
        chunk = format_openai_chunk("m", "", finish_reason="stop")
        data = json.loads(chunk[6:].strip())
        assert data["choices"][0]["finish_reason"] == "stop"


class TestFormatOpenaiResponse:
    def test_produces_valid_json(self):
        resp = format_openai_response("test-model", "full answer")
        data = json.loads(resp)
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "full answer"
        assert data["choices"][0]["finish_reason"] == "stop"
