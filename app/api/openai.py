# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c7228bda

"""OpenAI Chat Completions API 兼容层。"""
import json
import time
import uuid


def parse_openai_messages(messages: list[dict]) -> tuple[str, str, list[dict]]:
    """解析 OpenAI 格式消息列表，提取 system prompt、最后一条 user message 和历史。"""
    system_prompt = ""
    user_message = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_message = content
    # 返回完整的 messages 列表作为 history（去除最后一条 user message）
    history = [m for m in messages if m.get("role") != "system"]
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    return system_prompt, user_message or "", history


def format_openai_chunk(model: str, content: str, finish_reason: str | None = None) -> str:
    """格式化 OpenAI SSE 流式 chunk。"""
    choice: dict = {"index": 0, "delta": {}}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    else:
        choice["delta"]["content"] = content
    chunk = {
        "id": "chatcmpl-" + str(uuid.uuid4())[:8],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


def format_openai_response(model: str, content: str) -> str:
    """格式化 OpenAI 非流式完整响应。"""
    return json.dumps({
        "id": "chatcmpl-" + str(uuid.uuid4())[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }, ensure_ascii=False)
