"""OpenAI 兼容层已移除 — 开源版不包含聊天端点。"""
import json


def parse_openai_messages(payload: dict):
    return []


def format_openai_chunk(content: str, done: bool = False):
    return json.dumps({})


def format_openai_response(content: str):
    return json.dumps({})
