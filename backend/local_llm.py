"""[SHIM] 本地 LLM — 桥接到 app/llm/local.py。

开源版的 summarize() 尝试调 DeepSeek API 生成摘要，不可用时回退到关键词提取。
"""
from app.llm.local import LocalLLM  # noqa: F401
