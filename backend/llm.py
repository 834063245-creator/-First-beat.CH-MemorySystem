"""LLM 模块 — 开源版不生成文本，但保留完整格式器接口。

引擎只做记忆决策，语言生成由外部 Agent 的 LLM 完成。
此类继承自 app.llm.deepseek.DeepSeekLLM，格式器方法（_build_execute_directive 等）
完全可用，仅 generate/generate_stream 不产生文本。
"""
from app.llm.deepseek import DeepSeekLLM as _RealDeepSeekLLM


class DeepSeekLLM(_RealDeepSeekLLM):
    """DeepSeekLLM — 继承完整格式器，覆盖文本生成。

    格式器方法（_build_execute_directive / _build_memories_for_tool / _build_impulses）
    全部来自父类。generate() / generate_stream() 返回空，因为开源版不生成文本。
    """

    def generate(self, *args, **kwargs):
        """开源版不生成文本。"""
        return ""

    def generate_stream(self, *args, **kwargs):
        """开源版不生成文本。"""
        yield ""


def now_hint():
    return ""


def parse_dsml_tool_calls(content: str):
    return []


def strip_dsml(content):
    return content


def load_system_prompt():
    return ""


def extract_query_tags(text):
    return []
