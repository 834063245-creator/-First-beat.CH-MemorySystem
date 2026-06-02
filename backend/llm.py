"""LLM 模块 — 开源版不包含文本生成，仅保留格式器接口。

引擎只做记忆决策，语言生成由外部 Agent 的 LLM 完成。
DeepSeekLLM 类仅作为占位，不执行任何生成。
"""
import logging

logger = logging.getLogger(__name__)

# 格式桥接：从 app.llm.deepseek 导入格式器方法
from app.llm.deepseek import DeepSeekLLM as _DeepSeekFormatter

class DeepSeekLLM:
    """DeepSeek LLM 占位 — 开源版不生成文本。

    保留此类以满足 AppContext 等旧模块的初始化依赖。
    格式器方法（_build_execute_directive 等）已迁移至 app.llm.deepseek。
    """

    def __init__(self, *args, **kwargs):
        self._pattern_discovery = None
        logger.info("DeepSeekLLM 占位初始化 — 文本生成由外部 Agent 的 LLM 完成")

    def set_pattern_discovery(self, pd):
        """注入模式发现实例，供 MCP server 查询。"""
        self._pattern_discovery = pd


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
