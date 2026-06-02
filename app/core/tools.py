"""LLM 工具定义与调度 — 从 backend/main.py 迁移至此。

包含 OpenAI 兼容的工具定义常量，以及工具调用分发函数 _handle_tool_call。
"""
import json
import logging
import subprocess

from app.api.deps import AppContext

logger = logging.getLogger(__name__)


# ── 工具定义（OpenAI 兼容格式） ──────────────────────────────

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "实时搜索互联网获取最新信息。当你需要回答关于时事、实时数据、具体事实或任何你不确定的信息时，调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，尽量具体",
                },
            },
            "required": ["query"],
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取工作区文件的内容。当用户要求你「读一下某个文件」时调用。返回文件文本内容，只读不写。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径，可以是绝对路径或相对路径。例如 'backend/main.py' 或 'D:/amazing2/backend/main.py'",
                },
            },
            "required": ["path"],
        },
    },
}

LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "按 glob 模式列出工作区中的文件。支持通配符：* 匹配任意字符，** 递归目录。例如 '*.py'、'backend/**/*.txt'。当你想了解项目结构或找某个文件时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 匹配模式，如 'backend/*.py'、'**/*.md'、'backend/scripts/*.py'",
                },
            },
            "required": ["pattern"],
        },
    },
}

GREP_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "grep_files",
        "description": "在项目文件中搜索文本或正则表达式。当用户问「哪里定义了 xxx」「找一下包含 xxx 的文件」时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "要搜索的文本或正则表达式",
                },
                "glob_pattern": {
                    "type": "string",
                    "description": "文件过滤 glob 模式，默认 '**/*.py'，也支持 '**/*.md'、'**/*.{py,txt}' 等",
                },
            },
            "required": ["pattern"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "写入或覆写文件。如果要修改已有文件的部分内容，用 edit_file。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["path", "content"],
        },
    },
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "精确字符串替换——在文件中找到 old_str 并替换为 new_str。不改动文件其他部分。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_str": {"type": "string", "description": "要被替换的精确文本"},
                "new_str": {"type": "string", "description": "替换后的文本"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 shell 命令。用于运行脚本、编译代码、启动服务等。返回 stdout+stderr。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
            },
            "required": ["command"],
        },
    },
}

GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "glob",
        "description": "按 glob 模式搜索文件路径。例如 '**/*.py' 找所有 Python 文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob 匹配模式"},
                "root": {"type": "string", "description": "搜索根目录，默认当前项目"},
            },
            "required": ["pattern"],
        },
    },
}

TOOLS = [
    SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
    WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL,
]


# ── 工具调度 ──────────────────────────────────────────────────

async def handle_tool_call(tc: dict, extra_msgs: list, ctx: AppContext, *,
                           reasoning_content: str = "", is_stream: bool = False):
    """执行一个工具调用，追加结果到 extra_msgs。"""
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}

    asst_msg = {"role": "assistant", "tool_calls": [tc]}
    if reasoning_content:
        asst_msg["reasoning_content"] = reasoning_content

    if name == "search_web":
        from backend.search import search_web
        search_text = await search_web(args.get("query", ""))
        logger.info("%s搜索结果长度: %d", "流式" if is_stream else "", len(search_text))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": search_text})

    elif name == "read_file":
        from backend.workspace import read_file
        file_content = read_file(args.get("path", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": file_content})

    elif name == "list_files":
        from backend.workspace import list_files
        listing = list_files(args.get("pattern", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": listing})

    elif name == "grep_files":
        from backend.workspace import grep_files
        matched = grep_files(args.get("pattern", ""), args.get("glob_pattern", "**/*.py"))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": matched})

    elif name == "write_file":
        from app.tools.workspace import write_file
        result = write_file(args.get("path", ""), args.get("content", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "edit_file":
        from app.tools.workspace import edit_file
        result = edit_file(args.get("path", ""), args.get("old_str", ""), args.get("new_str", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "bash":
        try:
            r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=30)
            result = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            result = "命令执行超时（30s）"
        except Exception as e:
            result = f"执行失败: {e}"
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "glob":
        import glob as _glob
        matches = _glob.glob(args.get("pattern", ""), root_dir=args.get("root", "."), recursive=True)
        result = "\n".join(matches) if matches else "未匹配到文件"
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
