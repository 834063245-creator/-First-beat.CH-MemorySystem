"""[SHIM] 工作区工具 — 桥接到 app/tools/workspace.py。"""
from app.tools.workspace import (  # noqa: F401
    read_file,
    write_file,
    edit_file,
    list_files,
    grep_files,
)
