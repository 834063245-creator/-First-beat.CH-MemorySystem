"""工作区文件操作工具 — read_file / write_file / edit_file / list_files / grep_files。"""
import logging
import os

logger = logging.getLogger(__name__)

# 项目根目录，用于解析相对路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_path(path: str) -> str:
    """将相对路径转为绝对路径（基于项目根目录）。"""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def read_file(path: str) -> str:
    """读取文件内容。"""
    resolved = _resolve_path(path)
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"文件不存在: {path}"
    except PermissionError:
        return f"没有读取权限: {path}"
    except UnicodeDecodeError:
        try:
            with open(resolved, "r", encoding="gbk") as f:
                return f.read()
        except Exception as exc:
            return f"文件编码错误: {exc}"
    except Exception as exc:
        return f"读取失败: {exc}"


def write_file(path: str, content: str) -> str:
    """写入或覆写文件。"""
    resolved = _resolve_path(path)
    try:
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入: {path} ({len(content)} 字符)"
    except Exception as exc:
        return f"写入失败: {exc}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """精确字符串替换——在文件中找到 old_str 并替换为 new_str。"""
    resolved = _resolve_path(path)
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"文件不存在: {path}"
    except Exception as exc:
        return f"读取失败: {exc}"

    if old_str not in content:
        return f"未找到要替换的文本（{old_str[:60]}...），文件未修改。"

    if content.count(old_str) > 1:
        return (
            f"要替换的文本在文件中出现了 {content.count(old_str)} 次，"
            f"请提供更精确的 old_str 以确保唯一匹配。"
        )

    new_content = content.replace(old_str, new_str, 1)
    try:
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"已修改: {path}"
    except Exception as exc:
        return f"写入失败: {exc}"


def list_files(pattern: str) -> str:
    """按 glob 模式列出文件。"""
    import glob as _glob
    try:
        matches = _glob.glob(pattern, root_dir=_PROJECT_ROOT, recursive=True)
        if not matches:
            return f"未匹配到文件: {pattern}"
        return "\n".join(sorted(matches)[:200])
    except Exception as exc:
        return f"列出文件失败: {exc}"


def grep_files(pattern: str, glob_pattern: str = "**/*.py") -> str:
    """在项目文件中搜索文本或正则表达式。"""
    import glob as _glob
    import re

    try:
        files = _glob.glob(glob_pattern, root_dir=_PROJECT_ROOT, recursive=True)
    except Exception as exc:
        return f"glob 匹配失败: {exc}"

    if not files:
        return f"未匹配到文件: {glob_pattern}"

    results = []
    max_results = 50
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"正则表达式错误: {exc}"

    for fpath in sorted(files):
        if len(results) >= max_results:
            break
        full = os.path.join(_PROJECT_ROOT, fpath)
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append(f"{fpath}:{lineno}: {line.strip()[:200]}")
                        if len(results) >= max_results:
                            break
        except Exception:
            continue

    if not results:
        return f"在 {len(files)} 个文件中未找到匹配「{pattern}」的内容。"

    return "\n".join(results)
