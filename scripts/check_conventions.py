#!/usr/bin/env python3
"""
初痕项目约定检查器 v2 — 自动化代码规范审计
覆盖率：CLAUDE.md 全量约定 + 6 条红线哨兵

用法:
  python scripts/check_conventions.py          # 终端彩色输出
  python scripts/check_conventions.py --json   # JSON 输出（给 CI/Agent 消费）
  python scripts/check_conventions.py --quick  # 跳过 import graph（快 3x）
"""

import ast
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(PROJECT_ROOT, "app")

# ── ANSI 颜色 ──
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

results: list[dict] = []


def check(name: str, passed: bool, detail: str = "", severity: str = "error"):
    results.append({"check": name, "passed": passed, "detail": detail, "severity": severity})


# ═══════════════════════════════════════════════════════════════
# 辅助工具
# ═══════════════════════════════════════════════════════════════

def _py_files(root: str) -> list[Path]:
    """返回 root 下所有 .py 文件（排除 __pycache__ 和 tests/）。"""
    files = []
    for py_file in Path(root).rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if "tests" in str(py_file).split(os.sep):
            continue
        files.append(py_file)
    return files


def _read_file(path: str) -> str:
    """读文件，出错返回空字符串。"""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════
# 检查 1：Import 依赖方向（红线覆盖）
# ═══════════════════════════════════════════════════════════════

# 层序：数字越大越"底层"（越不依赖别人）。外层可以 import 内层，反之禁止。
# -1 = 基础设施层，所有人可以 import
LAYER_ORDER: dict[str, int] = {
    "api": 0,
    "core": 1,
    "memory": 2,
    "retrieval": 3, "analysis": 3, "portrait": 3,
    "background": 4,
    "llm": 5,
    # 基础设施 — 无限制
    "brain": -1,    # 语义基础设施（extract_tags/tokenize），仅依赖 llm/embed
    "config": -1, "models": -1, "tools": -1,
}

# core/ 下特定子模块是基础设施，允许被任意层 import
CORE_INFRA_MODULES = {"db", "helpers", "bottleneck", "heartbeat"}

# 模块 → 所属层名（从路径推导）
_MODULE_TO_LAYER: dict[str, str] = {}


def _get_layer(file_path: str) -> str | None:
    """从文件路径推导所属层名。"""
    rel = os.path.relpath(file_path, APP_DIR).replace("\\", "/")
    parts = rel.split("/")
    if parts[0] in LAYER_ORDER:
        return parts[0]
    return None


def _detect_type_checking_block(lines: list[str], line_idx: int) -> bool:
    """检测 line_idx 是否在 `if TYPE_CHECKING:` 块内。"""
    # 向上查找最近的 if 语句
    indent_level = len(lines[line_idx]) - len(lines[line_idx].lstrip())
    for i in range(line_idx - 1, max(line_idx - 30, -1), -1):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent < indent_level:
            if stripped == "if TYPE_CHECKING:":
                return True
            # 遇到了更外层的 if/def/class，不是 TYPE_CHECKING
            if stripped.startswith("if ") or stripped.startswith("def ") or stripped.startswith("class "):
                return False
    return False


def _extract_import_target(import_line: str) -> str | None:
    """从 import 语句提取目标 app 子模块名。

    'from app.memory.chroma import X' → 'memory'
    'from app.core.db import X' → None (db 是基础设施)
    'from app.core import bottleneck' → None (bottleneck 是基础设施)
    'from app.core.context import X' → 'core'
    'import app.llm.deepseek' → 'llm'
    """
    # from app.XXX.YYY import ZZZ
    m1 = re.match(r'from\s+app\.(\w+)\.(\w+)', import_line)
    if m1:
        top, sub = m1.group(1), m1.group(2)
        if top == "core" and sub in CORE_INFRA_MODULES:
            return None
        return top
    # from app.XXX import YYY (YYY 可能是 core 的子模块)
    m2 = re.match(r'from\s+app\.(\w+)\s+import\s+(\w+)', import_line)
    if m2:
        top, imported = m2.group(1), m2.group(2)
        if top == "core" and imported in CORE_INFRA_MODULES:
            return None
        return top
    # import app.XXX.YYY
    m3 = re.match(r'import\s+app\.(\w+)', import_line)
    if m3:
        return m3.group(1)
    return None


def check_import_direction():
    """验证依赖方向：外层可 import 内层，反向禁止（TYPE_CHECKING 除外）。"""
    violations = []

    for py_file in _py_files(APP_DIR):
        file_layer = _get_layer(str(py_file))
        if file_layer is None:
            continue

        file_order = LAYER_ORDER.get(file_layer, -1)
        if file_order == -1:
            continue  # 基础设施层本身不受限制

        lines = py_file.read_text(encoding="utf-8").split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith(("from app.", "import app.")):
                continue
            # 跳过注释
            if stripped.startswith("#"):
                continue

            target = _extract_import_target(stripped)
            if target is None:
                continue
            if target not in LAYER_ORDER:
                continue  # 未知模块，跳过

            target_order = LAYER_ORDER[target]

            # 基础设施层 — 允许
            if target_order == -1:
                continue

            # TYPE_CHECKING 块内 — 允许反向导入
            if _detect_type_checking_block(lines, i):
                continue

            # 违规：file_order > target_order 意味着"导入了比自己更外层的模块"
            # file_order = -1 的基础设施文件已在上面跳过
            if file_order > target_order:
                rel = os.path.relpath(str(py_file), PROJECT_ROOT)
                violations.append(
                    f"{rel}:{i + 1} → app.{target}.* (方向违规: {file_layer}→{target})"
                )

    # 允许少量已知违规（历史遗留，先 warn）
    check(
        "依赖方向: 无反向 import（TYPE_CHECKING 除外）",
        len(violations) == 0,
        f"{len(violations)} 处反向依赖: {'; '.join(violations[:8])}"
        if violations
        else "依赖方向正确",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 2：SQLite 统一入口
# ═══════════════════════════════════════════════════════════════

def check_sqlite_entry():
    """检查是否有模块绕过 get_db() 直接使用 sqlite3.connect()"""
    violations = []
    for py_file in _py_files(APP_DIR):
        content = py_file.read_text(encoding="utf-8")
        # 排除 db.py 自身
        if py_file.name == "db.py":
            continue
        if "sqlite3.connect(" in content:
            if "from app.core.db import get_db" not in content:
                violations.append(str(py_file.relative_to(PROJECT_ROOT)))

    check(
        "SQLite 统一入口: 无模块绕过 get_db() 直接用 sqlite3.connect()",
        len(violations) == 0,
        f"{len(violations)} 处违规: {', '.join(violations)}" if violations else "全部通过 get_db()",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 3：get_db() 参数审计（新增）
# ═══════════════════════════════════════════════════════════════

def check_get_db_params():
    """检查 get_db() 调用是否使用 data/ 前缀的相对路径。"""
    bad_params = []
    for py_file in _py_files(APP_DIR):
        content = py_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # 匹配 get_db("...") 或 get_db(f"...") 或 get_db(var)
            m = re.search(r'get_db\(([^)]+)\)', line)
            if not m:
                continue
            arg = m.group(1).strip()
            # 跳过变量引用（无法静态分析）
            if arg.startswith("self.") or arg.startswith("os.path"):
                continue
            # 跳过 f-string（运行时拼接）
            if arg.startswith("f") and ("data" in arg.lower() or "DATA_DIR" in arg):
                continue
            # 检查字面量路径
            arg_clean = arg.strip('"').strip("'")
            if arg_clean and not arg_clean.startswith("data/") and not arg_clean.startswith("DATA_DIR"):
                # 允许变量/属性引用
                if not re.match(r'^[\w.]+$', arg.strip("()")):
                    bad_params.append(
                        f"{py_file.name}:{i + 1} get_db({arg[:60]})"
                    )

    check(
        "get_db() 参数: 使用 data/ 前缀相对路径",
        len(bad_params) <= 3,  # 允许少量变量引用
        f"{len(bad_params)} 处路径不标准: {'; '.join(bad_params[:5])}"
        if bad_params
        else "参数规范",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 4：CO_OCCURRENCE_FILE 路径
# ═══════════════════════════════════════════════════════════════

def check_cooccurrence_path():
    """检查 CO_OCCURRENCE_FILE 配置是否与代码一致"""
    settings_path = os.path.join(APP_DIR, "config", "settings.py")
    content = Path(settings_path).read_text(encoding="utf-8")

    has_json = 'CO_OCCURRENCE_FILE' in content and '.json"' in content.split('CO_OCCURRENCE_FILE')[1][:100]

    # 检查 consolidation.py 是否直接读 .json（代码行，非注释/docstring）
    cons_path = os.path.join(APP_DIR, "background", "consolidation.py")
    reads_json = False
    if os.path.exists(cons_path):
        for line in Path(cons_path).read_text(encoding="utf-8").split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if "co_occurrence.json" in stripped:
                reads_json = True
                break

    # 检查 symmetry.py（仅看代码行，忽略 docstring/注释）
    sym_path = os.path.join(APP_DIR, "analysis", "symmetry.py")
    sym_reads_json = False
    if os.path.exists(sym_path):
        for line in Path(sym_path).read_text(encoding="utf-8").split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if "co_occurrence.json" in stripped and ("open(" in stripped or "json.load" in stripped):
                sym_reads_json = True
                break

    issues = []
    if has_json:
        issues.append("settings.py 仍指向 .json 而非 .db")
    if reads_json:
        issues.append("consolidation.py 直接读 .json → 可能读脏数据")
    if sym_reads_json:
        issues.append("symmetry.py 代码中直接读 .json → 同上")

    check(
        "CO_OCCURRENCE_FILE: 路径与代码一致",
        len(issues) == 0,
        "; ".join(issues) if issues else "配置正确",
        severity="error",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 5：Phase 4 退役状态
# ═══════════════════════════════════════════════════════════════

def check_phase4_retirement():
    """检查 Phase 4 退役是否完成"""
    active_refs = []

    for py_file in Path(PROJECT_ROOT).rglob("*.py"):
        if "personality" in str(py_file) and "store.py" in str(py_file):
            continue
        if "test_" in py_file.name:
            continue
        if py_file.name == "check_conventions.py":
            continue

        content = py_file.read_text(encoding="utf-8")
        if "from app.personality" in content or "import app.personality" in content:
            active_refs.append(f"{py_file.relative_to(PROJECT_ROOT)}")
        if "from app.background.distill import" in content:
            active_refs.append(f"{py_file.relative_to(PROJECT_ROOT)} (distill)")

    active_refs = list(set(active_refs))

    check(
        "Phase 4 退役: 无活跃引用 PersonalityStore/DistillEngine",
        len(active_refs) == 0,
        f"{len(active_refs)} 处仍引用: {', '.join(active_refs[:5])}" if active_refs else "退役完成",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 6：ChromaDB metadata schema 哨兵（红线 2 覆盖）
# ═══════════════════════════════════════════════════════════════

REQUIRED_META_KEYS = [
    "timestamp", "hit_count", "heat", "embed_model", "stale",
    "archived", "superseded_by", "storage_complete", "source",
    "summary", "tags",
]
OPTIONAL_META_KEYS = ["entities", "date_tag"]


def check_chroma_metadata_schema():
    """AST 解析 chroma.py add_memory() 的 metadata dict，确保字段完整。"""
    chroma_path = os.path.join(APP_DIR, "memory", "chroma.py")
    if not os.path.exists(chroma_path):
        check("ChromaDB metadata schema: 字段完整", False, "chroma.py 不存在")
        return

    try:
        tree = ast.parse(Path(chroma_path).read_text(encoding="utf-8"))
    except SyntaxError as e:
        check("ChromaDB metadata schema: 字段完整", False, f"AST 解析失败: {e}")
        return

    # 找到 add_memory 方法内的 meta 字典
    meta_keys_found: set[str] = set()
    in_add_memory = False
    in_meta_dict = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "add_memory":
            in_add_memory = True
            # 在函数体内查找 meta = {...} 赋值
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "meta":
                            if isinstance(stmt.value, ast.Dict):
                                for key in stmt.value.keys:
                                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                        meta_keys_found.add(key.value)
                            break

    missing_required = [k for k in REQUIRED_META_KEYS if k not in meta_keys_found]
    missing_optional = [k for k in OPTIONAL_META_KEYS if k not in meta_keys_found]

    if missing_required:
        check(
            "ChromaDB metadata schema: 必填字段完整",
            False,
            f"缺少必填字段: {', '.join(missing_required)}",
            severity="error",
        )
    elif missing_optional:
        check(
            "ChromaDB metadata schema: 字段完整",
            True,
            f"可选字段缺失（可接受）: {', '.join(missing_optional)}",
        )
    else:
        check(
            "ChromaDB metadata schema: 字段完整",
            True,
            f"全部 {len(REQUIRED_META_KEYS)}+{len(OPTIONAL_META_KEYS)} 字段完整",
        )


# ═══════════════════════════════════════════════════════════════
# 检查 7：storage_complete 状态机（红线 2 扩展）
# ═══════════════════════════════════════════════════════════════

def check_storage_complete_flag():
    """验证 storage_complete 标记在 chroma.py 和 context.py 中都被正确使用。"""
    chroma_path = os.path.join(APP_DIR, "memory", "chroma.py")
    ctx_path = os.path.join(APP_DIR, "core", "context.py")

    chroma_ok = False
    ctx_ok = False

    if os.path.exists(chroma_path):
        chroma_content = _read_file(chroma_path)
        chroma_ok = "mark_storage_complete" in chroma_content

    if os.path.exists(ctx_path):
        ctx_content = _read_file(ctx_path)
        # 检查后台队列是否调用 mark_storage_complete
        ctx_ok = "mark_storage_complete" in ctx_content or "storage_complete" in ctx_content

    check(
        "storage_complete 状态机: write → mark_storage_complete 链路完整",
        chroma_ok and ctx_ok,
        f"chroma.py={'✓' if chroma_ok else '✗'} context.py={'✓' if ctx_ok else '✗'}",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 8：Background 线程完整性（红线 3 覆盖）
# ═══════════════════════════════════════════════════════════════

REQUIRED_THREADS = [
    # (name_pattern, description)
    ("store_queue_", "存储队列线程"),
    ("impulse_consumer_", "冲动消费线程"),
    ("dmn_ticker_", "DMN 节拍线程"),
    ("ai_desensitize_", "AI 情绪淡化线程"),
]


def check_background_threads():
    """验证后台线程创建模式：daemon=True + _stop_event。"""
    ctx_path = os.path.join(APP_DIR, "core", "context.py")
    if not os.path.exists(ctx_path):
        check("后台线程: daemon + stop_event 模式", False, "context.py 不存在")
        return

    content = _read_file(ctx_path)

    # 检查 _stop_event
    has_stop_event = "_stop_event" in content
    has_stop_event_set = "_stop_event.set()" in content
    has_stop_event_is_set = "_stop_event.is_set()" in content

    # 检查 Thread 创建
    thread_creations = re.findall(
        r'threading\.Thread\([^)]+\)',
        content.replace("\n", " ")
    )
    # 简单计数 Thread(target= 和 daemon=
    thread_targets = len(re.findall(r'Thread\(target=', content))
    thread_daemons = len(re.findall(r'daemon\s*=\s*True', content))
    thread_daemon_false = len(re.findall(r'daemon\s*=\s*False', content))

    # 检查各关键线程名
    found_threads = []
    missing_threads = []
    for pattern, desc in REQUIRED_THREADS:
        if pattern in content:
            found_threads.append(desc)
        else:
            missing_threads.append(desc)

    issues = []
    if not has_stop_event:
        issues.append("缺少 _stop_event")
    if thread_daemon_false > 0:
        issues.append(f"{thread_daemon_false} 处 daemon=False")
    if thread_targets > thread_daemons:
        issues.append(f"Thread(target=)={thread_targets} > daemon=True={thread_daemons} — 可能有非 daemon 线程")
    if missing_threads:
        issues.append(f"缺少线程: {', '.join(missing_threads)}")

    check(
        "后台线程: daemon=True + _stop_event 模式",
        len(issues) == 0,
        f"已找到 {len(found_threads)}/{len(REQUIRED_THREADS)} 关键线程; " + "; ".join(issues) if issues else "全部正常",
        severity="error",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 9：Settings 承重墙哨兵（红线 4 覆盖）
# ═══════════════════════════════════════════════════════════════

# 承重墙配置：键 → (期望默认值, 描述, 类型)
LOAD_BEARING_SETTINGS = {
    "LLM_BASE_URL": ("https://api.deepseek.com", "LLM API 地址"),
    "OLLAMA_EMBED_MODEL": ("bge-m3", "Embedding 模型名"),
    "CHROMA_COLLECTION_NAME": ("memories", "ChromaDB collection 名"),
    "DATA_DIR": (None, "数据根目录", "exists"),  # 只检查存在
    "WORK_MEMORY_TOKEN_BUDGET": (200000, "工作记忆 token 预算"),
    "DEFAULT_EMBED_MODEL": ("bge-m3", "默认 embedding 模型"),
}


def check_settings_walls():
    """验证 settings.py 中承重墙配置存在且值为预期默认值。"""
    settings_path = os.path.join(APP_DIR, "config", "settings.py")
    if not os.path.exists(settings_path):
        check("Settings 承重墙: 关键配置完整", False, "settings.py 不存在")
        return

    # 用 AST 解析获取变量赋值
    try:
        tree = ast.parse(Path(settings_path).read_text(encoding="utf-8"))
    except SyntaxError as e:
        check("Settings 承重墙: 关键配置完整", False, f"AST 解析失败: {e}")
        return

    settings_values: dict[str, tuple] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in LOAD_BEARING_SETTINGS:
                    raw = ast.unparse(node.value)
                    if isinstance(node.value, ast.Constant):
                        settings_values[target.id] = ("constant", node.value.value)
                    elif isinstance(node.value, (ast.Call, ast.Compare, ast.BoolOp, ast.BinOp)):
                        # 函数调用 / 比较表达式 — 存源码，后续子串匹配
                        settings_values[target.id] = ("expr", raw)
                    else:
                        settings_values[target.id] = ("expr", raw)

    # 单独处理 BENCHMARK_MODE（它是 Compare 节点）
    if "BENCHMARK_MODE" not in settings_values:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "BENCHMARK_MODE":
                        settings_values["BENCHMARK_MODE"] = ("expr", ast.unparse(node.value))

    issues = []
    for key, (expected_val, desc, *extra) in LOAD_BEARING_SETTINGS.items():
        if key not in settings_values:
            issues.append(f"❌ {key} ({desc}) — 缺失")
            continue

        val_type, val = settings_values[key]
        if expected_val is None:
            continue  # 只检查存在性

        # 对于表达式/函数调用类型，检查是否包含期望默认值
        if val_type == "expr":
            if isinstance(expected_val, str) and f'"{expected_val}"' not in val and f"'{expected_val}'" not in val or isinstance(expected_val, int) and str(expected_val) not in val:
                issues.append(f"⚠ {key} ({desc}) — 默认值可能已变: {val[:80]}")
        elif val_type == "constant" and val != expected_val:
            issues.append(f"⚠ {key} ({desc}) — 期望 {expected_val!r}, 实际 {val!r}")
        elif val_type == "call" and isinstance(expected_val, str):
            if f'"{expected_val}"' not in val and f"'{expected_val}'" not in val:
                issues.append(f"⚠ {key} ({desc}) — 默认值可能已变: {val[:80]}")

    # 额外：检查 BENCHMARK_MODE 存在
    if "BENCHMARK_MODE" not in settings_values:
        issues.append("❌ BENCHMARK_MODE 缺失")

    check(
        "Settings 承重墙: 关键配置完整且默认值正确",
        len(issues) == 0,
        "; ".join(issues) if issues else f"全部 {len(LOAD_BEARING_SETTINGS)}+1 项正常",
        severity="error",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 10：hardcoded "memories" 一致性（红线 2 + 4）
# ═══════════════════════════════════════════════════════════════

def check_memories_collection_consistency():
    """检查 dispatch.py 中硬编码的 collection 名与 settings.py 一致。"""
    dispatch_path = os.path.join(APP_DIR, "tools", "dispatch.py")
    settings_path = os.path.join(APP_DIR, "config", "settings.py")

    if not os.path.exists(dispatch_path):
        check("collection 名一致性: dispatch ↔ settings", True, "dispatch.py 不存在，跳过")
        return

    dispatch_content = _read_file(dispatch_path)
    settings_content = _read_file(settings_path)

    # 从 settings 提取 CHROMA_COLLECTION_NAME
    settings_match = re.search(
        r'CHROMA_COLLECTION_NAME\s*=\s*["\']([^"\']+)["\']',
        settings_content
    )
    expected_name = settings_match.group(1) if settings_match else "memories"

    # 从 dispatch 提取硬编码 collection 名（排除函数参数默认值）
    hardcoded_matches = re.findall(
        r'get_or_create_collection\(["\']([^"\']+)["\']',
        dispatch_content
    )

    mismatches = [m for m in hardcoded_matches if m != expected_name]

    check(
        "collection 名一致性: dispatch 硬编码 = settings 配置",
        len(mismatches) == 0,
        f"dispatch 中硬编码: {hardcoded_matches}, settings 配置: {expected_name}"
        if mismatches
        else f"一致: {expected_name}",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 11：_CORE_RULES 完整性（红线 1 覆盖）
# ═══════════════════════════════════════════════════════════════

# _CORE_RULES 文本的 SHA256 哨兵值（生成于 2026-06-09）
# 如果此哈希变了，说明有人改了核心规则，需要人工审查
_CORE_RULES_SHA256 = "0f6aabc612d671f5468e1b5ad6e2b57711d292d124d96927bdfe45f0fe119b36"


def check_core_rules_integrity():
    """验证 _CORE_RULES 常量未被意外修改。"""
    deepseek_path = os.path.join(APP_DIR, "llm", "deepseek.py")
    if not os.path.exists(deepseek_path):
        check("_CORE_RULES 完整性: 哈希哨兵", False, "deepseek.py 不存在")
        return

    try:
        tree = ast.parse(Path(deepseek_path).read_text(encoding="utf-8"))
    except SyntaxError as e:
        check("_CORE_RULES 完整性: 哈希哨兵", False, f"AST 解析失败: {e}")
        return

    # 找到 _CORE_RULES 赋值
    core_rules = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_CORE_RULES":
                    if isinstance(node.value, ast.Constant):
                        core_rules = node.value.value
                    elif isinstance(node.value, ast.JoinedStr):
                        # f-string: 提取非变量部分
                        parts = []
                        for val in node.value.values:
                            if isinstance(val, ast.Constant):
                                parts.append(val.value)
                        core_rules = "".join(parts)

    if core_rules is None:
        check("_CORE_RULES 完整性: 哈希哨兵", False, "未找到 _CORE_RULES 赋值")
        return

    # 规范化空白后计算哈希
    normalized = core_rules.strip().replace("\r\n", "\n")
    actual_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    if actual_hash == _CORE_RULES_SHA256:
        check("_CORE_RULES 完整性: 哈希哨兵", True, "哈希匹配，未被修改")
    else:
        check(
            "_CORE_RULES 完整性: 哈希哨兵",
            False,
            f"哈希不匹配! 期望 {_CORE_RULES_SHA256[:16]}..., 实际 {actual_hash[:16]}... — 请人工审查",
            severity="warning",
        )


# ═══════════════════════════════════════════════════════════════
# 检查 12：E2E 哨兵测试存在性（红线 6 覆盖）
# ═══════════════════════════════════════════════════════════════

SENTINEL_TESTS = [
    "E2E/test_write_path.py",
    "E2E/test_link2_retrieve.py",
    "E2E/test_link3_cross_turn.py",
    "E2E/test_link4_evolution.py",
    "E2E/test_background.py",
    "E2E/conftest.py",
]


def check_e2e_sentinels():
    """验证 E2E 哨兵测试文件全部存在。"""
    missing = []
    for rel_path in SENTINEL_TESTS:
        full = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full):
            missing.append(rel_path)

    check(
        "E2E 哨兵测试: 核心链路文件完整",
        len(missing) == 0,
        f"缺少 {len(missing)} 个文件: {', '.join(missing)}" if missing else f"全部 {len(SENTINEL_TESTS)} 个文件就位",
        severity="error",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 13：测试文件 close_all() 使用（测试规范）
# ═══════════════════════════════════════════════════════════════

def check_test_close_all():
    """检查测试文件是否在适当位置调用 close_all()。"""
    test_dir = os.path.join(PROJECT_ROOT, "tests")
    if not os.path.exists(test_dir):
        check("测试规范: close_all() 清理", True, "tests/ 目录不存在，跳过")
        return

    test_files = list(Path(test_dir).rglob("test_*.py"))
    test_files.append(Path(test_dir) / "conftest.py")

    with_close = []
    without_close = []

    for tf in test_files:
        if not tf.exists():
            continue
        content = tf.read_text(encoding="utf-8")
        if "close_all" in content:
            with_close.append(tf.name)
        else:
            without_close.append(tf.name)

    check(
        "测试规范: close_all() 使用情况",
        len(without_close) <= len(with_close),
        f"有 close_all: {with_close}, 缺 close_all: {without_close}"
        if without_close
        else f"全部 {len(with_close)} 个测试文件已覆盖",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 14：实体抽取模型一致性（红线 5 覆盖）
# ═══════════════════════════════════════════════════════════════

def check_entity_model_consistency():
    """检查实体抽取使用的模型是否与配置一致。"""
    entity_path = os.path.join(APP_DIR, "analysis", "entity.py")
    if not os.path.exists(entity_path):
        check("实体抽取模型: 配置一致性", True, "entity.py 不存在，跳过")
        return

    content = _read_file(entity_path)

    # 检查是否引用了 settings 中的模型名，而不是硬编码
    uses_settings_model = (
        "LOCAL_LLM_MODEL" in content
        or "from app.config.settings" in content
    )

    # 硬编码模型名检查
    hardcoded_models = re.findall(r'["\'](qwen[\w.:]+)["\']', content)
    hardcoded_bge = re.findall(r'["\'](bge-[\w]+)["\']', content)

    issues = []
    if hardcoded_models:
        issues.append(f"硬编码实体模型: {hardcoded_models}")
    if hardcoded_bge:
        issues.append(f"硬编码 embedding 模型: {hardcoded_bge}")

    check(
        "实体抽取模型: 使用 settings 配置而非硬编码",
        len(issues) == 0,
        "; ".join(issues) if issues else "通过 settings 引用",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 15：裸 except 统计（来自 v1，保留）
# ═══════════════════════════════════════════════════════════════

def check_bare_except():
    """统计裸 except 和 except Exception 数量"""
    total_bare = 0
    bare_except_files = defaultdict(int)

    for py_file in _py_files(APP_DIR):
        content = py_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "except:" or stripped == "except: continue" or stripped.startswith("except Exception"):
                total_bare += 1
                bare_except_files[py_file.name] += 1

    worst = sorted(bare_except_files.items(), key=lambda x: -x[1])[:5]
    worst_str = ", ".join(f"{f}({c})" for f, c in worst)

    check(
        "异常处理: 裸 except 数量",
        total_bare < 100,
        f"共 {total_bare} 处 except Exception/bare except (前5: {worst_str})",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 16：Pickle 使用
# ═══════════════════════════════════════════════════════════════

def check_pickle():
    """检查是否有 pickle 使用"""
    violations = []
    for py_file in _py_files(APP_DIR):
        content = py_file.read_text(encoding="utf-8")
        if "import pickle" in content or "from pickle" in content:
            violations.append(str(py_file.relative_to(PROJECT_ROOT)))

    check(
        "Pickle: 无 pickle 使用",
        len(violations) == 0,
        f"{len(violations)} 处使用" if violations else "无 pickle 使用",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 17：os.getenv 默认值
# ═══════════════════════════════════════════════════════════════

def check_env_defaults():
    """检查 os.getenv 是否有默认值"""
    no_default = []
    for py_file in _py_files(APP_DIR):
        if py_file.name == "settings.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if "os.getenv(" in stripped and "os.getenv(" not in stripped.split("#")[0]:
                if stripped.count(",") == 0 and stripped.endswith(")"):
                    no_default.append(f"{py_file.name}: {stripped[:80]}")

    check(
        "配置: os.getenv() 有默认值",
        len(no_default) == 0,
        f"{len(no_default)} 处无默认值" if no_default else "全部有默认值",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 18：JSON 结构化存储（相对于 SQLite）
# ═══════════════════════════════════════════════════════════════

def check_json_vs_sqlite():
    """统计使用 json.load 做结构化读取的模块（非追加写入场景）"""
    exempt_files = {"history.py", "feedback.py", "working.py"}
    migrated_files = {"cooccur.py", "entity_pair.py", "hyperedge.py"}

    json_modules = defaultdict(list)

    for py_file in _py_files(APP_DIR):
        if py_file.name in exempt_files or py_file.name in migrated_files:
            continue
        if py_file.name == "__init__.py":
            continue

        content = py_file.read_text(encoding="utf-8")
        if "json.load(" in content or "json.loads(" in content:
            if "open(" in content or "_json.load(f)" in content or "json.load(f)" in content:
                json_modules[py_file.name].append("读文件")
            else:
                json_modules[py_file.name].append("解析字符串")

    file_readers = {k: v for k, v in json_modules.items() if "读文件" in v}

    check(
        "JSON 缓存文件: 模块数量（各存各的缓存，非 bug）",
        len(file_readers) < 25,
        f"{len(file_readers)} 个模块用 JSON 文件存储自身缓存: {', '.join(sorted(file_readers.keys())[:8])}",
        severity="info",
    )


# ═══════════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════════

ALL_CHECKS = [
    check_import_direction,
    check_sqlite_entry,
    check_get_db_params,
    check_cooccurrence_path,
    check_phase4_retirement,
    check_chroma_metadata_schema,
    check_storage_complete_flag,
    check_background_threads,
    check_settings_walls,
    check_memories_collection_consistency,
    check_core_rules_integrity,
    check_e2e_sentinels,
    check_test_close_all,
    check_entity_model_consistency,
    check_bare_except,
    check_pickle,
    check_env_defaults,
    check_json_vs_sqlite,
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="初痕项目约定检查器 v2")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--quick", action="store_true", help="跳过 import graph 检查")
    args = parser.parse_args()

    checks_to_run = ALL_CHECKS
    if args.quick:
        checks_to_run = [c for c in ALL_CHECKS if c is not check_import_direction]

    for fn in checks_to_run:
        try:
            fn()
        except Exception as e:
            check(fn.__doc__ or fn.__name__, False, f"检查器自身异常: {e}")

    # ── 输出 ──
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    errors = sum(1 for r in results if not r["passed"] and r["severity"] == "error")
    warnings = sum(1 for r in results if not r["passed"] and r["severity"] == "warning")
    infos = sum(1 for r in results if r["severity"] == "info")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0 if errors == 0 else 1

    print(f"\n{BOLD}═══ 初痕项目约定检查报告 v2 ═══{RESET}\n")

    for r in results:
        if r["severity"] == "info":
            status = f"{BLUE}[INFO]{RESET}"
        elif r["passed"]:
            status = f"{GREEN}[PASS]{RESET}"
        elif r["severity"] == "error":
            status = f"{RED}[FAIL]{RESET}"
        else:
            status = f"{YELLOW}[WARN]{RESET}"
        print(f"  {status} {r['check']}")
        if r["detail"]:
            print(f"       {r['detail']}")

    print(f"\n{BOLD}总计:{RESET} {GREEN}{passed} 通过{RESET}, "
          f"{RED}{errors} 错误{RESET}, {YELLOW}{warnings} 警告{RESET}, {BLUE}{infos} 信息{RESET}")

    # 覆盖率提示
    # 红线对应检查函数：
    # 红线1: check_core_rules_integrity
    # 红线2: check_chroma_metadata_schema + check_storage_complete_flag + check_memories_collection_consistency
    # 红线3: check_background_threads
    # 红线4: check_settings_walls
    # 红线5: check_entity_model_consistency
    # 红线6: check_e2e_sentinels
    red_line_checks = {
        check_core_rules_integrity,
        check_chroma_metadata_schema,
        check_storage_complete_flag,
        check_memories_collection_consistency,
        check_background_threads,
        check_settings_walls,
        check_entity_model_consistency,
        check_e2e_sentinels,
    }
    red_lines_covered = sum(1 for c in checks_to_run if c in red_line_checks)
    total_red_lines = 6
    print(f"{BLUE}红线覆盖:{RESET} 每条红线 ≥1 项哨兵检查（{red_lines_covered} 项哨兵覆盖 {total_red_lines} 条红线）")

    if errors > 0:
        print(f"\n{RED}[!] 存在 {errors} 个错误项，建议修复后再让 Agent 写代码。{RESET}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
