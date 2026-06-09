#!/usr/bin/env python3
"""
初痕项目约定检查器 v1 — 自动化代码规范审计
不需要读懂代码，跑一下看红绿即可。

用法:
  python scripts/check_conventions.py          # 终端彩色输出
  python scripts/check_conventions.py --json   # JSON 输出（给 CI/Agent 消费）
"""

import ast
import os
import sys
import json
from collections import defaultdict
from pathlib import Path

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(PROJECT_ROOT, "app")

# ── ANSI 颜色 ──
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

results: list[dict] = []


def check(name: str, passed: bool, detail: str = "", severity: str = "error"):
    results.append({"check": name, "passed": passed, "detail": detail, "severity": severity})


# ═══════════════════════════════════════════════════════════════
# 检查 1：SQLite 统一入口
# ═══════════════════════════════════════════════════════════════

def check_sqlite_entry():
    """检查是否有模块绕过 get_db() 直接使用 sqlite3.connect()"""
    violations = []
    for py_file in Path(APP_DIR).rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        # 排除 db.py 自身
        if py_file.name == "db.py":
            continue
        if "sqlite3.connect(" in content:
            # 检查是否已经 import get_db
            if "from app.core.db import get_db" not in content:
                violations.append(str(py_file.relative_to(PROJECT_ROOT)))

    check(
        "SQLite 统一入口: 无模块绕过 get_db() 直接用 sqlite3.connect()",
        len(violations) == 0,
        f"{len(violations)} 处违规: {', '.join(violations)}" if violations else "全部通过 get_db()",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 2：CO_OCCURRENCE_FILE 路径
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
            # 跳过注释和 docstring
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
# 检查 3：Phase 4 退役状态
# ═══════════════════════════════════════════════════════════════

def check_phase4_retirement():
    """检查 Phase 4 退役是否完成"""
    active_refs = []

    for py_file in Path(PROJECT_ROOT).rglob("*.py"):
        # 跳过废弃目录自身和测试
        if "personality" in str(py_file) and "store.py" in str(py_file):
            continue
        if "test_" in py_file.name:
            continue
        # 跳过约定检查器自身（字符串中可能包含模块名）
        if py_file.name == "check_conventions.py":
            continue

        content = py_file.read_text(encoding="utf-8")
        if "from app.personality" in content or "import app.personality" in content:
            active_refs.append(f"{py_file.relative_to(PROJECT_ROOT)}")
        if "from app.background.distill import" in content:
            active_refs.append(f"{py_file.relative_to(PROJECT_ROOT)} (distill)")

    # 去重
    active_refs = list(set(active_refs))

    check(
        "Phase 4 退役: 无活跃引用 PersonalityStore/DistillEngine",
        len(active_refs) == 0,
        f"{len(active_refs)} 处仍引用: {', '.join(active_refs[:5])}" if active_refs else "退役完成",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 4：裸 except 统计
# ═══════════════════════════════════════════════════════════════

def check_bare_except():
    """统计裸 except 和 except Exception 数量"""
    total_bare = 0
    bare_except_files = defaultdict(int)

    for py_file in Path(APP_DIR).rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "except:" or stripped == "except: continue":
                total_bare += 1
                bare_except_files[py_file.name] += 1
            elif stripped.startswith("except Exception"):
                total_bare += 1
                bare_except_files[py_file.name] += 1

    worst = sorted(bare_except_files.items(), key=lambda x: -x[1])[:5]
    worst_str = ", ".join(f"{f}({c})" for f, c in worst)

    check(
        "异常处理: 裸 except 数量",
        total_bare < 100,  # 阈值可调
        f"共 {total_bare} 处 except Exception/bare except (前5: {worst_str})",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 5：Pickle 使用
# ═══════════════════════════════════════════════════════════════

def check_pickle():
    """检查是否有 pickle 使用"""
    violations = []
    for py_file in Path(APP_DIR).rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if "import pickle" in content or "from pickle" in content:
            violations.append(str(py_file.relative_to(PROJECT_ROOT)))

    check(
        "Pickle: 无 pickle 使用",
        len(violations) == 0,
        f"{len(violations)} 处使用" if violations else "无 pickle 使用",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 6：os.getenv 默认值
# ═══════════════════════════════════════════════════════════════

def check_env_defaults():
    """检查 os.getenv 是否有默认值"""
    no_default = []
    for py_file in Path(APP_DIR).rglob("*.py"):
        if py_file.name == "settings.py":
            continue  # settings.py 是配置中心，有理由集中处理
        content = py_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if "os.getenv(" in stripped and not "os.getenv(" in stripped.split("#")[0]:
                # 简单检查：如果同一行没有第二个参数（默认值）
                if stripped.count(",") == 0 and stripped.endswith(")"):
                    no_default.append(f"{py_file.name}: {stripped[:80]}")

    check(
        "配置: os.getenv() 有默认值",
        len(no_default) == 0,
        f"{len(no_default)} 处无默认值" if no_default else "全部有默认值",
        severity="warning",
    )


# ═══════════════════════════════════════════════════════════════
# 检查 7：JSON 结构化存储（相对于 SQLite）
# ═══════════════════════════════════════════════════════════════

def check_json_vs_sqlite():
    """统计使用 json.load 做结构化读取的模块（非追加写入场景）"""
    # 这些是设计内的例外（追加写入日志）
    exempt_files = {"history.py", "feedback.py", "working.py"}
    # 这些是已迁移至 SQLite 的模块
    migrated_files = {"cooccur.py", "entity_pair.py", "hyperedge.py"}

    json_modules = defaultdict(list)

    for py_file in Path(APP_DIR).rglob("*.py"):
        if py_file.name in exempt_files or py_file.name in migrated_files:
            continue
        if py_file.name == "__init__.py":
            continue

        content = py_file.read_text(encoding="utf-8")
        if "json.load(" in content or "json.loads(" in content:
            # 区分：是在读文件还是在解析 API 响应/json 字段
            if "open(" in content or "_json.load(f)" in content or "json.load(f)" in content:
                json_modules[py_file.name].append("读文件")
            else:
                json_modules[py_file.name].append("解析字符串")

    # 只报告真正读文件的
    file_readers = {k: v for k, v in json_modules.items() if "读文件" in v}

    check(
        "JSON 缓存文件: 模块数量（各存各的缓存，非 bug）",
        len(file_readers) < 25,  # 信息性统计
        f"{len(file_readers)} 个模块用 JSON 文件存储自身缓存: {', '.join(sorted(file_readers.keys())[:8])}",
        severity="info",
    )


# ═══════════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════════

def main():
    checks = [
        check_sqlite_entry,
        check_cooccurrence_path,
        check_phase4_retirement,
        check_bare_except,
        check_pickle,
        check_env_defaults,
        check_json_vs_sqlite,
    ]

    for fn in checks:
        try:
            fn()
        except Exception as e:
            check(fn.__doc__ or fn.__name__, False, f"检查器自身异常: {e}")

    # ── 输出 ──
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    errors = sum(1 for r in results if not r["passed"] and r["severity"] == "error")
    warnings = sum(1 for r in results if not r["passed"] and r["severity"] == "warning")

    print(f"\n{BOLD}═══ 初痕项目约定检查报告 ═══{RESET}\n")

    for r in results:
        status = f"{GREEN}[PASS]{RESET}" if r["passed"] else (
            f"{RED}[FAIL]{RESET}" if r["severity"] == "error" else f"{YELLOW}[WARN]{RESET}"
        )
        print(f"  {status} {r['check']}")
        if r["detail"]:
            print(f"       {r['detail']}")

    print(f"\n{BOLD}总计:{RESET} {GREEN}{passed} 通过{RESET}, "
          f"{RED}{errors} 错误{RESET}, {YELLOW}{warnings} 警告{RESET}")

    if errors > 0:
        print(f"\n{RED}[!] 存在 {errors} 个错误项，建议修复后再让 Agent 写代码。{RESET}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
