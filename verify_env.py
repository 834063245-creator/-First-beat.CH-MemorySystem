"""初痕环境诊断脚本 — 一键检查所有依赖是否就绪。

用法: python verify_env.py
退出码: 0 表示全部通过，非 0 表示有问题需要修复。
"""

import sys
import os
import platform
import shutil
import socket

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"
exit_code = 0
warnings = 0


def check(title: str, condition: bool, fix_hint: str = "") -> None:
    global exit_code, warnings
    if condition:
        print(f"  {PASS} {title}")
    else:
        print(f"  {FAIL} {title}")
        if fix_hint:
            print(f"     -> {fix_hint}")
        exit_code = 1


def warn(title: str, hint: str = "") -> None:
    global warnings
    print(f"  {WARN} {title}")
    if hint:
        print(f"     -> {hint}")
    warnings += 1


# ── 头部 ──
print("=" * 60)
print("  初痕 · 环境诊断")
print("=" * 60)

# ── 1. 系统信息 ──
print("\n[1] 系统信息")
uname = platform.uname()
check(f"操作系统: {uname.system} {uname.release}", True)
check(f"架构: {uname.machine}", True)
check(f"Python: {sys.version.split()[0]}", True)
if sys.version_info < (3, 11):
    check("Python 版本 >= 3.11", False,
          "升级: https://python.org/downloads")
else:
    check(f"Python {sys.version_info.major}.{sys.version_info.minor} (>= 3.11)", True)

# 内存
try:
    import psutil
    mem = psutil.virtual_memory()
    mem_gb = mem.total / (1024**3)
    if mem_gb >= 4:
        check(f"内存: {mem_gb:.1f} GB", True)
    else:
        warn(f"内存仅 {mem_gb:.1f} GB (建议 4 GB+)")
    # 可用内存
    avail_gb = mem.available / (1024**3)
    if avail_gb < 1:
        warn(f"可用内存仅 {avail_gb:.1f} GB，可能影响 embedding 推理")
except ImportError:
    warn("psutil 未安装，无法检测内存")

# 磁盘空间
data_dir = os.getenv("DATA_DIR", "./data")
abs_data = os.path.abspath(data_dir)
try:
    usage = shutil.disk_usage(abs_data)
    free_gb = usage.free / (1024**3)
    if free_gb >= 2:
        check(f"磁盘可用: {free_gb:.1f} GB", True)
    else:
        warn(f"磁盘仅剩 {free_gb:.1f} GB (建议 2 GB+)")
except Exception:
    warn(f"无法检测磁盘空间: {abs_data}")

# ── 2. 项目文件完整性 ──
print("\n[2] 项目文件完整性")
required_files = [
    ("run.py", "启动入口"),
    ("requirements.txt", "完整依赖"),
    ("requirements-lite.txt", "轻量依赖"),
    ("verify_env.py", "本诊断脚本"),
    ("SETUP.md", "安装指南"),
    ("QUICKSTART.md", "快速上手"),
    ("README.md", "项目文档"),
    (".env.example", "配置模板"),
    ("app/core/circuit.py", "认知管线"),
    ("app/mcp/server.py", "MCP 服务"),
    ("app/llm/deepseek.py", "LLM 格式器"),
    ("app/llm/local.py", "本地 LLM"),
    ("app/tools/dispatch.py", "工具分发"),
    ("tests/conftest.py", "测试配置"),
]
for fpath, desc in required_files:
    check(f"{fpath} ({desc})", os.path.exists(fpath))

# ── 3. Python 依赖 ──
print("\n[3] Python 依赖")
core_deps = {
    "fastapi": ("Web 框架", "pip install fastapi"),
    "uvicorn": ("ASGI 服务器", "pip install uvicorn[standard]"),
    "chromadb": ("向量数据库", "pip install chromadb"),
    "httpx": ("HTTP 客户端", "pip install httpx"),
    "numpy": ("数值计算", "pip install numpy"),
    "jieba": ("中文分词", "pip install jieba"),
}
for pkg, (desc, install_cmd) in core_deps.items():
    try:
        __import__(pkg)
        check(f"{pkg} ({desc})", True)
    except ImportError:
        check(f"{pkg} ({desc})", False, install_cmd)

# 重型可选依赖
print("  可选依赖:")
for pkg, desc, hint in [
    ("torch", "PyTorch (本地 embedding)", "pip install torch 或使用 requirements-lite.txt"),
    ("transformers", "HuggingFace (模型加载)", "pip install transformers 或使用 requirements-lite.txt"),
]:
    try:
        __import__(pkg)
        check(f"  {pkg}", True)
    except ImportError:
        warn(f"  {pkg} 未安装", hint)

# ── 4. Ollama 服务 ──
print("\n[4] Ollama 服务")
ollama_url = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
ollama_ok = False
try:
    import httpx
    r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    if r.status_code == 200:
        ollama_ok = True
        check(f"Ollama 运行中 ({ollama_url})", True)
        models = r.json().get("models", [])
        model_names = [m["name"] for m in models]

        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        found = [n for n in model_names if n.startswith(embed_model)]
        if found:
            check(f"Embedding 模型: {embed_model}", True)
        else:
            check(f"Embedding 模型: {embed_model}", False,
                  f"ollama pull {embed_model}")

        if model_names:
            print(f"  已安装模型 ({len(models)} 个):")
            for m in models[:8]:
                size_mb = m.get("size", 0) / (1024**2)
                print(f"      {m['name']}  ({size_mb:.0f} MB)")
        else:
            warn("没有安装任何模型",
                 "ollama pull bge-m3")
    else:
        check(f"Ollama 响应异常 ({ollama_url})", False,
              f"HTTP {r.status_code}，请检查 Ollama 是否正常运行")
except ImportError:
    check(f"httpx 未安装", False, "pip install httpx")
except Exception as e:
    err = str(e)
    if "Connection refused" in err or "ConnectError" in err:
        check(f"Ollama 未启动 ({ollama_url})", False,
              "请打开 Ollama 应用或运行: ollama serve")
    elif "timed out" in err.lower() or "Timeout" in err:
        check(f"Ollama 连接超时 ({ollama_url})", False,
              "请确认 Ollama 服务正在运行且端口 11434 未被防火墙拦截")
    else:
        check(f"Ollama 连接失败 ({ollama_url})", False,
              f"{err[:80]}")

# ── 5. 端口检查 ──
print("\n[5] 端口检查")
port = 8082
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
    sock.close()
    check(f"端口 {port} 可用", True)
except OSError as e:
    sock.close()
    if "10048" in str(e) or "Address already in use" in str(e):
        warn(f"端口 {port} 已被占用", "可能已有初痕实例运行中")
    else:
        warn(f"端口 {port}: {e}")

# ── 6. 数据目录 ──
print("\n[6] 数据目录")
try:
    os.makedirs(data_dir, exist_ok=True)
    test_file = os.path.join(data_dir, ".write_test")
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
    check(f"数据目录可写: {os.path.abspath(data_dir)}", True)
except PermissionError:
    check(f"数据目录无写入权限: {data_dir}", False,
          "请修改目录权限或以管理员身份运行")
except Exception as e:
    check(f"数据目录异常: {data_dir}", False, str(e)[:60])

# ── 7. 配置文件 ──
print("\n[7] 配置文件")
env_file = ".env"
if os.path.exists(env_file):
    check(f".env 存在", True)
    # 检查关键配置
    with open(env_file, "r", encoding="utf-8") as f:
        content = f.read()
    key_checks = [
        ("OLLAMA_EMBED_MODEL", "Embedding 模型"),
        ("LOCAL_LLM_OLLAMA_URL", "Ollama 地址"),
    ]
    for key, desc in key_checks:
        if key in content:
            check(f"  配置项: {key}", True)
        else:
            warn(f"  缺少配置项: {key} ({desc})")
else:
    if os.path.exists(".env.example"):
        check(f".env 不存在", False,
              "运行: copy .env.example .env  然后编辑")
    else:
        check(f".env 和 .env.example 都不存在", False,
              "项目文件不完整，请重新 clone")

# ── 总结 ──
print()
print("=" * 60)
if exit_code == 0 and warnings == 0:
    print("  [OK] 全部通过！运行: python run.py")
elif exit_code == 0:
    print(f"  [OK] 环境就绪 (有 {warnings} 个提示)")
    print("  建议: 查看上方 [WARN] 项，通常不影响运行")
else:
    print(f"  [FAIL] {exit_code} 个错误, {warnings} 个警告")
    print("  请按上方提示修复后重试: python verify_env.py")
print("=" * 60)
sys.exit(exit_code)
