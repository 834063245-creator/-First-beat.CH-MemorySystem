"""初痕环境诊断脚本 — 一键检查所有依赖是否就绪。

用法: python verify_env.py
退出码: 0 表示全部通过，非 0 表示有问题需要修复。
"""

import sys
import os
import socket

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"
exit_code = 0


def check(title: str, condition: bool, fix_hint: str = "") -> None:
    global exit_code
    if condition:
        print(f"  {PASS} {title}")
    else:
        print(f"  {FAIL} {title}")
        if fix_hint:
            print(f"     ↳ {fix_hint}")
        exit_code = 1


print("=" * 55)
print("  初痕 · 环境诊断")
print("=" * 55)

# ── 1. Python 版本 ──
print("\n[1] Python 版本")
py_ver = sys.version_info
check(f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
      py_ver >= (3, 11),
      "需要 Python 3.11+，请升级: https://python.org/downloads")

# ── 2. 依赖包 ──
print("\n[2] 核心依赖包")
deps = {
    "fastapi": "Web 框架",
    "uvicorn": "ASGI 服务器",
    "chromadb": "向量数据库",
    "httpx": "HTTP 客户端",
    "numpy": "数值计算",
    "jieba": "中文分词",
}
for pkg, desc in deps.items():
    try:
        __import__(pkg)
        check(f"{pkg} ({desc})", True)
    except ImportError:
        check(f"{pkg} ({desc})", False, f"pip install {pkg}")

# torch / transformers 为可选
for pkg in ("torch", "transformers"):
    try:
        __import__(pkg)
        check(f"{pkg}", True)
    except ImportError:
        print(f"  {WARN} {pkg} 未安装（轻量模式可用，embedding 需远程服务）")

# ── 3. Ollama 连接 ──
print("\n[3] Ollama 服务")
ollama_url = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
try:
    import httpx
    r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    if r.status_code == 200:
        check(f"Ollama 连接 ({ollama_url})", True)
        # 检查 bge-m3 模型
        models = r.json().get("models", [])
        model_names = [m["name"] for m in models]
        bge_m3 = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        has_model = any(n.startswith(bge_m3) for n in model_names)
        check(f"Embedding 模型: {bge_m3}", has_model,
              f"运行: ollama pull {bge_m3}")
        check(f"已安装模型: {len(models)} 个", True)
        for m in models[:5]:
            print(f"       {m['name']}")
    else:
        check(f"Ollama 响应异常 ({ollama_url})", False,
              f"HTTP {r.status_code}")
except Exception as e:
    check(f"Ollama 连接 ({ollama_url})", False,
          f"无法连接: {str(e)[:60]}。请确认 Ollama 已启动")

# ── 4. 端口检查 ──
print("\n[4] 端口检查")
port = 8082
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
    sock.close()
    check(f"端口 {port} 可用", True)
except OSError:
    sock.close()
    print(f"  {WARN} 端口 {port} 已被占用（可能已有实例运行）")

# ── 5. 数据目录 ──
print("\n[5] 数据目录")
data_dir = os.getenv("DATA_DIR", "./data")
try:
    os.makedirs(data_dir, exist_ok=True)
    test_file = os.path.join(data_dir, ".write_test")
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
    check(f"数据目录可写: {os.path.abspath(data_dir)}", True)
except Exception as e:
    check(f"数据目录不可写: {data_dir}", False, str(e)[:60])

# ── 6. 配置文件 ──
print("\n[6] 配置文件")
env_file = ".env"
if os.path.exists(env_file):
    check(f".env 存在", True)
else:
    if os.path.exists(".env.example"):
        check(f".env 不存在", False, "运行: copy .env.example .env  并编辑")
    else:
        check(f".env 和 .env.example 都不存在", False, "项目文件不完整")

# ── 总结 ──
print()
print("=" * 55)
if exit_code == 0:
    print("  [OK] 环境检查全部通过！运行: python run.py")
else:
    print(f"  [FAIL] 发现 {exit_code} 处问题，请按上方提示修复")
print("=" * 55)
sys.exit(exit_code)
