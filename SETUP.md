# 初痕 · 安装与排查指南

## 系统要求

| 项目 | 最低要求 | 推荐 |
|------|----------|------|
| Python | 3.11+ | 3.12+ |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 2 GB（含 bge-m3 模型 ~1.2GB） | 5 GB+ |
| GPU | 不需要（CPU 推理） | NVIDIA GPU + CUDA |
| 操作系统 | Windows / macOS / Linux | — |

## 快速安装

### 1. 安装 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: 下载安装包
# https://ollama.com/download/windows
```

### 2. 拉取 Embedding 模型

```bash
ollama pull bge-m3
```

验证：

```bash
ollama list
# 应显示: bge-m3:latest
```

### 3. 安装 Python 依赖

```bash
# 克隆项目
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

# 完整安装（含本地 embedding）
pip install -r requirements.txt

# 或轻量安装（远程 embedding / 仅 MCP 服务）
pip install -r requirements-lite.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，确认 OLLAMA_EMBED_MODEL 和 LOCAL_LLM_OLLAMA_URL 正确
```

### 5. 启动

```bash
python run.py
# 服务运行于 http://localhost:8082
```

### 6. 验证

```bash
curl http://localhost:8082/health
# → {"status":"ok"}
```

或运行环境诊断脚本：

```bash
python verify_env.py
```

---

## 常见问题排查

| 症状 | 可能原因 | 解决方法 |
|------|----------|----------|
| `Connection refused` 访问 Ollama | Ollama 服务未启动 | 运行 `ollama serve` 或启动 Ollama 应用 |
| `model 'bge-m3' not found` | 模型未下载 | `ollama pull bge-m3` |
| `ModuleNotFoundError: torch` | 完整依赖未安装 | `pip install torch` 或使用 `requirements-lite.txt` |
| 端口 8082 被占用 | 已有服务运行 | 关闭旧进程：`netstat -ano | findstr :8082` |
| `DEEPSEEK_API_KEY` 401 | Key 无效 | 不带 DeepSeek 也能运行，摘要退化为关键词截断 |
| 导入 `app.core.circuit` 失败 | 项目根目录不在 Python 路径 | 确保从项目根目录运行 `python run.py` |
| ChromaDB 写入失败 | 磁盘空间不足或权限问题 | 检查 `data/` 目录权限，确保有 500MB+ 可用空间 |

### Ollama 连接测试

```bash
curl http://localhost:11434/api/tags
# 正常应返回模型列表 JSON
```

### Python 版本检查

```bash
python --version
# 应显示 Python 3.11.x 或更高
```

---

## Docker 部署（可选）

```bash
docker compose up -d
# Ollama + 初痕引擎一键启动
# 访问: http://localhost:8082
```

首次启动时 Docker 内的 Ollama 需要手动拉取模型：

```bash
docker exec -it chuchen-ollama ollama pull bge-m3
```

---

## 下一步

- [README](README.md) — 架构概述和 MCP 工具文档
- [CONTRIBUTING.md](CONTRIBUTING.md) — 贡献指南
- [AUTHOR.md](AUTHOR.md) — 作者的故事

还是跑不起来？[提个 Issue](https://github.com/834063245-creator/-First-beat.CH-MemorySystem/issues/new)。
