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

pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY（引擎需要 LLM 才能说话）
```

不填 Key 引擎也能启动——记忆、巩固、冲动、人格建模等后台功能照常运行，只是不会生成回复。

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

## 引擎目录结构

启动后会在 `./data/` 下自动创建以下文件：

```
data/
├── chroma/              # ChromaDB 向量库（记忆存储）
├── chat_history.jsonl   # 对话记录
├── working_memory.json  # 工作记忆摘要
├── impulse_state.json   # 冲动系统状态
├── dmn_state.json       # 巩固状态
├── topic_tree.json      # 话题树
├── co_occurrence.json   # 共现矩阵
├── pattern_cache.json   # 模式发现缓存（cache/ 目录）
├── personality_chroma/  # 人格标签库
├── behavior_chroma/     # 行为模式库
└── ai_chroma/           # AI 表达记忆库
```

---

## 常见问题排查

| 症状 | 可能原因 | 解决方法 |
|------|----------|----------|
| `Connection refused` 访问 Ollama | Ollama 服务未启动 | 运行 `ollama serve` 或启动 Ollama 应用 |
| `model 'bge-m3' not found` | 模型未下载 | `ollama pull bge-m3` |
| `ModuleNotFoundError` | 依赖未安装 | `pip install -r requirements.txt` |
| 端口 8082 被占用 | 已有服务运行 | 关闭旧进程：`netstat -ano \| findstr :8082` |
| `LLM_API_KEY` 401 | Key 无效或未配置 | 检查 `.env` 文件中的 `LLM_API_KEY` |
| 引擎不说话 | 没配置 API Key | 编辑 `.env`，填入有效的 LLM_API_KEY |
| 导入 `app.core.circuit` 失败 | 不在项目根目录 | 确保从项目根目录运行 `python run.py` |
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

- [README](README.md) — 架构概述和 API 文档
- [QUICKSTART.md](QUICKSTART.md) — 3 分钟快速上手
- [CONTRIBUTING.md](CONTRIBUTING.md) — 贡献指南
- [AUTHOR.md](AUTHOR.md) — 作者的故事

还是跑不起来？[提个 Issue](https://github.com/834063245-creator/-First-beat.CH-MemorySystem/issues/new)。
