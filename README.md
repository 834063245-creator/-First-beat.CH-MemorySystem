# 初痕 · First Beat — 认知型 AI 记忆引擎

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-214%20passed-green.svg)]()
[![MCP](https://img.shields.io/badge/MCP-10%20tools-orange.svg)]()
[English](README_EN.md)

👉 [快速安装？看这里](SETUP.md) | 🔧 [一键环境诊断](verify_env.py)

**不做文本生成。只做记忆。** 初痕是一个独立的认知引擎，通过 MCP 协议对任何 AI Agent 提供记忆服务。引擎负责检索、巩固、人格建模和认知决策，Agent 的 LLM 只管说话。

> 别人给 LLM 加记忆插件；初痕把 LLM 当作自己的语言皮层。

---

## 工作原理

```
  你的 AI Agent ─── MCP ──→ 初痕引擎 (localhost:8082)
      │                          │
      │ ── run_engine("用户说了什么") ──→
      │                          │  意图分析 → 多路检索 → 门控决策
      │                          │  人格匹配 → 冲动注入 → 情绪评估
      │                          │
      │ ←── 结构化认知上下文 ────  │
      │   {execute, memories,     │
      │    personality, impulses, │
      │    relationship, mood}    │
      │                          │
  LLM 生成回复                      │
      │                          │
      └── store_turn ──→ 入库 ────┘
```

---

## 快速启动

### 你需要

- **Python 3.11+**
- **Ollama** + bge-m3 模型

```bash
# 1. 安装 Ollama 并拉取 Embedding 模型
ollama pull bge-m3

# 2. 克隆 & 安装
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd chuchen

# 完整版（含 torch/transformers，适合纯本地运行）
pip install -r requirements.txt

# 轻量版（不含 torch/transformers，适合纯 MCP 服务模式）
# pip install -r requirements-lite.txt

# 3. 启动引擎
python run.py
# → 服务运行于 http://localhost:8082
```

### 验证

```bash
curl http://localhost:8082/health
# → {"status":"ok"}

# 或一键诊断所有环境依赖
python verify_env.py
```

> 安装遇到问题？查阅 [SETUP.md](SETUP.md) 详细排查指南。

---

## Docker 部署

### 一键启动（推荐）

```bash
docker compose up -d
# → Ollama (11434) + 初痕引擎 (8082) 同时启动
```

### 单独构建

```bash
docker build -t chuchen .
docker run -d -p 8082:8082 -v chuchen_data:/app/data chuchen
```

### docker-compose.yml

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama

  chuchen:
    build: .
    ports:
      - "8082:8082"
    environment:
      - LOCAL_LLM_OLLAMA_URL=http://ollama:11434
      - OLLAMA_EMBED_MODEL=bge-m3
    volumes:
      - chuchen_data:/app/data
    depends_on:
      ollama:
        condition: service_started
```

> 首次启动后需进入 Ollama 容器拉取模型：`docker exec chuchen-ollama ollama pull bge-m3`

---

## 10 个 MCP 工具

| 工具 | 输入 | 输出 |
|------|------|------|
| **`run_engine`** | 用户消息 | 完整认知上下文：意图、情绪、检索记忆、人格笔记、冲动信号、关系状态、执行指令 |
| **`store_turn`** | 用户消息 + AI 回复 | 对话入库确认 |
| **`query_memories`** | 查询文本 | 语义检索结果（含相关性、时间、情绪） |
| **`get_recent_history`** | N | 最近 N 轮对话 |
| **`get_memory_stats`** | — | 记忆总数、热度分布、情绪分布 |
| **`get_personality_tags`** | 来源(user/ai) | 人格标签列表 |
| **`get_topic_tree`** | — | 话题树结构 |
| **`get_relationship`** | — | 四维关系状态（熟悉度/信任度/亲密度/交互模式） |
| **`search_knowledge`** | 查询文本 | 知识库检索结果 |
| **`get_pattern_observations`** | — | 模式发现 + 自动调参记录 |

### `run_engine` 返回示例

```json
{
  "execute": {
    "tone": "warm",
    "formality": 0.3,
    "intimacy": 0.3,
    "response_mode": "question_first",
    "user_mood": "neutral",
    "user_intent": "emotional_sharing"
  },
  "personality": {
    "user": [{"content": "喜欢深入讨论技术", "hit_count": 12}],
    "ai":  [{"content": "回复风格偏理性分析", "hit_count": 8}]
  },
  "memories": [
    {"role": "fact", "summary": "用户正在开发AI记忆系统", "time_hint": "今天", "emotional_context": "用户情绪积极"}
  ],
  "impulses": ["关心一下用户关于「初痕项目」的进展"],
  "relationship": {
    "familiarity": 0.42,
    "trust": 0.68,
    "closeness": 0.35,
    "interaction_mode": "collaborator"
  }
}
```

---

## 接入 AI Agent

### 本地部署（推荐）

在 Agent 工作区创建 `.claude/mcp.json`：

```json
{
  "mcpServers": {
    "chuchen": {
      "url": "http://localhost:8082/mcp/jsonrpc"
    }
  }
}
```

### 远程部署

```json
{
  "mcpServers": {
    "chuchen": {
      "url": "https://your-server.com:8082/mcp/jsonrpc"
    }
  }
}
```

### 连接验证

启动 Agent 后尝试调用 `get_memory_stats`，如果能返回记忆统计信息说明连接成功。或直接 curl 测试：

```bash
curl -X POST http://localhost:8082/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_memory_stats","arguments":{}},"id":"1"}'
```

任何支持 MCP 的 Agent（Claude Code、Cursor 等）配置后即可直接调用全部 10 个工具。

---

## 架构

```
chuchen/
├── app/
│   ├── core/          # 认知管线：意图分析 · 门控决策 · 回路编排
│   ├── memory/        # ChromaDB 记忆库 + 工作记忆 + 共现/时间索引
│   ├── retrieval/     # 8路并行检索 + BM25/Embedding 两级精排
│   ├── background/    # 后台节律：4h/24h 巩固 · 5源冲动 · 蒸馏
│   ├── analysis/      # Russell 情绪环 · 实体提取 · 模式发现 · 人格对称性
│   ├── personality/   # 双人格系统（用户+AI 独立演化）
│   ├── mcp/           # MCP JSON-RPC 服务
│   ├── llm/           # 本地 embedding (bge-m3) + DeepSeek/本地 LLM 客户端
│   ├── api/           # REST 管理端点
│   ├── tools/         # 原子写入 · 工具分发
│   ├── brain/         # ChuchuCNN 自研字符级 CNN 模型 — 意图分类/情绪分析，500KB，<5ms
│   ├── config/        # 中央配置 · 环境变量默认值 · 路径工具
│   ├── models/        # Pydantic schemas（ChatRequest, ChatResponse 等）
│   └── knowledge/     # 知识库管理
├── backend/           # 旧模块桥接层（逐步迁移至 app/）
├── tests/             # 214+ 单测，本地可全部通过
├── scripts/           # 审计套件 · 报告对比
├── Dockerfile         # 容器化构建
├── docker-compose.yml # Ollama + 引擎一键部署
└── run.py             # 启动入口
```

---

## 审计套件

8 类审计覆盖语义检索、关键词、时间检索、排序、纠正反馈、人格一致性、工作记忆、时间节律。**每次修改检索逻辑后跑一次，确保分数不倒退。**

```bash
python scripts/audit.py           # 跑全部 8 类
python scripts/audit.py --quick   # 快速模式
```

对比两次审计的分数变化：

```bash
python scripts/compare_reports.py audit/report_before.json audit/report_after.json
```

审计报告保存在 `audit/` 目录（已加入 .gitignore）。

---

## 设计哲学

| # | 原则 | 含义 |
|---|------|------|
| 1 | **全文不加工** | 原文永不压缩。摘要和 Embedding 是翻译，不是加工 |
| 2 | **时间即骨架** | 时间参与组织、关联、浮现，不是衰减因子 |
| 3 | **行为即权重** | hit_count 决定权重，无时间衰减函数 |
| 4 | **引擎自有节律** | 巩固/冲动/蒸馏/模式发现独立运行，不依赖用户在线 |
| 5 | **引擎决策 → LLM 执行** | LLM 不拥有记忆、不调检索工具，只按引擎指令说话 |

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `OLLAMA_EMBED_MODEL` | 是 | Embedding 模型名，默认 `bge-m3` |
| `OLLAMA_NUM_THREADS` | 否 | CPU 线程数，默认 `4` |
| `LOCAL_LLM_OLLAMA_URL` | 是 | Ollama 地址，默认 `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key，用于工作记忆摘要增强 |
| `DEEPSEEK_BASE_URL` | 否 | DeepSeek API 地址，默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 否 | DeepSeek 模型名，默认 `deepseek-chat` |
| `LOCAL_LLM_ENABLED` | 否 | 启用本地 LLM（Ollama 摘要生成），默认 `false` |
| `LOCAL_LLM_MODEL` | 否 | 本地 LLM 模型名，默认 `qwen2.5:7b` |
| `LOCAL_LLM_TIMEOUT` | 否 | 本地 LLM 超时秒数，默认 `30` |
| `BOCHA_API_KEY` | 否 | 博查搜索 API Key，用于联网搜索 |
| `DATA_DIR` | 否 | 数据目录，默认 `./data` |
| `DEPLOY_MODE` | 否 | `full` / `lite` |
| `USERS` | 否 | 多用户认证（JSON 格式），如 `{"admin":"changeme"}` |
| `DEBUG_INCLUDE_PROMPT` | 否 | Debug 时是否包含 Prompt，默认 `false` |
| `IMPULSE_ACTIVE_PATH_B` | 否 | 冲动系统开关（主动开口），默认 `true` |
| `OLLAMA_MODELS` | 否 | Docker 专用：Ollama 模型挂载路径 |

详见 `.env.example`。

---

## 贡献

欢迎提 Issue 和 PR。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

当前版本采用 [MIT License](LICENSE)。后续版本及衍生作品可能采用不同许可，作者保留权利。

---

[📝 作者的话](AUTHOR.md)
