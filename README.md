# 初痕 · First Beat — 有自主节律的认知记忆引擎

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-258%20passed-green.svg)]()
[![MCP](https://img.shields.io/badge/MCP-10%20tools-orange.svg)]()
[English](README_EN.md)

👉 [快速安装](SETUP.md) | 🔧 [环境诊断](verify_env.py)

---

**别人给 LLM 加记忆插件；初痕让 LLM 当自己的语言皮层。**

不做文本生成。只做记忆。初痕是一个独立的认知引擎，通过 MCP 协议对任何 AI Agent 提供记忆服务。引擎负责检索、巩固、人格建模和认知决策，Agent 的 LLM 只负责说话。

---

## 和别的方案有什么不同

| | Mem0 | MemGPT / Letta | LangGraph | **初痕** |
|---|---|---|---|---|
| **中文文档** | ❌ | ❌ | ❌ | **✅ 中英双语** |
| **中文 README** | ❌ | ❌ | ❌ | **✅ 中英双版** |
| **中文分词/Tokenizer** | ❌ 依赖英文 spaCy | ❌ 无原生分词 | ❌ 无 | **✅ 汉字级 ChuchuTok** |
| **自有中文小模型** | ❌ 全调 GPT | ❌ 全调 LLM | ❌ 无 | **✅ 6 个 ChuchuCNN 中英双语, 500KB/个** |
| **离线可用** | ❌ 必须调 API | ❌ 必须调 API | ❌ 必须调 LLM | **✅ Ollama 可选，模型本地跑** |
| **架构** | LLM 提取事实 → 存向量库 | LLM 管理自身记忆 | 状态机编排 | **引擎决策 → LLM 执行** |
| **记忆检索** | 语义 + BM25 + 实体 | 语义 + 自编辑窗口 | (框架不提供) | **8 路并行检索 + 两级精排** |
| **人格建模** | ❌ | ❌ | ❌ | **用户 + AI 双人格独立演化** |
| **自主节律** | ❌ | ❌ | ❌ | **5 源冲动系统，引擎会主动开口** |
| **情绪分析** | ❌ | ❌ | ❌ | **Russell 二维情绪环 + ChuchuCNN** |
| **模式发现** | ❌ | ❌ | ❌ | **多时间尺度模式 + 自动调参** |
| **后台巩固** | ❌ | ❌ | ❌ | **4h/24h 双周期 + 蒸馏** |
| **事实时序推理** | ❌ | ❌ | ❌ | **✅ 双路径冲突检测 (情绪翻转 + 语义位移)** |
| **MCP 协议** | ❌ | ❌ | ❌ | **✅ 原生 MCP Server** |
| **部署方式** | 托管服务 / 自建 | 托管 API | Python 库 | **pip install → python run.py** |
| **零 API Key 启动** | ❌ | ❌ | ❌ | **✅ Clone 即跑** |
| **自研模型 (非调 LLM)** | ❌ | ❌ | ❌ | **✅ 6 分类器，3.5MB 总大小** |

核心差异就一句话：别人的记忆系统是 LLM 的被动工具，初痕是**有自己节律的独立器官**。引擎在后台自己跑巩固、蒸馏、冲动，不需要等用户发消息。

---

### 为什么没有 LongMemEval / LoCoMo 跑分

你可能注意到 Mem0 等系统会晒 LongMemEval、LoCoMo 等 benchmark 分数。初痕没有，原因很简单：

**这些 benchmark 测的是"事实召回"，不是"认知引擎"。**

它们的测试方式是：冷注入大量事实 → 问问题 → 测召回率。这本质上是在测一个**键值数据库的检索精度**，不是在测一个**有自主节律的认知引擎**的能力。

初痕的设计目标不是"存得多、查得准"，而是：
- 在对话中自然积累认知（不是冷注入）
- 理解用户的人格和情绪变化
- 在后台自主巩固、蒸馏、发现模式
- 在合适的时机主动开口

这些能力无法用"你问一条事实，我答一条事实"的 benchmark 来度量。

更直接地说：如果初痕的目标只是在 LongMemEval 上跑高分，我完全可以写一个专门的检索器去做这件事。但那就不是初痕了——它只是一个向量数据库，不是我想要的那个"认识用户"的东西。

事实上我已经做过一个了。早期有个叫 **Jarvis** 的实验版（现在躺在 D 盘里），SQLite + FAISS 向量检索 + LLM 提取事实，能做冷注入、事实召回、知识图谱，跑 benchmark 应该能拿个不错的分数。但它本质上还是"存进去、查出来"，没有自主节律，没有人格建模，没有冲动系统。它记性很好，但它不认识你。

初痕是在否定了 Jarvis 那条路之后从头做的。

**如果你有兴趣帮着跑 benchmark，非常欢迎。** 引擎的 MCP 接口是标准的，外部评测工具可以直接调用。提 Issue 或 PR 都可以，我会全力配合。先谢过了。

> 仓库里包含了我自己用的审计套件（`scripts/audit.py`），覆盖语义检索、关键词、时间检索、排序等 8 类场景的回归测试。虽然不是 LongMemEval 那样的行业标准，但对整个系统的功能完整性做了全面验证，可作为参考。

---

## 工作原理

```
 你的 AI Agent ─── MCP ──→ 初痕引擎 (localhost:8082)
     │                          │
     │ ── run_engine("用户说了什么") ──→
     │                          │  ① 意图/情绪分析
     │                          │  ② 8 路并行检索
     │                          │  ③ 认知状态分层 (fact/reference/background)
     │                          │  ④ 门控决策 (压抑不合适的冲动)
     │                          │
     │ ←── 结构化认知上下文 ────  │
     │   {execute, memories,     │
     │    personality, impulses, │
     │    relationship, mood}    │
     │                          │
 LLM 生成回复                      │
     │                          │
     └── store_turn ──→ 入库 ────┘

 后台自主运行（不依赖用户在线）：
   巩固 4h/24h · 冲动 5源泊松 · 蒸馏 · 模式发现
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

pip install -r requirements.txt      # 完整版
# pip install -r requirements-lite.txt  # 轻量版

# 3. 启动引擎
python run.py
# → 服务运行于 http://localhost:8082
```

### 验证

```bash
curl http://localhost:8082/health          # → {"status":"ok"}
python verify_env.py                        # 一键诊断
```

> 安装遇到问题？查阅 [SETUP.md](SETUP.md)。

---

## 10 个 MCP 工具

| 工具 | 输入 | 输出 |
|------|------|------|
| **`run_engine`** | 用户消息 | 意图/情绪/记忆/人格/冲动/关系/执行指令 |
| **`store_turn`** | 用户消息 + AI 回复 | 入库确认 |
| **`query_memories`** | 查询文本 | 语义检索结果（含相关性、时间、情绪） |
| **`get_recent_history`** | N | 最近 N 轮对话 |
| **`get_memory_stats`** | — | 记忆总数、热度分布、情绪分布 |
| **`get_personality_tags`** | source (user/ai) | 人格标签列表 |
| **`get_topic_tree`** | — | 话题树结构 |
| **`get_relationship`** | — | 熟悉度/信任度/亲密度/交互模式 |
| **`search_knowledge`** | 查询文本 | 知识库检索 |
| **`get_pattern_observations`** | — | 模式发现 + 自动调参记录 |

### `run_engine` 返回示例

```json
{
  "execute": {
    "tone": "caring",
    "formality": 0.1,
    "response_mode": "soothe",
    "user_mood": "negative",
    "user_intent": "emotional_sharing"
  },
  "memories": [
    {"role": "fact",   "summary": "用户最近压力大", "time_hint": "今天", "emotional_context": "用户情绪低落"},
    {"role": "reference", "summary": "用户上周提过项目deadline", "time_hint": "上周"}
  ],
  "personality": {
    "user": [{"content": "深夜容易情绪波动", "hit_count": 8}],
    "ai":   [{"content": "偏好先共情再给建议", "hit_count": 12}]
  },
  "impulses": ["你心里好像想起了什么——关于初痕项目的事"],
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

在 Agent 工作区创建 `.claude/mcp.json`（Claude Code），或任意支持 MCP 的客户端：

```json
{
  "mcpServers": {
    "chuchen": {
      "url": "http://localhost:8082/mcp/jsonrpc"
    }
  }
}
```

远程部署换成 `https://your-server.com:8082/mcp/jsonrpc`。

验证连接：

```bash
curl -X POST http://localhost:8082/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_memory_stats","arguments":{}},"id":"1"}'
```

---

## Docker

```bash
docker compose up -d   # Ollama + 引擎一键启动
```

首次启动后拉模型：`docker exec chuchen-ollama ollama pull bge-m3`

---

## 架构

```
app/
├── core/          # 认知管线：意图分析 · 门控决策 · 回路编排
├── memory/        # ChromaDB 记忆库 + 工作记忆 + 倒排/共现/时间索引
├── retrieval/     # 8 路并行检索 + BM25/Embedding 两级精排
├── background/    # 后台节律：4h/24h 巩固 · 5 源冲动 · 蒸馏
├── analysis/      # Russell 情绪环 · 实体提取 · 模式发现 · 人格对称性
├── personality/   # 双人格系统（用户 + AI 独立演化）
├── mcp/           # MCP JSON-RPC 服务
├── llm/           # 本地 embedding (bge-m3) + DeepSeek/本地 LLM
├── api/           # REST 管理端点
├── tools/         # 原子写入 · 工具分发
├── brain/         # 6 个 ChuchuCNN 自研字符级 CNN (共 3.5MB)
│   ├── model_intent/     # 意图分类 (7类，500KB)
│   ├── model_emotion/    # 情绪分类 (5类，500KB)
│   ├── model_urgency/    # 紧急度三分类 (500KB)
│   ├── model_negation/   # 否定检测二分类 (500KB)
│   ├── model_topic/      # 话题分类 (50类，567KB)
│   └── model_fact/       # 事实域判断 (二分类，494KB)
├── config/        # 中央配置
├── models/        # Pydantic schemas
└── knowledge/     # 知识库管理

backend/           # 旧模块桥接层（逐步迁移至 app/）
tests/             # 320+ 测试，5 层覆盖
```

---

## 设计哲学

| # | 原则 | 含义 |
|---|------|------|
| 1 | **原文不加工** | 原文永不压缩。摘要和 Embedding 是翻译，不是加工 |
| 2 | **时间即骨架** | 时间参与组织、关联、浮现，不是衰减因子 |
| 3 | **行为即权重** | hit_count 决定权重，无时间衰减函数 |
| 4 | **引擎自有节律** | 巩固/冲动/蒸馏/模式发现独立运行，不依赖用户在线 |
| 5 | **引擎决策 → LLM 执行** | LLM 不拥有记忆、不调检索工具，只按引擎指令说话 |

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `OLLAMA_EMBED_MODEL` | 是 | Embedding 模型名，默认 `bge-m3` |
| `LOCAL_LLM_OLLAMA_URL` | 是 | Ollama 地址，默认 `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 否 | DeepSeek API 地址 |
| `LOCAL_LLM_ENABLED` | 否 | 启用本地 LLM，默认 `true` |
| `LOCAL_LLM_MODEL` | 否 | 本地 LLM 模型名，默认 `qwen2.5:7b` |
| `BOCHA_API_KEY` | 否 | 博查搜索 API Key |
| `DATA_DIR` | 否 | 数据目录，默认 `./data` |
| `DEPLOY_MODE` | 否 | `full` / `lite` |
| `USERS` | 否 | 多用户认证 JSON |
| `IMPULSE_ACTIVE_PATH_B` | 否 | 冲动系统开关，默认 `true` |

详见 `.env.example`。

---

## 贡献

欢迎提 Issue 和 PR。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

[MIT License](LICENSE)。

---

[📝 作者的话](AUTHOR.md)
