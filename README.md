# 初痕 · First Beat — 自循环记忆体

> 初痕是一个自闭环的认知系统。引擎做决策，LLM 当它的嘴。填了 API Key 就自己活。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-258%20passed-green.svg)]()
[English](README_EN.md)

👉 [快速上手](QUICKSTART.md) | 🔧 [安装排查](SETUP.md) | [环境诊断](verify_env.py)

---

**别人给 LLM 加记忆插件；初痕让 LLM 当自己的语言皮层。**

初痕提供的是一个自循环的记忆基础设施。引擎自己在后台跑巩固、冲动、蒸馏、模式发现——然后在合适的时机，通过 LLM 自然地开口说话。上面想搭什么——聊天应用、桌宠、陪伴型 Agent——是你的事。初痕只管记忆和说话。

---

## 不是记忆插件，是一个记忆体

**大多数 AI 记忆项目（Mem0、Zep、Letta、MemOS、Cognee 等）做的是同一类事**：提供一套 SDK 或 API，让开发者把"记忆能力"嵌入到自己的 Agent 里。它们在各自的领域都做得很好——Mem0 每天处理上亿次调用，Zep 的时序图谱在 benchmark 上拿了第一，Letta 的 self-editing memory 是一个精巧的设计。

**初痕选了一条不同的方向**。它不提供 SDK、不开 API、不嵌到别人的系统里。它自己就是一个完整的可运行系统——认知、记忆、情感、冲动、语言输出全在一个进程里。用户填一个 LLM API Key，`python run.py`，就能跟它对话。

**不做“插件”的代价**之一就是，如果想要让它有“Agent”一样的工作能力，只能从整个闭环内部进行二次开发，这个工作量已经远远超过了我目前的能力。

**如果你能提供一些帮助的话“非常感谢”**。

这个方向对不对，路还远远没趟平。但它确实不是"又一个记忆插件"。


### 设计选择：紧耦合

整个管线是纠缠在一起的。功能区块可以定义——意图、情绪、人格、关系——但它们之间不是模块化接口调用，而是像生物神经网络一样紧密交织。这不是技术上的"做不到模块化"，而是设计上的主动选择：如果把意图分析和情绪感知拆成两个独立服务，它们之间就只剩数据协议的交互，丢掉了彼此感知和共振的能力。

### 运行闭环

整个系统由两层管线组成：请求-响应管线处理每一次对话，后台自主节律在用户不在时持续运行。两层通过记忆存储和冲动注入交织在一起——详见下方[工作原理](#工作原理)。

### 定位差异一览

| | Mem0 | Zep | Letta | MemOS | 初痕 |
|---|---|---|---|---|---|
| **本质** | 记忆 API 服务 | 时序知识图谱引擎 | Agent 记忆 OS | 记忆操作系统 | 自循环记忆体 |
| **集成者** | 开发者（pip/SDK） | 开发者（部署/API） | 开发者（Agent SDK） | 开发者（部署/API） | 想搭聊天/陪伴/桌宠产品的人 |
| **最终受益者** | Agent 的聊天用户 | Agent 的聊天用户 | Agent 的聊天用户 | Agent 的聊天用户 | 他们自己产品的最终用户 |
| **怎么用** | pip install → 调 API | 部署服务 → 调 API | pip install → Agent SDK | 部署 → API 调用 | pip install → 填 Key → run.py |
| **对外接口** | SDK / REST API | MCP / REST API | Python SDK / ADE | Memory API | REST / OpenAI 兼容 / SSE 流式 |
| **运行模式** | 嵌入 Agent 使用 | 嵌入 Agent 使用 | Agent 框架的一部分 | 需要上层应用 | 一个进程全闭环 |
| **后台自主节律** | — | — | sleeptime compute | — | 10 线程：巩固/冲动/蒸馏/模式 |
| **主动开口** | — | — | — | — | 引擎不等用户，想说就说 |
| **耦合方式** | 松耦合（分离式） | 松耦合 | 中耦合（Agent 控记忆） | 松耦合 | 紧耦合——不可拆分 |

这些差异没有谁对谁错。Mem0 和 Zep 选择了松耦合——提供灵活的 API/SDK，让开发者把记忆嵌入到任何架构里。初痕选择了紧耦合——提供一整个自循环的记忆体，别人拿它当基础设施，在上面搭自己想做的产品。两种选择对应两类不同的使用者。



---

## 为什么没有 LongMemEval / LoCoMo 跑分

你可能注意到 Mem0 等系统会晒 LongMemEval、LoCoMo 等 benchmark 分数。初痕没有，原因很简单：

**这些 benchmark 测的是"事实召回"，不是"认知引擎"。**

它们的测试方式是：冷注入大量事实 → 问问题 → 测召回率。这本质上是在测一个**键值数据库的检索精度**，不是在测一个**有自主节律的认知引擎**的能力。

初痕的设计目标不是"存得多、查得准"，而是：
- 在对话中自然积累认知（不是冷注入）
- 理解用户的人格和情绪变化
- 在后台自主巩固、蒸馏、发现模式
- 在合适的时机主动开口

这些能力无法用"你问一条事实，我答一条事实"的 benchmark 来度量。事实上早期有一个叫 Jarvis 的实验版——SQLite + FAISS，能跑 benchmark。但它本质上还是"存进去、查出来"，没有自主节律，没有人格建模，没有冲动系统。它记性很好，但它不认识你。

初痕是在否定了那条路之后从头做的。

**如果你有兴趣帮着跑 benchmark，非常欢迎。** 提 Issue 或 PR 都可以，我会全力配合。

> 仓库里包含审计套件（`scripts/audit.py`），覆盖语义检索、关键词、时间检索、排序等 8 类场景。虽然不是行业标准，但对系统功能做了全面验证，可作为参考。

---

## 工作原理

初痕由两层管线组成。请求-响应管线处理每一次对话——从用户消息进入，到 LLM 回复流出。后台自主节律在用户不在时持续运行——巩固记忆、蒸馏人格、发现模式、产生冲动并在合适时机主动开口。两层管线通过记忆存储和冲动注入交织在一起。

```
                         ┌─── 请求-响应管线 ───┐
                         │                      │
  用户消息                 │                      │         SSE 流式输出
  ───────→ Embedding ──→ 8路并行检索 ──→ 两级精排 ──→ CircuitOrchestrator
            (bge-m3)    (语义/BM25/标签/    (嵌入+命中)    │
                         实体/关键词/注意/                │
                         时间/共现)                       ├─ 意图分析（bge-m3 原型匹配）
                                                         ├─ 情绪分析（Russell 二维环）
                                                         ├─ 认知分层 + 门控决策
                                                         ├─ 冲动注入 + 行为预测
                                                         ├─ 关系状态评估
                                                         └─ 产出 UtteranceSpec
                                                                │
                                                                ▼
                                          LLM 生成回复 ←── LLMClient
                                          (prompt 含记忆+人格+冲动+执行指令)
                                                                │
                                                                ▼
                          ┌────────── 存储 ──────────┐
                          │                           │
                          ├─ chat_history.append() ──→ JSONL 对话记录
                          ├─ _enqueue_store_task() ──→ 队列 → worker
                          │                              │
                          │    ┌─────────────────────────┘
                          │    ▼
                          │   摘要 + 标签 + Embedding → ChromaDB 记忆库
                          │   冲突检测 ←── 新旧记忆对比 → 自动替换过时记忆
                          │
                          └─ working_memory 增量更新

  ┌──────────────────── 后台自主节律（10 线程，不等用户）─────────────────┐
  │                                                                        │
  │  5 冲动源（泊松节律）          冲动消费者              巩固引擎        │
  │  ┌─────────────────┐      ┌──────────────┐      ┌──────────────┐      │
  │  │ 情绪趋势          │      │              │      │ 浅巩固 4h     │      │
  │  │ 时间节律          ├──→ PriorityQueue ──→ 取信号 →LLM 发言       │      │
  │  │ 随机漫游          │      │ 存为[内心独白] │      │ │ 人格蒸馏    │      │
  │  │ 好奇心            │      └──────────────┘      │ │ 话题树重建  │      │
  │  │ 行为模式          │                            │ 深巩固 24h    │      │
  │  └─────────────────┘                            │ │ 认知画像提炼│      │
  │                                                  └──────────────┘      │
  │  DMN 空闲检测 ──→ 触发巩固    AI 巩固 ──→ AI 表达模式分析+蒸馏        │
  │  模式发现 ──→ 多时间尺度统计识别（零 LLM 调用）                        │
  └────────────────────────────────────────────────────────────────────────┘
```

### 请求-响应管线：一条消息的旅程

**① Embedding。** 用户消息到达后，首先通过 bge-m3（Ollama 本地推理）转为 1024 维向量。这一步完全本地，不消耗任何外部 API。

**② 检索。** 向量同时触发 8 条检索路径——语义相似、BM25 关键词、标签倒排、实体匹配、注意力漂移、时间触发、共现扩展、话题树分支。每条路径独立召回候选记忆，然后进入两级精排：第一级按 embedding 余弦相似度 + hit_count 加权评分，第二级按来源优先级（语义 > 预热 > 实体 > 关键词 > 标签 > 时间 > 共现）统一排序。最终选出 top-K 条记忆进入下一阶段。

**③ 回路编排（CircuitOrchestrator）。** 这是引擎的认知核心。它拿到检索结果后，依次执行：
- 意图分析：bge-m3 将用户消息与预定义的意图原型做语义匹配，判断是 casual / question / emotional_sharing / request / command
- 情绪分析：同样走 bge-m3 语义原型匹配，映射到 Russell 二维情绪环（效价 + 唤醒度），产出 emotion + intensity
- 认知分层：将检索到的记忆按 fact / reference / background 三级分层，决定哪些记忆需要显式注入 prompt，哪些作为背景信息
- 门控决策：根据意图 + 情绪 + 记忆置信度 + 关系状态，决定 tone（温暖/冷静/幽默）、formality（0~1）、response_mode（先共情/先了解/直接回答/先确认）
- 冲动注入：检查 PriorityQueue 中是否有后台冲动源产出的待消费信号，如有则注入到 UtteranceSpec
- 行为预测：基于马尔可夫链概率表，预测用户接下来可能说什么、需要什么
- 关系评估：综合 familiarity（熟悉度）、trust（信任度）、closeness（亲密度）和 interaction_mode，产出当前关系快照

所有步骤在一个方法调用内串行完成。不是微服务，不是 pipeline DAG——就是一组函数在一个线程里挨个执行。彼此之间不通过 JSON 通信，而是直接访问同一块内存里的数据结构。

**④ LLM 生成。** UtteranceSpec 交到 LLMClient。LLMClient 负责把它翻译成 LLM 能消费的格式——记忆格式化为 tool role 的 JSON、冲动转为自然语言提示、门控决策转为执行指令（"语气要温暖，先共情再回应"）、人格标签注入 system prompt。然后调 LLM API，流式返回文本。工具调用（搜索、读写文件、执行 shell）在这一步处理——LLM 可以调用工具，工具结果注入下一轮生成，最多两轮。

**⑤ 存储。** 回复生成后，两条存储路径并行触发。同步路径写 chat_history.jsonl（对话记录）和触发 working_memory 增量更新（最近 N 条对话的轻量摘要）。异步路径将消息放入内存队列，由队列 worker 消费：调用 LLM 生成摘要 → bge-m3 提取语义标签 → 实体抽取（qwen2.5:3b）→ 情绪分析 → 时间特征标注 → 写入 ChromaDB。入库时自动触发冲突检测——如果新记忆与旧记忆语义高度相似且时间更新，旧记忆被标记为 stale 并最终被替换。

### 后台自主节律：用户不在时做什么

后台管线不依赖用户消息。引擎启动后，10 个 daemon 线程独立运行，各自拥有自己的泊松节律或定时周期。

**冲动系统（6 个线程）。** 5 个冲动源各自独立运行——情绪趋势检测用户最近的情绪走向变化，时间节律发现用户在特定时段的行为模式，随机漫游从记忆库中随机抽取旧记忆，好奇心探索从未被提起过的话题，行为模式识别用户的行为范式。每个源按泊松分布独立触发，产出 (content, priority) 信号，经疲劳抑制后进入 PriorityQueue。第 6 个线程——冲动消费者——轮询队列：当用户空闲超过 2 分钟，取出优先级最高的信号，调用 LLM 将其转为自然语言发言，存为 `[内心独白]` 写入 chat_history 和 ChromaDB。引擎就是这样"主动开口"的——不等用户发消息，自己想说了就说。

**巩固引擎（2 个线程）。** 浅巩固每 4 小时触发一次：重建话题树、检测记忆冲突、执行人格蒸馏（从对话中提炼用户的行为模式、思维模式、偏好模式、沟通模式等标签）。深巩固每 24 小时触发一次：在浅巩固的基础上，进行跨天级别的模式对比、演变趋势检测、认知画像提炼。DMN 线程负责空闲检测——用户多久没说话了，是否到了该触发巩固的时间点。

**AI 巩固（1 个线程）。** 独立于用户记忆，分析 AI 自己的表达模式——AI 在什么情绪下用什么语气回复、AI 的表达习惯是否在变化。产出的 AI 人格标签与用户人格标签分开存储，在 LLM 生成时注入 system prompt 的"我自己的表达习惯"区。

**模式发现（无独立线程，由巩固触发）。** 多时间尺度统计模式识别——对话频率变化趋势、话题漂移速度、情绪波动周期。全部基于纯统计方法，零 LLM 调用。产出两个方向：tuning（自动调参建议，如情绪淡化开关、正式度偏移量、主动开口抑制）和 observation（人类可读的模式描述，注入 LLM prompt 作为背景信息）。

### 两层管线的交汇点

请求-响应管线和后台节律不是隔离的——它们在几个关键点交织：

- **记忆库是共享的。** 请求管线写入新记忆，后台巩固管线读取和重组这些记忆。用户刚聊完的内容，几分钟后就会被浅巩固纳入话题树。
- **冲动注入请求管线。** 后台冲动源产出的信号在 CircuitOrchestrator 的冲动注入步骤被检查——如果用户正在对话时有待消费的冲动信号，它会被注入 UtteranceSpec，影响 LLM 的回复方向。
- **人格标签双向流动。** 后台蒸馏产出的人格标签在 LLM 生成时被注入 system prompt，影响 LLM 对用户的理解。而 LLM 回复中的新信息又被存储、被蒸馏、反过来更新人格标签。
- **关闭任何一个方向，另一个方向也会退化。** 没有后台巩固，记忆只是堆积而不被理解。没有请求管线，后台冲动无人倾听。

---

## 快速启动

### 你需要

- **Python 3.11+**
- **Ollama** + bge-m3 模型
- （可选）**DeepSeek API Key** —— 不填也能跑，只是不会说话

```bash
# 1. 安装 Ollama 并拉取 Embedding 模型
ollama pull bge-m3

# 2. 克隆 & 安装
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

pip install -r requirements.txt

# 3. （可选）配置 LLM Key（支持 DeepSeek / OpenAI / 硅基流动 等）
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY 和 LLM_BASE_URL

# 4. 启动
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

## API

引擎暴露 REST API 和 OpenAI 兼容端点。接任何客户端（NextChat、Open WebUI、自定义前端、桌宠外壳），或者直接在代码里调。

### 聊天

| 端点 | 说明 |
|------|------|
| `POST /chat` | 对话（完整回复，含 trace） |
| `POST /chat/stream` | 对话（SSE 流式，含 reasoning + content + trace） |
| `POST /v1/chat/completions` | OpenAI 兼容端点 |

### 管理

| 端点 | 说明 |
|------|------|
| `GET /health` | 健康检查 |
| `GET /api/ping` | 心跳 |
| `GET /api/user-active` | 用户活跃心跳（前端每 10s 调用，供冲动系统判断空闲） |
| `GET /api/memories` | 记忆列表（支持语义搜索、标签筛选、分页） |
| `GET /api/memories/stats` | 记忆统计 |
| `GET /api/memories/{id}` | 记忆详情（含上下文） |
| `POST /api/memories/{id}/correct` | 纠正记忆摘要 |
| `DELETE /api/memories/{id}` | 删除记忆 |
| `POST /api/memories/feedback` | 提交记忆错误报告 |
| `GET /api/personalities` | 人格标签列表 |
| `GET /api/consolidation/status` | 巩固状态 |
| `GET /api/distill/status` | 蒸馏状态 |
| `GET /api/chat/history` | 聊天历史 |
| `GET /api/prompt` | 查看/修改系统提示词 |

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
├── core/          # 认知管线：回路编排 · 认知状态 · 门控决策 · 上下文管理
├── brain/         # 语义引擎核心 semantic.py（~240 行，零模型依赖）
│   ├── semantic.py        # 7 个语义函数：标签/意图/情绪/否定/紧急度/分词/实体
│   ├── models.py          # 兼容外壳（调 semantic.py）
│   ├── keywords.py        # 关键词常量
│   └── metrics.py         # 训练指标持久化
├── memory/        # ChromaDB 记忆库 + 工作记忆 + 倒排/共现/时间索引
├── retrieval/     # 8 路并行检索 + BM25/Embedding 两级精排
├── background/    # 后台节律：4h/24h 巩固 · 5 源冲动 · 蒸馏 · 冲突检测 · 生命周期
├── analysis/      # Russell 情绪环 · 实体提取 · 模式发现 · 人格对称性 · 行为预测
├── personality/   # 双人格系统（用户 + AI 独立演化）
├── llm/           # 本地 embedding (bge-m3) + DeepSeek 对话生成 + 摘要
├── api/           # REST 端点：聊天 · 记忆管理 · 人格 · 巩固 · 蒸馏
├── tools/         # 原子写入 · 工具分发 · 搜索 · 文件操作
├── config/        # 中央配置
└── models/        # Pydantic schemas

tests/             # 320+ 测试
scripts/           # 审计套件 + 工具脚本
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

## 技术栈

- Python 3.11+ / FastAPI / ChromaDB（本地持久化）
- Embedding: bge-m3 via Ollama（1024 维）
- 语义核: bge-m3（关键词抽取 / 意图情绪原型匹配）
- 实体抽取: Ollama qwen2.5:3b（入库异步，零感知）
- 否定检测: 白名单 + 距离规则（无模型）
- 紧急度: 10 行规则（无模型）
- BM25 分词: 字符 2-gram + rank-bm25
- 主 LLM: DeepSeek API（deepseek-v4-flash, 1M 上下文）
- 部署: Windows / macOS / Linux, Docker 可选

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `OLLAMA_EMBED_MODEL` | 是 | Embedding 模型名，默认 `bge-m3` |
| `LOCAL_LLM_OLLAMA_URL` | 是 | Ollama 地址，默认 `http://localhost:11434` |
| `LLM_API_KEY` | 否 | LLM API Key（不填则引擎不会说话） |
| `LLM_BASE_URL` | 否 | LLM API 地址，默认 `https://api.deepseek.com` |
| `LLM_MODEL` | 否 | 模型名，默认 `deepseek-v4-flash` |
| `DEEPSEEK_API_KEY` | 否 | （旧名，仍可用）等同 `LLM_API_KEY` |
| `LOCAL_LLM_ENABLED` | 否 | 启用本地 LLM，默认 `true` |
| `LOCAL_LLM_MODEL` | 否 | 本地 LLM 模型名，默认 `qwen2.5:7b` |
| `BOCHA_API_KEY` | 否 | 博查搜索 API Key |
| `DATA_DIR` | 否 | 数据目录，默认 `./data` |
| `USERS` | 否 | 多用户认证 JSON |
| `IMPULSE_ACTIVE_PATH_B` | 否 | 冲动系统开关，默认 `true` |

详见 `.env.example`。

---

## 审计

```bash
python scripts/audit.py              # 全量 8 类
python scripts/audit.py --quick      # 快速模式
python scripts/audit.py --category 1 # 单项（语义检索）
```

报告保存在 `audit/` 目录。

---

## 贡献

欢迎提 Issue 和 PR。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

[MIT License](LICENSE)。

---

[📝 作者的话](AUTHOR.md)
