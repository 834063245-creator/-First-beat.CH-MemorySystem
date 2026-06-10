# 初痕 · First Beat — 自循环记忆体

> 10 路并行检索 · 12 维认知画像 · 5 源泊松冲动 · 3 层存储架构 · 引擎自主节律 · 填 Key 就跑

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1098%20passed%2C%200%20failed-green.svg)]()
[![Coverage](https://img.shields.io/badge/line%20coverage-80%25-brightgreen.svg)]()
[![E2E](https://img.shields.io/badge/E2E%2BInt-163%20nodes%20%E2%9C%93-brightgreen.svg)]()
[English](README_EN.md)

👉 [快速上手](QUICKSTART.md) ([EN](QUICKSTART_EN.md)) | 🔧 [安装排查](SETUP.md) ([EN](SETUP_EN.md)) | [架构图](ARCHITECTURE_DIAGRAM.md) | [环境诊断](verify_env.py)

---

## 一句话

**别人给 LLM 加记忆插件；初痕让 LLM 当自己的语言皮层。**

初痕不是 SDK、不是 API 服务、不是 Agent 框架的插件。它是一个**自闭环的记忆基础设施**——10 路并行语义检索、12 维认知画像常驻注入、5 源泊松冲动在后台自主运行、3 层存储（向量 + 结构化 + 流式日志）覆盖从原文到关系的完整记忆谱系。引擎自己决策、自己巩固、自己发现模式——LLM 只是它的嘴。

**初痕做什么：** 填一个 LLM API Key，`python run.py`，你就有了一个会记住你、会主动开口、会在你离线时自己消化和沉淀的认知系统。在上面搭聊天应用、桌宠、陪伴型 Agent——那是你的事。初痕只管记忆和说话。

**初痕不做什么：** 不做多租户 SaaS。不做 SDK 嵌入。不做 LangChain/Mem0/Zep 的替代方案。它是 1 对 1 的——一个引擎服务一个用户，紧耦合，不可拆分。这是主动选择，不是做不到。

---

## 量化

| 指标 | 数值 |
|------|------|
| 检索路径 | **10 路**并行（语义 hot/cool · BM25 全文 · 关键词 · 标签 · 实体 · 共现 · 时间 · 话题树 · 注意力漂移） |
| 检索延迟 | **<500ms**（含 bge-m3 embedding，不含 LLM 生成） |
| 认知画像 | **12 维**（8 stable 常驻前缀缓存 + 4 dynamic 每轮更新） |
| 存储层 | **3 层**（ChromaDB 向量 + SQLite 结构化 + JSONL 流式日志） |
| 后台线程 | **10 个** daemon（5 冲动源 + 消费者 + DMN 巩固 + AI 巩固 + 情绪淡化 + 存储队列） |
| 记忆状态机 | **4 态**（hot → warm → cool → stale/archived），软降权，不硬屏蔽 |
| LongMemEval | **92%**（纠正后，100 题子集） |
| 测试 | **1,098** passed，0 failed，80% 行覆盖，76 测试文件（64 单元 + 6 E2E + 6 集成），E2E 5 链路 89 节点 |
| 代码量 | ~38,600 行 Python，161 文件 |
| 总 commit | **607**（跨 6 仓库：jarvis → amazing → amazing3 → amazing4 → amazing5 → First Beat） |
| 首次 commit | 2026-05-22（19 天从零到开源） |

[LongMemEval 实验报告](LONGMEMEVAL_REPORT.md) · [关于 benchmark 本身的批判](BENCHMARK_CRITIQUE.md)

---

## 为什么不是又一个记忆插件

大多数 AI 记忆项目（Mem0、Zep、Letta、MemOS 等）提供 SDK 或 API，让开发者把记忆嵌入到自己的 Agent 里。初痕选的是另一条路——自己就是一个完整的自循环记忆体，不嵌进任何系统。

**设计选择：紧耦合。** 意图、情绪、人格、关系不是独立微服务——它们在一个进程内直接访问同一块内存。不是做不到模块化，而是拆开后它们之间只剩数据协议，丢掉了彼此感知和共振的能力。

### 竞争对手 vs 初痕

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

## 基准测试

**LongMemEval: 92%（纠正后）** · 100 题精选子集 · DeepSeek V4 Flash 生成 · DeepSeek-Chat 评分

原始分数 80.0%，纠正 LLM-as-Judge 误判和数据集错题后达到 92%。初痕是目前已知唯一在 LongMemEval 上超过 90% 的中文原生自循环记忆体。

> 详细实验数据、逐题分析和评分纠正见 [`LONGMEMEVAL_REPORT.md`](LONGMEMEVAL_REPORT.md)。  
> 关于 LongMemEval 和 LoCoMo 作为记忆系统评测标准的根本性缺陷，见 [`BENCHMARK_CRITIQUE.md`](BENCHMARK_CRITIQUE.md)。

仓库内置审计套件（`scripts/audit.py`），覆盖语义检索、关键词检索、时间检索、排序准确性等 8 类场景，可作为系统功能验证参考。

---

## 工作原理

<details>
<summary><b>展开架构详情</b></summary>

初痕由两层管线组成。请求-响应管线处理每一次对话——从用户消息进入，到 LLM 回复流出。后台自主节律在用户不在时持续运行——巩固记忆、蒸馏人格、发现模式、产生冲动并在合适时机主动开口。两层管线通过记忆存储和冲动注入交织在一起。

```
                         ┌─── 请求-响应管线 ───┐
                         │                      │
  用户消息                 │                      │         SSE 流式输出
  ───────→ Embedding ──→ 10路并行检索 ──→ 引擎编织 ──→ CircuitOrchestrator
            (bge-m3)    (语义/BM25/标签/ (weave_context)   │
                         实体/注意/时间/   4层决策机制      │
                         话题树/共现)                      ├─ 意图分析（bge-m3 原型匹配）
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

  ┌──────────────────── 后台自主节律（不等用户）─────────────────┐
  │                                                                        │
  │  5 冲动源（泊松节律）          冲动消费者              巩固引擎        │
  │  ┌─────────────────┐      ┌──────────────┐      ┌──────────────┐      │
  │  │ 情绪趋势          │      │              │      │ 浅巩固 4h     │      │
  │  │ 时间节律          ├──→ PriorityQueue ──→ 取信号 →LLM 发言       │      │
  │  │ 随机漫游          │      │ 存为[内心独白] │      │ │ 画像浅更新  │      │
  │  │ 好奇心            │      └──────────────┘      │ │ 话题树重建  │      │
  │  │ 行为模式          │                            │ 深巩固 24h    │      │
  │  └─────────────────┘                            │ │ 画像深更新  │      │
  │                                                  │ │ 认知画像提炼│      │
  │                                                  └──────────────┘      │
  │  DMN 空闲检测 ──→ 触发巩固    AI 巩固 ──→ 镜像：独立 ConsolidationEngine    │
  │  模式发现 ──→ 多时间尺度统计识别（零 LLM 调用）                        │
  │  画像实时更新 ──→ 每轮对话后轻量更新（<100ms，不调 LLM）              │
  └────────────────────────────────────────────────────────────────────────┘
```

### 请求-响应管线：一条消息的旅程

**① Embedding。** 用户消息到达后，首先通过 bge-m3（Ollama 本地推理）转为 1024 维向量。这一步完全本地，不消耗任何外部 API。

**② 检索。** 向量同时触发 10 条检索路径——语义 hot、语义 cool、BM25 关键词、标签倒排、实体匹配、注意力漂移、时间触发、话题树分支、共现扩展。在 ThreadPoolExecutor 中并发执行（max_workers=7），各路独立召回候选记忆。候选记忆进入**引擎编织（weave_context）**——v2.0 引入的四层决策机制，替代固定的 TOP_K 截断：故事线编织（按实体/标签聚类，识别跨时间叙事和情绪趋势）→ 认知分层（fact / reference / background / suppressed）→ Token 预算分配（20000 token 软限制）→ 来源优先级排序。全程零 LLM 调用，延迟 < 150ms。v2.1 新增软降权体系：90 天线性衰减 + archived 上限 0.6 + stale 上限 0.3，不再硬屏蔽任何记忆。

**③ 回路编排（CircuitOrchestrator）。** 这是引擎的认知核心。它拿到检索结果后，依次执行：
- 意图分析：bge-m3 将用户消息与预定义的意图原型做语义匹配，判断是 casual / question / emotional_sharing / request / command
- 情绪分析：同样走 bge-m3 语义原型匹配，映射到 Russell 二维情绪环（效价 + 唤醒度），产出 emotion + intensity
- 认知分层：将检索到的记忆按 fact / reference / background 三级分层，决定哪些记忆需要显式注入 prompt，哪些作为背景信息
- 门控决策：根据意图 + 情绪 + 记忆置信度 + 关系状态，决定 tone（温暖/冷静/幽默）、formality（0~1）、response_mode（先共情/先了解/直接回答/先确认）
- 冲动注入：检查 PriorityQueue 中是否有后台冲动源产出的待消费信号，如有则注入到 UtteranceSpec
- 行为预测：基于马尔可夫链概率表，预测用户接下来可能说什么、需要什么
- 关系评估：综合 familiarity（熟悉度）、trust（信任度）、closeness（亲密度）和 interaction_mode，产出当前关系快照
- 画像注入：从 PortraitRenderer 渲染 stable 画像（8 维，命中 DeepSeek 前缀缓存）和 dynamic 画像（4 维，每轮更新），注入 UtteranceSpec——认知画像不走检索召回，常驻注入

所有步骤在一个方法调用内串行完成。不是微服务，不是 pipeline DAG——就是一组函数在一个线程里挨个执行。彼此之间不通过 JSON 通信，而是直接访问同一块内存里的数据结构。

**④ LLM 生成。** UtteranceSpec 交到 LLMClient。LLMClient 负责把它翻译成 LLM 能消费的格式——记忆格式化为 tool role 的 JSON、冲动转为自然语言提示、门控决策转为执行指令（"语气要温暖，先共情再回应"）、人格标签注入 system prompt。然后调 LLM API，流式返回文本。工具调用（搜索、读写文件、执行 shell）在这一步处理——LLM 可以调用工具，工具结果注入下一轮生成，最多两轮。

**⑤ 存储 + 画像实时更新。** 回复生成后，三条路径并行触发。同步路径写 chat_history.jsonl（对话记录）和触发 working_memory 增量更新（最近 N 条对话的轻量摘要）。异步路径将消息放入内存队列，由队列 worker 消费：本地 Ollama qwen2.5:3b 生成摘要（零 API 费用）→ bge-m3 提取语义标签 → 实体抽取（复用 qwen2.5:3b）→ 情绪分析 → 时间特征标注 → 写入 ChromaDB。入库时自动触发冲突检测——如果新记忆与旧记忆语义高度相似且时间更新，旧记忆被标记为 stale 并最终被替换。AI 记忆入库获得与用户记忆完全对等的元数据（实体、完整时间特征、session_continued 标记）。第三条路径——画像实时更新——将本轮的关系评估结果交给 PortraitWriter，触发轻量状态更新（仅 usr2/ai2 + usr4/ai4 维度，<100ms，不调 LLM）。同时将用户和 AI 消息写入各自对应的 AI 记忆库（ai_chroma），供后台巩固使用。

### 后台自主节律：用户不在时做什么

后台管线不依赖用户消息。引擎启动后，10 个 daemon 线程独立运行，各自拥有自己的泊松节律或定时周期。

**冲动系统（6 个线程）。** 5 个冲动源各自独立运行——情绪趋势检测用户最近的情绪走向变化，时间节律发现用户在特定时段的行为模式，随机漫游从记忆库中随机抽取旧记忆，好奇心探索从未被提起过的话题，行为模式识别用户的行为范式。每个源启动时先经过 120s 冷却期（等系统预热完），然后按泊松分布独立触发，产出 (content, priority) 信号，经疲劳抑制后进入 PriorityQueue。第 6 个线程——冲动消费者——轮询队列：当用户空闲超过 2 分钟，取出优先级最高的信号，调用 LLM 将其转为自然语言发言，存为 `[内心独白]` 写入 chat_history 和 ChromaDB。引擎就是这样"主动开口"的——不等用户发消息，自己想说了就说。

**巩固引擎 + 画像系统（DMN 合并 ticker）。** 浅巩固每 4 小时触发一次：重建话题树、检测记忆冲突、画像浅更新（提取器分析 → LLM 写条目 → 并入 PORTRAIT.md）。深巩固每 24 小时触发一次：跨天级别的模式对比、演变趋势检测、画像深更新（全局扫描 + LLM 合成）。画像实时更新每轮对话后触发（引擎特征提取，<100ms，不调 LLM）。DMN 线程负责空闲检测——用户多久没说话了，是否到了该触发巩固的时间点。

**AI 巩固（已合并到 DMN ticker）。** AI 拥有与用户侧完全镜像的独立 ConsolidationEngine 实例（ai_dmn），在 DMN 合并 ticker 中与用户侧共享 on_idle/浅巩固/深巩固触发 —— 不再使用独立 worker 线程。AI 记忆入库时获得与用户记忆完全对等的元数据（实体提取、完整时间特征、session_continued 标记）。AI 情绪淡化保留独立每小时定时器。产出的 AI 画像标签（ai1~ai6 维度）与用户画像（usr1~usr6 维度）统一存储在 PORTRAIT.md 中，在 LLM 生成时注入 system prompt。

**模式发现（无独立线程，由巩固触发）。** 多时间尺度统计模式识别——对话频率变化趋势、话题漂移速度、情绪波动周期。全部基于纯统计方法，零 LLM 调用。产出两个方向：tuning（自动调参建议，如情绪淡化开关、正式度偏移量、主动开口抑制）和 observation（人类可读的模式描述，注入 LLM prompt 作为背景信息）。

### 两层管线的交汇点

请求-响应管线和后台节律不是隔离的——它们在几个关键点交织：

- **记忆库是共享的。** 请求管线写入新记忆，后台巩固管线读取和重组这些记忆。用户刚聊完的内容，几分钟后就会被浅巩固纳入话题树。
- **冲动注入请求管线。** 后台冲动源产出的信号在 CircuitOrchestrator 的冲动注入步骤被检查——如果用户正在对话时有待消费的冲动信号，它会被注入 UtteranceSpec，影响 LLM 的回复方向。
- **画像双向流动。** 后台画像系统产出 12 维认知画像（usr1~usr6 + ai1~ai6），分 stable（8 维，命中缓存前缀）和 dynamic（4 维，每轮更新）两段常驻注入 system prompt。每轮对话后实时画像更新（<100ms），后台 4h/24h 巩固时触发浅/深画像更新。LLM 回复中的新信息又被存储、被分析、反过来更新画像条目。
- **关闭任何一个方向，另一个方向也会退化。** 没有后台巩固，记忆只是堆积而不被理解。没有请求管线，后台冲动无人倾听。

</details>

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
| `GET /api/consolidation/status` | 巩固状态 |
| `GET /api/portrait/render` | 画像渲染（stable + dynamic） |
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

> 详细设计决策和模块依赖见 [ARCHITECTURE.md](ARCHITECTURE.md)。

```
app/
├── core/          # 认知管线：回路编排 · 认知状态 · 门控决策 · 上下文管理 · 瓶颈监控 · 反馈
├── brain/         # 语义引擎核心 semantic.py（~240 行，零模型依赖）
│   ├── semantic.py        # 7 个语义函数：标签/意图/情绪/否定/紧急度/分词/实体
│   ├── models.py          # 兼容外壳（调 semantic.py）
│   ├── keywords.py        # 关键词常量
│   └── metrics.py         # 训练指标持久化
├── memory/        # ChromaDB（用户+AI 双集合）+ 工作记忆摘要 + 倒排/共现/实体对/时间索引
├── retrieval/     # 10 路并行检索 + 引擎编织（weave_context）四层决策 + v2.1 软降权
├── background/    # 后台节律：4h/24h 巩固 · 5 源冲动 · 蒸馏 · 镜像AI巩固 · 生命周期
├── analysis/      # Russell 情绪环 · 实体提取 · 模式发现 · 人格对称性 · 行为预测
├── portrait/      # 认知画像系统：12 维画像管理 · 实时/浅/深更新 · 渲染注入 · 提取器
├── personality/   # 双人格系统（Phase 4 退役中，由画像系统替代）
├── llm/           # 本地 embedding (bge-m3) + LLM 对话生成 + 本地摘要（qwen2.5:3b）
├── api/           # REST 端点：聊天 · 记忆管理 · 画像 · 巩固 · 反馈
├── tools/         # 原子写入 · 工具分发 · 搜索 · 文件操作
├── config/        # 中央配置
└── models/        # Pydantic schemas

tests/             # 64+ 单元测试文件（行覆盖率 80%+）
E2E/               # 端到端全链路回归（6 文件，5 链路）
integration/       # 集成测试（6 文件）
```

---

## 设计哲学

| # | 原则 | 含义 |
|---|------|------|
| 1 | **原文不加工** | 原文永不压缩。摘要和 Embedding 是翻译，不是加工 |
| 2 | **时间即骨架** | 时间参与组织、关联、浮现，不是衰减因子 |
| 3 | **行为即权重** | hit_count 决定权重，无时间衰减函数 |
| 4 | **引擎自有节律** | 巩固/冲动/模式发现独立运行，不依赖用户在线 |
| 5 | **引擎决策 → LLM 执行** | LLM 不拥有记忆、不调检索工具，只按引擎指令说话 |
| 6 | **画像常驻注入** | 认知画像每轮无条件注入 prompt，不走检索召回——对人的理解不应该取决于今天聊什么

---

## 技术栈

- Python 3.11+ / FastAPI / ChromaDB（本地持久化）
- Embedding: bge-m3 via Ollama（1024 维）
- 语义核: bge-m3（关键词抽取 / 意图情绪原型匹配）
- 实体抽取: Ollama qwen2.5:3b（入库异步，零感知；AI 侧与用户侧完全对等）
- 摘要生成: Ollama qwen2.5:3b（复用实体抽取模型，零 API 费用）
- 画像合成: 主 LLM（引擎提取特征 → LLM 写入/合并条目）
- 否定检测: 白名单 + 距离规则（无模型）
- 紧急度: 10 行规则（无模型）
- BM25 分词: 字符 2-gram + rank-bm25
- 主 LLM: DeepSeek API（deepseek-v4-flash, 1M 上下文，可替换为任意 OpenAI 兼容供应商）
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
| `LOCAL_LLM_ENABLED` | 否 | 启用本地 LLM（摘要 + 实体抽取），默认 `true` |
| `LOCAL_LLM_MODEL` | 否 | 本地 LLM 模型名，默认 `qwen2.5:7b`（实际摘要复用 qwen2.5:3b） |
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
