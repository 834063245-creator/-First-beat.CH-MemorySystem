# 初痕架构说明书

> 为什么这个系统长这样，以及各部分之间怎么配合。

---

## 目录

1. [核心设计决策](#核心设计决策)
2. [两层管线](#两层管线)
3. [认知状态层 — 引擎与 LLM 的边界](#认知状态层)
4. [检索管线 — 8 路并行](#检索管线)
5. [后台自主节律](#后台自主节律)
6. [记忆生命周期](#记忆生命周期)
7. [关键技术选择](#关键技术选择)
8. [已知技术债](#已知技术债)

---

## 核心设计决策

### 决策 1：引擎决策 → LLM 执行

**几乎所有 AI 记忆系统**的模式是：LLM 调用记忆检索工具，拿到结果后自己判断哪些有用、怎么用。

**初痕反过来**：引擎做完所有决策——检索哪条、置信度多少、用什么语气、要不要注入冲动——打包成一个 `UtteranceSpec` dataclass，LLM 只负责把它翻译成自然语言。LLM 不拥有记忆，不调检索工具，不决策。

**为什么：**
- LLM 有上下文窗口限制。把检索、排序、置信度判断交给引擎（纯算法），LLM 的窗口全部留给"说话"
- 决策可复现、可审计、可调试。引擎的每一步都有日志和耗时记录
- 换 LLM 供应商不影响记忆质量。引擎做的决策不依赖任何一个特定模型

### 决策 2：紧耦合而非松耦合

所有功能模块（意图、情绪、人格、关系、记忆）在一个进程内，函数直接调用，不通过 RPC/消息队列/微服务边界。

**为什么：**
- 意图分析和情绪感知之间存在信息交换——如果拆成两个独立服务，它们之间只剩数据协议，丢失了彼此感知的能力
- 紧耦合不意味着"无法扩展"——意味着"决策可以在所有上下文之间自由流动"

**代价：** 模块边界模糊，`ConsolidationEngine` 一个类管了太多事（已知，见[已知技术债](#已知技术债)）

### 决策 3：时间不是衰减因子，是组织骨架

大多数记忆系统用时间做指数衰减——越旧的记忆权重越低。初痕不用任何时间衰减函数。

**为什么：**
- 旧记忆未必不重要。一条 3 个月前关于"我害怕失败"的记忆，比昨天 10 条"今天吃了什么"更重要
- 时间参与组织（按时间段索引、按节律触发）、关联（共现矩阵）、浮现（冲动系统的好奇心源），但不参与评分衰减
- 权重由 `hit_count`（行为）决定：被回想得多的记忆自然权重高

### 决策 4：引擎有自己的节律，不等用户

后台 10 个 daemon 线程各跑各的泊松节律。巩固、冲动、蒸馏、模式发现——不管用户在不在线都跑。

**为什么：**
- 真正的记忆系统不是"用户问我才找"——是在用户不在的时候自主消化、整理、发现模式
- 冲动系统让引擎在用户空闲时主动开口，而不是永远被动等待

### 决策 5：原文永不压缩

原文存入 ChromaDB，摘要和 embedding 是翻译（用于检索），不是加工（不替代原文）。

**为什么：**
- 摘要丢失细节。一旦原文被丢弃，那些细节永远回不来
- 检索用摘要+embedding，但 LLM 看到的 prompt 里是原文

---

## 两层管线

```
请求-响应管线（每次对话）          后台自主节律（不等用户）
─────────────────────────        ─────────────────────────
用户消息                          5 冲动源  ——→ PriorityQueue
  │                               情绪趋势/时间节律/
  ├─ Embedding (bge-m3)           随机漫游/好奇心/行为模式
  ├─ 8路并行检索                   │
  ├─ 两级精排                     冲动消费者（空闲>2min）
  ├─ CircuitOrchestrator            │
  │   ├─ 意图分析                 LLM 生成 → [内心独白]
  │   ├─ 情绪分析                  
  │   ├─ 认知分层                 巩固引擎
  │   ├─ 门控决策                   ├─ 浅巩固 4h
  │   ├─ 冲动注入                   │   话题树重建
  │   ├─ 行为预测                   │   语义重复检测
  │   └─ 关系评估                   │   人格蒸馏
  │                                └─ 深巩固 24h
  ├─ LLM 生成回复                     归档评估
  │                                    话题笔记
  └─ 存储（chat_history + ChromaDB）
```

两层管线的交汇点：
- **记忆库共享** — 请求管线写，后台巩固读和重组
- **冲动注入** — 后台冲动信号在 CircuitOrchestrator 被检查，影响 LLM 回复方向
- **人格双向流动** — 后台蒸馏 → system prompt，LLM 回复 → 存储 → 更新人格

---

## 认知状态层

这是整场重构的基石——位于 `app/core/state.py`。

### 旧架构 vs 新架构

```
旧：引擎 → 文字纸条（"[高置信] 用户喜欢喝咖啡"）→ LLM 自己判断
新：引擎 → CognitiveState → LLM 只按决策执行
```

### MemoryDirective 的四级分层

| role | 含义 | LLM 该做什么 |
|------|------|-------------|
| `fact` | 引擎高置信 | 可以直接当作事实引用 |
| `reference` | 引擎有一定把握 | 需要带核实语气 |
| `background` | 上下文相关 | 用来调语气，不需要提及 |
| `suppressed` | 引擎已过滤 | 不给 LLM 看到 |

LLM 收到的 prompt 里没有置信度标签、没有来源标注、没有情绪元数据——只有"这件事你可以直接引用"或者"这件事你提一下但要核实"。

### UtteranceSpec 的构建过程

```
CircuitOrchestrator.process()
  ├─ 1. 意图分析     → UserMessageAnalysis (intent + emotion + urgency)
  ├─ 2. 情绪分析     → 补充 emotion_intensity
  ├─ 3. 认知分层     → fact / reference / background / suppressed
  ├─ 4. 门控决策     → GatingDecision (tone + formality + response_mode)
  ├─ 5. 冲动注入     → 检查 PriorityQueue，注入 ImpulseDirective
  ├─ 6. 行为预测     → mirror_prediction（下一步可能说什么）
  └─ 7. 关系评估     → RelationshipState (familiarity + trust + closeness)
                        │
                        ▼
                   UtteranceSpec → LLMClient.generate() → 回复
```

---

## 检索管线

位于 `app/retrieval/pipeline.py`。

### 8 路检索

| 路径 | 方法 | 特点 |
|------|------|------|
| ① 语义 hot | ChromaDB (heat=hot) | 高活跃记忆优先 |
| ② 语义 cool | ChromaDB (warm/cool) | 低活跃兜底，sim≥0.3 |
| ③ 关键词 | BM25 + 倒排索引 | AND → OR 退化 |
| ④ 标签 | 标签倒排索引 | 精确匹配 ≥1 个标签 |
| ⑤ 实体 | 实体名精确匹配 | PERSON/LOCATION/ORG |
| ⑥ 共现 | 共现矩阵扩展 | 跟已命中记忆共现过的 |
| ⑦ 时间触发 | TemporalPatternIndex | 当前时段的历史模式 |
| ⑧ 话题树 | 话题树分支扩展 | 同一话题簇的其他记忆 |
| ⑨ 注意力漂移 | 最近 3 轮加权 embedding | 模拟注意力惯性 |

### 意图门控

不同意图分配不同的检索路径配额：

| intent | semantic | tag | entity | time_expand |
|--------|----------|-----|--------|-------------|
| casual | 10 | 5 | 0 | 0 |
| recall | 20 | 8 | 5 | 5 |
| ask_fact | 25 | 10 | 5 | 0 |
| emotional_sharing | 12 | 5 | 0 | 3 |
| conflict | 25 | 10 | 5 | 5 |

### 去重和排序

1. 各路结果按 `id` 去重
2. 两级精排：embedding cosine + hit_count 加权
3. 来源优先级：semantic > dmn_preheat > entity > keyword > tag > time > co_occurrence

### 引擎编织（weave_context）

v2.0 引入：替代固定的 TOP_K 截断，全程零 LLM 调用，延迟目标 < 150ms。

**四层决策机制：**

```
候选记忆（8路检索结果）
    │
    ├─ 预处理：去 stale + 解析元数据
    │
    ├─ 判断 should_speak
    │     └─ 闲聊 + 候选 ≤3 → 不说（避免无意义回复）
    │
    ├─ 层一：故事线编织
    │     ├─ 按实体/标签聚类（跨时间）
    │     ├─ 计算时间跨度（≥1天才算故事线）
    │     └─ 提取情绪趋势（延续/翻转/持续积极/持续消极）
    │
    ├─ 层三：认知分层
    │     ├─ fact：故事线内记忆 + 语义距离 < 0.30
    │     └─ background/discard：其余按规则过滤
    │
    └─ 层四：Token 预算分配
          ├─ MAX_TOKENS = 2000（软限制）
          └─ 按叙事摘要截断，非硬截断
```

**关键设计：**

| 特性 | 实现方式 |
|------|---------|
| 叙事识别 | 按实体+标签聚类，提取跨时间模式 |
| 情绪趋势 | 检测正负情绪变化（同一实体的多次提及） |
| 来源感知 | semantic_hot(1.0) > entity(0.85) > kw(0.7) > ... |
| Token 控制 | 不是固定数量，而是 2000 token 软预算 |

---

## 后台自主节律

### 冲动系统（6 个线程）

```
5 个冲动源                  冲动消费者（1 个线程）
──────────                 ──────────────────────
情绪趋势 (10min)    ──┐    轮询 PriorityQueue
时间节律 (30min)    ──┤     │
随机漫游 (10min)    ──┼──→  ├─ 检查空闲（>2min）
好奇心   (20min)    ──┤     ├─ 取最高优先级
行为模式 (30min)    ──┘     ├─ LLM 生成自然语言
                            └─ 存为 [内心独白]
```

**内抑制机制：**
- 每个源有疲劳度（0~1），每发射一次 +0.15
- 疲劳度半衰期 15 分钟
- 有效优先级 = 基础优先级 × (1 - 疲劳度)
- 有效优先级 < 2 → 丢弃（抑制）

### 巩固引擎

| 级别 | 间隔 | 触发方式 | 内容 |
|------|------|---------|------|
| 浅巩固 | 4h | 独立线程 | 话题树重建、语义重复检测、标签嵌入索引、人格对称性、事实冲突检测 |
| 深巩固 | 24h | 独立线程 | 归档评估、话题笔记生成、情绪淡化 |
| 空闲巩固 | 按空闲时长 | DMN worker | Level 2 (回顾+预热)、Level 3 (日巩固+冲突扫描) |

### 模式发现（零 LLM）

`app/analysis/pattern_discovery.py` — 6h 运行一次，纯统计：

- **时间节律**：TemporalPatternIndex → 当前时段话题模式 → 自动调参
- **情绪锚点**：Russell 坐标 → 话题情绪关联
- **话题漂移**：前后半段话题分布对比
- **交互节奏**：会话长度、间隔分析
- **趋势检测**：线性回归斜率 → formality_shift / emotional_dampening 趋势

---

## 记忆生命周期

```
hot（刚创建/活跃命中）          warm（正常）           cool（冷）
     │                            │                    │
     │ hit_count 增长              │ 14天无人问津         │
     │                            ▼                    │
     │                         cool                    │
     │                                                 │
     │ 情绪翻转 / 事实更新                               │
     ▼                                                 ▼
  stale（被取代）                              archived（归档）
```

- **hot → warm**：自然过渡，由 hit_count 和活跃时间决定
- **warm → cool**：14 天内 hit_count=0，浅巩固时自动冷却
- **stale**：新记忆与旧记忆语义相似且情绪翻转 / 事实更新，旧的被标记 stale
- **archived**：话题簇中位数最后命中时间超过阈值（默认 30 天）

---

## 关键技术选择

| 层 | 选型 | 原因 |
|----|------|------|
| Embedding | bge-m3 via Ollama | 1024 维，中文友好，本地运行零 API 成本 |
| 向量存储 | ChromaDB | 本地持久化，hnsw 索引，无需外部服务 |
| 中文分词 | 字符 2-gram + bge-m3 KeyBERT | 不依赖 jieba，准确率相当 |
| 语义核 | bge-m3 原型匹配 | 意图/情绪分类不用训练分类器，惰性 cache |
| 情绪模型 | Russell 二维环（valence × arousal） | 比一维正负分类更细腻 |
| 实体抽取 | qwen2.5:3b (Ollama) | 本地运行，仅入库时异步调用 |
| 主 LLM | DeepSeek API (兼容 OpenAI) | 可通过改 BASE_URL/API_KEY/MODEL 换供应商 |

---

## 已知技术债

### ConsolidationEngine 职责过重（P1）

`app/background/consolidation.py` 约 1000 行，一个类承担了：空闲巩固、预热缓存、日巩固、冲突检测、话题笔记、冷热扫描、话题树重建、标签嵌入索引、事实冲突检测、人格对称性、归档评估。

**建议拆分方向：**
- `TopicNoteManager` — 话题笔记的读写和过期管理（~100 行）
- `ConflictDetector` — 事实冲突检测的三层漏斗逻辑（~150 行）
- `ArchivalManager` — 归档评估和执行（~80 行）
- `ConsolidationEngine` — 保留核心调度 + 预热 + 空闲触发

### O(n²) 全量扫描（P1 — 部分修复）

语义重复检测已从双层 for 循环改为 ChromaDB query（O(n log n)）。但 `_check_conflicts`（冲突预扫描）和 `_assess_archival`（归档评估）仍用 `list_all()` 全量扫描。记忆量 < 5000 条时影响不大，超过后需要分页或增量处理。

### 端到端回归测试（✅ 已完成）

`tests/test_e2e_regression.py` 已覆盖以下场景：
1. **多日对话记忆保持** - 7天前的记忆仍能正确检索
2. **情绪翻转检测** - 同一事实域的正负情绪翻转被正确识别
3. **人格标签一致性** - 连续对话后标签不漂移
4. **冲动系统疲劳抑制** - 连续同源信号被正确抑制
5. **倒排索引线程安全** - 并发读写不崩溃、不返回脏数据
6. **Russell 情绪环边界** - 空输入、让步句、亲密词等边界情况
7. **模式发现趋势检测** - 上升/下降/稳定趋势正确识别
8. **检索意图门控** - 不同意图正确分配检索配额
9. **事实冲突检测漏斗** - 三层过滤逻辑验证

### 观测性（P1 — 已有基础）

`/api/status` 端点提供了聚合快照。下一步：
- 前端 dashboard 消费此端点
- 添加 Prometheus metrics（`bottleneck.py` 的数据可以暴露为 metrics）
- 关键路径的 alerting（检索全部为空、冲动队列堆积、ChromaDB 写失败）

---

## 模块依赖图

```
app/
├── core/          ← 认知管线核心（不依赖 api/ 和 tools/）
│   ├── state.py         认知状态数据结构（MemoryDirective, UtteranceSpec）
│   ├── bottleneck.py    全链路耗时监控
│   ├── feedback.py      记忆错误报告
│   └── context.py       AppContext 服务容器
│
├── brain/         ← 语义引擎（零模型依赖，除 bge-m3）
│   └── semantic.py      7 个公开函数：标签/意图/情绪/否定/紧急度/分词/实体
│
├── memory/        ← 存储层
│   ├── chroma.py        ChromaDB 封装
│   ├── inverted.py      词/标签→记忆ID 倒排索引
│   ├── cooccur.py       共现矩阵
│   ├── affinity.py      话题亲和图
│   ├── temporal.py      时间模式索引
│   └── tree.py          话题树（层次聚类）
│
├── retrieval/     ← 检索管线（依赖 memory/ + brain/）
│   ├── pipeline.py      8 路检索 + 门控
│   └── scoring.py       两级精排
│
├── analysis/      ← 分析层（依赖 brain/）
│   ├── emotion.py       Russell 二维情绪环
│   ├── pattern_discovery.py  零 LLM 模式发现
│   ├── entity.py        实体抽取
│   └── predictor.py     行为预测（马尔可夫）
│
├── personality/   ← 双人格系统
│
├── background/    ← 后台自主节律（依赖所有上层）
│   ├── consolidation.py  巩固引擎
│   ├── impulse.py        冲动系统
│   └── lifecycle.py      线程生命周期管理
│
├── llm/           ← LLM 适配层
│   ├── deepseek.py      主 LLM 客户端（兼容 OpenAI API）
│   ├── embed.py         本地 embedding (bge-m3)
│   └── local.py         本地 LLM (qwen2.5:7b，用于摘要)
│
├── api/           ← REST 层（依赖所有上层）
│   ├── app.py           FastAPI 工厂
│   ├── chat.py          聊天端点
│   ├── system.py        系统/健康/状态端点
│   └── ...
│
└── tools/         ← 工具层（原子写入、搜索、文件操作）
```

**依赖方向：** api/ → core/ → brain/ + memory/ → retrieval/ + analysis/ + personality/ → background/ → llm/

**循环依赖控制：** `llm/` 和 `core/` 之间通过 TYPE_CHECKING 延迟导入避免循环。

---

*最后更新：2026-06-05*
