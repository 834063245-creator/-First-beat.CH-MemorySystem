# 初痕架构说明书

> 为什么这个系统长这样，以及各部分之间怎么配合。

---

## 目录

1. [核心设计决策](#核心设计决策)
2. [系统全景](#系统全景)
3. [认知状态层 — 引擎与 LLM 的边界](#认知状态层)
4. [认知画像系统 — PORTRAIT.md 统一汇聚](#认知画像系统)
5. [检索管线 — 10 路并行 + 编织](#检索管线)
6. [后台自主节律](#后台自主节律)
7. [记忆生命周期](#记忆生命周期)
8. [AI 自我表达记忆](#ai-自我表达记忆)
9. [工作记忆摘要](#工作记忆摘要)
10. [用户反馈闭环](#用户反馈闭环)
11. [E2E Benchmark 体系](#e2e-benchmark-体系)
12. [关键技术选择](#关键技术选择)
13. [已知技术债](#已知技术债)
14. [模块依赖图](#模块依赖图)

---

## 核心设计决策

### 决策 1：引擎决策 → LLM 执行

**几乎所有 AI 记忆系统**的模式是：LLM 调用记忆检索工具，拿到结果后自己判断哪些有用、怎么用。

**初痕反过来**：引擎做完所有决策——检索哪条、置信度多少、用什么语气、要不要注入冲动——打包成一个 `UtteranceSpec` dataclass，LLM 只负责把它翻译成自然语言。LLM 不拥有记忆，不调检索工具，不决策。

**为什么：**
- LLM 有上下文窗口限制。把检索、排序、置信度判断交给引擎（纯算法），LLM 的窗口全部留给"说话"
- 决策可复现、可审计、可调试。引擎的每一步都有日志和耗时记录
- 换 LLM 供应商不影响记忆质量。引擎做的决策不依赖任何一个特定模型

**代价：** prompt 注入结构依赖底座模型的缓存策略（当前为 DeepSeek 特调）。换模型需重构注入层（见[已知技术债](#已知技术债)）。

### 决策 2：紧耦合而非松耦合

所有功能模块（意图、情绪、人格、关系、记忆）在一个进程内，函数直接调用，不通过 RPC/消息队列/微服务边界。

**为什么：**
- 意图分析和情绪感知之间存在信息交换——如果拆成两个独立服务，它们之间只剩数据协议，丢失了彼此感知的能力
- 紧耦合不意味着"无法扩展"——意味着"决策可以在所有上下文之间自由流动"

**设计目标：** 1 对 1 服务（一个引擎服务一个用户），不是多租户。这个约束是主动选择，不是做不了。

**代价：** 模块边界模糊，`ConsolidationEngine` 一个类管了太多事（已知，见[已知技术债](#已知技术债)）

### 决策 3：时间不是衰减因子，是组织骨架

大多数记忆系统用时间做指数衰减——越旧的记忆权重越低。初痕不用任何时间衰减函数。

**为什么：**
- 旧记忆未必不重要。一条 3 个月前关于"我害怕失败"的记忆，比昨天 10 条"今天吃了什么"更重要
- 时间参与组织（按时间段索引、按节律触发）、关联（共现矩阵）、浮现（冲动系统的好奇心源），但不参与评分衰减
- 权重由 `hit_count`（行为）决定：被回想得多的记忆自然权重高

### 决策 4：引擎有自己的节律，不等用户

后台 10+ 个 daemon 线程各跑各的泊松节律。巩固、冲动、蒸馏、模式发现——不管用户在不在线都跑。

**为什么：**
- 真正的记忆系统不是"用户问我才找"——是在用户不在的时候自主消化、整理、发现模式
- 冲动系统让引擎在用户空闲时主动开口，而不是永远被动等待

### 决策 5：原文永不压缩

原文存入 ChromaDB，摘要和 embedding 是翻译（用于检索），不是加工（不替代原文）。

**为什么：**
- 摘要丢失细节。一旦原文被丢弃，那些细节永远回不来
- 检索用摘要+embedding，但 LLM 看到的 prompt 里是原文
- **教训：** v2.0 的缓存优化曾错误地让 LLM 只看到摘要——单 session 事实召回从 ~96% 跌到 79%。修复后（原文 + 摘要同时传 LLM）回归正常。这个 bug 在生产环境跑了两个月未被察觉，直到 LongMemEval benchmark 把盖子掀开。

### 决策 6：Benchmark 模式与认知管线隔离

`BENCHMARK_MODE=true` 环境变量触发。不影响正常代码路径，所有改动在 flag 后隔离。

**为什么：**
- Benchmark 测的是"搜出原文喂给 LLM"，系统的认知层（摘要、情绪、实体、编织、衰减）在这种场景下是噪音
- 通过 feature flag 做适配，而不是靠改代码或特调 prompt
- 保留检索管线的完整参与（10 路并行 + BM25 全文 + 全量兜底），只 bypass 认知过滤层

---

## 系统全景

```
请求-响应管线（每次对话）              后台自主节律（不等用户）
─────────────────────────            ─────────────────────────
用户消息                              5 冲动源  ──→ PriorityQueue
  │                                   情绪趋势/时间节律/
  ├─ 意图/情绪分类（bge-m3 原型）      随机漫游/好奇心/行为模式
  ├─ 10 路并行检索                     │
  ├─ 两级精排                         冲动消费者（空闲>2min）
  ├─ 引擎编织（weave_context）           │
  │   ├─ 故事线检测                   LLM 生成 → [内心独白]
  │   ├─ 认知分层 (fact/ref/bg/supp)
  │   ├─ 冲突检测                     巩固引擎（DMN 合并 ticker）
  │   └─ Token 预算                     ├─ 浅巩固 4h
  ├─ 回路调度（CircuitOrchestrator）    │   话题树重建/重复检测/
  │   ├─ 门控决策                       │   画像浅更新/冲突检测
  │   ├─ 冲动注入                       ├─ 深巩固 24h
  │   ├─ 行为预测                       │   归档评估/话题笔记/情绪淡化/
  │   ├─ 关系评估                       │   画像深更新
  │   └─ 画像注入（stable+dynamic）     ├─ AI 巩固（镜像 ai_dmn）
  ├─ LLM 生成回复                          与用户侧共享触发，
  └─ 存储（chat_history + ChromaDB        独立 ConsolidationEngine
      + AI 记忆库 + 工作记忆摘要）      │
      + 画像实时更新（usr2/ai2+4）     模式发现 6h
                                        时间/情绪/话题/节奏/趋势

                                      画像系统
                                        ├─ 实时更新（每轮对话后，<100ms）
                                        ├─ 浅更新（4h，提取器→LLM写条目）
                                        └─ 深更新（24h，全局扫描+LLM合成）
```

管线的交汇点：
- **记忆库共享** — 请求管线写，后台巩固读和重组
- **冲动注入** — 后台冲动信号在 CircuitOrchestrator 被检查，影响 LLM 回复方向
- **画像双向流动** — 后台画像系统 → portrait_stable/dynamic 常驻注入 system prompt，LLM 回复 → 存储 → 关系评估 → 实时更新画像 → 深巩固全局合成
- **工作记忆共享** — 请求管线写入对话摘要，下次请求带回上下文

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

LLM 收到的 prompt 里没有文字标签——置信度是连续值 `relevance`，情绪是原始值 `emotional_intensity` + `valence`，不做"高/中/低"或"情绪·正向/负向"的分类。LLM 自己判断权重。

### UtteranceSpec 的构建过程

```
CircuitOrchestrator.process()
  ├─ 1. 意图分析     → UserMessageAnalysis (intent + emotion + urgency)
  ├─ 2. 情绪分析     → 补充 emotion_intensity (感叹号/emoji/程度副词)
  ├─ 3. 检索         → 10 路并行 + 编织 + 认知分层
  ├─ 4. 门控决策     → GatingDecision (tone + formality + response_mode)
  ├─ 5. 冲动注入     → 检查 PriorityQueue，注入 ImpulseDirective
  ├─ 6. 行为预测     → mirror_prediction（Markov 预测下一步意图/话题）
  ├─ 7. 关系评估     → RelationshipState (familiarity + trust + closeness)
  ├─ 8. 画像注入     → PortraitRenderer.render_stable() + render_dynamic()
  │                    stable (8维) 注入 message[0] system prompt (前缀缓存命中)
  │                    dynamic (4维) 注入 message[N+1] system prompt (每轮更新)
  └─ 9. 人格注入     → [Phase 4 退役完成] 用户人格标签 + AI 自我表达标签
                        │
                        ▼
                   UtteranceSpec → LLMClient.generate() → 回复
```

---

## 认知画像系统

位于 `app/portrait/`。PORTRAIT.md 替代分散的 PersonalityStore + DistillEngine + personality_notes 注入，提供 12 维度的持久化认知画像，常驻注入 LLM prompt。

### 核心问题与被解决的设计缺陷

**旧设计：** 人格标签每次对话做语义检索召回（`rerank_tags(top_k=3)`）。聊 A 话题时看不到 B 话题蒸馏出的性格特征。认知产出（意图/情绪/关系/行为预测）算完即丢弃。蒸馏产出是孤立的 `{content, type, confidence}` 碎片标签，没有聚合为"这个人是什么样的人"的连贯画像。

**新设计：** 画像是一个 12 维度的结构化 Markdown 文档（PORTRAIT.md），每轮对话无条件注入 LLM prompt——不走检索召回。认知结论增量累积，每轮对话后实时更新关系快照维度（<100ms，不调 LLM），4h/24h 巩固时由引擎提取特征 → LLM 合成文本 → 并入画像。

### 12 维度结构

| 维度 | 用户侧 | AI 侧 | 更新层级 | 说明 |
|------|-------|-------|---------|------|
| 核心特征 | usr1 | ai1 | deep | 长期稳定的人格/表达特征 |
| 当前状态 | usr2 | ai2 | realtime | 情绪/精力/兴趣的当前快照 |
| 行为节律 | usr3 | ai3 | shallow | 时间模式/习惯频率 |
| 关系快照 | usr4 | ai4 | realtime | familiarity/trust/closeness |
| 兴趣图谱 | usr5 | ai5 | shallow | 话题偏好/知识覆盖 |
| 情绪图谱 | usr6 | ai6 | deep | 长期情绪轨迹/反应模式 |

stable 画像（8 维：usr1/3/5/6 + ai1/3/5/6）注入 message[0] system prompt，命中 DeepSeek 前缀缓存（>95% 命中率）。dynamic 画像（4 维：usr2/4 + ai2/4）注入 message[N+1] system prompt，每轮对话后实时更新。

### 三层更新节律

```
PortraitWriter
├─ realtime_update(utterance_spec, relationship_state)
│    每轮对话后触发（chat.py 中 await loop.run_in_executor）
│    仅更新 4 维度（usr2/ai2 + usr4/ai4）
│    引擎特征提取 + 规则合成，<100ms，不调 LLM
│
├─ shallow_update(app_context)
│    挂载 DMN 浅巩固节律（4h）
│    提取器扫描近期记忆 → 变化检测 → LLM 写条目 → 并入 PORTRAIT.md
│    覆盖 usr3/ai3 + usr5/ai5 + 填充 usr1/ai1 的新候选条目
│
└─ deep_update(app_context)
     挂载 DMN 深巩固节律（24h）
     全局扫描 → LLM 合成 → 合并/淘汰旧条目
     覆盖 usr1/ai1 + usr6/ai6，最低 20 轮对话门槛
```

### 画像注入格式

LLM 收到的 prompt 中，stable 画像作为独立段落注入 message[0] 的 system prompt 末尾，dynamic 画像作为独立 system message 注入消息列表 N+1 位：

```
message[0] system:  "你是初痕..." [核心人格 + 工具规则 + stable画像(8维)]
message[1] user:    历史对话...
message[2] assistant: ...
...
message[N] system:   [dynamic画像(4维) + session_context + now_hint()]
message[N+1] user:  当前用户消息
```

### 画像与检索的接触点

画像不走检索——它常驻注入 prompt。唯一的检索接触点在 `pipeline.py` 的精排阶段：如果候选记忆关联的实体/标签与画像条目高相关，给 +0.05 的 light boost。不会因为"没匹配到"而丢掉画像信息——画像已经在 prompt 里了。

### 模块结构

```
app/portrait/
├── __init__.py       # 模块入口
├── manager.py        # PORTRAIT.md 加载/解析/写入/条目 CRUD
├── state.py          # PortraitEntry / EntryStatus / EntryStateMachine
├── extractors.py     # 特征提取器（tag_counter / entity_aggregator / ...）
├── renderer.py       # 渲染为 stable/dynamic prompt 段
└── writer.py         # 三层更新（realtime / shallow / deep）
```

---

## 检索管线

位于 `app/retrieval/pipeline.py`。延迟目标 < 500ms（含 embedding）。

### 10 路并行检索

| 路径 | 方法 | 特点 |
|------|------|------|
| ① 语义 hot | ChromaDB (heat=hot) | 高活跃记忆优先 |
| ② 语义 cool | ChromaDB (warm/cool) | 低活跃兜底，sim≥0.3 |
| ③ 关键词 | 倒排索引（摘要） | AND → OR 退化 |
| ④ 标签 | 标签倒排索引 | 精确匹配 ≥1 个标签 |
| ⑤ 实体 | 实体名精确匹配 | PERSON/LOCATION/ORG 等 |
| ⑥ 共现 | 共现矩阵扩展 | 跟已命中记忆共现过的 |
| ⑦ 时间触发 | TemporalPatternIndex | 当前时段的历史模式 |
| ⑧ 话题树 | 话题树分支扩展 | 同一话题簇的其他记忆 |
| ⑨ 注意力漂移 | 最近 3 轮加权 embedding | 模拟注意力惯性 |
| ⑩ BM25 全文 | 全文 BM25Okapi 索引 | 对 ChromaDB 全部 document 建索引 |

路径 ⑩（v2.2）解决关键词倒排索引建在摘要上的漏检问题。路径⑨（v2.0）模拟人的注意力惯性——连续聊同一话题时，相关记忆被自动加权。

### 意图门控

不同意图分配不同的检索路径配额：

| intent | semantic | tag | entity | time_expand |
|--------|----------|-----|--------|-------------|
| casual | 10 | 5 | 0 | 0 |
| recall | 20 | 8 | 5 | 5 |
| ask_fact | 25 | 10 | 5 | 0 |
| emotional_sharing | 12 | 5 | 0 | 3 |
| conflict | 25 | 10 | 5 | 5 |

Benchmark 模式配额翻倍（放宽检索限制）。

### 去重和排序

1. 各路结果按 `id` 去重
2. 两级精排：embedding cosine + hit_count 加权 + recency_weight 软降权 + portrait light boost (+0.05)
3. v2.1 软降权公式：`recency_weight = 1.0 - (days_ago / 90) × (1.0 - 0.15)`，下限 0.15
   - archived 记忆上限 0.6
   - stale 记忆上限 0.3
4. 来源优先级：semantic > bm25_fulltext > entity > keyword > tag > time > co_occurrence > attention
5. **Benchmark 全量兜底：** 当 BENCHMARK_MODE=true 且 ChromaDB 记忆 ≤ 200 条时，直接全量返回，零检索遗漏
6. **Phase 4: 人格标签检索已退役** — 画像常驻注入替代了每次对话的 `rerank_tags(top_k=3)` 语义召回。画像不走检索，每轮无条件注入 prompt。

### 引擎编织（weave_context）

v2.0 引入：替代固定的 TOP_K 截断，全程零 LLM 调用，延迟目标 < 150ms。

**四层决策机制：**

```
候选记忆（10 路检索结果）
    │
    ├─ 预处理：去 stale + 解析元数据
    │
    ├─ 判断 should_speak
    │     └─ 闲聊 + 候选 ≤3 → 不说（避免无意义回复）
    │
    ├─ 层一：故事线编织
    │     ├─ 按实体/标签聚类（跨时间）
    │     ├─ 计算时间跨度（≥1 天才算故事线）
    │     └─ 提取情绪趋势（延续/翻转/持续积极/持续消极）
    │
    ├─ 层二：认知分层
    │     ├─ fact：故事线内记忆 + 语义距离 < 0.30 × source_boost
    │     ├─ reference / background：其余按 relevance 分级
    │     └─ suppressed：引擎过滤，不给 LLM
    │
    └─ 层三：Token 预算分配
          ├─ MAX_TOKENS = 20000（软限制）
          └─ 按叙事摘要截断，非硬截断
```

**关键设计：**

| 特性 | 实现方式 |
|------|---------|
| 叙事识别 | 按实体+标签聚类，提取跨时间模式 |
| 情绪趋势 | 检测正负情绪变化（同一实体的多次提及） |
| 来源感知 | semantic_hot(1.0) > entity(0.85) > kw(0.7) > ... |
| Token 控制 | 不是固定数量，而是 20000 token 软预算 |
| 闲聊抑制 | intent=casual + 候选≤3 → should_speak=False |

### V2 prompt 注入格式

v2.0 采用 JSON + tool role 注入（替代 v1 的纯文本 `【记忆】` 段落）：

```json
{
  "id": "mem_003",
  "time": "2026-06-04 15:35",
  "relative_time": "1天前",
  "summary": "用户自称痞老板，喜欢深夜写代码，每天喝三杯咖啡",
  "document": "完整原文，不再截断",
  "source": "semantic_hot",
  "hit_count": 12,
  "relevance": 0.92,
  "stale": false,
  "emotional_intensity": 3,
  "emotion_valence": "positive"
}
```

**关键设计原则：**
- 引擎只筛选（编织层 token 预算），不截断——`summary` 和 `document` 完整透传，无硬编码 `[:N]` 截断
- 情绪和置信度给原始值（`relevance: 0.92`、`emotional_intensity: 3`），不做文字标签——LLM 自己判断权重
- 记忆走 `tool` role 注入（API 原生识别为外部事实），与历史对话 `user/assistant` 分离
- system prompt + 历史对话构成稳定前缀，可被 DeepSeek prompt 缓存命中

---

## 后台自主节律

### 冲动系统（6 个线程 — 5 源 + 1 消费者）

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
- 超时信号（> TTL）自动丢弃

### 巩固引擎

| 级别 | 间隔 | 触发方式 | 内容 |
|------|------|---------|------|
| 浅巩固 | 4h | DMN 合并 ticker | 话题树重建、语义重复检测、标签嵌入索引、画像浅更新、人格对称性、事实冲突检测、实体对演化、冷热转换 |
| 深巩固 | 24h | DMN 合并 ticker | 归档评估、话题笔记生成、画像深更新、情绪淡化（高 arousal 旧记忆） |
| 空闲巩固 | 按空闲时长 | DMN worker | Level 1 预热（重建检索缓存）、Level 2 回顾（>4h）、Level 3 日巩固（>24h） |
| AI 巩固 | 与用户侧同步 | DMN 合并 ticker 共享触发 | AI 侧拥有完整 ConsolidationEngine 镜像（ai_dmn），和用户侧共享 on_idle/浅巩固/深巩固触发；AI 情绪淡化保留独立每小时定时器 |

### Phase 0b: AI 巩固镜像

AI 侧不再使用独立的 worker 线程简化逻辑。DMN 合并 ticker 中用户侧触发浅/深巩固时，AI 侧同步触发——两个 ConsolidationEngine 实例（`self.dmn` 和 `self.ai_dmn`）在同一个 ticker 循环中依次执行。AI 记忆入库时获得与用户记忆完全对等的元数据：实体提取（qwen2.5:3b）、完整 10 字段时间特征、session_continued 标记、基于 full_text（用户+AI）的双维度情绪分析。

### 蒸馏引擎（零 LLM — Phase 4 已退役/已删除）

`app/background/distill.py` — 已删除。纯统计方法曾由画像系统逐步替代，现画像系统已完全接管：
- 标签共现聚类
- 时间模式检测（时段→话题关联）
- 情绪关联分析
- 趋势分析

### 模式发现（零 LLM）

`app/analysis/pattern_discovery.py` — 6h 运行一次，纯统计，5 种模式：

- **时间节律**：TemporalPatternIndex → 当前时段话题模式 → 引擎自动调参
- **情绪锚点**：Russell 坐标 → 话题情绪关联
- **话题漂移**：前后半段话题分布对比
- **交互节奏**：会话长度、间隔分析
- **趋势检测**：线性回归斜率 → formality_shift / emotional_dampening 趋势

产出写入 `pattern_cache.json`，注入 prompt 的 `[模式观察]` 段。

### 线程生命周期管理

`app/background/lifecycle.py` — 统一注册/启动/停止：
- 崩溃自动重启（最多 5 次/小时/线程）
- 优雅退出（stop_event）
- 线程存活监控

---

## 记忆生命周期

```
hot（刚创建 / 情绪强度≥2）     warm（正常）          cool（冷）
     │                            │                    │
     │ hit_count 增长              │ 14 天无人问津        │
     │                            ▼                    │
     │                         cool                    │
     │                                                 │
     │ 情绪翻转 / 事实更新                               │
     ▼                                                 ▼
  stale（被取代，软降权）                       archived（归档）
  recency_weight ≤ 0.3                           recency_weight ≤ 0.6
```

### 状态转换

| 转换 | 触发条件 | 行为 |
|------|---------|------|
| 新建 → hot | `emotional_intensity >= 2` 或已有高情感 | 初始热度设为 hot |
| 新建 → warm | 上述条件不满足 | 初始热度设为 warm |
| → cool | 14 天无命中自动冷却 | 浅巩固时检查 |
| → stale | 新记忆与旧记忆语义相似 + 情绪翻转 / 事实更新 | 旧记忆标记 `stale=True`，记录 `superseded_by` |
| → archived | 话题簇中位数最后命中时间超过 90 天 | 标记 archived |

### v2.1: 软降权体系

不再硬屏蔽任何记忆。所有记忆留在候选池，通过 `recency_weight` 软降权：

- **正常记忆**：90 天线性衰减到 0.15
- **stale 记忆**：上限 0.3，不进 `fact_memories`，路由到 `stale_context`
- **archived 记忆**：上限 0.6
- **被报错记忆**：error_count ↑ → score ↓

stale 记忆注入 LLM 时携带 `stale_reason` 和 `superseded_by`，LLM 可作背景理解变化过程，但不得作为当前事实引用。

### 情绪衰减（独立于巩固调度器）

每 50 次 `increment_hit_count` 触发检查。3 天未命中的高 intensity 记忆，intensity 自然衰减。不依赖巩固调度器，在线路上自然发生。

---

## AI 自我表达记忆

独立的 ChromaDB 集合（`ai_memories`），存储 AI 的回复风格和表达习惯。v2.3 获得与用户记忆完全对等的元数据支持。

- **写入**：每次对话后分析 AI 回复，提取表达方式标签。AI 记忆入库获得完整元数据：实体提取（qwen2.5:3b）、10 字段时间特征、session_continued 标记、基于 full_text（用户+AI）的双维度情绪分析——与用户侧完全对等
- **检索**：R10 路径 — 当前语境下匹配历史表达风格
- **巩固**：AI 拥有独立 ConsolidationEngine 实例（ai_dmn），在 DMN 合并 ticker 中与用户侧共享 on_idle/浅巩固/深巩固触发。AI 情绪淡化保留独立每小时定时器
- **注入**：AI 画像标签（ai1~ai6 维度）统一存储在 PORTRAIT.md 中，由 PortraitRenderer 渲染后注入 system prompt——替代了旧的 `personality_notes_ai` 取前 5 条模式

---

## 工作记忆摘要

`app/memory/working.py` — 增量维护对话脉络，替代完整对话历史的注入。

```
旧方案：每次请求注入全部对话历史 → token 爆炸
新方案：工作记忆摘要（~3K tokens）+ 最近 5 轮原文（~2K tokens）
```

- **增量摘要**：每次对话后由本地 LLM（零 API 成本）增量更新
- **话题切换检测**：话题重叠率 < 30% 时触发全量重写
- **锁保护**：`RLock` 保护读写竞态
- **版本号**：每次更新递增，用于一致性校验

这不替代 ChromaDB 的记忆检索——它专门解决"我们正在聊什么"的瞬时上下文问题。

---

## 用户反馈闭环

`app/core/feedback.py` — 记忆质量的外部纠偏。

| 操作 | 机制 | 效果 |
|------|------|------|
| 用户报错 | `log_error_report()` → error_reports.jsonl | 检索时降权，error_count 越高 score 越低 |
| 用户纠正 | 新事实覆盖旧事实 + stale 标记 | 被纠正记忆获 +0.3 boost，同 tag 群组 +0.1 |
| 用户否定 | downvote | -0.3 惩罚 |
| 清除报错 | `clear_memory_errors()` | 追加清除标记，恢复权重 |

所有反馈走 JSONL 追加写入（不重写文件），保证并发安全。

---

## E2E Benchmark 体系

`E2E/` — 6 个测试文件，89 个检查节点，5 条链路。真实 ChromaDB + 真实 bge-m3 + 真实本地 LLM。不 mock。

| 链路 | 文件 | 节点数 | 核心问题 |
|------|------|--------|---------|
| 一：写入 | `test_write_path.py` | 12 | "存进去了吗？存对了吗？" |
| 二：检索+编织+认知 | `test_link2_retrieve.py` | 35 | "找得到吗？编织对了吗？回复靠谱吗？" |
| 三：跨轮记忆 | `test_link3_cross_turn.py` | 9 | "隔几轮还记得吗？换种问法还行吗？" |
| 四：记忆演化 | `test_link4_evolution.py` | 16 | "时间过了记忆质量退化了吗？" |
| 独立：后台节律 | `test_background.py` | 17 | "不等用户的时候，系统在干什么？干对了吗？" |

**设计原则：**
- 测试数据隔离：每个用例使用独立 ChromaDB 集合/临时目录
- 固定 random seed 保证可复现
- 五链路各自独立计分，**不加权合成一个数字**（加权总分掩盖问题）
- 哪条低修哪条

完整规格见 `BENCHMARK_SPEC.md`。

---

## 关键技术选择

| 层 | 选型 | 原因 |
|----|------|------|
| Embedding | bge-m3 via Ollama | 1024 维，中文友好，本地运行零 API 成本 |
| 向量存储 | ChromaDB | 本地持久化，HNSW 索引，无需外部服务 |
| 全文检索 | BM25Okapi（内存） | benchmark 适配 < 10K 条，生产可扩展磁盘索引 |
| 中文分词 | 字符 2-gram + bge-m3 KeyBERT | 不依赖 jieba，准确率相当 |
| 语义核 | bge-m3 原型匹配 | 意图/情绪分类不用训练分类器，惰性 cache |
| 情绪模型 | Russell 二维环（valence × arousal） | 比一维正负分类更细腻 |
| 实体抽取 | qwen2.5:3b (Ollama) | 本地运行，仅入库时异步调用 |
| 摘要生成 | qwen2.5:3b (Ollama) | 本地运行，零 API 费用（替代原 DeepSeek） |
| 主 LLM | DeepSeek API (兼容 OpenAI) | 可通过改 BASE_URL/API_KEY/MODEL 换供应商 |
| 部署 | Docker + docker-compose | Ollama 独立容器 + 应用容器，带 healthcheck |
| 数据加密 | 不加密（本地 1 对 1 部署） | 无网络传输，安全边界在宿主机层 |

---

## 已知技术债

### ConsolidationEngine 职责过重（P1）

`app/background/consolidation.py` 一个类承担了：空闲巩固、预热缓存、日巩固、冲突检测、话题笔记、冷热扫描、话题树重建、标签嵌入索引、人格对称性、归档评估。

**建议拆分方向：**
- `TopicNoteManager` — 话题笔记的读写和过期管理
- `ConflictDetector` — 事实冲突检测的三层漏斗逻辑
- `ArchivalManager` — 归档评估和执行
- `ConsolidationEngine` — 保留核心调度 + 预热 + 空闲触发

### PersonalityStore / DistillEngine 向 Portrait 迁移（P1 — 已完成）

Phase 4 已全部完成：personality/ 和 background/distill.py 模块已物理删除。画像系统完全接管，所有认知产出（意图/情绪/关系/行为预测）经 PortraitWriter 三层更新引擎写入 PORTRAIT.md。蒸馏相关 API（`/api/personalities`、`/api/distill/status`）已移除。已完成：
- consolidation.py 中的人格蒸馏调用已替换为 portrait_writer.shallow_update()
- personality/ 和 background/distill.py 模块已移除
- settings.py 中的 PERSONALITY/DISTILL 配置常量已清理

### Prompt 注入依赖 DeepSeek 缓存策略（P1 — 部分改善）

v2.3 将 system prompt 拆分为 stable 前缀（message[0]，命中缓存）+ dynamic 后缀（message[N+1]，每轮更新），使传统 prompt 缓存对流式对话更友好。stable 段包含核心人格 + 工具规则 + 8 维画像，命中率 >95%。但整体注入结构仍依赖 DeepSeek 的缓存策略。换底座模型时需重新评估。

### O(n²) 全量扫描（P1 — 部分修复）

语义重复检测已从双层 for 循环改为 ChromaDB query（O(n log n)）。但 `_check_conflicts` 和 `_assess_archival` 仍用 `list_all()` 全量扫描。记忆量 < 5000 条时影响不大，超过后需要分页或增量处理。

### 测试同步（P2 — 已修复）

已移除旧 test_personality.py / test_personality_deep.py / test_behavior.py，新增 5 个 portrait 测试文件。test_thread_safety.py 仍需同步更新。

### 观测性（P1 — 已有基础）

`/api/status` 端点提供了聚合快照。下一步：
- 前端 dashboard 消费此端点
- Prometheus metrics（`bottleneck.py` 的数据可暴露为 metrics）
- 关键路径告警（检索全部为空、冲动队列堆积、ChromaDB 写失败）

### 无 CI/CD Pipeline（P2）

当前无 GitHub Actions 或其他 CI。E2E 测试需手动运行。已写入 AUTHOR.md 协作需求中。

---

## 模块依赖图

```
app/
├── core/          ← 认知管线核心
│   ├── state.py         认知状态数据结构（MemoryDirective, UtteranceSpec）
│   ├── circuit.py       回路调度器（意图→检索→门控→注入→LLM）
│   ├── bottleneck.py    全链路耗时监控
│   ├── feedback.py      记忆错误报告与纠正
│   ├── conflict.py      冲突检测与消解（零自动裁决）
│   └── context.py       AppContext 服务容器
│
├── brain/         ← 语义引擎（零模型依赖，除 bge-m3）
│   ├── semantic.py      7 个公开函数：标签/意图/情绪/否定/紧急度/分词
│   ├── keywords.py      意图/情绪关键词表 + 程度副词（统一常量来源）
│   └── models.py        语义模型加载
│
├── memory/        ← 存储层
│   ├── chroma.py        ChromaDB 封装（用户记忆 + AI 记忆双集合）
│   ├── inverted.py      词/标签 → 记忆 ID 倒排索引
│   ├── cooccur.py       共现矩阵
│   ├── entity_pair.py   实体对共现追踪
│   ├── affinity.py      话题亲和图
│   ├── temporal.py      时间模式索引
│   ├── tree.py          话题树（层次聚类）
│   ├── working.py       工作记忆摘要（增量对话脉络）
│   ├── history.py       对话历史管理
│   └── tag_index.py     标签嵌入索引
│
├── retrieval/     ← 检索管线
│   ├── pipeline.py      10 路检索 + 门控 + 编织 + benchmark 全量兜底
│   ├── scoring.py       两级精排（cosine + hit_count + recency_weight）
│   └── bm25_fulltext.py BM25 全文索引（内存）
│
├── analysis/      ← 分析层（零 LLM）
│   ├── emotion.py       Russell 二维情绪环
│   ├── pattern_discovery.py  5 模式发现（时间/情绪/话题/节奏/趋势）
│   ├── entity.py        实体抽取（qwen2.5:3b 离线调用）
│   ├── symmetry.py      人格对称性分析（用户/AI 双矩阵盲区检测）
│   └── predictor.py     行为预测（Markov 链）
│
├── portrait/      ← 认知画像系统（引擎提取 + LLM 合成）
│   ├── manager.py       PORTRAIT.md 加载/解析/写入/条目 CRUD
│   ├── state.py         PortraitEntry / EntryStatus / EntryStateMachine
│   ├── extractors.py    特征提取器（tag_counter/entity_aggregator/...）
│   ├── renderer.py      渲染 stable(8维)/dynamic(4维) prompt 段
│   └── writer.py        三层更新（realtime/shallow/deep）
│
│
├── background/    ← 后台自主节律
│   ├── consolidation.py  巩固引擎（浅/深/空闲三级，用户+AI 双实例）
│   ├── impulse.py        冲动系统（5 源 + 消费者 + 内抑制）
│   ├── distill.py        蒸馏引擎（零 LLM，Phase 4 退役中）
│   └── lifecycle.py      线程生命周期（崩溃重启 + 限流）
│
├── llm/           ← LLM 适配层
│   ├── deepseek.py      主 LLM 客户端（兼容 OpenAI API）+ 画像注入
│   ├── embed.py         本地 embedding (bge-m3 via Ollama)
│   └── local.py         本地 LLM (qwen2.5:3b，摘要/实体)
│
├── api/           ← REST 层
│   ├── app.py           FastAPI 工厂
│   ├── chat.py          聊天端点 + benchmark 注入 + 管理重置 + 画像实时更新
│   │   ├─ /chat             普通对话
│   │   ├─ /chat/stream      流式对话
│   │   ├─ /v1/chat/completions  OpenAI 兼容
│   │   ├─ /benchmark/inject  benchmark 直接入库（走存储管线，不调 LLM）
│   │   └─ /admin/reset       清空所有记忆和索引
│   ├── system.py        系统/健康/状态端点
│   ├── memories.py      记忆查询端点
│   ├── consolidation.py 巩固状态端点
│   ├── portrait.py      画像渲染端点
│   └── chat_history.py  对话历史端点
│
└── tools/         ← 工具层
    ├── atomic.py         原子文件写入
    ├── workspace.py      工作区操作
    └── dispatch.py       记忆查询 dispatch
```

**依赖方向：** api/ → core/ → brain/ + memory/ → retrieval/ + analysis/ + portrait/ → background/ → llm/

**循环依赖控制：** `llm/` 和 `core/` 之间通过 TYPE_CHECKING 延迟导入避免循环。

---

*最后更新：2026-06-14*
