# 初痕 (First Beat) CH Memory System

> **这是项目的唯一权威文档。** 其他所有 .md 都以此文档为准。Agent 启动时自动加载。
> 修改代码后必须同步更新本文档。最后修订 2026-06-19 (Phase 5 完成：ChromaDB 移除，Qdrant 唯一后端)。

---

## 1. 这是什么

一个 **本地优先的 AI Agent 记忆引擎**。给大模型装上长期记忆——存对话、检索回忆、建画像、主动发起话题。

- 156 个 .py 文件 · ~40,000 行源码 · ~13,400 行测试（60 文件）
- DeepSeek API (主 LLM) + Ollama 本地 (bge-m3 embedding / qwen2.5 实体抽取)
- FastAPI 服务 · 单用户单实例 · Benchmark 模式可选

---

## 2. 项目地图

```
d:\First Beat CH Memory System\
│
├── run.py                          # 启动入口
├── requirements.txt                # Python 依赖
├── prompt.txt                      # LLM 系统提示词（外部文件，PROMPT_FILE 环境变量指向）
├── verify_env.py                   # 环境验证：检查依赖/Ollama/目录结构 (238行)
├── CLAUDE.md                       # ← 你正在读的文件，唯一文档
│
├── Dockerfile                      # 容器构建
├── docker-compose.yml              # 容器编排
├── .dockerignore
├── .env.example                    # 环境变量模板
├── .env                            # 实际环境变量 (gitignore)
├── .gitignore
├── LICENSE                         # MIT License
├── skills-lock.json                # 技能版本锁定
│
├── README.md                       # 项目说明（中文）
├── README_EN.md                    # 项目说明（英文）
├── QUICKSTART.md                   # 快速上手（中文）
├── QUICKSTART_EN.md                # 快速上手（英文）
├── SETUP.md                        # 环境搭建（中文）
├── SETUP_EN.md                     # 环境搭建（英文）
├── ARCHITECTURE.md                 # 架构详解（中文）
├── ARCHITECTURE_DIAGRAM.md         # 架构图
├── ARCHITECTURE_EN.md              # 架构详解（英文）
├── AUTHOR.md                       # 作者/项目背景（中文）
├── AUTHOR_EN.md                    # 作者/项目背景（英文）
├── AUTHOR_PERSONAL.md              # 作者个人说明
├── BENCHMARK_CRITIQUE.md           # Benchmark 方法批判
├── CONTRIBUTING.md                 # 贡献指南（中文）
├── CONTRIBUTING_EN.md              # 贡献指南（英文）
├── EVEROS_INSIGHTS.md              # EverOS 设计洞见
├── SPEC_DRIFT.md                   # DriftTracker 规格说明
├── SPEC_MIGRATION.md                # 存储&推理基础设施迁移蓝图 (v1.6)
│
├── app/                            # ========== 主源码 ==========
│   ├── api/                        # FastAPI 层：接收请求，返回响应
│   │   ├── chat.py                 #   核心：/chat 端点是引擎闭环入口 (627行)
│   │   ├── system.py               #   系统状态、prompt 管理、健康检查
│   │   ├── memories.py             #   记忆 CRUD API
│   │   ├── openai.py               #   OpenAI 兼容层
│   │   ├── chat_history.py         #   对话历史 API：分页回传 JSONL 历史 (24行)
│   │   ├── consolidation.py        #   巩固 API：手动触发后台巩固 (18行)
│   │   ├── deps.py                 #   FastAPI 依赖注入：get_context/get_user (10行)
│   │   ├── health.py               #   健康检查端点 /health (9行)
│   │   └── app.py                  #   FastAPI app 工厂
│   │
│   ├── core/                       # 引擎核心：决策、上下文、基础设施
│   │   ├── context.py              #   ★ AppContext：服务容器，管理所有子系统和后台线程 (965行)
│   │   ├── circuit.py              #   ★ ChatCircuit：单次对话的处理管线 (748行)
│   │   ├── state.py                #   CognitiveState：引擎决策数据结构，LLM 看到的唯一接口 (303行)
│   │   ├── tools.py                #   LLM 工具定义（OpenAI 格式的 tool schemas）
│   │   ├── conflict.py             #   记忆冲突解决：事实矛盾检测 → stale 标记
│   │   ├── db.py                   #   DEPRECATED: Phase 3 退役桩，仅 close_all() no-op (Phase 5 删)
│   │   ├── helpers.py              #   工具函数：计时、trace、JSONL 缓存
│   │   ├── heartbeat.py            #   用户心跳追踪（共享给后台线程和 API）
│   │   ├── bottleneck.py           #   端到端延迟监控
│   │   ├── auth.py                 #   单用户认证（X-Chuhen-User header）
│   │   ├── feedback.py             #   错误报告 JSONL
│   │   ├── metadata.py.bak         #   REMOVED: DEPRECATED 旧的元数据提取 (2026-06-14)
│   │   └── user_context.py         #   多用户 AppContext 管理器
│   │
│   ├── memory/                     # 记忆存储层：Qdrant + JSONL（ChromaDB 已移除）
│   │   ├── qdrant.py               #   ★ QdrantService：唯一向量后端，量化+payload索引+LRU缓存 (~1400行)
│   │   ├── qdrant_cooccur.py       #   CoOccurrenceStore：共现独立 collection
│   │   ├── qdrant_hyperedge.py     #   HyperEdgeStore：超边独立 collection
│   │   ├── history.py              #   ChatHistory：对话历史 JSONL，内存缓存最近 500 条 (270行)
│   │   ├── working.py              #   工作记忆摘要：增量 LLM 摘要，替代注入全量历史 (165行)
│   │   ├── inverted.py             #   词→记忆ID 倒排索引，线程安全增量更新 (157行)
│   │   ├── temporal.py             #   TemporalPatternIndex：话题时间规律发现 (171行)
│   │   ├── tree.py                 #   话题树：从标签亲和图自动聚类 (179行)
│   │   ├── affinity.py             #   话题亲和图：标签级关联网络 (86行)
│   │   └── tag_index.py            #   标签嵌入索引：bge-m3 + 余弦最近邻查找 (140行)
│   │
│   ├── retrieval/                  # 检索管线：多路并行召回
│   │   ├── pipeline.py             #   ★ 核心：run_chat_retrieval() 14步 + retrieve_all() 9路并行 (703行)
│   │   ├── scoring.py              #   统一评分函数 compute_score() (40行)
│   │   └── reranker.py.bak         #   REMOVED: 嵌入余弦相似度重排序 (96行, UNUSED, 2026-06-14)
│   │
│   ├── llm/                        # LLM 适配层
│   │   ├── deepseek.py             #   ★ LLMClient：主 LLM 调用，10段消息结构，前缀缓存优化 (1051行)
│   │   ├── embed.py                #   ★ 嵌入层：local_embed()，四级缓存+请求合并，bge-m3 (390行)
│   │   └── local.py                #   本地 LLM 封装 (qwen2.5:7b)，用于摘要生成 (118行)
│   │
│   ├── portrait/                   # 画像系统：12维认知画像（替代旧 PersonalityStore）
│   │   ├── manager.py              #   ★ PortraitManager：PORTRAIT.md 生命周期管理，YAML+HTML注释 (541行)
│   │   ├── writer.py               #   ★ PortraitWriter：三层更新引擎（实时/浅层/深层）(726行)
│   │   ├── renderer.py             #   PortraitRenderer：画像→Prompt 片段（稳定8维+动态4维）(167行)
│   │   ├── state.py                #   PortraitEntry 状态机：pending→active→cooling→decayed (104行)
│   │   └── extractors.py           #   特征提取纯函数：关键词/置信度/情绪翻转/标签热度 (135行)
│   │
│   ├── background/                 # 后台节律系统
│   │   ├── consolidation.py        #   ★ ConsolidationEngine：空闲+4h/24h 周期认知巩固 (1071行)
│   │   ├── impulse.py              #   ★ ImpulseScheduler：4源泊松冲动信号→LLM内心独白 (565行)
│   │   └── lifecycle.py            #   后台线程生命周期管理：统一启停+崩溃重启 (115行)
│   │
│   ├── analysis/                   # 分析层：零 LLM 统计推断
│   │   ├── pattern_discovery.py    #   PatternDiscovery：4检测器，纯统计，6h 周期 (478行)
│   │   ├── predictor.py            #   BehaviorPredictor：n步马尔可夫链行为预测 (194行)
│   │   ├── emotion.py              #   Russell 2D 情绪环状模型 (180行)
│   │   ├── entity.py               #   实体抽取：Ollama qwen2.5:3b + 正则 + KeyBERT 三级 (58行)
│   │   ├── symmetry.py             #   PersonaSymmetry：双矩阵差分→AI理解盲区 (130行)
│   │   ├── drift.py                 #   DriftTracker：偏移率追踪(spend/frugal/drift)，纯规则 EMA (140行)
│   │   └── self_mirror.py           #   SelfMirror：AI自我镜像，检索相似情绪下的历史回应 (170行)
│   │
│   ├── brain/                      # 语义基础设施（NLP工具箱，纯函数，不调检索）
│   │   ├── semantic.py             #   7个公共函数：extract_tags/classify_intent/analyze_emotion... (477行)
│   │   ├── keywords.py             #   共享关键词常量 (56行)
│   │   ├── metrics.py              #   训练指标持久化 (138行)
│   │   ├── export_training_data.py  #   训练数据导出 (103行)
│   │   └── models.py               #   兼容壳，重导出 semantic.py (21行)
│   │
│   ├── tools/                      # 工具基础设施（允许任意层 import）
│   │   ├── dispatch.py             #   引擎内部工具：query_memory/analyze_pattern/count_memories (815行)
│   │   ├── atomic.py               #   原子文件写入：temp + os.replace() (33行)
│   │   ├── search.py               #   网页搜索（Bocha API）(54行)
│   │   └── workspace.py            #   文件系统操作：read/write/edit/list/grep (128行)
│   │
│   ├── config/
│   │   ├── settings.py             #   ★ 所有配置，环境变量优先，60+ 配置项 (249行)
│   │   └── paths.py                #   路径工具，新旧布局兼容 (26行)
│   │
│   └── models/
│       └── schemas.py              #   Pydantic 请求/响应模型 (52行)
│
├── tests/                          # ========== 测试 (62文件, 13,729行) ==========
│   ├── conftest.py                 #   共享 fixture：isolated_env, seeded_env
│   ├── test_*.py                   #   单元测试覆盖全模块
│   └── (共62个测试文件，覆盖 api/core/memory/retrieval/llm/portrait/analysis/brain/tools)
│
├── E2E/                            # ========== 端到端哨兵测试 (5链路, 89节点) ==========
│   ├── conftest.py                 #   E2E 共享 fixture
│   ├── test_write_path.py          #   链路1: 存进去了吗？(12节点)
│   ├── test_link2_retrieve.py      #   链路2: 找得到吗？(35节点)
│   ├── test_link3_cross_turn.py    #   链路3: 跨轮还记得吗？(9节点)
│   ├── test_link4_evolution.py     #   链路4: 时间过了退化了吗？(16节点)
│   └── test_background.py          #   链路0: 后台在干什么？(17节点)
│
├── integration/                    # ========== 集成测试 (6文件) ==========
│   ├── conftest.py
│   ├── test_int_consolidation.py
│   ├── test_int_history_flow.py
│   ├── test_int_retrieval_paths.py
│   ├── test_int_scoring.py
│   └── test_int_write_retrieve.py
│
├── LongMemEval/                    # ========== LongMemEval Benchmark ==========
│   ├── LONGMEMEVAL_REPORT.md       #   中文评估报告
│   ├── LONGMEMEVAL_REPORT_EN.md    #   英文评估报告
│   └── longmemeval_custom_results.jsonl  # 评估结果
│
├── audit/                          # ========== 审计报告存档 ==========
│   └── report_*.json               #   各轮审计 JSON 报告
│
├── htmlcov/                        # ========== 覆盖率报告 (自动生成) ==========
│
├── scripts/                        # ========== 运维脚本 ==========
│   ├── audit.py                    #   记忆审计 v4：8类生产数据检索层测试 (737行)
│   ├── check_conventions.py        #   代码规范检查：依赖方向/import 审查/SQLite 规范 (1015行)
│   ├── compare_reports.py          #   审计报告对比：两轮审计 diff 工具 (70行)
│   ├── verify_infra.py             #   Phase 0: Qdrant+vLLM 连通性验证 (270行)
│   │                               #   (migrate_to_qdrant.py / phase0_5_verify.py 已随 Phase 5 删除)
    ├── stress_test_1m.py            #   Phase 4: 百万级压力测试 性能基准
│   └── pre-push                    #   pre-push hook 备份：push 前跑 pytest tests/
│
├── .claude/                        # ========== Claude Code 配置 ==========
│   ├── settings.json               #   项目级设置
│   ├── settings.local.json         #   本地覆盖 (gitignore)
│   ├── mcp.json                    #   MCP 服务器配置
│   ├── skills/                     #   自定义技能
│   └── workflows/                  #   审计工作流定义
│       ├── ch-audit-full.js        #     全量审计工作流
│       ├── ch-audit-quick.js       #     增量审计工作流
│       ├── ch-audit-dimensions.md  #     审计维度定义
│       └── ch-audit-prompt.md      #     审计 Prompt
│
├── .hologram/                      # ========== Hologram 引擎状态 (运行时) ==========
│   ├── cache/pipeline_cache.json
│   ├── memory/MEMORY.md
│   ├── sessions/
│   └── timeline.db
│
└── data/                           # ========== 运行时数据 ==========
    ├── PORTRAIT.md                  #   画像文件（运行时读写，非文档）
    ├── qdrant/                      #   Qdrant 用户记忆向量库（本地嵌入式）
    ├── ai_qdrant/                   #   Qdrant AI 自我记忆向量库（本地嵌入式）
    │                                #   (旧 chroma*/behavior_chroma/personality_chroma 已随 Phase 5 弃用)
    ├── *.db                         #   SQLite：co_occurrence, entity_pairs, hyper_edges, timeline等
    ├── *.json                       #   JSON 缓存：affinity, tree, temporal_patterns, working_memory等
    ├── chat_history.jsonl           #   对话历史
    ├── store_failures.jsonl         #   入库失败记录
    └── cache/pattern_cache.json     #   模式发现缓存
```

---

## 3. 一次对话的完整数据流

这是理解整个系统的关键。当用户发一条消息时，以下是完整的调用链：

```
用户发送 "我最近想换Python了，Rust太难了"
│
├─ 1. API 层 [app/api/chat.py]
│   POST /chat → chat_endpoint() → 获取/创建 AppContext
│
├─ 2. 预处理 [app/core/circuit.py → ChatCircuit.run()]
│   ├─ intent = classify_intent(user_msg)           # 7类意图分类，~50μs
│   ├─ emotion = analyze_emotion(user_msg)           # Russell 2D valence+arousal
│   ├─ entities = extract_entities(user_msg)         # Ollama qwen2.5:3b + 正则
│   └─ gate = basal_ganglia_gate(intent, emotion)    # 门控：决定回应模式
│
├─ 3. 实时画像更新 [app/portrait/writer.py]
│   └─ realtime_update()                             # <100ms, 无LLM, 更新dim2/4
│
├─ 4. 检索 [app/retrieval/pipeline.py]
│   ├─ run_chat_retrieval()
│   │   ├─ 加载 DMN 预热缓存
│   │   ├─ retrieve_all() ─── 9路并行检索 ──────┐
│   │   │   ├─ path1: semantic (Qdrant 向量)      │
│   │   │   ├─ path2: keyword (倒排索引)          │
│   │   │   ├─ path3: tag (标签匹配)              │
│   │   │   ├─ path4: entity (实体匹配)           │
│   │   │   ├─ path5: temporal (时间模式)         │  ThreadPoolExecutor
│   │   │   ├─ path6: topic (话题树扩展)          │  (max_workers=8)
│   │   │   ├─ path7: attention (注意力漂移)       │
│   │   │   ├─ path8: fulltext (MatchText 全文)   │
│   │   │   ├─ path9: ai_memory (AI 自我记忆)     │
│   │   │   └─ + co_occurrence (共现扩展) ────────┘ ← 依赖前9路seen_ids
│   │   ├─ 注意力邻近度评分 (weighted-average embedding drift)
│   │   ├─ 近期性权重 (90天线性衰减)
│   │   └─ 纠错反馈 boost/downgrade
│
├─ 5. 认知状态组装 [app/core/state.py]
│   └─ CognitiveState 构建:
│       ├─ memories (检索结果+分数+权重)
│       ├─ portrait_stable (8维, message[0])
│       ├─ portrait_dynamic (4维, message[N+1])
│       ├─ impulses (冲动信号→自然想法)
│       ├─ pattern_observations (模式发现)
│       └─ session_context (工作记忆摘要+织线上下文)
│
├─ 6. LLM 调用 [app/llm/deepseek.py]
│   generate() → 10段消息序列 → POST DeepSeek API
│   │
│   │  message[0]:   system   ← 稳定系统提示词 (前缀缓存 >95%)
│   │  message[1..N]: user/assistant ← 历史对话
│   │  message[N+1]: system   ← 动态画像 + drift偏移 + session_context + now_hint
│   │  message[N+2]: assistant ← tool_call: retrieve_memories
│   │  message[N+3]: tool     ← 记忆 JSON
│   │  message[N+4]: assistant ← tool_call: natural_thoughts (如有冲动)
│   │  message[N+5]: tool     ← 冲动文本
│   │  message[N+6]: system   ← self_mirror + execute_directive
│   │  message[N+7]: assistant ← tool_call: pattern_observations (如有)
│   │  message[N+8]: tool     ← 模式观察
│   │  message[last]: user    ← 当前用户消息
│
├─ 7. 后处理 [app/api/chat.py]
│   ├─ 解析 LLM 响应 (DSML tool_call 格式兼容)
│   ├─ 流式返回给用户
│   └─ 触发生命周期钩子:
│       ├─ _store_conversation() → Qdrant 入库 (异步队列)
│       ├─ pulse() → 冲动信号检查
│       ├─ sync_turn() → 对话历史追加
│       └─ DMN ticker → 空闲检测→巩固调度
│
└─ 8. 记忆入库 [app/core/context.py]
    _store_conversation() 异步执行:
    ├─ 本地 LLM 摘要 (qwen2.5:7b)
    ├─ 标签提取 (bge-m3 KeyBERT)
    ├─ 实体抽取 (qwen2.5:3b + 正则)
    ├─ 情绪分析 (Russell 2D)
    ├─ embedding 计算 (bge-m3, 1024维)
    ├─ Qdrant upsert (user + AI 双写)
    ├─ 冲突检测 → mark_stale
    ├─ 情绪反转检测
    ├─ EntityPair / HyperEdge 更新
    └─ 重试3次, 失败写入 store_failures.jsonl
```

---

### 3.1 画像系统 (Portrait) 深入

12 维认知画像，存于 `data/PORTRAIT.md`（YAML frontmatter + HTML 注释元数据）。**这是理解画像代码的关键背景。**

#### 12 维度定义

```
用户.1 核心特征     ← 深巩固(24h) LLM重述  数据源: DMN 巩固统计 + PatternDiscovery + symmetry 盲区
用户.2 当前状态     ← 实时(<100ms) 引擎直写 数据源: UtteranceSpec 情绪/意图/关注焦点
用户.3 行为节律     ← 浅巩固(4h) LLM合成   数据源: BehaviorPredictor + TemporalPatternIndex + PatternDiscovery
用户.4 关系快照     ← 实时 引擎直写         数据源: RelationshipState (trust/closeness/familiarity/interaction_mode)
用户.5 兴趣图谱     ← 浅巩固(4h) LLM合成   数据源: Qdrant tag 分布 + TopicAffinity + TopicTree
用户.6 情绪图谱     ← 浅巩固(4h) LLM合成   数据源: 入库 emotion + PatternDiscovery + 情绪淡化统计

AI.1-6            ← 完全镜像用户维度         数据源: AI Qdrant 独立记忆库 + AI ConsolidationEngine
```

#### 条目四态操作（引擎删, LLM 写）

| 操作 | 谁做 | 触发 |
|------|------|------|
| **保留** | LLM 合成时保留 | 近 14 天有关联 tag 活动 |
| **修改** | LLM 合成时更新 | hot↔warm↔cooling 档位变化, 证据数波动 >30% |
| **删除** | **引擎直接删** | cooling>30天, 证据归零 |
| **新增** | LLM 合成时加入 | 新 tag 密度达标, 新情绪触发关联 |

**原则: 引擎做删除判断和分类, LLM 做文本合成。引擎不改文本, LLM 不做删除决策。**

#### 条目状态机

```
pending → active → cooling → decayed(删除)
  ↑         ↓         ↓
  └─────────┴─────────┘ (重新激活)
```

- `pending`: 实时层标记, 等浅巩固确认。**不注入 prompt。**
- `active`: 确认后注入, confidence ≥ 0.40
- `cooling`: >14 天未观察, **不注入 prompt**, 保留在文件
- `decayed`: >30 天未观察, 下次持久化时物理删除

#### 渲染→Prompt

画像在注入 LLM 前要剥离引擎元数据:

1. **过滤**: 去掉 pending / cooling / confidence<0.40 的条目
2. **剥离**: 去掉 `` `高 · 23条证据 · tags:...` `` 等引擎运维信息
3. **分组**: 8 稳定维度(usr1/3/5/6 + ai1/3/5/6) → `message[0]`, 4 动态维度(usr2/4 + ai2/4) → `message[N+1]`

#### 画像→检索 boost（唯一接触点）

画像不介入检索策略, 只在精排阶段加一个轻量 boost:

| 匹配 | boost |
|------|-------|
| hot 话题 | +0.2 |
| warm 话题 / 关注焦点 / AI 积累域 | +0.1 |
| 负向触发话题 | **-0.2** (降权不屏蔽) |

公式: `final_score = cosine_sim×0.5 + hit_conf×0.2 + source_weight×0.2 + portrait_boost×0.1`, boost ∈ [-0.2, +0.3]

---

## 4. 架构铁律

### 依赖方向（不可逆向）

```
api/ → core/ → memory/ → retrieval/ + analysis/ + portrait/ → background/ → llm/
                 ↑
brain/ ──────────┘  (语义基础设施, 仅依赖 llm/embed, 允许被任意层 import)
tools/ ──────────┘  (工具基础设施, 允许被任意层 import)
helpers.py, bottleneck.py, heartbeat.py ──┘  (基础设施, 允许被任意层 import)
```

### 核心设计决策

| 决策 | 含义 |
|------|------|
| **引擎决策 → LLM 执行** | LLM 不调内部记忆检索工具，不出记忆决策；LLM 仅调外部功能工具（search_web/read_file 等）。引擎决定"记住什么/检索什么/何时巩固"，LLM 只做文本合成 |
| **原文永不压缩** | Qdrant 存原文，摘要只是索引。prompt 中传原文，不做有损压缩 |
| **时间不做衰减** | 用 hit_count 做权重，不用时间衰减函数。记忆靠「被用了多少次」自然排序 |
| **1对1服务** | 一个 AppContext 服务一个用户，不同用户完全隔离 (data/users/{name}/) |
| **无 pickle** | 任何场景都不用 pickle |

### 存储规范

```
向量+元数据 → Qdrant (本地嵌入式/远程服务器)  → app/memory/qdrant.py
流式/追加   → JSONL 追加写入                 → open(path, "a")
缓存        → JSON (仅模块内部, 不共享)       → 各自的 _cache.json
```

- **唯一存储后端为 Qdrant** (Phase 5：ChromaDB 已移除)：向量 int8 量化 + payload 索引 + LRU 20K embedding 缓存
- `QDRANT_URL` 留空 → 本地嵌入式文件模式（`data/qdrant`, `data/ai_qdrant`）；设 `http(s)://` → 连服务器（docker 注入）
- **无 `STORAGE_BACKEND` 开关、无 ChromaDB 回退**；`chromadb` 依赖已删，`requirements.txt` 用 `qdrant-client`
- **CoOccurrence / HyperEdge 为独立 Qdrant collection**（`qdrant_cooccur.py` / `qdrant_hyperedge.py`）
- **Phase 3 已删 SQLite 存储**：cooccur.py、entity_pair.py、hyperedge.py 已移除
- **AppContext 存储属性**：`memory_service` / `ai_memory_service`（旧 `chroma_service` 命名已废）
- **向量检索**用 `client.query_points(...).points`（qdrant-client 1.10+ 弃用 `search`）；点 ID 必须 UUID/整数
- **追加写入的数据（对话历史、错误报告）继续用 JSONL**
- **模块私有的缓存 JSON 可以保留**

---

## 5. 🛑 红线：绝对不能碰的区域

> 改了会静默崩溃，不会报错。不主动修改，除非用户明确要求。

### 红线 1：LLM Prompt 消息结构

**位置**: `app/llm/deepseek.py` — `generate()` 方法

```
message[0]:   system        ← stable prompt + _CORE_RULES + portrait_stable(8维)
message[1..N]: user/assistant ← 历史对话
message[N+1]: system        ← portrait_dynamic(4维) + session_context + now_hint()
message[N+2]: assistant     ← tool_call: retrieve_memories
message[N+3]: tool          ← 记忆 JSON 数组
message[N+4]: assistant     ← tool_call: natural_thoughts  (有冲动才加)
message[N+5]: tool          ← 冲动文本
message[N+6]: system        ← execute_directive
message[N+7]: assistant     ← tool_call: pattern_observations  (有内容才加)
message[N+8]: tool          ← 模式观察
message[last]: user         ← 当前消息
```

**不能动**: `_CORE_RULES` 常量、tool role 注入模式、消息顺序。动了前缀缓存命中率暴跌。

### 红线 2：Qdrant Payload Schema

**位置**: `app/memory/qdrant.py` — `QdrantService.add_memory()` 方法

**payload 字段名不可改名/删除**: `timestamp, hit_count, heat, embed_model, stale, archived, superseded_by, storage_complete, source, summary, tags, entities, date_tag`

- `storage_complete` flag 控制后台队列是否重试入库（`mark_storage_complete(memory_id)` 置位）
- collection 名 `"memories"` / `"ai_memories"`（settings `MEMORIES_COLLECTION` / `AI_COLLECTION`），dispatch.py 硬编码 `"memories"`

### 红线 3：后台线程

**位置**: `app/core/context.py` + `app/background/`

| 线程 | 启动位置 | 静默死亡后果 |
|------|---------|------------|
| `store_queue_{dir}` | context.py | 对话不再自动入库 |
| `impulse_consumer_{dir}` | context.py | 引擎不再主动开口 |
| `dmn_ticker_{dir}` | context.py | 巩固+画像更新全部停止 |
| `ai_desensitize_{dir}` | context.py | AI 情绪淡化停止 |
| 4个冲动源泊松线程 | impulse.py | 冲动信号不再产生 |

**不能动**: daemon=True 属性、`_stop_event` 检查逻辑、DMN ticker 合并逻辑。

### 红线 4：Settings 承重墙

**位置**: `app/config/settings.py`

| 配置 | 改了会怎样 |
|------|-----------|
| `LLM_BASE_URL` / `LLM_MODEL` | 前缀缓存策略可能完全失效 |
| `OLLAMA_EMBED_MODEL = "bge-m3"` | 旧记忆 embedding 全部作废，需全量重建 |
| `BENCHMARK_MODE` | 非 benchmark 开它会降低回复质量 |
| `DATA_DIR` | 所有路径全变 |
| `WORK_MEMORY_TOKEN_BUDGET` | 改太小丢对话上下文 |
| `QDRANT_QUANTIZATION` | Phase 4 int8 量化开关；关闭→向量 4× 膨胀，开启→搜索精度微降 |
| `QDRANT_EMB_CACHE_MAX` | 改太小→注意力评分退化；改太大→内存压力（默认 20K ≈ 80MB） |

### 红线 5：Embedding 和实体抽取

- `local_embed()` 依赖 Ollama `bge-m3`，9 路检索全部依赖它
- 实体抽取依赖 Ollama `qwen2.5:3b`，改模型→实体对/超边质量下降
- Phase 4 embedding 缓存：`_emb_cache` 为 `OrderedDict` LRU，`_emb_cache_put()` 统一写入口。外部不要直接写 `_emb_cache` 或在外面加 `_emb_cache_lock`→死锁

### 红线 6：E2E 哨兵测试

5 条链路是系统健康的体温计。E2E 失败 = 核心链路断了。

```
E2E/test_write_path.py       — "存进去了吗？"
E2E/test_link2_retrieve.py   — "找得到吗？"
E2E/test_link3_cross_turn.py — "跨轮还记得吗？"
E2E/test_link4_evolution.py  — "时间过了退化了吗？"
E2E/test_background.py       — "后台在干什么？"
```

---

## 6. 编码规范

### 必须
- 新存储逻辑走 Qdrant（`app/memory/qdrant.py`）—— CoOccurrenceStore / HyperEdgeStore / QdrantService
- 测试后调用 `from app.core.db import close_all; close_all()`（兼容桩，Phase 5 移除）
- 异常捕获至少用 `except Exception:`，禁止裸 `except:`
- 遵循依赖方向，不引入反向 import

### 禁止
- `sqlite3.connect()` 直连（Phase 3 后 SQLite 已退役）
- 跨模块共享的 JSON 文件做结构化存储
- pickle
- 在 embedding 缓存逻辑外加锁
- 修改 E2E conftest 的 fixture 初始化逻辑

### 线程安全
- 所有共享状态用 `threading.Lock()` 保护
- 文件写入用 `atomic_write()` (temp + os.replace)
- JSONL 追加用 `atomic_append()`

---

## 7. 运行命令

```bash
python run.py                          # 启动服务 (端口 8000)
python -m pytest tests/ -q             # 单元测试
python -m pytest E2E/ -v               # E2E 测试
python -m pytest integration/ -v       # 集成测试
python scripts/audit.py                # 记忆审计 (8类, 全量)
python scripts/audit.py --quick        # 快速审计
python scripts/check_conventions.py    # 代码规范检查
python scripts/compare_reports.py      # 审计报告对比
python scripts/verify_infra.py         # Phase 0: Qdrant+vLLM 连通性验证
python scripts/verify_infra.py --quick #   仅快速验证
python scripts/stress_test_1m.py           # Phase 4: 百万级压力测试 (默认 10K)
python scripts/stress_test_1m.py --count 100000 --server http://localhost:6333  # 10万条服务器模式
BENCHMARK_MODE=true python run.py      # Benchmark 模式
```

### Pre-push Hook

`git push` 时自动跑 `pytest tests/`，失败则阻止推送。安装/重装：

```bash
cp scripts/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

### 审计系统

```bash
/ch-audit              # 全量审计 (10维度 × 5路并行 Agent)
/ch-audit --quick      # 增量审计 (仅 git diff)
/ch-audit --dim=1,4,7  # 聚焦审计
```

---

## 8. 当前状态

### 最近完成 (2026-06-19)

- ✅ **Phase 5 完成：ChromaDB 彻底移除，Qdrant 成为唯一后端**：
  - 删 `app/memory/chroma.py`、`requirements.txt` 去 `chromadb` 加 `qdrant-client`、删 `STORAGE_BACKEND`/`TEST_BACKEND` 开关
  - `context.py` 单一 Qdrant 路径；`chroma_service`/`ai_chroma_service` → `memory_service`/`ai_memory_service` 全项目重命名
  - 删 `migrate_to_qdrant.py`、`phase0_5_verify.py`（chroma 时代脚本）；audit.py / check_conventions.py / verify_env.py / run.py 切 Qdrant
  - **修复此前从未端到端跑通的 Qdrant 路径隐藏 bug**：
    - Windows 本地模式探测（盘符 `C:` 的冒号误判为 URL）→ 改用 scheme 前缀判断，`QDRANT_URL` 默认空走本地文件模式
    - qdrant-client 1.10+ 删除 `client.search`，改 `query_points(...).points`（语义检索之前静默失败）
    - 补全 `QdrantService.count/clear_all/_local_index_build/update_entity_co_counts`；修 `mark_storage_complete(memory_id)` 与 `add_memory` 签名（chroma 兼容 + 红线 2 payload）；`clear()` 改按 ID 删（本地模式 delete_collection 重建不清数据）
  - 测试：tests 984 通过（仅 10 个 `weave_context` 为 master 上既存失败）、integration 19 通过、audit_pipeline 29 通过；check_conventions 15 通过 0 错误
  - **注意**：原 `data/chroma*` 旧数据按用户决定丢弃，Qdrant 从空开始

- ✅ **Phase 4 百万级硬骨头完成**：
  - **量化**: `scalar_int8` 量化配置，4GB→1GB 向量存储。`_build_quantization_config()` 应用于 3 类 collection。
  - **Payload 索引**: 13 字段索引（memories）、3 字段（co_occurrence）、3 字段（hyper_edges），全部幂等。
  - **Embedding 缓存 LRU**: `OrderedDict` LRU 淘汰，`last_hit_time DESC` 分批 scroll，`_emb_cache_put()` 统一写入口，上限 20K。
  - **压力测试**: `scripts/stress_test_1m.py` — 可配置规模（1K~1M），P50/P95/P99 对照 §10.2 阈值判定。
  - **本地 Payload 索引**: `_LocalPayloadIndex` (~200行) — Python 侧内存索引，补偿 Qdrant 本地模式无服务端索引。热数据过滤 0.3ms（26x vs 暴力扫描），增量维护 + 启动异步构建。
- ✅ **Phase 3 SQLite 迁移完成**：CoOccurrenceStore / HyperEdgeStore 替代 cooccur.py / entity_pair.py / hyperedge.py
  - `app/memory/qdrant.py` 新增 `CoOccurrenceStore` (~280行)、`HyperEdgeStore` (~280行)
  - cooccur.py、entity_pair.py、hyperedge.py 已删除
  - `entity_co_counts` 入库时预计算存入 Qdrant payload（替代 EntityPairTracker）
  - context.py 初始化全部切换到新 stores（ChromaDB 回退路径自动创建本地 Qdrant 客户端）
  - PersonaSymmetry 通过 `export_for_symmetry()` 透明接入
  - db.py 保留兼容桩 (`close_all` no-op)，Phase 5 删除
  - 1065 tests pass (6 预存 flaky，与改动无关)
- ✅ **Phase 2 杀全量模式完成**：`_translate_filter()` (8种运算符)、`_QdrantCollectionCompat` ChromeDB→Qdrant API 适配器、删 `bm25_fulltext.py` (Qdrant MatchText 替代)、pipeline.py 全路径 API 翻译、context.py _collection 适配、dispatch.py 兼容。1084 tests pass。
- ✅ **Phase 1 QdrantService 完成**：`app/memory/qdrant.py` (550行→~980行)、context.py STORAGE_BACKEND 切换、dispatch.py Qdrant 兼容、portrait/writer.py PersonaSymmetry bugfix。
- ✅ **Phase 0.5 原型验证完成**：V2 HNSW 召回率 1.0、V3 中文标签匹配 8/8、V4 子串匹配行为已文档化、V5 CoOccurrence 本地模式通过、V6 embedding 签名兼容。V1 需 Ollama+vLLM 服务（当前不可达）
- ✅ **Phase 0 infra 搭建完成**：docker-compose + settings.py 切换开关 + verify_infra.py + migrate_to_qdrant.py

### 计划中

- ⏳ **asyncio 化**：Qdrant 迁移完成后独立执行。目标——`pipeline.py` 的 `ThreadPoolExecutor` 9 路并发 → `asyncio.gather()`、`embed.py` 同步 HTTP → `httpx.AsyncClient`、`context.py` 后台线程 → asyncio Task。**不在迁移 spec 范围内**，两个工作解耦。vLLM HTTP 层的 `embed.py`/`local.py` 改写优先用 `httpx.AsyncClient`（新代码不引入同步 HTTP 债务），外部暂时 `asyncio.to_thread()` 包一层

### 最近完成 (2026-06-14)

- ✅ **8项性能修复**：hit-count 批量写入（200次→1次 ChromaDB 往返）、AI 入库非阻塞（fire-and-forget）、Executor 生命周期治理（消除线程泄漏）、队列磁盘轮询降频（1s→30s）、stats() 计数器替代全量遍历、N-gram 缓存早停优化、死代码移除（reranker/metadata→.bak）、存储步骤并行化（asyncio.gather）
- ✅ 删 GitHub Actions CI，换 pre-push hook（push 前本地跑 pytest，挂了不让推）
- ✅ 偏移率追踪 + 自我镜像：2 个新特性落地 (DriftTracker + SelfMirror)
- ✅ Phase 4 退役：`app/personality/` 删除，画像系统完全接管
- ✅ 共现/实体对/超边 v3 迁移：JSON → SQLite
- ✅ 审计 v4：Ollama 降级跳过 + 清理旧训练脚本
- ✅ 审计修复第2轮：14文件/8批次，加权分 70.5%→73.3%

### 已知问题（不要主动修）

- `app/tools/dispatch.py:695` — 裸 `except: continue`，吞 KeyboardInterrupt
- 全项目约 227 处 `except Exception` — 大部分是后台线程防御性捕获，有意为之，不要批量替换

### 已修复

- ✅ co_occurrence 路径不一致 → CoOccurrenceTracker.export_for_symmetry()
- ✅ PersonaSymmetry 直接读文件 → 改用 from_dicts=True

---

*Agent 启动时自动加载。本文档是项目约定的唯一权威来源——如果代码与本文档冲突，以本文档为准。修改红线区域或约定前，必须跟用户确认。*
