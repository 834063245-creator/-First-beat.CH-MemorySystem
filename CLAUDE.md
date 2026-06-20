# 初痕 (First Beat) CH Memory System

> **这是项目的唯一权威文档。** 其他所有 .md 都以此文档为准。Agent 启动时自动加载。
> 修改代码后必须同步更新本文档。最后修订 2026-06-21 (Embedding 统一切换 bge-m3→qwen_embed，检索+入库+注入全用同一空间，3584维，1007 tests green)。

---

## 1. 这是什么

一个 **本地优先的 AI Agent 记忆引擎**。给大模型装上长期记忆——存对话、检索回忆、建画像、主动发起话题。

- 156 个 .py 文件 · ~40,000 行源码 · ~13,400 行测试（60 文件）
- DeepSeek API (主 LLM) + qwen_embed 本地 (3584维，纯 Python+numpy) + Ollama 本地 (qwen2.5 实体抽取/摘要)
- **新增：本地 CVEC 残差注入模式** — `LOCAL_LLM_MODE=true` 时引擎走本地 qwen2.5 + 16 模块分层 steering，不走 API
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
│   │   └── tag_index.py            #   标签嵌入索引：qwen_embed + 余弦最近邻查找 (140行)
│   │
│   ├── retrieval/                  # 检索管线：多路并行召回
│   │   ├── pipeline.py             #   ★ 核心：run_chat_retrieval() 14步 + retrieve_all() 9路并行 (703行)
│   │   ├── scoring.py              #   统一评分函数 compute_score() (40行)
│   │   └── reranker.py.bak         #   REMOVED: 嵌入余弦相似度重排序 (96行, UNUSED, 2026-06-14)
│   │
│   ├── llm/                        # LLM 适配层
│   │   ├── deepseek.py             #   ★ LLMClient：主 LLM 调用，10段消息结构，前缀缓存优化 (1051行)
│   │   ├── steering.py             #   ★ SteeringInjector：本地 CVEC 残差注入引擎 (文本路径, 528行)
│   │   ├── steering_direct.py       #   ★★ SteeringDirect：绕过文本中转，结构化数值直出残差向量 (530行, NEW 2026-06-21)
│   │   ├── embed.py                #   ★ 嵌入层：local_embed()，两级缓存，qwen_embed 3584维 (193行，v3: 切自 bge-m3)
│   │   ├── qwen_embed.py           #   ★ qwen2.5 独立嵌入模型，查表 351x 加速，检索/入库/注入统一后端 (221行)
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
│   │   ├── impulse.py              #   ★ ImpulseScheduler：5源泊松冲动信号→LLM内心独白 [连线⑤]
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
│   ├── verify_cognitive_wiring.py  #   认知五连线 smoke 验证 (20 项检查)
│   ├── cognitive_trace.py          #   认知追踪器：场景推演+冲突检测+LLM 影响估算
│   ├── steering_phase5_inject.py   #   Phase 5: 引擎注入闭环（文本级），build_steering_segments
│   ├── steering_phase6_embed_inject.py # Phase 6: 嵌入层注入对比（EMBED vs TEXT，已证伪）
│   ├── steering_phase7_layer2_cvec.py  # ★ Phase 7: 残差分层注入（llama_set_adapter_cvec）
│   ├── steering_phase7_debug.py        # Phase 7: 调试/扫参（α + 层号）
│   ├── steering_phase8_layered.py      # ★★ Phase 8: 16 模块分层注入（15 向量 → L3-26）
│   ├── verify_steering_direct.py   # ★ SteeringDirect smoke 验证：7 项检测，概念向量语义+trajectory
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

> **2026-06-21 新增本地 CVEC 模式**：`LOCAL_LLM_MODE=true` 时，步骤 6 走 `steering.py:SteeringInjector.generate()` 替代 DeepSeek API。
> 引擎产出 16 模块短文本 → qwen_embed → CVEC 分层注入到本地 qwen2.5 残差流，不走 prompt 拼装。

这是理解整个系统的关键。当用户发一条消息时，以下是完整的调用链：

```
用户发送 "我最近想换Python了，Rust太难了"
│
├─ 1. API 层 [app/api/chat.py]
│   POST /chat → chat_endpoint() → 获取/创建 AppContext
│   ┌─ 远程模式 (默认): user_ctx.llm_client.generate(cognitive_state=utterance_spec)
│   └─ 本地 CVEC 模式 (LOCAL_LLM_MODE=true): user_ctx.steering_injector.generate(user_msg, utterance_spec)
│
├─ 2. 预处理 [app/core/circuit.py → ChatCircuit.run()]
│   ├─ intent = classify_intent(user_msg)           # 7类意图分类，~50μs
│   ├─ emotion = analyze_emotion(user_msg)           # Russell 2D valence+arousal
│   ├─ entities = extract_entities(user_msg)         # Ollama qwen2.5:3b + 正则
│   └─ basal_ganglia_gate(...)                       # [连线②] 画像情绪→语气收敛
│                                                   # [连线④] 行为预测→响应模式预调
│
├─ 3. 实时画像更新 [app/portrait/writer.py]
│   ├─ realtime_update()                             # <100ms, 无LLM, 更新dim2/4
│   └─ [连线①] 消费 error_reports → 关联条目降 confidence   # 2026-06-20
│
├─ 4. 检索 [app/retrieval/pipeline.py]
│   ├─ [连线④] BehaviorPredictor.predict() → mirror_prediction  # 缓存供门控用
│   ├─ [连线③] compute_portrait_boost_map() → weave_context 阈值放宽
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
│       ├─ pulse() → 冲动信号检查 [连线⑤ 画像探索冲动参与]
│       ├─ sync_turn() → 对话历史追加
│       └─ DMN ticker → 空闲检测→巩固调度
│
└─ 8. 记忆入库 [app/core/context.py]
    _store_conversation() 异步执行:
    ├─ 本地 LLM 摘要 (qwen2.5:7b)
    ├─ 标签提取 (qwen_embed KeyBERT)
    ├─ 实体抽取 (qwen2.5:3b + 正则)
    ├─ 情绪分析 (Russell 2D)
    ├─ embedding 计算 (qwen_embed, 3584维)
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
| 5个冲动源泊松线程 | impulse.py | 冲动信号不再产生 [连线⑤] |

**不能动**: daemon=True 属性、`_stop_event` 检查逻辑、DMN ticker 合并逻辑。

### 红线 4：Settings 承重墙

**位置**: `app/config/settings.py`

| 配置 | 改了会怎样 |
|------|-----------|
| `LLM_BASE_URL` / `LLM_MODEL` | 前缀缓存策略可能完全失效 |
| `OLLAMA_EMBED_MODEL = "qwen_embed"` | v3: 已切到 qwen_embed，不再依赖 Ollama embedding |
| `BENCHMARK_MODE` | 非 benchmark 开它会降低回复质量 |
| `DATA_DIR` | 所有路径全变 |
| `WORK_MEMORY_TOKEN_BUDGET` | 改太小丢对话上下文 |
| `QDRANT_QUANTIZATION` | Phase 4 int8 量化开关；关闭→向量 4× 膨胀，开启→搜索精度微降 |
| `QDRANT_EMB_CACHE_MAX` | 改太小→注意力评分退化；改太大→内存压力（默认 20K ≈ 80MB） |

### 红线 5：Embedding 和实体抽取

- `local_embed()` 依赖 `qwen_embed`（纯 Python+numpy，3584维），9 路检索+入库+注入全部依赖它
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
python run.py                          # 启动服务 (端口 8000, 默认远程 DeepSeek API)
LOCAL_LLM_MODE=true python run.py      # 启动服务 (本地 CVEC 模式, qwen2.5 + 残差注入)
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
python scripts/verify_cognitive_wiring.py   # 认知五连线 smoke 验证 (20 项，改完连线就跑)
python scripts/cognitive_trace.py            # 认知追踪器：造场景→推演→冲突检测→估算 LLM 影响
python scripts/verify_steering_direct.py       # SteeringDirect smoke 验证 (7项)
python scripts/calibrate_trajectory.py --module gate_tone        # Trajectory 标定: dry-run 单模块
python scripts/calibrate_trajectory.py --priority high --dry-run   # Trajectory 标定: 高优先级 6 模块
python scripts/calibrate_trajectory.py --module gate_tone --scan-alpha  # Trajectory 标定: 扫 alpha
python scripts/calibrate_trajectory.py --module gate_tone --live      # Trajectory 标定: live 实际生成
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

### 最近完成 (2026-06-21) — 直接向量注入引擎落地 ⭐⭐

- ✅ **`app/llm/steering_direct.py` 落地** — 530 行，绕过文本中转，模块结构化数值直出残差向量
  - **ConceptVectorBuilder** — 4 种零训练向量构造方法：
    - 方法一：锚点插值 (标量 → 向量, emotion/trust/formality)
    - 方法二：概念方向 (变化量 → 向量, drift/predictor)
    - 方法三：类别查表 (离散值 → 向量, tone/mode/intent)
    - 方法四：Tag 混合 (标签列表 → 向量, portrait/interest/impulse)
  - **TrajectoryShaper** — 5 种 shape 函数展开基向量到 28 层 (uniform/gradient/early/late/peak)
  - **16 ModuleSteeringConfig** — 每模块独立配置 (层范围 + shape + α)
  - **16 提取器** — 各从 UtteranceSpec 提取结构化值，零文本产出
  - 初始化 1147ms 预计算 ~100 token 嵌入，运行时 19.9ms 构建全 trajectory
- ✅ **`app/llm/steering.py` 集成** — 新增 `_setup_cvec_direct()` / `_generate_with_cvec_direct()`，双模式分支
- ✅ **`app/config/settings.py`** — 新增 `STEERING_DIRECT` 配置开关（默认关闭，实验特性）
- ✅ **`scripts/verify_steering_direct.py`** — 7 项 smoke 验证全通过：
  - 情绪 valence 方向分离度 0.50，arousal 分离度 0.53
  - Trust 锚点插值完美对称
  - 5 种 tone 类别语义结构合理
  - 7 种 trajectory shape 验证通过
  - 端到端：26/28 层活跃，19.9ms 构建
  - Direct vs Text 向量相似度分析完成
- ✅ **1007 tests passed, 0 failed** — 零回归

### 最近完成 (2026-06-21) — ChatML 格式修复 + Trajectory 标定首轮 ⭐

- ✅ **ChatML prompt 格式修复** — `steering.py` 全部 6 个 generate 方法切到 Qwen2.5 原生的 `<|im_start|>` / `<|im_end|>` 格式 + stop tokens
  - 旧格式 `用户消息: ...\n回复:` 触发多轮对话循环（模型生成假用户消息+回复无限循环）
  - 新格式用 ChatML system/user/assistant 角色帧定，`stop=["<|im_end|>", "<|im_start|>"]` 精确截断
  - 三路对比验证：无steering(110s) / 文本(84s) / direct(74s) 全部干净单轮回复，零循环
- ✅ **`scripts/calibrate_trajectory.py` 升级** — 410 行，完整 live shape sweep + alpha scan + cross-module analysis
  - `--compare` 三路对比 (no-steering / text / direct)
  - `--module X --live` 单模块扫 6 shapes，实际生成回复供人工对比
  - `--cross-module` 全模块层贡献分析 + 共居冲突检测
  - `--scan-alpha` 固定 shape 扫 α 范围
- ✅ **gate_tone 标定完成** — 实验数据 driven：
  - 7 shape × 1 scenario 实测对比：`gradient_up` 产生最强共情开头（"不要怀疑自己"）
  - `late` 产生温暖结尾（"加油！"）但 gradient_up 整体更优
  - 配置变更：shape `late` → `gradient_up`，层范围 L24-28 → L16-28
- ✅ **relationship_state 标定完成**：
  - 4 shape 实测对比：`gradient_down` 最优（"不要怀疑自己"）——与 gate_tone 相反
  - 结论：关系感知在浅中层编码（gradient_down 浅强深弱），门控语气在深层编码（gradient_up 深强浅弱）
  - 配置变更：shape `gradient_up` → `gradient_down`
- ✅ **portrait_emotion 确认** — 当前 `peak:12:4` + L8-15 合理，中层情绪编码位置正确
- ✅ **cross-module 层贡献分析** — 15 模块层分布健康：L1-2 空，身份 L3-5，记忆 L5-12，情绪 L8-15，关系 L18-27，门控 L16-28。**零强冲突**（所有 co-inhabiting 向量 cos < 0.2）
- ✅ **1007 tests passed, 0 failed** — 零回归

### 待标定模块 (优先级中/低)

- ⏳ portrait_identity (shape=early L3-5) — 当前合理，待验证
- ⏳ impulse_signal (shape=peak:10:3 L8-15) — 冲动激活场景少，低优先级
- ⏳ drift_context (shape=gradient_up L15-22) — drift 只在情绪场景激活，中优先级
- ⏳ memories 1-5 (shape=gradient_down L5-12) — 当前合理，低优先级
- ⏳ self_mirror / behavior_predictor / portrait_interest — 低优先级

- ✅ **检索管线全量切 qwen_embed** — `app/llm/embed.py` 从 Ollama HTTP (bge-m3) 切换到纯 Python+numpy qwen_embed (3584维)
  - 去掉 Ollama HTTP 客户端、请求合并器、n-gram 近似缓存（~200行删除）
  - 保留请求级缓存 + LRU 全局缓存
  - 速度 351x（3247 vs 9 emb/s），Recall@5=100% 持平 bge-m3
- ✅ **Qdrant collection 维度 1024→3584** — qdrant.py / qdrant_cooccur.py / qdrant_hyperedge.py 全部更新
- ✅ **settings.py 更新** — `DEFAULT_EMBED_MODEL="qwen_embed"`，EMBED_MODELS 新增 qwen_embed 条目
- ✅ **全项目注释清理** — 所有 bge-m3 引用替换为 qwen_embed（20+ 文件）
- ✅ **conftest mock 适配** — mock `_embed_via_qwen` 替代 `_embed_via_ollama`，维度 3584
- ✅ **test_embed.py 重写** — 移除 n-gram/合并器测试，维度 3584
- ✅ **1007 tests passed, 0 failed** — 零回归，embed 测试从 skip 变真实运行

### 最近完成 (2026-06-21) — 残差注入引擎落地 ⭐

- ✅ **`app/llm/steering.py` 落地** — 310 行，`build_steering_segments()` + `SteeringInjector` 单例
  - `build_steering_segments(utterance_spec)` → 16 模块各产短中文，从 UtteranceSpec 直接提取
  - `SteeringInjector.generate(user_message, utterance_spec)` → qwen_embed → CVEC 分层注入 → 本地生成
  - `SteeringInjector.generate_stream(...)` — 流式生成，与 `LLMClient.generate_stream()` 接口兼容
  - MODULE_LAYER_MAP: 16 个模块 × 层号 + α 值（来自 Phase 8 实验标定）
  - 线程安全：单例 + Lock 保护 CVEC buffer，单用户自然无争用
- ✅ **`app/api/chat.py` 本地模式分支** — 两个端点各加 `if local_llm_mode` 分支
  - 流式 (`/chat/stream`): `_steering_stream()` async wrapper → SSE 同格式输出
  - 非流式 (`/chat`): `steering_injector.generate()` 直接返回，跳过工具调用循环
  - 本地模式 v1 暂不支持工具调用（纯对话），后续补
- ✅ **`app/config/settings.py` 新增 5 配置项** — `LOCAL_LLM_MODE` / `QWEN_GGUF_PATH` / `STEERING_ENABLED` / `STEERING_STRENGTH` / `MINGW_BIN_DIR`
- ✅ **`app/core/context.py`** — AppContext 条件创建 `steering_injector`
- ✅ **Smoke 验证通过** — CVEC 注入后回复从"建议从简单项目开始"变"我理解你的感受"，共情明显提升
- ✅ **994 tests passed, 0 failed** — 零回归

### 最近完成 (2026-06-20)

- ✅ **认知层五条连线落地** — 画像·反馈·预测·门控·冲动不再孤立运行：
  - **连线① Feedback→Portrait**: 用户纠错("你记错了")→ `feedback.py:get_recent_corrected_ids()` → `PortraitWriter.realtime_update()` 降关联条目 confidence + 标 PENDING
  - **连线② Portrait→Gate**: 画像 usr6 情绪趋势 → `basal_ganglia_gate()` 收敛语气 (warm→soft) + 拉高 formality
  - **连线③ Portrait→weave_context**: 画像热度 tag → `compute_portrait_boost_map()` → `weave_context()` 分层阈值乘 tag_multiplier → 热点记忆更容易进 fact 层
  - **连线④ Predictor→Gate**: `BehaviorPredictor` 预测 → `basal_ganglia_gate(mirror_prediction=)` 预调 response_mode
  - **连线⑤ Portrait→Impulse**: 新增第 5 泊松冲动源 `source_portrait_curiosity()` — 画像 usr2/usr5/usr6 → 定向探索冲动
  - 改动 ~150 行，5 文件 (feedback.py / writer.py / circuit.py / impulse.py / context.py)，165 已有测试全绿
  - **冲突分析**：② 和 ④ 都修改 tone，但②先执行收敛后④的 caring 分支(需 tone==warm)不再触发 — 合理，长期低落时不应强行 caring
  - **认知追踪器** `scripts/cognitive_trace.py`：造假用户→逐线推演→冲突检测→估算 LLM prompt 变化。以后改阈值/关键词/boost 值前先跑一把看效果。
  - **Smoke 验证** `scripts/verify_cognitive_wiring.py`：20 项端到端检查，改完连线就跑。

- ✅ **残差分层注入 Phase 7/8/9 验证通过** — 引擎向量通过 `llama_set_adapter_cvec` 打入残差流，零 C++ 改动：
  - **Phase 7** (4 场景 × 4 条件): 单向量注入，CVEC-L2 最优——回复从"去学 Rust 教程"变"换到 Python 也完全没问题"
  - **Phase 8** (4 场景 × 2 条件): 16 模块 15 条向量分层注入 L3-26，全通
  - **Phase 9** (全 28 层注入): 同一向量 ×28 层不会炸——LayerNorm 每层兜底。ALL×28 比 L2-only 效果**更好**
  - **Steering Trajectory 概念**: 每个模块在 28 层上各有一个不同的向量 → 16 模块 × 28 层 = 448 个可独立调节的 steering knob
  - **prompt vs trajectory**: prompt 是广播（全层同一方式 attend），trajectory 是精确制导（浅层植身份/中层偏语气/深层约决策/末层调措辞）
  - **脚本**: `steering_phase7_layer2_cvec.py` / `steering_phase8_layered.py` / `steering_phase7_debug.py`
  - **下一步**: 模块直接产出残差向量（绕过文本中转）→ 引擎结构化数值 → linear projection → d_model × 28 layers

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

- ⏳ **Trajectory 标定继续** — 首轮完成高优先级 3 模块(gate_tone/portrait_emotion/relationship_state)，剩余 12 模块待标定。每模块需 ~10 分钟 live sweep（GTX 1060 6GB, 8 tok/s）。
- ⏳ **本地模式工具调用** — 目前 v1 纯对话（ChatML 格式），本地 qwen2.5 自带 tool call 能力待接入 `chat.py` 的 `for tool_round in range(2)` 循环
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

## 9. 认知层五连线 — 架构说明

> **五条连线让画像系统从"只记录不参与"变成引擎决策的活跃参与者。**

### 9.1 连线总览

```
用户纠错 ──①──→ 画像 confidence ↓
画像情绪 ──②──→ 门控语气收敛
画像热度 ──③──→ 检索阈值放宽
行为预测 ──④──→ 响应模式预调
画像标签 ──⑤──→ 定向探索冲动
```

| 连线 | 数据源 | 目标 | 触发频率 | 文件 |
|------|--------|------|---------|------|
| ① Feedback→Portrait | `error_reports.jsonl` | `PortraitEntry.confidence` | 每轮对话 | feedback.py, writer.py |
| ② Portrait→Gate | `usr6` 情绪条目 | `GatingDecision.tone` | 每轮对话 | circuit.py |
| ③ Portrait→weave | `compute_portrait_boost_map()` | 检索分层阈值 | 每轮对话 | circuit.py |
| ④ Predictor→Gate | `BehaviorPredictor` | `GatingDecision.response_mode` | 每轮对话 | circuit.py |
| ⑤ Portrait→Impulse | usr2/usr5/usr6 标签 | 泊松冲动(λ=900s) | 后台线程 | impulse.py, context.py |

### 9.2 执行顺序与冲突

五条连线在 `ChatCircuit.process()` 中的执行顺序：

```
process():
  ① realtime_update() 消费反馈 → 调整画像条目
  ④ mirror_prediction = BehaviorPredictor.predict() → 缓存结果
  ③ compute_portrait_boost_map() → 传入 weave_context()
  ② basal_ganglia_gate(ctx_obj=..., mirror_prediction=...) → 门控
  ⑤ 后台线程独立运行，不参与 process()
```

**已知冲突与化解**：
- **② vs ④ tone 冲突**：②先执行，从 usr6 收敛 tone=soft；④的 caring 分支检查 `tone=="warm"` 才触发 → ②收敛后④自然跳过。结论：**长期低落时不会强行 caring，行为正确。**
- **③ boost vs ① 降权**：①把条目标 PENDING 后，③的 boost map 仍按原值 boost（未实时同步 confidence 下降）。**暂未出问题，但改 boost 计算时需注意。**

### 9.3 认知追踪器 (`scripts/cognitive_trace.py`)

**用途**：改任何认知层参数前，先跑一把看效果。

**原理**：
1. 造假用户（画像条目 + 历史记忆 + 纠错记录）
2. 逐条连线调用真实引擎函数，记录 before/after
3. 检测多线同时修改同一变量的冲突
4. 估算最终 LLM prompt 的变化

**用法**：
```bash
python scripts/cognitive_trace.py
```

输出包括：场景设定 → 五线逐条推演 → 冲突/重叠分析 → LLM prompt 影响估算表格。

**什么时候跑**：
- 改了 boost 值、负面关键词、冲动间隔等参数
- 新增/删除连线
- 怀疑某条连线没生效时看 trace 输出

### 9.4 连线 Smoke 验证 (`scripts/verify_cognitive_wiring.py`)

**用途**：改完连线代码后快速确认 20 项基础功能没断。

```bash
python scripts/verify_cognitive_wiring.py   # 预期 20/20
```

**覆盖**：纠错降权、门控收敛、阈值放宽、预测预调、冲动产出。

---

*Agent 启动时自动加载。本文档是项目约定的唯一权威来源——如果代码与本文档冲突，以本文档为准。修改红线区域或约定前，必须跟用户确认。*
