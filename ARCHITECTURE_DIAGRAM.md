# 初痕全链路可视化架构图

> Mermaid 图表集。GitHub 原生渲染，直接可看。
> 最后更新：2026-06-09

---

## 一、系统全景

```mermaid
flowchart TB
    subgraph ONLINE["🌐 请求-响应管线（每次对话）"]
        direction TB
        USER["👤 用户消息"]
        EMBED["bge-m3 Embedding<br/>1024维 · 本地 · 零成本"]
        RETRIEVE["🔍 10路并行检索 · 零人格检索"]
        WEAVE["🧵 引擎编织 weave_context<br/>故事线 · 分层 · Token预算"]
        CIRCUIT["⚡ CircuitOrchestrator<br/>意图/情绪/门控/冲动/关系/画像"]
        LLM_GEN["🤖 LLM 生成回复<br/>stable画像(前缀缓存) + dynamic画像"]
        STORE["💾 存储管线 + 画像实时更新"]
        USER --> EMBED --> RETRIEVE --> WEAVE --> CIRCUIT --> LLM_GEN --> STORE
    end

    subgraph OFFLINE["🌙 后台自主节律（不等用户）"]
        direction TB
        IMPULSE["💭 冲动系统<br/>5源 + PriorityQueue + 消费者"]
        CONSOL["🏗️ 巩固引擎（DMN合并ticker）<br/>浅4h / 深24h / 空闲<br/>用户 + AI 双实例镜像"]
        PORTRAIT_BG["🖼️ 画像系统<br/>实时 · 浅4h · 深24h<br/>引擎提取 + LLM合成"]
        PATTERN["📊 模式发现<br/>6h · 5模式 · 零LLM"]
        DMN["🧠 DMN空闲检测"]
    end

    subgraph STORAGE_LAYER["💾 存储层"]
        CHROMA[("ChromaDB<br/>用户记忆")]
        AI_CHROMA[("ChromaDB<br/>AI记忆·元数据对等")]
        CHAT_HISTORY[("chat_history.jsonl")]
        WM[("工作记忆摘要")]
        PORTRAIT_MD[("PORTRAIT.md<br/>12维认知画像")]
        INDEXES["倒排 · 共现 · 实体对<br/>时间 · 话题树 · 标签嵌入"]
    end

    subgraph API_LAYER["🔌 API层"]
        REST["REST / SSE / OpenAI兼容"]
        HEALTH["健康检查 / 状态"]
    end

    ONLINE --> STORAGE_LAYER
    OFFLINE --> STORAGE_LAYER
    STORAGE_LAYER --> ONLINE
    STORAGE_LAYER --> OFFLINE
    API_LAYER --> ONLINE

    style ONLINE fill:#1a1a2e,stroke:#e94560,color:#eee
    style OFFLINE fill:#1a1a2e,stroke:#0f3460,color:#eee
    style STORAGE_LAYER fill:#1a1a2e,stroke:#16213e,color:#eee
    style API_LAYER fill:#1a1a2e,stroke:#533483,color:#eee
```

---

## 二、请求-响应管线：一条消息的完整旅程

```mermaid
flowchart LR
    A["👤 用户消息"] --> B["① Embedding<br/>bge-m3 → 1024维向量<br/>Ollama本地 · ~50ms"]
    B --> C["② 意图/情绪分类<br/>bge-m3原型匹配<br/>intent + emotion + urgency"]
    C --> D["③ 10路并行检索<br/>ThreadPoolExecutor(7)<br/>各路独立召回"]
    D --> E["④ 去重 + 两级精排<br/>cosine + hit_count<br/>+ v2.1 recency_weight"]
    E --> F{"⑤ weave_context<br/>引擎编织 · 零LLM · <150ms"}
    
    F --> F1["层一：故事线<br/>实体/标签聚类<br/>情绪趋势检测"]
    F1 --> F2["层二：认知分层<br/>fact / reference<br/>background / suppressed"]
    F2 --> F3["层三：Token预算<br/>20000 token软限制<br/>按叙事摘要截断"]
    
    F3 --> G["⑥ CircuitOrchestrator<br/>回路编排 · 8步决策"]
    G --> G1["意图分析"]
    G --> G2["情绪分析"]
    G --> G3["门控决策<br/>tone · formality · mode"]
    G --> G4["冲动注入"]
    G --> G5["行为预测<br/>Markov链"]
    G --> G6["关系评估<br/>familiarity/trust/closeness"]
    G --> G7["画像注入<br/>stable(8维)→前缀缓存<br/>dynamic(4维)→每轮更新"]
    
    G7 --> H["⑦ UtteranceSpec<br/>打包所有决策 → dataclass<br/>portrait_stable + portrait_dynamic"]
    H --> I["⑧ LLMClient.generate()<br/>tool-role JSON注入记忆<br/>system prompt含stable画像+门控指令<br/>DeepSeek前缀缓存命中>95%"]
    I --> J["⑨ SSE流式输出"]

    J --> K1["同步：chat_history写入"]
    J --> K2["同步：工作记忆增量更新"]
    J --> K3["异步：队列 → worker<br/>摘要(qwen2.5:3b) + 标签 + 实体<br/>→ ChromaDB + 倒排 + 共现"]
    J --> K4["同步：画像实时更新<br/>usr2/ai2+usr4/ai4<br/><100ms · 不调LLM"]
    
    style A fill:#e94560,color:#fff
    style J fill:#0f3460,color:#fff
    style H fill:#533483,color:#fff
```

---

## 三、10路并行检索

```mermaid
flowchart TB
    QUERY["🔍 查询 Embedding"] --> PARALLEL

    subgraph PARALLEL["ThreadPoolExecutor max_workers=7"]
        direction LR
        P1["① 语义hot<br/>ChromaDB heat=hot<br/>高活跃优先"]
        P2["② 语义cool<br/>ChromaDB warm/cool<br/>低活跃兜底 sim≥0.3"]
        P3["③ BM25全文<br/>BM25Okapi<br/>全部document全文索引"]
        P4["④ 关键词<br/>倒排索引(摘要)<br/>AND→OR退化"]
        P5["⑤ 标签<br/>标签倒排<br/>精确匹配≥1标签"]
        P6["⑥ 实体<br/>实体名精确匹配<br/>PERSON/LOC/ORG"]
        P7["⑦ 共现<br/>共现矩阵扩展<br/>已命中记忆的关联"]
        P8["⑧ 时间触发<br/>TemporalPatternIndex<br/>当前时段历史模式"]
        P9["⑨ 话题树<br/>话题树分支扩展<br/>同话题簇记忆"]
        P10["⑩ 注意力漂移<br/>近3轮加权embedding<br/>模拟注意力惯性"]
    end

    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 & P9 & P10 --> MERGE["去重: 按mem_id"]
    MERGE --> RANK["两级精排<br/>cosine + hit_count<br/>+ recency_weight"]
    RANK --> RESULT["候选记忆池 → weave_context"]

    style QUERY fill:#e94560,color:#fff
    style MERGE fill:#533483,color:#fff
    style RESULT fill:#0f3460,color:#fff
```

### 意图门控配额

| intent | semantic | tag | entity | time_expand |
|--------|:--------:|:---:|:------:|:-----------:|
| casual | 10 | 5 | 0 | 0 |
| recall | 20 | 8 | 5 | 5 |
| ask_fact | 25 | 10 | 5 | 0 |
| emotional_sharing | 12 | 5 | 0 | 3 |
| conflict | 25 | 10 | 5 | 5 |
| *benchmark* | *×2~5* | *×2~4* | *→10~20* | *→5~10* |

### v2.1 软降权公式

```
recency_weight = 1.0 - (days_ago / 90) × (1.0 - 0.15)   下限 0.15
  · archived 记忆上限 0.6
  · stale    记忆上限 0.3
  · 被报错   记忆 error_count ↑ → score ↓
```

---

## 四、引擎编织 weave_context（四层决策 · 零LLM · <150ms）

```mermaid
flowchart TB
    CANDIDATES["候选记忆池<br/>各路去重后"] --> PRE["预处理<br/>去stale · 解析元数据<br/>计算recency_weight"]
    
    PRE --> CHECK{"should_speak?<br/>intent=casual + 候选≤3"}
    CHECK -->|"否"| SILENT["不说<br/>避免无意义回复"]
    CHECK -->|"是"| L1

    subgraph WEAVE["四层决策引擎"]
        L1["层一：故事线编织<br/>━━━━━━━━━━━━━<br/>按实体/标签聚类<br/>计算时间跨度 ≥1天<br/>提取情绪趋势<br/>延续/翻转/持续积极/持续消极"]
        L1 --> L2["层二：认知分层<br/>━━━━━━━━━━━━━<br/>fact: 故事线内 + sem_dist<0.30<br/>reference: relevance中等<br/>background: 上下文相关<br/>suppressed: 引擎过滤"]
        L2 --> L3["层三：来源排序<br/>━━━━━━━━━━━━━<br/>semantic_hot(1.0)<br/>> bm25(0.90)<br/>> entity(0.85)<br/>> keyword(0.75)<br/>> tag(0.70)<br/>> time(0.65)<br/>> cooccur(0.60)<br/>> attention(0.55)"]
        L3 --> L4["层四：Token预算<br/>━━━━━━━━━━━━━<br/>MAX_TOKENS=20000软限制<br/>按叙事摘要截断<br/>非硬截断 · 完整透传"]
    end

    L4 --> STALE["stale处理<br/>不进fact → stale_context<br/>带stale_reason+superseded_by"]
    STALE --> OUTPUT["WovenContext<br/>fact_memories<br/>+ reference<br/>+ background<br/>+ stale_context<br/>+ narrative情绪趋势"]

    style CANDIDATES fill:#e94560,color:#fff
    style OUTPUT fill:#0f3460,color:#fff
    style WEAVE fill:#1a1a2e,stroke:#533483,color:#eee
```

---

## 五、CircuitOrchestrator 回路编排

```mermaid
flowchart LR
    INPUT["WovenContext<br/>+ 用户消息"] --> S1

    subgraph ORCHESTRATOR["CircuitOrchestrator.process() · 8步串行"]
        S1["① 意图分析<br/>━━━━━━━<br/>bge-m3原型匹配<br/>casual/recall/ask_fact/<br/>emotional_sharing/conflict"]
        S1 --> S2["② 情绪分析<br/>━━━━━━━<br/>Russell二维环<br/>valence × arousal<br/>+ intensity(0~1)"]
        S2 --> S3["③ 认知分层<br/>━━━━━━━<br/>MemoryDirective<br/>fact/reference/<br/>background/suppressed"]
        S3 --> S4["④ 门控决策<br/>━━━━━━━<br/>tone · formality<br/>response_mode<br/>suppression理由"]
        S4 --> S5["⑤ 冲动注入<br/>━━━━━━━<br/>PriorityQueue检查<br/>有效优先级≥2<br/>→ ImpulseDirective"]
        S5 --> S6["⑥ 行为预测<br/>━━━━━━━<br/>Markov链<br/>mirror_prediction<br/>下一步意图/话题"]
        S6 --> S7["⑦ 关系评估<br/>━━━━━━━<br/>RelationshipState<br/>familiarity/trust<br/>closeness/mode"]
        S7 --> S8["⑧ 画像注入<br/>━━━━━━━<br/>render_stable(8维)<br/>→ message[0] 前缀缓存<br/>render_dynamic(4维)<br/>→ message[N+1] 每轮更新"]
    end

    S8 --> OUTPUT["UtteranceSpec<br/>全部决策打包 → LLM<br/>portrait_stable + portrait_dynamic"]

    style INPUT fill:#e94560,color:#fff
    style OUTPUT fill:#0f3460,color:#fff
    style ORCHESTRATOR fill:#1a1a2e,stroke:#533483,color:#eee
```

---

## 六、后台自主节律（全线程视图）

```mermaid
flowchart TB
    subgraph IMPULSE["💭 冲动系统 · 6线程"]
        direction LR
        IS1["情绪趋势<br/>10min · Poisson"]
        IS2["时间节律<br/>30min · Poisson"]
        IS3["随机漫游<br/>10min · Poisson"]
        IS4["好奇心<br/>20min · Poisson"]
        IS5["行为模式<br/>30min · Poisson"]
        
        IS1 & IS2 & IS3 & IS4 & IS5 --> FATIGUE["疲劳抑制<br/>每次+0.15 · 半衰15min<br/>有效优先级 = 基础 × (1-疲劳)"]
        FATIGUE --> PQ["PriorityQueue<br/>有效优先级<2 → 丢弃<br/>>TTL → 丢弃"]
        PQ --> CONSUMER["冲动消费者<br/>空闲>2min触发<br/>LLM生成→[内心独白]"]
    end

    subgraph CONSOLIDATION["🏗️ 巩固引擎 · 3级 · 用户+AI双实例"]
        direction TB
        SHALLOW["浅巩固 · 4h<br/>━━━━━━━━<br/>话题树重建<br/>语义重复检测<br/>标签嵌入索引<br/>画像浅更新<br/>人格对称性<br/>事实冲突检测<br/>实体对演化<br/>冷热转换"]
        DEEP["深巩固 · 24h<br/>━━━━━━━━<br/>归档评估 90天<br/>话题笔记生成<br/>画像深更新<br/>情绪淡化"]
        IDLE["空闲巩固<br/>━━━━━━━━<br/>Level1: 预热+重建缓存<br/>Level2: >4h回顾<br/>Level3: >24h日巩固"]
    end

    subgraph PORTRAIT_BG["🖼️ 画像系统 · 三层更新"]
        direction TB
        PRT_RT["实时更新 · 每轮<br/>引擎提取特征·规则合成<br/><100ms · 不调LLM<br/>usr2/ai2 + usr4/ai4"]
        PRT_SH["浅更新 · 4h<br/>提取器扫描+LLM写条目<br/>usr3/ai3 + usr5/ai5<br/>+ usr1/ai1新候选"]
        PRT_DP["深更新 · 24h<br/>全局扫描+LLM合成<br/>usr1/ai1 + usr6/ai6<br/>最低20轮门槛"]
    end

    subgraph OTHER["其他后台线程"]
        AI_C["AI巩固 · 镜像<br/>独立ConsolidationEngine<br/>共享DMN ticker触发<br/>情绪淡化 独立1h定时器"]
        PAT["模式发现 · 6h<br/>5模式 · 零LLM"]
        DMN["DMN · ~5min<br/>空闲检测+触发巩固<br/>用户+AI双引擎"]
    end

    LIFECYCLE["🛡️ lifecycle.py<br/>崩溃自动重启 · 5次/h上限 · 优雅退出"]
    
    IMPULSE & CONSOLIDATION & OTHER --> LIFECYCLE

    style FATIGUE fill:#533483,color:#fff
    style PQ fill:#e94560,color:#fff
    style CONSUMER fill:#0f3460,color:#fff
```

### 内抑制细节

```
疲劳度增长:  每次发射 +0.15
半衰期:      15分钟
抑制阈值:    有效优先级 < 2
TTL过期:     超时自动丢弃
速率限制:    MAX_PER_HOUR / MIN_INTERVAL
```

---

## 七、记忆生命周期

```mermaid
stateDiagram-v2
    [*] --> hot: 新建·情绪强度≥2
    [*] --> warm: 新建·普通
    
    hot --> hot: hit_count增长
    warm --> cool: 14天无人问津
    hot --> cool: 14天无命中
    
    hot --> stale: 情绪翻转/事实更新
    warm --> stale: 语义重复+新事实
    
    cool --> archived: 话题簇中位数<br/>最后命中>90天
    
    stale --> [*]: 软降权<br/>recency_weight≤0.3<br/>不进fact→stale_context
    archived --> [*]: 软降权<br/>recency_weight≤0.6

    note right of stale: v2.1不再硬屏蔽<br/>保留为背景参考<br/>LLM了解变化过程
    note right of archived: 90天归档<br/>仍可被检索
```

### 情绪衰减（独立于巩固调度器）

```
触发:  每50次 increment_hit_count
条件:  3天未命中 + 高emotional_intensity
效果:  intensity自然衰减
特点:  不依赖巩固调度器，在线路上自然发生
```

---

## 八、存储层全景

```mermaid
flowchart TB
    subgraph CHROMA["ChromaDB · 本地持久化 · HNSW索引"]
        direction LR
        USER_COL["📁 用户记忆集合<br/>━━━━━━━━<br/>document: 原文<br/>metadata.summary<br/>metadata.tags<br/>metadata.emotion<br/>metadata.heat<br/>metadata.hit_count<br/>metadata.stale<br/>metadata.archived<br/>metadata.superseded_by"]
        AI_COL["📁 AI记忆集合<br/>━━━━━━━━<br/>document: AI回复<br/>metadata.summary<br/>metadata.tags<br/>metadata.emotion<br/>独立巩固"]
    end

    subgraph INDEXES["内存索引层"]
        direction LR
        INV["倒排索引<br/>词→mem_id<br/>标签→mem_id"]
        CO["共现矩阵<br/>mem_id→关联记忆"]
        EP["实体对<br/>实体共现计数"]
        AFF["话题亲和图<br/>话题间边权重"]
        TEMP["时间模式索引<br/>时段→话题"]
        TREE["话题树<br/>层次聚类"]
        TAG_EMB["标签嵌入<br/>tag→1024维"]
    end

    subgraph FILES["文件存储"]
        CHAT[("chat_history.jsonl<br/>对话记录")]
        WM[("working_memory.json<br/>工作记忆摘要")]
        ERROR[("error_reports.jsonl<br/>用户反馈")]
        PATTERN[("pattern_cache.json<br/>模式发现产出")]
        BLIND[("blind_spots.json<br/>人格对称盲区")]
        PORTRAIT_FILE[("PORTRAIT.md<br/>12维认知画像")]
        PERSONA[("personality_tags.json<br/>人格标签[退役中]")]
    end

    CHROMA --> INDEXES
    INDEXES --> CHROMA
    FILES

    style CHROMA fill:#1a1a2e,stroke:#0f3460,color:#eee
    style INDEXES fill:#1a1a2e,stroke:#533483,color:#eee
    style FILES fill:#1a1a2e,stroke:#16213e,color:#eee
```

---

## 九、模块依赖图

```mermaid
flowchart TB
    subgraph API["api/ · REST层"]
        CHAT["chat.py<br/>对话+流式+OpenAI兼容<br/>benchmark注入+画像实时"]
        SYS["system.py<br/>健康/状态"]
        MEM_API["memories.py<br/>记忆CRUD+反馈"]
        PORT_API["portrait.py<br/>画像渲染"]
        CONS_API["consolidation.py"]
    end

    subgraph CORE["core/ · 认知核心"]
        STATE["state.py<br/>CognitiveState<br/>UtteranceSpec"]
        CIRC["circuit.py<br/>CircuitOrchestrator<br/>含画像注入"]
        CONFLICT["conflict.py<br/>冲突消解"]
        FEEDBACK["feedback.py<br/>用户反馈闭环"]
        CTX["context.py<br/>AppContext<br/>画像初始化+AI巩固镜像"]
    end

    subgraph BRAIN["brain/ · 语义引擎"]
        SEM["semantic.py<br/>7函数·零模型依赖"]
        KW["keywords.py<br/>统一常量表"]
    end

    subgraph PORTRAIT["portrait/ · 画像系统"]
        PRT_MGR["manager.py<br/>PORTRAIT.md CRUD"]
        PRT_ST["state.py<br/>EntryStateMachine"]
        PRT_EXT["extractors.py<br/>特征提取器"]
        PRT_RND["renderer.py<br/>stable/dynamic渲染"]
        PRT_WRT["writer.py<br/>三层更新"]
    end

    subgraph MEMORY["memory/ · 存储层"]
        CHROMA_M["chroma.py<br/>ChromaDB封装"]
        INV_M["inverted.py"]
        CO_M["cooccur.py"]
        TREE_M["tree.py"]
        WM_M["working.py<br/>工作记忆"]
        TEMP_M["temporal.py"]
    end

    subgraph RETRIEVAL["retrieval/ · 检索管线"]
        PIPE["pipeline.py<br/>10路检索+编织<br/>Portrait light boost"]
        SCORE["scoring.py<br/>精排+软降权"]
        BM25["bm25_fulltext.py"]
    end

    subgraph ANALYSIS["analysis/ · 分析层"]
        EMO["emotion.py<br/>Russell环"]
        ENTITY["entity.py<br/>qwen实体"]
        PD["pattern_discovery.py<br/>5模式"]
        SYM["symmetry.py<br/>人格对称"]
        PRED["predictor.py<br/>Markov"]
    end

    subgraph BACKGROUND["background/ · 后台节律"]
        CONS_BG["consolidation.py<br/>巩固引擎·用户+AI双实例"]
        IMP_BG["impulse.py<br/>冲动系统"]
        DIST_BG["distill.py<br/>蒸馏引擎·退役中"]
        LIFE_BG["lifecycle.py<br/>生命周期"]
    end

    subgraph LLM["llm/ · LLM适配"]
        DS["deepseek.py<br/>主LLM·画像注入"]
        EMB["embed.py<br/>bge-m3"]
        LOCAL["local.py<br/>qwen2.5:3b"]
    end

    API --> CORE
    API --> PORTRAIT
    CORE --> PORTRAIT
    CORE --> BRAIN
    CORE --> MEMORY
    CORE --> RETRIEVAL
    RETRIEVAL --> MEMORY
    RETRIEVAL --> BRAIN
    CORE --> ANALYSIS
    BACKGROUND --> MEMORY
    BACKGROUND --> ANALYSIS
    BACKGROUND --> CORE
    BACKGROUND --> PORTRAIT
    BACKGROUND --> LLM
    API --> LLM
    CORE --> LLM

    style CORE fill:#1a1a2e,stroke:#e94560,color:#eee
    style RETRIEVAL fill:#1a1a2e,stroke:#0f3460,color:#eee
    style BACKGROUND fill:#1a1a2e,stroke:#533483,color:#eee
    style PORTRAIT fill:#1a1a2e,stroke:#e94560,color:#eee
```

**依赖方向**: `api/ → core/ → brain/ + memory/ → retrieval/ + analysis/ → background/ → llm/`

---

## 十、Prompt 注入结构

```mermaid
flowchart LR
    subgraph SYSTEM_PROMPT_STABLE["System Prompt message[0]（稳定前缀 → DeepSeek缓存命中>95%）"]
        direction TB
        SP1["核心人格 + 工具规则"]
        SP2["stable画像 (8维)<br/>usr1/3/5/6 + ai1/3/5/6"]
        SP3["人格标签 [退役中]"]
        SP4["模式观察"]
        SP5["话题笔记"]
        SP6["引擎调参"]
    end

    subgraph SYSTEM_PROMPT_DYNAMIC["System Prompt message[N+1]（动态 → 每轮更新）"]
        direction TB
        SD1["dynamic画像 (4维)<br/>usr2/4 + ai2/4"]
        SD2["session_context"]
        SD3["now_hint 时间感知"]
    end

    subgraph HISTORY["历史对话 user/assistant"]
        H1["工作记忆摘要 ~3K tokens"]
        H2["最近5轮原文 ~2K tokens"]
    end

    subgraph TOOL_ROLE["Tool Role · 记忆注入（JSON）"]
        direction TB
        M1["fact记忆<br/>relevance高 · 原文完整"]
        M2["reference记忆<br/>relevance中 · 带核实语气"]
        M3["background记忆<br/>调语气 · 不提及"]
        M4["stale_context<br/>stale=true · 了解变化"]
    end

    CURRENT["当前用户消息 user"]

    SYSTEM_PROMPT_STABLE --> HISTORY --> TOOL_ROLE --> SYSTEM_PROMPT_DYNAMIC --> CURRENT

    style SYSTEM_PROMPT_STABLE fill:#1a1a2e,stroke:#0f3460,color:#eee
    style SYSTEM_PROMPT_DYNAMIC fill:#1a1a2e,stroke:#533483,color:#eee
    style TOOL_ROLE fill:#1a1a2e,stroke:#e94560,color:#eee
```

---

## 十一、E2E Benchmark 体系

```mermaid
flowchart LR
    subgraph LINK1["链路一: 写入 · 12节点"]
        W1["embedding → summary<br/>→ tags → entities<br/>→ emotion → storage<br/>→ 索引更新"]
    end

    subgraph LINK2["链路二: 检索+编织+认知 · 35节点"]
        R1["意图→配额→WM<br/>→10路检索→去重<br/>→精排→编织<br/>→分层→冲突→Token<br/>→门控→人格→冲动<br/>→LLM回复验证"]
    end

    subgraph LINK3["链路三: 跨轮 · 9变体"]
        X1["短跨/长跨/同义改写<br/>注意力惯性/话题切换<br/>情绪翻转/WM延续<br/>关系演化/冲突修正"]
    end

    subgraph LINK4["链路四: 演化 · 16节点"]
        M1["浅巩固: 话题树/重复<br/>Supersede/标签嵌入<br/>亲和图/画像浅更新/冷热<br/>实体对/对称性<br/>深巩固: 归档/笔记/淡化<br/>画像深更新<br/>独立: 情绪衰减/AI巩固镜像<br/>反馈闭环/原文不变"]
    end

    subgraph LINK5["链路五: 后台 · 17节点"]
        B1["5冲动源→疲劳→抑制<br/>→消费→TTL→浅/深巩固<br/>→画像实时/浅/深<br/>→模式发现→DMN→AI巩固<br/>→线程存活+重启"]
    end

    LINK1 --> LINK2 --> LINK3 --> LINK4 --> LINK5

    style LINK1 fill:#1a1a2e,stroke:#e94560,color:#eee
    style LINK2 fill:#1a1a2e,stroke:#0f3460,color:#eee
    style LINK3 fill:#1a1a2e,stroke:#533483,color:#eee
    style LINK4 fill:#1a1a2e,stroke:#16213e,color:#eee
    style LINK5 fill:#1a1a2e,stroke:#1a1a2e,color:#eee
```

**5 链路 · 各自独立计分 · 不加权合成一个数字**

---

*初痕 · First Beat — 自循环记忆体*
