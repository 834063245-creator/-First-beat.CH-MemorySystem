# SPEC: 初痕记忆引擎 — 存储与推理基础设施迁移

> **版本**: v1.6 · **日期**: 2026-06-16 · **状态**: Phase 0 ✅ → Phase 0.5 进行中
> **关联文档**: CLAUDE.md（项目唯一权威文档）
> **评审**: 2026-06-15 三轮评审。v1.4 补全 HyperEdge/检索管线/迁移脚本。v1.5 回退路径合并。v1.6 payload 全字段原生类型——entities/entity_co_counts/memory_ids 从 JSON 字符串改为 Qdrant 原生 list[dict]/dict/list[str]，删除所有 json.dumps/loads，MatchText→MatchValue where applicable。
> **目标规模**: 十万级起步，百万级架构储备

---

## 目录

1. [动机](#1-动机)
2. [目标架构](#2-目标架构)
3. [组件映射：旧 → 新](#3-组件映射旧--新)
4. [数据模型设计](#4-数据模型设计)
5. [API 映射：ChromaDB → Qdrant](#5-api-映射chromadb--qdrant)
6. [API 映射：Ollama → vLLM](#6-api-映射ollama--vllm)
7. [SQLite 迁移策略](#7-sqlite-迁移策略)
8. [检索管线重设计](#8-检索管线重设计)
9. [Phase 分步执行计划](#9-phase-分步执行计划)
10. [验收标准](#10-验收标准)
11. [风险登记册](#11-风险登记册)
12. [回滚策略](#12-回滚策略)

---

## 1. 动机

### 现状瓶颈

| 瓶颈 | 症状 | 根因 |
|------|------|------|
| **ChromaDB metadata 查询弱** | 几乎全量 `list_all()` → Python 侧过滤 | ChromaDB `where` 不支持复合条件、不支持聚合 |
| **SQLite 同步冗余** | 3 个独立 .db 文件（cooccur/entity_pair/hyperedge）与 ChromaDB 并行维护，数据重复 | metadata 存 JSON blob，无法查询，被迫外挂 SQLite |
| **全量加载成习惯** | `list_all()` / 全量 `get()` / 全量 `emb_cache` / 全量 `bm25` / 全量 `inverted_index` | ChromaDB 不给力的 API 逼出这些模式 |
| **Ollama 瓶颈** | HTTP 单条 embed 延迟不可控，依赖本地 GPU | vLLM 批量推理 + OpenAI 兼容接口更标准 |
| **百万级完全不可达** | 现有写模式在十万级就崩 | 全量加载 × N 处 = 线性崩溃 |

### 目标

```
现状:  ChromaDB + SQLite×3 + JSONL  + Ollama(bge-m3/qwen2.5)
       └── 全量加载模式遍布各模块

目标:  Qdrant(payload 存元数据)  + JSONL(异步日志)  + vLLM(bge-m3/qwen2.5)
       └── 服务端过滤 + 分页 + 量化 + 分区
       └── embedding 接口不变，HTTP 层换
       └── 删 SQLite 同步逻辑，元数据进 Qdrant payload
```

---

## 2. 目标架构

### 2.1 存储层

```
                        ┌──────────────────────┐
                        │   Python 进程          │
                        │                       │
  ┌─────────────────┐   │  ┌─────────────────┐  │
  │  vLLM embed      │◄──┼──│  llm/embed.py   │  │
  │  :8001           │   │  └─────────────────┘  │
  │  bge-m3          │   │                       │
  └─────────────────┘   │  ┌─────────────────┐  │
                         │  │  llm/local.py   │  │
  ┌─────────────────┐   │  │  brain/semantic │  │
  │  vLLM chat       │◄──┼──│                 │  │
  │  :8002           │   │  └─────────────────┘  │
  │  qwen2.5:3b      │   │                       │
  └─────────────────┘   │  ┌─────────────────┐  │
                         │  │  memory/         │  │
  ┌─────────────────┐   │  │  qdrant.py (新)  │  │
  │  Qdrant 服务      │◄──┼──│                 │  │
  │  :6333           │   │  └─────────────────┘  │
  │  向量 + payload  │   │                       │
  └─────────────────┘   │  ┌─────────────────┐  │
                         │  │  JSONL 文件      │  │
                         │  │  chat_history    │  │
                         │  │  store_failures  │  │
                         │  │  error_reports   │  │
                         │  └─────────────────┘  │
                         └──────────────────────┘

  删掉的:
  ✗ data/chroma/         (ChromaDB PersistentClient)
  ✗ data/co_occurrence.db
  ✗ data/entity_pairs.db
  ✗ data/hyper_edges.db
  ✗ app/core/db.py        (SQLite 连接池)
  ✗ app/memory/cooccur.py
  ✗ app/memory/entity_pair.py
  ✗ app/memory/hyperedge.py
  ✗ app/retrieval/bm25_fulltext.py
```

### 2.2 推理层

> **已解决（附录 B#5）**: vLLM 单实例不支持同时跑 embedding 模型和 chat 模型。采用双实例部署。

```
  现状 Ollama (单实例):
    :11434/api/embeddings  → bge-m3 单条 embed
    :11434/api/embed       → bge-m3 批量 embed
    :11434/api/generate    → qwen2.5:3b 摘要 + 实体抽取（同一模型，不同 prompt）

  目标 vLLM (双实例):
    :8001/v1/embeddings    → bge-m3 单条+批量 (OpenAI 兼容)     ← 实例1
    :8002/v1/chat/completions → qwen2.5:3b 摘要+实体抽取         ← 实例2
```

**公开接口不变：**
- `local_embed(text) → list[float] | None` — 签名不变
- `local_embed_batch(texts) → list[list[float] | None]` — 签名不变
- `local_embed_async(text) → list[float] | None` — 签名不变
- `LocalLLM.generate(prompt, max_tokens) → str | None` — 签名不变
- `LocalLLM.summarize(text, max_chars) → str` — 签名不变
- `extract_entities(text) → list[dict]` — 签名不变

---

## 3. 组件映射：旧 → 新

### 3.1 模块级映射

| 现状 | 目标 | 操作 |
|------|------|------|
| `app/memory/chroma.py` (711行) | `app/memory/qdrant.py` (新建) | 重写，API 方法名保持一致 |
| `app/core/db.py` (50行) | — | **删除** |
| `app/memory/cooccur.py` (292行) | Qdrant 独立 collection `co_occurrence` | **删除**，逻辑移到 qdrant.py |
| `app/memory/entity_pair.py` (236行) | Qdrant payload `entity_co_counts` 字段 | **删除**，入库时预计算 |
| `app/memory/hyperedge.py` (461行) | Qdrant 独立 collection `hyper_edges` | **删除**，逻辑移到 qdrant.py |
| `app/memory/inverted.py` (157行) | 不变（仅改数据源为 Qdrant scroll） | 无改动，纯 Python 数据结构，公开 API 不变 |
| `app/memory/temporal.py` (171行) | 不变（读 Qdrant payload） | 改数据源 |
| `app/memory/tree.py` (179行) | 不变 | 改数据源 |
| `app/memory/affinity.py` (86行) | 不变 | 改数据源 |
| `app/memory/tag_index.py` (140行) | 不变（需重建缓存） | 切换 vLLM 后标签 embedding 缓存失效，需重建 `data/tag_index.json` |
| `app/memory/history.py` (270行) | 不变（JSONL） | 无改动 |
| `app/memory/working.py` (165行) | 不变（JSON） | 无改动 |
| `app/llm/embed.py` (390行) | 仅改 HTTP 层 | 改 2 个私有函数 |
| `app/llm/local.py` (118行) | 仅改 HTTP 层 | 改 2 个方法 |
| `app/brain/semantic.py` (477行) | 仅改 HTTP 层 | 改 1 处调用 |
| `app/retrieval/pipeline.py` (703行) | Qdrant 查询 | 改所有 `_collection.xxx` 调用 |
| `app/retrieval/bm25_fulltext.py` (110行) | — | **删除**，Qdrant 全文替代 |
| `app/retrieval/scoring.py` (40行) | 不变 | 无改动 |
| `app/core/context.py` (945行) | QdrantService + 删 tracker init | 改初始化段 |
| `app/core/conflict.py` | 类型标注改 | 改参数类型 |
| `app/tools/dispatch.py` (815行) | Qdrant 查询 | 改 `_get_chroma_collection()` |
| `app/api/chat.py` | `ctx.chroma_service` → `ctx.qdrant_service` | 改名 |
| `app/api/memories.py` | 同上 | 改名 |
| `app/api/system.py` | 同上 | 改名 |
| `app/background/consolidation.py` (1071行) | 同上 | 改名 |
| `app/background/impulse.py` (565行) | 同上 | 改名 |
| `app/analysis/self_mirror.py` (170行) | 同上 | 改名 |
| `app/config/settings.py` (249行) | 新配置项 | 增删配置 |

### 3.2 不改动的模块

```
✅ app/llm/deepseek.py       — 主 LLM，不碰（红线）
✅ app/core/circuit.py       — 管线不变（底层透明）
✅ app/core/state.py         — 数据结构不变
✅ app/core/tools.py         — 工具定义不变
✅ app/core/heartbeat.py     — 基础设施不变
✅ app/core/helpers.py       — 基础设施不变
✅ app/core/bottleneck.py    — 基础设施不变
✅ app/portrait/*            — 画像系统不变（只读 QdrantService）
✅ app/analysis/emotion.py   — 纯计算不变
✅ app/analysis/drift.py     — 纯计算不变
✅ app/analysis/symmetry.py  — 从 Qdrant CoOccurrence collection 读，接口不变
✅ app/analysis/predictor.py — 读数据 → 从 Qdrant 读，接口不变
✅ app/analysis/pattern_discovery.py — 同上
✅ app/tools/atomic.py       — 工具函数不变
✅ app/tools/search.py       — 工具函数不变
✅ app/tools/workspace.py    — 工具函数不变
```

---

## 4. 数据模型设计

### 4.1 Qdrant Collection: `memories`

每个 point 的 payload 承载原本分散在 ChromaDB metadata + entity_pair 中的全部信息：

```yaml
id: "uuid-string"                          # Qdrant point id
vector: [1024-dim float32]                 # bge-m3 embedding
payload:
  # ── 原有 ChromaDB metadata ──
  user_message: "..."                      # 用户原话
  ai_message: "..."                        # AI 原话
  document: "用户：...\nAI：..."           # 拼接全文
  summary: "..."                           # LLM 摘要
  tags: "Python,Rust,编程"                 # 逗号分隔
  timestamp: 1718400000.0                  # Unix 时间戳
  hit_count: 5                             # 命中次数
  last_hit_time: 1718500000.0              # 最后命中时间
  heat: "hot"                              # hot | warm | cool
  embed_model: "bge-m3"                    # 嵌入模型
  stale: false                             # 是否被取代
  archived: false                          # 是否归档
  superseded_by: ""                        # 替代者 ID
  supersede_reason: ""                     # 替代原因
  superseded_at: ""                        # 替代时间
  storage_complete: true                   # 入库完成标记
  source: "user"                           # user | ai
  date_tag: "2026-06-15"                   # 日期标签

  # ── 预计算时间特征 ──
  year: 2026
  month: 6
  day: 15
  week: 24
  day_of_week: 1                           # 0=周一
  quarter: 2
  season: "summer"
  year_month: "2026-06"

  # ── 情绪 ──
  emotion_valence: 0.7                     # Russell 2D 效价
  emotion_arousal: 0.5                     # Russell 2D 唤醒度
  emotion_valence_bin: "positive"          # positive | negative | neutral
  emotional_intensity: 3                   # 0-5 强度

  # ── 实体 ──
  entities: [{"text": "Python", "type": "TECHNOLOGY"}, ...]   # 原生 list[dict]，Qdrant 可对 entities[].text 建 keyword 索引

  # ── 替代 SQLite entity_pair.py（入库时预计算，非查询时派生）──
  entity_co_counts: {"Python": 15, "Rust": 8, ...}            # 原生 dict[str,int]，Python 端聚合，不需 Qdrant 索引
```

> **为什么 CoOccurrence 不进 payload**：热门记忆可能与 500+ 条其他记忆共现，`co_occurring` JSON 会膨胀到数百 KB。PersonaSymmetry 需要全局 Top-N 排序（`ORDER BY count DESC LIMIT 10000`），分散到 payload 后无法高效查询。CoOccurrence 保留为独立 collection（§4.3）。

### 4.2 Qdrant Collection: `co_occurrence`

> **设计决策**: CoOccurrence 不进 payload。原因：(1) 热门记忆的 `co_occurring` JSON 膨胀到数百 KB；(2) PersonaSymmetry 需要全局 `ORDER BY count DESC LIMIT 10000`，分散到 payload 无法高效查询；(3) 独立 collection 支持按 count 排序的 scroll。

```yaml
id: auto-generated (UUID)
vector: [1024-dim float32]                 # id_a 对应记忆的 embedding（用于语义近邻检索）
payload:
  id_a: "<memory_id>"                      # 记忆 A 的 ID
  id_b: "<memory_id>"                      # 记忆 B 的 ID（保证 id_a < id_b 字典序）
  count: 5                                 # 共现次数
  last_time: 1718500000.0                  # 最后共现时间 (Unix timestamp)
```

**操作映射**：

```python
# record(memory_ids): 同轮出现的记忆对，count += 1
#   → Qdrant: 先 retrieve 现有点的 count，再 upsert（合并器锁保护）
#   → 优化: 使用 Qdrant recommend API 直接获取相似记忆，天然共现

# query(memory_ids) → list[{id, count}]:
#   → Qdrant scroll: filter id_a IN (...) OR id_b IN (...)
#   → 客户端合并去重，按 count 降序

# export_for_symmetry(limit=10000):
#   → Qdrant scroll: order_by count DESC, limit=10000
#   → 客户端转为 {id_a: {id_b: count, ...}, ...}

# LTD 衰减:
#   → 改为查询时降权: score *= max(0.5, 1.0 - days_since_last/14)
#   → 删除定时全表 UPDATE 任务
```

> ⚠️ **向量过时风险**: co_occurrence 点的 `vector` 存的是 `id_a` 对应记忆的 embedding。当记忆被删除或 embedding 更新时，cooccurrence 向量不会级联更新。该向量仅用于「通过 cooccurrence 找语义相似记忆」这一辅助场景——它是**尽力而为的快照**，不用于精确排名的语义搜索。核心查询路径（query/export_for_symmetry）走 `scroll` + `order_by count`，不依赖向量。

### 4.3 Qdrant Collection: `hyper_edges`

```yaml
id: auto-generated
vector: [1024-dim float32]                 # 参与实体的平均 embedding
payload:
  entities: ["Python", "VSCode", "IDE"]     # 排序后的实体列表 (原生 list[str]，可建 keyword 索引)
  memory_ids: ["id1", "id2", "id3"]         # 关联的记忆 ID (原生 list[str])
  created_at: "2026-06-15T10:30:00"        # 创建时间 (ISO string)
  edge_size: 3                             # 实体数 (int)
```

### 4.4 Qdrant Collection: `ai_memories`

与 `memories` 结构相同，存 AI 自我记忆。

### 4.5 Payload 索引策略

```
必须建索引的字段（高频过滤条件）:
  memories collection:
    - heat              (keyword / enum)
    - timestamp         (float, range queries)
    - last_hit_time     (float, 用于 embedding 缓存排序和 recency 过滤)
    - emotional_intensity (integer, range queries)
    - emotion_valence_bin (keyword)
    - stale             (bool)
    - archived          (bool)
    - date_tag          (keyword)
    - source            (keyword)

  co_occurrence collection:
    - id_a              (keyword, 精确匹配查询)
    - id_b              (keyword, 精确匹配查询)
    - count             (integer, 排序用)

文本索引 (Qdrant text index，用于 bm25 替代路径):
  - document          (text, bm25 替代路径用 MatchText 搜索)

按需索引 (原生 list/dict 类型可使用嵌套路径建索引):
  - entities[].text   (keyword/text, 实体名的精确/分词匹配)
  - entities[].type   (keyword, 按实体类型过滤，如 "PERSON" / "TECHNOLOGY")

按需建索引:
  - year_month        (keyword, 分区时用)
```

### 4.6 量化配置

```
prod:
  on_disk: true                           # 向量存磁盘，mmap访问
  quantization: scalar_int8               # 4GB → 1GB
  hnsw: m=16, ef_construct=100, ef=64     # 召回率 > 0.99

benchmark:
  on_disk: false                          # 小数据集，全在内存
  quantization: none
```

---

## 5. API 映射：ChromaDB → Qdrant

### 5.1 写入

```python
# ── add_memory ──
# ChromaDB
self._collection.add(
    ids=[memory_id],
    documents=[document],
    embeddings=[embedding],
    metadatas=[meta],
)
# Qdrant
self._client.upsert(
    collection_name=self._collection_name,
    points=[models.PointStruct(
        id=memory_id,
        vector=embedding,
        payload={**meta, "document": document},
    )],
)

# ── mark_storage_complete ──
# ChromaDB
self._collection.update(ids=[memory_id], metadatas=[{"storage_complete": True}])
# Qdrant
self._client.set_payload(
    collection_name=self._collection_name,
    payload={"storage_complete": True},
    points=[memory_id],
)

# ── update_memory ──
# ChromaDB
self._collection.update(ids=[id], metadatas=[meta], embeddings=[emb])
# Qdrant
self._client.overwrite_payload(...)
self._client.update_vectors(...)

# ── batch_increment_hit_count ──
# 现状: 一次 get + 一次 update（ChromaDB 特性）
# Qdrant: 每条记忆 set_payload 增量更新 hit_count 和 last_hit_time
# 合并器锁保护，一次持锁内完成全部 set_payload 调用
```

### 5.2 查询

```python
# ── 语义检索 (path1) ──
# ChromaDB
col.query(query_embeddings=[q], n_results=50,
          where={"heat": "hot"},
          include=["documents","metadatas","distances"])
# Qdrant
client.search(
    collection_name=coll,
    query_vector=q,
    query_filter=models.Filter(must=[
        models.FieldCondition(key="heat", match=models.MatchValue(value="hot")),
    ]),
    limit=50,
    with_payload=True,
    with_vectors=False,
)

# ── list_memories (分页+过滤) ──
# 现状: list_all_cached() → Python filter → sort → slice
# Qdrant: scroll + filter + order_by
client.scroll(
    collection_name=coll,
    scroll_filter=models.Filter(must=[...]),
    limit=per_page,
    offset=offset,
    order_by=models.OrderBy(key="timestamp", direction=models.Direction.DESC),
    with_payload=True,
)

# ── list_since / list_before ──
# 现状: collection.get(where={"timestamp": {"$gte": since_ts}})
# Qdrant: scroll with Range filter on timestamp

# ── query_by_emotion ──
# 现状: list_all() → Python filter valence range
# Qdrant: scroll with Range filter on emotion_valence

# ── get_memories_by_timerange ──
# 现状: list_all() → Python filter timestamp range
# Qdrant: scroll with Range filter on timestamp
```

### 5.3 管理

```python
# ── delete_memory ──
# ChromaDB: self._collection.delete(ids=[mid])
# Qdrant: client.delete(collection_name=coll, points_selector=[mid])

# ── count ──
# ChromaDB: self._collection.count()
# Qdrant: client.count(collection_name=coll).count

# ── stats ──
# 现状: 运行计数器 + count()
# Qdrant: count() + payload 聚合或运行计数器（保持）

# ── embedding 缓存 ──
# 现状: ChromaService._emb_cache (dict) + _emb_cache_lock + _get_embedding_cached()
#       外部访问: ctx.chroma_service._emb_cache[mid] = emb   (context.py 入库时写)
#                ctx.chroma_service._get_embedding_cached(mid)  (pipeline.py 注意力评分)
# Qdrant: 保留 _emb_cache: dict + _emb_cache_lock + _get_embedding_cached()
#         外部写: ctx.qdrant_service._emb_cache[mid] = emb  (与旧接口一致)
#         外部读: ctx.qdrant_service._get_embedding_cached(mid)  (不变)
#         重建策略: 启动时 scroll 最近 20K 条（按 last_hit_time DESC），不阻塞请求
```

### 5.4 关键差异：Filter 翻译层

```python
# 需要一个 ChromaDB where → Qdrant Filter 的翻译函数
# 用在 list_since, list_before, 情感淡化扫描, dispatch query 等场景

def _translate_filter(chroma_where: dict) -> models.Filter:
    """ChromaDB where dict → Qdrant Filter.

    支持的运算符: $gte, $lte, $gt, $lt, $eq, $ne, $in, $contains, $and, $or
    """
    conditions = []

    for key, value in chroma_where.items():
        if key == "$and":
            # $and: [{"key": {"$gte": v}}, {"key": {"$lte": v}}]
            # → Qdrant Filter(must=[...])
            sub_conditions = []
            for sub_clause in value:
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            conditions.append(models.Filter(must=sub_conditions))
        elif key == "$or":
            # $or: [{"key1": {"$eq": v1}}, {"key2": {"$eq": v2}}]
            # → Qdrant Filter(should=[...])
            sub_conditions = []
            for sub_clause in value:
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            conditions.append(models.Filter(should=sub_conditions))
        else:
            conditions.append(_build_condition(key, value))

    return models.Filter(must=conditions)


def _build_condition(key: str, value) -> models.Condition:
    """构建单个字段条件。
    
    注意: $and / $or 由上层 _translate_filter 处理，不进入本函数。
    """
    if not isinstance(value, dict):
        # 简单等值: {"heat": "hot"}
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    for op, val in value.items():
        if op == "$gte":
            return models.FieldCondition(key=key, range=models.Range(gte=val))
        elif op == "$lte":
            return models.FieldCondition(key=key, range=models.Range(lte=val))
        elif op == "$gt":
            return models.FieldCondition(key=key, range=models.Range(gt=val))
        elif op == "$lt":
            return models.FieldCondition(key=key, range=models.Range(lt=val))
        elif op == "$eq":
            return models.FieldCondition(key=key, match=models.MatchValue(value=val))
        elif op == "$ne":
            return models.Filter(must_not=[
                models.FieldCondition(key=key, match=models.MatchValue(value=val))
            ])
        elif op == "$in":
            return models.FieldCondition(key=key, match=models.MatchAny(any=val))
        elif op == "$contains":
            # ChromaDB $contains → Qdrant MatchText (需 text index)
            return models.FieldCondition(key=key, match=models.MatchText(text=str(val)))

    raise ValueError(f"Unsupported operator in: {value}")
```

**覆盖验证**（对照实际代码中的 where 子句）：

| 位置 | ChromaDB where | Qdrant 翻译 |
|------|---------------|-------------|
| `chroma.py:275` | `{"emotional_intensity": {"$gte": 1}}` | `Range(gte=1)` ✅ |
| `chroma.py:562` | `{"timestamp": {"$gte": since_ts}}` | `Range(gte=since_ts)` ✅ |
| `chroma.py:608` | `{"timestamp": {"$lt": before_ts}}` | `Range(lt=before_ts)` ✅ |
| `pipeline.py:408` | `{"heat": "hot"}` | `MatchValue("hot")` ✅ |
| `pipeline.py:420` | `{"heat": {"$in": ["warm","cool"]}}` | `MatchAny(["warm","cool"])` ✅ |
| `dispatch.py:615` | `{"$and": [w, {"$or": [{"ev":{"$eq":v}}, {"evb":{"$eq":v}}]}]}` | `Filter(must=[..., Filter(should=[MatchValue(v), MatchValue(v)])])` ✅ |
| `dispatch.py:697` | `{"$and": [{"ts":{"$gte":...}}, {"ts":{"$lte":...}}]}` | `Filter(must=[Range(gte...), Range(lte...)])` ✅ |
| `dispatch.py:762` | `{"tags": {"$contains": kw}}` | `MatchText(kw)` ⚠️ 见下文 |
| `context.py:653` | `{"tags": {"$contains": t}}` | `MatchText(t)` ⚠️ 见下文 |

> ⚠️ **$contains → MatchText 语义差异（仅影响 tags 和 document 的 bm25 替代路径）**: ChromaDB `$contains` 是子字符串匹配。Qdrant `MatchText` 是**分词后匹配**。对于 `dispatch.py:762` 和 `context.py:653` 的旧 `$contains` 查询：如果 inverted_index 已处理标签精确匹配，这些旧查询路径应改为走 `inverted_index.query_tags()` 或 Qdrant `entities[].text` keyword 匹配（原生 `list[dict]` 支持嵌套索引），不再依赖 MatchText 的子串行为。

---

## 6. API 映射：Ollama → vLLM

### 6.1 Embedding

```python
# ── 单条 embed ──
# Ollama:  POST /api/embeddings  {"model":"bge-m3","prompt":t}
# vLLM:    POST /v1/embeddings   {"model":"bge-m3","input":t}
# 响应映射:
#   Ollama:  resp["embedding"]                    # list[float]
#   vLLM:    resp["data"][0]["embedding"]          # list[float]
# 
# 改动范围: embed.py _embed_via_ollama() → _embed_via_vllm()
# 约 10 行改动

# ── 批量 embed ──
# Ollama:  POST /api/embed    {"model":"bge-m3","input":[...]}
# vLLM:    POST /v1/embeddings {"model":"bge-m3","input":[...]}
# 响应映射:
#   Ollama:  resp["embeddings"][i]                # list[float]
#   vLLM:    resp["data"][i]["embedding"]          # list[float]
#
# 改动范围: embed.py _embed_via_ollama_batch() → _embed_via_vllm_batch()
# 约 10 行改动

# ── 不受影响的 ──
# local_embed()           — 四级缓存不变
# local_embed_batch()     — 批量逻辑不变
# local_embed_async()     — asyncio 包装不变
# 请求合并器 (_coalesced_embed) — 不变
# n-gram 近似缓存          — 不变
# 全局 LRU 缓存           — 不变
# 请求级缓存              — 不变
```

### 6.2 本地 LLM (摘要 / 实体抽取)

```python
# ── 摘要 (LocalLLM.generate / summarize) ──
# Ollama:  POST /api/generate  {"model":"qwen2.5:3b","prompt":p,"stream":false}
# vLLM:    POST /v1/chat/completions  {
#             "model":"qwen2.5:3b",
#             "messages":[{"role":"user","content":p}],
#             "temperature":0.3,"max_tokens":1024}
# 响应映射:
#   Ollama:  resp["response"]
#   vLLM:    resp["choices"][0]["message"]["content"]
#
# 改动范围: local.py LocalLLM.generate()
# 约 10 行改动

# ── 实体抽取 (brain/semantic.py) ──
# Ollama:  POST /api/chat  {...}
# vLLM:    POST /v1/chat/completions  {...}
# 响应映射: 同上
#
# 改动范围: brain/semantic.py _extract_entities_via_ollama()
# 约 10 行改动

# ⚠️ 注意: qwen2.5:3b 的 Ollama model name 可能和 vLLM 注册名不同
#    需要 settings.py 中配置 VLLM_CHAT_MODEL
```

### 6.3 配置变更

```python
# settings.py 删除:
OLLAMA_EMBED_MODEL = "bge-m3"
LOCAL_LLM_OLLAMA_URL = "http://localhost:11434"

# settings.py 新增:
VLLM_EMBED_URL = "http://localhost:8001"     # vLLM 实例1: bge-m3
VLLM_CHAT_URL = "http://localhost:8002"      # vLLM 实例2: qwen2.5:3b
VLLM_EMBED_MODEL = "bge-m3"                  # vLLM 中注册的名字
VLLM_CHAT_MODEL = "qwen2.5:3b"               # vLLM 中注册的名字

# EMBED_MODELS dict 中的 provider 字段:
# "provider": "ollama" → "provider": "vllm"

# 回滚开关:
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "vllm")  # "ollama" for rollback
```

---

## 7. SQLite 迁移策略

### 7.1 CoOccurrenceTracker → 独立 Qdrant Collection

> **架构决策（评审修正）**: 原方案将 `co_occurring` JSON 塞入每条 memory 的 payload。被否决——热门记忆与 500+ 条共现时 JSON 膨胀，PersonaSymmetry 全局 Top-N 排序无法实现。改为独立 collection。

**现状 SQL** (cooccur.py):

```sql
-- record(): INSERT ... ON CONFLICT DO UPDATE
INSERT INTO cooccurrence(id_a, id_b, count, last_time)
VALUES (?, ?, 1, ?) ON CONFLICT(id_a, id_b) DO UPDATE SET
count = cooccurrence.count + 1, last_time = excluded.last_time

-- query(): SELECT where id_a IN (...) OR id_b IN (...)
SELECT id_a, id_b, count FROM cooccurrence
WHERE id_a IN (?,?,...) OR id_b IN (?,?,...)
```

**目标 Qdrant** (collection `co_occurrence`):

```python
class CoOccurrenceStore:
    """独立 Qdrant collection: co_occurrence
    
    每条 point = 一对记忆的共现次数。
    支持高效 query() 和 export_for_symmetry()。
    """

    def record(self, memory_ids: list[str]):
        """同轮出现的记忆对，count += 1。
        
        合并器锁保护，一次持锁内完成全部读写。
        """
        now = time.time()
        pairs = []
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                a, b = sorted([memory_ids[i], memory_ids[j]])
                pairs.append((a, b))

        if not pairs:
            return

        # 批量检索现有点
        point_ids = [f"{a}||{b}" for a, b in pairs]
        existing = self._client.retrieve(
            collection_name="co_occurrence",
            ids=point_ids,
            with_payload=["count"],
        )
        existing_map = {p.id: (p.payload or {}).get("count", 0) for p in existing}

        # 批量 upsert
        points = []
        for (a, b), pid in zip(pairs, point_ids):
            new_count = existing_map.get(pid, 0) + 1
            points.append(models.PointStruct(
                id=pid,
                vector=self._get_memory_embedding(a),  # id_a 的 embedding
                payload={"id_a": a, "id_b": b, "count": new_count, "last_time": now},
            ))
        self._client.upsert(collection_name="co_occurrence", points=points)

    def query(self, memory_ids: list[str]) -> list[dict]:
        """查询给定记忆集的所有共现伙伴，按 count 降序。
        
        替代原 SQL: WHERE id_a IN (...) OR id_b IN (...)
        """
        mset = set(memory_ids)
        # Qdrant scroll 不支持 OR，分两次 scroll 然后合并
        # 上限: 传入 ID 数 × 100，最低 1000，最高 5000——防止热门记忆膨胀
        max_per_field = max(1000, min(len(memory_ids) * 100, 5000))
        all_points = []
        for field in ["id_a", "id_b"]:
            points, _ = self._client.scroll(
                collection_name="co_occurrence",
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key=field, match=models.MatchAny(any=memory_ids)),
                ]),
                with_payload=["id_a", "id_b", "count"],
                limit=max_per_field,
            )
            all_points.extend(points)

        # 合并去重，排除自身，按 count 降序
        seen, partners = set(memory_ids), {}
        for pt in all_points:
            p = pt.payload
            partner = p["id_b"] if p["id_a"] in mset else p["id_a"]
            if partner not in seen:
                partners[partner] = partners.get(partner, 0) + p["count"]

        return sorted(
            [{"id": k, "count": v} for k, v in partners.items()],
            key=lambda x: -x["count"],
        )

    def export_for_symmetry(self, limit: int = 10000) -> dict[str, dict[str, int]]:
        """全局 Top-N 共现导出，供 PersonaSymmetry 使用。
        
        替代原 SQL: SELECT ... ORDER BY count DESC LIMIT 10000
        """
        points, _ = self._client.scroll(
            collection_name="co_occurrence",
            with_payload=["id_a", "id_b", "count"],
            order_by=models.OrderBy(key="count", direction=models.Direction.DESC),
            limit=limit,
        )
        data: dict[str, dict[str, int]] = {}
        for pt in points:
            a, b, cnt = pt.payload["id_a"], pt.payload["id_b"], pt.payload["count"]
            data.setdefault(a, {})[b] = cnt
            data.setdefault(b, {})[a] = cnt
        return data
```

**LTD 衰减**: 改为检索时降权，删除定时全表 UPDATE 任务：

```python
def _apply_ltd_decay(self, score: float, last_time: float) -> float:
    """查询时衰减: 超过7天降权，超过30天减半，超过90天归零。"""
    days = (time.time() - last_time) / 86400
    if days > 90:
        return 0.0
    if days > 30:
        return score * 0.5
    if days > 7:
        return score * max(0.5, 1.0 - days / 14)
    return score
```

### 7.2 EntityPairTracker → 入库时预计算

**策略变更（评审修正）**: 原方案为「查询时从 entities payload 实时聚合」——对 path4(entity_match) 需要 retrieve 50-200 条记忆并逐条 JSON 解析，性能远差于一次 SQL 查询。改为**入库时预计算**，结果存入 payload 的 `entity_co_counts` 字段。

**入库时**（在 `_store_conversation()` 的 entities 提取后）:

```python
def _update_entity_co_counts(self, memory_id: str, entities: list[dict]):
    """入库后更新该条记忆涉及实体的全局共现计数。
    
    策略: 对 payload entity_co_counts 字段做增量写——
    定期（每 100 次入库）从 Qdrant 取样重算，非实时全量。
    entities 现在是原生 list[dict]，通过 entities[].text keyword 匹配查询。
    """
    # 简化方案: 不单独维护 entity_pair 表
    # entity_co_counts 在入库时预计算一次，后续检索直接用
    entity_names = [e.get("text", "") for e in entities if e.get("text")]
    if not entity_names:
        return

    # 从 Qdrant 查询这些 entity 的当前计数，取 top 50
    existing_memories = self._client.scroll(
        collection_name=self._coll,
        scroll_filter=models.Filter(should=[
            models.FieldCondition(key="entities[].text", match=models.MatchValue(value=en))
            for en in entity_names
        ]),
        with_payload=["entity_co_counts"],
        limit=50,
    )[0]

    # 聚合 + 更新当前记忆的 entity_co_counts
    aggregated: dict[str, int] = {}
    for mem in existing_memories:
        prev = (mem.payload or {}).get("entity_co_counts", {})
        for en, cnt in prev.items():
            aggregated[en] = aggregated.get(en, 0) + cnt

    self._client.set_payload(
        collection_name=self._coll,
        payload={"entity_co_counts": aggregated},
        points=[memory_id],
    )
```

**查询时**（替代 entity_pair query）:

```python
def get_entity_co_counts(self, memory_ids: list[str]) -> dict[str, int]:
    """从 payload 直接读 entity_co_counts，替代 SQLite entity_pair query。"""
    points = self._client.retrieve(
        collection_name=self._coll,
        ids=memory_ids,
        with_payload=["entity_co_counts"],
    )
    merged: dict[str, int] = {}
    for pt in points:
        co = (pt.payload or {}).get("entity_co_counts", {})
        for en, cnt in co.items():
            merged[en] = merged.get(en, 0) + cnt
    return dict(sorted(merged.items(), key=lambda x: -x[1]))
```

> ⚠️ **概率近似**: `entity_co_counts` 是入库时从 top-50 记忆采样聚合的，非全量精确计数。在百万级下 50 条样本覆盖率仅 0.05%。这是一个**检索排序用的辅助信号**——不应作为精确过滤条件（如"只返回 entity_co_counts >= 3 的结果"）。如需精确实体共现，走 `hyper_edges` collection 的 entity_index。
>
> **entity_pair 删除对检索管线的影响**: entity_pair 的所有调用都在写入路径（`context.py:608` 的 `record()`），不涉及检索管线。pipeline.py 的 entity_match 路径走的是 `inverted_index.get_exact()`，两者无耦合。entity_pair 的迁移是纯粹的写入优化——实体对计数从 SQLite 搬到 payload `entity_co_counts` 字段，检索阶段不可见。

### 7.3 HyperEdgeIndex → 独立 Qdrant Collection

> **现状**: SQLite 三表（`hyper_edge` + `entity_index` + `entity_edge`），461 行。8 个公开方法：`record` / `expand` / `get_memory_ids` / `cluster_key` / `cluster_entities` / `remove_memory` / `clear` / `stats`。
> **目标**: Qdrant 独立 collection `hyper_edges`，单表存储。`cluster_key` 是纯内存集合运算不调存储，其余 7 个方法全部重写。

#### 7.3.1 数据模型（复用 §4.3）

```yaml
Collection: hyper_edges
  id: auto-generated (UUID)
  vector: [1024-dim float32]                  # 参与实体的平均 bge-m3 embedding
  payload:
    entities: '["Python","VSCode","IDE"]'     # 排序后的实体列表 (JSON string)
    memory_ids: '["id1","id2","id3"]'         # 关联记忆 ID (JSON string)
    created_at: "2026-06-15T10:30:00"         # ISO 时间戳
    edge_size: 3                              # 实体数（冗余，加速过滤）
```

**Payload 索引**（Phase 4 创建）：
- `entities` — keyword index（`MatchValue` 查询，原生 `list[str]`）
- `created_at` — float/date index（裁剪排序）
- `edge_size` — integer index（可选，过滤大超边）

#### 7.3.2 完整 Pseudocode

```python
class HyperEdgeStore:
    """独立 Qdrant collection: hyper_edges

    替代 SQLite 三表结构。每条 point = 一个超边（一组实体 + 关联记忆ID）。
    MAX_EDGES=10000 上限约束总行数，裁剪时保留最近一半。
    """

    MAX_EDGES = 10000
    EXPAND_TOP_K = 10
    SCROLL_LIMIT_PER_ENTITY = 500   # 单实体展开时最多滚动的超边数

    def __init__(self, client, collection_name: str, embed_fn):
        self._client = client
        self._coll = collection_name
        self._embed_fn = embed_fn    # local_embed_batch, 用于计算实体平均向量

    # ═══════════════════════════════════════════════════
    # 写入
    # ═══════════════════════════════════════════════════

    def record(self, entities: list[str], memory_id: str):
        """记录一组实体在同一段对话中共现。

        替代原 SQLite: INSERT hyper_edge + UPDATE entity_index(entity+co_entities+memory_ids)
                       + entity_edge (三表事务, ~45行)
        """
        entities = sorted(set(e for e in entities
                              if isinstance(e, str) and len(e) >= 2))
        if len(entities) < 2:
            return

        # 计算参与实体的平均向量
        avg_vec = self._compute_avg_embedding(entities)

        point_id = str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._coll,
            points=[models.PointStruct(
                id=point_id,
                vector=avg_vec,
                payload={
                    "entities": entities,           # 原生 list[str]
                    "memory_ids": [memory_id],      # 原生 list[str]
                    "created_at": datetime.utcnow().isoformat(),
                    "edge_size": len(entities),
                },
            )],
        )

        # 超边数超限 → 裁剪最老的一半
        total = self._client.count(collection_name=self._coll).count
        if total > self.MAX_EDGES:
            self._prune()

    # ═══════════════════════════════════════════════════
    # 查询
    # ═══════════════════════════════════════════════════

    def expand(self, entity_names: list[str], top_k: int = None) -> dict[str, int]:
        """展开实体 → 返回 {related_entity: total_weight}。

        替代原 SQL: SELECT co_entities FROM entity_index WHERE entity = ?
                    (逐实体 N+1 查询, ~12行)
        新方案: 对每个输入实体 scroll 匹配超边，客户端聚合权重。
        """
        if top_k is None:
            top_k = self.EXPAND_TOP_K
        if not entity_names:
            return {}

        input_set = set(entity_names)
        scores: dict[str, int] = defaultdict(int)

        for ename in entity_names:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="entities", match=models.MatchValue(value=ename)
                    ),
                ]),
                with_payload=["entities"],
                limit=self.SCROLL_LIMIT_PER_ENTITY,
            )
            for pt in pts:
                try:
                    edge_entities = set(pt.payload["entities"])
                except (TypeError, KeyError):
                    continue
                # 每条匹配超边中的非输入实体各 +1 权重
                for e in edge_entities - input_set:
                    scores[e] += 1

        return dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])

    def get_memory_ids(self, entity_names: list[str],
                       max_memories: int = 50) -> list[str]:
        """给定实体名，收集所有关联超边的记忆 ID，按出现次数降序。

        替代原 SQL: entity_edge JOIN hyper_edge (逐实体 N+1, ~20行)
        """
        if not entity_names:
            return []

        scored: dict[str, int] = defaultdict(int)
        for ename in entity_names:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="entities", match=models.MatchValue(value=ename)
                    ),
                ]),
                with_payload=["memory_ids"],
                limit=self.SCROLL_LIMIT_PER_ENTITY,
            )
            for pt in pts:
                try:
                    mids = pt.payload.get("memory_ids", [])
                except (TypeError, KeyError):
                    mids = []
                for mid in mids:
                    scored[mid] += 1

        return [mid for mid, _ in
                sorted(scored.items(), key=lambda x: -x[1])[:max_memories]]

    def cluster_key(self, entities: list[str],
                    existing_groups: list[set[str]],
                    min_overlap: int = 2) -> int | None:
        """纯内存集合运算——不调 Qdrant。逻辑与旧实现完全一致。"""
        if not entities:
            return None
        entity_set = set(entities)
        best_idx, best_overlap = None, 0
        for i, group in enumerate(existing_groups):
            overlap = len(entity_set & group)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        return best_idx if best_overlap >= min_overlap else None

    def cluster_entities(self, entities: list[str],
                         min_overlap: int = 2) -> set[str]:
        """通过超边扩展实体集合。

        替代原 SQL: entity_edge JOIN hyper_edge (两次查询, ~25行)
        """
        if not entities:
            return set()

        input_set = set(entities)
        result = set(entities)

        for ename in entities:
            pts, _ = self._client.scroll(
                collection_name=self._coll,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="entities", match=models.MatchValue(value=ename)
                    ),
                ]),
                with_payload=["entities"],
                limit=200,
            )
            for pt in pts:
                try:
                    edge_entities = set(pt.payload["entities"])
                except (TypeError, KeyError):
                    continue
                if len(input_set & edge_entities) >= min_overlap:
                    result |= edge_entities

        return result

    # ═══════════════════════════════════════════════════
    # 维护
    # ═══════════════════════════════════════════════════

    def remove_memory(self, memory_id: str):
        """删除记忆时同步清理超边。

        替代原 SQL: 全表 scan hyper_edge → 条件更新/删除 → 清理 entity_edge
                    → 清理 entity_index (40行, 三表操作)
        新方案: scroll 全量超边（上限 10000），客户端过滤，批量更新。
        """
        pts, _ = self._client.scroll(
            collection_name=self._coll,
            with_payload=["memory_ids"],
            limit=self.MAX_EDGES + 100,  # 安全余量
        )

        updates, deletes = [], []
        for pt in pts:
            try:
                mids = pt.payload.get("memory_ids", [])
            except (TypeError, KeyError):
                mids = []
            if memory_id not in mids:
                continue
            mids.remove(memory_id)
            if mids:
                updates.append((pt.id, mids))
            else:
                deletes.append(pt.id)

        # 批量更新
        for pt_id, new_mids in updates:
            self._client.set_payload(
                collection_name=self._coll,
                payload={"memory_ids": new_mids},
                points=[pt_id],
            )
        if deletes:
            self._client.delete(
                collection_name=self._coll,
                points_selector=deletes,
            )

    def _prune(self):
        """裁剪最老的超边，保留最近一半。

        替代原 SQL: DELETE oldest edges (子查询) → DELETE orphan entity_edge
                    → DELETE + 重建 entity_index (60行, 三表操作)
        """
        keep = self.MAX_EDGES // 2
        pts, _ = self._client.scroll(
            collection_name=self._coll,
            with_payload=["created_at"],
            order_by=models.OrderBy(
                key="created_at", direction=models.Direction.ASC
            ),
            limit=self.MAX_EDGES,
        )
        if len(pts) <= keep:
            return

        to_delete = [pt.id for pt in pts[:len(pts) - keep]]
        self._client.delete(
            collection_name=self._coll,
            points_selector=to_delete,
        )
        logger.info("超边索引裁剪: %d → %d", len(pts), len(to_delete))

    def clear(self):
        """清空所有超边。删除并重建 collection 比逐条删快。"""
        self._client.delete_collection(self._coll)
        # QdrantService 负责重建 collection schema

    def stats(self) -> dict:
        total = self._client.count(collection_name=self._coll).count
        return {"total_hyperedges": total}

    # ═══════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════

    def _compute_avg_embedding(self, entities: list[str]) -> list[float]:
        """计算实体名称列表的平均 bge-m3 embedding。

        用于超边的向量字段——参与实体的语义中心点。
        向量仅用于「语义检索相关超边」这一辅助场景；
        核心查询路径（expand/get_memory_ids）走 scroll + MatchValue（keyword 匹配），不依赖向量。
        """
        embs = self._embed_fn(entities)
        valid = [e for e in embs if e is not None]
        if not valid:
            return [0.0] * 1024
        n = len(valid)
        return [sum(dim) / n for dim in zip(*valid)]
```

#### 7.3.3 与旧实现的差异

| 方面 | SQLite 三表 | Qdrant 单 collection |
|------|------------|---------------------|
| `record()` | 三表事务 (INSERT hyper_edge + INSERT entity_edge + UPDATE entity_index) | 单次 upsert |
| `expand()` | `SELECT co_entities FROM entity_index` (一次查询) | 每输入实体一次 scroll + 客户端聚合 |
| `get_memory_ids()` | `entity_edge JOIN hyper_edge` (一次 JOIN) | 每输入实体一次 scroll |
| `remove_memory()` | 全表 scan + 三表清理 (40行) | scroll 全量 (≤10K) + 客户端过滤 + 批量 set_payload |
| `_prune()` | DELETE + 重建 entity_index (60行) | scroll → 批量 delete |
| `cluster_key()` | 无变化 | 纯内存运算，不调存储 |

> ⚠️ **expand/get_memory_ids 从单次 SQL 查询变为 N 次 scroll**：当输入实体数较少时（典型 3-8 个），这是可接受的。如果未来出现 >20 个实体的查询场景，优化为单次 scroll + `should` filter（但这要求 Qdrant 支持 OR 语义的 keyword 匹配——`MatchAny` 替代逐 entity 的 `MatchValue` scroll）。

---

## 8. 检索管线重设计

### 8.1 检索管线不变：9 路并行，仅 API 翻译

> **设计原则**: 不合并路径、不改变分数体系、不改 heat 过滤逻辑。旧 10 路中 9 路
> 结构完全不变，仅底层 ChromaDB API 翻译为 Qdrant API。唯一删除：bm25_fulltext
> （Qdrant MatchText 原生替代）。
>
> **inverted_index 保留**（157 行，全 Python，零存储依赖）——`build()` / `build_tags()`
> 改从 Qdrant scroll 获取数据（context.py 初始化时），公开 API（`query` / `query_tags` /
> `get_exact` / `add` / `add_tags` / `remove`）完全不变。依赖 inverted_index 的 5 条
> 检索路径（keyword / tag_match / entity_match / temporal / topic）**代码一行不改**。

#### 路径对照

```
旧 10 路                                  新 10 路                        pipeline.py
─────────────────────────────────────    ─────────────────────────────    ──────────
① semantic_hot    ChromaDB query         Qdrant search + heat="hot"      ✏️ API翻译
② semantic_cool   ChromaDB query         Qdrant search + heat IN warm/cool ✏️ API翻译
③ keyword         inverted_index.query   ←── 不变 ──→                   ✅ 不动
④ tag_match       inverted_index.query_tags ←── 不变 ──→                ✅ 不动
⑤ entity_match    inverted_index.get_exact ←── 不变 ──→                 ✅ 不动
⑥ temporal        temporal_index.query()  temporal_index.query()         ✅ 不动
                   → inverted_index       → inverted_index (同上)
⑦ topic_tree      topic_tree.expand()    topic_tree.expand()             ✅ 不动
                   → inverted_index       → inverted_index (同上)
⑧ attention_drift ChromaDB query         Qdrant search                   ✏️ API翻译
⑨ bm25_fulltext   BM25 → col.get(ids)    Qdrant scroll + MatchText       ✏️ 重写
⑩ ai_expression   ChromaDB query         Qdrant search                   ✏️ API翻译
+ co_occurrence   co_tracker.query()     CoOccurrenceStore.query()       ✏️ 换后端
```

#### 逐路 API 翻译 (仅 4 路需要改)

**①② semantic_hot / semantic_cool — heat 二段检索原封不动**

```python
# ① semantic_hot（原封不动的二段检索，仅 API 换）
# ChromaDB (旧):
hot = col.query(query_embeddings=[q], n_results=min(sem_n, 200),
                where={"heat": "hot"},
                include=["documents", "metadatas", "distances"])
# Qdrant (新):
hot = client.search(collection_name=coll, query_vector=q,
    query_filter=Filter(must=[FieldCondition(key="heat", match=MatchValue(value="hot"))]),
    limit=min(sem_n, 200), with_payload=True)

# ② semantic_cool — 不变
cool = client.search(collection_name=coll, query_vector=q,
    query_filter=Filter(must=[FieldCondition(key="heat", match=MatchAny(any=["warm","cool"]))]),
    limit=remain, score_threshold=1.0 - MIN_SIMILARITY, with_payload=True)
```

**⑧ attention_drift — 不变**

```python
# ChromaDB (旧):
results = col.query(query_embeddings=[center], n_results=10, ...)
# Qdrant (新):
results = client.search(collection_name=coll, query_vector=center,
    limit=10, with_payload=True)
```

**⑨ bm25_fulltext → Qdrant MatchText（唯一真正删除的路径）**

```python
# 旧: BM25 内存索引全文搜索 → col.get(ids=bm25_ids)
# 新: Qdrant scroll + MatchText on document
results, _ = client.scroll(
    collection_name=coll,
    scroll_filter=Filter(must=[
        FieldCondition(key="document", match=MatchText(text=user_message)),
    ]),
    limit=100 if _BM else 20,
    with_payload=True,
)
# 固定分 0.35 不变
local = [_make_mem(r.id, r.payload, r.payload.get("document",""), 0.35, "bm25_fulltext")
         for r in results]
```

> ⚠️ MatchText 对中文的分词取决于 Qdrant tokenizer。此路径仅替代 bm25（原固定分 0.35，在 10 路中是最弱的辅助信号），即使分词有偏差也影响有限。Phase 0.5 验证对完整中文句子的命中率即可——不需要逐标签测试。

**⑩ ai_expression — 不变**

```python
# ChromaDB (旧):
ai_results = ai_col.query(query_embeddings=[q], n_results=5, ...)
# Qdrant (新):
ai_results = client.search(collection_name=ai_coll, query_vector=q,
    limit=5, with_payload=True)
```

**co_occurrence expand — CoOccurrenceStore.query()（§7.1）**

#### inverted_index 适配（context.py 初始化时，非 pipeline.py）

```python
# inverted_index.py 本身零改动——它是纯 Python 数据结构。
# 仅 context.py 中 build/build_tags 调用的数据源切换：

# ChromaDB (旧):
summaries = [(r["id"], r["metadata"]["summary"]) for r in col.get(include=["metadatas"])]
self.inverted_index.build(summaries)

# Qdrant (新):
pts, _ = client.scroll(collection_name=coll, with_payload=["summary"], limit=100000)
summaries = [(pt.id, pt.payload.get("summary","")) for pt in pts]
self.inverted_index.build(summaries)

# build_tags 同理：Qdrant scroll with_payload=["tags"] 替代 ChromaDB get
```

#### 删除模块清单

```
✗ app/retrieval/bm25_fulltext.py — Qdrant MatchText on document 原生替代
```

### 8.2 全量加载清零清单

| 位置 | 全量操作 | 替换 |
|------|---------|------|
| `chroma.list_all()` | 全量 `get()` | Qdrant `scroll()` 分页 |
| `chroma.list_all_cached()` | 全量 `get()` + 5min TTL | **删除方法**。分页 scroll 下"全量缓存"无意义——调用方 (`list_memories`, `consolidation`) 改为各自加 filter 的 scroll。consolidation 需要全量扫时走分页 scroll（带 timestamp filter 缩小范围）；API 分页直接走 Qdrant scroll + offset/limit |
| `chroma.query_by_emotion()` | `list_all()` → Python filter | Qdrant `scroll` + `Range` filter |
| `chroma.get_memories_by_timerange()` | `list_all()` → Python filter | Qdrant `scroll` + `Range` filter |
| `chroma._apply_emotional_desensitization()` | `get(where=...)` 全量 | Qdrant `scroll` + filter（无需分页，scroll 原生支持流式读取全部匹配行） |
| `chroma._build_embedding_cache()` | 全量 ID + 分批 `get` | Qdrant `scroll` 取最近 N 条（按 `last_hit_time DESC`） + LRU 上限，启动时不阻塞 |
| `dispatch._get_chroma_collection()` | `get(where=...)` | Qdrant `scroll` + filter |
| `bm25_fulltext` | 全量 `get(docs)` 到内存 | **删除模块**，Qdrant text index 替代 |
| `inverted_index` build/build_tags | 全量 `get()` 拉取 summaries/tags | Qdrant `scroll()` 按需拉取，公开 API 不变 |
| `cooccur.query()` | SQL `WHERE id_a IN (...) OR id_b IN (...)` | Qdrant `co_occurrence` collection scroll（独立 collection） |
| `cooccur.export_for_symmetry()` | SQL `ORDER BY count DESC LIMIT 10000` | Qdrant `co_occurrence` scroll + order_by count DESC |
| `hyperedge.expand()` | 全量 scan 超边表 | Qdrant scroll + filter |
| `hyperedge.expand()` | SQLite `SELECT co_entities FROM entity_index` (逐实体 N+1) | Qdrant `hyper_edges` scroll + keyword MatchValue + 客户端聚合（§7.3.2 `expand()`） |
| `hyperedge.get_memory_ids()` | SQLite `entity_edge JOIN hyper_edge` (逐实体 N+1) | Qdrant `hyper_edges` scroll + keyword MatchValue + 客户端聚合（§7.3.2 `get_memory_ids()`） |
| `hyperedge.cluster_entities()` | SQLite `entity_edge JOIN hyper_edge` (两次查询) | Qdrant `hyper_edges` scroll + keyword MatchValue（§7.3.2 `cluster_entities()`） |
| `hyperedge.remove_memory()` | SQLite 全表 scan + 三表清理 (40行) | Qdrant `hyper_edges` scroll 全量 (≤10K) + 批量 set_payload（§7.3.2 `remove_memory()`） |
| `hyperedge._prune()` | SQLite DELETE + 重建 entity_index (60行) | Qdrant `hyper_edges` scroll order_by created_at → 批量 delete（§7.3.2 `_prune()`） |
| `consolidation` 巩固扫描 | `list_all()` / `list_since()` | Qdrant scroll + filter |
| `tag_index` | `embed_fn()` 批量标签 | 从 Qdrant payload 读 tags 列 |

---

## 9. Phase 分步执行计划

```
Phase 0  ██░░░░░░░░  infra: docker-compose 加 Qdrant + vLLM×2            1天
Phase 0.5 █░░░░░░░░  原型验证: 1000条真实数据端到端，验证关键假设            1天
Phase 1  ███░░░░░░░  vLLM + Qdrant 核心层（embed HTTP + QdrantService）   4天
                    (合并原 Phase 1+2，避免混合向量库风险 — 评审 §5)
Phase 2  ██░░░░░░░░  杀全量模式（list_all→scroll，filter 翻译层）          2天
Phase 3  ███░░░░░░░  SQLite 元数据迁 Qdrant（cooccur/hyperedge 迁移）      3天
Phase 4  ███░░░░░░░  百万级硬骨头（量化、分区、embedding 缓存上限）          3天
Phase 5  ██░░░░░░░░  清理（删旧代码、改测试、更新文档）                     2天
        ──────────────────────────────────────────────────────────
        合计                                                                 16天

每个 Phase 结束:
  ✅ pytest tests/ 全部通过
  ✅ E2E 主链路通过
  ✅ 系统可启动运行
```

### Phase 0 交付物 (infra)

- [ ] `docker-compose.yml` 加 `qdrant` + `vllm-embed` + `vllm-chat` 三个服务
- [ ] `app/config/settings.py` 加 Qdrant + vLLM×2 配置项（含回滚开关 STORAGE_BACKEND / EMBED_PROVIDER）
- [ ] **GPU 资源规划**: bge-m3 ~1.3GB VRAM (vLLM 实例1)，qwen2.5:3b ~6GB VRAM (vLLM 实例2)，Qdrant 可跑 CPU。单卡 ≥8GB 可全承载
- [ ] 连通性验证脚本：Qdrant 创建 collection、vLLM embed 返回正确维度 (1024)
- [ ] 数据迁移脚本 `scripts/migrate_to_qdrant.py` 完成以下核心逻辑：
      - **分批读取**: ChromaDB `get(limit=N, offset=M)` 分批导出，每批 500 条，避免全量加载
      - **格式转换**: ChromaDB metadata → Qdrant payload（字段名不变）。关键类型转换：
        - `entities`: ChromaDB JSON 字符串 `'[{"text":"...","type":"..."}]'` → Qdrant 原生 `list[dict]`（`json.loads()`）
        - `entity_co_counts`: 从旧 SQLite entity_pair 表聚合 → Qdrant 原生 `dict[str,int]`
        - hyper_edges `entities`/`memory_ids`: 旧 SQLite JSON 字符串 → Qdrant 原生 `list[str]`（`json.loads()`）
        - 其他字段（`tags`/`heat`/`summary` 等）保持原有字符串/数值类型，直接赋值
      - **进度显示**: tqdm 进度条 + 每 1000 条打印吞吐量
      - **校验**: 迁移后逐批对比 `count()` — ChromaDB vs Qdrant；抽样 100 条对比 id/document/embedding 三字段一致
      - **断点续传**: 用 `data/migration_checkpoint.json` 记录已完成批次，中断后跳过已迁移的 batch
      - **干跑模式**: `--dry-run` 只校验不写入，用于 Phase 0 测试连通性
- [ ] `TEST_BACKEND=qdrant|chromadb` 环境变量引入，在 Phase 1-3 过渡期允许测试动态选择后端

### Phase 0.5 交付物 (原型验证) ⚠️ 通过后才能进 Phase 1

- [ ] 从现有 ChromaDB 导出 1000 条真实记忆
- [ ] vLLM bge-m3 输出向量 vs Ollama bge-m3 余弦相似度 ≥ 0.99
- [ ] Qdrant HNSW 参数对比 ChromaDB 召回率（m=16, ef_construct=100, ef=64）
- [ ] CoOccurrence 独立 collection 方案的 query/record/export_for_symmetry 原型跑通
- [ ] Qdrant text index 对逗号分隔中文标签的匹配精度（仅影响 bm25 替代路径，见 R10）
- [ ] 原型结论写入本文档「附录 C: 原型验证结果」

**Phase 0.5 通过标准（全部必须满足才能进入 Phase 1）：**

| 验证项 | 阈值 | 不通过处理 |
|--------|------|-----------|
| vLLM vs Ollama 同文本向量余弦相似度 | ≥ 0.99（100 条样本，取最小值） | 全量重建 embedding；若 <0.95 则回退到 Ollama，vLLM 部署方案需重新评估 |
| Qdrant HNSW 召回率 vs ChromaDB | ≥ 0.95（top-50 search，200 条查询） | 调参 (m/ef_construct/ef)；两次迭代仍不达标则降级为 exact search |
| Qdrant text index 中文标签匹配 | 逗号分隔标签 `"Python,Rust,编程"` → `MatchText("编程")` 必须返回 true | 切换到 keyword 数组方案，tags 存储格式从逗号分隔字符串改为 JSON 数组 |
| Qdrant text index 子串匹配 | `"编程语言"` vs `MatchText("编程")` 行为文档化（不强制要求通过） | 评估对 `dispatch.py:762` / `context.py:653` 的 `$contains` 查询的影响，变更策略写入附录 C |
| CoOccurrence 独立 collection | 10K 条数据 record/query/export_for_symmetry 延迟 <100ms | 保留 SQLite cooccur 表作为备选方案，CoOccurrenceStore 双写两边 |
| Embedding 兼容性 | `local_embed()` 签名/返回值格式不变；请求合并器正常工作 | 修改 embed.py 适配层直至通过 |

### Phase 1 交付物 (核心层: vLLM + Qdrant)

> **合并原 Phase 1+2**：避免 vLLM embedding 写入 ChromaDB 造成混合向量库（评审 §5）。

- [ ] `app/llm/embed.py` HTTP 层改完（`_embed_via_ollama` → `_embed_via_vllm`）
- [ ] `app/llm/local.py` HTTP 层改完（`generate` / `summarize` → vLLM chat API）
- [ ] `app/brain/semantic.py` HTTP 层改完（实体抽取 → vLLM chat API）
- [ ] `app/memory/qdrant.py` 实现完整 QdrantService（API 方法名保持与 ChromaService 一致）
- [ ] 全项目 ChromaService → QdrantService 改名（16 文件）+ `_get_chroma_collection()` → QdrantService 方法
- [ ] 🔧 **BUGFIX**: 修正 `portrait/writer.py:590` PersonaSymmetry 调用——当前传的是 tracker 对象，应传 `export_for_symmetry()` 结果 + `from_dicts=True`
- [ ] 🧹 **缓存重建**: 删除 `data/tag_index.json`，切换 vLLM 后用 bge-m3 新向量重建标签嵌入索引
- [ ] `pytest tests/` 全部通过

### Phase 2 交付物 (杀全量模式)

- [ ] `qdrant.py` 中所有 `list_all()` → `scroll()` + filter
- [ ] `_translate_filter()` 实现并测试全部 8 种运算符覆盖
- [ ] `pipeline.py` 中所有 `_collection.get()` → Qdrant `retrieve` / `search`
- [ ] **删除** `bm25_fulltext.py`（Qdrant MatchText 替代）
- [ ] inverted_index 数据源切换（context.py 中 `build()`/`build_tags()` 的数据从 ChromaDB `get()` → Qdrant `scroll()`）

### Phase 3 交付物 (SQLite 迁移)

- [ ] CoOccurrence 独立 Qdrant collection 实现（record/query/export_for_symmetry）
- [ ] entity_pair 逻辑 → 入库时预计算 entity_co_counts，存入 payload
- [ ] hyperedge 逻辑 → Qdrant `hyper_edges` collection
- [ ] PersonaSymmetry 从 CoOccurrence collection 读取（替代 export_for_symmetry→SQLite）
- [ ] 删 `app/core/db.py`
- [ ] 删 `app/memory/cooccur.py` / `entity_pair.py` / `hyperedge.py`

### Phase 4 交付物 (百万级硬骨头)

- [ ] Qdrant quantization 配置启用（scalar_int8）
- [ ] Payload 索引创建（§4.5 清单全部）
- [ ] Embedding 缓存重建策略：LRU 上限 + 热度过滤，启动时不阻塞
- [ ] 100 万条压力测试通过（性能基准见 §10.2）

### Phase 5 交付物 (清理)

- [ ] `requirements.txt` 删 chromadb，加 qdrant-client
- [ ] `Dockerfile` 删 chromadb 构建依赖
- [ ] `docker-compose.yml` 删 ollama 服务
- [ ] 更新 CLAUDE.md（项目地图、数据流、红线——ChromaDB 红线移除，Qdrant 红线新增）
- [ ] 更新 README / SETUP / ARCHITECTURE 文档
- [ ] 全部测试 mock 更新（~35 文件）
- [ ] E2E 5 链路全部通过（需 Qdrant + vLLM 服务运行；CI 中可用 TEST_BACKEND 环境变量控制）
- [ ] 代码规范检查通过
- [ ] 全项目 grep 确认无残留 `chroma_service` / `ChromaService` / `chromadb` 引用（除回滚路径）
- [ ] ChromaDB 数据在 Qdrant 稳定运行 ≥7 天后删除

---

## 10. 验收标准

### 10.1 功能等价性

```
✅ 用户发一条消息 → 检索→LLM 调用→响应→入库，流程完整
✅ 10 路检索（或等价物）结果与迁移前语义一致（余弦 > 0.95）
✅ 画像更新正常工作
✅ 后台巩固 4h/24h 周期正常工作
✅ 情绪淡化正常工作
✅ 冲突解决正常工作
✅ /chat /memories /system API 返回格式不变
✅ 所有 62 个测试文件通过
✅ E2E 5 条链路全部通过
```

### 10.2 性能基准

```
场景: 1 万条记忆

操作                  ChromaDB(现状)    Qdrant(目标)    判定
─────────────────────────────────────────────────────────
add_memory            <50ms            <50ms            🟢
语义 search(top50)    <20ms            <30ms            🟢
list_memories(分页)   <100ms           <30ms            🟢 更快
stats()               <1ms             <5ms             🟢
batch_hit_count(100)  <20ms            <100ms           🟡 略慢,可接受
全量扫描(巩固)          <3s              <2s              🟢

场景: 100 万条记忆

操作                  Qdrant(目标)
────────────────────────────────────
语义 search(top50)    <50ms
分页 list              <100ms
stats()               <10ms
全量扫描               <10s (分页)
```

### 10.3 内存占用

```
场景: 100 万条记忆

组件                  ChromaDB(现状)    Qdrant(目标)
─────────────────────────────────────────────────
向量 (1024×f32)       4GB 全加载       1GB (int8量化, mmap, 按需分页)
Payload               随意              约 800B/条 = 800MB (不含 co_occurring)
CoOccurrence          独立 SQLite       独立 Qdrant collection ≈ 200MB
Embedding 缓存        100MB (10K条)     200MB (20K条, 按 last_hit_time DESC 排序)
                      ⚠️ 扩到50K=200MB  策略: 缓存最近命中过的 20K 条记忆 embedding。last_hit_time 降序自然偏向活跃记忆。命中率预期 >80%
其他缓存              ~200MB            ~200MB
─────────────────────────────────────────────────
总计                  ~5.3GB            ~2.4GB
```

> **缓存上限决策**: 原方案提出 50K 上限。评审分析：在百万级下 _build_embedding_cache 遍历全量 ID（scroll 100万条取最近 50K）耗时会显著增加。改为 20K，缓存策略从「热度过滤（只缓存 heat=hot）」改为「按 `last_hit_time DESC` 取最近 20K 条」——`last_hit_time` 降序自然同时覆盖 hot 记忆和用户最近翻出的旧记忆，平衡命中率和启动成本。

---

## 11. 风险登记册

| ID | 风险 | 概率 | 影响 | 缓解 | Phase |
|----|------|------|------|------|-------|
| R1 | Qdrant HNSW 参数不当，召回率下降 | 中 | 高 | Phase 0.5 原型验证对比 ChromaDB 召回率 | 0.5 |
| R2 | Payload filter 翻译层遗漏边界条件 ($contains/$and) | 中 | 中 | §5.4 覆盖清单全部 8 个实际 where 子句，写测试 | 1,2 |
| R3 | HyperEdge 重建逻辑迁移出错 | 高 | 中 | Phase 3 专门验证 hyperedge expand/get_memory_ids/cluster | 3 |
| R4 | 数据迁移脚本损毁源数据 | 低 | 极高 | 迁移脚本只读源数据，写目标；跑完校验再删旧数据 | 0,5 |
| R5 | vLLM 的 bge-m3 输出向量与 Ollama 不一致 | 中 | 极高 | Phase 0.5 首日 verify: 同文本 → 两边向量余弦 >0.99；不通过则全量重建 embedding | 0.5 |
| R6 | single-writer lock 在百万级下排队过长 | 低 | 中 | Phase 4 压力测试验证；必要时 CoOccurrence 异步写入 | 4 |
| R7 | E2E 测试全断 | 高 | 高 | 每个 Phase 跑完就修，不攒到最后 | 1-5 |
| R8 | ChromaDB metadata schema 字段名被无意改动 | 中 | 高 | Qdrant payload 字段名完全保持，迁移脚本做 diff 验证 | 1 |
| R9 | CoOccurrence 独立 collection scroll 性能不足 | 中 | 中 | Phase 0.5 原型验证 1000 条 → 推算 10万条延迟；备选：保留 SQLite cooccur 表 | 0.5,3 |
| R10 | Qdrant text index 对中文句子的分词行为未知，bm25 替代路径召回低 | **中** (v1.4 降级：inverted_index 保留后，仅影响 bm25 替代路径) | 低 (bm25 是固定分 0.35 的辅助信号) | Phase 0.5 验证完整中文句子的 MatchText 命中率；不通过则回退为无 bm25 路径（10→9 路），影响可控 | 0.5 |
| R11 | 多用户隔离策略不明确导致数据混合 | 低 | 高 | Phase 0 设计时确定 collection 命名规范（§13） | 0 |
| R12 | `$contains` → `MatchText` 子串匹配语义差异导致假阴性 | **低** (v1.4 降级：inverted_index 保留后，tags/entities 的精确匹配走 inverted_index，MatchText 仅用于 bm25 的 document 搜索) | 低 | Phase 0.5 文档化差异行为；备选方案已就绪（inverted_index 健在） | 0.5 |
| R13 | 切换 vLLM 后 tag_index.py 缓存未重建，余弦近邻返回错误结果 | 中 | 中 | Phase 1 显式删除 `data/tag_index.json`，用 vLLM bge-m3 新向量重建 | 1 |

---

## 12. 回滚策略

### 12.1 数据层回滚

```
Phase 0-1: 无风险 — 不改存储
Phase 2:   Qdrant 并行运行，ChromaDB 数据不删
Phase 3:   同上，全量数据仍在 ChromaDB
Phase 4:   同上，SQLite 文件不删
Phase 5:   同上
Phase 6:   删旧数据前确保 Qdrant 运行 ≥7 天无事故

回滚方式:
  1. 改 settings.py: STORAGE_BACKEND = "chromadb" (保留切换开关)
  2. 重启服务
  3. ChromaDB 数据完整，SQLite 文件完整
```

### 12.2 推理层回滚

```
Phase 1 vLLM → 回滚方式:
  1. 改 settings.py: EMBED_PROVIDER = "ollama"
  2. 重启服务
  3. 切换环境变量 VLLM_BASE_URL → 空，OLLAMA_EMBED_MODEL → "bge-m3"
```

### 12.3 切换开关设计

```python
# app/config/settings.py
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "qdrant")  # "chromadb" for rollback
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "vllm")      # "ollama" for rollback

# Qdrant 连接配置
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # None = 本地开发无认证
```

---

## 13. 多用户隔离策略

> **背景**: 当前 ChromaDB 通过 `data/users/{name}/chroma/` 目录隔离。Qdrant 是网络服务，没有目录概念。需要明确隔离机制。

### 13.1 方案：Qdrant Collection 前缀隔离

```
用户 "alice" 的数据:
  Qdrant collections:
    alice_memories            ← 替代 data/users/alice/chroma/
    alice_ai_memories         ← 替代 data/users/alice/ai_chroma/
    alice_co_occurrence       ← 替代 data/users/alice/co_occurrence.db
    alice_hyper_edges         ← 替代 data/users/alice/hyper_edges.db

用户 "bob" 的数据:
    bob_memories
    bob_ai_memories
    bob_co_occurrence
    bob_hyper_edges
```

**QdrantService 初始化时按用户动态选择 collection 名：**

```python
class QdrantService:
    def __init__(self, user_name: str, source: str = "user"):
        prefix = f"{user_name}_"
        self._memories_coll = f"{prefix}memories" if source == "user" else f"{prefix}ai_memories"
        self._ai_coll = f"{prefix}ai_memories"
        self._cooc_coll = f"{prefix}co_occurrence"
        self._hyper_coll = f"{prefix}hyper_edges"
```

### 13.2 替代方案（备选）

如果 Qdrant collection 数量膨胀成问题（用户数 > 100），可改用 **payload-based 多租户**：

```python
# 所有用户共享 collection，payload 中加 user_id 字段
# 所有查询都加 must=[FieldCondition(key="user_id", match=MatchValue(value=user_name))]
```

- 优先采用 **方案 A (prefix)**，用户数 < 50 时简单可靠
- 用户数 > 50 时迁移到 **方案 B (payload 多租户)**
- Phase 0 docker-compose 默认单用户，Phase 4 压力测试验证多用户隔离

### 13.3 迁移过渡期测试策略

Phase 1-3 期间 ChromaDB 和 Qdrant 代码路径并存。通过环境变量动态选择后端：

```python
# app/config/settings.py
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "qdrant")  # "chromadb" 回退
# app/core/context.py
if STORAGE_BACKEND == "qdrant":
    self.memory_service = QdrantService(user_name, source="user")
else:
    self.memory_service = ChromaService(...)
```

```bash
# 每个 Phase 出口必须通过两套测试:
TEST_BACKEND=chromadb python -m pytest tests/ -q   # 旧后端不受影响
TEST_BACKEND=qdrant  python -m pytest tests/ -q   # 新后端通过
```

---

## 附录 A: 文件变更总清单

### 新增
- `app/memory/qdrant.py` — QdrantService（替代 chroma.py + cooccur.py + entity_pair.py + hyperedge.py）
- `SPEC_MIGRATION.md` — 本文件

### 删除
- `app/core/db.py`
- `app/memory/chroma.py`
- `app/memory/cooccur.py`
- `app/memory/entity_pair.py`
- `app/memory/hyperedge.py`
- `app/retrieval/bm25_fulltext.py`
- `data/chroma/` (目录)
- `data/ai_chroma/` (目录)
- `data/co_occurrence.db`
- `data/entity_pairs.db`
- `data/hyper_edges.db`

### 修改
- `app/llm/embed.py` — 改 HTTP 层（ollama → vLLM embed API）
- `app/llm/local.py` — 改 HTTP 层（ollama → vLLM chat API）
- `app/brain/semantic.py` — 改 HTTP 层（实体抽取 ollama → vLLM）
- `app/core/context.py` — ChromaService → QdrantService，删各 tracker init
- `app/core/conflict.py` — 参数类型标注（ChromaDB → Qdrant）
- `app/retrieval/pipeline.py` — 4 路 API 翻译（①②⑧⑩ semantic/attention/ai search）+ 删 bm25 + co_occurrence 换后端；inverted 相关 5 路不动
- `app/tools/dispatch.py` — `_get_chroma_collection()` → Qdrant scroll
- `app/api/chat.py` — `ctx.chroma_service` → `ctx.qdrant_service`
- `app/api/memories.py` — 同上
- `app/api/system.py` — 同上
- `app/background/consolidation.py` — 同上
- `app/background/impulse.py` — 同上
- `app/analysis/self_mirror.py` — 同上
- `app/analysis/symmetry.py` — 读 CoOccurrence collection（替代 export_for_symmetry→SQLite）
- `app/portrait/writer.py` — 修正 PersonaSymmetry 调用（from_dicts=True）
- `app/config/settings.py` — 增删配置项
- `requirements.txt` — 删 chromadb，加 qdrant-client
- `Dockerfile` — 删 chromadb 构建依赖
- `docker-compose.yml` — 删 ollama，加 qdrant + vllm-embed + vllm-chat
- `CLAUDE.md` — 文档更新（项目地图、数据流、红线）
- `README.md` / `ARCHITECTURE.md` / `SETUP.md` 等文档
- 全部测试文件 (~35 个)

---

## 附录 B: 未决事项（已评审，全部决议）

| # | 问题 | 决议 | 位置 |
|---|------|------|------|
| 1 | Qdrant collection 名 | 保持 `"memories"` / `"ai_memories"`，多用户前缀 `"{user}_"` | §13 |
| 2 | CoOccurrence LTD 衰减 | **查询时降权**，删除定时全表 UPDATE | §7.1 |
| 3 | HyperEdge 是否存向量 | **存向量**（参与实体的平均 bge-m3 embedding），用于语义检索相关超边 | §4.3 |
| 4 | BM25 全文检索 | **删除 BM25 模块**，Qdrant text index 完全替代 | §8.1 |
| 5 | vLLM bge-m3 + qwen2.5:3b 共存 | **双实例**：:8001 (bge-m3) + :8002 (qwen2.5:3b) | §2.2, §6.3 |
| 6 | Embedding 缓存大小 | **20K + last_hit_time DESC**（缓存最近命中的 20K 条），非热度过滤 | §10.3 |
| 7 | $contains → MatchText 语义 | **Phase 0.5 验证**，差异文档化；备选 keyword 数组 | §5.4, R12 |
| 8 | tag_index.py 缓存重建 | **Phase 1 显式删除并重建**，切换 vLLM 后旧 Ollama 向量失效 | §3.1, R13 |
| 9 | writer.py:590 PersonaSymmetry bug | **Phase 1 修复**（迁移到 Phase 1，非 Phase 5） | §9 Phase 1 |
| 10 | CoOccurrence 向量过时 | **文档化为尽力而为快照**，核心查询走 scroll + order_by，不依赖向量 | §4.2 |
| 11 | entity_co_counts 精度 | **概率近似**，入库采样 top-50 聚合，非精确计数 | §7.2 |

## 附录 C: 原型验证结果

> 本附录在 **Phase 0.5 完成后** 填入。包含：
> - vLLM vs Ollama 向量余弦相似度数据
> - Qdrant HNSW 召回率 vs ChromaDB
> - CoOccurrence 独立 collection 性能
> - Qdrant text index 中文标签匹配精度
> - 任何设计变更决策

---

*本文档是迁移蓝图的唯一权威来源。执行过程中发现问题必须回溯更新本文档。*
