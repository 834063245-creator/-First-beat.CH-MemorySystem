# 集成测试 SPEC（中间层）

> 本文档为 LLM 代码生成的输入规范。每条链路描述：测什么、怎么 mock、断言什么。
> 生成的测试文件应放在 `tests/` 目录下，文件名见各链路标题。

---

## 0. 全局约定

### 0.1 核心原则

- **用真实 ChromaDB + 真实 embedding**（Ollama bge-m3），不 mock
- **只 mock LLM 文本生成**（摘要/标签/情绪/实体），因为 LLM 输出不稳定
- 每个测试目标 ≤ 2 秒，整套 ≤ 30 秒
- 测的是"组件之间能不能跑通"，不是单个函数的内部逻辑

### 0.2 Fixture 复用

复用 `tests/conftest.py` 已有 fixture：

| Fixture | 说明 | 模式 |
|---------|------|------|
| `isolated_env` | 隔离 AppContext + 临时数据目录 | BENCHMARK_MODE=true |
| `isolated_env_no_bm` | 同上 | BENCHMARK_MODE=false |
| `seeded_env` | isolated_env + 12 条种子记忆 | BENCHMARK_MODE=true |

### 0.3 Mock 清单

#### BENCHMARK_MODE=true（isolated_env 默认）

Benchmark 路径只调 embedding + 标签提取，不调 LLM 摘要和情绪分析。
**只需 mock 1 个函数**：

| 函数 | 导入路径 | 真实签名 | Mock 返回值 |
|------|---------|---------|------------|
| `extract_tags` | `app.brain.semantic.extract_tags` | `(text: str, topk: int = 5) -> list[str]` | 根据关键词返回固定标签列表 |

> `local_embed` 走真实 Ollama，不 mock。

#### BENCHMARK_MODE=false（isolated_env_no_bm）

非 Benchmark 路径调用完整的 LLM 管线。
**需要 mock 3 个函数**：

| 函数 | 导入路径 | 真实签名 | Mock 返回值 |
|------|---------|---------|------------|
| `LocalLLM.summarize` | `app.llm.local.LocalLLM.summarize` | `(self, text: str, max_chars=200, *, fast=False) -> str` | 返回 `text[:50]` 截断 |
| `extract_tags` | `app.brain.semantic.extract_tags` | `(text: str, topk: int = 5) -> list[str]` | 根据关键词返回固定标签列表 |
| `extract_entities` | `app.analysis.entity.extract_entities` | `(text: str) -> list[dict]` | 返回 `[]` 或固定实体 |
| `analyze_emotion_2d` | `app.analysis.emotion.analyze_emotion_2d` | `(text: str) -> tuple[float, float, str]` | 返回 `(0.0, 0.3, "neutral")` |

> **注意**：`_extract_noun_tags`（`app.core.context` 内部函数）实际调用的是 `app.brain.semantic.extract_tags`，
> 所以 mock 路径应 patch `app.brain.semantic.extract_tags`，不要 patch `app.core.context._extract_noun_tags`。

### 0.4 Mock 桩函数模板

```python
# ── 标签提取桩（通用，benchmark/non-benchmark 都用）──
def _mock_extract_tags(text: str, topk: int = 5) -> list[str]:
    tag_map = {
        "Rust": ["Rust", "编程"], "Python": ["Python", "编程"],
        "猫": ["宠物", "猫"], "狗": ["宠物", "狗"],
        "东京": ["旅行", "东京"], "大阪": ["旅行", "大阪"],
        "工作": ["工作", "职场"], "压力": ["工作", "压力"],
        "健身": ["健身", "运动"], "跑步": ["健身", "跑步"],
    }
    tags = []
    for kw, tl in tag_map.items():
        if kw in text:
            tags.extend(tl)
    return list(set(tags))[:topk] if tags else ["通用"]

# ── 摘要桩（仅 non-benchmark）──
def _mock_summarize(self, text: str, max_chars: int = 200, *, fast: bool = False) -> str:
    return text.strip()[:max_chars]

# ── 实体桩（仅 non-benchmark）──
def _mock_extract_entities(text: str) -> list[dict]:
    return []  # 简化，不测实体

# ── 情绪桩（仅 non-benchmark）──
def _mock_analyze_emotion_2d(text: str) -> tuple[float, float, str]:
    if any(w in text for w in ["愤怒", "压力", "焦虑"]):
        return (-0.6, 0.8, "negative")
    if any(w in text for w in ["开心", "有趣", "棒"]):
        return (0.7, 0.6, "positive")
    return (0.0, 0.3, "neutral")
```

### 0.5 命名规范

- **类名**：`TestInt{链路名}`（如 `TestIntWriteRetrieve`）
- **方法名**：`test_{行为}_{条件}`（如 `test_write_then_inverted_index_finds_it`）
- **每个方法只验证一个跨组件行为**

### 0.6 关键注意事项

1. `_store_conversation` 在 BENCHMARK_MODE 下**不调 LLM 摘要和情绪分析**，只做 embed + 标签 + 写库
2. `_store_conversation` 是同步方法，内部有 `time.sleep` 等待队列，测试中需加 `time.sleep(0.3~0.5)` 等入库完成
3. `retrieve_all` 的签名：`(user_message: str, query_embedding: list | None, ctx_obj, intent=None, cached_tags=None) -> list[dict]`
4. 返回的 dict 字段：`id, document, metadata, distance, source, summary, hit_count`
5. `compute_score` 的签名：`(similarity: float, hit_count: int, attention_boost=0.0, bm25_score=0.0, source_bonus=0.0, error_penalty=0.0) -> float`
6. `RERANK_ATTENTION_WEIGHT = 0.0`（settings.py），因此 `compute_score` 的 `attention_boost` 参数**对分数无影响**，测试中应使用 `source_bonus` 代替
7. `_store_conversation` 在 BENCHMARK_MODE 下**不写 chat_history**（chat_history.append 只在 API 层 `app/api/chat.py` 调用），链路 1 不应测试 chat_history 写入
8. BENCHMARK_MODE=true 会强制 `IS_LITE=True` + `LITE_DISABLE_BACKGROUND_TASKS=True`，导致 `ctx.dmn = None`。巩固测试需**手动构建 ConsolidationEngine**（从 ctx 的子组件拼装）

---

## 1. 链路 1：写入→检索闭环

**文件**：`tests/test_int_write_retrieve.py`
**Fixture**：`isolated_env`（BENCHMARK_MODE=true）
**Mock**：仅需 mock `extract_tags`
**测试数量**：5 个

| # | 方法名 | 行为描述 | 断言 |
|---|--------|---------|------|
| 1 | `test_write_increments_chroma_count` | 写入 1 条对话后 | `chroma_service.count()` 比写入前 +1 |
| 2 | `test_write_populates_inverted_index` | 写入含"Rust"的对话后 | `inverted_index.query(["Rust"], min_match=1)` 返回 ≥1 条 |
| 3 | `test_write_populates_tag_index` | 写入含"Rust"的对话后 | `inverted_index.query_tags(["编程"])` 返回非空 set |
| 4 | `test_write_stores_correct_metadata` | 写入对话后 | ChromaDB 记录的 metadata 含 `tags` 字段，且包含预期标签 |
| 5 | `test_write_then_semantic_search` | 写入"橘猫去宠物医院"后 | 用 `local_embed("猫咪生病")` 做 ChromaDB query 返回 ≥1 条 |

**数据准备**：每个测试独立写入 1~2 条对话，不依赖 seeded_env。

---

## 2. 链路 2：检索通路验证

**文件**：`tests/test_int_retrieval_paths.py`
**Fixture**：`isolated_env`（BENCHMARK_MODE=true）
**Mock**：仅需 mock `extract_tags`
**测试数量**：5 个

| # | 方法名 | 行为描述 | 断言 |
|---|--------|---------|------|
| 1 | `test_retrieve_all_returns_results` | 写入 2 条不同话题记忆 → `retrieve_all("编程", None, ctx)` | 返回 list 且 len ≥ 1 |
| 2 | `test_retrieve_all_deduplicates` | 写入 1 条记忆 → `retrieve_all` 多路命中 | 返回的 ID 列表无重复：`len(ids) == len(set(ids))` |
| 3 | `test_retrieve_all_scores_bounded` | 写入 1 条记忆 → `retrieve_all` | 每条结果的 `score` 在 `[0.0, 1.0]` 之间 |
| 4 | `test_retrieve_all_empty_store` | 空数据库 → `retrieve_all` | 返回 `[]`，不抛异常 |
| 5 | `test_keyword_path_finds_by_tag` | 写入含标签"旅行"的记忆 | `inverted_index.query_tags(["旅行"])` 返回非空 set |

**注意**：`retrieve_all` 内部会调 `local_embed`（真实 Ollama），所以 embedding 是真的。

---

## 3. 链路 3：对话历史流转

**文件**：`tests/test_int_history_flow.py`
**Fixture**：`isolated_env`（BENCHMARK_MODE=true）
**Mock**：无需 mock（ChatHistory 纯文件 IO，不涉及 LLM）
**测试数量**：5 个

| # | 方法名 | 行为描述 | 断言 |
|---|--------|---------|------|
| 1 | `test_append_then_get_recent` | `chat_history.append` 写入 3 轮 | `get_recent(n=3)` 返回 3 条 |
| 2 | `test_context_by_timestamp` | 写入 5 轮 → 查第 3 轮时间戳上下文 | `get_context_by_timestamp(ts, before=2, after=2)` 返回非 None |
| 3 | `test_history_persists_to_jsonl` | 写入 1 条 → 检查文件 | `data_dir/chat_history.jsonl` 存在且行数 ≥ 1 |
| 4 | `test_delete_by_timestamp` | 写入 → 删除 → 查最近 | `delete_by_timestamp(ts)` 返回 True，`get_recent` 不含该条 |
| 5 | `test_update_chroma_id` | 写入 → `update_chroma_id(ts, "abc123")` | `get_records_snapshot()` 中该条含 `chroma_id: "abc123"` |

---

## 4. 链路 4：巩固流水线

**文件**：`tests/test_int_consolidation.py`
**Fixture**：自定义 `consolidation_env`（基于 `isolated_env` + mock 标签写入 12 条种子 + 手动构建 ConsolidationEngine）
**Mock**：仅需 mock `extract_tags`（巩固过程中标签重建会调用）
**测试数量**：4 个

> **为什么手动构建 DMN**：BENCHMARK_MODE=true 下 `ctx.dmn = None`（IS_LITE 强制启用），
> 因此 fixture 中用 ctx 的子组件（chroma_service, personality_store, behavior_store, chat_history, co_tracker）
> 手动实例化 `ConsolidationEngine`。

| # | 方法名 | 行为描述 | 断言 |
|---|--------|---------|------|
| 1 | `test_shallow_consolidation_no_error` | 12 条种子 → `dmn.consolidate_shallow()` | 不抛异常 |
| 2 | `test_deep_consolidation_no_error` | 12 条种子 → `dmn.consolidate_deep()` | 不抛异常 |
| 3 | `test_consolidation_state_dict` | 12 条种子 → 浅巩固 → `dmn.get_state_update()` | 返回 dict，含 `topics` 键 |
| 4 | `test_consolidation_writes_state_file` | 12 条种子 → 浅巩固 → 检查 `dmn_state.json` | 文件存在且为有效 JSON 字典 |

**注意**：
- 巩固过程较长（含多次 ChromaDB 查询），每个测试可能 5~20 秒，属于可接受范围

---

## 5. 链路 5：评分与排序

**文件**：`tests/test_int_scoring.py`
**Fixture**：无需 fixture（`compute_score` 纯计算函数）
**Mock**：无需 mock
**测试数量**：5 个

| # | 方法名 | 行为描述 | 断言 |
|---|--------|---------|------|
| 1 | `test_score_increases_with_similarity` | 同 hit_count，sim 0.3 vs 0.9 | `score(0.9) > score(0.3)` |
| 2 | `test_score_increases_with_hits` | 同 sim，hit 1 vs 100 | `score(100) > score(1)` |
| 3 | `test_score_bounded_01` | 遍历 sim∈{0,0.5,1} × hits∈{0,10,1000} | 所有分数 ∈ `[0.0, 1.0]` |
| 4 | `test_source_bonus_increases_score` | 同参数，source_bonus 0.0 vs 0.1 | `score(bonus=0.1) > score(bonus=0.0)` |
| 5 | `test_error_penalty_decreases_score` | 同参数，penalty 0.0 vs 0.3 | `score(penalty=0.0) > score(penalty=0.3)` |

---

## 6. 执行命令

```bash
# 跑全部集成测试
pytest tests/test_int_*.py -v --tb=short

# 跑单条链路
pytest tests/test_int_write_retrieve.py -v

# 跑单个测试
pytest tests/test_int_write_retrieve.py::TestIntWriteRetrieve::test_write_then_semantic_search -v
```

## 7. 依赖检查清单

跑集成测试前确认：

- [ ] Ollama 已启动（`ollama serve`）
- [ ] bge-m3 模型已拉取（`ollama pull bge-m3`）
- [ ] 项目依赖已安装（`pip install -r requirements.txt`）

---

## 8. 与现有测试的关系

| 现有测试 | 层次 | 与集成测试的区别 |
|---------|------|----------------|
| `tests/test_retrieval_pipeline.py` | 单元测试 | 直接实例化 InvertedIndex/CoOccurrenceTracker，不经过 AppContext |
| `tests/test_int_retrieval_paths.py`（新增） | 集成测试 | 通过 AppContext 写入真实 ChromaDB，再调 retrieve_all 走完整管线 |
| `E2E/test_write_path.py` | E2E | 用真实 LLM + 真实 embedding + 真实 ChromaDB，最重最慢 |
