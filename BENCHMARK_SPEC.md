# 初痕端到端 Benchmark 规格书

> 本文档定义五条全链路的输入、关键节点、输出条件。
> 不做实现假设，只定义"什么是对的"。

---

## 总则

- **测试环境**：真实 ChromaDB + 真实 bge-m3 + 真实本地 LLM（不 mock）
- **测试数据隔离**：每个 benchmark 用例使用独立的 ChromaDB 集合 / 临时目录，不污染生产数据
- **可复现性**：固定 random seed，相同输入 → 相同分数
- **评分维度**：每条链路独立计分，不加权合成一个数字（加权总分掩盖问题）

---

## 链路一：写入链路

### 触发
```
POST /chat  { "message": "<用户消息>" }
```

### 关键节点与预期

| # | 节点 | 输入 | 预期输出 | 验证方法 |
|---|------|------|---------|---------|
| W1 | embedding | 用户消息文本 | `list[float]`，len=1024，非全零，非 NaN | `assert len(emb) == 1024` |
| W2 | summary | 用户消息 + AI 回复 | 中文摘要，长度 20~200 字，含关键实体 | `assert 20 <= len(s) <= 200` |
| W3 | tags | 用户消息 + 摘要 | `list[str]`，≥2 个标签，每个 ≥2 字 | `assert len(tags) >= 2` |
| W4 | entities | 用户消息 | `list[dict]`，可空，非空时每项含 text + type | schema 校验 |
| W5 | emotion | 用户消息 | valence(float) + arousal(float) + category(str) | valence ∈ [-1,1], arousal ∈ [0,1] |
| W6 | time_features | 当前时间戳 | year/month/day/day_of_week/hour/season/time_period 全覆盖 | 24 小时每个小时映射到正确 time_period |
| W7 | ChromaDB 存储 | 所有上述字段 | 1 条新记录，id 非空，metadata 完整 | `collection.get(ids=[mid])` 返回完整记录 |
| W8 | chat_history 存储 | 用户消息 + AI 回复 | jsonl 追加一行，含 timestamp/user_message/llm_reply | 解析最后一行 JSON，字段齐全 |
| W9 | 倒排索引 | tags + summary | 新标签 → 新记忆 ID 的映射存在 | `inverted.query_tags(["新标签"])` 包含该 ID |
| W10 | 实体对存储 | 提取到的 entities | entity_pairs.json 中新增对应实体对的共现计数 | 读取 entity_pairs.json，对应 key 的 count ≥ 1 |
| W11 | AI 自我记忆存储 | AI 回复文本 + 分析结果 | ai_memories 集合新增 1 条记录，metadata 含 summary/tags/emotion | `ai_collection.get(ids=[ai_mid])` 返回完整记录 |
| W12 | 回复不崩 | 整个请求 | HTTP 200，response 字段非空 | `assert resp.response` |

### 计分
```
写入链路分 = 通过的节点数 / 12
```

---

## 链路二：检索+编织+认知链路

### 触发
```
POST /chat  { "message": "<查询消息>" }
```
前提：已通过链路一写入 N 条已知记忆（N ≥ 10，覆盖多话题）。

### 关键节点与预期

| # | 节点 | 输入 | 预期输出 | 验证方法 |
|---|------|------|---------|---------|
| R1 | 意图分类 | 查询文本 | intent ∈ {recall, emotional_sharing, ask_fact, conflict, casual} | 与人工标注比对 |
| R2 | 门控配额 | intent | 各路径配额符合 _INTENT_ROUTES 定义 | `route["semantic"] >= 预期最小值` |
| R3 | Working Memory 更新 | 用户消息 + AI 回复 + 当前 WM digest | digest 更新，覆盖本轮关键实体/话题；话题偏移 ≥30% 时触发全量重写 | 新 digest 包含本轮消息关键实体 |
| R4 | 语义检索 | query embedding | 返回 N 条，其中"自身"（用存入时的原文查）在 top-20 内 | self-retrieval hit |
| R5 | 关键词检索 | query keywords | 精确命中 ≥1 个 tag 的记忆被返回 | `any(tag in result.tags)` |
| R6 | 实体检索 | query entities | 精确命中 ≥1 个实体的记忆被返回 | 同上 |
| R7 | 共现扩展 | 已命中记忆 ID | 返回的记忆中包含与已命中记忆共现过的 | ID 出现在 cooccur 结果中 |
| R8 | 时间触发 | 当前时段 | 同时段历史记忆被返回（如果有） | 验证返回记忆的 time_period |
| R9 | 话题树扩展 | 已命中记忆的话题 | 同话题簇的记忆被返回 | 验证返回记忆的 topic_cluster |
| R10 | AI 表达检索 | AI reply embedding | ai_memories 中相似历史表达被检索到 | 写入 ≥3 条 AI 记忆后，语义相似的 AI 历史表达在检索结果中 |
| R11 | 注意力漂移 | 最近 3 轮对话 | attention_proximity 字段非 None（值可为零） | 字段存在性 |
| R12 | 去重 | 多路结果 | 同一 memory_id 只出现一次 | `len(set(ids)) == len(ids)` |
| R13 | 精排顺序 | 去重后候选 | 高 similarity + 高 hit_count 的记忆排在前面 | 前 3 条 score ≥ 后面 |
| R14 | 新近度权重 | 候选记忆 | 90 天线性衰减到 0.15；archived 上限 0.6；stale 上限 0.3，折入 score | 30 天前记忆的 recency_weight ≈ 0.67；archived 记忆 ≤ 0.6 |
| R15 | 行为预测 | 最近 3 轮用户消息 | Markov chain 预测下一意图/话题概率分布，top-1 概率 > 0 | `predictor.predict()` 返回非空 dict，概率之和 ≈ 1 |
| R16 | 编织-故事线 | 候选记忆 | 同实体跨 ≥2 天的记忆被检出 narrative | `len(wc.narratives) >= 0`（有则格式正确） |
| R17 | 编织-故事线情绪趋势 | 同故事线记忆 | trend ∈ {延续, 出现翻转, 持续积极, 持续消极} | 与同故事线记忆的 valence 一致 |
| R18 | 编织-分层 | 候选记忆 | semantic_dist < 0.30 × source_boost → fact，其余 → discard | `wc.fact_memories` 只含高置信记忆 |
| R19 | 编织-stale 处理 | stale=True 的记忆 | 不进 fact，进 stale_context 或 discard | stale 记忆不在 wc.fact_memories 中 |
| R20 | 冲突检测 | 候选 fact 记忆 | 语义矛盾被检出（若存在），冲突对含 both_ids + conflict_type；无矛盾时返回空列表 | 注入矛盾事实对 → assert len(conflicts) >= 1；无矛盾 → 空 |
| R21 | 编织-Token 预算 | fact_memories | 总 token ≤ 20000 | `sum(len(doc)//2 + 10 for doc in facts) <= 20000` |
| R22 | 编织-闲聊不发言 | intent=casual + 候选 ≤ 3 | `should_speak = False` | assert not wc.should_speak |
| R23 | 认知分层 | 编织后的记忆 | MemoryDirective.role ∈ {fact, reference, background, suppressed}，stale 不在 fact | stale 记忆的 role ≠ "fact" |
| R24 | 关系状态 | 最近 30 轮对话 | RelationshipState 含 familiarity/trust/closeness/interaction_mode，值在 [0,1] | 各字段非 None，连续互动后 familiarity 上升 |
| R25 | 情绪状态推断 | intent + emotion | user_mood ∈ {positive, negative, neutral}；affective_context ∈ {intimate, focused_work, casual_chat, conflict} | emotional_sharing + negative → affective_context="intimate" |
| R26 | 门控-tone | intent + emotion + relationship | tone 匹配场景（如 emotional_sharing + negative → "caring"；高 closeness → 更亲密 tone） | 与预期 tone 比对 |
| R27 | 门控-mode | intent + emotion + relationship | response_mode 匹配场景（如 conflict → "confirm"） | 与预期 mode 比对 |
| R28 | 引擎调参覆盖 | PatternDiscovery tuning | emotional_dampening → tone="neutral" + intensity 压制；formality_shift → formality 调整 | 注入 emotional_dampening 信号后验证 gate.tone="neutral" |
| R29 | 冲动检查 | 空闲时间 | 空闲 >2min 时冲动队列被检查，有效优先级 ≥2 的信号被注入 | 检查 ImpulseDirective |
| R30 | 人格注入-用户 | personality_notes | system prompt 含用户人格标签 | 验证最终 prompt 含用户标签 |
| R31 | 人格注入-AI | personality_notes_ai (source=ai) | system prompt 含 AI 自我表达习惯标签 | 验证最终 prompt 含 AI 人格标签，source=ai |
| R32 | 话题笔记注入 | 当前话题 tags | DMN.get_topic_notes() 返回匹配的话题笔记，注入 prompt | prompt 中含话题笔记的关键词，匹配当前话题 |
| R33 | 模式观察注入 | PatternDiscovery + PersonaSymmetry | prompt 含 [模式观察] 段，来源为 PatternDiscovery.get_observations() + blind_spots.json | prompt 中含 "[模式观察]" 文本；blind_spots.json 存在时其 observation 出现在 prompt 中 |
| R34 | LLM 回复含引用 | fact 记忆 | 回复中包含 fact 记忆的关键实体或事实 | `assert any(entity in response for entity in fact_entities)` |
| R35 | LLM 回复不含 suppressed | suppressed 记忆 | 回复中不出现 suppressed 记忆的内容 | 字符串匹配 |

### 计分
```
检索+编织+认知链路分 = 通过的节点数 / 35
```

---

## 链路三：跨轮记忆链路

### 触发
```
第 1 轮  POST /chat { "message": "我最近在<做某事>" }
第 2 轮  POST /chat { "message": "<无关话题>" }
...
第 K 轮  POST /chat { "message": "我之前跟你说过什么来着" }
```

### 测试变体

| 变体 | 描述 | 轮数 | 验证点 |
|------|------|------|--------|
| X1 短跨 | 中间隔 1 轮无关对话 | 3 轮 | 第 3 轮检索命中第 1 轮的记忆 |
| X2 长跨 | 中间隔 5+ 轮无关对话（不同话题） | 7 轮 | 第 7 轮检索仍命中第 1 轮 |
| X3 同义改写 | 第 K 轮用不同措辞问同一件事 | 3 轮 | 编织后的回复引用原事实 |
| X4 注意力惯性 | 连续 3 轮聊同一话题 | 3 轮 | 第 3 轮的 attention_proximity > 第 1 轮 |
| X5 话题切换 | 前 2 轮聊 A，第 3 轮换 B | 3 轮 | 第 3 轮的 attention 权重主要落在 B |
| X6 情绪翻转 | 第 1 轮 "喜欢 X"，第 K 轮 "X 让我崩溃" | 3 轮 | 第 1 轮记忆被标记 stale |
| X7 WM 跨轮延续 | 连续 3 轮聊同一话题，第 4 轮问"我们之前在聊什么" | 4 轮 | WM digest 包含第 1~3 轮的关键实体，LLM 能引用前文 |
| X8 关系演化 | 连续 5 轮积极互动（用户表达满意/感谢） | 5 轮 | 第 5 轮 RelationshipState.familiarity > 第 1 轮 |
| X9 冲突修正 | 第 1 轮 "我叫张三"，第 2 轮 "不对，我叫李四" | 2 轮 | 第 1 轮记忆被 supersede，后续检索命中"李四"而非"张三" |

### 计分
```
跨轮链路分 = 通过的变体数 / 9
```

---

## 链路四：记忆演化链路

### 触发
```
第 1 天   写入 N 条记忆（N ≥ 10，覆盖 ≥3 个话题）
第 N 天   触发浅巩固（手动或等待 4h）
第 N+1 天 触发深巩固（手动或等待 24h）
第 N+2 天 查询
```
> 注：M13~M15 不依赖巩固调度器，各有独立触发路径，详见各节点说明。

### 浅巩固 (4h / idle ≥ Level 2)

| # | 节点 | 预期 | 验证方法 |
|---|------|------|---------|
| M1 | 话题树重建 | 同话题记忆聚集到同一分支 | 验证话题簇内记忆的 tag 重叠率 |
| M2 | 语义重复检测 | 高度相似的记忆被识别 | 检测到重复对的 sim > 0.9 |
| M3 | Supersede 链路 | 检测到的重复对中，旧的被标记 stale + superseded_by 指向新记忆 ID | `old_mem.stale == True` 且 `old_mem.superseded_by == new_mem.id` |
| M4 | Tag Embedding 索引 | tag_embeddings.json 重建，每个标签有对应 embedding | 文件存在，标签数 ≥ 总标签数，每个 embedding len=1024 |
| M5 | Topic Affinity 图 | topic_affinity.json 更新，话题间边的权重反映共现频率 | 文件存在，高共现话题对的 affinity > 低共现对 |
| M6 | 人格蒸馏 | 从对话中提取用户/AI 标签 | 蒸馏后标签数量 ≥ 蒸馏前 |
| M7 | 冷热转换 | 14 天未命中的记忆标记 cool | 验证 metadata.heat |
| M8 | Entity Pair 演化 | entity_pairs.json 中新增/更新实体对的共现计数和时间戳 | 文件存在，巩固后关键实体对计数 ≥ 巩固前 |
| M9 | 人格对称性 | 比较用户/AI 双共现矩阵，检出盲区（distribution gap ≥ 0.3），写入 cache/blind_spots.json | blind_spots.json 存在，observations 非空且含 tag/gap/user_related/ai_related |

### 深巩固 (24h / idle ≥ Level 3)

| # | 节点 | 预期 | 验证方法 |
|---|------|------|---------|
| M10 | 归档评估 | 30 天未命中的话题簇被归档 | 验证 archived 标记 |
| M11 | 话题笔记 | 对每个显著话题簇生成笔记 | 笔记文件存在且格式正确 |
| M12 | 情绪淡化-巩固触发 | 高 arousal 的旧记忆 emotional_intensity 下降 | 对比前后 intensity |

### 独立触发（不依赖巩固调度器）

| # | 节点 | 触发路径 | 预期 | 验证方法 |
|---|------|---------|------|---------|
| M13 | 情绪衰减 | 每 50 次 `increment_hit_count` 触发检查，3 天未命中 | 高 emotional_intensity 的记忆 intensity 自然衰减 | 对比 3 天前后 intensity，不应需要巩固才触发 |
| M14 | AI 自我巩固 | 独立 1h 定时线程 | ai_memories 集合也经历浅/深巩固，不抛异常 | AI 记忆集合的 topic_tree / duplicates / archived 等操作正常完成 |
| M15 | 用户反馈闭环 | 用户通过 API 纠正/报错记忆时写入 jsonl | 被报错的记忆检索时降权（error_count 越高 score 越低）；被纠正的记忆获 boost（+0.3），同 tag 群组 boost（+0.1），downvote 惩罚（-0.3） | 注入 3 条 error_report → 对应记忆的 score 低于同类未报错记忆；注入 correction → score 高于同类 |
| M16 | 原文不变 | 任意巩固前后 | 巩固前后 document 字段 hash 一致 | MD5 比对 |

### 计分
```
记忆演化链路分 = 通过的节点数 / 16
```

---

## 独立项：后台节律

### 触发
不经过 HTTP。通过时间推进 / 手动触发各周期事件。

### 子项

| # | 子系统 | 周期 | 预期 | 验证方法 |
|---|--------|------|------|---------|
| B1 | 情绪趋势冲动源 | 10min | 情绪趋势有变化时产生信号 | 检查 PriorityQueue |
| B2 | 时间节律冲动源 | 30min | 当前时段有历史模式时产生信号 | 检查 PriorityQueue |
| B3 | 随机漫游冲动源 | 10min | 随机产生信号 | 检查 PriorityQueue |
| B4 | 好奇心冲动源 | 20min | 对低命中率的记忆产生信号 | 信号关联到低 hit_count 记忆 |
| B5 | 行为模式冲动源 | 30min | 检测到行为模式时产生信号 | 检查 PriorityQueue |
| B6 | 疲劳度增长 | 每次发射 | 同源每次发射疲劳度 +0.15 | `fatigue_new = fatigue_old + 0.15` |
| B7 | 疲劳度半衰 | 15min | 疲劳度自然衰减 | `fatigue_now < fatigue_15min_ago` |
| B8 | 冲动抑制 | 有效优先级 <2 | 信号被丢弃不进队列 | `PriorityQueue` 中无该信号 |
| B9 | 冲动触发 | 空闲 >2min + 有效优先级 ≥2 | 消费者取到信号，LLM 生成 [内心独白] | chat_history 中出现 [内心独白] |
| B10 | 冲动 TTL 过期 | 队列中存活 > TTL | 超时信号被丢弃，不进 consumer | 注入 priority=99 信号，等待超过 TTL → get_next() 返回 None |
| B11 | 浅巩固触发 | 4h | consolidate_shallow() 被调用且不抛异常 | 无异常 + 耗时在合理范围 |
| B12 | 深巩固触发 | 24h | consolidate_deep() 被调用且不抛异常 | 无异常 + 耗时在合理范围 |
| B13 | 模式发现触发 | 6h | PatternDiscovery.run() 被调用且不抛异常；5 种模式（时间模式、情绪锚点、话题漂移、互动节奏、趋势检测）各有输出 | 检查 pattern_cache.json，5 种模式的 key 均存在且非空 |
| B14 | DMN idle-check | ~5min (Poisson) | DMN 线程检测用户空闲后触发 Level 1 预热（重建检索缓存）；空闲 >4h/24h 时分别触发浅/深巩固 | Level 1 预热后检索缓存时间戳更新；巩固触发链正确 |
| B15 | AI 巩固线程 | 1h | AI 自我表达记忆的独立巩固线程正常运行 | AI 记忆集合的 topic_tree / duplicates / notes 按时更新 |
| B16 | 线程存活 | 持续 | 所有 daemon 线程均存活（含 DMN、AI 巩固线程） | `threading.enumerate()` 检查 |
| B17 | 线程重启 | 崩溃后 | 线程崩溃后 lifecycle 自动重启 | 模拟线程崩溃 → 验证恢复 |

### 计分
```
后台节律分 = 通过的子项数 / 17
```

---

## 五链路分数总表

| 链路 | 子项数 | 核心问题 |
|------|--------|---------|
| 一：写入 | 12 | "存进去了吗？存对了吗？" |
| 二：检索+编织+认知 | 35 | "找得到吗？编织对了吗？回复靠谱吗？" |
| 三：跨轮记忆 | 9 | "隔几轮还记得吗？换种问法还行吗？" |
| 四：记忆演化 | 16 | "时间过了记忆质量退化了吗？" |
| 独立：后台节律 | 17 | "不等用户的时候，系统在干什么？干对了吗？" |

**不计算加权总分。五条链路各自独立计分。哪条低修哪条。**

---

*初稿 · 待你确认后进入实现阶段*
