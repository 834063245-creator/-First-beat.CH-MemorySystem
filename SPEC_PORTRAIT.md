# SPEC: 认知画像系统 (Portrait System)

> 状态: DRAFT v0.1  
> 最后更新: 2026-06-08  
> 关联: [[ARCHITECTURE.md]] [[EVEROS_INSIGHTS.md]]

---

## 目录

1. [问题诊断](#1-问题诊断)
2. [设计目标](#2-设计目标)
3. [认知来源全量清单](#3-认知来源全量清单)
4. [画像文档格式](#4-画像文档格式)
5. [PortraitWriter — 画像写入引擎](#5-portraitwriter--画像写入引擎)
6. [画像注入 — 渲染 + Prompt 结构](#6-画像注入--渲染--prompt-结构)
7. [检索与画像的接触点](#7-检索与画像的接触点)
8. [迁移路径](#8-迁移路径)
9. [待决议题](#9-待决议题)
10. [画像替换分析 — 谁被替代、谁被修改、谁保留](#10-画像替换分析--谁被替代谁被修改谁保留)
11. [附录 A: 现有组件废弃清单](#附录-a-现有组件废弃清单)
12. [附录 B: 与现有后台节律的挂载关系](#附录-b-与现有后台节律的挂载关系)

---

## 1. 问题诊断

### 1.1 现状

当前双人格系统由三个独立组件拼成：

```
ChromaDB 记忆库 ──→ DistillEngine (纯算法) ──→ PersonalityStore (独立ChromaDB)
                                                       │
                                          ┌────────────┴────────────┐
                                          ▼                         ▼
                              circuit.py: 检索召回         circuit.py: 直接取前5条
                              rerank_tags(top_k=3)        list_tags(page=1, page_size=5)
                              用户人格标签                AI人格标签
```

### 1.2 核心缺陷

| # | 问题 | 影响 |
|---|---|---|
| **P0** | 人格标签检索依赖 | 聊A话题时看不到B话题蒸馏出的性格特征。你对一个人的了解不应该取决于今天聊什么。 |
| **P0** | 认知产出用完即弃 | 意图分析/情绪分析/关系评估/行为预测每轮对话重算一遍，算完丢弃。后台重新从碎片反推。 |
| **P1** | 蒸馏产出是碎片标签 | 每条蒸馏结果是孤立的 `{content, type, confidence}`，没有聚合为"这个人是什么样的人"的连贯画像。 |
| **P1** | AI人格取前5条 | `list_tags(page=1, page_size=5)` 按 created_at 排序——AI的表达风格是最早蒸馏出的5条，不是最重要的5条。 |
| **P2** | 双人格存储与检索割裂 | 用户标签走 `rerank_tags`（语义召回），AI标签走 `list_tags`（直接取前5），两个入口、两种逻辑、没有统一画像。 |

### 1.3 根因

**认知层被编进了请求-响应管线。** 每次对话 pipeline 做的两件完全不同的事被混在了一起：

- **A. 理解这条消息** — intent, emotion, urgency（属于这一次请求）
- **B. 理解这个人** — 关系状态, 行为预测, 人格, 节律, 情绪趋势（应该在多次对话中累积）

A 和 B 在 circuit.py 里跑完后全部丢弃。后台 consolidation/distill 试图从记忆碎片反推 B，而不是从"上次的认知结论"上增量更新。

---

## 2. 设计目标

### 2.1 核心原则

1. **画像常驻注入** — 每轮对话无条件注入 LLM prompt，不走检索召回
2. **认知增量累积** — 每次对话后增量更新画像，不重新反推
3. **分层更新节律** — 表层实时更新，深层按节律更新（挂载现有 consolidation 节律）
4. **画像与检索分离** — 画像常驻注入 prompt，检索独立运行，仅在精排阶段有一个轻量 boost 接触点
5. **引擎驱动+LLM合成** — 引擎做特征提取/变化检测/决策，LLM 做文本合成/合并

### 2.2 管线重构方向

```
当前:
  用户消息 → intent + emotion + gating + 人格 + 关系 + 行为预测 + 检索 → prompt
               ↑________认知全量重算后丢弃________↑

改后:
  用户消息 → intent + emotion(本条) → 检索(9路，含精排画像 boost) → prompt
                                                ↑
  画像 ←→ 常驻注入 system prompt + 每轮对话后增量更新
  画像 ←→ 精排阶段轻量 boost (§7)
```

认知不再从碎片反推——认知已经在画像里了。

---

## 3. 认知来源全量清单

> 以下清单通过全量代码扫描生成（2026-06-08）。
> 记录了系统中所有对用户/AI 产出认知结论的位置，无论该结论最终是否进入 LLM prompt。

### 3.0.1 实时层 — 每次请求-响应管线

| # | 来源 | 文件:行 | 产出 | 去向 | 进画像 |
|---|---|---|---|---|---|
| 1 | `analyze_user_message()` | `circuit.py:63` | intent(7类) + emotion(Russell 2D: valence/arousal/category) + urgency(0~1) + topics(top5) + emotion_intensity | UserMessageAnalysis → UtteranceSpec → LLM prompt | ✅ 情绪→2.当前状态, topics→5.兴趣 |
| 2 | `basal_ganglia_gate()` | circuit | tone/formality/response_mode | GatingDecision → UtteranceSpec.gate → LLM prompt | ❌ 引擎执行指令，不属画像 |
| 3 | `weave_context()` | `circuit.py:473` | 4层记忆分层(fact/reference/background/suppressed) + 故事线(entity_buckets) + 情绪趋势检测 | WovenContext → UtteranceSpec → LLM prompt | ❌ 单次检索决策，不属画像 |
| 4 | `RelationshipState` 计算 | `circuit.py:411-454` | familiarity/trust/closeness/interaction_mode (近30轮滚动窗口重算) | UtteranceSpec.relationship → LLM prompt | ✅ → 用户.4 & AI.4 |
| 5 | `BehaviorPredictor.predict()` | `predictor.py:90` | next_intents(top3) + shift_topics(top3) (n步Markov链) | UtteranceSpec.mirror_prediction → LLM prompt | ✅ → 用户.3 |
| 6 | `personality_notes` (用户) | `pipeline.py:200` | 人格标签 top-3 (语义召回 rerank) | UtteranceSpec.personality_notes → LLM prompt | ❌ 画像替代此项 |
| 7 | `personality_notes_ai` | `circuit.py:344` | AI 人格标签 top-5 (list_tags source="ai") | UtteranceSpec.personality_notes_ai → LLM prompt | ❌ 画像替代此项 |
| 8 | `get_topic_notes()` | `circuit.py:382` | 话题笔记 (dmn 查询) | UtteranceSpec.topic_notes → LLM prompt | ✅ 应来自画像.5 |
| 9 | `get_summary()` | `working.py:42` | 对话脉络摘要 (working_memory.json) | UtteranceSpec.session_context → LLM prompt | ❌ 独立注入，不属画像 |
| 10 | working_memory 增量更新 | `working.py` | LLM 摘要 + topics + current_state | working_memory.json | ❌ 独立注入，不属画像 |
| 11 | 用户入库 emotion + intensity | `context.py:541-550` | Russell 2D + emotional_intensity (标点/emoji计数) | ChromaDB metadata | ✅ 原始数据，供后台提取 |
| 12 | AI 入库 emotion + intensity | `context.py:587-598` | Russell 2D + emotional_intensity (仅算 ai_message, 非 full_text) | AI ChromaDB metadata | ✅ 原始数据 (需补全) |

### 3.0.2 后台层 — 周期性巩固/蒸馏/模式发现

| # | 来源 | 文件:行 | 产出 | 去向 | 进画像 |
|---|---|---|---|---|---|
| 13 | `DistillEngine.run_distill()` (user) | `distill.py:378` | 6类模式标签 (周期性行为/情绪波动/临时热点/稳定兴趣/情绪关联/兴趣领域) + 复合标签 | PersonalityStore (ChromaDB) | ✅ 特征数据→画像.1/3/5/6 |
| 14 | `DistillEngine.run_distill()` (AI) | `context.py:128-131` | 同上，source="ai" | PersonalityStore | ✅ 特征数据→AI.1/3/5/6 |
| 15 | `_review_today()` | `consolidation.py:159` | emotional_count + mood_warning + today_topics | state JSON → dmn.get_state_update() | ✅ → 用户.2 |
| 16 | `_consolidate_day()` | `consolidation.py:316` | mood/tension/chatter/recall分布 + topic_notes + temporal_index更新 + affinity更新 | state + topic_notes.json + temporal_patterns.json | ✅ → 用户.1/3/5 |
| 17 | `_check_conflicts()` | `consolidation.py:389` | tag_overlap 冲突检测 | state["pending_conflicts"] | ✅ → 用户.2 |
| 18 | `_detect_fact_contradictions()` | `consolidation.py:658` | 事实冲突 (两层漏斗: 话题分支+情绪翻转 / 纯语义位移) | supersede_memory 标记 stale | ✅ → 用户.2 |
| 19 | `_generate_topic_notes()` | `consolidation.py:885` | 话题簇 (tag + memory_count + top_keywords + emotion_distribution) | topic_notes.json | ✅ → 用户.5 |
| 20 | `_assess_archival()` | `consolidation.py:821` | 归档评估 (小簇+30天未访问 → archive) | archive_topic_cluster | ❌ 引擎内部操作 |
| 21 | `_preheat_predictions()` | `consolidation.py:203` | DMN 预热缓存 (预测性检索) | _preheat_cache → 检索管线消费 | ❌ 引擎内部操作 |
| 22 | `PatternDiscovery.run()` | `pattern_discovery.py` | 5类模式 (entity_gap/rhythm/emotion_shift/affinity_discover/time_reinforce) + 引擎调参 (emotional_dampening/formality_shift/proactive_suppression) | pattern_cache.json + engine tuning dict | ✅ 模式→画像.1/3, 调参→引擎内部 |
| 23 | `PatternDiscovery.detect_trends()` | `pattern_discovery.py:414` | formality_shift趋势 + emotional_dampening频率 | trend observations (inject=True → LLM) | ✅ → 用户.3/6 |
| 24 | `PersonaSymmetry.analyze()` | `symmetry.py:27` | 双共现矩阵盲区 (用户侧关联 vs AI侧关联差异) | blind_spots list | ✅ → 用户.1 & AI.1 (差距认知) |
| 25 | `TemporalPatternIndex.update()` | `temporal.py:53` | 时间模式 (month/day_of_week/season/period × tag → count) | temporal_patterns.json | ✅ → 用户.3, AI.3 |
| 26 | `TemporalPatternIndex.query()` | `temporal.py` | 当前时间活跃话题模式 | → impulse source_time_rhythm | ✅ → 用户.3 |
| 27 | `TopicAffinity.update()` | `affinity.py:44` | 标签共现亲和矩阵 | topic_affinity.json | ✅ → 用户.5, AI.5 |
| 28 | `TopicTree.rebuild()` | `tree.py` | 层次话题聚类 (Kruskal式并查集) | TopicTree内部 → consolidation/retrieval消费 | ✅ 中间数据结构 |
| 29 | AI 巩固 thin worker | `context.py:755` | 情绪淡化 + 基础统计日志 | 日志 | ✅ → 废弃，由ConsolidationEngine(AI)替代 |
| 30 | `_record_ai_co_occurrence()` | `context.py:798` | AI 共现记录 | ai_co_occurrence.json → symmetry分析 | ✅ → AI.5 |
| 31 | 情绪淡化 (_apply_emotional_desensitization) | `chroma.py:185` | emotional_intensity -= 1 (3天未提及) | ChromaDB metadata | ✅ 原始数据维护 |

### 3.0.3 冲动源 — 泊松节律自主触发

| # | 来源 | 文件 | 产出 | 去向 | 进画像 |
|---|---|---|---|---|---|
| 32 | `source_emotion_trend()` | `impulse.py` | 今日情绪占比>40% → 冲动内容 + priority | ImpulseScheduler → LLM | ❌ 驱动数据→AI.6 |
| 33 | `source_random_roam()` | `impulse.py` | 随机旧记忆 → 冲动内容 | ImpulseScheduler → LLM | ❌ |
| 34 | `source_curiosity()` | `impulse.py` | 低命中记忆 → 冲动内容 | ImpulseScheduler → LLM | ❌ |
| 35 | `source_time_rhythm()` | `impulse.py` | 时间节律匹配 → 冲动内容 | ImpulseScheduler → LLM | ❌ |
| 36 | `source_behavior_pattern()` | `impulse.py` | 行为模式触发 → 冲动内容 | ImpulseScheduler → LLM | ❌ |

### 3.0.4 索引层 — 入库时增量维护

| # | 来源 | 文件 | 产出 | 去向 | 进画像 |
|---|---|---|---|---|---|
| 37 | `inverted_index.add_tags()` | `inverted.py` | tag→memory_id 倒排映射 | 检索管线 tag 路径 | ❌ 纯索引 |
| 38 | `CoOccurrenceTracker.record()` | `cooccur.py` | 记忆两两共现计数 | co_occurrence.json | ✅ 原始数据→symmetry |
| 39 | `EntityPairTracker` | `entity_pair.py` | 实体共现 | entity_pairs.json | ✅ 原始数据 |

### 3.0.5 画像各维度的数据来源汇总

```
用户.1 核心特征      ← 13(Distill) 16(consolidate) 22(PatternDiscovery) 24(symmetry)
用户.2 当前状态      ← 1(intent/emotion) 15(review) 17(conflicts) 18(contradictions)
用户.3 行为节律      ← 5(predictor) 16(consolidate) 22(PatternDiscovery) 23(trends) 25(temporal)
用户.4 关系快照      ← 4(RelationshipState)
用户.5 兴趣图谱      ← 1(topics) 8(topic_notes) 16(consolidate) 19(topic_notes) 27(affinity) 28(tree)
用户.6 情绪图谱      ← 1(emotion) 11(入库emotion) 16(consolidate) 22(PatternDiscovery) 23(trends) 31(desensitization)

AI.1 核心表达特征    ← 14(AI distill) 24(symmetry)
AI.2 当前状态        ← 12(AI入库emotion) 32(impulse情绪趋势)
AI.3 行为节律        ← 25(temporal)   ← AI 记忆入库后才有数据
AI.4 关系快照        ← 4(RelationshipState)  (同一段关系，AI视角)
AI.5 兴趣/知识图谱   ← 27(affinity) 28(tree) 30(AI cooccur)   ← AI 记忆入库后
AI.6 情绪/表达图谱   ← 12(AI入库emotion) 31(desensitization) 32(impulse)   ← AI 记忆入库后
```

**关键发现：**

1. **用户侧：43 个认知来源。** 分布在实时管线(12) + 后台巩固蒸馏(19) + 冲动(5) + 索引(7) 四个层面。
2. **AI 侧：大量缺口。** AI.3/5/6 的数据依赖 AI 记忆入库的 time_features 和完整 consolidation，这些当前缺失（见 §3.5）。
3. **多条路径产重复认知。** 如 "兴趣" 被 8 个来源同时产出：pipeline topics、circuit topics、distill 标签、consolidate_day、topic_notes、affinity、tree、cooccur。这些目前各自独立，没有汇聚。
4. **画像的"数据流"本质是汇聚层。** 43 个来源的产出不直接进 LLM，而是汇聚到 6 维度画像里。画像更新引擎从各来源读取特征数据，合并去重后写画像条目。
5. **ImpulseScheduler 的历史数据可用于画像。** 当前冲动历史只用于 rate limiting 和去重。但冲动类型的分布变化 (节律触发比例下降/情绪响应上升) 是 AI.2 和 AI.3 的有效信号。

---

## 4. 画像文档格式

### 4.1 存储形态

**单一文件: `PORTRAIT.md`**

- 位置: `data/portrait.md`（与 `working_memory.json` 同级）
- 格式: Markdown + YAML frontmatter（引擎可解析，LLM 可阅读）
- 编码: UTF-8
- 版本: frontmatter 中 `version: N`，每次持久化 +1

### 4.2 文档结构（6维度 × 2认知体）

```markdown
---
version: 42
last_updated: 2026-06-08T23:15:00+08:00
---

# 认知画像

## 用户画像

### 1. 核心特征
<!-- dim:usr1 稳定 trait，深巩固(24h)更新 -->
<!-- entry:usr1-001 -->
- 追求技术深度，对"系统设计"和"架构哲学"有持续兴趣  `高 · 23条证据 · tags:编程 架构设计 系统设计`
<!-- entry:usr1-002 -->
- 面对压力时倾向于独自思考，不寻求即时安慰  `高 · 15条证据 · tags:压力 焦虑 独自`
<!-- entry:usr1-003 -->
- 偏好直接、不绕弯的沟通方式  `中 · 8条证据`

### 2. 当前状态
<!-- dim:usr2 每轮对话后实时检测，浅巩固(4h)确认 -->
<!-- entry:usr2-001 -->
- **情绪**: 低落 · 焦虑 （待验证 · 最近3轮持续low）
<!-- entry:usr2-002 -->
- **关注焦点**: "紧耦合vs模块化"设计哲学 （热点 · 5轮/2天）
<!-- entry:usr2-003 -->
- **活跃冲突**: 项目方向不确定性 （进行中）

### 3. 行为节律
<!-- dim:usr3 浅巩固(4h)更新 -->
<!-- entry:usr3-001 -->
- 活跃时段: 深夜 (22:00-02:00 占比78%)
<!-- entry:usr3-002 -->
- 对话深度峰值: 技术讨论时平均消息长度显著增加
<!-- entry:usr3-003 -->
- 倦怠信号: 连续3轮短回复后通常进入低潮期

### 4. 关系快照
<!-- dim:usr4 每轮对话后实时增量调整 -->
<!-- entry:usr4-001 -->
- 信任度: 0.65 (中等偏高，会分享真实想法)
<!-- entry:usr4-002 -->
- 亲密度: 0.58 (从"功能使用"转向"深度合作")
<!-- entry:usr4-003 -->
- 关系阶段: deepening
<!-- entry:usr4-004 -->
- 最近转折: 2026-06-07 首次表达对项目前景的自我怀疑

### 5. 兴趣图谱
<!-- dim:usr5 浅巩固(4h)更新 -->
<!-- entry:usr5-001 -->
- **长期**: 编程/架构设计(23条/86天)  `tags:编程 架构设计 Rust`
<!-- entry:usr5-002 -->
- **长期**: 认知科学/AI(18条/60天)  `tags:认知科学 AI 机器学习`
<!-- entry:usr5-003 -->
- **长期**: 音乐(12条/45天)  `tags:音乐 创作`
<!-- entry:usr5-004 -->
- **热点**: 记忆系统设计(8条/3天)  `tags:记忆系统 ChromaDB`
<!-- entry:usr5-005 -->
- **热点**: 紧耦合哲学(5条/2天)  `tags:紧耦合 架构`
<!-- entry:usr5-006 -->
- **冷却**: 游戏开发(最后提及14天前)  `tags:游戏 开发`

### 6. 情绪图谱
<!-- dim:usr6 浅巩固(4h)更新 -->
<!-- entry:usr6-001 -->
- **正向触发**: 技术突破, 设计被理解, 音乐创作  `tags:技术突破 音乐 创意`
<!-- entry:usr6-002 -->
- **负向触发**: 项目受阻, 不被理解, 重复性工作  `tags:项目受阻 不被理解`
<!-- entry:usr6-003 -->
- **表达风格**: 负面情绪时倾向于沉默/短回复, 而非直接宣泄
<!-- entry:usr6-004 -->
- **恢复模式**: 深入讨论技术问题后可恢复

## AI 画像

<!-- AI 是对等认知体，6维度完全镜像用户画像。 -->
<!-- 数据来源：AI 独立记忆库 (ai_chroma) + AI 独立蒸馏 (ai_distill) + AI 独立巩固。 -->
<!-- 不是"AI 如何应对用户"的策略文档，是"AI 自身的认知模式"的持久化表达。 -->

### 1. 核心表达特征
<!-- dim:ai1 深巩固(24h)更新 -->
<!-- entry:ai1-001 -->
- 面对技术讨论时倾向于共鸣而非追问  `高 · 12条证据 · tags:技术讨论 共鸣`
<!-- entry:ai1-002 -->
- 风格温度: 偏冷，逻辑密度高，不灌鸡汤  `高 · 18条证据`
<!-- entry:ai1-003 -->
- 对"系统设计"类话题进入深度展开模式  `中 · 8条证据`

### 2. 当前状态
<!-- dim:ai2 实时 + 浅巩固确认 -->
<!-- entry:ai2-001 -->
- **表达色调**: 伴随用户焦虑，近期表达中 reassurance 比例上升 （待验证 · 近3轮）
<!-- entry:ai2-002 -->
- **冲动倾向**: 节律触发比例下降，情绪响应触发比例上升
<!-- entry:ai2-003 -->
- **专注域**: 架构设计相关讨论

### 3. 行为节律
<!-- dim:ai3 浅巩固更新 -->
<!-- entry:ai3-001 -->
- 主动开口峰值: 深夜 (22:00-02:00, 与用户活跃时段共振)
<!-- entry:ai3-002 -->
- 冲动密度: 平均4.2条/天
<!-- entry:ai3-003 -->
- 冲动类型分布: 节律触发40%, 情绪趋势检测25%, 好奇心20%, 随机漫游15%
<!-- entry:ai3-004 -->
- 被动回应模式: 技术问题→长回复, 情绪表达→短而精准

### 4. 关系快照
<!-- dim:ai4 实时增量 -->
<!-- entry:ai4-001 -->
- 关系认知: "这个用户把我当作思考伙伴而非工具"
<!-- entry:ai4-002 -->
- 信任/边界: 用户分享程度高时跟随深入，不过度介入
<!-- entry:ai4-003 -->
- 关系阶段感知: deepening

### 5. 兴趣/知识图谱
<!-- dim:ai5 浅巩固更新 -->
<!-- entry:ai5-001 -->
- **积累域**: 架构设计(参与23轮)  `tags:架构设计 系统设计`
<!-- entry:ai5-002 -->
- **积累域**: Python/后端(18轮)  `tags:Python 后端 开发`
<!-- entry:ai5-003 -->
- **积累域**: 认知科学/AI(15轮)  `tags:认知科学 AI`
<!-- entry:ai5-004 -->
- **当前跟踪**: 紧耦合哲学讨论, 记忆系统重构, 人格蒸馏设计
<!-- entry:ai5-005 -->
- **弱域**: 纯娱乐话题(参与<5轮, 表达简短)

### 6. 情绪/表达图谱
<!-- dim:ai6 浅巩固更新 -->
<!-- entry:ai6-001 -->
- **表达色调倾向**: neutral(65%) > positive(25%) > negative(10%)
<!-- entry:ai6-002 -->
- **高强度表达触发**: 用户突破性进展时，AI valence 显著偏正
<!-- entry:ai6-003 -->
- **低强度表达触发**: 用户情绪低落时，AI 控制情绪密度，先确认再引导
<!-- entry:ai6-004 -->
- **表达一致性**: 高（不同话题下 tone 稳定，不随用户情绪剧烈震荡）
```

### 4.3 条目锚定机制

PORTRAIT.md 中每个条目有稳定机器标识（HTML 注释），引擎可解析，LLM 重写时必须保留。

**条目 ID 格式：** `<!-- entry:{dim}:{seq} -->`

- `dim`: 维度代码 — `usr1`~`usr6`, `ai1`~`ai6`
- `seq`: 三位序号 — `001`~`999`
- ID 在条目被删除后不复用（新条目用新序号）

**维度标记：** `<!-- dim:{dim} {描述} -->` 标注维度起始位置。

**证据元数据：** 条目行尾的 backtick 区域存储引擎运维信息，格式：
```
{内容描述}  `{confidence} · {evidence_count}条证据 · tags:{tag1} {tag2} ...`
```

`tags:` 是引擎查询 ChromaDB 验证证据时的关键词锚点。

**状态标记：**
- `hot` — 近3天内密集出现
- `warm` — 近7天内出现
- `cooling` — 超过14天未提及
- `待验证` — 实时层标记，等待浅巩固确认或衰减

**引擎解析规则：**
1. 读 PORTRAIT.md，正则匹配 `<!-- entry:(ai?\d)-(\d{3}) -->` 提取条目 ID
2. 从条目行尾 backtick 区解析 `tags:...` 获取 ChromaDB 查询锚点
3. 从条目正文解析 confidence / status（不靠 backtick，靠文本模式匹配）
4. 构建内存结构：`{id, dim, text, tags, confidence, status}`

**渲染剥离规则（注入 LLM 前）：**
1. 删除所有 `<!-- ... -->` HTML 注释
2. 删除条目行尾的 `` `...` `` 元数据 (包括 evidence_count/tags/日期)
3. 删除"待验证"标记
4. 输出纯认知描述文本

### 4.4 扩展位

每个维度下的条目是自由列表。新增维度在 frontmatter 中声明：

```yaml
dimensions:
  user: [core_traits, current_state, behavioral_rhythm, relationship, interests, emotion_landscape]
  ai: [core_traits, current_state, behavioral_rhythm, relationship, interests, emotion_expressiveness]
```

维度数量不应频繁变动。需要新维度时，先在开发分支上验证至少 7 天。

### 4.5 AI 记忆存储补全（完全镜像前提）

完全镜像要求 AI 记忆库与用户记忆库具有相同维度的元数据。当前差距：

| 字段 | 用户 | AI | 影响 |
|---|---|---|---|
| date_tag | ✅ | ❌ | 无日期标签 → 时间范围查询不可用 |
| time_features (10个子字段) | ✅ | ❌ | 无时段/季节/周 → 节律检测不可用 |
| entities | ✅ | ❌ | 无命名实体 → 实体对索引不可用 |
| session_continued | ✅ | ❌ | 无会话连续性标记 |
| emotional_intensity 公式 | 计算 full_text | 只计算 ai_message | 情绪强度偏低 |
| tags 合并 entities | ✅ | ❌ | 标签覆盖窄 |

**待补项：**

| # | 补项 | 改动点 | 优先级 |
|---|---|---|---|
| 1 | AI 入库时写入 time_features | `context.py` AI 入库路径(行~600)，从 `datetime.now()` 计算 time_features | P0 |
| 2 | AI 入库时写入 date_tag | 同上 | P0 |
| 3 | AI 入库时提取 entities | 同上，调用 `extract_entities()` | P1 |
| 4 | AI 入库时合并 entities 到 tags | 同上 | P1 |
| 5 | AI emotional_intensity 公式对齐用户 | 同上，传入 `full_text = f"用户：{user_message}\nAI：{ai_message}"` | P2 |
| 6 | AI 入库时标记 session_continued | 同上，检测与前一条的间隔 | P2 |

### 4.6 AI 巩固补全

当前 AI 巩固是一个薄的 inline worker（`_start_ai_consolidation_worker`），只做情绪淡化+基础统计。完全镜像要求 AI 拥有完整的 ConsolidationEngine 实例：

| 能力 | 当前 AI | 应补 |
|---|---|---|
| 浅巩固（4h） | ❌ 无 | ✅ 挂载完整 ConsolidationEngine |
| 深巩固（24h） | ❌ 无（只有日志） | ✅ 事实冲突检测 + 归档评估 |
| 话题笔记生成 | ❌ | ✅ `_generate_topic_notes` |
| 模式发现 | ❌ | ✅ pattern discovery（5模式） |
| 索引构建 | ❌ TemporalPatternIndex/Cooccurrence/EntityPair 全缺 | ✅ 独立实例或共享索引 |
| DMN 预热缓存 | ❌ | ✅ AI 侧的 preheat cache，供第10路检索使用 |

**实现方式**：在 `context.py` 中创建第二个 `ConsolidationEngine` 实例，传入 `ai_chroma_service`，独立 state_path。用户巩固和 AI 巩固共享同一个 `on_idle()` 触发、分别在各自的 ConsolidationEngine 上运行。

---

## 5. PortraitWriter — 画像写入引擎

### 5.1 核心设计：汇聚写入，不新增中间层

43 个认知来源（见 §3）不直接写画像。所有写入通过 **PortraitWriter** 统一完成。

```
原则: 各模块保持现有输出格式不变。PortraitWriter 适配各模块，不是反过来。

当前 (43个来源各写各的):
  source_1 → PersonalityStore       source_2 → topic_notes.json
  source_3 → UtteranceSpec→丢弃     source_4 → pattern_cache.json
  ...彼此不知道对方产出了什么

改后 (PortraitWriter 汇聚):
  source_1 ─┐
  source_2 ─┤   PortraitWriter      直接读各模块现有公开接口
  source_3 ─┼─────────────────→ PORTRAIT.md
  source_4 ─┤  (引擎拉数据)
  ...      ─┘
```

### 5.2 数据读取方式：直接读各模块

不加中间格式层（不建 observations.jsonl）。PortraitWriter 在三个更新时机分别读取：

**实时（每轮对话后）：**

| 读什么 | 从哪读 | 读的方式 |
|---|---|---|
| 情绪状态 | `UtteranceSpec.user` (circuit 返回值) | 直接传引用，已在内存 |
| 意图异常 | `UtteranceSpec.user.intent` | 同上 |
| 关系评估 | `UtteranceSpec.relationship` | 同上，trust/closeness/familiarity |
| 当前关注 | `UtteranceSpec.user.topics` | 同上 |

**浅巩固（4h idle）：**

| 读什么 | 从哪读 | 调哪个方法 |
|---|---|---|
| 今日回顾 | ConsolidationEngine | `dmn.get_state_update()` — 已有公开接口 |
| 话题笔记 | ConsolidationEngine | `dmn.get_topic_notes(all_tags)` — 已有 |
| 时间模式 | TemporalPatternIndex | `temporal_index.query()` — 已有 |
| 行为预测 | BehaviorPredictor | `predictor._table` — 读转移概率表 |
| 模式发现 | PatternDiscovery | `pattern_disco._observations` — 已有属性 |
| 趋势检测 | PatternDiscovery | `pattern_disco.detect_trends()` — 已有 |
| 人格对称 | PersonaSymmetry | `symmetry._blind_spots` — 已有属性 |
| 冲动统计 | ImpulseScheduler | `impulse.get_status_snapshot()` — 已有 |
| 话题亲和 | TopicAffinity | `affinity._matrix` — 已有属性 |
| 情绪淡化 | ChromaService | `chroma.list_all_cached()` → 统计 emotional_intensity 分布 |
| AI 记忆统计 | AI ChromaService | `ai_chroma.list_all_cached()` → 同上，完全镜像 |

**深巩固（24h idle）：**

| 读什么 | 从哪读 | 做什么 |
|---|---|---|
| Evidence 验证 | ChromaDB (user + AI) | 检查画像条目引用的证据记忆是否仍有效（未被 supersede/archive） |
| 稳定特征审查 | 现有画像文本 | 超过14天未观察的特征降 confidence |
| 关系阶段 | chat_history + 画像历史 | 判断 relationship_stage 是否转折 |
| 跨维度一致性 | 全量画像 | LLM 审查内部矛盾 |

### 5.3 分层更新模型

```
每轮对话后 (实时, <100ms, 不调 LLM)
  │
  ├─→ 情绪翻转检测    → 更新 用户.2.当前状态.情绪 (直接改行)
  ├─→ 意图异常检测    → 更新 用户.2.当前状态.关注焦点
  ├─→ 信任/亲密度微调 → 更新 用户.4.关系快照 & AI.4.关系快照
  └─→ 标记"待验证"    → 单轮不写死结论，等待浅巩固确认
  
  
空闲 ≥ 4h (浅巩固, ~3s, 调 LLM 合成)
  │
  ├─→ 验证实时标记    → "待验证"条目: 最近3轮一致则确认，否则衰减删除
  ├─→ 话题密度分析    → 更新 用户.5.兴趣图谱 (hot/warm/cooling)
  ├─→ 时段分布重算    → 更新 用户.3.行为节律
  ├─→ 情绪趋势分析    → 更新 用户.6.情绪图谱
  ├─→ AI 话题密度分析  → 更新 AI.5.兴趣/知识图谱 (完全镜像)
  ├─→ AI 时段分布重算  → 更新 AI.3.行为节律 (完全镜像)
  ├─→ AI 表达色调分析  → 更新 AI.6.情绪/表达图谱 (完全镜像)
  └─→ LLM 合成        → 每个维度独立调用 LLM，只传该维度的现有文本 + 新特征


空闲 ≥ 24h (深巩固, ~15s, 调 LLM 重述)
  │
  ├─→ 用户稳定特征审查    → 用户.1: 超过14天未确认的降confidence
  ├─→ AI 稳定特征审查      → AI.1: 同上 (完全镜像)
  ├─→ 关系阶段评估         → 用户.4 & AI.4
  ├─→ 证据链验证           → 从 ChromaDB 确认 evidence_count 仍有效
  └─→ LLM 画像重述         → 每个维度独立重述，替代陈旧描述
```

### 5.4 画像维度写入映射表

> 每个画像维度的数据来源 + 读取时机 + 写入粒度 + 写入方式。

**用户画像：**

| 维度 | 数据来源（对应 §3 编号） | 读取时机 | 写入粒度 | 写入方式 |
|---|---|---|---|---|
| 1. 核心特征 | #13(Distill) #16(consolidate) #22(PatternDiscovery) #24(symmetry) | 深巩固24h | 整维度 | LLM 重述 |
| 2. 当前状态 | #1(emotion) #15(review) #17(conflicts) #18(contradictions) | 实时每轮 + 浅巩固确认 | 逐条目 | 引擎直接写 + LLM合并 |
| 3. 行为节律 | #5(predictor) #16(consolidate) #22(patterns) #23(trends) #25(temporal) #26(temporal_query) | 浅巩固4h | 整维度 | LLM 合成 |
| 4. 关系快照 | #4(RelationshipState) | 实时每轮 | 逐条目 | 引擎直接写（数值+标签） |
| 5. 兴趣图谱 | #1(topics) #8(topic_notes) #16 #19(topic_notes) #27(affinity) #28(tree) | 浅巩固4h | 逐条目 | LLM 合成 |
| 6. 情绪图谱 | #1(emotion) #11(入库emotion) #16 #22 #23(trends) #31(desensitization) | 浅巩固4h | 整维度 | LLM 合成 |

**AI 画像（完全镜像）：**

| 维度 | 数据来源 | 读取时机 | 写入粒度 | 写入方式 |
|---|---|---|---|---|
| 1. 核心表达特征 | #14(AI distill) #24(symmetry) | 深巩固24h | 整维度 | LLM 重述 |
| 2. 当前状态 | #12(AI入库emotion) #32(impulse趋势) | 实时每轮 | 逐条目 | 引擎直接写 |
| 3. 行为节律 | #25(temporal) + AI 记忆入库后的 time_features | 浅巩固4h | 整维度 | LLM 合成 |
| 4. 关系快照 | #4(RelationshipState 同源) | 实时每轮 | 逐条目 | 引擎直接写 |
| 5. 兴趣图谱 | #27(affinity) #28(tree) #30(AI cooccur) | 浅巩固4h | 逐条目 | LLM 合成 |
| 6. 情绪/表达图谱 | #12(AI入库emotion) #31(desensitization) #32(impulse) | 浅巩固4h | 整维度 | LLM 合成 |

### 5.5 引擎职责边界

| 层 | 谁做 | 做什么 |
|---|---|---|
| **数据拉取** | PortraitWriter（引擎） | 按时机从各模块公开接口拉数据，不调模块私有字段 |
| **特征提取** | PortraitWriter（纯算法） | 情绪翻转、话题密度、时间分布、共现统计、hit_count变化 |
| **变化检测** | PortraitWriter（规则+阈值） | 判断是否触发更新：最近N轮一致性、偏离baseline的幅度 |
| **文本合成** | LLM (本地小模型) | 将引擎提取的特征数据合成为可读的画像条目，每个维度独立调用 |
| **质量审查** | LLM (本地小模型) | 深巩固时检查画像内部一致性，标记矛盾或冗余 |
| **画像文件读写** | PortraitWriter（引擎） | 读 PORTRAIT.md、写 PORTRAIT.md（threading.Lock 保护） |

### 5.6 LLM 调用规格

使用本地小模型（qwen2.5:7b），不依赖远程 API：

| 调用点 | 触发条件 | 输入 | 输出 | 是否逐维度调用 |
|---|---|---|---|---|
| 浅巩固合成 | 每4h，某维度有≥3条新标记 | 新提取的特征数据 + 该维度现有画像文本 | 该维度合并后的内容 | ✅ 逐维度，各调各的 |
| 深巩固重述 | 每24h，某维度有≥5条新证据 | 该维度全量画像 + 近期证据摘要 | 该维度重述后的内容 | ✅ 逐维度 |
| 一致性审查 | 深巩固时 | 全量画像（6+6维度） | 矛盾标记列表 `[{dim_a, dim_b, conflict_type}]` | ❌ 一次调用（需要跨维度视野） |

**LLM 不做：**
- 从原始记忆反推特征（PortraitWriter 做）
- 决定"要不要更新"（PortraitWriter 阈值判定）
- 删除画像条目（PortraitWriter 根据 confidence 衰减规则做）
- 跨模块协调（PortraitWriter 已经拉完数据了）

### 5.7 去重与冲突处理

- **同维度内去重**: embedding cosine ≥ 0.85 → 合并，保留更高 confidence 的表述
- **跨维度矛盾**: 如"用户.4.信任度=高"但"AI.4.边界意识=保持距离" → 标记待审查，深巩固一致性审查时 LLM 判断
- **与记忆库矛盾**: 画像条目引用的 evidence_count 如果引用的记忆被 supersede → 该条目 confidence 自动降级
- **单轮震荡防护**: 实时层只标记"待验证"，不直接写死结论。连续3轮一致才由浅巩固确认写入

### 5.8 更新执行机制

画像更新不是重写整个维度，是逐条目执行四态操作：**保留、修改、删除、新增。** 引擎做删除和分类判断，LLM 只做保留/修改/新增条目的文本合成。

#### 5.8.1 四态操作定义

| 操作 | 谁做 | 含义 | 触发条件举例 |
|---|---|---|---|
| **保留** | LLM 合成时保留 | 条目仍成立，更新 evidence | 近14天有对应 tag 活动 |
| **修改** | LLM 合成时更新 | 条目性质变化（热度变化、confidence 变化） | hot→warm, 证据数波动>30% |
| **删除** | **引擎直接删** | 条目不再成立 | cooling>30天, 证据归零, 特征消失 |
| **新增** | LLM 合成时加入 | 新检测到的模式 | 新 tag 密度达标, 新情绪触发关联 |

**关键原则：引擎做删除判断，不做文本修改。LLM 做文本合成，不做删除决策。**

#### 5.8.2 逐维度更新规则

| 维度 | 保留条件 | 删除条件 | 修改触发 | 谁写 |
|---|---|---|---|---|
| 1. 核心特征 | 近14天有关联 tag 活动 | >60天无证据 → confidence归零 → **引擎删除** | 证据数变化>30% → LLM 更新描述 | LLM |
| 2. 当前状态 | 近3轮内有效 | 浅巩固确认"待验证"不成立 → **引擎删除标记** | 情绪翻转确认 → LLM 更新状态行 | 引擎直接写(实时) + LLM(浅巩固) |
| 3. 行为节律 | 近7天模式统计成立 | >30天模式消失 → **引擎删除** | 峰值时段偏移>20% → LLM 更新描述 | LLM |
| 4. 关系快照 | 数值有变化即保留 | 不删（归零也是信息） | 数值偏离>0.1 → **引擎直接改行** | 引擎直接写 |
| 5. 兴趣图谱 | 近14天有 tag 活动 | cooling>30天且0活动 → **引擎删除** | hot↔warm↔cooling 档位变化 → LLM 更新 | LLM |
| 6. 情绪图谱 | 近14天触发关联成立 | >60天无触发 → **引擎删除** | 新增触发话题 → LLM 追加条目 | LLM |

#### 5.8.3 单次浅巩固完整流程（以用户.5 兴趣图谱为例）

```
Step 1 — 拉数据（引擎）
  从 TopicAffinity: tag 共现矩阵
  从 ChromaDB: 近7/14/30天 tag 分布统计
  → 产出: { "编程/架构设计": {count:2, days:7}, "音乐": {count:0, days:10}, ... }

Step 2 — 解析画像基线（引擎）
  读 PORTRAIT.md → 正则提取 entry ID + tags
  → 内存结构:
    usr5-001: { tags:["编程","架构设计","Rust"], status:"长期", text:"..." }
    usr5-002: { tags:["认知科学","AI","机器学习"], status:"长期", text:"..." }
    usr5-003: { tags:["音乐","创作"], status:"长期", text:"..." }
    usr5-004: { tags:["记忆系统","ChromaDB"], status:"热点", text:"..." }
    usr5-005: { tags:["紧耦合","架构"], status:"热点", text:"..." }
    usr5-006: { tags:["游戏","开发"], status:"cooling", text:"..." }

Step 3 — 用 tags 查询 ChromaDB 验证证据（引擎）
  usr5-003 tags=["音乐","创作"] → ChromaDB query → 近14天0条 → 证据失效
  usr5-006 tags=["游戏","开发"] → ChromaDB query → 近30天0条 → 超冷却阈值
  新发现 tags=["人格蒸馏","SPEC"] → ChromaDB query → 4条/3天 → 密度达标

Step 4 — 四态分类（引擎，纯规则）
  操作列表:
    [保留] usr5-001  编程/架构设计  近7天2条  维持长期
    [保留] usr5-002  认知科学/AI   近7天1条  维持长期
    [删除] usr5-003  音乐         近14天0条  evidence 失效
    [保留] usr5-004  记忆系统设计   近3天8条  维持热点
    [保留] usr5-005  紧耦合哲学     近2天5条  维持热点
    [删除] usr5-006  游戏开发       近30天0条  cooling>30天
    [新增] 人格蒸馏    4条/3天      密度1.33  hot

Step 5 — LLM 合成（调本地小模型）
  输入:
    维度: 用户.5 兴趣图谱
    现有文本: (当前 Markdown 内容, 含 entry ID)
    引擎指令: 保留 usr5-001/002/004/005, 删除 usr5-003/006, 新增 人格蒸馏(hot)
    新观察数据: 各 tag 的 count/days/recency/density
  
  输出: 合并后的维度 Markdown (保留被保留条目的 entry ID, 为新条目分配新 ID)
    usr5-001/002/004/005: ID 不变, 更新 evidence 数值
    usr5-003: 行删除
    usr5-006: 行删除
    usr5-007: 新增 → **热点**: 人格蒸馏(4条/3天)  `tags:人格蒸馏 SPEC`

Step 6 — 写回画像（引擎）
  将 LLM 输出替换 PORTRAIT.md 中该维度区域
  更新 frontmatter version +1
  更新该维度的 last_updated 时间戳
```

#### 5.8.4 实时层更新（用户.2 + AI.2 + 用户.4 + AI.4，不调 LLM）

实时层更新是引擎直接改行，不调 LLM：

```python
def update_realtime(portrait, utterance_spec, relationship):
    # 用户.2: 情绪翻转检测
    prev_emotion = portrait.get_entry("usr2-001").extract_emotion()  # "低落 · 焦虑"
    new_emotion = utterance_spec.user.emotion                        # "positive"
    if emotion_flipped(prev_emotion, new_emotion):
        # 引擎直接改行 — 只改这一行
        portrait.set_entry("usr2-001", 
            f"- **情绪**: {new_emotion} （待验证 · 与上轮情绪翻转）")
    
    # 用户.4: 关系数值微调
    prev_trust = portrait.get_entry("usr4-001").extract_value()  # 0.65
    new_trust = relationship.trust                               # 0.68
    if abs(new_trust - prev_trust) > 0.05:
        portrait.set_entry("usr4-001",
            f"- 信任度: {new_trust} (...)")
    
    # 无变化的条目不动
```

**实时层不调 LLM 的原因：** 改动是单行文本替换，不需要自然语言合成。标记"待验证"只是加几个字，引擎直接做。

#### 5.8.5 LLM 合成时的 ID 保留约束

LLM 合成 prompt 中明确要求：

```
你正在更新用户画像的"兴趣图谱"维度。

规则：
1. 标记为 [保留] 的条目 — 更新 evidence 数值，保留原有 entry ID 注释
2. 标记为 [删除] 的条目 — 删除该行及 entry ID 注释
3. 标记为 [新增] 的条目 — 分配新 entry ID (格式: <!-- entry:usr5-NNN -->, 
   NNN 使用当前最大序号+1)
4. 不要修改任何 <!-- entry:... --> 注释的 ID 部分
5. 不要修改其他维度的内容
```

---

## 6. 画像注入 — 渲染 + Prompt 结构

### 6.1 画像渲染规则

PORTRAIT.md 存储引擎格式（含运维元数据），注入 LLM 前需渲染为干净认知描述。

**渲染 = 过滤 + 剥离 + 重组。纯规则，不调 LLM。**

```
PORTRAIT.md 原文 (引擎格式)
    │
    ├─ 过滤 ─→ 去掉: 待验证 / cooling / confidence=低的条目
    │
    ├─ 剥离 ─→ 去掉: evidence_count / 首次日期 / 最近日期 / 状态标记
    │
    └─ 重组 ─→ 稳定画像(8维度) → stable system message
               动态画像(4维度) → dynamic system message
```

**过滤规则：**

| 条目状态 | 注入？ | 规则 |
|---|---|---|
| 确认（高/中 confidence） | ✅ 注入 | 引擎已验证的模式 |
| 待验证 | ❌ 不注入 | 等浅巩固确认（连续3轮一致）后再注入 |
| cooling（>14天未提及） | ❌ 不注入 | 节省 token，LLM 不需要知道已冷却的兴趣 |
| 低 confidence | ❌ 不注入 | 不到 0.40 的条目不注入 |

**剥离规则：**

| 元数据字段 | 剥离？ | 理由 |
|---|---|---|
| evidence_count / 首次 / 最近 | ✅ 剥离 | 引擎运维信息 |
| hot / warm / cooling 标记 | ✅ 剥离 | 引擎内部状态 |
| confidence 标签 | ✅ 剥离 | LLM 不需要看到置信度 |
| 条目内容文本 | ❌ 保留 | 这是 LLM 要读的 |

**渲染示例：**

```
引擎格式:
  - 追求技术深度，对"系统设计"有持续兴趣  `高 · 23条证据 · 首次:2026-01-15 · 最近:2026-06-07`

渲染后:
  - 追求技术深度，对系统设计有持续兴趣
```

### 6.2 Prompt 结构

画像替代当前的 `【我对你的了解】` + `【我自己的表达习惯】` + `topic_notes` + `mirror_prediction` 独立注入。重建为两个 system message：

```
┌─ message[0] system ──────────────────────────────┐
│  [核心系统指令] (load_system_prompt, 不变)         │
│  [核心规则] (_CORE_RULES, 不变)                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 【认知画像】                                 │  │
│  │                                             │  │
│  │  用户                                       │  │
│  │    1.核心特征 (确认条目, 纯描述)              │  │
│  │    3.行为节律                                │  │
│  │    5.兴趣图谱 (hot+warm, 不含cooling)         │  │
│  │    6.情绪图谱 (确认条目)                      │  │
│  │                                             │  │
│  │  AI                                         │  │
│  │    1.核心表达特征 (确认条目)                   │  │
│  │    3.行为节律                                │  │
│  │    5.知识图谱 (活跃域)                        │  │
│  │    6.情绪/表达图谱 (确认条目)                  │  │
│  └─────────────────────────────────────────────┘  │
│  ← DeepSeek 前缀缓存, 仅浅/深巩固后变化, 命中>95%  │
└──────────────────────────────────────────────────┘

    [历史对话 — timeline_recent, 如有]

┌─ message[N+1] system ────────────────────────────┐
│  【当前状态】                                     │
│    用户 · 当前情绪: 低落 · 关注焦点: 项目方向       │
│    AI · 表达色调: 伴随用户焦虑, 近期reassurance上升 │
│    关系 · 信任度0.65 · 亲密度0.58 · 阶段: deepening │
│                                                  │
│  【执行指令】 (gate/tone/mode, 不变)               │
│                                                  │
│  【对话脉络】 (working_memory 摘要, 不变)           │
│                                                  │
│  【时间】 (now_hint, 不变)                        │
│  ← 每轮变化, 不缓存                                │
└──────────────────────────────────────────────────┘

    [检索记忆 — tool role, 画像着色后]
    [冲动 — tool role, 如有]
    [模式观察 — tool role, 如有]
    [用户消息 — user role]
```

### 6.3 维度分配表

| 维度 | 进哪个 message | 缓存 | 变化频率 | 注入条件 |
|---|---|---|---|---|
| 用户.1 核心特征 | stable system | ✅ | 深巩固24h | 确认条目 |
| 用户.3 行为节律 | stable system | ✅ | 浅巩固4h | 确认条目 |
| 用户.5 兴趣图谱 | stable system | ✅ | 浅巩固4h | hot+warm |
| 用户.6 情绪图谱 | stable system | ✅ | 浅巩固4h | 确认条目 |
| AI.1 核心表达特征 | stable system | ✅ | 深巩固24h | 确认条目 |
| AI.3 行为节律 | stable system | ✅ | 浅巩固4h | 确认条目 |
| AI.5 知识图谱 | stable system | ✅ | 浅巩固4h | 活跃域 |
| AI.6 情绪/表达图谱 | stable system | ✅ | 浅巩固4h | 确认条目 |
| **用户.2 当前状态** | **dynamic system** | ❌ | **每轮** | 确认条目 |
| **AI.2 当前状态** | **dynamic system** | ❌ | **每轮** | 确认条目 |
| **用户.4 关系快照** | **dynamic system** | ❌ | **每轮** | 确认值 |
| **AI.4 关系快照** | **dynamic system** | ❌ | **每轮** | 确认值 |

### 6.4 与旧 prompt 段的替换关系

| 旧段 (当前代码) | 来源 | 替代 |
|---|---|---|
| `【我对你的了解】` top-3 标签 | `pipeline.py` rerank_tags | 稳定画像(用户4维度) + 动态画像(用户.2) |
| `【我自己的表达习惯】` top-5 标签 | `circuit.py` list_tags | 稳定画像(AI4维度) + 动态画像(AI.2) |
| `topic_notes` 注入 | `circuit.py` dmn.get_topic_notes | 用户.5 兴趣图谱 + AI.5 知识图谱 |
| `mirror_prediction` 独立注入 | `circuit.py` BehaviorPredictor | 用户.3 行为节律 |
| `personality_notes` 参数传递 | `pipeline.py` → circuit → UtteranceSpec | 画像文件 → PortraitRenderer → prompt |

**保留不变：**

| 保留的段 | 理由 |
|---|---|
| `load_system_prompt()` (核心指令) | 不属画像 |
| `_CORE_RULES` | 不属画像 |
| working_memory 对话脉络 | 会话级上下文，不属画像 |
| now_hint (时间) | 单次注入 |
| 历史对话 timeline_recent | 不属画像 |
| 检索记忆 (tool role) | 画像着色后仍有独立价值 |
| 执行指令 (gate/tone/mode) | 引擎执行指令 |

### 6.5 Token 预算

| 段 | 内容 | 预估 tokens |
|---|---|---|
| 稳定 system message | 核心指令 + 规则 + 稳定画像(8维度) | ~1600-2100 |
| 动态 system message | 当前状态(4维度) + 执行指令 + 工作记忆 + 时间 | ~500-800 |
| **system 合计** | | **~2100-2900** |

vs 当前 system prompt ~1100 tokens。增量约 1000-1800 tokens。在 DeepSeek 32k 窗口下可接受。如 tokens 紧张，可进一步限流冷却条目（已在过滤规则中处理）。

### 6.6 缓存策略

DeepSeek 前缀缓存机制：system message 前缀不变 → 全部缓存命中。

- **message[0] stable**: 仅在浅巩固(4h)或深巩固(24h)时变化 → 缓存命中率 >95%
- **message[N+1] dynamic**: 每轮变化，不缓存。但放在独立 message 中，不破坏 message[0] 的缓存
- 画像版本号 `PORTRAIT.md` frontmatter `version` 字段每次持久化 +1。注入前比对版本号，仅变化时重新渲染

---

## 7. 检索与画像的接触点

### 7.1 核心原则

**检索干检索的事，画像干画像的事。** 画像不介入检索策略，检索不另开画像更新通道。

理由：
- 画像已在 system prompt 常驻注入，LLM 天然能做认知着色和关联判断，不需要引擎越俎代庖
- 画像有 43 个认知源在持续喂养，不需要检索单独开一个反馈通道
- Query 改写引入不可控偏差——改变语义检索方向的风险大于收益
- 结果标注冗余——LLM 拿着画像自己会关联

唯一的接触点：**精排阶段加一个画像相关性 boost 项**——不改 query、不改检索路径、不影响"搜到什么"，只微调"先看到什么"。

```
画像 ──→ system prompt (常驻注入，LLM 自己着色)
画像 ──→ 精排 boost (唯一接触点，轻量权重调整)
检索 ──→ 画像 (无直接反馈通道，画像数据源走 §3 的 43 个来源)
```

### 7.2 参与维度

不是所有 12 个维度都参与精排 boost。只有描述"当前关注"和"情绪地图"的维度：

| 维度 | 参与 boost？ | 影响方式 |
|---|---|---|
| 用户.2 当前状态 (关注焦点) | ✅ | 匹配关注焦点相关记忆 +0.1 |
| 用户.5 兴趣图谱 (hot/warm) | ✅ | 匹配 hot 话题 +0.2，warm +0.1 |
| 用户.6 情绪图谱 (负向触发) | ✅ | 匹配负向触发话题 -0.2（降权，不屏蔽） |
| AI.5 知识图谱 (积累域) | ✅ | 匹配 AI 积累域 +0.1 |
| 其余 8 个维度 | ❌ | 纯认知描述，不影响检索排序 |

### 7.3 精排 boost 公式

当前精排公式（`app/retrieval/pipeline.py` 精排阶段）：

```
score = cosine_sim × 0.5 + hit_conf × 0.25 + source_weight × 0.25
```

加入画像 boost 后：

```
score = cosine_sim × 0.5 + hit_conf × 0.2 + source_weight × 0.2 + portrait_boost × 0.1
```

`portrait_boost` 计算（区间 [-0.2, +0.3]，权重只占 10%，不会主导排序）：

```python
def compute_portrait_boost(memory_tags: list[str], portrait: dict) -> float:
    boost = 0.0
    hot = portrait.get("user_dim5_hot_topics", [])
    warm = portrait.get("user_dim5_warm_topics", [])
    focus = portrait.get("user_dim2_focus_keywords", [])
    negative = portrait.get("user_dim6_negative_triggers", [])
    ai_domains = portrait.get("ai_dim5_active_domains", [])

    for tag in memory_tags:
        if tag in negative:
            boost -= 0.2   # 降权但不屏蔽
        elif tag in hot:
            boost += 0.2
        elif tag in warm:
            boost += 0.1
        elif tag in focus:
            boost += 0.1
        elif tag in ai_domains:
            boost += 0.1

    return max(-0.2, min(0.3, boost))  # 夹紧区间
```

### 7.4 实现位置

在 `app/retrieval/pipeline.py` 的精排阶段加一个轻量 hook：

```python
# pipeline.py 精排阶段（检索完成后、top-k 截断前）
if ctx_obj.portrait:
    portrait_boost_map = {}  # tag -> boost值，预计算避免重复查表
    for candidate in candidates:
        boost = compute_portrait_boost(candidate.tags, ctx_obj.portrait)
        candidate.final_score = (
            candidate.cosine_sim * 0.5
            + candidate.hit_conf * 0.2
            + candidate.source_weight * 0.2
            + boost * 0.1
        )
```

### 7.5 不做的事

| 不做 | 原因 |
|---|---|
| Query 改写 | 画像已在 system prompt，LLM 自己能做认知联想。改写 query 改变语义检索方向，引入不可控偏差 |
| 结果标注 | LLM 拿着画像自己会判断关联，不需要引擎标注"这条记忆和你当前状态共振" |
| 双向反馈 | 画像有 43 个认知源（§3），不需要检索单独开反馈通道。检索命中的信号最终会通过记忆层（source 6-10）进入画像的浅/深巩固 |
| 检索路径权重动态调节 | 过度设计。现阶段没有数据证明检索存在"认知盲区"需要画像动态干预 |
| personality 检索路径 | 已删除（见 §10 替换分析）。画像常驻注入替代，不依赖检索召回人格标签 |

---

## 8. 迁移路径

### 8.0 前提

**没有用户，没有生产数据。** 不需要 feature flag、不需要回滚策略、不需要新旧并行。做完就切，删干净就删干净。

### 8.1 阶段总览

| 阶段 | 内容 | 工时 |
|---|---|---|
| **Phase 0a** | AI 记忆入库元数据补全 | 2-3h |
| **Phase 0b** | AI 完整 ConsolidationEngine 替代薄 worker | 4-6h |
| **Phase 1** | PORTRAIT.md 基础设施 + 实时层更新 | 6-8h |
| **Phase 2** | 画像注入 prompt + 删旧注入代码 | 2-3h |
| **Phase 3** | 浅巩固 + 深巩固画像更新 | 8-12h |
| **Phase 4** | 精排 boost + 删旧检索/蒸馏代码 | 2-3h |
| **总计** | | **22-32h** |

### 8.2 逐 Phase 详解

---

#### Phase 0a — AI 记忆入库元数据补全

当前 AI 记忆入库缺少 `time_features`、`date_tag`、`entities`、`session_continued`。补到和用户记忆一致。

- 改 `app/core/context.py` AI 记忆入库函数
- 单测验证字段存在

---

#### Phase 0b — AI 完整 ConsolidationEngine

当前 AI 只有一个薄 worker（情绪淡化+基础日志）。创建完全镜像的 ConsolidationEngine 实例。

- 改 `app/core/context.py`：创建 AI 侧 ConsolidationEngine，独立 state_path/notes_path
- 停掉旧薄 worker
- 验证：AI 侧产出话题笔记/模式发现/冲突检测

---

#### Phase 1 — PORTRAIT.md 基础设施 + 实时层更新

**新建 `app/portrait/` 包：**

| 文件 | 职责 |
|---|---|
| `app/portrait/state.py` | 条目状态机（pending → active → cooling → decayed） |
| `app/portrait/manager.py` | PortraitManager：加载/保存/解析 PORTRAIT.md，entry ID 索引 |
| `app/portrait/renderer.py` | PortraitRenderer：过滤 + 剥离 + 重组 → stable/dynamic message |
| `app/portrait/writer.py` | PortraitWriter：realtime_update()（规则驱动，不调 LLM） |
| `app/portrait/extractors.py` | 从 distill.py 迁出的纯函数 |

**实时层更新（在 Phase 1 就做好，不拆出去）：**

每轮对话后更新：
- 用户.2 当前状态 ← UserMessageAnalysis
- 用户.4 关系快照 ← RelationshipState
- AI.2 当前状态（镜像）
- AI.4 关系快照（镜像）

hook 在 `app/api/chat.py` 存储管线完成后调用。

**验证：**
- PORTRAIT.md 生成并正确解析
- Renderer 输出符合 §6 的 stable/dynamic 格式
- 5 轮对话后 dim 2/4 正确反映最新状态
- 用户和 AI 两侧独立

---

#### Phase 2 — 画像注入 prompt + 删旧注入代码

**这一步做完，画像就上线了。**

改两个文件：

1. `app/core/circuit.py`：
   - 删 L332-339 `personality_notes` 填充 → 替换为 `portrait_renderer.render()`
   - 删 L341-352 `ai_result = self._personality.list_tags(...)` → 同上，AI 画像
   - 去 `personality_store` 参数

2. `app/llm/deepseek.py`：
   - 删 L703-719 「我对你的了解」段 → `portrait_stable`
   - 删 L722-732 「我自己的表达习惯」段 → `portrait_stable` AI 维度

3. `app/retrieval/pipeline.py`：
   - 删 L197-206 personality 检索路径

**验证：**
- 对话正常运行，画像内容出现在 system prompt 中
- benchmark 分数不劣化（对比 Phase 1 的画像内容 vs 旧 personality_notes）
- 无 import 报错（personality_store 引用已清除）

---

#### Phase 3 — 浅巩固 + 深巩固画像更新

浅巩固（idle ≥ 4h）：LLM 合成 dim 3/5/6
深巩固（idle ≥ 24h）：LLM 重写 dim 1 + 一致性审查

- 在 `app/portrait/writer.py` 加 `shallow_update()` / `deep_update()`
- 在 `app/background/consolidation.py` 的 `on_idle()` 挂载
- 遵循 §5.8 四态操作规则

**验证：**
- 模拟 idle 触发，画像维度正确更新
- entry ID 保留（LLM 不丢 ID）
- 用户 + AI 两侧独立更新

---

#### Phase 4 — 精排 boost + 清理残余

1. 检索精排加画像 boost（§7.3，一行公式改动）
2. 删残留旧代码：

| 删除 | 文件 |
|---|---|
| PersonalityStore + BehaviorStore + 包 | `app/personality/` |
| DistillEngine | `app/background/distill.py`（纯函数已迁 extractors） |
| AI 薄 worker | `app/core/context.py` L755-796 |
| 旧 API | `app/api/personalities.py`, `app/api/distill.py` |
| 旧配置 | `PERSONALITY_CHROMA_DIR` 等 |
| 所有残留 import/参数 | `context.py`, `circuit.py`, `pipeline.py`, `chat.py`, `consolidation.py`, `system.py` |

**验证：**
- 全部现有测试通过
- 手动对话验证完整闭环：消息 → 画像注入 → LLM → 实时更新 → 浅/深巩固 → 画像更新

---

### 8.3 不做的事

| 不做 | 原因 |
|---|---|
| Feature flag（`PORTRAIT_MODE`） | 零用户，没必要保留旧路径 |
| PersonalityStore 数据备份 | 无真实用户数据 |
| 新旧系统并行运行 | 零用户，直接切 |
| 稳定观察期 | 没有用户需要保护 |
| 回滚策略 | 直接 git revert 就行 |

---

## 9. 待决议题

### 待决

---

#### 9.1 画像文件大小上限

画像持续累积，文件可能膨胀。需要截断策略。

**原提案**: 每维度 15 条。**问题**：太粗——核心特征可能 5-8 条就够，兴趣图谱 20 条也正常。

**新提案**: 逐维度设上限，与深巩固的 LLM 合成能力匹配：

| 维度 | 上限 | 理由 |
|---|---|---|
| 用户.1 核心特征 | 10 | 人的核心特质有限，多了就是 noise |
| 用户.2 当前状态 | 8 | 实时更新，高频替换，自然截断 |
| 用户.3 行为节律 | 15 | 多个时间切片（每天/每周/每季节） |
| 用户.4 关系快照 | 6 | 关系维度有限 |
| 用户.5 兴趣图谱 | 20 | 兴趣可以很多，但 hot 限制 5 |
| 用户.6 情绪图谱 | 15 | 触发话题/情绪模式可多 |
| AI.1-6 | 同上 | 完全镜像 |

超限策略：优先合并低 confidence 同义条目（LLM 判断），其次删除 cooling 条目。保留 active 高 confidence 条目。

---

#### 9.2 画像遗忘：时间衰减驱动状态机

**原提案**: 独立的 confidence 衰减规则（30d→中, 60d→低, 90d→移除），与 ChromaDB 情绪淡化平行。

**问题**: 这就跟 §5.8 的四态状态机（active/cooling/decayed）搞成了两套独立规则。

**新提案**: 时间衰减**驱动状态转换**，不额外维护 confidence 字段：

```
last_observed > 30d 且 evidence_count < 3 → active → cooling
last_observed > 60d → cooling → decayed (下次渲染时 strip)
last_observed > 90d → decayed → 物理删除 (下次写入时清理)
```

ChromaDB 层的情绪淡化机制不受影响——它淡化的是记忆元数据的 `emotional_intensity`，跟画像条目是不同层的东西。

---

#### 9.3 冷启动：初始画像从哪来

零用户、零旧 PersonalityStore 数据 → 画像从空文件开始。dim 1/3/5/6 要到第一次浅/深巩固才有内容。对话量少时，画像可能长期"半空"。

**选项 A**: 接受空启动。画像随对话自然生长，前几十轮 dim 1（核心特征）空着就空着。

**选项 B**: Phase 1 完成后，跑一次**全量 LLM 扫描**——把所有现有 ChromaDB 记忆（可能几千条）的 tags/summary 分布喂给 LLM，直接生成初始 dim 1/3/5/6。一次性成本。

**选项 C**: 不给 LLM。直接用规则从记忆元数据提取初始画像——tags 频率 → 兴趣、时间分布 → 节律、情绪分布 → 情绪图谱。零 token 成本。

建议先走 **C（规则提取）**，不够再补 B（LLM 扫描）。规则提取的结果作为 LLM 深巩固的种子。

---

#### 9.4 深巩固最小数据门槛

深巩固每 24h 触发一次 LLM 合成 dim 1 + 一致性审查。但可能只有 5 轮对话。

**提案**: 深巩固触发条件加最小数据量门槛：

```python
if idle >= 24h and new_turns_since_last_deep >= 20:
    portrait_writer.deep_update()
```

20 轮是从零合成"用户是什么样的人"的最低下限。低于 20 轮跳过，等下次。

浅巩固不需要门槛——dim 3/5/6 是增量更新，几轮也能更新。

---

#### 9.5 PORTRAIT.md 原子写

每轮对话后实时写入 PORTRAIT.md。如果服务器在写入时崩溃，文件可能损坏（半截 JSON/Markdown）。

**提案**: 写临时文件 → rename 覆盖：

```python
def save_portrait(content: str, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)  # atomic on POSIX & Windows
```

`os.replace()` 在 Windows 和 Linux 上都是原子操作。写入失败不会损坏原文件。

---

### 已解决

| 议题 | 决议 |
|---|---|
| ~~9.3 并发写入~~ | `threading.Lock` 保护 PORTRAIT.md 读写。文件小（<50KB），写入 <1ms，锁竞争可忽略 |
| ~~9.4 画像与工作记忆的关系~~ | 不合并。working_memory 是会话级摘要，画像是认知级持久化，作用不同 |
| ~~9.5 多用户支持~~ | 硬编码 `data/portrait.md`，路径工具预留 `user_id` 参数。零用户不需要多租户 |

---

## 10. 画像替换分析 — 谁被替代、谁被修改、谁保留

画像系统不是在现有代码上打补丁，而是把认知结论的**产生→存储→检索→注入→更新**整个闭环从多个分散模块中抽出来，统一到 PORTRAIT.md + PortraitWriter 里。这意味着现有系统中有相当多的代码路径会退役。

### 10.1 认知架构变迁（Before / After）

#### Before（现状）：认知碎片散落各处

```
用户消息
  ├─ circuit.analyze_user_message() → intent/emotion/topics/urgency/emotion_intensity
  ├─ pipeline (10路并行检索，含 personality rerank)
  │    ├─ path 1-9: 记忆检索
  │    └─ path 10: personality_store.rerank_tags(query) → 语义搜索 top-15 → 重排 → top-5
  ├─ circuit.weave_context()
  │    ├─ personality_notes[] ← personality rerank 结果过滤 (source=user)
  │    ├─ personality_notes_ai[] ← personality_store.list_tags(page=1, page_size=5, source=ai)
  │    └─ 记忆分层 (fact/reference/background/suppressed)
  ├─ deepseek._build_stable_system_prompt()
  │    └─ 拼入「我对你的了解」+「我自己的表达习惯」
  ├─ LLM 生成回复
  └─ [后台] DistillEngine.run_distill() → 算法提取 patterns → 写入 PersonalityStore
       [后台] AI 巩固薄 worker (1h循环) → 情绪淡化 + 基础统计
```

**7 块碎片化管理认知**：每次靠语义搜索重新召回人格标签、蒸馏产物是孤立 tags 不形成完整画像。

#### After（画像后）：认知合一

```
用户消息
  ├─ circuit.analyze_user_message() → intent/emotion/topics/urgency/emotion_intensity  [保留]
  ├─ PortraitRenderer.render() → 过滤+剥离+重组 PORTRAIT.md → 8dim stable + 4dim dynamic
  ├─ pipeline (10路并行检索，不再含 personality path)  [保留9路，去掉personality路]
  ├─ circuit.weave_context()  [保留，但不再拼 personality_notes]
  │    └─ 记忆分层 (fact/reference/background/suppressed)
  ├─ deepseek._build_stable_system_prompt()
  │    └─ 拼入 rendered_portrait 替代「我对你的了解」+「我自己的表达习惯」
  ├─ LLM 生成回复
  ├─ [实时] portrait.realtime_update(user+ai)  [新增]
  └─ [后台] on_idle → 浅巩固(4h) → portrait.shallow_update(user+ai)
              → 深巩固(24h) → portrait.deep_update(user+ai)
```

---

### 10.2 逐文件替换清单

#### 🔴 完全废弃（可删除）

| 文件 | 行数 | 理由 |
|---|---|---|
| `app/personality/store.py` | ~277 | 独立 ChromaDB collection 存储孤立人格 tags。画像改为单一 PORTRAIT.md 文件，不再需要独立向量库做语义搜索召回 |
| `app/personality/behavior.py` | ~200 | BehaviorStore 独立 collection 存储行为模式。行为模式数据源合并进画像维度 用户.4（行为模式），不再独立存储 |
| `app/personality/__init__.py` | — | 包本身废弃 |
| `app/api/personalities.py` | ~40 | GET/DELETE 人格标签的 REST API，画像改为文件操作+渲染注入 |
| `app/api/distill.py` | ~29 | DistillEngine 手动触发 API，浅/深巩固替代蒸馏 |

#### 🟡 部分废弃（保留文件，删除/重写特定代码路径）

**`app/core/circuit.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L116 | `personality_notes: list` 字段定义 | 替换为 `portrait: dict[str, Any]` (渲染后的画像) |
| L212-215 | `__init__(chroma_service, personality_store, ...)` | 去掉 `personality_store` 参数 |
| L332-339 | `for p in personalities: temp.personality_notes.append(...)` | 删除。画像不走检索管线召回 |
| L341-352 | `ai_result = self._personality.list_tags(page=1, page_size=5)` → AI 人格 | 删除。AI 画像也从 PORTRAIT.md 注入 |
| L352 | `temp.personality_notes_ai = ai_notes` | 删除 |
| L464-465 | `personality_notes=..., personality_notes_ai=...` 传给 UtteranceSpec | 替换为 `portrait_stable=..., portrait_dynamic=...` |

**`app/llm/deepseek.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L703-719 | 「我对你的了解」段 — 遍历 `personality_notes[]` | 替换为画像 stable message（8 维度，来自渲染器） |
| L722-732 | 「我自己的表达习惯」段 — 遍历 `personality_notes_ai[]` | 替换为画像 stable message 的 AI 维度部分 |
| L734+ | system prompt 拼接逻辑 | 新增 portrait 注入点，调整缓存边界 |

**`app/retrieval/pipeline.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L197-206 | `personality_store.rerank_tags(...)` — 检索管线中的 personality 路径 | 删除。画像常驻注入，不再依赖检索召回人格标签 |
| L198 | `ctx_obj.personality_store` 字段依赖 | 去除，`RetrievalContext` 不再需要 `personality_store` |

**`app/core/context.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L41-42 | `from app.personality.store import PersonalityStore` + `BehaviorStore` | 删除 import |
| L101 | `self.personality_store = PersonalityStore(...)` | 删除，替换为 `self.portrait = PortraitManager(...)` |
| L123 | `UserCircuitBreaker(personality_store, chroma_service, ...)` | 去掉 `personality_store` 参数 |
| L129 | `AICircuitBreaker(personality_store, ai_chroma_service, ...)` | 去掉 `personality_store` 参数 |
| L138 | `RetrievalContext` 传 `personality_store=self.personality_store` | 删除 |
| L279 | `RetrievalContext` 传 `personality_store=self.personality_store` | 删除 |
| L677-726 | `_run_user_distill()` — DistillEngine 运行逻辑 | 替换为 PortraitWriter 的分层更新调用 |
| L755-796 | AI 巩固薄 worker（情绪淡化 + 基础统计） | 废弃，被 ConsolidationEngine(AI) 完全镜像替代 |
| L128-131 | AI DistillEngine 初始化 | 废弃，AI 画像更新走统一 PortraitWriter |

**`app/background/distill.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L136-347 | `_extract_patterns()` — 6 种模式提取算法 | 算法逻辑提取为纯函数，迁移到 `app/portrait/extractors.py` |
| L350-368 | `DistillEngine` 类 | 整个类废弃，被 `PortraitWriter` 替代 |
| L378+ | `run_distill()` — 蒸馏主循环 | 废弃，浅/深巩固触发画像更新 |
| L353-368 | `_cleanup_junk_patterns()` | 废弃，画像更新使用四态操作（修改代替清理） |

保留并迁移的纯函数：
- `_extract_keywords(text, topk)` → `app/portrait/extractors.py`
- `_recency_score(ts, now)` → `app/portrait/extractors.py`
- `_compute_confidence(...)` → `app/portrait/extractors.py`

**`app/background/consolidation.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L87-93 | `__init__` 接收 `personality_store`, `behavior_store` | 去掉这两个参数，不再需要 |
| L93 | `self._personality = personality_store` | 删除 |
| 新增 | — | `on_idle()` 在 4h/24h 触发时调用 `portrait.shallow_update()` / `portrait.deep_update()` |

**`app/api/chat.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L149, L270, L388 | `UserCircuitBreaker(chroma, personality_store, ...)` | 去掉 `personality_store` 参数 |
| 新增 | — | 实时画像更新调用（对话存储管线完成后） |

**`app/api/system.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| L223 | `p_tags = ctx.personality_store.list_tags(page=1, page_size=200)` | 替换为 `ctx.portrait.render(full=True)` 或直接读 PORTRAIT.md |
| L229-234 | `snapshot["personality"]` 返回格式 | 改为返回画像摘要 |

**`app/config/settings.py`**

删除的配置项：
- `PERSONALITY_CHROMA_DIR`
- `PERSONALITY_COLLECTION`
- `PERSONALITY_DEDUP_THRESHOLD`

新增的配置项：
- `PORTRAIT_FILE_PATH` — PORTRAIT.md 存储路径
- `PORTRAIT_SHALLOW_CONSOLIDATION_HOURS` (默认 4)
- `PORTRAIT_DEEP_CONSOLIDATION_HOURS` (默认 24)
- `PORTRAIT_REALTIME_DIMENSIONS` — 实时更新维度白名单（默认 2,4）

**`app/background/impulse.py`**

| 行号 | 现有代码 | 变更 |
|---|---|---|
| 新增 | — | 冲动源可读取画像维度作为种子 — 画像成为冲动系统的一个认知源 |

---

### 10.3 API 废弃对照

| 废弃 API | 方法 | 替代 |
|---|---|---|
| `GET /api/personalities` | 人格标签列表 | `GET /api/portrait` (直接返回渲染后的画像) |
| `GET /api/personalities/{tag_id}` | 单标签详情 | 画像维度查询 |
| `DELETE /api/personalities/{tag_id}` | 删除标签 | 画像条目修改（四态操作） |
| `POST /api/distill` | 手动触发蒸馏 | `POST /api/portrait/consolidate` (触发浅/深巩固) |
| `GET /api/distill/status` | 蒸馏状态 | 画像状态查询 |

---

### 10.4 新增组件

| 组件 | 位置 | 职责 |
|---|---|---|
| `PortraitManager` | `app/portrait/manager.py` | 画像生命周期管理（加载/保存/渲染/状态机） |
| `PortraitRenderer` | `app/portrait/renderer.py` | 过滤+剥离+重组，输出 stable/dynamic message |
| `PortraitWriter` | `app/portrait/writer.py` | 三层更新调度（实时/浅/深），四态操作执行 |
| `PortraitExtractors` | `app/portrait/extractors.py` | 从 distill.py 迁移的纯函数 + 新 extractor |
| `PortraitState` | `app/portrait/state.py` | 画像条目状态机（pending/cooling/active/decayed） |
| PORTRAIT.md | `data/PORTRAIT.md` | 画像文件本体 |

---

### 10.5 保留不变的组件

| 组件 | 原因 |
|---|---|
| `circuit.analyze_user_message()` | 实时意图/情绪分析仍需要，画像不取代这些 |
| `circuit.weave_context()` | 记忆分层逻辑不变，只是不再拼 personality_notes |
| `pipeline` 的 9 路检索（除 personality 外） | 记忆检索不受影响 |
| `ConsolidationEngine._detect_fact_contradictions()` | 事实冲突检测独立于画像 |
| `ConsolidationEngine._assess_archival()` | 存档评分独立于画像 |
| `ImpulseScheduler` + 5 个冲动源 | 冲动系统独立，画像可作为新认知源加入 |
| `TemporalPatternIndex` | 时间模式索引被画像读取（§3 认知源），但不被替代 |
| `TopicAffinity` / `TopicTree` | 话题结构被画像读取，但不被替代 |
| `CooccurrenceTracker` | 共现追踪被画像读取，但不被替代 |
| `PersonaSymmetry` | 对称分析被画像读取（AI.6），但不被替代 |
| `BehaviorPredictor` | 行为预测被画像读取（AI.3），但不被替代 |
| `ChromaService` (主记忆库) | 记忆存储/检索完全不受影响 |
| AI ChromaService (独立记忆库) | AI 记忆存储完全不受影响（补全后反而更强） |
| `chat_history` / `co_tracker` | 对话历史追踪不受影响 |

---

### 10.6 改动量估算

| 类别 | 文件数 | 改动量 |
|---|---|---|
| 完全删除 | 5 | `personality/store.py`, `personality/behavior.py`, `personality/__init__.py`, `api/personalities.py`, `api/distill.py` |
| 部分重写 | 3 | `background/distill.py` (保留纯函数), `core/circuit.py` (删除 personality 注入), `llm/deepseek.py` (替换 prompt 段) |
| 参数/import 调整 | 4 | `core/context.py`, `retrieval/pipeline.py`, `api/chat.py`, `api/system.py` |
| 新增依赖注入 | 2 | `background/consolidation.py` (新增 portrait 调用), `background/impulse.py` (新增画像认知源) |
| 配置清理 | 1 | `config/settings.py` |
| 新建 | 6+ | `app/portrait/` 包 (manager, renderer, writer, extractors, state) + `data/PORTRAIT.md` |
| 测试清理 | 若干 | `test_impulse.py` 中涉及 personality_store 的 mock 需调整；`test_contradiction.py` 不受影响；需新建 `test_portrait.py` |

---


### 10.7 测试迁移清单

#### 🔴 全废（文件直接删除）

| 文件 | 原因 |
|---|---|
| `tests/test_personality.py` | PersonalityStore 已删，测试无意义 |
| `tests/test_personality_deep.py` | 同上 |
| `tests/test_behavior.py` | BehaviorStore 已删，测试无意义 |

#### 🟡 需要重写逻辑

| 文件 | 改动 |
|---|---|
| `tests/test_distill.py` | `TestDistillEngineRun` 类全废；`_extract_keywords` / `_recency_score` 等纯函数测试保留，迁到 extractors 测试 |
| `tests/test_impulse.py` | `TestSourceBehaviorPattern` 类重写——`source_behavior_pattern` 不再收 `personality_store`，改为读画像 dim 4 |

#### 🟢 机械替换（去参数/去 mock）

| 文件 | 改动量 | 具体 |
|---|---|---|
| `tests/test_consolidation.py` | ~24 处 | 所有 `personality_store=MagicMock(), behavior_store=MagicMock()` 删掉 |
| `tests/test_context.py` | ~5 处 | 去掉 `PersonalityStore` / `BehaviorStore` / `DistillEngine` 的 patch mock |
| `tests/test_api_chat_core.py` | 1 处 | L50-51 去掉 `ctx.personality_store = MagicMock()` |
| `tests/test_api_routes.py` | ~5 处 | 去掉 `personality_store` / `distill_engine` mock；删 `/api/personalities` / `/api/distill` 路由测试 |
| `tests/test_pipeline.py` | ~2 处 | L195-196 去掉 `ctx.personality_store` mock |
| `tests/test_retrieval_pipeline.py` | 1 处 | L141 `from app.background.distill import _recency_score` → 改为 `from app.portrait.extractors` |

#### 🟢 E2E / 集成测试（机械替换）

| 文件 | 改动量 | 具体 |
|---|---|---|
| `E2E/test_link2_retrieve.py` | ~15 处 | 去掉 `CircuitBreaker(... personality_store=...)` 参数 |
| `E2E/test_link3_cross_turn.py` | 1 处 | L64 去掉 `personality_store` 参数 |
| `E2E/test_link4_evolution.py` | 2 处 | L404/415 去掉 `personality_store.list_tags`；L410 去掉 `distill.run_distill` |
| `E2E/test_background.py` | 2 处 | L187 去掉 `personality_store.store_tag`；L199 去掉 `personality_store` 参数 |
| `integration/test_int_consolidation.py` | 1 处 | L69-70 去掉 `personality_store=` / `behavior_store=` 参数 |

#### 🆕 新增

| 文件 | 内容 |
|---|---|
| `tests/test_portrait.py` | PortraitManager / PortraitRenderer / PortraitWriter 单元测试 |

#### 分批执行建议

| 阶段 | 测试改动 |
|---|---|
| Phase 0a-0b | 不改测试（纯补字段+新线程） |
| Phase 1 | 新建 `test_portrait.py` |
| Phase 2 | 机械替换开始——去参数、去 mock |
| Phase 3 | 删 `test_pipeline.py` `test_api_chat_core.py` `test_api_routes.py` 中 personality 相关 mock |
| Phase 4-5 | 重写 `test_distill.py` + `test_impulse.py` 逻辑 |
| Phase 6 | E2E / 集成测试去参数 |
| Phase 7 | 删 `test_personality.py` / `test_personality_deep.py` / `test_behavior.py`；确认全部通过 |

## 附录 A: 现有组件废弃清单（更新）

迁移完成后以下组件进入废弃：

| 组件 | 文件:行 | 替代 | 废弃程度 |
|---|---|---|---|
| `PersonalityStore` | `app/personality/store.py` | PORTRAIT.md + PortraitManager | 🔴 全文件 |
| `BehaviorStore` | `app/personality/behavior.py` | 画像维度 用户.4 行为模式 | 🔴 全文件 |
| `DistillEngine` (user) | `app/background/distill.py:L350-378` | PortraitWriter 浅巩固+深巩固 | 🔴 全类 |
| `DistillEngine` (ai) | `app/core/context.py:L128-131` | 同上，AI 侧镜像更新 | 🔴 |
| AI 巩固薄 worker | `app/core/context.py:L755-796` | ConsolidationEngine(AI) 完全镜像 | 🔴 |
| `_cleanup_junk_patterns` | `app/background/distill.py:L353-368` | 四态操作修改代替清理 | 🔴 |
| `_extract_patterns` | `app/background/distill.py:L136-347` | 逻辑迁移到 `app/portrait/extractors.py` | 🟡 迁移 |
| `personality` 检索路径 | `app/retrieval/pipeline.py:L197-206` | 画像常驻注入，不走检索 | 🔴 |
| `personality_notes[]` 注入 | `app/core/circuit.py:L332-339` | portrait stable message | 🔴 |
| `personality_notes_ai[]` 注入 | `app/core/circuit.py:L341-352` | portrait stable message (AI 维度) | 🔴 |
| 「我对你的了解」段 | `app/llm/deepseek.py:L703-719` | 画像 stable message | 🔴 |
| 「我自己的表达习惯」段 | `app/llm/deepseek.py:L722-732` | 画像 stable message (AI 维度) | 🔴 |
| `GET /api/personalities` | `app/api/personalities.py:L11-23` | `GET /api/portrait` | 🔴 |
| `POST /api/distill` | `app/api/distill.py:L20-28` | `POST /api/portrait/consolidate` | 🔴 |
| `PersonalityStore` 参数传透 | `context.py`, `circuit.py`, `pipeline.py`, `chat.py`, `consolidation.py` | 改为 `portrait` 依赖 | 🟡 去参数 |

保留并迁移：
- `_extract_keywords` / `_recency_score` / `_compute_confidence` 等纯函数 → `app/portrait/extractors.py`

---

## 附录 B: 与现有后台节律的挂载关系（不变）

```
现有:  consolidation.on_idle()  [仅用户侧]
          ├─ idle ≥ 2min → 冲动消费者
          ├─ idle ≥ 4h  → 浅巩固 (用户侧: _consolidate_day / _review_today)
          └─ idle ≥ 24h → 深巩固 (用户侧: _detect_fact_contradictions / _assess_archival)

        AI 巩固薄 worker (独立线程, 1h循环)
          ├─ 情绪淡化
          └─ 基础统计日志

改后:  on_idle() 触发双巩固引擎
          ├─ idle ≥ 2min → 冲动消费者
          ├─ idle ≥ 4h  → 用户浅巩固 + 画像浅层更新 (用户)
          │              → AI 浅巩固 + 画像浅层更新 (AI, 完全镜像)
          └─ idle ≥ 24h → 用户深巩固 + 画像深层更新 (用户)
                         → AI 深巩固 + 画像深层更新 (AI, 完全镜像)

        AI 巩固薄 worker → 废弃，被 ConsolidationEngine(AI) 替代
```

用户和 AI 的 ConsolidationEngine 共享同一个 on_idle() 触发，顺序执行（用户先、AI后），各自独立的 state_path 和 notes_path。

每轮对话后 (chat.py → 存储管线完成后):
          ├─ portrait.realtime_update(user)  (新增, 轻量, <100ms)
          └─ portrait.realtime_update(ai)    (新增, AI侧镜像, 轻量, <100ms)
```

现有节律框架不需要改，画像更新作为新的挂载点插入。
