# SPEC：残差注入引擎 — 16 模块独立产出残差，精准打入指定层

> **状态**：核心假设已验证（2026-06-20），**引擎落地完成（2026-06-21）** — `app/llm/steering.py` 投产，chat.py 双模式分支，994 tests green。方案从"第 0 层统一注入"收敛为"每模块独立残差 → 分层注入"。
> **推导历程**：从"给实例加手"→ 全架构改造 → 并行干爆设备 → 混动杀实时 → MCP 碾平认知层 → logit bias 浅层 → activation steering 空间对齐 → 第 0 层 embedding 注入 → **残差分层注入（当前方案）**。
> **核心洞见**：Transformer 残差流从第 0 层到最后一层始终是同一个 ℝ^d_model 空间——没有任何变形、投影、门控。任何模块产出的向量，只要在这个空间里，`h += r` 就永远合法。公式是唯一的真理，变化的只是流经公式的数据。

---

## 1. 这是什么

**引擎 16 个模块各自产出短中文 → tokenize → embedding 表 → 残差向量 → 在 Transformer 指定层做加法：**

```
h_L = h_L + Attention(Norm(h_L)) + r_module × α
```

**两步走：**
- **第一步（最小可用）**：所有引擎产出 → 第 0 层 embedding 拼进输入序列（跨模型零改动，已文本级验证）
- **第二步（最终形态）**：每个模块 → 独立残差向量 → 各自注入指定层（qwen2.5 标准残差，本文档主目标）

跟当前做法的核心区别：当前引擎产出的文本拼进 prompt（LLM 可选看或不看），改后残差向量直接加在 hidden state 上（物理不可跳过，跟 Transformer 自己的残差是同一个动作）。

---

## 2. 为什么是这个方案

### 之前所有方向为什么撞墙

| 方向 | 撞墙原因 |
|------|---------|
| 给手（Agent 框架） | 全架构改造，认知层被 Action 感染，需重新发明 haras engineering |
| 并行架构 | 设备内存/CPU 干爆 |
| 混动模式 | LLM 延迟杀实时管线（TTFT 200-500ms × N 个决策点） |
| MCP 接入 | 认知分层被碾平，画像/门控/关系全废 |
| 双轨入库 | if-else 分类做不到——每句话同时是关系和任务 |
| logit bias | 0 层深度，只做代数加法，不过 attention |
| activation steering | 空间不对齐（embedding space ≠ hidden space），需改 C++ |

### 这个方案为什么不同

1. **不改引擎逻辑**。引擎 16 个模块照常产出，只是出口从"拼 prompt"换成"产残差向量"。
2. **不改模型架构**。不做层间注入，不碰 attention/FFN 内部——只利用 Transformer 自带的残差连接。
3. **不需要训练**。模型自己的 tokenizer + embedding 表就是最好的编码器——短中文→tokenize→embed 这条路径零训练。
4. **语义空间天然对齐**。引擎文本用本地模型自己的 embedding 产出向量→就在模型的语义空间里。
5. **不可跳过**。prompt 是"建议"——模型可以忽视。残差 `h += r` 是输入 tensor 的一部分——attention 物理上必须处理。
6. **残差加法永远合法**。Transformer 残差流从第 0 层到最后一层始终是同一个 ℝ^d_model 空间，没有变形/投影/门控。任何向量加进去，LayerNorm 自然兜底，模型训练时已通过 dropout 学会容忍外来信号。
7. **公式一样，效果千变万化**。每个模块产残差的公式一模一样（tokenize→embed→加法），但注入不同层效果完全不同——浅层(L1-L10)流动语义特征，深层(L21-L32)流动逻辑特征。同一句"用户是资深程序员"，在 L5 被当语义属性消费，在 L25 被当决策约束消费。
8. **Agent 框架不需要内置**。本地 LLM 自带 tool call + 规划能力，引擎只需提供认知上下文，LLM 自己就是 Agent 大脑。
9. **单轨入库零改动**。Agent 中间产物不入 Qdrant（留在对话记录 JSONL），只入用户消息 + AI 最终回复。老管线完全不变。

---

## 3. 架构全景

```
用户: "我最近想换Python了，Rust太难了"
  │
  ▼
┌─── 引擎 (完全不变) ──────────────────────────────────────┐
│                                                          │
│  run_chat_retrieval()      → ~50条候选记忆 (9路并行)       │
│  analyze_user_message()    → intent + emotion + topics    │
│  mirror_prediction         → 行为预测                      │
│  portrait_boost_map        → 画像热点 tag boost            │
│  weave_context()           → fact/reference/stale 分组     │
│  CognitiveState 组装        → 置信度评分 + mood + DMN     │
│  basal_ganglia_gate()      → tone/formality/response_mode │
│  drift_tracker             → 偏移率                       │
│  self_mirror               → AI 自我镜像                  │
│  RelationshipState         → trust/closeness/familiarity  │
│  portrait_renderer         → portrait_stable/dynamic      │
│  impulse                   → 冲动信号                     │
│                                                          │
│  ↓ 引擎产出 (每个模块独立，各自短中文)                        │
│                                                          │
│  portrait_identity:  "用户是资深程序员，偏好Rust和Python"    │
│  portrait_emotion:   "用户近期工作压力大，情绪脆弱"          │
│  portrait_interest:  "用户关注架构设计、技术栈选型"          │
│  relation:           "信任度0.7，collaborator关系模式"     │
│  gate_tone:          "语气caring，不要直接，不要过于轻快"    │
│  relevant_memory_1:  "06/15用户说Rust太难了想放弃"         │
│  relevant_memory_2:  "05/28用户说Python写起来确实舒服"      │
│  relevant_memory_3:  "05/12架构重构搞了一周终于差不多了"     │
│  narrative:          "06/15→今日 Python/Rust/编程 5次提及" │
│  impulse:            "主动话题: 技术栈选型讨论"             │
│                                                          │
│  每个模块产出 1-2 句短中文 (10-30 token)                    │
│  总计 ~100-200 token (vs 现在 ~800-2000 token prompt)     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌─── 本地 Ollama ───────────────────────────────────────────┐
│                                                          │
│  qwen2.5 tokenizer + embedding 表:                        │
│    用户消息 → [token_id] → [vec]（原始素材）                │
│    引擎短句 → [token_id] → [vec]（加工指令）                │
│                                                          │
│  Layer 1 输入: 仅用户消息 token embeddings                  │
│  Layer 1 处理: attention → 语义表示                        │
│                                                          │
│  Layer 2 输入: 语义表示 + 引擎残差向量 ─── 加工指令注入      │
│    全部在 qwen 自己的语义空间，残差 h += r                   │
│                                                          │
│  Layer 2 → ... → Layer N → 生成                           │
│                                                          │
│  输出: "我能感觉到你最近一直在纠结这件事。Rust确实..."       │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
              用户看到回复

(入库管线、后台线程全部不变)
```

### Agent 循环（由外部 Agent 框架负责，不嵌入引擎）

> **决策（2026-06-21）**：工具调用不进引擎。引擎只管认知注入（残差向量→模型残差流）。工具调度/权限/错误恢复全在外部 Agent 框架（如 Claude Code）。引擎给 LLM 戴上"认知眼镜"，外部框架给 LLM 装上"手"。

```
用户: "帮我看看那个bug"
  │
  ▼
引擎 (不变) → 认知上下文 embedding 注入
  │
  ▼
本地 qwen2.5 ──┬── 外部 Agent 框架 (工具调度)
  │              ├─ read_file("auth.py")
  │              ├─ grep("空指针", "auth.py")
  │              ├─ write_file("auth.py", fix)
  │              └─ bash("pytest tests/")
  │              ↑ 中间调用留在外部框架 transcript，不入 Qdrant
  │
  ▼
最终回复: "找到了，auth.py第42行空指针，修好了，测试通过"
  │
  ▼
Qdrant 入库 (老管线，零改动)
  只入: (user_msg, ai_final_response) 这一对
```


---

## 4. 跟当前做法的对比

| | 当前 (API prompt) | 第一步 (第 0 层 embedding) | 最终形态 (残差分层注入) |
|---|------------------|--------------------------|----------------------|
| LLM | DeepSeek API (HTTP) | 本地 Ollama qwen2.5 | 本地 qwen2.5 / 更大模型 |
| 引擎产出形式 | 拼进 10 段消息结构 | 各模块独立短句 | 16 模块独立短句 → 残差向量 |
| 注入点 | message[0], [N+1] 等 | 第 0 层 embedding | `h_l += r` 指定层 |
| 模型是否可跳过 | 是 | 否 (输入序列的一部分) | **否（残差加法，物理不可跳过）** |
| 控制粒度 | 全消息共享 | 全层共享，attention 自然分配 | **每模块指定层号** |
| Token 限制 | API 上下文窗口 | 本地模型限制 | 本地模型限制 |
| 延迟 | HTTP 往返 200-500ms | 本地推理 | 本地推理 |
| 引擎 token 消耗 | ~800-2000 token | ~100-200 token | ~100-200 token |

---

## 5. 需要做的事

### 5.1 引擎侧改造

**新建模块** `app/llm/steering.py`：

```python
def build_steering_segments(utterance_spec: UtteranceSpec) -> list[str]:
    """将 UtteranceSpec 分解为各模块独立短句列表。
    每个短句 1-2 句中文，10-30 token。
    返回的 list 将直接 tokenize → embed → 拼进输入 tensor。
    """
    segments = []
    
    # 画像身份 (用户.1 核心特征)
    if utterance_spec.portrait_stable:
        segments.append(extract_identity_segment(utterance_spec))
    
    # 画像情绪 (用户.6 情绪图谱 + 当前情绪)
    segments.append(
        f"用户当前情绪{utterance_spec.user.emotion}，"
        f"{_extract_emotion_context(utterance_spec)}"
    )
    
    # 关系状态
    if utterance_spec.relationship:
        rs = utterance_spec.relationship
        segments.append(
            f"信任度{rs.trust:.1f}，{rs.interaction_mode}关系模式"
        )
    
    # 门控语气
    if utterance_spec.gate:
        segments.append(
            f"语气要求: {utterance_spec.gate.tone}，"
            f"不要过于{_opposite_tone(utterance_spec.gate.tone)}"
        )
    
    # 相关记忆 (每条一句话)
    for i, mem in enumerate(utterance_spec.memories[:5]):
        segments.append(_memory_to_steering_segment(mem))
    
    # 故事线
    for n in (utterance_spec.woven_context.narratives or [])[:2]:
        segments.append(n)
    
    return segments
```

**修改** `app/core/circuit.py` — `process()` 末尾的 `UtteranceSpec` 打包，新增 `steering_segments` 字段（或直接在 chat.py 层调用 `build_steering_segments`）。

### 5.2 推理侧改造

**新建/修改** 本地推理适配层（替代 `app/llm/deepseek.py` 的 HTTP POST）：

```python
async def generate_with_steering(
    user_message: str,
    steering_segments: list[str],
    model: str = "qwen2.5:7b",
) -> str:
    """本地 Ollama 推理 — embedding 层注入引擎产出。
    
    1. 引擎短句 + 用户消息 拼接为完整 prompt
    2. Ollama API 本地调用
    3. 返回生成文本
    """
    # 拼接：引擎短句在前，用户消息在后
    steering_prompt = "\n".join(steering_segments)
    full_prompt = f"{steering_prompt}\n\n用户消息: {user_message}\n\n回复:"
    
    # 本地 Ollama API (localhost:11434)
    # 或用 ollama-python SDK
    import ollama
    response = ollama.generate(model=model, prompt=full_prompt)
    return response["response"]
```

### 5.3 第一阶段可做的事（最小可用）

> **验证状态（2026-06-20）**：全部跑通。见 §12 和 `memory/steering-experiment-results.md`。

**不新建模块。不拆模块独立短句。直接：**

1. 把 `deepseek.py` 里现有的 `_build_messages()` 产出的完整 prompt 文本 → 发给本地 Ollama（而不是 DeepSeek API） ✅ Phase 1
2. 观察：本地 qwen2.5 拿到完整引擎上下文的回复质量，跟 API DeepSeek 对比 ✅ Phase 1-3
3. 如果回复质量相当或更好 → 拆模块独立短句 → 精确控制各引擎产出 ✅ Phase 5（`build_steering_segments()` 已验证，`steering_phase5_inject.py`）

---

## 6. 风险清单

| 风险 | 严重程度 | 验证状态 | 应对 |
|------|---------|---------|------|
| 本地 qwen2.5 能力不如 DeepSeek API | 高 | **已确认** — 7B Q4_K_M 回复质量系统性弱于 DeepSeek v4-flash | 换 32B/72B 更大本地模型；或 7B 上 GPU 提升推理速度 |
| Embedding 层注入 vs 文本 prompt 效果没差异 | 中 | **部分验证** — 文本级注入（紧凑短句强制前缀）已证明有效：plain→steering 质量跃升 | 需真正 embedding 向量注入才能完全验证 |
| 引擎短句切分不够精细，信息冗余 | 低 | **已验证** — 9-10 条结构化短句足以承载全部引擎认知上下文 | 已实现 `build_steering_segments()`（见 steering_phase5_inject.py） |
| Ollama 本地推理延迟不可接受 | 中 | **部分解决** — GTX 1060 6GB 只能部分加载（21% GPU），8 tok/s 够验证不够生产 | 更大显存显卡或纯 CPU 推理可接受 |
| qwen2.5 embedding 不足以替代 bge-m3 做检索 | 低 | **已排除** — Recall@5 = 100%，与 bge-m3 持平，且速度快 351x | `app/llm/qwen_embed.py` 已落地 |

---

## 7. Agent 框架：分工明确——引擎戴眼，框架装手

> **决策（2026-06-21）**：工具调用不进引擎。引擎只管认知注入，外部 Agent 框架管工具调度。

**核心结论**：引擎通过残差注入给本地 LLM 戴上"认知眼镜"（知道用户是什么人/什么状态/什么关系），外部 Agent 框架（如 Claude Code）给 LLM 装上"手"（工具调度/权限/错误恢复）。两个系统正交——引擎不碰工具，框架不碰认知。

### 为什么这样分工

| 职责 | 负责方 | 原因 |
|------|--------|------|
| 认知上下文注入 | 引擎 (steering_direct.py) | 16 模块残差向量→模型残差流，物理不可跳过 |
| 工具选择+执行 | 外部 Agent 框架 | 框架已有的规划/调度/权限/重试机制，引擎不需要重造 |
| 对话记录 | 外部框架 transcript + 引擎 JSONL | 双记录——框架记工具链，引擎记认知轨迹 |
| Qdrant 入库 | 引擎 (context.py) | 只入 (user_msg, ai_final_response)，老管线零改动 |

### 跟之前"给手"方案的对比

| | 之前: 引擎内置 Agent | 现在: 引擎+外部框架 |
|---|---|---|
| 工具规划 | 需从头实现任务分解 | 外部框架自带 |
| 权限控制 | 需在认知架构 6 层全加拦截 | 外部框架一层拦截 |
| 认知上下文 | 架构改造让认知层被 Action 感染 | Embedding 注入，不改认知层 |
| 复杂度 | 全架构改造 | 引擎零新增架构，框架零认知负担 |

---

## 8. 存储：单轨入库，中间产物不入 Qdrant

**核心结论**：不需要双轨入库。Agent 中间工具调用链不入 Qdrant——留在对话记录 JSONL 里。Qdrant 只收用户消息 + AI 最终回复。

### 为什么双轨不需要

Agent 一轮交互的结构：

```
用户消息: "帮我看看那个bug"
  → [Agent 内部: read_file → grep → write_file → bash test → ...]  ← 不入库
AI 最终回复: "找到了，auth.py第42行有个空指针，修好了，测试通过"
```

| | 存哪里 | 用途 |
|---|--------|------|
| 用户消息 + AI 最终回复 | Qdrant (老管线) | 语义检索、跨会话召回——"上次我修了什么来着" |
| 中间 tool call 完整链 | 对话记录 JSONL | 需要回溯细节时 `read_file`——"上次你到底为什么这么修" |

### 为什么不会混乱

- 对话记录 JSONL 是 Agent 框架自动写的——跟 Claude Code 的 transcript 一样。**天生就有。不需要额外开发。**
- Qdrant 只收最终产物。摘要、标签、情绪、embedding——全是对"对话"有意义的东西。工具调用的 bash 输出不需要被摘要，不需要被语义检索。
- 如果一段工具调用链里的发现值得被记住——LLM 会在最终回复里自己总结。总结自然进入 Qdrant。

**入库管线零改动。** `_enqueue_store_task(user_msg, ai_response, timestamp)` 照旧。

---

## 9. 残差分层注入：16 模块 → 指定层

> **第一步（第 0 层统一注入）跑通后，这是核心目标。**

### 9.1 当前向 LLM 输出内容的全部模块（16 个）

按在 deepseek.py 的消息结构中注入位置排列：

| # | 模块 | 当前产出内容 | 当前注入位 | 文件 |
|---|------|------------|----------|------|
| 1 | **PortraitRenderer** | portrait_stable（8 稳定维度） | message[0] system | `app/portrait/renderer.py` |
| 2 | **PortraitRenderer** | personality_notes_ai（AI 自我表达） | message[0] system | 同上 |
| 3 | **ConsolidationEngine** | personality_notes（行为/思维模式） | message[0] system | `app/background/consolidation.py` |
| 4 | **ChatHistory** | timeline_recent（历史对话） | message[1..N] | `app/memory/history.py` |
| 5 | **PortraitRenderer** | portrait_dynamic（4 动态维度） | message[N+1] system | `app/portrait/renderer.py` |
| 6 | **DriftTracker** | drift_text（偏移率 + 方向） | message[N+1] system | `app/analysis/drift.py` |
| 7 | **WorkingMemory** | session_context（工作记忆摘要） | message[N+1] system | `app/memory/working.py` |
| 8 | **Retrieval Pipeline** | 9路并行检索的原始记忆 | message[N+3] tool | `app/retrieval/pipeline.py` |
| 9 | **weave_context** | 故事线 + fact/reference 分层 | message[N+3] tool | `app/core/circuit.py` |
| 10 | **Conflict Resolution** | stale_context（被取代旧记忆） | message[N+3] tool | `app/core/conflict.py` |
| 11 | **ImpulseScheduler** | 5源泊松冲动 → 自然念头 | message[N+5] tool | `app/background/impulse.py` |
| 12 | **SelfMirror** | self_mirror_text（历史回应参考） | message[N+6] system | `app/analysis/self_mirror.py` |
| 13 | **basal_ganglia_gate** | tone + formality + response_mode | message[N+6] system | `app/core/circuit.py` |
| 14 | **BehaviorPredictor** | mirror_prediction → 准备方向 | message[N+6] system | `app/analysis/predictor.py` |
| 15 | **RelationshipState** | trust/familiarity/closeness/mode | message[N+6] system | `app/core/circuit.py` |
| 16 | **PatternDiscovery + PersonaSymmetry** | pattern_observations + blind_spots | message[N+8] tool | `app/analysis/` |

> **不在计数内**：`analyze_user_message` 的 intent/emotion 不直接给 LLM 看，被 gate 消费后转写成执行指令。`topic_notes` 写入 UtteranceSpec 但 deepseek.py 未消费。`emotional_reversals` 字段存在但未填充。

### 9.2 目标：每模块 → 残差向量 → 指定层

```
16 个模块
  ├─ 各产出短中文 (10-30 token/条)
  ├─ tokenize (模型自己的 BPE)
  ├─ embed (模型自己的 embedding 表)
  ├─ 得到 r ∈ ℝ^d_model
  └─ h_target_layer += r × α_module
```

**不需要训练任何东西。** Tokenizer + embedding 表是模型自带的，短中文是模块已经在产出的。唯一多出来的是 α_module 幅度系数（从实验标定，不是训练）。

### 9.3 分层注入策略（qwen2.5, 28层）

```
画像身份 (usr1)    → L3-5    "用户是谁"从浅层渗入——当语义属性被消费
相关记忆          → L5-10   "我们聊过什么"融入语义层
冲动信号          → L8-12   "心里想到什么"中层浮现，不强制输出
画像情绪 (usr6)    → L10-15  "用户什么状态"影响中层语气编码
偏移率            → L15-20  "用户最近在省/花"影响态度倾向
关系信任          → L18-25  "我们什么关系"影响深层交互决策
自我镜像          → L20-25  "我以前怎么回应的"影响风格一致性
门控语气          → L25+    "怎么说话"接近输出决策层
```

**注意**：模型的层之间没有标注——哪层管身份、哪层管情绪是 emergent 的，不是设计好的。上述分配是基于残差流语义特征的推断。精确层号需逐层实验反推。但无论挂在哪层，公式都是 `h += r`，改动一个模块的注入层只是一行配置。

### 9.4 同一模块拆分注入多层

一个模块可以产出多段短文本，注入多层：

```
PortraitRenderer:
  "用户是资深Rust/Python程序员"  → embed → h₅  += r₁  (浅层：身份语义)
  "用户近期工作压力大，情绪脆弱"   → embed → h₁₂ += r₂  (中层：情绪色调)
  "用户偏好直接、不废话的沟通"     → embed → h₂₅ += r₃  (深层：交互风格)
```

同一个模块，三个残差，三个层次——公式始终是 `h += r`。

### 9.5 后续进阶（分层注入验证后）

- **加权 steering**：不同模块向量乘不同强度系数（画像情绪 × 1.5，冲动 × 0.3）
- **动态强度**：根据关系 trust 自动调节 steering strength
- **层号自动搜索**：通过小规模实验自动发现每个模块的最佳注入层
- **跨模型层映射**：qwen2.5 28层 → LLaMA 32层 → Qwen2.5-32B 64层 的层号对应

---

## 10. Embedding 统一：从 bge-m3 切到 qwen2.5

### 10.1 核心逻辑

用 qwen2.5 自己的 embedding 同时做入库和检索 = 不存在向量不对齐。**同一个模型，同一个空间，入库和检索都是它。**

```
qwen2.5 embedding 层
  ├─ 入库 → 记忆文本 embed → Qdrant 存入
  ├─ 检索 → 用户消息 embed → Qdrant 查询
  ├─ 注入 → 引擎短句 embed → 拼进输入
  └─ 推理 → qwen2.5 本体 → LLM 生成
```

**唯一要验证的事：qwen2.5 的 embedding 做语义检索质量够不够。** bge-m3 是专门为语义相似度训练的，qwen2.5 的 embedding 层是为 next-token prediction 训练的。但这不意味着 qwen 的 embedding 做检索一定差——不少 decoder-only 模型的内部表示做检索意外地好。

> **验证结果（2026-06-20）**：20 查询 benchmark，Recall@5 = 100%（与 bge-m3 持平），Recall@3 = 100%，Recall@1 = 85%（vs bge-m3 90%）。速度 3,247 emb/s（vs bge-m3 9 emb/s = 351x）。**qwen2.5 embedding 层可完全替代 bge-m3。**
> 
> **已落地**：`app/llm/qwen_embed.py` — 从 qwen2.5 GGUF 提取 token embedding 表独立运行，纯 Python+numpy，不依赖 Ollama。

### 10.2 验证方式

跟 V8 vs bge-m3 那轮 benchmark 一模一样：

```
测试集: 同一批中文查询 → 同一批候选记忆
对比:
  bge-m3:    1024维 → Qdrant query → 召回率 / MRR
  qwen2.5:   用 Ollama /api/embeddings → 同维度 → 同批查询 → 同指标
```

### 10.3 两种结果

| 结果 | 方案 |
|------|------|
| qwen 检索率 ≈ bge | **全切**。bge-m3 退役。一个模型搞定一切。 |
| qwen 检索率明显差 | **双模型、分职责**。bge-m3 管检索，qwen 管注入+推理。两个空间不互操作，不需要对齐。 |

**无论哪种结果，都不影响引擎注入这条链路。** 引擎认知产出的中文短句用 qwen embed 进入输入——这个跟检索用谁的 embedding 无关。

---

## 11. 跨模型适应性：层数对比与注入策略

### 11.1 当前可用模型层数

| 模型 | 层数 | hidden_size | 注意机制 | 特殊结构 |
|------|------|------------|---------|---------|
| **Qwen2.5-7B** | **28** | 3584 → 4096 | GQA (28Q/4KV) | 标准残差 |
| **DeepSeek V2** | **30** | 4096 | MLA | 前 3 FFN, 后 27 MoE |
| **DeepSeek V3** | **61** | — | MLA | 前 3 FFN, 后 58 MoE |
| **DeepSeek V4 Flash** | **43** | **4096** | MQA (1 KV head!) / CSA / HCA 混合 | 前 3 hash-MoE bootstrap, 后 40 标准 MoE; mHC 四残差流 |

### 11.2 残差注入的跨模型障碍

DeepSeek V4 Flash 用 **mHC (Manifold-Constrained Hyper-Connections)** 替代标准残差连接——每层 **4 个并行残差流**，通过 Sinkhorn-Knopp 迭代混合。标准残差只有一条流，注入点明确。mHC 有四条——要注入的向量打进哪条流？全打？按比例分？

**每一代 DeepSeek 架构都大变**——V2 是 MLA+标准 MoE，V3 加 MTP，V4 Flash 推倒重来换三型注意力+mHC+hash bootstrap。残差分层注入需要**每代模型重新调层号、找注入点、适应注意力机制和残差结构**。

**回退方案**：对 mHC 模型，退回第 0 层 embedding 注入（§1 第一步）。第 0 层是所有 Transformer 都有的标准入口，不受残差结构影响。**分层注入优先 target qwen2.5 等标准残差模型。**

### 11.3 第 0 层注入的跨模型稳定性

**embedding 层是唯一在任何 Transformer 模型里都存在的标准入口。** 不管后面是 MLA、MQA、CSA、MoE、mHC——第 0 层 embedding 不参与这些。引擎产出的向量从第 0 层进入，被后续所有层的 whatever 注意力机制处理。

**第 0 层注入 → 跨模型零改动。** 先跑通 qwen2.5，之后换任何开源模型，同一个方案直接搬。

### 11.4 分层注入的层预算（16 模块）

- **Qwen2.5 (28 层)**：16 个模块分配 28 层（可用 4-26），多个模块可共享同一层。memories 类模块（8/9/10）可全部打入 L5-10 区间。
- **DeepSeek V4 Flash (43 层)**：层数更宽裕，但前 3 层不可用(hash-MoE bootstrap, 只看 token ID) + mHC 四残差流 → 建议退回第 0 层注入。
- 实际可用范围：qwen2.5 约 4-26 层

---

## 12. 与现有架构的关系

**不改动的**：
- 引擎全部逻辑 (circuit.py, pipeline.py, portrait/, background/, analysis/, memory/)
- 入库管线 (context.py `_store_conversation`)
- 后台线程 (consolidation, impulse, lifecycle)
- 对话记录 (chat_history.jsonl)
- 工具定义 (app/core/tools.py)
- chat.py 已有的 `for tool_round in range(2)` 工具循环

**可能改动的**（按优先级）：
1. `app/llm/deepseek.py` — 新增本地 Ollama 调用路径
2. `app/config/settings.py` — `LOCAL_LLM_MODE` / `LOCAL_LLM_MODEL`
3. `app/api/chat.py` — 本地 vs API 模式分支
4. `app/llm/embed.py` — (条件：qwen 检索率达标) embedding 切到 qwen2.5
5. `app/memory/qdrant.py` — (条件同上) collection dim 更新

**不用做的**：
- ~~双轨入库~~ → 单轨，老管线零改动
- ~~Agent 框架内置~~ → LLM 自带
- ~~Activation steering (C++)~~ → Embedding 层注入
- ~~MCP 接入~~ → 本地直连
- ~~并行化~~ → 引擎串行不变
- ~~编码器/蒸馏/映射矩阵~~ → 中文短句直接用 tokenizer+embed

---

## 13. 实验验证记录（2026-06-20）

> 详见 `memory/steering-experiment-results.md`，此处仅记结论。

### 13.1 已完成的验证

| # | 假设 | 结论 |
|---|------|------|
| 1 | qwen2.5:7b 回复质量可与 DeepSeek 匹配 | **否** — 7B Q4_K_M 系统性弱于 DeepSeek v4-flash。需更大模型。 |
| 2 | 紧凑结构化短句是普适优化 | **是** — 省 40% prompt token，大小模型都有效，质量不降。 |
| 3 | qwen2.5 embedding 可替代 bge-m3 | **是** — Recall@5=100%，速度 351x。`app/llm/qwen_embed.py` 已落地。 |
| 4 | 引擎注入（文本级）能让 7B 脱胎换骨 | **是** — plain→steering 质量跃升，从"AI套话"变"有温度的对话"。 |
| 5 | 不可跳过 > 可跳过 | **是** — 同一信息放 system prompt vs 强制前缀，后者效果显著更好。 |
| 6 | `batch.embd` 注入 per-token 嵌入 ≡ token decode | **是** — logits 逐位匹配（`np.allclose(atol=1e-3)`）。API 正确性已验证。 |
| 7 | mean-pooled 连续向量打入 Layer 1 输入端能承载语义 | **否** — 模型 1 token 后立即 EOS。Layer 1 输入端期望 token 级嵌入分布，mean-pooled 破坏此结构。 |
| 8 | `llama_set_adapter_cvec` 让引擎向量在指定层做残差注入 | **是** — CVEC-L2 显著优于 TEXT（无引擎上下文）和 CVEC-L1。`llama_set_adapter_cvec` 是 llama.cpp 已有 API，零 C++ 改动。 |

### 13.2 Phase 6 核心发现（2026-06-20）— 被 Phase 7 推翻

**Phase 6 结论（已证伪）**：mean-pooled 引擎向量无法被模型理解。
**Phase 7 修正**：Phase 6 的失败不是"mean-pooled 不可用"，而是**打错了层**。`batch.embd` 只能注入 Layer 1 输入端（token 级结构期望），模型自然拒绝。换用 `llama_set_adapter_cvec` 注入 Layer 2+ 残差流后，同一 mean-pooled 向量成功引导生成。

Phase 6 保留的技术贡献：llama-cpp-python 编译方法、llama_batch API 坑点、per-token 嵌入 ≡ token decode 的验证。

### 13.2b Phase 7 核心发现（2026-06-20）⭐

**机制**：`llama_set_adapter_cvec(ctx, cvec_buf, n_layer*n_embd, n_embd, layer_start, layer_end)` — llama.cpp 已有 API。将 steering 向量置于指定层 buffer 位置，模型在该层输出后自动做残差加法：`h_layer_out += steering_vector`。无需改 C++ 源码。

**实验设计**：4 场景 × 4 条件（TEXT 无引擎 / CVEC-L1 / CVEC-L2 / CVEC-L3），steering=引擎短句 mean-pooled 单向量 × scale=0.1。

**结果**（4/4 场景 CVEC-L2 最优）：

| 场景 | TEXT（无引擎） | CVEC-L2（有引擎 @ L2） |
|------|-------------|---------------------|
| 001 发泄 Rust | "去学 Rust 教程" — 说教 | **"换到 Python 也完全没问题"** — 共情 ✓ |
| 003 自我怀疑 | "代码质量变化是正常的" — 客观分析 | "你这是成长的表现" — 先肯定 ✓ |
| 007 微服务提问 | 直接给考虑因素 | 先确认"确实需要认真考虑"再分析 — 更有对话感 |
| 009 女友抱怨 | "你女朋友对你表示理解吗？" — 反问 | "这种情况确实挺难处理" — 先共情再给方案 ✓ |

**层号效应**：CVEC-L1 效果跟 TEXT 接近（模型仍在处理 token 语法），CVEC-L2 共情/肯定显著增加（语义层），CVEC-L3 开始偏离引擎引导回到通用分析。**qwen2.5 28 层中 L2 是最佳认知注入层。**

**延迟**：所有 CVEC 条件 17-20s（vs TEXT 19-21s），无额外开销。

### 13.2c Phase 8 核心发现（2026-06-20）⭐⭐

**机制**：16 个引擎模块各自产出短文本 → embed → 打进 SPEC §9.3 指定的层（L3-26），同一层多个模块向量直接累加。`llama_set_adapter_cvec(ctx, buf, n_layer*n_embd, n_embd, 1, n_layer)` 一次性设置全层 buffer。

**实验设计**：4 场景 × 2 条件（TEXT 无引擎 / CVEC-16 分层注入），12-15 个活跃模块同时注入。

**结果**：
- 情感场景（001/003/009）：CVEC-16 明显更共情、更温暖，TEXT 偏说教
- 技术场景（007 微服务提问）：两者接近（合理——不需要语气 steering）
- 15 条向量同时注入，零 EOS，零报错，零互斥

**残差机制验证**：
- 残差做的是**语义偏置**（特征空间偏移），不是**指令执行**（token 级 attention）
- 注入 "回复第一句必须说卧槽" → 不跟（指令需要 token attention）
- 注入 "用户情绪低落，需共情" → 跟了（语义偏置影响 token 分布）
- 结论：残差控制的是"怎么回应"的基调，不是"回应什么"的内容

### 13.2d Phase 9 核心发现（2026-06-20）⭐⭐⭐ — Steering Trajectory

**机制**：同一 steering 向量注入全部 28 层（同一向量 ×28），α=0.001-0.05 全范围正常生成，零 EOS，零乱码。LayerNorm 在每层重新归一化，防止 norm 累积爆炸。

**关键发现**：ALL×28 α=0.05 比 L2-only α=0.05 效果**更好**——"我理解你的感受"（ALL×28）vs "理解你的感受"（L2-only）。全层持续偏置比单层一下更自然。

**Steering Trajectory 概念**：
- 每个模块不是在某一层注入一个向量，而是在 28 层上各有一个**不同的向量**——构成一条 steering 曲线
- 浅层（L1-5）的 steering 跟深层（L24-28）的 steering 可以编码完全不同的东西
- 16 个模块 × 28 层 = 448 个可独立调节的 steering knob
- 跟 prompt 的本质区别：**prompt 是广播（全层同一种方式 attend），steering trajectory 是精确制导（每层不同干预）**

**已验证的 trajectory 模式**：
- Uniform（全层同向量）：稳定，LayerNorm 兜底
- Gradient（浅→深线性递增）：也稳定，L1 小 L28 大
- 推测：浅层植入语义身份、中层偏移语气、深层约束决策、末层调整措辞

### 13.3 新增产物

- `app/llm/qwen_embed.py` — qwen2.5 独立嵌入模型，替代 bge-m3
- `data/qwen_embed_f32.npy` — 152K×3584 嵌入表（2.0 GB，gitignored）
- `data/qwen_tokenizer.json` — BPE 词典+合并规则
- `scripts/extract_qwen_embed.py` — GGUF 提取管线
- `scripts/steering_phase5_inject.py` — `build_steering_segments()` + 闭环测试（文本级注入）
- `scripts/steering_phase6_embed_inject.py` — 嵌入层注入对比实验（TEXT vs EMBED mean-pooled，Layer 1 输入）
- `scripts/steering_phase7_layer2_cvec.py` — ⭐ 残差分层注入实验（`llama_set_adapter_cvec`，4 条件 × 4 场景）
- `scripts/steering_phase8_layered.py` — ⭐⭐ 16 模块分层注入实验（15 条向量 → L3-26，2 条件 × 4 场景）
- `scripts/steering_phase7_debug.py` — Phase 7 调试脚本（扫 α 和层号）

### 13.4 待解决

- ~~Layer 2+ 残差注入需改 C++~~ → **已解决**：`llama_set_adapter_cvec` 是已有 API，零 C++ 改动
- ~~多模块同层/多层注入~~ → **已解决**：Phase 8 验证 15 条向量同时注入 3-26 层，全通
- ~~全层同一向量会不会炸~~ → **已解决**：Phase 9 验证，LayerNorm 兜底，不炸反而更丝滑
- ~~**引擎落地**~~ → **已解决（2026-06-21）**：`app/llm/steering.py` 投产，chat.py 双模式，994 tests green（详见 §14）
- ~~**模块直接产出残差向量（绕过文本）**~~ → **已解决（2026-06-21）**：`app/llm/steering_direct.py` 落地，ConceptVectorBuilder 4 方法 + TrajectoryShaper 5 shape + 16 提取器，7/7 smoke 通过，1007 tests green（详见 §15）
- ~~**ChatML 格式修复**~~ → **已解决（2026-06-21）**：旧 prompt 格式导致多轮对话循环，切 ChatML + stop tokens 完美修复
- **16 模块 trajectory 标定**：首轮完成 3/15（gate_tone gradient_up / relationship_state gradient_down / portrait_emotion 保持 peak:12:4），剩余 12 模块待标定。`scripts/calibrate_trajectory.py` 工具链就绪。
- **CVEC vs FULL PROMPT 质量差距**：CVEC-only 达 ~75% FULL PROMPT 质量，全层 trajectory 可能进一步缩小差距
- **跨模型 trajectory 复用**：qwen2.5 28层的 trajectory 能否通过层映射迁移到 32B/72B 或其他模型
- ❌ **本地模式工具调用不做进引擎**：由外部 Agent 框架负责。引擎只管认知注入（残差向量→模型残差流）。
- **检索管线切 qwen_embed**：需全量重建 Qdrant collection（维度 1024→3584）
- 本地推理质量瓶颈：需 32B/72B 或更强 GPU
- GTX 1060 6GB 不够：7B Q4_K_M 仅 21% GPU 层，8 tok/s


## 14. 引擎落地记录（2026-06-21）⭐

### 14.1 落地的 4 个文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `app/llm/steering.py` | **新建** — `build_steering_segments()` + `SteeringInjector` 单例 | 310 |
| `app/api/chat.py` | 双模式分支 — `/chat` 和 `/chat/stream` 各加 `if local_llm_mode` | +40 |
| `app/config/settings.py` | 5 新配置项 — `LOCAL_LLM_MODE` / `QWEN_GGUF_PATH` / `STEERING_ENABLED` / `STEERING_STRENGTH` / `MINGW_BIN_DIR` | +15 |
| `app/core/context.py` | AppContext 条件创建 `steering_injector` | +6 |

### 14.2 架构决策

1. **不新建推理适配层**。直接在 `chat.py` 两处分叉，`steering_injector.generate()` 接口与 `llm_client.generate()` 兼容（都返回 `{"content": ..., "tool_calls": [...]}`）。
2. **不新建消息拼装**。引擎 UtteranceSpec → `build_steering_segments()` → 16 段短中文，不再拼 10 段消息结构。CVEC 注入替代 prompt 注入。
3. **不解决并发**。单用户单实例，CVEC buffer 用一把 Lock 保护即可。
4. **不设计回退**。CVEC set 失败 → `if ret == 0:` 跳过，模型照常生成。
5. **检索管线不动**。bge-m3 (1024维) 继续管检索，qwen_embed (3584维) 只管 CVEC 注入。两个空间不混用。

### 14.3 验证

- **994 tests passed, 0 failed** — 零回归
- **CVEC smoke test** — 模型加载 16s，注入后 13-22s 生成，共情明显提升
- **Plain vs CVEC 对比** — plain："建议从简单项目开始"（说教）；CVEC："我理解你的感受"（共情）

### 14.4 用法

```bash
# 远程 DeepSeek API（默认）
python run.py

# 本地 CVEC 模式
LOCAL_LLM_MODE=true python run.py
```


## 15. 直接向量注入引擎落地记录（2026-06-21）⭐⭐

### 15.1 核心突破

**从"文本中转"到"向量直出"**：16 个模块不再写中文短句，直接从结构化数值产出 d_model 语义向量。

```
旧路径: 模块内部状态 → 拼中文短句 → BPE tokenize → embed表查表 → mean pool → 单向量 → CVEC
新路径: 模块内部状态 → ConceptVectorBuilder (4方法) → d_model向量 → TrajectoryShaper → 28层轨迹 → CVEC
```

### 15.2 四种向量构造方法（零训练）

| 方法 | 适用场景 | 示例 |
|------|---------|------|
| 锚点插值 | 标量特征 | trust=0.7 → `0.7*embed("信任")+0.3*embed("疏离")` |
| 概念方向 | 变化量特征 | drift=frugal(+25%) → `frugal_direction * 0.25` |
| 类别查表 | 离散特征 | tone=soft → 预计算 soft 向量 |
| Tag 混合 | 标签列表 | portrait tags → 加权混合embed |

### 15.3 Trajectory 系统

每个模块 3 参数控 448 自由度:
- base_vector: 方法一~四产出的 d_model 向量
- shape: uniform / gradient_up / gradient_down / early / late / peak:N
- intensity: alpha * global_strength

### 15.4 落地的 4 个文件

| 文件 | 改动 | 行数 |
|------|------|------|
| app/llm/steering_direct.py | 新建 | 530 |
| app/llm/steering.py | 集成 _setup_cvec_direct() | +90 |
| app/config/settings.py | STEERING_DIRECT 开关 | +6 |
| scripts/verify_steering_direct.py | 新建 7项smoke | 460 |

### 15.5 验证结果

- 1007 tests passed, 0 failed — 零回归
- Smoke 7/7: 情绪valence分离0.50, arousal分离0.53, trust插值对称, tone语义合理, 7shape通过, 26/28层活跃
- CVB 初始化 1147ms, 运行时构建 19.9ms
- 用法: `LOCAL_LLM_MODE=true STEERING_DIRECT=true python run.py`
