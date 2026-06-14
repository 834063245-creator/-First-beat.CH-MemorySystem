# SPEC: 偏移率 + 自我镜像

> 状态: DRAFT · 2026-06-14
> 两个独立特性的联合设计文档, 因为它们在注入层有共用点

---

## Part A: 偏移率追踪 (Drift Velocity)

### A1. 问题

画像系统知道「用户是谁」, 不知道「用户正在怎么变」。同一个人三天前在学 Rust, 今天说 Rust 太难退回 Python——画像看到两个兴趣标签, 看不到中间的放弃轨迹。

### A2. 核心概念

```
spend (愿投, +)  ←──→  frugal (省钱, λ)  ←──→  drift (放弃, -)
                                                 ├─ 放弃 (深)
                                                 ├─ 妥协 (中)
                                                 └─ 烦躁 (浅)
```

### A3. 检测 (纯规则, 零 LLM)

信号词表匹配, 三档 drift 独立检测:

```
"花钱/付费/值得投资/效率优先"     → spend
"免费/省钱/自己搞/开源/性价比"    → frugal
"不管了/随便/放弃/不做了"        → drift_放弃
"算了/将就/凑合/能用就行"        → drift_妥协
"烦死了/劝退/坑爹/垃圾/有毒"     → drift_烦躁
```

优先级: drift > spend > frugal。同方向连续决策 EMA 合并 (α=0.7)。

### A4. 文件

| 文件 | 职责 | 行数 |
|------|------|------|
| `app/analysis/drift.py` | DriftTracker: detect(), comprehensive_offset(), _log() | ~200 |
| `app/config/settings.py` | +`DRIFT_DECISION_LOG` 路径 | +2 |
| `app/core/context.py` | `_store_conversation()` 后挂 detect() | +5 |
| `app/portrait/writer.py` | 浅巩固读取 comprehensive_offset → 更新 用户.2 | +8 |

### A5. 注入

偏移状态写入 `CognitiveState` 新字段, 在 `message[N+1]` (dynamic system) 中与画像动态维度一同注入:

```
【当前状态】
用户 2. 当前状态: ... | 偏移: frugal(+25%) 连续3轮节省倾向
```

---

## Part B: 自我镜像 (Self-Mirror)

### B1. 问题

AI 在面对用户情绪时, 每次都是"第一次"。它不知道自己上次遇到类似情境时用了什么策略、效果如何。不是"让 LLM 自己反思", 而是让引擎把 AI 过去的行为模式**作为上下文喂给它**, 它自己决定这次要不要不一样。

### B2. 核心概念

```
当前情境: 用户情绪低落 · 焦虑
         ↓
引擎查询 AI ChromaDB → 找"AI 在过去类似情境下的回应"
         ↓
对每条回应 → ChatHistory 取用户下一轮反应 → analyze_emotion_2d() 标记有效性
         ↓
组装成自我镜像片段 → 注入 prompt
```

### B3. 数据管道

| 步骤 | 从哪读 | 具体查询 |
|------|--------|---------|
| 1. 当前用户情绪 | `UtteranceSpec.user.emotion` | valence/arousal/category |
| 2. AI 过去的回应 | AI ChromaDB | 按情绪标签检索 AI 记忆中跟当前情绪接近的 |
| 3. 用户下一轮反应 | ChatHistory | `get_context_by_timestamp(ai_response_ts, before=0, after=1)` |
| 4. 反应情绪分析 | `analyze_emotion_2d()` | 对用户反应跑情绪 → 标记正向/负向 |
| 5. 组装 prompt | 新模块 | 最多 3 条, 每条 ≤80 字 |

### B4. 新模块: `app/analysis/self_mirror.py`

```python
class SelfMirror:
    """AI 自我镜像生成器——查 AI 过去在相似情绪下的回应, 组装为 prompt 片段。
    纯读操作, 零 LLM 调用, 不落盘。
    """

    def build_mirror(self, user_emotion: dict, ai_chroma, chat_history, *, limit=3) -> str:
        """输入当前用户情绪, 输出自我镜像 prompt 段, 或空字符串。"""
        # 1. 去 AI ChromaDB 检索相似情绪下的 AI 记忆
        ai_memories = ai_chroma.query_by_emotion(
            valence_range=(user_emotion["valence"] - 0.3, user_emotion["valence"] + 0.3),
            limit=10
        )
        if not ai_memories:
            return ""

        # 2. 对每条 AI 记忆, 查用户下一轮反应
        episodes = []
        for mem in ai_memories:
            ts = mem["metadata"]["timestamp"]
            ctx = chat_history.get_context_by_timestamp(ts, before=0, after=1)
            if not ctx or not ctx.get("context_after"):
                continue
            user_reaction = ctx["context_after"][0]["user_message"]
            e_valence, e_arousal, e_cat = analyze_emotion_2d(user_reaction)

            episodes.append({
                "when": ts,
                "ai_response": mem["document"][:120],
                "user_reaction": user_reaction[:80],
                "reaction_valence": e_valence,
                "effective": e_valence > 0.3,  # 用户反应正向 = 有效
            })
            if len(episodes) >= limit:
                break

        if not episodes:
            return ""

        # 3. 组装
        return self._render_mirror(episodes, user_emotion)

    @staticmethod
    def _render_mirror(episodes: list[dict], current_emotion: dict) -> str:
        """组装为 LLM 可读的自我镜像段落。"""
        emo_label = current_emotion.get("category", "neutral")
        lines = [f"【自我镜像 — 面对{emo_label}情绪的你】"]
        for i, ep in enumerate(episodes, 1):
            tag = "✓ 用户转向正向" if ep["effective"] else "✗ 用户情绪未改善"
            lines.append(
                f"{i}. {ep['when'][:10]} | 你: {ep['ai_response'][:80]}...\n"
                f"   用户反应: {ep['user_reaction'][:50]}... ({tag})"
            )
        return "\n".join(lines)
```

### B5. 文件

| 文件 | 职责 | 行数 |
|------|------|------|
| `app/analysis/self_mirror.py` | SelfMirror 类 | ~150 |
| `app/core/context.py` | AppContext 持有 SelfMirror 实例 | +3 |
| `app/llm/deepseek.py` | `_build_execute_directive()` 接受 self_mirror 参数, 拼接 | +8 |
| `app/core/circuit.py` | gate 之后调 `self_mirror.build_mirror()` → 传入 CognitiveState | +5 |

### B6. Prompt 注入位置 (关键)

自我镜像是策略性上下文, 不属于工具调用协议, 不属于系统指令。最佳位置: **融入 `message[N+6]` (execute_directive) 消息体**, 与执行指令并列:

```
message[N+6]: system ─────────────────────────────────────
│
│  【自我镜像 — 面对负向情绪的你】
│  1. 2026-06-10 | 你: "理解你的沮丧, 架构设计确实..." 
│     用户反应: "嗯, 让我重新想想..." (✓ 用户转向正向)
│  2. 2026-06-08 | 你: "要不要深入聊聊具体的困难点？"
│     用户反应: "算了不说了" (✗ 用户情绪未改善)
│
│  【执行指令】
│  意图: emotional_sharing | 情绪: negative
│  回应模式: gentle | tone: 先确认再引导
│  注意: 上次追问策略效果差, 这次可考虑沉默陪伴或技术话题转移
│
└──────────────────────────────────────────────────────
```

**为什么是这个位置:**
1. 在记忆和冲动之后 (AI 已有完整上下文)
2. 在执行指令之前 (AI 先看到自己的过去再看到指令)
3. 不新增 message 索引 (不破坏缓存结构)
4. 紧邻用户消息 (反射内容最新鲜)

### B7. 边界条件

| 条件 | 行为 |
|------|------|
| AI ChromaDB 无相似情绪记忆 | 返回空串, 不注入 |
| 找到了但用户下一轮是系统消息 | 跳过该条 |
| 找到了但用户下一轮情绪中性 (|valence|<0.2) | 进入列表, 标记"中性" |
| 相同日期多条相似情境 | 只取最差异化的 3 条 (按 reaction_valence 分散采样) |

---

## Part C: 两者交汇

### C1. 注入层共用

偏移率 (A) 和自我镜像 (B) 在 prompt 中是相邻的信息:

```
message[N+1] dynamic system:
  【当前状态】
  用户情绪: 低落 · 焦虑 | 偏移: drift(-40%) 从Rust→Python撤退
  ...

message[N+6] system:
  【自我镜像 — 面对负向情绪的你】     ← B
  ...
  【执行指令】                        ← 原有
  ...
```

两者不互相依赖, 但都是从「用户当前状态」推导出「AI 应该知道什么」。

### C2. 引擎决策层共用

gate 计算 GatingDecision 时, 可以同时读偏移率和自我镜像的结果:

```python
# circuit.py gate 逻辑
gating = basal_ganglia_gate(intent, emotion)

# 偏移率调制
if drift.direction == "drift" and drift.offset < -40:
    gating.tone = "不要追问, 不要给方案"

# 自我镜像调制
if mirror.last_similar_was_ineffective:
    gating.tone = "避免上次的策略"
```

gate 集中决策, LLM 和子系统都只读结果。
