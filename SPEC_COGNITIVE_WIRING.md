# SPEC：认知层五条连线 — 画像·反馈·预测·门控·冲动互联

> **目标**：让引擎的画像系统不再只是"记录"，而是真正参与引擎的检索、门控、冲动决策。
> **总改动量**：~145 行，5 个文件，0 个新文件。
> **风险**：低。所有改动都是"在已有参数/已有 ctx_obj 上追加读取逻辑"，不改变现有数据流结构。

---

## 前置条件

确认以下文件存在且未被人为大幅改动：

- `app/core/feedback.py` — 当前 45 行，含 `log_error_report()` 和 `clear_memory_errors()`
- `app/portrait/writer.py` — 含 `realtime_update()` 方法，方法末尾调用 `self._manager.save()`
- `app/core/circuit.py` — 含 `basal_ganglia_gate()` 函数（第 112 行附近）和 `CircuitOrchestrator` 类
- `app/background/impulse.py` — 含 `ImpulseScheduler` 类和 `SOURCE_CONFIG` 列表（第 281 行附近）
- `app/core/context.py` — 第 165 行附近创建 `ImpulseScheduler` 实例

---

## 连线①：Feedback → Portrait（用户纠错闭环）

**问题**：`feedback.log_error_report()` 只往 `data/error_reports.jsonl` 追加日志，没有任何代码消费这些日志。用户说"你记错了"之后，引擎行为完全不变。

**方案**：`PortraitWriter.realtime_update()` 在每轮对话后读取近 24 小时的错误报告，找到关联的画像条目，降低其 confidence 并标记为 PENDING。

### 改动 1.1 — 文件：`app/core/feedback.py`

**位置**：文件末尾，`clear_memory_errors()` 函数之后（当前最后一行是 `return 0`）

**插入**：

```python
def get_recent_corrected_ids(data_dir: str = "data", since_hours: int = 24) -> set[str]:
    """读取近 N 小时内被用户标记为错误的 memory_id 集合。

    供 PortraitWriter 消费：用户说"记错了"→关联画像条目标记为待验证。
    """
    path = os.path.join(data_dir, "error_reports.jsonl")
    if not os.path.exists(path):
        return set()
    cutoff = time.time() - since_hours * 3600
    ids: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 跳过 clear 标记，只看 error 报告
                if rec.get("action") == "clear":
                    continue
                if rec.get("timestamp", 0) > cutoff and rec.get("memory_id"):
                    ids.add(rec["memory_id"])
    except OSError:
        pass
    return ids
```

**验证**：改完后 `app/core/feedback.py` 应包含 3 个函数：`log_error_report`、`clear_memory_errors`、`get_recent_corrected_ids`。

---

### 改动 1.2 — 文件：`app/portrait/writer.py`

**位置**：`realtime_update()` 方法内，在 `self._turns_since_last_deep += 1` 之前插入。

**查找特征码**：找到以下两行：
```python
        self._turns_since_last_deep += 1
        self._manager.save()
```

**替换为**：
```python
        # ── 反馈消费：用户"记错了" → 关联画像条目 confidence 下降 ──
        try:
            from app.core.feedback import get_recent_corrected_ids
            corrected = get_recent_corrected_ids()
            if corrected:
                for entry_id, entry in list(self._manager._entries.items()):
                    # 检查条目标签或文本是否引用了被纠正的 memory_id
                    entry_text_and_tags = entry.text + " " + " ".join(entry.tags)
                    if any(mid in entry_text_and_tags for mid in corrected):
                        entry.confidence = max(0.1, entry.confidence - 0.3)
                        entry.status = EntryStatus.PENDING
                        logger.info("画像条目 %s confidence 降至 %.2f（反馈纠正）",
                                    entry_id, entry.confidence)
        except Exception:
            pass  # 反馈消费失败不影响主链路

        self._turns_since_last_deep += 1
        self._manager.save()
```

**验证**：搜索 `realtime_update` 方法，确认在 `_turns_since_last_deep` 之前有上述反馈消费块。

---

## 连线②：Portrait → Gate（画像情绪趋势调制门控语气）

**问题**：`basal_ganglia_gate()` 的情绪→语气映射只看当前消息。用户连续两周低落但当前消息是 casual，门控仍用 warm 语气。

**方案**：在已有的"引擎调参覆盖"块中，追加一段读 `portrait_manager` 的 `usr6`（情绪图谱）维度，发现 ≥2 条负面活跃条目时收敛语气。

### 改动 2.1 — 文件：`app/core/circuit.py`

**位置**：函数 `basal_ganglia_gate()` 内，现有的 `# ── 引擎调参覆盖 ──` 块中。

**查找特征码**：
```python
    # ── 引擎调参覆盖 ──────────────────────────────────────────
    if ctx_obj is not None:
        try:
            tuning = ctx_obj._pattern_discovery.get_tuning()
            if tuning.get("emotional_dampening"):
                tone = "neutral"
                pfc.emotion_intensity = max(0.0, pfc.emotion_intensity - 1.0)
            if tuning.get("formality_shift"):
                formality = max(0.0, min(1.0, formality + tuning["formality_shift"] * 0.15))
        except Exception:
            pass
```

**替换为**：
```python
    # ── 引擎调参覆盖 ──────────────────────────────────────────
    if ctx_obj is not None:
        try:
            tuning = ctx_obj._pattern_discovery.get_tuning()
            if tuning.get("emotional_dampening"):
                tone = "neutral"
                pfc.emotion_intensity = max(0.0, pfc.emotion_intensity - 1.0)
            if tuning.get("formality_shift"):
                formality = max(0.0, min(1.0, formality + tuning["formality_shift"] * 0.15))
        except Exception:
            pass

        # Phase 2: 画像情绪趋势调制语气
        try:
            portrait = getattr(ctx_obj, "portrait", None)
            if portrait is not None and not portrait.is_empty:
                usr6_entries = portrait.get_dim_entries("usr6")
                negative_keywords = ("低落", "焦虑", "沮丧", "压力", "烦躁", "疲惫",
                                     "negative", "anxious", "depressed", "frustrated")
                negative_count = sum(
                    1 for e in usr6_entries
                    if e.status.value == "active"
                    and any(kw in e.text for kw in negative_keywords)
                )
                if negative_count >= 2 and pfc.intent not in ("conflict",):
                    # 长期低落 → 收敛直接语气，避免过于轻快
                    if tone in ("warm", "direct"):
                        tone = "soft"
                    formality = max(formality, 0.4)
        except Exception:
            pass
```

**验证**：搜索 `basal_ganglia_gate` 函数体，确认 `引擎调参覆盖` 块内包含画像情绪趋势读取逻辑。

---

## 连线③：Portrait → weave_context（画像热度调制检索分层阈值）

**问题**：`weave_context()` 的 `MIN_FACT_DIST = 0.30` 是写死的。用户极度关注的话题不会因此放宽检索阈值。

**方案**：`weave_context()` 新增可选参数 `portrait_boost`，在分层阈值计算时乘以 tag 命中 boost；`process()` 中预先计算 boost map 并传入。

### 改动 3.1 — 文件：`app/core/circuit.py`，方法 `weave_context()`

**位置**：方法签名所在行。

**查找特征码**：
```python
    def weave_context(
        self,
        candidates: list[dict],
        cognitive_state,
    ) -> "WovenContext":
```

**替换为**：
```python
    def weave_context(
        self,
        candidates: list[dict],
        cognitive_state,
        portrait_boost: dict[str, float] | None = None,
    ) -> "WovenContext":
```

---

### 改动 3.2 — 同一文件，`weave_context()` 方法内

**位置**："层三：分层"段落。找到 `for m in active:` 循环内 `MIN_FACT_DIST = 0.30` 之后的阈值计算区域。

**查找特征码**：
```python
        for m in active:
            mid = m.get("id", "")
            dist = m.get("distance", 0.5)
            source = m.get("source", "semantic")
            is_stale = m.get("_stale", False)

            # story line 里的：非 stale → fact，stale → stale_context
            if mid in used_in_narrative:
```

**替换为**：
```python
        # ── 画像 boost 预计算：tag → 调制系数 ──
        _pboost = portrait_boost or {}

        for m in active:
            mid = m.get("id", "")
            dist = m.get("distance", 0.5)
            source = m.get("source", "semantic")
            is_stale = m.get("_stale", False)

            # 画像 tag boost：命中画像热点 → 放宽阈值
            tag_multiplier = 1.0
            if _pboost:
                for tag in m.get("_tags", []):
                    tag_multiplier = max(tag_multiplier, 1.0 + _pboost.get(tag, 0.0))

            # story line 里的：非 stale → fact，stale → stale_context
            if mid in used_in_narrative:
```

---

### 改动 3.3 — 同一文件，同一循环内，阈值计算行

**查找特征码**（在 `if mid in used_in_narrative:` 之后的 else 分支中）：
```python
            threshold = MIN_FACT_DIST * boost
```

**替换为**：
```python
            threshold = MIN_FACT_DIST * boost * tag_multiplier
```

---

### 改动 3.4 — 文件：`app/core/circuit.py`，方法 `process()` 内

**位置**：`weave_context` 调用之前。当前调用为 `self.weave_context(memories, prefrontal)`。

**查找特征码**：
```python
        # ── 引擎编织：从候选集中织出上下文（替代 TOP_K 截断）──
        woven = self.weave_context(memories, prefrontal)
```

**替换为**：
```python
        # ── 引擎编织：从候选集中织出上下文（替代 TOP_K 截断）──
        # 预计算画像 boost map，供 weave_context 调制分层阈值
        _portrait_boost = {}
        try:
            if hasattr(ctx_obj, "portrait") and ctx_obj.portrait is not None:
                _portrait_boost = ctx_obj.portrait.compute_portrait_boost_map()
        except Exception:
            pass

        woven = self.weave_context(memories, prefrontal, portrait_boost=_portrait_boost)
```

**验证**：确认 `weave_context` 调用的第二个参数位置出现 `portrait_boost=_portrait_boost`。

---

## 连线④：Predictor → Gate（行为预测影响响应模式）

**问题**：`mirror_prediction` 已在 `process()` 中计算并传入 `UtteranceSpec` 给 LLM，但 `basal_ganglia_gate()` 完全不使用它来做自己的门控决策。

**方案**：给 `basal_ganglia_gate()` 增加 `mirror_prediction` 参数，在调参块中根据预测意图预先调制 `response_mode`。

### 改动 4.1 — 文件：`app/core/circuit.py`，函数 `basal_ganglia_gate()`

**位置**：函数签名。

**查找特征码**：
```python
def basal_ganglia_gate(
    prefrontal: UserMessageAnalysis,
    memories: list,
    impulses: list,
    personality_notes: list,
    ctx_obj=None,
) -> GatingDecision:
```

**替换为**：
```python
def basal_ganglia_gate(
    prefrontal: UserMessageAnalysis,
    memories: list,
    impulses: list,
    personality_notes: list,
    ctx_obj=None,
    mirror_prediction: dict | None = None,
) -> GatingDecision:
```

---

### 改动 4.2 — 同一函数，引擎调参覆盖块末尾（连线②画像情绪块之后、`return` 之前）

**查找特征码**：
```python
        except Exception:
            pass

    return GatingDecision(
```

**替换为**：
```python
        except Exception:
            pass

    # Phase 2: 行为预测预调门控模式
    if mirror_prediction:
        predicted_intent = mirror_prediction.get("predicted_intent", "")
        if predicted_intent == "ask_fact" and response_mode == "auto":
            response_mode = "direct_answer"
        elif predicted_intent == "emotional_sharing" and tone == "warm":
            tone = "caring"

    return GatingDecision(
```

---

### 改动 4.3 — 同一文件，`process()` 方法内，`basal_ganglia_gate` 调用处

**查找特征码**（第 396 行附近）：
```python
        # ④ 响应门控
        gate = basal_ganglia_gate(
            prefrontal, fact_memories, impulses, temp.personality_notes, ctx_obj)
```

**替换为**：
```python
        # ④ 响应门控
        gate = basal_ganglia_gate(
            prefrontal, fact_memories, impulses, temp.personality_notes,
            ctx_obj, mirror_prediction=mirror_prediction)
```

**验证**：确认 `mirror_prediction` 变量在 `process()` 中已存在于调用之前（它确实在约 240 行处计算），且此处正确传入了关键字参数。

---

## 连线⑤：Portrait → Impulse（画像驱动物探索冲动源）

**问题**：5 个冲动源全是时间/情绪/随机驱动。画像已经知道用户关注什么、对什么敏感，冲动系统完全不知道。

**方案**：新增 `source_portrait_curiosity` 冲动源，从画像 `usr2`（关注焦点）+ `usr5`（兴趣图谱）取探索候选，排除 `usr6` 负向触发话题，产出定向探索冲动。注册为第 5 个泊松源。

### 改动 5.1 — 文件：`app/background/impulse.py`

**位置**：在 `source_curiosity()` 函数之后、`# ── 调度器 ──` 注释之前插入新函数。

**查找特征码**：
```python
        return (picked, 15)
    except Exception as exc:
        logger.debug("curiosity 源异常: %s", exc)
    return None


# ── 调度器 ──────────────────────────────────────────────────
```

**替换为**：
```python
        return (picked, 15)
    except Exception as exc:
        logger.debug("curiosity 源异常: %s", exc)
    return None


def source_portrait_curiosity(portrait_manager=None, all_mems=None, **kwargs) -> tuple | None:
    """画像驱动的好奇心源：对用户关注但引擎了解不足的话题主动探索。

    数据来源：
      - usr2（当前状态/关注焦点） → extract_focus_keywords()
      - usr5（兴趣图谱）           → extract_hot_topics()
      - usr6（情绪图谱）           → 排除负向触发，避免踩雷

    优先级 20（介于随机漫游 18 和好奇心 15 之间，画像引导比随机更有价值）。
    """
    if portrait_manager is None:
        return None
    try:
        focus_tags = portrait_manager.extract_focus_keywords()
        hot_tags = portrait_manager.extract_hot_topics()
        neg_tags = set(portrait_manager.extract_negative_triggers())

        # 合并候选：关注焦点优先
        candidates = []
        for tag in focus_tags:
            if tag not in neg_tags:
                candidates.append(tag)
        for tag in hot_tags:
            if tag not in neg_tags and tag not in candidates:
                candidates.append(tag)

        if not candidates:
            logger.debug("portrait_curiosity 跳过: 无可用候选标签")
            return None

        # 加权随机：关注焦点（前几个）权重更高
        weights = [1.5 if i < len(focus_tags) else 1.0 for i in range(len(candidates))]
        total_w = sum(weights)
        r = random.uniform(0, total_w)
        cumulative = 0.0
        picked_tag = candidates[-1]
        for tag, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                picked_tag = tag
                break

        # 检查记忆库中该 tag 的覆盖深度：少则探索，多则跳过
        if all_mems:
            tagged_count = sum(
                1 for m in all_mems
                if picked_tag in ((m.get("metadata") or {}).get("tags", "") or "")
            )
            if tagged_count >= 10:
                logger.debug("portrait_curiosity 跳过: tag '%s' 已覆盖 %d 条",
                             picked_tag, tagged_count)
                return None

        return (f"我注意到你最近常提到「{picked_tag}」，想多聊聊这个话题吗？", 20)
    except Exception as exc:
        logger.debug("portrait_curiosity 源异常: %s", exc)
    return None


# ── 调度器 ──────────────────────────────────────────────────
```

---

### 改动 5.2 — 同一文件，`ImpulseScheduler` 类的 `SOURCE_CONFIG`

**位置**：`SOURCE_CONFIG` 列表。

**查找特征码**：
```python
    SOURCE_CONFIG = [
        ("情绪趋势", source_emotion_trend, 600),    # 平均每 10 分钟
        ("时间节律", source_time_rhythm, 1800),     # 平均每 30 分钟
        ("随机漫游", source_random_roam, 600),       # 平均每 10 分钟
        ("好奇心", source_curiosity, 1200),          # 平均每 20 分钟
    ]
```

**替换为**：
```python
    SOURCE_CONFIG = [
        ("情绪趋势", source_emotion_trend, 600),    # 平均每 10 分钟
        ("时间节律", source_time_rhythm, 1800),     # 平均每 30 分钟
        ("随机漫游", source_random_roam, 600),       # 平均每 10 分钟
        ("好奇心", source_curiosity, 1200),          # 平均每 20 分钟
        ("画像探索", source_portrait_curiosity, 900),  # 平均每 15 分钟
    ]
```

---

### 改动 5.3 — 同一文件，`ImpulseScheduler.__init__()`

**位置**：`__init__` 签名和构造函数体。

**查找特征码**：
```python
    def __init__(self, state_path: str, temporal_pattern_index=None):
        self._state_path = state_path
        self._temporal_index = temporal_pattern_index
```

**替换为**：
```python
    def __init__(self, state_path: str, temporal_pattern_index=None,
                 portrait_manager=None):
        self._state_path = state_path
        self._temporal_index = temporal_pattern_index
        self._portrait_manager = portrait_manager
```

---

### 改动 5.4 — 同一文件，`ImpulseScheduler.start_source_workers()`

**位置**：`kwargs_map` 字典构建处。

**查找特征码**：
```python
        kwargs_map = {
            "情绪趋势": {"memory_service": memory_service},
            "时间节律": {"memory_service": memory_service, "temporal_pattern_index": self._temporal_index},
            "随机漫游": {"memory_service": memory_service},
            "好奇心": {"memory_service": memory_service},
        }
```

**替换为**：
```python
        kwargs_map = {
            "情绪趋势": {"memory_service": memory_service},
            "时间节律": {"memory_service": memory_service, "temporal_pattern_index": self._temporal_index},
            "随机漫游": {"memory_service": memory_service},
            "好奇心": {"memory_service": memory_service},
            "画像探索": {"memory_service": memory_service, "portrait_manager": self._portrait_manager},
        }
```

---

### 改动 5.5 — 文件：`app/core/context.py`

**位置**：创建 `ImpulseScheduler` 实例处（第 165 行附近）。

**查找特征码**：
```python
        self.impulse_scheduler = ImpulseScheduler(
            state_path=f"{data_dir}/impulse_state.json",
            temporal_pattern_index=self.temporal_pattern_index,
        )
```

**替换为**：
```python
        self.impulse_scheduler = ImpulseScheduler(
            state_path=f"{data_dir}/impulse_state.json",
            temporal_pattern_index=self.temporal_pattern_index,
            portrait_manager=self.portrait,
        )
```

**验证**：确认 `self.portrait` 在 `context.py` 中先于 `ImpulseScheduler` 创建（第 128 行 `self.portrait = PortraitManager(PORTRAIT_FILE_PATH)` 确实在第 165 行的 ImpulseScheduler 之前）。

---

## 改动清单汇总

| # | 文件 | 改动类型 | 行数估计 |
|---|------|---------|---------|
| 1.1 | `app/core/feedback.py` | 新增函数 | +25 |
| 1.2 | `app/portrait/writer.py` | `realtime_update()` 内插入 | +16 |
| 2.1 | `app/core/circuit.py` | `basal_ganglia_gate()` 内替换 | +15 |
| 3.1 | `app/core/circuit.py` | `weave_context()` 签名 | 改 1 行 |
| 3.2 | `app/core/circuit.py` | `weave_context()` 循环内插入 | +8 |
| 3.3 | `app/core/circuit.py` | 阈值计算行 | 改 1 行 |
| 3.4 | `app/core/circuit.py` | `process()` 内 weave 调用前插入 | +8 |
| 4.1 | `app/core/circuit.py` | `basal_ganglia_gate()` 签名 | 改 1 行 |
| 4.2 | `app/core/circuit.py` | `basal_ganglia_gate()` 调参块后插入 | +7 |
| 4.3 | `app/core/circuit.py` | `process()` 内 gate 调用 | 改 1 行 |
| 5.1 | `app/background/impulse.py` | 新增冲动源函数 | +55 |
| 5.2 | `app/background/impulse.py` | SOURCE_CONFIG 追加 | +1 行 |
| 5.3 | `app/background/impulse.py` | `__init__` 签名 | 改 1 行 + 1 行赋值 |
| 5.4 | `app/background/impulse.py` | kwargs_map 追加 | +1 行 |
| 5.5 | `app/core/context.py` | ImpulseScheduler 构造传参 | +1 行 |
| | **合计** | | **~145 行** |

---

## 测试检查清单

改动完成后，按顺序执行以下验证（每个改动独立验证，不要一次跑全部）：

### 连线①验证

```bash
# 1. 确认 feedback 模块导入无语法错误
py -c "from app.core.feedback import get_recent_corrected_ids; print('OK')"

# 2. 空数据目录下返回空集合
py -c "from app.core.feedback import get_recent_corrected_ids; assert get_recent_corrected_ids('data') == set(); print('OK')"
```

### 连线②验证

```bash
# 3. 确认 circuit 模块可导入
py -c "from app.core.circuit import basal_ganglia_gate; print('OK')"
```

### 连线③验证

```bash
# 4. 确认 weave_context 新签名生效
py -c "from app.core.circuit import CircuitOrchestrator; import inspect; sig = inspect.signature(CircuitOrchestrator.weave_context); assert 'portrait_boost' in sig.parameters; print('OK')"
```

### 连线④验证

```bash
# 5. 确认 basal_ganglia_gate 新签名生效
py -c "from app.core.circuit import basal_ganglia_gate; import inspect; sig = inspect.signature(basal_ganglia_gate); assert 'mirror_prediction' in sig.parameters; print('OK')"
```

### 连线⑤验证

```bash
# 6. 确认新冲动源可导入
py -c "from app.background.impulse import source_portrait_curiosity; print('OK')"

# 7. 确认 SOURCE_CONFIG 含 5 个源
py -c "from app.background.impulse import ImpulseScheduler; assert len(ImpulseScheduler.SOURCE_CONFIG) == 5; print('OK')"
```

### 全量语法检查

```bash
# 8. 所有改动文件语法检查
py -m py_compile app/core/feedback.py && echo "feedback OK"
py -m py_compile app/portrait/writer.py && echo "writer OK"
py -m py_compile app/core/circuit.py && echo "circuit OK"
py -m py_compile app/background/impulse.py && echo "impulse OK"
py -m py_compile app/core/context.py && echo "context OK"
```

### 关键测试（不改动逻辑，只确认已有测试仍然通过）

```bash
# 9. 画像相关测试
py -m pytest tests/test_portrait_writer.py -x -q

# 10. 电路相关测试
py -m pytest tests/test_circuit.py tests/test_circuit_gate.py tests/test_circuit_branches.py -x -q

# 11. 冲动相关测试
py -m pytest tests/test_impulse.py -x -q
```

---

## 执行顺序

严格按 ① → ② → ③ → ④ → ⑤ 顺序执行。每条连线改完立即跑对应验证命令，确认通过后再做下一条。不要一次改完五个再测。

## 注意事项

1. **所有改动都是追加/替换，不要删除任何已有代码行**。
2. **所有新增逻辑都放在 `try/except` 中**，失败时 `pass` 不阻断主链路。
3. **不要修改任何 import 语句**（`from __future__ import annotations` 不要加，本项目不使用）。
4. **缩进严格使用 4 空格**，与现有代码风格一致。
5. 如果查找特征码匹配失败（文件已被改动），停下来报告，不要强行粘贴。
