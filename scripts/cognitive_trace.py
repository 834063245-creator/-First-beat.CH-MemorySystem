# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: e0d48e5e

"""认知追踪器 — 把五条连线在黑盒里的行为全摊开。

用法: python scripts/cognitive_trace.py

造一个真实场景（长期焦虑的程序员），跑完整管线，输出：
  1. 每条连线的输入→处理→输出
  2. 冲突检测（两条线是否同时动了同一个变量）
  3. 冗余检测（是否多条线做同一件事）
  4. 最终 LLM prompt 的影响（开了五连线 vs 不开的 diff）
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ===========================================================
# 工具函数
# ===========================================================

def hr(title: str = ""):
    """打印分隔线"""
    width = 70
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{'='*side} {title} {'='*(width - side - len(title) - 2)}")
    else:
        print("─" * width)

def kv(key: str, value, indent: int = 2):
    """打印键值对"""
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{key}: {value:.3f}")
    elif isinstance(value, list) and len(value) > 10:
        print(f"{prefix}{key}: [{len(value)} items]")
    elif isinstance(value, dict):
        print(f"{prefix}{key}: {json.dumps(value, ensure_ascii=False, default=str)[:120]}")
    else:
        print(f"{prefix}{key}: {value}")

def warn(msg: str):
    print(f"  !!  {msg}")

def good(msg: str):
    print(f"  [OK] {msg}")

def info(msg: str):
    print(f"  -- {msg}")

# ===========================================================
# 场景搭建
# ===========================================================

def build_scenario():
    """构建一个真实场景：长期焦虑的程序员「小陈」"""
    from app.portrait.state import PortraitEntry, EntryStatus
    ACTIVE = EntryStatus.ACTIVE
    from app.portrait.manager import PortraitManager
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="cognitive_trace_")

    # ── 画像：小陈的 12 维画像 ──
    portrait_path = os.path.join(tmpdir, "PORTRAIT.md")
    pm = PortraitManager(portrait_path)

    # usr6: 情绪图谱 — 长期低落
    pm.set_entry("usr6-001", "用户长期工作压力大，频繁提到「好累」「学不动了」",
                 tags=["压力", "疲惫"], confidence=0.75, status=ACTIVE,
                 last_observed="2026-06-18")
    pm.set_entry("usr6-002", "用户对职业发展感到焦虑，担心被 AI 取代",
                 tags=["焦虑", "职业"], confidence=0.65, status=ACTIVE,
                 last_observed="2026-06-15")
    pm.set_entry("usr6-003", "用户最近睡眠不好，深夜还活跃",
                 tags=["失眠", "焦虑"], confidence=0.55, status=ACTIVE,
                 last_observed="2026-06-19")

    # usr5: 兴趣图谱 — 技术栈
    pm.set_entry("usr5-001", "深度关注 Rust 语言，想用它做 side project",
                 tags=["Rust", "编程"], confidence=0.90, status=ACTIVE,
                 last_observed="2026-06-19")
    pm.set_entry("usr5-002", "对 Python 生态很熟悉，日常主力语言",
                 tags=["Python", "后端"], confidence=0.85, status=ACTIVE,
                 last_observed="2026-06-18")
    pm.set_entry("usr5-003", "想转 AI 方向但不确定从哪里开始",
                 tags=["AI", "机器学习", "职业"], confidence=0.70, status=ACTIVE,
                 last_observed="2026-06-12")

    # usr2: 当前状态
    pm.set_entry("usr2-001", "关注焦点: 职业发展方向选择，Rust vs Python vs AI",
                 tags=["Rust", "Python", "AI", "职业"], status=ACTIVE,
                 last_observed="2026-06-20")
    pm.set_entry("usr2-002", "近期活动: 刷 LeetCode，投简历",
                 tags=["面试", "算法"], status=ACTIVE,
                 last_observed="2026-06-19")

    # usr1: 核心特征
    pm.set_entry("usr1-001", "用户是 3 年经验的後端工程师，主要用 Python",
                 tags=["Python", "后端", "工程师"], status=ACTIVE,
                 last_observed="2026-06-10")

    pm.save()

    # ── 错误报告：用户之前纠正过一次错误记忆 ──
    error_path = os.path.join(tmpdir, "error_reports.jsonl")
    from app.core.feedback import log_error_report
    log_error_report("mem-java-001", "我从来没喜欢过Java", "user", data_dir=tmpdir)
    log_error_report("mem-java-002", "记错了，我没有Java项目经验", "user", data_dir=tmpdir)

    # ── 构造记忆候选集（模拟检索管线返回） ──
    now = time.time()
    memories = [
        {"id": "mem-rust-001", "distance": 0.15, "source": "semantic",
         "metadata": {"tags": "Rust,编程", "entities": ["Rust"],
                      "timestamp": now - 86400,
                      "emotion_valence_bin": "positive",
                      "summary": "想用Rust写一个高性能的Web框架"}},
        {"id": "mem-python-001", "distance": 0.22, "source": "semantic",
         "metadata": {"tags": "Python,后端", "entities": ["Python"],
                      "timestamp": now - 172800,
                      "emotion_valence_bin": "positive",
                      "summary": "Python的异步生态越来越好了"}},
        {"id": "mem-job-001", "distance": 0.28, "source": "tag_match",
         "metadata": {"tags": "职业,面试", "entities": ["面试"],
                      "timestamp": now - 43200,
                      "emotion_valence_bin": "negative",
                      "summary": "今天面试又被问Rust，完全不会"}},
        {"id": "mem-ai-001", "distance": 0.35, "source": "co_occurrence",
         "metadata": {"tags": "AI,机器学习", "entities": ["AI"],
                      "timestamp": now - 259200,
                      "emotion_valence_bin": "neutral",
                      "summary": "AI 替代程序员的话题"}},
        {"id": "mem-stress-001", "distance": 0.18, "source": "entity_match",
         "metadata": {"tags": "压力,焦虑", "entities": ["压力"],
                      "timestamp": now - 3600,
                      "emotion_valence_bin": "negative",
                      "summary": "学不动了，太多新技术"}},
        {"id": "mem-java-001", "distance": 0.40, "source": "co_occurrence",
         "metadata": {"tags": "Java", "entities": ["Java"],
                      "timestamp": now - 604800,
                      "emotion_valence_bin": "negative",
                      "stale": True,
                      "summary": "Java太难用了"}},
        {"id": "mem-rust-002", "distance": 0.25, "source": "keyword",
         "metadata": {"tags": "Rust,内存安全", "entities": ["Rust"],
                      "timestamp": now - 300000,
                      "emotion_valence_bin": "positive",
                      "summary": "Rust 的内存安全模型很吸引人"}},
    ]

    return tmpdir, pm, memories, error_path


# ===========================================================
# 追踪器核心
# ===========================================================

@dataclass
class TraceEntry:
    """一条连线的追踪记录"""
    name: str
    input_state: dict = field(default_factory=dict)
    output_state: dict = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_trace():
    tmpdir, pm, memories, error_path = build_scenario()

    traces: list[TraceEntry] = []

    # =======================================================
    hr("场景设定")
    print(f"  用户: 小陈，3年後端工程师")
    print(f"  当前消息: 「最近在学Rust，但Python的工作好找多了，好焦虑」")
    print(f"  画像状态: {sum(1 for e in pm._entries.values() if e.should_inject)} 条活跃 / {len(pm._entries)} 条总计")
    print(f"  检索候选: {len(memories)} 条记忆")
    print(f"  历史纠错: 2 条错误报告 (关于 Java)")
    hr()

    # =======================================================
    hr("连线① 追踪: Feedback → Portrait")
    # =======================================================
    t1 = TraceEntry(name="① Feedback→Portrait")

    from app.core.feedback import get_recent_corrected_ids
    corrected = get_recent_corrected_ids(data_dir=tmpdir)
    t1.input_state = {"error_report_ids": list(corrected), "count": len(corrected)}
    kv("错误报告中的 memory_id", list(corrected))

    # 检查哪些画像条目引用了被纠正的记忆
    affected_entries = []
    for entry_id, entry in list(pm._entries.items()):
        entry_text_and_tags = entry.text + " " + " ".join(entry.tags)
        for mid in corrected:
            if mid in entry_text_and_tags:
                affected_entries.append((entry_id, entry.confidence, entry.status))
                # 模拟纠错
                entry.confidence = max(0.1, entry.confidence - 0.3)
                from app.portrait.state import EntryStatus
                entry.status = EntryStatus.PENDING
                break

    t1.output_state = {"affected": affected_entries}
    t1.changes = [f"{eid}: conf {old:.2f}→{entry.confidence:.2f}, status {old_st}→PENDING"
                  for eid, old, old_st in affected_entries
                  for entry_id, entry in [(eid, pm._entries.get(eid))] if entry]

    if affected_entries:
        for c in t1.changes:
            warn(c)
        info(f"效果: 下次浅巩固时这 {len(affected_entries)} 条会被 LLM 重新评估")
    else:
        good("无画像条目引用被纠正的记忆（正常——纠错的是Java，画像里没有Java条目）")
        info("如果用户纠正过「Python太难」之类的，画像里 usr5 的 Python 条目就会被降权")

    traces.append(t1)

    # =======================================================
    hr("连线② 追踪: Portrait → Gate（画像情绪→门控语气）")
    # =======================================================
    t2 = TraceEntry(name="② Portrait→Gate")

    from app.core.circuit import basal_ganglia_gate, UserMessageAnalysis

    usr6_entries = pm.get_dim_entries("usr6")
    t2.input_state = {
        "usr6_entries": [(e.id, e.text[:40], e.status.value) for e in usr6_entries],
        "usr6_active_count": len([e for e in usr6_entries if e.status.value == "active"]),
    }
    kv("usr6 情绪条目", [(e.id, e.text[:50]) for e in usr6_entries])

    # 门控前：默认语气
    pfc = UserMessageAnalysis(
        intent="casual", emotion="negative", urgency=0.6,
        topics=["Rust", "Python", "职业"], raw_text="最近在学Rust，但Python的工作好找多了，好焦虑",
        confidence=0.75, emotion_intensity=0.7,
    )

    # 不传 portrait 的门控（模拟连线②关闭）
    gate_without = basal_ganglia_gate(pfc, [], [], [], ctx_obj=None)
    kv("无画像时 tone", gate_without.tone)
    kv("无画像时 formality", gate_without.formality)
    kv("无画像时 response_mode", gate_without.response_mode)

    # 传 portrait 的门控（连线②生效）
    class FakeCtx:
        portrait = pm
        class _pattern_discovery:
            @staticmethod
            def get_tuning(): return {}

    gate_with = basal_ganglia_gate(pfc, [], [], [], ctx_obj=FakeCtx())

    t2.output_state = {
        "tone_before": gate_without.tone,
        "tone_after": gate_with.tone,
        "formality_before": gate_without.formality,
        "formality_after": gate_with.formality,
    }

    negative_keywords = ("低落", "焦虑", "沮丧", "压力", "烦躁", "疲惫",
                         "negative", "anxious", "depressed", "frustrated")
    negative_active = sum(
        1 for e in usr6_entries
        if e.status.value == "active" and any(kw in e.text for kw in negative_keywords)
    )
    kv("负面活跃条目数", negative_active)

    if gate_without.tone != gate_with.tone:
        t2.changes.append(f"tone: {gate_without.tone} → {gate_with.tone}")
        warn(f"语气被画像情绪收敛: {gate_without.tone} → {gate_with.tone}")
        info(f"原因: usr6 有 {negative_active} 条负面活跃条目 ≥ 2 条阈值")
    else:
        good(f"语气未变化 ({gate_with.tone})——可能负面条目不足或 intent 是 conflict")

    if gate_without.formality != gate_with.formality:
        t2.changes.append(f"formality: {gate_without.formality:.2f} → {gate_with.formality:.2f}")
        warn(f"正式度被画像拉高: {gate_without.formality:.2f} → {gate_with.formality:.2f}")

    traces.append(t2)

    # =======================================================
    hr("连线③ 追踪: Portrait → weave_context（画像tag→检索阈值）")
    # =======================================================
    t3 = TraceEntry(name="③ Portrait→weave_context")

    boost_map = pm.compute_portrait_boost_map()
    t3.input_state = {"boost_map": boost_map, "boosted_tags": list(boost_map.keys())}
    kv("boost map", boost_map)

    from app.core.circuit import CircuitOrchestrator

    orch = CircuitOrchestrator(
        memory_service=None, impulse_scheduler=None,
        dmn_engine=None, chat_history=None,
        co_tracker=None, mirror_neuron=None,
    )

    class FakeCognitive:
        intent = "ask_fact"

    # 无 boost 的编织
    wc_no_boost = orch.weave_context(memories, FakeCognitive())

    # 有 boost 的编织
    wc_boosted = orch.weave_context(memories, FakeCognitive(), portrait_boost=boost_map)

    t3.output_state = {
        "fact_count_without": len(wc_no_boost.fact_memories),
        "fact_count_with": len(wc_boosted.fact_memories),
        "ref_count_without": len(wc_no_boost.reference_memories),
        "ref_count_with": len(wc_boosted.reference_memories),
    }

    # 逐条对比
    fact_ids_no = {m["id"] for m in wc_no_boost.fact_memories}
    fact_ids_with = {m["id"] for m in wc_boosted.fact_memories}
    promoted = fact_ids_with - fact_ids_no
    demoted = fact_ids_no - fact_ids_with

    if promoted:
        for mid in promoted:
            m = next((x for x in memories if x["id"] == mid), None)
            tags = (m.get("metadata", {}).get("tags", "") if m else "")
            boost_hits = {t: boost_map.get(t, 0) for t in (tags.split(",") if isinstance(tags, str) else tags)}
            t3.changes.append(f"晋升 fact: {mid} (tags: {tags}, boost命中: {boost_hits})")
            good(f"晋升 fact: {mid} — tags '{tags}' 被画像 boost 命中 → 阈值放宽")
    if demoted:
        for mid in demoted:
            t3.changes.append(f"降级: {mid}")
            warn(f"降级: {mid}（不应发生——有 boost 只有晋升没有降级）")
    if not promoted and not demoted:
        info("本轮无阈值跨越——boost 值不足以改变分层结果")

    traces.append(t3)

    # =======================================================
    hr("连线④ 追踪: Predictor → Gate（行为预测→预调模式）")
    # =======================================================
    t4 = TraceEntry(name="④ Predictor→Gate")

    # 当前用户消息的情绪+意图
    kv("当前 intent", pfc.intent)
    kv("当前 emotion", pfc.emotion)

    # 场景A: 用户焦虑地问 Rust——预测可能是 ask_fact
    pred_a = {"predicted_intent": "ask_fact"}
    gate_a = basal_ganglia_gate(pfc, [], [], [], ctx_obj=FakeCtx(),
                                mirror_prediction=pred_a)
    t4.input_state["prediction_a"] = pred_a
    kv("预测 ask_fact 时 response_mode", gate_a.response_mode)
    if gate_a.response_mode != gate_with.response_mode:
        t4.changes.append(f"response_mode: {gate_with.response_mode} -> {gate_a.response_mode}")
        good(f"预测 ask_fact → response_mode 从 {gate_with.response_mode} 变为 {gate_a.response_mode}")

    # 场景B: 用户情绪化——预测 emotional_sharing
    pred_b = {"predicted_intent": "emotional_sharing"}
    gate_b = basal_ganglia_gate(pfc, [], [], [], ctx_obj=FakeCtx(),
                                mirror_prediction=pred_b)
    t4.input_state["prediction_b"] = pred_b
    kv("预测 emotional_sharing 时 tone", gate_b.tone)
    if gate_b.tone != gate_with.tone:
        t4.changes.append(f"tone: {gate_with.tone} -> {gate_b.tone}")
        good(f"预测 emotional_sharing → tone 从 {gate_with.tone} 变为 {gate_b.tone}")
    else:
        info(f"预测 emotional_sharing 未改变 tone (当前={gate_b.tone})——②已将其收敛为 soft，④的 caring 分支需 tone==warm 才触发")

    # 冲突检测：连线② vs 连线④
    hr("  +== 冲突分析: 连线② vs 连线④ ==+")

    # ② 设了 tone=soft（因为 usr6 负面 ≥2）
    # ④ 在 emotional_sharing + warm 时会设 tone=caring
    # 但 ② 已经把 tone 改成 soft 了，不是 warm，所以 ④ 的 caring 不会触发
    tone_after_2 = gate_with.tone  # soft (由②设置)
    if tone_after_2 == "soft":
        info("② 先执行，tone=soft → ④ 的 emotional_sharing 检查 'tone==\"warm\"' 失败 → ④ 不覆盖")
        info("结论: ② 优先级高于 ④ 的 caring 分支（这是合理的——长期低落时不应强行 caring）")
    else:
        warn("② 和 ④ 可能同时修改 tone，需要检查执行顺序")
    print("  +==============================+")

    traces.append(t4)

    # =======================================================
    hr("连线⑤ 追踪: Portrait → Impulse（画像驱动好奇心）")
    # =======================================================
    t5 = TraceEntry(name="⑤ Portrait→Impulse")

    from app.background.impulse import source_portrait_curiosity

    focus = pm.extract_focus_keywords()
    hot = pm.extract_hot_topics()
    neg = pm.extract_negative_triggers()
    t5.input_state = {
        "focus_keywords": focus, "hot_topics": hot, "negative_triggers": neg,
    }
    kv("关注焦点 (usr2)", focus)
    kv("热点话题 (usr5)", hot)
    kv("负向触发 (usr6)", neg)

    # 候选去重去负
    candidates = []
    for tag in focus:
        if tag not in neg: candidates.append(tag)
    for tag in hot:
        if tag not in neg and tag not in candidates: candidates.append(tag)

    kv("可用候选标签", candidates)
    kv("被排除的负面标签", [t for t in focus + hot if t in neg])

    # 模拟当前记忆库的覆盖情况
    all_mems_sim = [
        {"metadata": {"tags": "Rust,编程"}},
        {"metadata": {"tags": "Rust,内存安全"}},
        {"metadata": {"tags": "Python,后端"}},
        {"metadata": {"tags": "Python,Django"}},
        {"metadata": {"tags": "AI,机器学习"}},
    ]
    for tag in candidates:
        count = sum(1 for m in all_mems_sim
                    if tag in ((m.get("metadata") or {}).get("tags", "") or ""))
        status = "XX 覆盖充足(≥10)" if count >= 10 else f">> 覆盖不足({count}/10)"
        kv(f"  {tag} 覆盖", f"{count} 条 → {status}")

    result = source_portrait_curiosity(portrait_manager=pm, all_mems=all_mems_sim)
    t5.output_state = {"impulse_result": result}

    if result:
        content, priority = result
        t5.changes.append(f"产出冲动: priority={priority}, content='{content}'")
        good(f"画像探索产出: 「{content}」(优先级 {priority})")
    else:
        info("本轮无画像探索冲动产出（覆盖充足或候选枯竭）")

    traces.append(t5)

    # =======================================================
    hr("全景总结")
    # =======================================================

    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  连线   │  触发?  │  做了什么                                 │")
    print("  |─────────────────────────────────────────────────────────────|")
    for t in traces:
        triggered = "[OK]" if t.changes else "—"
        action = t.changes[0][:52] if t.changes else "(本轮无效果)"
        print(f"  │  {t.name:20s} │  {triggered:4s}  │  {action:52s} │")
    print("  └─────────────────────────────────────────────────────────────┘")

    # 冲突摘要
    print()
    hr("冲突/重叠分析")
    conflicts = []

    # ② vs ④: 都修改 tone
    t2_tone_changed = any("tone" in c for c in t2.changes)
    t4_tone_changed = any("tone" in c for c in t4.changes)
    if t2_tone_changed and t4_tone_changed:
        conflicts.append("!!  连线②和④都修改了 tone——②先执行④后执行，④可能覆盖②或反之。需确认预期行为。")
    elif t2_tone_changed:
        conflicts.append("ii  仅连线②修改了 tone，④未触发（正常——②的语气收敛后④不再匹配 warm）")
    elif t4_tone_changed:
        conflicts.append("ii  仅连线④修改了 tone，②未触发")

    # ③ vs ①: ①降了 confidence 的条目，③是否还在 boost？
    t1_affected = [c.split(":")[0] for c in t1.changes]
    for tag, boost in boost_map.items():
        # 检查这个 tag 对应的条目是否被①降权了
        for eid, entry in pm._entries.items():
            if tag in entry.tags and entry.status.value == "pending":
                conflicts.append(f"!!  标签 '{tag}' 被③ boost +{boost}，但关联条目 {eid} 已被①标为 PENDING (conf={entry.confidence:.2f})。boost 应随 confidence 降低")

    # ⑤ vs ①: ①降权的 topic ⑤还在探索？
    for c in t5.changes:
        for tag in candidates:
            for eid, entry in pm._entries.items():
                if tag in entry.tags and entry.status.value == "pending":
                    conflicts.append(f"!!  连线⑤可能探索 '{tag}'，但其关联条目 {eid} 已被①降权——可能探索了已被用户否定的话题")

    if not conflicts:
        good("本轮无检测到冲突")
    else:
        for c in conflicts:
            warn(c)

    # LLM Prompt 影响估算
    hr("LLM Prompt 影响估算")
    print()
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print("  │  message[0] (system):                                        │")
    if t2_tone_changed:
        print("  │    ← ②: 画像情绪趋势 → tone=soft, formality≥0.4               │")
    if t3.changes:
        print(f"  │    ← ③: {len(promoted)} 条记忆因画像 boost 进入 fact 层        │")
    print("  │                                                              │")
    print("  │  message[N+1] (dynamic portrait):                            │")
    if t1.changes:
        print(f"  │    ← ①: {len(t1.changes)} 条画像条目标 PENDING，下次浅巩固重评  │")
    print("  │                                                              │")
    print("  │  message[N+4..N+5] (natural_thoughts):                       │")
    if t5.changes:
        content = t5.changes[0].split("'")[1] if "'" in t5.changes[0] else "?"
        print(f"  │    ← ⑤: 画像探索冲动注入: 「{content}」                        │")
    print("  │                                                              │")
    print("  │  message[last] (user):                                       │")
    if t4.changes:
        print(f"  │    ← ④: 行为预测预调 → response_mode 已优化                   │")
    print("  │                                                              │")
    print(f"  │  最终 LLM 看到的语气: tone={gate_with.tone},                   │")
    print(f"  │                      formality={gate_with.formality:.2f},                │")
    print(f"  │                      mode={gate_with.response_mode}                  │")
    print(f"  │  传递给 LLM 的 fact 记忆: {len(wc_boosted.fact_memories)} 条                   │")
    print("  └──────────────────────────────────────────────────────────────┘")

    # 清理
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return traces, conflicts


if __name__ == "__main__":
    run_trace()
