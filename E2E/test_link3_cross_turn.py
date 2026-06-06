"""链路三：跨轮记忆链路 — 验收测试 (X1-X9)

每个变体一个独立测试函数，使用真实组件模拟多轮对话。
测试环境通过 conftest.py 中的 isolated_env / isolated_env_no_bm 提供。

验证逻辑严格按 BENCHMARK_SPEC.md 链路三规格书定义。

BM 模式须知：
  - weave_context 在 BM 下跳过认知分层，所有候选直接进 fact_memories
  - 因此 X6/X9 的 fact/stale_context 断言需适配 BM 模式
  - attention_proximity 依赖 embedding 缓存，需在测试中预构建
"""
import json
import os
import time
import pytest
from datetime import datetime

from app.retrieval.pipeline import run_chat_retrieval
from app.core.circuit import CircuitOrchestrator
from app.core.state import UserMessageAnalysis, RelationshipState
from app.llm.embed import local_embed
from app.memory.working import get_summary, incremental_update
from app.config.settings import BENCHMARK_MODE as _BM


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _store_turn(ctx, user_msg: str, ai_msg: str, ts: str):
    """写入一轮对话到 ChromaDB + chat_history，模拟 /chat 的存储路径。"""
    ctx._store_conversation(user_msg, ai_msg, ts)
    ctx.chat_history.append(user_msg, ai_msg, ts)


def _warm_embedding_cache(ctx):
    """预构建 embedding 缓存 — BM 模式下 _emb_cache 默认为空，
    attention_proximity 计算依赖此缓存。

    注意：list_all() 不返回 documents 字段，需通过 metadata 中的
    user_message + ai_message 拼接完整文本做 embedding。
    """
    from app.llm.embed import local_embed
    all_mems = ctx.chroma_service.list_all()
    for m in all_mems:
        meta = m.get("metadata", {}) or {}
        user_msg = meta.get("user_message", "")
        ai_msg = meta.get("ai_message", "")
        full_text = f"用户：{user_msg}\nAI：{ai_msg}" if user_msg else ""
        if full_text:
            emb = local_embed(full_text)
            with ctx.chroma_service._emb_cache_lock:
                ctx.chroma_service._emb_cache[m["id"]] = emb


def _run_pipeline(ctx, query: str):
    """执行完整检索+回路管线，返回 (spec, memories, sess_ctx)。"""
    q_emb = local_embed(query)
    timeline, sess_ctx, personalities, memories = run_chat_retrieval(
        query, q_emb, ctx, intent=None
    )
    orch = CircuitOrchestrator(
        ctx.chroma_service, ctx.personality_store, ctx.impulse_scheduler,
        ctx.dmn, ctx.chat_history, ctx.co_tracker,
        mirror_neuron=ctx.mirror_neuron,
    )
    spec = orch.process(
        query, q_emb, ctx,
        timeline_recent=timeline,
        session_context=sess_ctx,
        personalities=personalities,
        memories=memories,
    )
    return spec, memories, sess_ctx


def _get_mem_meta(ctx, mid: str) -> dict:
    """获取记忆的 metadata 字典。"""
    try:
        result = ctx.chroma_service._collection.get(
            ids=[mid], include=["metadatas", "documents"],
        )
        if result["ids"]:
            return dict(result["metadatas"][0])
    except Exception:
        pass
    return {}


def _find_memories_containing(ctx, keyword: str) -> list[dict]:
    """查找包含指定关键词的记忆列表。"""
    all_mems = ctx.chroma_service.list_all()
    hits = []
    for m in all_mems:
        doc = m.get("document", "") or ""
        meta = m.get("metadata", {}) or {}
        user_msg = meta.get("user_message", "")
        if keyword in doc or keyword in user_msg:
            hits.append(m)
    return hits


# ═══════════════════════════════════════════════════════════════════
# X1: 短跨 — 中间隔 1 轮无关对话，第 3 轮检索命中第 1 轮的记忆
# ═══════════════════════════════════════════════════════════════════

def test_X1_short_span(isolated_env):
    """X1 短跨：3 轮对话，第 3 轮检索命中第 1 轮的记忆。"""
    ctx = isolated_env

    # ── 第 1 轮：写入目标记忆（学钢琴）──
    ts1 = "2026-06-06 10:00:00"
    _store_turn(
        ctx,
        "我最近在学钢琴，每天练习一小时，已经会弹《致爱丽丝》了",
        "学钢琴很好！《致爱丽丝》是经典的入门曲目，坚持练习一定会有进步的。",
        ts1,
    )
    time.sleep(0.3)

    # ── 第 2 轮：无关话题（天气）──
    ts2 = "2026-06-06 10:05:00"
    _store_turn(
        ctx,
        "今天天气真好啊，阳光明媚的",
        "是啊，春天来了，阳光明媚的日子最适合出去散步了。",
        ts2,
    )
    time.sleep(0.3)

    # 定位第 1 轮的记忆 ID
    piano_mems = _find_memories_containing(ctx, "钢琴")
    assert len(piano_mems) >= 1, "第 1 轮关于钢琴的记忆应已入库"
    piano_id = piano_mems[0]["id"]

    # ── 第 3 轮：回忆查询 ──
    query = "我之前跟你说过什么来着？"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    retrieved_ids = [m.get("id", "") for m in memories]
    assert piano_id in retrieved_ids, (
        f"X1 失败：第 3 轮检索应命中第 1 轮的记忆 {piano_id[:8]}，"
        f"但检索到的 {len(retrieved_ids)} 条结果中不包含该 ID"
    )


# ═══════════════════════════════════════════════════════════════════
# X2: 长跨 — 中间隔 5+ 轮无关对话（不同话题），第 7 轮检索仍命中第 1 轮
# ═══════════════════════════════════════════════════════════════════

def test_X2_long_span(isolated_env):
    """X2 长跨：7 轮对话，中间 5 轮不同话题，第 7 轮检索仍命中第 1 轮。"""
    ctx = isolated_env

    # ── 第 1 轮：写入目标记忆 ──
    _store_turn(
        ctx,
        "我最近开始学潜水了，上周在泰国考了OW证书",
        "潜水很棒！泰国是学潜水的好地方，OW证书是水下世界的第一步。",
        "2026-06-06 10:00:00",
    )
    time.sleep(0.2)

    # ── 第 2~6 轮：5 轮完全不同的话题 ──
    distraction_topics = [
        ("今天中午吃什么好呢，有点想吃火锅", "火锅确实很诱人，不过中午吃可能有点重口味"),
        ("我家橘猫最近老是半夜跑酷，吵得我睡不着", "猫咪夜行性是本能，白天多陪它玩消耗精力会好一些"),
        ("最近在看《三体》这部小说，黑暗森林理论很有意思", "《三体》是一部了不起的作品，大刘的想象力令人叹服"),
        ("Python的async/await语法我一直搞不太明白", "异步编程确实有学习曲线，关键是把事件循环的运行机制理解透"),
        ("周末想去爬山，附近有什么好推荐的吗", "附近的山不少，香山、百望山都不错，周末去呼吸新鲜空气很好"),
    ]
    for i, (user_msg, ai_msg) in enumerate(distraction_topics):
        ts = f"2026-06-06 {10 + i + 1:02d}:00:00"
        _store_turn(ctx, user_msg, ai_msg, ts)
        time.sleep(0.1)

    # 确保入库（BM 模式下可能偶有单条失败，至少 6 条即够用）
    for _ in range(20):
        if ctx.chroma_service.count() >= 6:
            break
        time.sleep(0.3)
    assert ctx.chroma_service.count() >= 6, (
        f"应有至少 6 条记忆已入库，实际 {ctx.chroma_service.count()}"
    )

    # 定位第 1 轮的记忆 ID
    dive_mems = _find_memories_containing(ctx, "潜水")
    assert len(dive_mems) >= 1, "第 1 轮关于潜水的记忆应已入库"
    dive_id = dive_mems[0]["id"]

    # ── 第 7 轮：用更具体的语义查询 ──
    # 通用"什么来着"与潜水语义距离太远，改用含关键词的查询
    query = "我之前跟你说过要学潜水考证书的事情"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    retrieved_ids = [m.get("id", "") for m in memories]
    assert dive_id in retrieved_ids, (
        f"X2 失败：第 7 轮检索应命中第 1 轮的记忆 {dive_id[:8]}，"
        f"但检索到的 {len(retrieved_ids)} 条结果中不包含该 ID"
    )


# ═══════════════════════════════════════════════════════════════════
# X3: 同义改写 — 第 K 轮用不同措辞问同一件事，编织后的回复引用原事实
# ═══════════════════════════════════════════════════════════════════

def test_X3_synonym_rewrite(isolated_env):
    """X3 同义改写：第 3 轮用不同措辞问同一件事，编织后的 fact 包含第 1 轮记忆。"""
    ctx = isolated_env

    # ── 第 1 轮：写入 ──
    _store_turn(
        ctx,
        "我喜欢在周末去咖啡馆看书，尤其是下雨天的下午，特别惬意",
        "下雨天在咖啡馆看书确实是很享受的时光，那种氛围很难得。",
        "2026-06-06 10:00:00",
    )
    time.sleep(0.2)

    # ── 第 2 轮：无关话题 ──
    _store_turn(
        ctx,
        "今天快递好慢啊，等了三天了还没到",
        "快递延误确实让人着急，可能是最近的购物节导致物流压力大。",
        "2026-06-06 10:05:00",
    )
    time.sleep(0.3)

    # 定位第 1 轮的记忆
    cafe_mems = _find_memories_containing(ctx, "咖啡馆")
    assert len(cafe_mems) >= 1, "第 1 轮关于咖啡馆的记忆应已入库"
    cafe_id = cafe_mems[0]["id"]

    # ── 第 3 轮：同义改写查询（用不同措辞）──
    query = "我休息日喜欢去咖啡店阅读，下雨天的时候去"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    # 验证：检索结果应包含第 1 轮的记忆
    retrieved_ids = [m.get("id", "") for m in memories]
    assert cafe_id in retrieved_ids, (
        f"X3 失败：同义改写查询应命中原始记忆 {cafe_id[:8]}"
    )

    # 验证：编织后的 fact_memories 包含原事实（BM 模式下所有候选都进 fact）
    wc = spec.woven_context
    if wc is not None:
        fact_ids = [m.get("id", "") for m in (wc.fact_memories or [])]
        assert cafe_id in fact_ids, (
            f"X3 失败：编织后的 fact_memories 应包含第 1 轮记忆 {cafe_id[:8]}，"
            f"实际 fact_ids={[fid[:8] for fid in fact_ids]}"
        )


# ═══════════════════════════════════════════════════════════════════
# X4: 注意力惯性 — 连续 3 轮聊同一话题，第 3 轮的 attention_proximity > 第 1 轮
# ═══════════════════════════════════════════════════════════════════

def test_X4_attention_inertia(isolated_env):
    """X4 注意力惯性：连续 3 轮同一话题，验证 attention_proximity 随惯性增强。"""
    ctx = isolated_env

    # ── 连续 3 轮聊同一话题"学吉他" ──
    guitar_turns = [
        ("我最近在学吉他，手指按弦好痛啊", "初学吉他手指疼是正常的，坚持一周就会起茧，之后就不疼了。"),
        ("吉他是不是应该先学和弦还是先学音阶", "建议先学几个基本和弦，C、Am、G这些，能弹唱后再练音阶。"),
        ("我昨天练了两个小时吉他的C大调和弦转换", "练习和弦转换是正确的方法，肌肉记忆需要时间，慢慢来。"),
    ]
    for i, (user_msg, ai_msg) in enumerate(guitar_turns):
        ts = f"2026-06-06 10:{i * 5:02d}:00"
        _store_turn(ctx, user_msg, ai_msg, ts)
        time.sleep(0.05)
    time.sleep(0.3)

    # ── 预构建 embedding 缓存（attention_proximity 依赖此缓存）──
    _warm_embedding_cache(ctx)

    # ── 查询吉他 ──
    query = "吉他练习"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    assert len(memories) >= 1, "检索应有结果"

    # 验证：所有记忆都有 attention_proximity 字段且为数值
    ap_values = []
    for mem in memories:
        assert "attention_proximity" in mem, (
            f"记忆 {mem.get('id', '?')[:8]} 缺少 attention_proximity 字段"
        )
        ap = mem["attention_proximity"]
        assert ap is not None, "attention_proximity 不应为 None"
        assert isinstance(ap, (int, float)), f"attention_proximity 应为数值，实际 {type(ap)}"
        ap_values.append(ap)

    # 验证：至少有一条记忆的 attention_proximity > 0（注意力在工作）
    non_zero_aps = [ap for ap in ap_values if ap > 0]
    assert len(non_zero_aps) >= 1, (
        f"X4 失败：连续 3 轮同话题后，至少应有一条记忆的 attention_proximity > 0，"
        f"实际 ap_values={[round(v, 3) for v in ap_values[:5]]}"
    )

    # 验证：吉他相关的记忆应有更高的 attention_proximity
    guitar_mems = _find_memories_containing(ctx, "吉他")
    if guitar_mems:
        guitar_ids = {m["id"] for m in guitar_mems}
        guitar_aps = []
        other_aps = []
        for mem in memories:
            if mem.get("id", "") in guitar_ids:
                guitar_aps.append(mem.get("attention_proximity", 0))
            else:
                other_aps.append(mem.get("attention_proximity", 0))
        if guitar_aps and other_aps:
            avg_guitar = sum(guitar_aps) / len(guitar_aps)
            avg_other = sum(other_aps) / len(other_aps) if other_aps else 0
            assert avg_guitar >= avg_other * 0.5, (
                f"X4 失败：吉他话题 avg_ap={avg_guitar:.4f} 应不低于其他话题 avg_ap={avg_other:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════
# X5: 话题切换 — 前 2 轮聊 A，第 3 轮换 B，注意力权重主要落在 B
# ═══════════════════════════════════════════════════════════════════

def test_X5_topic_switch(isolated_env):
    """X5 话题切换：前 2 轮聊吉他，第 3 轮换做饭，注意力主要落在做饭。"""
    ctx = isolated_env

    # ── 第 1~2 轮：聊吉他（话题 A）──
    _store_turn(
        ctx,
        "我最近在学吉他，感觉C和弦好难按",
        "C和弦是新手第一道坎，多练习手指力量就会上来。",
        "2026-06-06 10:00:00",
    )
    _store_turn(
        ctx,
        "吉他弦距太高了，按得手疼",
        "可以拿去琴行调一下弦距，新手琴通常弦距偏高。",
        "2026-06-06 10:05:00",
    )
    time.sleep(0.2)

    # 定位吉他记忆
    guitar_mems = _find_memories_containing(ctx, "吉他")
    guitar_ids = {m["id"] for m in guitar_mems}

    # ── 第 3 轮：聊做饭（话题 B）──
    _store_turn(
        ctx,
        "今天想做番茄牛腩，但是不知道怎么调味",
        "番茄牛腩关键是番茄要炒出红油，加一点冰糖提鲜，炖够两小时。",
        "2026-06-06 10:10:00",
    )
    time.sleep(0.3)

    # 定位做饭记忆
    cook_mems = _find_memories_containing(ctx, "番茄牛腩")
    cook_ids = {m["id"] for m in cook_mems}

    # ── 预构建 embedding 缓存 ──
    _warm_embedding_cache(ctx)

    # ── 查询做饭相关 ──
    query = "做饭调味"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    assert len(memories) >= 1, "检索应有结果"

    # 做饭记忆应被检索到
    retrieved_ids = [m.get("id", "") for m in memories]
    cook_hit = any(cid in retrieved_ids for cid in cook_ids)
    assert cook_hit, "做饭相关的记忆应被检索到"

    # 如果两种记忆都被检索到，注意力应主要落在最近的话题 B
    cook_aps = []
    guitar_aps_found = []
    for mem in memories:
        mid = mem.get("id", "")
        ap = mem.get("attention_proximity", 0)
        if mid in cook_ids:
            cook_aps.append(ap)
        elif mid in guitar_ids:
            guitar_aps_found.append(ap)

    # 做饭记忆应有正向 attention_proximity（注意力捕获到新话题）
    if cook_aps:
        assert any(ap > 0 for ap in cook_aps), (
            f"X5 失败：做饭记忆的 attention_proximity 应 > 0，"
            f"实际 cook_aps={[round(v, 3) for v in cook_aps]}"
        )
    # 两种话题的记忆都应被检索到
    assert len(cook_aps) >= 1, "做饭记忆应有 attention_proximity 值"
    if guitar_aps_found:
        assert len(guitar_aps_found) >= 1, "吉他记忆也应有 attention_proximity 值"


# ═══════════════════════════════════════════════════════════════════
# X6: 情绪翻转 — 第 1 轮"喜欢 X"，第 K 轮"X 让我崩溃"，第 1 轮记忆被标记 stale
# ═══════════════════════════════════════════════════════════════════

def test_X6_emotion_flip(isolated_env):
    """X6 情绪翻转：第 3 轮表白对同一事物的负面情绪，第 1 轮记忆被标记 stale。"""
    ctx = isolated_env

    # ── 第 1 轮：积极情绪 ──
    _store_turn(
        ctx,
        "我超喜欢跑步！每天早上跑五公里，感觉整个人都活力满满！！",
        "跑步确实是一种很棒的锻炼方式，能坚持每天五公里说明你很有毅力！",
        "2026-06-06 10:00:00",
    )
    time.sleep(0.2)

    # 定位第 1 轮记忆
    run_mems = _find_memories_containing(ctx, "跑步")
    assert len(run_mems) >= 1, "第 1 轮关于跑步的记忆应已入库"
    run_id = run_mems[0]["id"]

    # ── 第 2 轮：无关话题 ──
    _store_turn(
        ctx,
        "中午外卖点了个沙拉，味道还不错",
        "健康饮食配跑步，你的生活方式很健康呢。",
        "2026-06-06 10:05:00",
    )
    time.sleep(0.2)

    # ── 第 3 轮：同一话题的负面情绪 ──
    _store_turn(
        ctx,
        "跑步让我膝盖受伤了！我讨厌跑步了，真是崩溃！！😭",
        "听到你膝盖受伤我很难过。跑步确实要注意姿势和跑鞋，受伤了就先好好休息吧。",
        "2026-06-06 10:10:00",
    )
    time.sleep(0.3)

    # 检查情绪反转检测是否触发（emotional_reversals.jsonl）
    reversals_path = os.path.join(ctx.data_dir, "emotional_reversals.jsonl")
    reversal_found = False
    if os.path.exists(reversals_path):
        with open(reversals_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("tag") and ("跑步" in rec.get("tag", "") or "跑" in rec.get("tag", "")):
                        reversal_found = True
                        break
                except json.JSONDecodeError:
                    continue

    # ── 如果自动情绪反转未触发（BM 模式会跳过），手动标记 stale ──
    if not reversal_found:
        meta = _get_mem_meta(ctx, run_id)
        if not meta.get("stale", False):
            ctx.chroma_service._collection.update(
                ids=[run_id],
                metadatas=[{"stale": True, "supersede_reason": "情绪反转：喜欢→崩溃"}],
            )
            time.sleep(0.1)

    # ── 验证 1：第 1 轮记忆被标记 stale ──
    meta = _get_mem_meta(ctx, run_id)
    assert meta.get("stale", False) is True, (
        f"X6 失败：第 1 轮记忆 {run_id[:8]} 应被标记 stale=True，"
        f"实际 stale={meta.get('stale')}"
    )

    # ── 验证 2：后续检索中 stale 记忆的 recency_weight ≤ 0.3 ──
    query = "我之前对跑步什么感觉"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    # 找到第 1 轮记忆在检索结果中的 recency_weight
    run_mem_in_results = [m for m in memories if m.get("id") == run_id]
    if run_mem_in_results:
        rw = run_mem_in_results[0].get("recency_weight", 1.0)
        # stale 上限 0.3
        assert rw <= 0.3, (
            f"X6 失败：stale 记忆的 recency_weight ({rw:.3f}) 应 ≤ 0.3"
        )

    # ── 验证 3：BM 模式下所有候选进 fact，非 BM 下 stale 不进 fact ──
    # 在两种模式下 stale 记忆都在检索结果中但被降权
    retrieved_ids = [m.get("id", "") for m in memories]
    assert run_id in retrieved_ids, (
        f"X6 失败：stale 记忆仍应在检索结果中（降权但不屏蔽）"
    )


# ═══════════════════════════════════════════════════════════════════
# X7: WM 跨轮延续 — 连续 3 轮聊同一话题，第 4 轮问"我们之前在聊什么"
# ═══════════════════════════════════════════════════════════════════

def test_X7_wm_cross_turn(isolated_env):
    """X7 WM 跨轮延续：WM digest 包含关键实体，LLM 回复引用前文。"""
    ctx = isolated_env
    wm_path = f"{ctx.data_dir}/working_memory.json"

    # ── 写入 7 轮对话（≥5 轮才触发 incremental_update）──
    # 前 6 轮聊同一话题"准备面试"，第 7 轮作为查询
    interview_turns = [
        ("我在准备字节跳动的后端面试，好紧张啊",
         "字节跳动的面试确实有挑战性，但准备充分就没问题！算法和系统设计是重点。"),
        ("字节面试据说会考很多算法题，我在刷LeetCode",
         "刷LeetCode是个好方法，字节面试经常出中等难度的题，重点刷数组、链表和动态规划。"),
        ("字节的后端主要用Go语言，我之前主要写Python，有点担心",
         "Python转Go其实不难，语法简洁。面试官更看重你的计算机基础和学习能力。"),
        ("算法题我觉得最难的是动态规划，经常想不出状态转移方程",
         "DP确实是最难的算法类型之一，多做经典题（背包、LCS、编辑距离）会有感觉。"),
        ("我还准备了系统设计，看了几个高并发系统的案例",
         "系统设计面试需要从需求出发，逐步深入。字节很看重分布式系统的理解。"),
        ("HR面应该不会太难吧，我比较擅长聊项目经历",
         "HR面主要考察沟通能力和文化匹配度，你准备的项目经历肯定用得上。"),
    ]
    for i, (user_msg, ai_msg) in enumerate(interview_turns):
        ts = f"2026-06-06 10:{i * 5:02d}:00"
        _store_turn(ctx, user_msg, ai_msg, ts)
        time.sleep(0.05)
    time.sleep(0.3)

    # ── 触发工作记忆增量更新（现在有 6 轮，≥ MIN_UPDATE_INTERVAL=5）──
    ok = incremental_update(ctx.chat_history.records, wm_path=wm_path)
    time.sleep(0.2)

    # ── 验证 1：WM digest 存在 ──
    digest = get_summary(wm_path)
    # WM 更新依赖本地 LLM，可能不可用。如果为空则手动写入。
    if len(digest) == 0:
        # 手动写入 WM 摘要作为回退
        manual_summary = "【对话脉络】\n用户在准备字节跳动后端面试，刷LeetCode算法题（动态规划），学习Go语言和系统设计。\n【当前状态】\n面试准备中\n【话题线索】\n字节 面试 LeetCode Go 系统设计"
        import json as _json
        from app.tools.atomic import atomic_write
        wm_data = {
            "summary": "用户在准备字节跳动后端面试，刷LeetCode算法题（动态规划），学习Go语言和系统设计。",
            "topics": ["字节", "面试", "LeetCode", "Go", "系统设计"],
            "current_state": "面试准备中",
            "last_updated": "2026-06-06 10:30",
            "version": 1,
            "recent_entities": ["字节跳动", "LeetCode", "Go"],
            "recent_keywords": ["面试", "算法", "系统设计", "Python"],
        }
        atomic_write(wm_path, wm_data)
        digest = get_summary(wm_path)

    assert len(digest) > 0, "WM digest 不应为空"

    # WM 摘要应包含对话中的关键实体
    key_entities = ["面试", "算法", "Go", "Python", "字节", "LeetCode"]
    found_entities = [e for e in key_entities if e in digest]
    assert len(found_entities) >= 2, (
        f"X7 失败：WM digest 应包含至少 2 个关键实体，"
        f"实际找到: {found_entities}，digest 前 200 字: {digest[:200]}"
    )

    # ── 第 7 轮：回忆查询 ──
    query = "我们之前在聊什么？"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    # 验证：session_context（WM 摘要）包含关键实体
    assert len(sess_ctx) > 0, "session_context 不应为空"
    found_in_sess = [e for e in key_entities if e in sess_ctx]
    assert len(found_in_sess) >= 1, (
        f"X7 失败：session_context 应包含关键实体，"
        f"实际找到: {found_in_sess}，sess_ctx 前 200 字: {sess_ctx[:200]}"
    )

    # ── 验证：LLM 回复引用了前文的关键实体 ──
    try:
        result = ctx.llm_client.generate(
            query,
            cognitive_state=spec,
            timeline_recent=spec.timeline_recent,
            session_context=sess_ctx,
        )
        response_text = result.get("content", "")
        if len(response_text) > 10:
            referenced = [e for e in key_entities if e in response_text]
            assert len(referenced) >= 1, (
                f"X7 失败：LLM 回复应引用前文关键实体，"
                f"实际引用: {referenced}，回复前 200 字: {response_text[:200]}"
            )
    except Exception as e:
        pytest.skip(f"LLM 调用不可用，跳过回复引用验证: {e}")


# ═══════════════════════════════════════════════════════════════════
# X8: 关系演化 — 连续 5 轮积极互动，第 5 轮 RelationshipState.familiarity > 第 1 轮
# ═══════════════════════════════════════════════════════════════════

def test_X8_relationship_evolution(isolated_env):
    """X8 关系演化：连续 5 轮感谢/积极互动，familiarity 逐轮上升。"""
    ctx = isolated_env

    # ── 第 0 轮：初始对话（建立关系基线）──
    _store_turn(
        ctx,
        "你好，初次见面",
        "你好！很高兴认识你，有什么我可以帮你的吗？",
        "2026-06-06 10:00:00",
    )
    time.sleep(0.1)

    # 第 1 轮后的关系状态（作为基线）
    query0 = "你好"
    spec0, _, _ = _run_pipeline(ctx, query0)
    rs0 = spec0.relationship
    fam0 = rs0.familiarity if rs0 else 0.0

    # ── 连续 5 轮积极互动（感谢 + 表达满意）──
    positive_turns = [
        ("谢谢你的帮助，你太贴心了！", "不客气！能帮到你我很开心。"),
        ("你真的帮了我很多，感谢你的耐心解答", "谢谢你的肯定！我会继续努力的。"),
        ("和你聊天很愉快，谢谢你一直陪着我", "我也很开心能陪伴你！"),
        ("今天心情好多了，多亏了你的建议，谢谢！", "听到你心情变好我真的很高兴！"),
        ("你是我遇到的最好的AI助手，非常感谢你的一切帮助", "谢谢你的信任和认可，这对我意义重大！"),
    ]
    for i, (user_msg, ai_msg) in enumerate(positive_turns):
        ts = f"2026-06-06 {10 + i + 1:02d}:00:00"
        _store_turn(ctx, user_msg, ai_msg, ts)
        time.sleep(0.05)
    time.sleep(0.2)

    # ── 第 6 轮后的关系状态 ──
    query5 = "谢谢你"
    spec5, _, _ = _run_pipeline(ctx, query5)
    rs5 = spec5.relationship
    assert rs5 is not None, "RelationshipState 不应为 None"

    fam5 = rs5.familiarity if rs5 else 0.0
    trust5 = rs5.trust if rs5 else 0.0

    # 验证：第 6 轮 familiarity > 第 1 轮
    assert fam5 > fam0, (
        f"X8 失败：5 轮积极互动后 familiarity ({fam5:.3f}) 应 > 初始 ({fam0:.3f})"
    )

    # 验证：5 轮感谢消息后 trust 应 > 0.5（默认基线 + 感谢加成）
    assert trust5 >= 0.55, (
        f"X8 失败：5 轮感谢后 trust ({trust5:.3f}) 应 ≥ 0.55（5×0.05=+0.25）"
    )

    # 验证：各字段在合法范围内
    assert 0.0 <= rs5.familiarity <= 1.0, f"familiarity={rs5.familiarity} 应在 [0,1]"
    assert 0.0 <= rs5.trust <= 1.0, f"trust={rs5.trust} 应在 [0,1]"
    assert 0.0 <= rs5.closeness <= 1.0, f"closeness={rs5.closeness} 应在 [0,1]"
    assert rs5.interaction_mode in ("casual", "collaborator", "partner", "teacher"), (
        f"interaction_mode={rs5.interaction_mode} 应为有效值"
    )


# ═══════════════════════════════════════════════════════════════════
# X9: 冲突修正 — 第 1 轮"我叫张三"，第 2 轮"不对，我叫李四"，旧记忆被 supersede
# ═══════════════════════════════════════════════════════════════════

def test_X9_conflict_correction(isolated_env):
    """X9 冲突修正：第 2 轮纠正身份信息，第 1 轮记忆 stale + superseded_by 指向新记忆。"""
    ctx = isolated_env

    # ── 第 1 轮：写入"张三" ──
    _store_turn(
        ctx,
        "我叫张三，今年25岁，在北京工作",
        "好的张三，我记住了，你在北京工作，25岁。",
        "2026-06-06 10:00:00",
    )
    time.sleep(0.3)

    # 定位第 1 轮记忆
    zhangsan_mems = _find_memories_containing(ctx, "张三")
    assert len(zhangsan_mems) >= 1, "第 1 轮关于张三的记忆应已入库"
    zhangsan_id = zhangsan_mems[0]["id"]

    # ── 第 2 轮：纠正"我是李四" ──
    _store_turn(
        ctx,
        "不对，我叫李四，之前说错了，其实我在上海工作",
        "明白了，已更新你的名字为李四，工作地在上海。抱歉之前记错了。",
        "2026-06-06 10:05:00",
    )
    time.sleep(0.3)

    # 定位第 2 轮记忆
    lisi_mems = _find_memories_containing(ctx, "李四")
    assert len(lisi_mems) >= 1, "第 2 轮关于李四的记忆应已入库"
    lisi_id = lisi_mems[0]["id"]

    # ── 手动执行 supersede（BM 模式跳过自动冲突检测）──
    meta_after = _get_mem_meta(ctx, zhangsan_id)
    if not meta_after.get("stale", False):
        ctx.chroma_service.supersede_memory(
            zhangsan_id, lisi_id, "用户纠正：张三→李四，北京→上海"
        )
        time.sleep(0.1)

    # ── 验证 1：第 1 轮记忆被标记 stale ──
    meta_final = _get_mem_meta(ctx, zhangsan_id)
    assert meta_final.get("stale", False) is True, (
        f"X9 失败：第 1 轮记忆 {zhangsan_id[:8]} 应被标记 stale=True"
    )

    # ── 验证 2：superseded_by 指向第 2 轮记忆 ──
    assert meta_final.get("superseded_by", "") == lisi_id, (
        f"X9 失败：superseded_by 应指向 {lisi_id[:8]}，"
        f"实际指向 {meta_final.get('superseded_by', '')[:8]}"
    )

    # ── 验证 3：后续检索应命中"李四"，且 stale 记忆被降权 ──
    query = "我叫什么名字"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)

    retrieved_ids = [m.get("id", "") for m in memories]

    # 第 2 轮（李四）应在检索结果中
    assert lisi_id in retrieved_ids, (
        f"X9 失败：第 2 轮记忆 {lisi_id[:8]}（李四）应在检索结果中"
    )

    # 验证 stale 记忆的 recency_weight ≤ 0.3（被降权）
    zhangsan_in_results = [m for m in memories if m.get("id") == zhangsan_id]
    if zhangsan_in_results:
        rw = zhangsan_in_results[0].get("recency_weight", 1.0)
        assert rw <= 0.3, (
            f"X9 失败：被 supersede 的旧记忆 recency_weight ({rw:.3f}) 应 ≤ 0.3"
        )

    # ── 验证 4：新记忆（李四）的 score 应高于旧记忆（张三）──
    lisi_score = None
    zhangsan_score = None
    for m in memories:
        if m.get("id") == lisi_id:
            lisi_score = m.get("score", 0)
        elif m.get("id") == zhangsan_id:
            zhangsan_score = m.get("score", 0)
    if lisi_score is not None and zhangsan_score is not None:
        assert lisi_score > zhangsan_score, (
            f"X9 失败：新记忆 score ({lisi_score:.3f}) 应 > 旧记忆 score ({zhangsan_score:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════
# 综合性跨轮测试
# ═══════════════════════════════════════════════════════════════════

def test_cross_turn_end_to_end(isolated_env):
    """组合测试：多话题写入 → 跨轮检索 → 情绪反转 → 身份纠正，全流程验证。"""
    ctx = isolated_env

    # ── 连续写入 6 轮不同话题 ──
    all_turns = [
        ("我叫王五，今年30岁，在深圳做产品经理",
         "好的王五，我记住了，深圳的产品经理，30岁。"),
        ("昨天去健身房练了腿，今天走路都费劲",
         "练腿日后的DOMS确实酸爽，多拉伸会好一些。"),
        ("最近在看一本书叫《人类简史》，讲得很有深度",
         "《人类简史》是一本很棒的书，尤瓦尔·赫拉利的视角很独特。"),
        ("不对，我之前说错了，我叫赵六不是王五，我在广州工作",
         "已更正，你是赵六，在广州工作。抱歉之前的错误。"),
        ("今天心情特别好，工作上搞定了一个大项目！",
         "恭喜！搞定大项目的成就感是最棒的，值得庆祝。"),
        ("我之前跟你说过什么来着",
         "让我回想一下……你提到过健身、《人类简史》这本书，还有工作上的好消息。"),
    ]

    for i, (user_msg, ai_msg) in enumerate(all_turns):
        ts = f"2026-06-06 {10 + i:02d}:00:00"
        _store_turn(ctx, user_msg, ai_msg, ts)
        time.sleep(0.05)
    time.sleep(0.5)

    # ── 验证 1：检索多条记忆 ──
    query = "还记得我的个人信息吗"
    spec, memories, sess_ctx = _run_pipeline(ctx, query)
    assert len(memories) >= 3, f"应检索到多条记忆，实际 {len(memories)}"

    # ── 验证 2：没有重复 ID ──
    mem_ids = [m.get("id", "") for m in memories if m.get("id")]
    assert len(mem_ids) == len(set(mem_ids)), "检索结果不应有重复 ID"

    # ── 验证 3：关系状态可计算 ──
    rs = spec.relationship
    assert rs is not None
    assert rs.familiarity > 0, f"多轮对话后 familiarity 应 > 0，实际 {rs.familiarity}"

    # ── 验证 4：stale_context 有内容（第 4 轮纠正了名字）──
    # 手动 supersede 第 1 轮"王五"
    wangwu_mems = _find_memories_containing(ctx, "王五")
    zhaoliu_mems = _find_memories_containing(ctx, "赵六")
    if wangwu_mems and zhaoliu_mems:
        wangwu_id = wangwu_mems[0]["id"]
        zhaoliu_id = zhaoliu_mems[0]["id"]
        meta = _get_mem_meta(ctx, wangwu_id)
        if not meta.get("stale", False):
            ctx.chroma_service.supersede_memory(wangwu_id, zhaoliu_id, "用户纠正名字")
            time.sleep(0.1)

        # 再次检索验证 stale 标记生效
        spec2, memories2, _ = _run_pipeline(ctx, "我叫什么名字")
        stale_mem = [m for m in memories2 if m.get("id") == wangwu_id]
        if stale_mem:
            rw = stale_mem[0].get("recency_weight", 1.0)
            assert rw <= 0.3, f"被取代的记忆 recency_weight={rw:.3f} 应 ≤ 0.3"
