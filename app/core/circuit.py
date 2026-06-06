"""回路调度器 — 用户消息的标准处理通路。

编排顺序：意图分析 → 记忆检索 → 一致性校验 → 响应选控（门控）

意图/情绪分类由 app.brain.semantic 驱动（bge-m3 原型匹配）。
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from app.core.state import UserMessageAnalysis, GatingDecision, UtteranceSpec
from app.brain.keywords import (
    INTENT_KEYWORDS, EMOTION_KEYWORDS,
    INTENSIFIERS, EMOTION_REPEAT_PATTERN, WORK_KEYWORDS,
)
from app.brain.semantic import (
    classify_intent as _sem_classify_intent,
    analyze_emotion as _sem_analyze_emotion,
    classify_urgency as _sem_classify_urgency,
    extract_tags as _sem_extract_tags,
    detect_negation as _sem_detect_negation,
)

logger = logging.getLogger(__name__)

# ── 回路①：用户消息分析 ────────────────────────────

# ── 从统一常量表导入（唯一来源 app/brain/keywords.py） ──
_INTENT_MAP = INTENT_KEYWORDS
_EMOTION_WORDS = EMOTION_KEYWORDS
_INTENSIFIERS = INTENSIFIERS
_EMOTION_REPEAT_PATTERN = EMOTION_REPEAT_PATTERN
_WORK_KEYWORDS = WORK_KEYWORDS

def _compute_emotion_intensity(text: str) -> float:
    """根据感叹号、emoji、程度副词、重复字估算情绪强度 0~1。"""
    score = 0.0
    # 感叹号
    exclaim = text.count("！") + text.count("!")
    score += min(exclaim * 0.15, 0.3)
    # emoji
    import re
    emoji_pattern = re.compile("[\U0001F300-\U0001F9FF☀-➿︀-️]+")
    emoji_count = len(emoji_pattern.findall(text))
    score += min(emoji_count * 0.1, 0.2)
    # 程度副词
    for w in _INTENSIFIERS:
        if w in text:
            score += 0.1
    # 重复字模式（好好好、哈哈哈）
    for p in _EMOTION_REPEAT_PATTERN:
        if p in text:
            score += 0.2
    # 长文本情绪增强
    if len(text) > 80:
        score += 0.1
    return min(score, 1.0)


def analyze_user_message(user_message: str, chat_history=None) -> UserMessageAnalysis:
    """分析用户消息的意图和情绪 — 纯语义层（bge-m3 原型匹配）。

    Ollama 不可用时降级为默认值（casual / neutral），不崩溃。
    """
    if not user_message:
        return UserMessageAnalysis(intent="casual", emotion="neutral")

    _t0 = __import__('time').perf_counter()
    text = user_message.strip()

    # 预先计算一次 embedding，复用于意图分类和情绪分析（省 2 次 Ollama HTTP 往返）
    from app.llm.embed import local_embed
    _text_emb = local_embed(text)

    # 语义层一步到位：意图、情绪、紧急度、关键词
    intent = _sem_classify_intent(text, embedding=_text_emb)
    emotion = _sem_analyze_emotion(text, embedding=_text_emb)
    urgency = _sem_classify_urgency(text)
    topics = _sem_extract_tags(text, topk=5)
    emotion_intensity = _compute_emotion_intensity(text)
    _t1 = __import__('time').perf_counter()

    # 否定检测（情绪词被否定 → 回 neutral）
    if emotion != "neutral":
        target_words = _EMOTION_WORDS.get(emotion, [])
        if any(_sem_detect_negation(text, w) for w in target_words):
            emotion = "neutral"

    # 置信度基线（embedding 原型匹配）
    confidence = 0.7

    from app.core import bottleneck as _b
    elapsed = (_t1 - _t0) * 1000
    logger.debug("user_analysis: %.1fms intent=%s emotion=%s urgency=%.2f topics=%s",
                 elapsed, intent, emotion, urgency, topics)
    if elapsed > 500:
        _b.record("user_analysis", elapsed)

    return UserMessageAnalysis(
        intent=intent, emotion=emotion, urgency=urgency,
        topics=topics, raw_text=text, confidence=confidence,
        emotion_intensity=emotion_intensity,
    )


# ── 回路④：响应门控（门控决策） ────────────────────────────


def basal_ganglia_gate(
    prefrontal: UserMessageAnalysis,
    memories: list,
    impulses: list,
    personality_notes: list,
    ctx_obj=None,
) -> GatingDecision:
    """根据用户消息分析结果，决定输出语气 + 压制不合适的冲动。"""
    tone = "warm"
    formality = 0.3
    intimacy = 0.0
    response_mode = "auto"
    suppression_reasons = []

    pfc = prefrontal

    if pfc.intent == "emotional_sharing":
        if pfc.emotion in ("negative", "intimate", "frustrated"):
            tone = "caring"
            formality = 0.1
            response_mode = "soothe"
        else:
            tone = "warm"
            response_mode = "question_first"

    elif pfc.intent == "conflict":
        tone = "soft"
        formality = 0.5
        response_mode = "confirm"

    elif pfc.intent == "recall":
        tone = "direct" if pfc.emotion == "neutral" else "warm"
        response_mode = "auto"

    elif pfc.intent == "ask_fact":
        tone = "direct"
        formality = 0.4
        response_mode = "direct_answer"

    elif pfc.intent == "request":
        tone = "direct"
        formality = 0.3
        response_mode = "direct_answer"

    elif pfc.intent == "meta":
        tone = "direct"
        formality = 0.4
        response_mode = "direct_answer"

    # 亲密程度
    if pfc.emotion == "intimate":
        intimacy = 0.7
    elif pfc.urgency > 0.5:
        intimacy = 0.0
    elif pfc.emotion_intensity and pfc.emotion_intensity > 0.5:
        intimacy = 0.5
    else:
        intimacy = 0.3

    # 压制工作类冲动（亲密/情绪场景下）
    impulses_to_show = list(impulses)
    if pfc.intent in ("emotional_sharing", "conflict"):
        filtered = []
        for imp in impulses_to_show:
            target = getattr(imp, 'target_concept', str(imp))
            if any(kw in target for kw in _WORK_KEYWORDS):
                suppression_reasons.append(
                    f"冲动被压制：工作类+{pfc.intent}场景 → {target[:40]}"
                )
                continue
            filtered.append(imp)
        impulses_to_show = filtered

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

    return GatingDecision(
        tone=tone, formality=formality, intimacy=intimacy,
        response_mode=response_mode,
        suppression_reasons=suppression_reasons,
        memories_to_show=memories,
        impulses_to_show=impulses_to_show,
    )


# ── 回路主调度器 ────────────────────────────────────────────


class CircuitOrchestrator:
    """编排一次用户消息的完整处理通路。"""

    def __init__(self, chroma_service, personality_store, impulse_scheduler,
                 dmn_engine, chat_history, co_tracker, mirror_neuron=None):
        self._chroma = chroma_service
        self._personality = personality_store
        self._impulse = impulse_scheduler
        self._dmn = dmn_engine
        self._chat_history = chat_history
        self._co_tracker = co_tracker
        self._mirror_neuron = mirror_neuron

    def process(
        self,
        user_message: str,
        query_embedding: list,
        ctx_obj,
        *,
        timeline_recent: Optional[list] = None,
        session_context: Optional[str] = None,
        personalities: Optional[list] = None,
        memories: Optional[list] = None,
    ) -> UtteranceSpec:
        """回路①→②→③→④ 顺序执行，返回 UtteranceSpec。

        调用方需先执行检索管线得到 timeline_recent/session_context/personalities/memories，
        传入本方法做后续分析 + 门控。
        """
        from app.retrieval.pipeline import run_chat_retrieval

        _ticks = [("start", __import__('time').perf_counter())]
        def _log_step(name):
            import time as _t
            from app.core import bottleneck as _b
            ms = (_t.perf_counter() - _ticks[-1][1]) * 1000
            _b.record(name, ms)
            _ticks.append((name, _t.perf_counter()))
        _log_step('prep')

        # ① 先跑用户消息分析（语义层 bge-m3 原型匹配）
        prefrontal = analyze_user_message(user_message, self._chat_history)

        # ② 若调用方未提供检索结果，自动补充（传 intent 让 pipeline 复用语义结果）
        if memories is None:
            timeline_recent, session_context, personalities, memories = run_chat_retrieval(
                user_message, query_embedding, ctx_obj,
                intent=prefrontal.intent if prefrontal.confidence >= 0.6 else None,
            )
        personalities = personalities or []
        memories = memories or []

        _log_step('user_analysis')
        # 行为预测：预测用户下一步行为模式
        mirror_prediction = None
        if self._mirror_neuron is not None:
            try:
                mirror_prediction = self._mirror_neuron.predict(
                    prefrontal.intent, prefrontal.topics)
            except Exception as exc:
                logger.debug("行为预测跳过: %s", exc)

        _log_step('mirror_predict')
        # ── 引擎编织：从候选集中织出上下文（替代 TOP_K 截断）──
        woven = self.weave_context(memories, prefrontal)
        if not woven.should_speak:
            memories = []
        else:
            memories = woven.fact_memories
        _log_step('weave_context')

        # ③ 一致性校验 + 巩固状态注入
        from app.core.state import CognitiveState
        temp = CognitiveState()
        import math as _math

        # v2.1: 收集 stale 记忆，不屏蔽只标记
        stale_mems = []

        # ── 编织后的记忆直接作为 fact，跳过 CognitiveState 二次评分 ──
        # v2.1: stale 记忆不屏蔽，收集到 stale_context 供 LLM 做背景参考
        if woven.should_speak and woven.fact_memories:
            for mem in woven.fact_memories:
                meta = mem.get("metadata") or {}
                if meta.get("stale", False):
                    stale_mems.append(mem)
                    continue
                temp.add_fact(mem, certainty=0.7)
            # v2.1: 编织器已把 stale 路由到 woven.stale_context
            stale_mems.extend(woven.stale_context)
            # reference 级记忆：编织器分出的次高置信记忆
            for mem in (woven.reference_memories or []):
                temp.add_reference(mem, certainty=0.35)
        else:
            for mem in memories:
                meta = mem.get("metadata") or {}
                source = mem.get("source", "semantic")
                dist = mem.get("distance", 0.5)
                stale = meta.get("stale", False)
                if stale:
                    stale_mems.append(mem)
                    continue

                # 连续置信度：组合语义距离 + hit_count + 来源可靠性
                semantic_conf = 1.0 - dist
                hc = meta.get("hit_count", 0) or 0
                hit_conf = min(_math.log(hc + 1) / _math.log(11), 1.0) if hc > 0 else 0.0
                source_weight = {
                    "semantic": 1.0, "dmn_preheat": 0.85, "entity_match": 0.8,
                    "kw_match": 0.65, "tag_match": 0.6, "keyword_expand": 0.55,
                    "text_match": 0.6, "time_rhythm": 0.4, "co_occurrence": 0.35,
                    "bm25_fulltext": 0.75,
                }.get(source, 0.5)
                certainty = 0.5 * semantic_conf + 0.25 * hit_conf + 0.25 * source_weight
                certainty = max(0.0, min(1.0, certainty))

                if certainty >= 0.6:
                    temp.add_fact(mem, certainty=certainty)
                elif certainty >= 0.35:
                    temp.add_reference(mem, certainty=certainty)
                else:
                    temp.add_background(mem)

        for p in personalities:
            if isinstance(p, dict):
                c = p.get("content", "")
                t = p.get("type", "")
                if c and p.get("source", "user") == "user":
                    temp.personality_notes.append({"content": c, "type": t})
            elif isinstance(p, str):
                temp.personality_notes.append(p)

        # AI 人格标签（source=ai，用于【我自己的表达习惯】）
        ai_notes = []
        try:
            ai_result = self._personality.list_tags(page=1, page_size=5)
            for item in ai_result.get("items", []):
                if item.get("source", "user") == "ai":
                    content = item.get("content", "")
                    if content:
                        ai_notes.append({"content": content, "type": item.get("type", "")})
        except Exception as exc:
            logger.debug("AI 人格标签获取跳过: %s", exc)
        temp.personality_notes_ai = ai_notes

        if self._dmn is not None:
            try:
                self._dmn.apply_to_cognitive_state(temp)
            except Exception as exc:
                logger.debug("DMN 注入失败: %s", exc)

        # M4 fix: 设置 user_mood 和 affective_context（之前永为 None）
        _emotion_to_mood = {"positive": "positive", "negative": "negative",
                            "frustrated": "negative", "intimate": "positive"}
        temp.user_mood = _emotion_to_mood.get(prefrontal.emotion, "neutral")
        _intent_to_affective = {
            "conflict": "conflict", "emotional_sharing": "casual_chat",
            "recall": "casual_chat", "ask_fact": "focused_work",
            "request": "focused_work", "meta": "casual_chat",
        }
        temp.affective_context = _intent_to_affective.get(prefrontal.intent, "casual_chat")
        # emotional_sharing + 负面情绪 → intimate
        if prefrontal.intent == "emotional_sharing" and prefrontal.emotion in ("negative", "intimate", "frustrated"):
            temp.affective_context = "intimate"

        # 行为预测写入状态
        if mirror_prediction:
            temp.mirror_prediction = mirror_prediction

        # 话题笔记检索
        topic_notes = []
        try:
            if self._dmn is not None and prefrontal.topics:
                topic_notes = self._dmn.get_topic_notes(prefrontal.topics)
        except Exception as exc:
            logger.debug("话题笔记检索跳过: %s", exc)

        # M3 fix: fact 级记忆 + reference 级记忆都给 LLM
        fact_memories = [d for d in temp.primary]
        ref_memories = [d for d in temp.secondary]

        # 冲动收集
        impulses = []
        try:
            gap = self._impulse.idle_gap_minutes(self._chat_history)
            if gap is not None and gap >= 2:
                imp = self._impulse.get_next()
                if imp:
                    from app.core.state import ImpulseDirective
                    impulses.append(ImpulseDirective(
                        intent="share_observation",
                        target_concept=imp.get("content", ""),
                        emotional_tone="neutral",
                    ))
        except Exception as exc:
            logger.debug("冲动收集跳过: %s", exc)

        # ④ 响应门控
        gate = basal_ganglia_gate(
            prefrontal, fact_memories, impulses, temp.personality_notes, ctx_obj)

        _log_step('response_gate')
        # ── 关系维度更新（基于当前轮对话信号） ────────────────────────
        from app.core.state import RelationshipState
        rs = RelationshipState()
        if self._chat_history is not None:
            try:
                recent = self._chat_history.get_recent(n=30)
                if len(recent) > 0:
                    rs.familiarity = min(1.0, len(recent) * 0.02)

                    err_count = sum(
                        1 for r in recent
                        if "记错" in r.get("user_message", "") or "不对" in r.get("user_message", "")
                    )
                    thanks_count = sum(
                        1 for r in recent
                        if "谢谢" in r.get("user_message", "") or "感谢" in r.get("user_message", "")
                    )
                    rs.trust = max(0.0, min(1.0, 0.5 + thanks_count * 0.05 - err_count * 0.1))

                    intimate_count = sum(
                        1 for r in recent
                        if "想你" in r.get("user_message", "") or "爱" in r.get("user_message", "")
                    )
                    sad_count = sum(
                        1 for r in recent
                        if "难过" in r.get("user_message", "") or "烦" in r.get("user_message", "")
                    )
                    rs.closeness = max(0.0, min(1.0, intimate_count * 0.1 + sad_count * 0.05))

                    tech_topics = ["架构", "代码", "Rust", "bug", "部署", "系统", "重构"]
                    emotional_words = ["难过", "开心", "感动", "压力", "累", "焦虑"]
                    user_msgs = " ".join(r.get("user_message", "") for r in recent[-10:])
                    tech_score = sum(1 for w in tech_topics if w in user_msgs)
                    emo_score = sum(1 for w in emotional_words if w in user_msgs)
                    if tech_score > emo_score * 2:
                        rs.interaction_mode = "collaborator"
                    elif emo_score > tech_score * 2:
                        rs.interaction_mode = "partner"
                    else:
                        rs.interaction_mode = "casual"
            except Exception:
                pass  # 关系更新失败不影响主链路
        logger.debug("关系维度: familiarity=%.2f trust=%.2f closeness=%.2f mode=%s",
                     rs.familiarity, rs.trust, rs.closeness, rs.interaction_mode)

        return UtteranceSpec(
            user=prefrontal,
            memories=fact_memories,
            reference_memories=ref_memories,
            impulses=gate.impulses_to_show,
            gate=gate,
            timeline_recent=timeline_recent,
            session_context=session_context,
            personality_notes=temp.personality_notes,
            personality_notes_ai=temp.personality_notes_ai,
            mirror_prediction=mirror_prediction or {},
            topic_notes=topic_notes,
            relationship=rs,
            stale_context=stale_mems,
            woven_context=woven,
        )

    def weave_context(
        self,
        candidates: list[dict],
        cognitive_state,
    ) -> "WovenContext":
        """引擎编织记忆上下文。替代原来的 TOP_K 截断。

        全程纯规则 + 统计，零 LLM 调用。延迟目标 < 150ms。
        """
        from app.models.schemas import WovenContext
        from app.config.settings import BENCHMARK_MODE
        from collections import defaultdict
        import math

        if not candidates:
            return WovenContext(should_speak=False, total_candidates=0)

        # ── Benchmark 模式：跳过认知编织，所有候选直接作为 fact ──
        if BENCHMARK_MODE:
            wc = WovenContext(
                should_speak=True,
                total_candidates=len(candidates),
                fact_memories=list(candidates),
                total_tokens=sum(
                    len((m.get("metadata") or {}).get("summary", "")
                        or m.get("document", "")) // 2 + 10
                    for m in candidates
                ),
            )
            return wc

        wc = WovenContext(total_candidates=len(candidates))

        # ── 预处理：取元数据（v2.1: 不再跳过 stale — 旧记忆只降权不屏蔽）──
        active = []
        for m in candidates:
            meta = m.get("metadata") or {}
            tags = meta.get("tags", "")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            elif not isinstance(tags, list):
                tags = []
            m["_tags"] = tags
            m["_entities"] = meta.get("entities", []) or []
            ts = meta.get("timestamp", 0)
            try:
                m["_ts"] = float(ts)
            except (ValueError, TypeError):
                m["_ts"] = 0
            m["_stale"] = meta.get("stale", False)
            m["_archived"] = meta.get("archived", False)
            active.append(m)

        if not active:
            return WovenContext(should_speak=False, total_candidates=len(candidates))

        # ── 判断要不要说 ──
        intent = getattr(cognitive_state, 'intent', 'casual') if hasattr(cognitive_state, 'intent') else 'casual'
        if intent == 'casual' and len(active) <= 3:
            # 闲聊 + 候选少 = 不说
            wc.should_speak = False
            return wc
        wc.should_speak = True

        # ═══════════════════════════════════════════════════
        # 层一：故事线（同实体 + 跨时间）
        # ═══════════════════════════════════════════════════
        entity_buckets = defaultdict(list)
        for m in active:
            raw_entities = m.get("_entities", [])
            if isinstance(raw_entities, str):
                try:
                    raw_entities = json.loads(raw_entities)
                except (json.JSONDecodeError, TypeError):
                    raw_entities = []
            if not isinstance(raw_entities, list):
                raw_entities = []
            entities = m.get("_tags", []) + raw_entities
            entities = list(set(e for e in entities if isinstance(e, str) and len(e) >= 2))
            if entities:
                key = tuple(sorted(entities[:3]))
                entity_buckets[key].append(m)

        narratives = []
        used_in_narrative = set()
        for key, bucket in entity_buckets.items():
            if len(bucket) < 2:
                continue
            bucket.sort(key=lambda x: x.get("_ts", 0))
            timespan_days = (bucket[-1].get("_ts", 0) - bucket[0].get("_ts", 0)) / 86400
            if timespan_days < 1:
                continue  # 同一天内的不算故事线

            # 拼 narrative
            entity_str = "/".join(key)
            first_ts = bucket[0].get("_ts", 0)
            last_ts = bucket[-1].get("_ts", 0)
            from datetime import datetime
            first_date = datetime.fromtimestamp(first_ts).strftime("%m月%d日") if first_ts else "?"
            last_date = datetime.fromtimestamp(last_ts).strftime("%m月%d日") if last_ts else "?"

            # 情绪变化判断
            valences = []
            for m in bucket:
                v = m.get("metadata", {}).get("emotion_valence_bin", "neutral") if m.get("metadata") else "neutral"
                valences.append(v)
            trend = "延续"
            if "positive" in valences and "negative" in valences:
                trend = "出现翻转"
            elif all(v == "positive" for v in valences):
                trend = "持续积极"
            elif all(v == "negative" for v in valences):
                trend = "持续消极"

            narrative = (
                f"{first_date}首次提到{entity_str}，"
                f"共提及{len(bucket)}次，跨越{int(timespan_days)}天，{trend}。"
            )
            narratives.append(narrative)
            for m in bucket:
                used_in_narrative.add(m.get("id", ""))

        wc.narratives = narratives[:5]  # 最多 5 条故事线

        # ═══════════════════════════════════════════════════
        # 层三：分层（fact / background / discard）
        # v2.1: stale 记忆不屏蔽，进 stale_context 供 LLM 做背景参考
        # ═══════════════════════════════════════════════════
        MIN_FACT_DIST = 0.30  # 语义距离阈值

        for m in active:
            mid = m.get("id", "")
            dist = m.get("distance", 0.5)
            source = m.get("source", "semantic")
            is_stale = m.get("_stale", False)

            # story line 里的：非 stale → fact，stale → stale_context
            if mid in used_in_narrative:
                if is_stale:
                    wc.stale_context.append(m)
                else:
                    wc.fact_memories.append(m)
                continue

            # 语义距离判断 + 来源权重
            # dist < MIN_FACT_DIST × boost → fact（低置信来源要求更近的语义距离）
            source_boost = {"semantic_hot": 1.0, "semantic_cool": 0.9,
                            "entity_match": 0.85, "kw_match": 0.7,
                            "tag_match": 0.7, "co_occurrence": 0.6,
                            "time_triggered": 0.5, "attention_drift": 0.8,
                            "topic_expand": 0.55,
                            "bm25_fulltext": 0.75,
                            "ai_expression": 0.7}
            boost = source_boost.get(source, 0.8)
            threshold = MIN_FACT_DIST * boost

            if dist < threshold:
                if is_stale:
                    wc.stale_context.append(m)  # stale → 背景参考，不作为事实
                else:
                    wc.fact_memories.append(m)
            elif dist < 0.45:
                # reference 级：引擎有一定把握，LLM 需带核实语气
                if not is_stale:
                    wc.reference_memories.append(m)
            # else: discard 级，不处理

        # ═══════════════════════════════════════════════════
        # 层四：Token 预算分配
        # ═══════════════════════════════════════════════════
        MAX_TOKENS = 20000  # 给 fact_memories 的最大 token 预算
        estimated_tokens = 0
        final_facts = []

        # narrative 已经压缩得很短，不占太多 token
        for n in wc.narratives:
            estimated_tokens += len(n) // 2  # 中文字符估算为 0.5 token

        for m in wc.fact_memories:
            doc = m.get("document", "") or m.get("summary", "") or ""
            summary = (m.get("metadata") or {}).get("summary", "") or ""
            text = summary[:200] if summary else doc[:200]
            tokens = len(text) // 2 + 10  # +10 预留 overhead
            estimated_tokens += tokens
            if estimated_tokens <= MAX_TOKENS:
                final_facts.append(m)
            else:
                break

        wc.fact_memories = final_facts
        wc.total_tokens = estimated_tokens
        return wc


@dataclass
class GateResult:
    """门控决策输出。"""
    tone: str = "warm"
    formality: float = 0.3
    response_mode: str = "auto"
    intimacy: float = 0.0
    suppression_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "rule"


class GateDecisionMaker:
    """门控决策器 — 纯规则（策略映射表，不需模型）。"""

    def __init__(self, model_name: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        pass

    def load(self) -> bool:
        return True

    def decide(self, intent: str, emotion: str,
               context: dict | None = None) -> GateResult:
        return self._rule_decide(intent, emotion)

    def _rule_decide(self, intent: str, emotion: str) -> GateResult:
        tone = "warm"
        formality = 0.3
        response_mode = "auto"
        intimacy = 0.3

        if intent == "emotional_sharing":
            if emotion in ("negative", "intimate", "frustrated"):
                tone = "caring"
                formality = 0.1
                response_mode = "soothe"
                intimacy = 0.6
            else:
                tone = "warm"
                response_mode = "question_first"
        elif intent == "conflict":
            tone = "soft"
            formality = 0.5
            response_mode = "confirm"
            intimacy = 0.1
        elif intent == "recall":
            tone = "direct" if emotion == "neutral" else "warm"
            response_mode = "auto"
        elif intent in ("ask_fact", "meta"):
            tone = "direct"
            formality = 0.4
            response_mode = "direct_answer"
        elif intent == "request":
            tone = "direct"
            formality = 0.3
            response_mode = "direct_answer"

        if emotion == "intimate":
            intimacy = max(intimacy, 0.7)

        return GateResult(
            tone=tone, formality=formality, response_mode=response_mode,
            intimacy=intimacy, confidence=0.8, source="rule",
        )

