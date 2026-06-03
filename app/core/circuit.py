"""回路调度器 — 用户消息的标准处理通路。

编排顺序：意图分析 → 记忆检索 → 一致性校验 → 响应选控（门控）
不重写现有模块，只编排"先走谁再走谁"。

模型增强：analyze_user_message() 可通过 brain 参数使用小模型。
用法：analyze_user_message(msg, brain=brain) → 模型优先+规则兜底
     analyze_user_message(msg)              → 纯规则（原有行为）
"""

import logging
import threading
from typing import Optional

from app.core.state import UserMessageAnalysis, GatingDecision, UtteranceSpec

logger = logging.getLogger(__name__)

# ── 模型增强开关 ──────────────────────────────────────

_CHUCHEN_BRAIN = None
_CHUCHEN_BRAIN_LOADED = False

def get_brain() -> Optional[object]:
    """获取 ChuchenBrain 实例（ChuchuCNN > Ollama > 规则，失败返回 None）。"""
    global _CHUCHEN_BRAIN, _CHUCHEN_BRAIN_LOADED
    if _CHUCHEN_BRAIN_LOADED:
        return _CHUCHEN_BRAIN
    _CHUCHEN_BRAIN_LOADED = True
    try:
        from app.brain.models import ChuchenBrain
        _CHUCHEN_BRAIN = ChuchenBrain(model_name="qwen2.5:3b")
        status = _CHUCHEN_BRAIN.load_all()
        if any(status.values()):
            logger.info("ChuchenBrain 已加载: %s", status)
        else:
            logger.info("ChuchenBrain 未连接Ollama，使用纯规则")
        return _CHUCHEN_BRAIN
    except Exception:
        _CHUCHEN_BRAIN = None
        return None


# ── 回路①：用户消息分析 ────────────────────────────

_INTENT_MAP = {
    "recall": ["记得", "之前", "上次", "以前", "曾经", "想起", "是不是说过",
               "那时候", "那会儿", "还记得", "记不记得"],
    "emotional_sharing": ["想", "觉得", "感觉", "心情", "难过", "开心", "烦",
                          "累", "困", "疲惫", "焦虑", "担心", "感动", "温暖",
                          "梦到", "失眠", "心疼", "好烦", "好累", "好开心",
                          "好难过"],
    "conflict": ["不对", "不是", "你错了", "别说了", "你搞错了", "乱说",
                 "你没听懂", "不是这样"],
    "ask_fact": ["什么", "怎么", "为什么", "如何", "能不能", "请问",
                 "啥", "是不是"],
    "request": ["帮我", "请你", "需要你", "帮我查", "帮我找", "帮我写",
                "帮我改", "帮我看看", "能不能帮我"],
    "meta": ["你是谁", "你能做什么", "你会什么", "你叫什么", "你有什么功能",
             "你是吗"],
}

_EMOTION_WORDS = {
    "intimate": ["想你", "爱", "心疼", "抱", "陪", "温暖", "梦到", "亲", "在乎",
                 "想你", "爱你", "抱抱"],
    "positive": ["开心", "高兴", "好", "棒", "喜欢", "感动", "幸福", "感谢",
                 "太棒", "太好了", "不错", "厉害"],
    "negative": ["难过", "烦", "累", "焦虑", "担心", "生气", "讨厌", "失望",
                 "痛苦", "崩溃", "孤独", "压力", "郁闷", "烦躁"],
    "frustrated": ["烦死了", "受不了", "无语", "气死", "崩溃", "不想说了",
                   "够了", "算了吧"],
}

# 程度副词 — 用于计算情绪强度
_INTENSIFIERS = {"很", "非常", "特别", "太", "超级", "极其", "好", "真", "真的", "实在"}
_EMOTION_REPEAT_PATTERN = ["好好好", "哈哈哈", "呜呜呜", "啊啊啊", "嘿嘿嘿"]

_WORK_KEYWORDS = [
    "bug", "代码", "修", "调试", "部署", "重构",
    "PR", "commit", "改bug", "熬夜", "加班", "上线",
    "项目", "进度", "需求", "排期", "接口", "数据库",
]

# 否定词
_NEGATION_WORDS = {"不", "没", "别", "不要", "没有", "不用", "不会", "不是"}

# ── Embedding 意图原型（惰性初始化） ──
_INTENT_PROTOTYPES = {
    "recall": [
        "你还记得我之前说的吗",
        "上次我们聊到那个事情",
        "我之前提过一个项目",
        "我记得你之前说过",
    ],
    "emotional_sharing": [
        "我今天心情不太好",
        "好开心啊终于搞定了",
        "我觉得好累啊",
        "最近压力好大",
    ],
    "conflict": [
        "你说的不对",
        "不是这样的",
        "你搞错了",
        "你理解错了",
    ],
    "ask_fact": [
        "今天天气怎么样",
        "你知道这个怎么用吗",
        "这是什么意思",
        "请问这个功能怎么用",
    ],
    "request": [
        "帮我查一下这个",
        "帮我写一段代码",
        "能不能帮我看看这个问题",
        "帮我改一下这个",
    ],
    "meta": [
        "你是谁开发的",
        "你能做什么事情",
        "你都有什么功能",
        "你是什么模型",
    ],
}

# 惰性加载的 embedding
_PROTO_EMBEDDINGS = None
_INTENT_EMBED_LOCK = threading.Lock()


def _get_proto_embeddings() -> dict[str, list[float]]:
    """惰性计算所有意图原型的 embedding。"""
    global _PROTO_EMBEDDINGS
    if _PROTO_EMBEDDINGS is not None:
        return _PROTO_EMBEDDINGS
    with _INTENT_EMBED_LOCK:
        if _PROTO_EMBEDDINGS is not None:
            return _PROTO_EMBEDDINGS
        try:
            from local_embed import local_embed_batch

            result = {}
            for intent, examples in _INTENT_PROTOTYPES.items():
                embs = local_embed_batch(examples)
                valid = [e for e in embs if e is not None]
                if valid:
                    # 对每个意图所有原型 embedding 取均值
                    import numpy as np
                    result[intent] = np.mean(valid, axis=0).tolist()
            _PROTO_EMBEDDINGS = result
            logger.debug("意图原型 embedding 已计算: %d 个意图", len(result))
        except Exception as exc:
            logger.warning("意图原型 embedding 计算失败: %s", exc)
            _PROTO_EMBEDDINGS = {}
        return _PROTO_EMBEDDINGS


def _detect_negation(text: str, emotion_words: list[str]) -> bool:
    """检测情感词前是否出现否定词。"""
    import re
    for ew in emotion_words:
        idx = text.find(ew)
        if idx < 0:
            continue
        # 取情感词前最多 6 个字符
        before = text[max(0, idx - 6):idx]
        for neg in _NEGATION_WORDS:
            if neg in before:
                return True
    return False


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


def analyze_user_message(user_message: str, chat_history=None,
                         query_embedding: Optional[list] = None,
                         brain: Optional[object] = None) -> UserMessageAnalysis:
    """分析用户消息的意图和情绪。

    决策优先级：ChuchuCNN(字符CNN) > Ollama > 关键词规则
    - brain 参数传入时：先走小模型，intent 和 emotion 独立控制置信度
    - 模型结果中任一字段置信度不足时，该字段降级到关键词兜底
    - 模型完全不可用时（brain=None/异常），走原有 keyword+embedding 双路
    """
    if not user_message:
        return UserMessageAnalysis(intent="casual", emotion="neutral")

    text = user_message.strip()
    urgency = _compute_urgency(text)
    topics = _extract_topics(text)
    emotion_intensity = _compute_emotion_intensity(text)

    # ── 模型路径（ChuchuCNN > Ollama > 规则，自带降级） ─────────
    model_intent = None
    model_emotion = None
    if brain is not None:
        try:
            model_intent = brain.classify_intent(text)
            model_emotion = brain.analyze_emotion(text)
        except Exception:
            pass  # 模型异常 → 走原有逻辑

    if model_intent is not None and model_intent.source == "model":
        # ── 独立置信度检查：各字段各自决策 ──
        intent_ok = model_intent.confidence >= 0.6
        emotion_ok = (model_emotion is not None
                      and model_emotion.source == "model"
                      and model_emotion.confidence >= 0.5)

        if intent_ok or emotion_ok:
            final_intent = model_intent.intent if intent_ok else _keyword_intent(text)
            final_emotion = model_emotion.primary if emotion_ok else _keyword_emotion(text)
            final_confidence = model_intent.confidence if intent_ok else 0.4

            # 情绪否定检测
            if final_emotion != "neutral":
                target_words = _EMOTION_WORDS.get(final_emotion, [])
                if _detect_negation(text, target_words):
                    final_emotion = "neutral"
            if _has_explicit_negation(text) and final_emotion != "neutral":
                final_emotion = "neutral"

            return UserMessageAnalysis(
                intent=final_intent, emotion=final_emotion,
                urgency=urgency, topics=topics, raw_text=text,
                confidence=final_confidence,
                emotion_intensity=emotion_intensity,
            )

    # ── 原有 keyword + embedding 双路（完整兜底） ──────────
    kw_intent = _keyword_intent(text)
    kw_emotion = _keyword_emotion(text)
    if kw_emotion != "neutral":
        target_words = _EMOTION_WORDS.get(kw_emotion, [])
        if _detect_negation(text, target_words):
            kw_emotion = "neutral"

    kw_confidence = 0.3
    if kw_intent != "casual":
        kw_confidence = 0.6
    if kw_intent != "casual" and kw_emotion != "neutral":
        kw_confidence = 0.8

    urgency = _compute_urgency(text)
    topics = _extract_topics(text)
    emotion_intensity = _compute_emotion_intensity(text)

    # ── embedding 路径 ────────────────────────────────────
    emb_intent = None
    emb_confidence = 0.0

    if query_embedding is not None:
        try:
            import numpy as np
            protos = _get_proto_embeddings()
            if protos:
                query_arr = np.array(query_embedding, dtype=np.float32)
                best_intent = "casual"
                best_sim = 0.0
                for intent, proto_emb in protos.items():
                    proto_arr = np.array(proto_emb, dtype=np.float32)
                    sim = float(np.dot(query_arr, proto_arr))
                    if sim > best_sim:
                        best_sim = sim
                        best_intent = intent
                if best_sim >= 0.5:
                    emb_intent = best_intent
                    emb_confidence = min(0.5 + (best_sim - 0.5) * 2, 0.95)
        except Exception as exc:
            logger.debug("embedding 意图分类跳过: %s", exc)

    # ── 双路投票 ──────────────────────────────────────────
    if emb_intent is not None and emb_confidence > kw_confidence:
        final_intent = emb_intent
        final_emotion = kw_emotion if kw_emotion != "neutral" else _keyword_emotion(text)
        if final_emotion != "neutral":
            target_words = _EMOTION_WORDS.get(final_emotion, [])
            if _detect_negation(text, target_words):
                final_emotion = "neutral"
        if _has_explicit_negation(text) and final_emotion != "neutral":
            final_emotion = "neutral"
    else:
        final_intent = kw_intent
        final_emotion = kw_emotion

    if final_intent == "ask_fact" and "吗" in text:
        has_emotion = any(kw in text for words in _EMOTION_WORDS.values() for kw in words)
        if has_emotion and emb_intent is not None and emb_intent in ("emotional_sharing", "recall"):
            final_intent = emb_intent
            final_emotion = kw_emotion

    if emb_intent is not None and emb_confidence > kw_confidence:
        final_confidence = emb_confidence
    else:
        final_confidence = kw_confidence

    return UserMessageAnalysis(
        intent=final_intent, emotion=final_emotion, urgency=urgency,
        topics=topics, raw_text=text, confidence=final_confidence,
        emotion_intensity=emotion_intensity,
    )


def _keyword_intent(text: str) -> str:
    """纯关键词意图检测。"""
    # 优先级：conflict > emotional_sharing > recall > ask_fact > request > meta > casual
    for kw in _INTENT_MAP["conflict"]:
        if kw in text:
            return "conflict"
    for kw in _INTENT_MAP["emotional_sharing"]:
        if kw in text:
            return "emotional_sharing"
    for kw in _INTENT_MAP["recall"]:
        if kw in text:
            return "recall"
    for kw in _INTENT_MAP["request"]:
        if kw in text:
            return "request"
    for kw in _INTENT_MAP["ask_fact"]:
        if kw in text:
            return "ask_fact"
    for kw in _INTENT_MAP["meta"]:
        if kw in text:
            return "meta"
    return "casual"


def _keyword_emotion(text: str) -> str:
    """纯关键词情绪检测。intimate > frustrated > negative > positive > neutral。"""
    # 优先匹配 intimate 和 frustrated（它们更具体）
    for kw in _EMOTION_WORDS["intimate"]:
        if kw in text:
            return "intimate"
    for kw in _EMOTION_WORDS["frustrated"]:
        if kw in text:
            return "frustrated"
    for kw in _EMOTION_WORDS["negative"]:
        if kw in text:
            return "negative"
    for kw in _EMOTION_WORDS["positive"]:
        if kw in text:
            return "positive"
    return "neutral"


def _has_explicit_negation(text: str) -> bool:
    """检查是否有明确的否定句式。"""
    import re
    patterns = [
        r"不\w*[难过开心高兴好累烦]",
        r"没\w*[难过开心高兴好累烦]",
        r"别\w*[说了提]",
    ]
    return any(re.search(p, text) for p in patterns)


def _compute_urgency(text: str) -> float:
    urgency = 0.0
    if "!" in text or "！！" in text:
        urgency += 0.3
    if len(text) > 100:
        urgency += 0.2
    if "急" in text or "马上" in text or "立刻" in text:
        urgency += 0.4
    return min(urgency, 1.0)


def _extract_topics(text: str) -> list:
    try:
        import jieba.analyse
        return jieba.analyse.extract_tags(text, topK=5)
    except Exception:
        return []


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

        # ② 若调用方未提供检索结果，自动补充（纯兼容）
        if memories is None:
            timeline_recent, session_context, personalities, memories = run_chat_retrieval(
                user_message, query_embedding, ctx_obj)
        personalities = personalities or []
        memories = memories or []
        _ticks = [("start", __import__('time').perf_counter())]
        def _log_step(name):
            import time as _t
            from app.core import bottleneck as _b
            ms = (_t.perf_counter() - _ticks[-1][1]) * 1000
            _b.record(name, ms)
            _ticks.append((name, _t.perf_counter()))
        _log_step('prep')

        # ① 用户消息分析（传入 query_embedding 启用 embedding 路径，传入 brain 启用模型增强）
        prefrontal = analyze_user_message(user_message, self._chat_history,
                                           query_embedding=query_embedding,
                                           brain=get_brain())

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
        # ③ 一致性校验 + 巩固状态注入
        from app.core.state import CognitiveState
        temp = CognitiveState()
        import math as _math
        for mem in memories:
            meta = mem.get("metadata") or {}
            source = mem.get("source", "semantic")
            dist = mem.get("distance", 0.5)
            stale = meta.get("stale", False)
            if stale:
                temp.suppressed_ids.add(mem.get("id", ""))
                continue

            # 连续置信度：组合语义距离 + hit_count + 来源可靠性
            semantic_conf = 1.0 - dist
            hc = meta.get("hit_count", 0) or 0
            hit_conf = min(_math.log(hc + 1) / _math.log(11), 1.0) if hc > 0 else 0.0
            source_weight = {
                "semantic": 1.0, "dmn_preheat": 0.85, "entity_match": 0.8,
                "kw_match": 0.65, "tag_match": 0.6, "keyword_expand": 0.55,
                "text_match": 0.6, "time_rhythm": 0.4, "co_occurrence": 0.35,
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
        )
