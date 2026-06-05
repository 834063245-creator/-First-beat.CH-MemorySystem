"""认知状态层 — 引擎决策的数据结构，LLM 皮层的唯一接口。

这是整场重构的基石：
  旧架构：引擎 → 文字纸条 → LLM 自己判断
  新架构：引擎 → CognitiveState → LLM 只按决策执行

所有决策由引擎（检索管线/DMN/冲动/蒸馏）完成，
LLM 不再需要理解置信度、来源、标签等元数据。
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import WovenContext


# ── 关系模型 ─────────────────────────────────────────────────


@dataclasses.dataclass
class RelationshipState:
    """轻量关系模型，纯滚动窗口（近 30 轮），不落盘。

    每次 process() 中根据 ChatHistory 重新计算，
    重启后自动从近 30 轮对话恢复。
    """
    familiarity: float = 0.0    # 熟悉度 0~1
    trust: float = 0.5          # 信任度 0~1，默认 0.5（中性起点）
    closeness: float = 0.0      # 亲密度 0~1
    interaction_mode: str = "casual"  # casual / collaborator / partner / teacher


# ── 记忆决策 ────────────────────────────────────────────────

MemoryRole = Literal["fact", "reference", "background", "suppressed"]


@dataclasses.dataclass
class MemoryDirective:
    """引擎对一条记忆的完整决策。

    role 含义：
      fact       — 引擎高置信，LLM 可直接当作事实引用
      reference  — 引擎有一定把握，LLM 需带核实语气
      background — 上下文相关，LLM 用来调语气但不需提及
      suppressed — 引擎已过滤，不给 LLM 看到

    注入 prompt 时不再携带任何元数据标签（置信度/来源/情绪），
    LLM 只需知道自己该怎么做。
    """

    memory_id: str
    summary: str
    document: str = ""
    role: MemoryRole = "reference"

    # 引擎算出来的置信度，用于内部决策，不给 LLM 看到
    certainty: float = 0.0

    # 情绪上下文（可选），注入时写作一句话而非标签
    emotional_context: Optional[str] = None

    # 时间信息（可选），注入时写作 "昨天下午" 而非时间戳
    time_hint: Optional[str] = None

    # 认知来源——这条记忆是怎么被想起来的，用于 prompt 分维度展示
    source: str = "semantic"

    # 情绪原始值——不给标签，LLM 自己判断
    emotional_intensity: int = 0
    emotion_valence_bin: str = ""

# ── 冲动指令 ────────────────────────────────────────────────

ImpulseIntent = Literal["recall", "check", "share_observation", "emotional_check"]


@dataclasses.dataclass
class ImpulseDirective:
    """冲动信号的结构化表示，不再产出话术文字。

    intent: 引擎决定的意图
    target: 概念对象（不是完整句子）
    emotional_tone: LLM 开口时的语气
    """

    intent: ImpulseIntent
    target_concept: str
    emotional_tone: str = "neutral"
    priority: float = 0.0
    affective_mode: str = "direct"  # "direct" / "caring" / "metaphor" / "postpone"


# ── 认知状态 ────────────────────────────────────────────────

@dataclasses.dataclass
class CognitiveState:
    """一次聊天轮次中，引擎已经做完的决策。

    检索管线构建此对象，prompt 构建器消费它。
    LLM 收到的是：事实 / 参考 / 背景 / 冲动信号（如有），
    以及引擎对用户情绪和今日话题的判断。
    """

    # 分层记忆
    primary: list[MemoryDirective] = dataclasses.field(default_factory=list)
    secondary: list[MemoryDirective] = dataclasses.field(default_factory=list)
    background: list[MemoryDirective] = dataclasses.field(default_factory=list)

    # 引擎已过滤（这些记忆不被写入 prompt）
    suppressed_ids: set[str] = dataclasses.field(default_factory=set)

    # 冲动信号
    impulse: Optional[ImpulseDirective] = None

    # 引擎对用户的状态判断
    user_mood: Optional[str] = None  # "positive" / "negative" / "neutral"

    # 情境情感态 — 引擎对当前对话气氛的判断
    affective_context: Optional[str] = None
    # "intimate" / "focused_work" / "casual_chat" / "conflict"

    # 今日话题（引擎从今日记忆中提取）
    today_topics: list[str] = dataclasses.field(default_factory=list)

    # 活跃冲突（旧记忆与新增记忆矛盾）
    active_conflicts: list[tuple[str, str, str]] = dataclasses.field(
        default_factory=list
    )
    # (tag, 旧版本摘要, 新版本摘要)

    # 行为预测（引擎对用户下一步行为的预判）
    mirror_prediction: Optional[dict] = None
    # {"next_intent": "recall", "shift_topics": ["话题A", "话题B"]} 或 None

    # 人格画像中与当前轮次相关的条目（str 或 {"content": str, "type": str}）
    personality_notes: list = dataclasses.field(default_factory=list)

    # 情绪反转事件（引擎检测到的用户情绪变化）
    emotional_reversals: list[dict] = dataclasses.field(default_factory=list)
    # [{"tag": "...", "old_valence": "positive", "new_valence": "negative"}, ...]

    # ── 便捷构造 ──────────────────────────────────────────

    @classmethod
    def empty(cls) -> CognitiveState:
        return cls()

    def add_fact(self, mem: dict, *, certainty: float = 0.9) -> MemoryDirective:
        """从原始记忆 dict 构造 fact 级别指令。"""
        d = self._from_mem_dict(mem, role="fact", certainty=certainty)
        self.primary.append(d)
        return d

    def add_reference(self, mem: dict, *, certainty: float = 0.6) -> MemoryDirective:
        """构造 reference 级别指令。"""
        d = self._from_mem_dict(mem, role="reference", certainty=certainty)
        self.secondary.append(d)
        return d

    def add_background(self, mem: dict) -> MemoryDirective:
        """构造 background 级别指令。"""
        d = self._from_mem_dict(mem, role="background", certainty=0.3)
        self.background.append(d)
        return d

    @staticmethod
    def _from_mem_dict(
        mem: dict, *, role: MemoryRole, certainty: float
    ) -> MemoryDirective:
        meta = mem.get("metadata") or {}
        ts = meta.get("timestamp", 0)
        time_hint = None
        if ts:
            try:
                from datetime import datetime

                # 兼容字符串时间戳（如 "2026-05-31 10:29:47"）和数字时间戳
                if isinstance(ts, str):
                    dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                else:
                    dt = datetime.fromtimestamp(float(ts))
                now = datetime.now()
                days_ago = (now - dt).days
                if days_ago == 0:
                    time_hint = "今天"
                elif days_ago == 1:
                    time_hint = "昨天"
                elif days_ago < 7:
                    time_hint = f"{days_ago}天前"
                elif days_ago < 30:
                    time_hint = f"{days_ago // 7}周前"
                else:
                    time_hint = dt.strftime("%Y-%m-%d")
            except (OSError, ValueError):
                pass

        emotional_context = None
        ei = meta.get("emotional_intensity", 0)
        valence = meta.get("emotion_valence_bin", "") or ""
        if ei and valence == "positive" and ei >= 2:
            emotional_context = "用户当时情绪积极"
        elif ei and valence == "negative" and ei >= 2:
            emotional_context = "用户当时情绪低落"

        return MemoryDirective(
            memory_id=mem.get("id", ""),
            summary=meta.get("summary", "") or mem.get("document", "") or "",
            document=mem.get("document", "") or "",
            role=role,
            certainty=certainty,
            emotional_context=emotional_context,
            time_hint=time_hint,
            source=mem.get("source", "semantic"),
            emotional_intensity=ei,
            emotion_valence_bin=valence,
        )

    def add_conflict(self, tag: str, old: str, new: str):
        self.active_conflicts.append((tag, old, new))

    def set_impulse(self, intent: ImpulseIntent, target: str, tone: str = "neutral"):
        self.impulse = ImpulseDirective(
            intent=intent, target_concept=target, emotional_tone=tone
        )


# ── 回路调度输出数据结构 ──────────────────────────────────


@dataclasses.dataclass
class UserMessageAnalysis:
    """引擎对用户消息的第一眼分析——意图、情绪、紧迫度。"""
    intent: str = "casual"
    # "recall" / "ask_fact" / "emotional_sharing" / "casual" / "request" / "conflict" / "meta"
    emotion: str = "neutral"
    # "positive" / "negative" / "intimate" / "neutral" / "frustrated"
    urgency: float = 0.0     # 0.0~1.0
    topics: list = dataclasses.field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    emotion_intensity: float = 0.0  # 0.0~1.0，基于感叹号/emoji/程度副词


@dataclasses.dataclass
class GatingDecision:
    """回路④：响应门控决策。

    决定哪些内容适合现在给 LLM 看，以及用什么语气。
    LLM 只看到 fact 级记忆和未被压制的冲动。
    """
    tone: str = "warm"
    # "warm" / "caring" / "direct" / "soft" / "neutral"
    formality: float = 0.3   # 0.0=随意  1.0=正式
    intimacy: float = 0.0    # 0.0~1.0
    response_mode: str = "auto"
    # "auto" / "question_first" / "soothe" / "direct_answer" / "confirm"
    suppression_reasons: list = dataclasses.field(default_factory=list)
    memories_to_show: list = dataclasses.field(default_factory=list)
    impulses_to_show: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AiState:
    """AI 在本次回应中的表达状态（回复后捕获）。"""
    emotion: str = "neutral"    # AI 回复的情绪倾向
    tone: str = "warm"          # AI 使用的语气
    formality: float = 0.3      # 正式度 0~1
    emotion_intensity: float = 0.0


@dataclasses.dataclass
class UtteranceSpec:
    """引擎最终交给 LLM 渲染的完整规格。"""
    user: UserMessageAnalysis = dataclasses.field(default_factory=UserMessageAnalysis)
    ai: AiState = dataclasses.field(default_factory=AiState)
    memories: list = dataclasses.field(default_factory=list)
    # 只有 role="fact" 且门控放行的 MemoryDirective
    reference_memories: list = dataclasses.field(default_factory=list)
    # role="reference" 级记忆，渲染为【参考信息】段
    impulses: list = dataclasses.field(default_factory=list)
    # 门控放行的 ImpulseDirective
    gate: GatingDecision = dataclasses.field(default_factory=GatingDecision)
    timeline_recent: Optional[list] = None
    session_context: Optional[str] = None
    personality_notes: list = dataclasses.field(default_factory=list)
    personality_notes_ai: list = dataclasses.field(default_factory=list)
    """AI 自我模型的表达习惯描述（source=ai 的人格标签）。"""
    mirror_prediction: Optional[dict] = None
    emotional_reversals: list[dict] = dataclasses.field(default_factory=list)
    topic_notes: list[dict] = dataclasses.field(default_factory=list)
    relationship: Optional[RelationshipState] = None
    stale_context: list = dataclasses.field(default_factory=list)
    """v2.1: 被取代但保留为背景参考的记忆（stale=True 但不屏蔽）"""
    woven_context: Optional["WovenContext"] = None
