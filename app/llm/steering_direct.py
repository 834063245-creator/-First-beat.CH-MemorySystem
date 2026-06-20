# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: d7e3f1a9

"""
直接向量注入引擎 — 16 模块绕过文本中转，结构化数值直出残差向量。

核心洞见:
  152K × 3584 的 embed 表是模型语义空间的"调色盘"。
  不需要写中文句子 → tokenize → embed → mean pool。
  直接在空间里做向量运算：锚点插值、概念方向、类别查表。

四种构造方法:
  方法一: 锚点插值 — 对标量特征 (emotion valence, trust, formality)
  方法二: 概念方向 — 对标量变化量 (drift, 增量预测)
  方法三: 类别查表 — 对离散特征 (tone, mode, intent)
  方法四: Tag 混合 — 对标签列表 (portrait tags, interest, impulse)

Trajectory 系统:
  每个模块不产出一个向量，而是产出一条 28 层轨迹。
  3 参数控 448 自由度: base_vector + shape + intensity。

用法:
  from app.llm.steering_direct import build_steering_trajectory

  trajectory = build_steering_trajectory(utterance_spec)
  # trajectory.shape = (n_layer, n_embd), 可直接喂给 llama_set_adapter_cvec
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# §1 概念向量构建器
# ═══════════════════════════════════════════════════════════════════

class ConceptVectorBuilder:
    """从 embed 表预计算锚点/方向，运行时零文本 O(1) 产出语义向量。

    初始化时预计算 ~100 个 token 的 embed 和方向向量（<100ms），
    运行时所有模块直接做向量运算，不碰 tokenizer。
    """

    def __init__(self):
        from app.llm.qwen_embed import get_qwen_embedder
        self._emb = get_qwen_embedder()
        self._dim = self._emb.dim  # 3584

        # ── 预计算所有锚点中心 ──
        self._anchors: dict[str, np.ndarray] = {}
        self._directions: dict[str, np.ndarray] = {}
        self._categories: dict[str, dict[str, np.ndarray]] = {}

        self._precompute_emotion()
        self._precompute_trust()
        self._precompute_formality()
        self._precompute_drift()
        self._precompute_tone()
        self._precompute_mode()
        self._precompute_intent()
        self._precompute_generic()

        logger.info(
            "ConceptVectorBuilder ready: %d anchors, %d directions, %d category-sets, dim=%d",
            len(self._anchors), len(self._directions),
            sum(len(v) for v in self._categories.values()), self._dim,
        )

    # ── 预计算 ──────────────────────────────────────────────

    def _center(self, words: list[str]) -> np.ndarray:
        """多 token 的 mean-pooled 嵌入中心 → 单位向量。"""
        vecs = [self._emb.embed(w) for w in words]
        v = np.mean(vecs, axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _direction(self, pos_words: list[str], neg_words: list[str]) -> np.ndarray:
        """正负两极的中心差 → 单位方向向量。"""
        d = self._center(pos_words) - self._center(neg_words)
        n = float(np.linalg.norm(d))
        return (d / n).astype(np.float32) if n > 0 else d.astype(np.float32)

    def _precompute_emotion(self):
        # Russell 2D: valence (正负) + arousal (高低)
        self._directions["valence"] = self._direction(
            ["开心", "快乐", "满足", "幸福", "温暖", "欣慰", "期待", "感动", "轻松", "舒服"],
            ["难过", "痛苦", "绝望", "愤怒", "焦虑", "恐惧", "沮丧", "失落", "孤独", "崩溃"],
        )
        self._directions["arousal"] = self._direction(
            ["激动", "兴奋", "紧张", "焦虑", "愤怒", "惊喜", "狂喜", "惊恐", "急迫", "热切"],
            ["平静", "放松", "疲惫", "无聊", "困倦", "安详", "淡漠", "麻木", "慵懒", "镇定"],
        )

    def _precompute_trust(self):
        self._anchors["trust_high"] = self._center(
            ["信任", "亲近", "熟悉", "默契", "放心", "依赖", "坦诚", "理解", "包容", "支持"])
        self._anchors["trust_low"] = self._center(
            ["陌生", "疏离", "怀疑", "戒备", "警惕", "隔阂", "冷淡", "防范", "不信任", "距离"])
        self._anchors["closeness_high"] = self._center(
            ["亲密", "亲近", "密切", "紧密", "深厚", "浓烈", "挚友", "知己", "家人", "贴心"])
        self._anchors["familiarity_high"] = self._center(
            ["熟悉", "了解", "认识", "习惯", "老友", "旧识", "熟络", "相知", "默契", "常见"])

    def _precompute_formality(self):
        self._anchors["formal_high"] = self._center(
            ["正式", "专业", "严谨", "客观", "规范", "标准", "官方", "学术", "商务", "礼仪"])
        self._anchors["formal_low"] = self._center(
            ["随意", "轻松", "口语", "亲切", "日常", "自由", "随性", "白话", "聊天", "闲谈"])

    def _precompute_drift(self):
        self._directions["spend"] = self._direction(
            ["投入", "付费", "购买", "投资", "花钱", "值得", "效率", "氪金", "消费", "升级"],
            ["克制", "节约", "省着", "不买", "观望", "犹豫", "精打细算", "控制", "收敛", "保守"],
        )
        self._directions["frugal"] = self._direction(
            ["节省", "免费", "白嫖", "克制", "开源", "性价比", "便宜", "不花钱", "省", "抠"],
            ["奢侈", "挥霍", "浪费", "大方", "铺张", "乱花", "不差钱", "任性", "随意买", "挥金"],
        )
        self._directions["drift"] = self._direction(
            ["放弃", "随便", "将就", "退出", "算了", "不做了", "无所谓", "摆烂", "躺平", "不管了"],
            ["坚持", "执着", "追求", "精益求精", "不放弃", "认真", "努力", "拼搏", "进取", "专注"],
        )

    def _precompute_tone(self):
        self._categories["tone"] = {
            "soft":     self._center(["温柔", "共情", "理解", "柔和", "轻声", "安抚", "体贴"]),
            "caring":   self._center(["关怀", "照顾", "温暖", "呵护", "心疼", "在意", "共情"]),
            "warm":     self._center(["友好", "热情", "亲切", "温暖", "阳光", "开朗", "热络"]),
            "direct":   self._center(["直接", "简洁", "高效", "干脆", "利落", "不废话", "直白"]),
            "neutral":  self._center(["客观", "中性", "理性", "平衡", "公正", "冷静", "分析"]),
        }

    def _precompute_mode(self):
        self._categories["interaction_mode"] = {
            "casual":       self._center(["随意", "轻松", "朋友", "闲聊", "日常"]),
            "collaborator": self._center(["合作", "专业", "同事", "协作", "配合"]),
            "partner":      self._center(["亲密", "信任", "搭档", "伙伴", "战友"]),
            "teacher":      self._center(["指导", "教学", "导师", "讲解", "传授"]),
        }

    def _precompute_intent(self):
        self._categories["next_intent"] = {
            "recall":             self._center(["回忆", "想起", "之前", "记得", "追溯"]),
            "ask_fact":           self._center(["查询", "事实", "确认", "核实", "了解"]),
            "emotional_sharing":  self._center(["分享", "情绪", "感受", "倾诉", "表达"]),
            "casual":             self._center(["闲聊", "日常", "随便", "聊天", "寒暄"]),
            "conflict":           self._center(["矛盾", "冲突", "纠正", "反驳", "质疑"]),
            "request":            self._center(["请求", "需要", "帮忙", "帮我", "想要"]),
        }

    def _precompute_generic(self):
        # 身份相关锚点 (供 portrait tags 使用)
        self._anchors["identity_base"] = self._center(
            ["程序员", "工程师", "开发者", "技术", "编程", "代码"])
        self._anchors["interest_base"] = self._center(
            ["兴趣", "爱好", "关注", "喜欢", "热衷", "探索"])
        self._anchors["thinking_base"] = self._center(
            ["思考", "分析", "推理", "逻辑", "判断", "决策"])
        self._anchors["feeling_base"] = self._center(
            ["感受", "体验", "心情", "情绪", "状态", "心态"])

    # ── 方法一: 锚点插值 (标量 → 向量) ─────────────────────

    def from_scalar(
        self, value: float, anchor_high: str, anchor_low: str = "",
        *,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """标量值在高低锚点间线性插值。

        Args:
            value: 标量值
            anchor_high: 高端锚点 key (在 self._anchors 中)
            anchor_low: 低端锚点 key (空则用零向量)
            value_range: value 的取值范围

        Returns:
            单位向量 [dim]
        """
        lo, hi = value_range
        t = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.5

        v_high = self._anchors.get(anchor_high)
        if v_high is None:
            return np.zeros(self._dim, dtype=np.float32)

        if anchor_low:
            v_low = self._anchors.get(anchor_low)
            if v_low is None:
                return v_high * t
            v = v_high * t + v_low * (1.0 - t)
        else:
            v = v_high * t

        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    # ── 方法二: 概念方向 (标量变化量 → 向量) ──────────────

    def from_direction(
        self, value: float, direction_name: str,
        *,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        """标量值缩放预计算的方向向量。

        零值 → 零向量 (不加 steering)。适合"变化量"语义。

        Args:
            value: 标量值 (在 value_range 内)
            direction_name: 方向 key (在 self._directions 中)

        Returns:
            向量 [dim] (非归一化，大小 = |value|)
        """
        d = self._directions.get(direction_name)
        if d is None:
            return np.zeros(self._dim, dtype=np.float32)

        lo, hi = value_range
        t = max(-1.0, min(1.0, 2.0 * (value - lo) / (hi - lo) - 1.0)) if hi > lo else 0.0
        return (d * t).astype(np.float32)

    # ── 方法三: 类别查表 (离散值 → 向量) ──────────────────

    def from_category(
        self, category: str, category_set: str,
    ) -> Optional[np.ndarray]:
        """离散类别 → 预计算的单位向量。

        Args:
            category: 类别名 (如 "soft", "collaborator")
            category_set: 类别集合名 (如 "tone", "interaction_mode")

        Returns:
            单位向量 [dim]，或 None (未知类别)
        """
        table = self._categories.get(category_set, {})
        return table.get(category)

    # ── 方法四: Tag 混合 (标签列表 → 向量) ──────────────────

    def from_tags(
        self, tags: list[str], weights: list[float] | None = None,
    ) -> Optional[np.ndarray]:
        """多标签加权混合 → 单位向量。

        每个 tag 独立 embed 后加权求和+归一化。

        Args:
            tags: 中文标签列表
            weights: 对应权重 (None = 等权)

        Returns:
            单位向量 [dim]，或 None (空标签)
        """
        if not tags:
            return None

        if weights is None:
            weights = [1.0] * len(tags)
        else:
            weights = list(weights)[:len(tags)]

        vecs = []
        w_sum = 0.0
        for tag, w in zip(tags, weights):
            if not tag or not tag.strip():
                continue
            v = self._emb.embed(tag.strip())
            vecs.append(v * w)
            w_sum += abs(w)

        if not vecs or w_sum == 0:
            return None

        v = np.sum(vecs, axis=0) / w_sum
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    # ── 便利方法: 直接 embed 文本 (过渡期用) ──────────────

    def from_text(self, text: str) -> np.ndarray:
        """文本 → embed → 单位向量。过渡期用于 memories 模块。"""
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)
        v = self._emb.embed(text.strip())
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════
# §2 Trajectory 塑形器
# ═══════════════════════════════════════════════════════════════════

class TrajectoryShaper:
    """将基向量 + shape 配置展开为 28 层的轨迹数组。

    5 种 shape 函数:
      uniform      — 全层等值 (Phase 9 已验证安全)
      gradient_up  — 浅→深线性递增 (适合门控/决策约束)
      gradient_down— 浅→深线性递减 (适合身份语义)
      peak:N       — 以第 N 层为中心的高斯峰 (精确靶向)
      early        — 指数衰减，强于前 10 层
      late         — 指数上升，强于后 10 层
    """

    @staticmethod
    def shape_fn(shape: str, n_layer: int = 28) -> np.ndarray:
        """返回 shape 函数在 0..n_layer-1 上的值数组 [n_layer]。

        Args:
            shape: "uniform" | "gradient_up" | "gradient_down" |
                   "peak:N[:W]" | "early" | "late"
            n_layer: 总层数

        Returns:
            float32[n_layer], 范围 [0, 1]
        """
        layers = np.arange(n_layer, dtype=np.float32)

        if shape == "uniform":
            return np.ones(n_layer, dtype=np.float32)

        if shape == "gradient_up":
            # L1=0.15 → L28=1.0 线性递增
            return (0.15 + 0.85 * layers / (n_layer - 1)).astype(np.float32) if n_layer > 1 \
                else np.ones(1, dtype=np.float32)

        if shape == "gradient_down":
            # L1=1.0 → L28=0.15 线性递减
            return (1.0 - 0.85 * layers / (n_layer - 1)).astype(np.float32) if n_layer > 1 \
                else np.ones(1, dtype=np.float32)

        if shape == "early":
            # exp(-layer/5): L1≈0.82, L10≈0.14, L28≈0.004
            return np.exp(-layers / 5.0).astype(np.float32)

        if shape == "late":
            # exp(-(27-layer)/5): L1≈0.004, L18≈0.14, L28≈0.82
            return np.exp(-(n_layer - 1 - layers) / 5.0).astype(np.float32)

        if shape.startswith("peak:"):
            # peak:N[:W] 如 "peak:12" 或 "peak:12:3"
            parts = shape.split(":")
            center = float(parts[1]) if len(parts) > 1 else 14.0
            width = float(parts[2]) if len(parts) > 2 else 3.0
            # 高斯: exp(-(x-center)² / (2*width²))
            return np.exp(-((layers - center) ** 2) / (2.0 * width ** 2)).astype(np.float32)

        logger.warning("TrajectoryShaper: unknown shape '%s', fallback to uniform", shape)
        return np.ones(n_layer, dtype=np.float32)

    @staticmethod
    def expand(
        base_vec: np.ndarray,
        shape: str,
        intensity: float,
        n_layer: int = 28,
    ) -> np.ndarray:
        """基向量 → 28 层轨迹。

        Args:
            base_vec: 单位向量 [d_model]
            shape: shape 名称
            intensity: 全局强度倍率 (0~1 建议)
            n_layer: 总层数

        Returns:
            trajectory[n_layer, d_model]
        """
        sf = TrajectoryShaper.shape_fn(shape, n_layer)  # [n_layer]
        trajectory = np.outer(sf * intensity, base_vec)  # [n_layer, d_model]
        return trajectory.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# §3 模块配置 — 16 模块 × (extractor + layer_range + shape + alpha)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModuleSteeringConfig:
    """单个模块的 steering 配置。

    layer_start/end 为 1-indexed (与 llama_set_adapter_cvec 一致)。
    shape 决定 28 层上的强度分布。
    alpha 为基强度系数（来自 Phase 8 实验标定）。
    """
    name: str                        # 模块名 (日志用)
    layer_start: int                 # 注入起始层 (1-indexed)
    layer_end: int                   # 注入结束层 (1-indexed, inclusive)
    shape: str = "uniform"           # trajectory shape
    alpha: float = 0.08              # 基强度系数
    extractor: str = ""              # 提取函数名


# ── 16 模块配置表 ───────────────────────────────────────────

MODULE_DIRECT_CONFIG: list[ModuleSteeringConfig] = [
    # ═══ 浅层: 身份语义 (L3-5) ═══
    ModuleSteeringConfig(
        name="portrait_identity",   layer_start=3,  layer_end=5,   shape="early",
        alpha=0.08, extractor="identity"),
    ModuleSteeringConfig(
        name="portrait_identity_ai", layer_start=3, layer_end=5,   shape="early",
        alpha=0.05, extractor="ai_identity"),

    # ═══ 语义层: 相关记忆 (L5-12) ═══
    ModuleSteeringConfig(
        name="relevant_memory_1",   layer_start=5,  layer_end=12,  shape="gradient_down",
        alpha=0.06, extractor="memory_1"),
    ModuleSteeringConfig(
        name="relevant_memory_2",   layer_start=5,  layer_end=12,  shape="gradient_down",
        alpha=0.06, extractor="memory_2"),
    ModuleSteeringConfig(
        name="relevant_memory_3",   layer_start=5,  layer_end=12,  shape="gradient_down",
        alpha=0.06, extractor="memory_3"),
    ModuleSteeringConfig(
        name="relevant_memory_4",   layer_start=5,  layer_end=12,  shape="gradient_down",
        alpha=0.06, extractor="memory_4"),
    ModuleSteeringConfig(
        name="relevant_memory_5",   layer_start=5,  layer_end=12,  shape="gradient_down",
        alpha=0.06, extractor="memory_5"),

    # ═══ 中层: 情绪+冲动 (L8-15) ═══
    ModuleSteeringConfig(
        name="portrait_emotion",    layer_start=8,  layer_end=15,  shape="peak:12:4",
        alpha=0.10, extractor="emotion"),
    ModuleSteeringConfig(
        name="impulse_signal",      layer_start=8,  layer_end=15,  shape="peak:10:3",
        alpha=0.05, extractor="impulse"),

    # ═══ 中深层: 偏移率+兴趣 (L15-22) ═══
    ModuleSteeringConfig(
        name="drift_context",       layer_start=15, layer_end=22,  shape="gradient_up",
        alpha=0.07, extractor="drift"),
    ModuleSteeringConfig(
        name="portrait_interest",   layer_start=15, layer_end=22,  shape="uniform",
        alpha=0.07, extractor="interest"),

    # ═══ 深层: 关系+镜像 (L18-27) ═══
    # Calibrated 2026-06-21: gradient_down produces best empathy ("不要怀疑自己")
    # for relationship — trust/closeness encoding benefits from stronger shallow
    # presence in this layer range, unlike tone which needs deep-layer gradient_up.
    ModuleSteeringConfig(
        name="relationship_state",  layer_start=18, layer_end=27,  shape="gradient_down",
        alpha=0.08, extractor="relationship"),
    ModuleSteeringConfig(
        name="self_mirror",         layer_start=20, layer_end=27,  shape="late",
        alpha=0.05, extractor="mirror"),

    # ═══ 中深层→输出层: 门控语气 (L16-28, gradient_up) ═══
    # Calibrated 2026-06-21: gradient_up produces strongest empathetic opening
    # ("不要怀疑自己") vs late ("加油!" closing). Broader range lets tone
    # influence emotional framing earlier while strengthening at output layers.
    ModuleSteeringConfig(
        name="gate_tone",           layer_start=16, layer_end=28,  shape="gradient_up",
        alpha=0.12, extractor="tone"),
    ModuleSteeringConfig(
        name="behavior_predictor",  layer_start=24, layer_end=28,  shape="late",
        alpha=0.05, extractor="predictor"),
]


# ═══════════════════════════════════════════════════════════════════
# §4 模块向量提取器 — 16 个 extract_*() 函数
# ═══════════════════════════════════════════════════════════════════
# 每个函数签名: (utterance_spec, cvb: ConceptVectorBuilder) → Optional[np.ndarray]
# 返回 None 表示该模块本轮无产出，跳过注入。

def _extract_identity(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """画像身份 (usr1 核心特征) — tags → 加权混合。"""
    portrait_stable = getattr(spec, "portrait_stable", "") or ""
    if not portrait_stable or not isinstance(portrait_stable, str):
        return None
    # 从 portrait_stable 文本提取关键标签
    tags = _tokenize_cn_phrases(portrait_stable)
    if not tags:
        return cvb.from_text(portrait_stable[:150])
    return cvb.from_tags(tags[:5])


def _extract_ai_identity(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """AI 自我身份 (ai1) — notes → 混合。"""
    notes = getattr(spec, "personality_notes_ai", []) or []
    texts = []
    for n in notes[:2]:
        if isinstance(n, dict):
            texts.append(str(n.get("content", ""))[:80])
        elif isinstance(n, str):
            texts.append(n[:80])
    combined = " ".join(texts).strip()
    if not combined:
        return None
    return cvb.from_text(combined)


def _extract_emotion(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """情绪 (Russell 2D) — valence/arousal 方向 + 画像情绪上下文。

    这是概念方向方法（方法二）的核心应用。
    valence → valence_direction, arousal → arousal_direction, 加权求和。
    """
    user = getattr(spec, "user", None)
    valence = 0.0
    arousal = 0.5
    intensity = 0.0

    if user:
        emotion_str = getattr(user, "emotion", "") or ""
        # 从情绪字符串推断 Russell 坐标
        v, a = _emotion_to_russell(emotion_str)
        valence = v
        arousal = a
        intensity = getattr(user, "emotion_intensity", 0.0) or 0.0

    # 如果情绪强度很低，减弱 steering
    if abs(valence) < 0.15 and abs(arousal - 0.5) < 0.15 and intensity < 0.2:
        return None

    vec_v = cvb.from_direction(valence, "valence", value_range=(-1.0, 1.0))
    vec_a = cvb.from_direction(arousal - 0.5, "arousal", value_range=(-0.5, 0.5))

    # 合并: valence 权重 0.6, arousal 权重 0.4
    vec = vec_v * 0.6 + vec_a * 0.4
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return None
    return (vec / n).astype(np.float32)


def _extract_impulse(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """冲动信号 — target_concept 标签 → embed。"""
    impulses = getattr(spec, "impulses", []) or []
    if not impulses:
        return None
    tags = []
    for imp in impulses[:2]:
        target = ""
        if isinstance(imp, dict):
            target = imp.get("target_concept", "")
        else:
            target = getattr(imp, "target_concept", "")
        if target and target.strip():
            tags.append(target.strip())
    return cvb.from_tags(tags) if tags else None


def _extract_drift(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """偏移率 — spend/frugal/drift 方向加权。

    drift_text 格式: "偏移: frugal(+25%) 连续3轮节省倾向"
    解析方向和幅度。
    """
    drift_text = getattr(spec, "drift_text", "") or ""
    if not drift_text:
        return None

    direction, magnitude = _parse_drift_text(drift_text)
    if direction == "neutral" or magnitude < 0.05:
        return None

    if direction.startswith("spend"):
        return cvb.from_direction(magnitude, "spend", value_range=(0.0, 1.0))
    elif direction.startswith("frugal"):
        return cvb.from_direction(magnitude, "frugal", value_range=(0.0, 1.0))
    elif direction.startswith("drift"):
        return cvb.from_direction(magnitude * 1.5, "drift", value_range=(0.0, 1.5))
    return None


def _extract_interest(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """兴趣图谱 (usr5) — 人格 notes 中的兴趣标签混合。"""
    notes = getattr(spec, "personality_notes", []) or []
    tags = []
    for n in notes[:3]:
        content = ""
        if isinstance(n, dict):
            content = str(n.get("content", ""))
        elif isinstance(n, str):
            content = n
        if content:
            tags.extend(_tokenize_cn_phrases(content))
    if not tags:
        return None
    return cvb.from_tags(list(dict.fromkeys(tags))[:5])  # 去重保序


def _extract_relationship(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """关系状态 — trust/closeness/familiarity 锚点插值 + mode 查表。"""
    rs = getattr(spec, "relationship", None)
    if rs is None:
        return None

    trust = getattr(rs, "trust", 0.5)
    closeness = getattr(rs, "closeness", 0.0)
    mode = getattr(rs, "interaction_mode", "casual")

    # trust 锚点插值
    vec_trust = cvb.from_scalar(trust, "trust_high", "trust_low")
    vec_close = cvb.from_scalar(closeness, "closeness_high")
    vec_mode = cvb.from_category(mode, "interaction_mode")

    # 加权混合: trust 50% + closeness 20% + mode 30%
    parts = [vec_trust * 0.5, vec_close * 0.2]
    if vec_mode is not None:
        parts.append(vec_mode * 0.3)
    else:
        # mode 未知时，重新分配权重给 trust
        parts[0] = vec_trust * 0.65
        parts[1] = vec_close * 0.35

    vec = sum(parts)
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return None
    return (vec / n).astype(np.float32)


def _extract_mirror(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """自我镜像 — 历史回应文本 embed（过渡期用 from_text，后续 Qdrant 直出）。"""
    text = getattr(spec, "self_mirror_text", "") or ""
    if not text or not text.strip():
        return None
    return cvb.from_text(text.strip()[:150])


def _extract_tone(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """门控语气 — tone 类别查表 + formality 标量混合。"""
    gate = getattr(spec, "gate", None)
    if gate is None:
        return None

    tone = getattr(gate, "tone", "warm")
    formality = getattr(gate, "formality", 0.3)

    vec_tone = cvb.from_category(tone, "tone")
    vec_form = cvb.from_scalar(formality, "formal_high", "formal_low")

    if vec_tone is None:
        return None

    # tone 70% + formality 30%
    vec = vec_tone * 0.7 + vec_form * 0.3
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return None
    return (vec / n).astype(np.float32)


def _extract_predictor(spec, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """行为预测 — next_intent 类别查表。"""
    mp = getattr(spec, "mirror_prediction", None) or {}
    if not mp:
        return None

    nexts = mp.get("next_intents") or [mp.get("next_intent", "")]
    if not nexts or not nexts[0]:
        return None

    vec = cvb.from_category(nexts[0], "next_intent")
    if vec is None:
        return None

    # 如果有第二个预测意图，50-50 混合
    if len(nexts) > 1 and nexts[1]:
        vec2 = cvb.from_category(nexts[1], "next_intent")
        if vec2 is not None:
            vec = (vec + vec2) * 0.5
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec = vec / n

    return vec.astype(np.float32)


def _extract_memory(spec, idx: int, cvb: ConceptVectorBuilder) -> Optional[np.ndarray]:
    """相关记忆 — 摘要文本 embed。

    TODO: 后续改为 Qdrant 直出嵌入向量（方法三）。
    当前过渡方案: 摘要文本 → embed。
    """
    memories = getattr(spec, "memories", []) or []
    if idx >= len(memories):
        return None
    mem = memories[idx]
    if isinstance(mem, dict):
        summary = mem.get("summary", "") or mem.get("document", "") or ""
    else:
        summary = getattr(mem, "summary", "") or getattr(mem, "document", "") or ""
    if not summary or not str(summary).strip():
        return None
    return cvb.from_text(str(summary)[:150])


# ── 提取器注册表 ────────────────────────────────────────────

_EXTRACTOR_REGISTRY: dict[str, callable] = {
    "identity":      _extract_identity,
    "ai_identity":   _extract_ai_identity,
    "emotion":       _extract_emotion,
    "impulse":       _extract_impulse,
    "drift":         _extract_drift,
    "interest":      _extract_interest,
    "relationship":  _extract_relationship,
    "mirror":        _extract_mirror,
    "tone":          _extract_tone,
    "predictor":     _extract_predictor,
    "memory_1":      lambda s, c: _extract_memory(s, 0, c),
    "memory_2":      lambda s, c: _extract_memory(s, 1, c),
    "memory_3":      lambda s, c: _extract_memory(s, 2, c),
    "memory_4":      lambda s, c: _extract_memory(s, 3, c),
    "memory_5":      lambda s, c: _extract_memory(s, 4, c),
}


# ═══════════════════════════════════════════════════════════════════
# §5 主入口 — 构建完整 steering trajectory
# ═══════════════════════════════════════════════════════════════════

def build_steering_trajectory(
    utterance_spec,
    n_layer: int = 28,
    n_embd: int = 3584,
    global_strength: float = 1.0,
) -> np.ndarray:
    """从 UtteranceSpec 直接构建 28 层 steering trajectory。

    每个模块:
      1. extractor 从 utterance_spec 提取结构化值
      2. ConceptVectorBuilder 产出 d_model 语义向量 (零文本)
      3. TrajectoryShaper 展开为 28 层轨迹
      4. 所有模块轨迹累加到最终 buffer

    Args:
        utterance_spec: 引擎产出的 UtteranceSpec
        n_layer: 模型层数 (qwen2.5 = 28)
        n_embd: 隐藏维度 (qwen2.5 = 3584)
        global_strength: 全局强度倍率 (来自 STEERING_STRENGTH 配置)

    Returns:
        np.ndarray[n_layer, n_embd] — 可直接展平喂给 llama_set_adapter_cvec
    """
    cvb = _get_cvb()
    trajectory = np.zeros((n_layer, n_embd), dtype=np.float32)

    active_count = 0

    for cfg in MODULE_DIRECT_CONFIG:
        extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)
        if extractor_fn is None:
            logger.warning("Unknown extractor '%s' for module '%s'", cfg.extractor, cfg.name)
            continue

        try:
            base_vec = extractor_fn(utterance_spec, cvb)
        except Exception:
            logger.debug("Extractor '%s' failed for module '%s'", cfg.extractor, cfg.name, exc_info=True)
            continue

        if base_vec is None:
            continue

        # TrajectoryShaper 展开
        mod_traj = TrajectoryShaper.expand(
            base_vec, cfg.shape, cfg.alpha * global_strength, n_layer,
        )

        # 只累加到模块配置的层范围
        l0 = max(0, cfg.layer_start - 1)       # 0-indexed
        l1 = min(n_layer, cfg.layer_end)        # exclusive
        trajectory[l0:l1] += mod_traj[l0:l1]
        active_count += 1

    if active_count == 0:
        logger.debug("build_steering_trajectory: 0 active modules")
    else:
        logger.debug("build_steering_trajectory: %d/%d modules active",
                     active_count, len(MODULE_DIRECT_CONFIG))

    return trajectory


def trajectory_to_cvec_buffer(trajectory: np.ndarray) -> np.ndarray:
    """将 (n_layer, n_embd) trajectory 展平为 llama_set_adapter_cvec 所需格式。

    Returns:
        np.ndarray[n_layer * n_embd], float32, C-contiguous
    """
    return np.ascontiguousarray(trajectory.ravel(), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# §6 辅助函数
# ═══════════════════════════════════════════════════════════════════

# ── 全局单例 ────────────────────────────────────────────────

_cvb: Optional[ConceptVectorBuilder] = None


def _get_cvb() -> ConceptVectorBuilder:
    global _cvb
    if _cvb is None:
        _cvb = ConceptVectorBuilder()
    return _cvb


# ── 情绪字符串 → Russell 坐标 ──────────────────────────────

# 从 emotion.py 的 _VALENCE_AROUSAL_MAP 摘取关键映射
_EMOTION_VA_MAP: dict[str, tuple[float, float]] = {
    # 正面+高唤醒
    "开心": (0.80, 0.70), "激动": (0.90, 0.85), "惊喜": (0.90, 0.80),
    "兴奋": (0.95, 0.90), "快乐": (0.85, 0.70), "期待": (0.70, 0.60),
    "感动": (0.90, 0.65), "温暖": (0.80, 0.40), "希望": (0.70, 0.50),
    "勇气": (0.70, 0.70), "坚定": (0.60, 0.60), "轻松": (0.70, 0.20),
    "舒服": (0.75, 0.20), "满足": (0.70, 0.25), "欣慰": (0.65, 0.30),
    "爽": (0.85, 0.65), "赞": (0.70, 0.55), "完美": (0.85, 0.60),
    "高兴": (0.80, 0.65), "喜悦": (0.80, 0.60), "愉悦": (0.80, 0.55),
    "放心": (0.60, 0.15),
    # 正面+低唤醒
    "安静": (0.50, 0.15), "安详": (0.55, 0.10), "慵懒": (0.50, 0.20),
    "平静": (0.50, 0.15), "释然": (0.60, 0.25), "安心": (0.65, 0.15),
    # 负面+高唤醒
    "愤怒": (-0.85, 0.85), "焦虑": (-0.70, 0.75), "恐惧": (-0.85, 0.80),
    "紧张": (-0.60, 0.70), "烦躁": (-0.65, 0.65), "生气": (-0.75, 0.70),
    "恼火": (-0.70, 0.70), "暴躁": (-0.80, 0.80), "崩溃": (-0.90, 0.90),
    "压力": (-0.50, 0.60), "惶恐": (-0.80, 0.75),
    # 负面+低唤醒
    "难过": (-0.80, 0.25), "伤心": (-0.85, 0.30), "痛苦": (-0.90, 0.35),
    "绝望": (-0.95, 0.25), "失望": (-0.70, 0.30), "失落": (-0.65, 0.25),
    "委屈": (-0.60, 0.30), "无奈": (-0.45, 0.20), "疲惫": (-0.30, 0.10),
    "累": (-0.30, 0.10), "沮丧": (-0.70, 0.30), "悲观": (-0.75, 0.20),
    "厌倦": (-0.55, 0.25), "无聊": (-0.30, 0.15), "孤独": (-0.70, 0.20),
    "寂寞": (-0.65, 0.20), "郁闷": (-0.60, 0.25), "憋屈": (-0.55, 0.40),
    "遗憾": (-0.45, 0.20), "内疚": (-0.60, 0.35), "挫败": (-0.65, 0.40),
    "尴尬": (-0.40, 0.35), "担心": (-0.50, 0.45), "困惑": (-0.30, 0.30),
    "迷茫": (-0.40, 0.25), "后悔": (-0.55, 0.35),
    # 中性
    "好奇": (0.30, 0.45), "惊讶": (0.10, 0.55), "正常": (0.10, 0.20),
    "一般": (0.0, 0.15), "neutral": (0.0, 0.30),
    # 复合/口语
    "emo": (-0.50, 0.40), "破防": (-0.85, 0.80), "无语": (-0.40, 0.30),
    "烦": (-0.55, 0.55), "烦人": (-0.60, 0.60), "烦死": (-0.75, 0.75),
    "气人": (-0.65, 0.60), "气死": (-0.80, 0.80), "吓人": (-0.60, 0.65),
    "真香": (0.75, 0.55), "酸": (-0.30, 0.30),
}


def _emotion_to_russell(emotion_str: str) -> tuple[float, float]:
    """情绪字符串 → (valence, arousal) in (-1..1, 0..1)。"""
    if not emotion_str:
        return (0.0, 0.5)

    emotion_str = emotion_str.strip()

    # 精确匹配
    if emotion_str in _EMOTION_VA_MAP:
        return _EMOTION_VA_MAP[emotion_str]

    # 模糊匹配：找最长的子串命中
    best = None
    best_len = 0
    for key, va in _EMOTION_VA_MAP.items():
        if key in emotion_str and len(key) > best_len:
            best = va
            best_len = len(key)

    if best is not None:
        return best

    # Fallback: 基于情绪类别的粗略映射
    emotion_lower = emotion_str
    if any(w in emotion_lower for w in ["开心", "快乐", "高兴", "喜悦", "兴奋"]):
        return (0.80, 0.65)
    if any(w in emotion_lower for w in ["难过", "伤心", "痛苦", "绝望", "崩溃"]):
        return (-0.80, 0.30)
    if any(w in emotion_lower for w in ["愤怒", "生气", "暴躁", "恼火"]):
        return (-0.80, 0.80)
    if any(w in emotion_lower for w in ["焦虑", "紧张", "担心", "恐惧"]):
        return (-0.60, 0.70)
    if any(w in emotion_lower for w in ["疲惫", "累", "无聊", "厌倦"]):
        return (-0.30, 0.10)
    if any(w in emotion_lower for w in ["温暖", "感动", "幸福", "满足"]):
        return (0.80, 0.45)
    if any(w in emotion_lower for w in ["期待", "希望", "好奇"]):
        return (0.60, 0.50)

    # positive / negative / neutral 兜底
    if "positive" in emotion_lower or "正面" in emotion_lower:
        return (0.60, 0.45)
    if "negative" in emotion_lower or "负面" in emotion_lower:
        return (-0.50, 0.40)
    if "intimate" in emotion_lower:
        return (0.70, 0.40)
    if "frustrated" in emotion_lower:
        return (-0.55, 0.55)

    return (0.0, 0.5)


# ── 漂移文本解析 ────────────────────────────────────────────

def _parse_drift_text(text: str) -> tuple[str, float]:
    """解析 drift_text → (direction, magnitude)。

    drift_text 格式: "偏移: frugal(+25%) 连续3轮节省倾向"
                     "偏移: spend(+40%) 用户连续投入"
                     "偏移: drift_放弃(-60%) 用户想放弃"
    """
    text = str(text)
    direction = "neutral"
    magnitude = 0.0

    if "spend" in text:
        direction = "spend"
    elif "frugal" in text:
        direction = "frugal"
    elif "drift" in text:
        # 判断 drift_放弃 / drift_妥协 / drift_烦躁
        for sub in ["放弃", "妥协", "烦躁"]:
            if sub in text:
                direction = f"drift_{sub}"
                break
        if direction == "drift":
            direction = "drift_放弃"  # default

    # 提取百分比
    import re
    m = re.search(r'\(([+-]?\d+)%\)', text)
    if m:
        magnitude = abs(int(m.group(1))) / 100.0
    else:
        magnitude = 0.25  # 默认幅度

    # clamp
    magnitude = max(0.0, min(1.0, magnitude))

    return direction, magnitude


# ── 中文短语切分 ────────────────────────────────────────────

def _tokenize_cn_phrases(text: str) -> list[str]:
    """从中文文本提取 2-4 字短语（简单规则，不依赖分词器）。"""
    import re
    text = str(text)
    # 移除标点和空白
    text = re.sub(r'[，。！？、；：""''（）\s\n\[\]【】★·\->]+', ' ', text)
    parts = text.split()
    phrases = []
    for part in parts:
        part = part.strip()
        if 2 <= len(part) <= 8:
            phrases.append(part)
    # 去重保序
    seen = set()
    result = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result
