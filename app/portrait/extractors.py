"""画像特征提取器 — 纯函数集合，不依赖外部状态。

从 app/background/distill.py 迁移的纯函数 + 新增的画像专用特征提取器。
所有函数无副作用，输入→输出，可独立测试。
"""

import math
import re
from datetime import datetime, timezone
from typing import Optional


# ── 从 distill.py 迁移的纯函数 ────────────────────────────

def extract_keywords(text: str, topk: int = 10) -> list[str]:
    """用 jieba TF-IDF 提取关键词，过滤停用词和短词。

    从 app/background/distill.py._extract_keywords 迁移（区别：不依赖模块级 STOP_WORDS）。
    """
    if not text or not text.strip():
        return []
    try:
        from app.brain.semantic import extract_tags
        words = extract_tags(text, topk=topk * 2)
        from app.config.settings import STOP_WORDS
        return [w for w in words if len(w) >= 2 and w.lower() not in STOP_WORDS][:topk]
    except (ImportError, Exception):
        return []


def recency_score(ts: float, now: Optional[float] = None) -> float:
    """计算时间戳的新近度分数 [0, 1]。

    从 app/background/distill.py._recency_score 迁移。
    1.0 = 刚刚, 0.0 = 90天前或更早。
    """
    if now is None:
        now = datetime.now().timestamp()
    age_days = (now - ts) / 86400.0
    if age_days <= 0:
        return 1.0
    if age_days >= 90:
        return 0.0
    return max(0.0, 1.0 - age_days / 90.0)


def compute_confidence(
    evidence_count: int,
    recency: float,
    consistency: float = 1.0,
) -> float:
    """综合计算画像条目的置信度 [0, 1]。

    从 app/background/distill.py._compute_confidence 迁移。

    Args:
        evidence_count: 证据数量
        recency: 新近度分数
        consistency: 一致性分数（多条证据的一致性）
    """
    evidence_score = min(1.0, math.log(evidence_count + 1) / math.log(10))
    return round(evidence_score * 0.4 + recency * 0.4 + consistency * 0.2, 2)


# ── 画像专用特征提取器 ──────────────────────────────────

def detect_emotion_flip(prev_emotion: str, new_emotion: str) -> bool:
    """检测情绪是否发生了翻转（正→负 或 负→正）。"""
    POSITIVE = {"positive", "happy", "excited", "intimate"}
    NEGATIVE = {"negative", "frustrated", "sad", "angry", "anxious"}

    prev_is_pos = prev_emotion.lower() in POSITIVE
    prev_is_neg = prev_emotion.lower() in NEGATIVE
    new_is_pos = new_emotion.lower() in POSITIVE
    new_is_neg = new_emotion.lower() in NEGATIVE

    return (prev_is_pos and new_is_neg) or (prev_is_neg and new_is_pos)


def compute_tag_density(tag_counts: dict[str, int], days: int) -> dict[str, float]:
    """计算标签密度（每天出现次数）。

    Args:
        tag_counts: {tag: count_in_period}
        days: 观察窗口天数

    Returns:
        {tag: density}
    """
    if days <= 0:
        return {}
    return {tag: count / days for tag, count in tag_counts.items()}


def classify_tag_heat(tag: str, count: int, days: int, last_seen_days: float) -> str:
    """分类标签热度。

    Returns:
        'hot' | 'warm' | 'cooling'
    """
    density = count / max(days, 1)
    if density >= 1.0 and last_seen_days <= 3:
        return "hot"
    elif last_seen_days <= 7:
        return "warm"
    else:
        return "cooling"


def extract_emotion_category(valence: float) -> str:
    """将 Russell 2D valence 值映射到情绪类别。

    Args:
        valence: -1.0 ~ 1.0 (负向 ~ 正向)

    Returns:
        'positive' | 'neutral' | 'negative'
    """
    if valence > 0.2:
        return "positive"
    elif valence < -0.2:
        return "negative"
    else:
        return "neutral"


def tag_similarity(tags_a: list[str], tags_b: list[str]) -> float:
    """计算两个标签列表的 Jaccard 相似度。"""
    if not tags_a or not tags_b:
        return 0.0
    set_a = set(tags_a)
    set_b = set(tags_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
