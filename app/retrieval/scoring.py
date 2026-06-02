"""统一评分函数 — 替代 backend/main.py 和 app/tools/dispatch.py 中各自的内联评分。"""
import math

from app.config.settings import (
    RERANK_SEMANTIC_WEIGHT,
    RERANK_ATTENTION_WEIGHT,
    RERANK_HIT_WEIGHT,
    RERANK_LN_MAX,
)


def compute_score(
    similarity: float,
    hit_count: int,
    attention_boost: float = 0.0,
    bm25_score: float = 0.0,
    source_bonus: float = 0.0,
    error_penalty: float = 0.0,
) -> float:
    """统一评分：语义(0.7) + 注意力(可选的) + hit_count(0.3) + 辅助修正。

    Args:
        similarity: 语义相似度（0~1，如 distance → 1-similarity）
        hit_count: 命中计数
        attention_boost: 注意力偏移量（0~1）
        bm25_score: BM25 原始分数
        source_bonus: 来源加成（如文本匹配 +0.1）
        error_penalty: 错误惩罚（误差数量 × 系数）

    Returns:
        归一化分数（0~1 区间）
    """
    semantic_part = RERANK_SEMANTIC_WEIGHT * similarity
    attention_part = RERANK_ATTENTION_WEIGHT * attention_boost
    hc_bonus = RERANK_HIT_WEIGHT * min(
        math.log(max(hit_count, 0) + 1) / RERANK_LN_MAX, 1.0
    )
    return (
        semantic_part + attention_part + hc_bonus + source_bonus - error_penalty
    )
