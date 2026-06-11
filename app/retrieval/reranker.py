"""Reranker — 基于 embedding 余弦相似度的本地精排，带纠正反馈加成。"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 8,
    correction_boosts: Optional[dict[str, float]] = None,
    error_counts: Optional[dict[str, int]] = None,
    attention_boosts: Optional[dict[str, float]] = None,
    attention_weight: float = 0.0,
    portrait_boost_map: Optional[dict[str, float]] = None,
    portrait_weight: float = 0.0,
) -> list[dict]:
    """基于 embedding 余弦相似度精排，附带纠正/错误反馈 + 画像 boost。

    输入: query + candidates（每条 dict 需有 "summary" 或 "document" 字段）
         attention_boosts: {memory_id: proximity} — 注意力漂移分数
         attention_weight: 注意力分数在最终评分中的加权（0=不启用）
         portrait_boost_map: {memory_id: boost_value} — 画像相关性加权（区间[-0.2, +0.3]）
         portrait_weight: 画像 boost 在最终评分中的加权（0=不启用，推荐 0.1）
    输出: 按相似度降序排列的 top_k 条 candidates，附带 _rr_score
    """
    if not candidates:
        return candidates

    boosts = correction_boosts or {}
    errs = error_counts or {}
    attn = attention_boosts or {}

    try:
        from app.llm.embed import local_embed, local_embed_batch

        query_emb = local_embed(query)
        if query_emb is None:
            logger.warning("query embedding 失败，跳过精排")
            return candidates[:top_k]

        q = np.array(query_emb, dtype=np.float32)

        # ── 批量 embedding：收集所有候选文本，一次 Ollama HTTP 调用 ──
        texts: list[str] = []
        text_indices: list[int] = []  # texts pos → candidates pos
        empty_copies: list[dict] = []  # 无文本的候选直接给 0 分
        for i, c in enumerate(candidates):
            text = (c.get("summary") or c.get("document") or "")[:200]
            mid = c.get("id", "")
            if not text:
                c_copy = c.copy()
                c_copy["_rr_score"] = 0.0
                empty_copies.append(c_copy)
                continue
            texts.append(text)
            text_indices.append(i)

        batch_embs = local_embed_batch(texts) if texts else []

        scored = list(empty_copies)
        for pos, c_idx in enumerate(text_indices):
            c = candidates[c_idx]
            mid = c.get("id", "")
            emb = batch_embs[pos] if pos < len(batch_embs) else None
            if emb is not None:
                d = np.array(emb, dtype=np.float32)
                sim = float(np.dot(q, d))  # 已归一化，余弦=点积
            else:
                sim = 0.0

            # 纠正反馈加成 & 错误报告惩罚（在主路径上生效）
            boost = boosts.get(mid, 0.0) * 0.1
            penalty = errs.get(mid, 0) * 0.05
            # 注意力漂移加权（指数加权后的最近对话上下文相似度）
            attn_score = attn.get(mid, 0.0)
            # 画像 boost（Phase 4: 精排阶段画像相关性微调）
            portrait_boost = (portrait_boost_map or {}).get(mid, 0.0)
            sim = sim + boost - penalty + attention_weight * attn_score + portrait_weight * portrait_boost

            c_copy = c.copy()
            c_copy["_rr_score"] = sim
            scored.append(c_copy)

        scored.sort(key=lambda x: x.get("_rr_score", 0), reverse=True)
        return scored[:top_k]
    except Exception as e:
        logger.warning("Rerank 失败，回退原排序: %s", e)
        return candidates[:top_k]
