# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: dcafb902

"""记忆冲突用户消解 — 用户确认后标记旧记忆 stale。

流程：
  1. LLM 在回复中提到冲突（从 CognitiveState 拿到 pending_conflicts）
  2. 用户下一条消息确认 → check_resolution() 检测到
  3. 旧记忆标记 stale=True → 后续检索自动过滤

零自动裁决，必须用户确认才执行。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 确认信号 — 用户表达"对，之前说的不对"时的多字短语
# 避免单字（"对"/"是"/"嗯"），否则大量无辜消息误触冲突扫描
_CONFIRM_SIGNALS = frozenset({
    "没错", "是的", "对了",
    "记错", "不对", "不是", "记错了",
    "改了", "变了", "更正", "纠正", "变心",
    "戒",                 # "咖啡我戒了" → 确认之前矛盾
    "对，之前记错了", "是的，改了", "你说得对",
    "确实不是", "之前搞错了", "纠正一下",
})

# 消解有效期（秒）：24 小时内检测到的冲突才接受消解
_RESOLUTION_TTL = 86400


def check_resolution(
    user_message: str,
    pending_conflicts: list[dict],
    memory_service,
    co_tracker,
) -> dict | None:
    """检查用户消息是否确认了某条冲突的旧记忆是错的。

    参数:
        user_message: 用户当前消息
        pending_conflicts: DMN 的 pending_conflicts 列表
        memory_service: QdrantService 实例（用于标记 stale）
        co_tracker: CoOccurrenceStore 实例（清理孤儿关联）

    返回:
        被消解的冲突信息 dict（含 tag/old_id/old_summary），或 None
    """
    if not user_message or not pending_conflicts:
        return None

    # ① 用户是否说了确认信号？
    has_signal = any(s in user_message for s in _CONFIRM_SIGNALS)
    if not has_signal:
        return None

    now = __import__("time").time()

    for conflict in pending_conflicts:
        # ② 消解有效期检查
        detected_at = conflict.get("detected_at", 0)
        if detected_at and now - detected_at > _RESOLUTION_TTL:
            continue

        shared_tags = conflict.get("shared_tags", [])
        if not shared_tags:
            continue

        # ③ 用户消息中提到冲突话题了吗？
        tag_matched = any(tag in user_message for tag in shared_tags)
        if not tag_matched:
            continue

        # ④ 执行消解
        old_id = conflict.get("old_id_full") or conflict.get("old_id")
        if not old_id:
            continue

        try:
            memory_service._collection.update(
                ids=[old_id],
                metadatas=[{"stale": True}],
            )
            logger.info(
                "冲突消解 ✓ tag=%s old=%s new=%s",
                shared_tags[0],
                old_id[:8],
                (conflict.get("new_id_full") or conflict.get("new_id", ""))[:8],
            )
        except Exception as e:
            logger.warning("冲突消解写入失败: %s", e)
            continue

        # 清理共现孤儿
        try:
            co_tracker.remove(old_id)
        except Exception:
            pass

        return {
            "tag": shared_tags[0],
            "old_id": old_id,
            "old_summary": conflict.get("old_summary", ""),
            "new_summary": conflict.get("new_summary", ""),
        }

    return None
