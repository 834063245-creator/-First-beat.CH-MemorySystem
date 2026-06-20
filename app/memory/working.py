# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: af347f45

"""工作记忆摘要 — 增量维护对话脉络，替代原始对话原文的 500K token 注入。

每次对话后由本地 LLM（零成本）增量更新摘要，下次请求只带摘要（~3K tokens）
+ 最近 5 轮原文（~2K tokens），替代原来的 500K tokens 原始对话。
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

WM_LOCK = threading.RLock()

# 增量摘要触发配置
MIN_UPDATE_INTERVAL = 5       # 轮数下限，防频繁触发
TOPIC_SHIFT_THRESHOLD = 0.3   # 话题重叠率低于此值时触发


def _load(wm_path: str) -> dict:
    """读取工作记忆文件。"""
    with WM_LOCK:
        if not os.path.exists(wm_path):
            return {"summary": "", "topics": [], "current_state": "", "last_updated": "", "version": 0}
        try:
            with open(wm_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"summary": "", "topics": [], "current_state": "", "last_updated": "", "version": 0}


def _save(data: dict, wm_path: str):
    """写入工作记忆文件。"""
    with WM_LOCK:
        from app.tools.atomic import atomic_write
        atomic_write(wm_path, data)


def get_summary(wm_path: str) -> str:
    """获取当前对话脉络摘要文本，用于注入 prompt。"""
    wm = _load(wm_path)
    parts = []
    if wm.get("summary"):
        parts.append(f"【对话脉络】\n{wm['summary']}")
    if wm.get("current_state"):
        parts.append(f"【当前状态】\n{wm['current_state']}")
    if wm.get("topics"):
        parts.append(f"【话题线索】\n{' '.join(wm['topics'][:5])}")
    if wm.get("recent_keywords") or wm.get("recent_entities"):
        extra = []
        if wm.get("recent_entities"):
            extra.append(f"实体：{' '.join(wm['recent_entities'][:8])}")
        if wm.get("recent_keywords"):
            extra.append(f"关键词：{' '.join(wm['recent_keywords'][:8])}")
        parts.append("【最近提及】\n" + "\n".join(extra))
    return "\n\n".join(parts) if parts else ""


def _topic_shift_detected(recent_turns: list[dict], wm_topics: list[str]) -> bool:
    """检测最近几轮对话的话题是否与当前摘要话题有显著差异。"""
    if not wm_topics:
        return True
    recent_text = " ".join(
        t.get("user_message", "") for t in recent_turns[-5:] if t.get("user_message")
    )
    if not recent_text.strip():
        return False
    from app.brain.semantic import extract_tags
    current_tags = extract_tags(recent_text, topk=8)
    current_tags = [w for w in current_tags if len(w) >= 2]
    if not current_tags:
        return False
    wm_set = set(wm_topics)
    current_set = set(current_tags)
    overlap = len(wm_set & current_set) / max(len(current_set), 1)
    return overlap < TOPIC_SHIFT_THRESHOLD


def incremental_update(conversation_turns: list[dict], *, wm_path: str) -> bool:
    """增量更新工作记忆摘要。由后台线程调用，不阻塞对话。

    通过话题变化检测触发更新，避免固定轮数下连续相同话题浪费资源。
    满足最少轮数（MIN_UPDATE_INTERVAL）且话题分布发生显著偏移时触发。
    输入是最近 N 轮完整对话 + 旧摘要，输出是新摘要。
    """
    # 先用轮数做下限检查（至少 MIN_UPDATE_INTERVAL 轮）
    if len(conversation_turns) < MIN_UPDATE_INTERVAL:
        return False

    # 然后做话题变化检测
    with WM_LOCK:
        wm = _load(wm_path)

    recent_for_detect = conversation_turns[-MIN_UPDATE_INTERVAL:]
    if not _topic_shift_detected(recent_for_detect, wm.get("topics", [])):
        return False  # 话题没变，跳过

    # 取增量部分：上次摘要之后的新对话
    recent_turns = conversation_turns[-(MIN_UPDATE_INTERVAL * 2):]

    # 记录当前版本号，写回时做乐观检测
    old_version = wm.get("version", 0)

    # 构造 prompt 给本地 LLM（兼容 ChatHistory 的 user_message/llm_reply 格式）
    old_summary = wm.get("summary", "无历史摘要")
    turn_lines = []
    for t in recent_turns:
        ts = t.get('timestamp', '')[:16]
        user = t.get('user_message', '')
        reply = t.get('llm_reply', '')
        if user:
            turn_lines.append(f"[{ts}] 用户：{user[:200]}")
        if reply:
            turn_lines.append(f"[{ts}] 助手：{reply[:200]}")
    turns_text = "\n".join(turn_lines)

    prompt_text = (
        f"以下是用户和AI助手之间的最近一段对话，以及当前的对话摘要。\n\n"
        f"当前摘要：\n{old_summary}\n\n"
        f"最近对话：\n{turns_text}\n\n"
        f"请综合以上信息，输出一份更新后的对话脉络摘要，要求：\n"
        f"1. 保持简洁（100字以内）\n"
        f"2. 包含：当前在讨论什么、用户的核心关注点\n"
        f"3. 不要遗漏关键背景\n"
        f"4. 用中文"
    )

    try:
        from app.llm.local import LocalLLM
        llm = LocalLLM()
        new_summary = llm.summarize(prompt_text)
        if not new_summary or len(new_summary) < 10:
            logger.warning("工作记忆摘要生成结果太短，跳过更新")
            return False

        # 写回时统一加锁，重新读取并做乐观版本检测
        with WM_LOCK:
            wm = _load(wm_path)
            if wm.get("version", 0) != old_version:
                logger.info("工作记忆版本已变更(v%s != v%s)，跳过本轮更新", wm.get("version", 0), old_version)
                return False
            wm["summary"] = new_summary[:500]
            wm["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            wm["version"] = wm.get("version", 0) + 1

            try:
                from app.brain.semantic import extract_tags
                topics = extract_tags(new_summary, topk=5)
                wm["topics"] = [t for t in topics if len(t) >= 2]
                wm["recent_entities"] = extract_tags(new_summary, topk=5)
                all_text = " ".join([t.get("user_message", "")[:200] for t in recent_turns if t.get("user_message")])
                wm["recent_keywords"] = extract_tags(all_text, topk=8)
            except Exception:
                pass

            _save(wm, wm_path)

        logger.info("工作记忆摘要已更新 (v%d): %s", wm["version"], new_summary[:60])
        return True
    except Exception as exc:
        logger.warning("工作记忆摘要更新失败: %s", exc)
        return False
