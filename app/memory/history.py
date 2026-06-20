# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 21d8a7e5

"""原始对话历史，同步写入，供时间线近端检索使用。独立于记忆系统。"""
import json
import logging
import os
import threading
from typing import List

from app.brain.semantic import extract_tags

from app.tools.atomic import atomic_append

logger = logging.getLogger(__name__)


class ChatHistory:
    """原始对话历史，同步写入，供时间线检索使用"""

    def __init__(self, path="./data/chat_history.jsonl", max_memory=500):
        self.path = path
        self.max_memory = max_memory
        self.records = []
        self._lock = threading.Lock()
        # 内存映射：timestamp → memory_id（Qdrant point ID）
        # 向后兼容：JSONL 中 chroma_id 和 memory_id 字段均可读取
        self._memory_map: dict[str, str] = {}
        self._load()

    def _load(self):
        """启动时加载最近 max_memory 条记录，过滤逻辑删除标记。"""
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            return
        deleted_ts: set[str] = set()
        with open(self.path, encoding="utf-8") as f:
            all_lines = f.readlines()
            # 扫描全文件收集删除标记
            for line in all_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") == "delete":
                    target = rec.get("target_timestamp", "")
                    if target:
                        deleted_ts.add(target)
            # 加载非删除标记的最后 N 条
            for line in all_lines[-self.max_memory:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") == "delete":
                    continue
                if rec.get("timestamp", "") in deleted_ts:
                    continue
                self.records.append(rec)

    def append(self, user_message: str, llm_reply: str, timestamp: str):
        """同步调用，LLM生成回复后立即写入"""
        record = {
            "user_message": user_message,
            "llm_reply": llm_reply,
            "timestamp": timestamp,
        }
        with self._lock:
            self.records.append(record)
            if len(self.records) > self.max_memory:
                self.records = self.records[-self.max_memory:]
        atomic_append(self.path, json.dumps(record, ensure_ascii=False))

    def update_memory_id(self, timestamp: str, memory_id: str):
        """异步入库完成后回写 memory_id 到 JSONL 文件和内存映射。"""
        with self._lock:
            self._memory_map[timestamp] = memory_id
            for rec in self.records:
                if rec.get("timestamp") == timestamp:
                    # 新字段名；同时保留 chroma_id 向后兼容旧 JSONL 读取逻辑
                    rec["memory_id"] = memory_id
                    rec["chroma_id"] = memory_id
                    break

    def delete_by_timestamp(self, timestamp: str) -> bool:
        """逻辑删除指定 timestamp 的对话记录。追加 delete marker 到 JSONL，不重写文件。"""
        with self._lock:
            before = len(self.records)
            self.records = [r for r in self.records if r.get("timestamp") != timestamp]
            if len(self.records) == before:
                return False
            self._memory_map.pop(timestamp, None)
            # 追加逻辑删除标记（不重写文件）
            try:
                atomic_append(self.path, json.dumps(
                    {"action": "delete", "target_timestamp": timestamp},
                    ensure_ascii=False,
                ))
            except Exception as e:
                logger.error("逻辑删除标记写入失败: %s", e)
                return False
        return True

    def delete_by_memory_id(self, memory_id: str) -> bool:
        """按 memory_id 逻辑删除对话记录。"""
        with self._lock:
            # 向后兼容：同时匹配 memory_id 和 chroma_id 字段
            target_ts = None
            for r in self.records:
                if r.get("memory_id") == memory_id or r.get("chroma_id") == memory_id:
                    target_ts = r.get("timestamp", "")
                    break
            if not target_ts:
                return False
            before = len(self.records)
            self.records = [r for r in self.records
                            if r.get("memory_id") != memory_id and r.get("chroma_id") != memory_id]
            if len(self.records) == before:
                return False
            try:
                atomic_append(self.path, json.dumps(
                    {"action": "delete", "target_timestamp": target_ts},
                    ensure_ascii=False,
                ))
            except Exception as e:
                logger.error("逻辑删除标记写入失败（memory_id）: %s", e)
                return False
        return True

    def get_context_by_timestamp(self, timestamp: str, before: int = 3, after: int = 3) -> dict:
        """动态读取某条消息的上下文：前后 N 轮对话。

        返回 {"context_before": [...], "context_after": [...]}，
        before/after 各最多 N 条，按时间正序排列。
        """
        with self._lock:
            idx = None
            for i, rec in enumerate(self.records):
                if rec.get("timestamp") == timestamp:
                    idx = i
                    break
            if idx is None:
                return {"context_before": [], "context_after": []}
            start = max(0, idx - before)
            end = min(len(self.records), idx + after + 1)
            ctx_before = []
            for j in range(start, idx):
                r = self.records[j]
                ctx_before.append({"user": r.get("user_message", ""), "ai": r.get("llm_reply", "")})
            ctx_after = []
            for j in range(idx + 1, end):
                r = self.records[j]
                ctx_after.append({"user": r.get("user_message", ""), "ai": r.get("llm_reply", "")})
            return {"context_before": ctx_before, "context_after": ctx_after}

    def get_context_by_memory_id(self, memory_id: str, before: int = 3, after: int = 3) -> dict:
        """通过 memory_id 查找上下文。向后兼容 chroma_id 字段。"""
        with self._lock:
            for i, rec in enumerate(self.records):
                if rec.get("memory_id") == memory_id or rec.get("chroma_id") == memory_id:
                    start = max(0, i - before)
                    end = min(len(self.records), i + after + 1)
                    ctx_before = []
                    for j in range(start, i):
                        r = self.records[j]
                        ctx_before.append({"user": r.get("user_message", ""), "ai": r.get("llm_reply", "")})
                    ctx_after = []
                    for j in range(i + 1, end):
                        r = self.records[j]
                        ctx_after.append({"user": r.get("user_message", ""), "ai": r.get("llm_reply", "")})
                    return {"context_before": ctx_before, "context_after": ctx_after}
        return {"context_before": [], "context_after": []}

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算文本的 token 数。异常时降级为 len//3。"""
        try:
            return max(1, len(text) // 3)
        except Exception:
            return len(text) // 3 if text else 0

    def _get_recent_by_token_budget(self, token_budget: int) -> list[dict]:
        """按 token 预算从最新记录往前倒序遍历，填满预算后按时间正序返回。

        至少返回 1 条记录（防止预算过小时返回空）。
        """
        if not self.records:
            return []
        total = 0
        result = []
        for rec in reversed(self.records):
            user_tokens = self._estimate_tokens(rec.get("user_message", ""))
            ai_tokens = self._estimate_tokens(rec.get("llm_reply", ""))
            tokens = user_tokens + ai_tokens
            if total + tokens > token_budget and result:
                break
            total += tokens
            result.append(dict(rec))
            if total >= token_budget:
                break
        # 至少返回 1 条
        if not result and self.records:
            result = [dict(self.records[-1])]
        # 按时间正序返回
        result.reverse()
        # 合并 memory_id（向后兼容 chroma_id）
        for rec in result:
            cid = self._memory_map.get(rec["timestamp"]) or rec.get("memory_id") or rec.get("chroma_id")
            if cid:
                rec["memory_id"] = cid
        return result

    def get_records_snapshot(self) -> list[dict]:
        """返回 records 的线程安全快照（浅拷贝）。"""
        with self._lock:
            return list(self.records)

    def get_recent(self, n: int = 5, token_budget: int = None) -> list[dict]:
        """取最近记录。

        token_budget 不为 None 时按 token 预算倒序填充，至少返回 1 条。
        token_budget 为 None 时取最近 n 条（默认 5），向后兼容。
        """
        if token_budget is not None:
            return self._get_recent_by_token_budget(token_budget)
        with self._lock:
            recent = self.records[-n:]
            merged = []
            for r in recent:
                rec = dict(r)
                cid = self._memory_map.get(rec["timestamp"]) or rec.get("memory_id") or rec.get("chroma_id")
                if cid:
                    rec["memory_id"] = cid
                merged.append(rec)
        return merged

    @staticmethod
    def annotate_chunks(records: list[dict], chunk_size: int = 10) -> list[str]:
        """从后往前分组，为整段对话提取话题关键词（一次 embedding 调用）。

        返回逐行字符串列表，可直接拼入 prompt 的【最近发生了什么】区。
        """
        lines = []
        all_text = " ".join(r.get("user_message", "") for r in records)
        all_keywords = extract_tags(all_text, topk=10) if all_text.strip() else []
        filtered_all = [kw for kw in all_keywords if len(kw) > 1 and kw not in
                        ("的", "了", "是", "我", "你", "他", "她", "它", "们", "在", "有", "和", "就", "不", "也", "这", "那", "都", "要")]

        for chunk_start in range(len(records), 0, -chunk_size):
            chunk = records[max(0, chunk_start - chunk_size):chunk_start]
            if not chunk:
                continue
            if chunk_start == len(records) and len(filtered_all) >= 3:
                lines.append(f"[话题：{', '.join(filtered_all[:3])}]")
            for rec in chunk:
                ts = rec.get("timestamp", "")
                if len(ts) >= 16:
                    ts = ts[:16]
                is_monologue = rec.get("user_message", "") == "[内心独白]"
                if is_monologue:
                    lines.append(f"[{ts}] 内心独白：{rec.get('llm_reply', '')}")
                else:
                    lines.append(f"[{ts}] 用户：{rec.get('user_message', '')}")
                    lines.append(f"[{ts}] 助手：{rec.get('llm_reply', '')}")
        return lines
