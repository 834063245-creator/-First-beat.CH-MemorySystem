"""LLM 调用层 — 通用文本生成 + 工具调用。"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

import httpx
import threading

if TYPE_CHECKING:
    from app.core.state import CognitiveState, UtteranceSpec
    from app.analysis.pattern_discovery import PatternDiscovery

logger = logging.getLogger(__name__)


def now_hint() -> str:
    """返回当前时间的格式化字符串，与记忆时间戳格式一致。"""
    from datetime import datetime
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"当前时间：[{now.year}-{now.month:02d}-{now.day:02d} {now.hour:02d}:{now.minute:02d} {weekdays[now.weekday()]}]"

from app.config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)


_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    os.getenv("PROMPT_FILE", "prompt.txt"),
)


def load_system_prompt() -> str:
    try:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return ""





# ── 缓存友好：核心规则作为常量，不重复构建 ────────────────

_CORE_RULES = (
    "\n\n【记忆使用核心规则——不可修改】\n"
    "1. 如果【记忆】区写着「没有找到相关记忆」，则对于任何需要回忆"
    "用户个人或历史对话的问题，你必须如实回答「我不记得」。"
    "绝不能编造任何事实、属性或实体。\n"
    "2. 你可以使用自己的通用知识回答常识性问题（如编程、百科、天气等），不受第一条限制。\n"
    "3. 绝对不要将不同记忆中的实体或属性合并、替换或混淆。\n"
    "4. 【事实记忆】中的内容是经过系统核实的历史记录，可直接引用。"
    "如果内容本身就能回答用户的问题，就直接使用它回答。\n"
    '5. 如果【自然浮现的念头】区有内容，顺着那个念头自然地开口。\n'
    "不需要解释它从哪里来，不需要加「我注意到」之类的引导语，"
    "就只是顺着心里的感觉自然地说话。\n"
    "你有以下可用工具：search_web（实时搜索）、read_file（读文件）、"
    "list_files（列出文件）、grep_files（搜索文件内容）。"
    "需要时直接调用，不需请示。"
)


# ===================================================================
# LLM Client — 主模型回答（通用，不绑死供应商）
# ===================================================================

class LLMClient:
    """通用 LLM 客户端：基于记忆上下文生成回答。"""

    def __init__(self):
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL
        self.model = LLM_MODEL
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._pattern_discovery: Optional["PatternDiscovery"] = None

    def set_pattern_discovery(self, pd: "PatternDiscovery"):
        self._pattern_discovery = pd

    async def aclose(self):
        """关闭底层 httpx 客户端。"""
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass

    def _build_pattern_observations(self) -> str:
        """返回模式观察文本（供 prompt 注入）。"""
        obs = []
        if self._pattern_discovery:
            try:
                obs.extend(self._pattern_discovery.get_observations())
            except Exception:
                logger.debug("模式观察读取失败")

        # 人格对称性观察（盲区检测，由 consolidate_shallow 写入）
        try:
            import os, json
            cache_dir = os.path.dirname(self._pattern_discovery._cache_path)
            blind_spots_path = os.path.join(cache_dir, "blind_spots.json")
            if os.path.exists(blind_spots_path):
                with open(blind_spots_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                obs.extend(data.get("observations", []))
        except Exception:
            pass

        return "\n".join(obs) if obs else ""

    @staticmethod
    def _timeline_to_messages(timeline_recent: list[dict]) -> list[dict]:
        """历史对话 → user/assistant 消息对，保证 API 级别上下文连续性。

        在消息内容前标注时间，让 LLM 能感知"这句是几小时前还是上周说的"。
        """
        msgs = []
        for rec in timeline_recent:
            user = rec.get("user_message", "")
            ai = rec.get("llm_reply", "")
            ts = rec.get("timestamp", "")
            if isinstance(ts, (int, float)):
                from datetime import datetime
                ts = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            elif len(str(ts)) >= 16:
                ts = str(ts)[:16]
            else:
                ts = ""
            time_tag = f"[{ts}] " if ts else ""

            if user == "[内心独白]":
                if ai:
                    msgs.append({"role": "assistant", "content": f"{time_tag}[内心独白] {ai}"})
            else:
                if user:
                    msgs.append({"role": "user", "content": f"{time_tag}{user}"})
                if ai:
                    msgs.append({"role": "assistant", "content": f"{time_tag}{ai}"})
        return msgs

    async def generate(
        self,
        user_message: str,
        memories: Optional[List[dict]] = None,
        *,
        cognitive_state: Optional["UtteranceSpec"] = None,
        tools: Optional[List[dict]] = None,
        extra_messages: Optional[List[dict]] = None,
        max_tokens: int = 32768,
        personalities: Optional[List[str]] = None,
        timeline_recent: Optional[List[dict]] = None,
        session_context: Optional[str] = None,
    ) -> dict:
        """生成回答，支持工具调用和追加消息。"""
        if cognitive_state is not None:
            system_prompt = self._build_stable_system_prompt(
                cognitive_state, session_context
            )
            mem_count = len(cognitive_state.memories)
        else:
            memories = memories or []
            system_prompt = self._build_prompt(
                memories, personalities=personalities,
                timeline_recent=timeline_recent, session_context=session_context,
            )
            mem_count = len(memories)
        total_chars = len(system_prompt) + len(user_message)
        if extra_messages:
            for m in extra_messages:
                total_chars += len(str(m.get("content", "")))
        logger.info("DeepSeek 请求: prompt ~%d chars, 记忆 %d 条, tools=%s", total_chars, mem_count, bool(tools))
        messages = [{"role": "system", "content": system_prompt}]
        if timeline_recent:
            messages.extend(self._timeline_to_messages(timeline_recent))
        # 记忆走 tool role（API 原生识别为外部事实）
        if cognitive_state:
            memories_text = self._build_memories_for_tool(cognitive_state)
            if memories_text and memories_text != "[]":
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "retrieve_memories",
                        "type": "function",
                        "function": {"name": "retrieve_memories", "arguments": "{}"}
                    }]
                })
                messages.append({
                    "role": "tool", "tool_call_id": "retrieve_memories",
                    "content": memories_text
                })

            # 冲动也走 tool role（与记忆合并注入）
            impulses_text = self._build_impulses(cognitive_state)
            if impulses_text:
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "natural_thoughts",
                        "type": "function",
                        "function": {"name": "natural_thoughts", "arguments": "{}"}
                    }]
                })
                messages.append({
                    "role": "tool", "tool_call_id": "natural_thoughts",
                    "content": impulses_text
                })

            # 【执行】作为独立 system message（API 给 system role 更高优先级）
            execute_text = self._build_execute_directive(cognitive_state)
            if execute_text:
                messages.append({"role": "system", "content": execute_text})

            # 【模式观察】注入（tool role，与记忆/冲动同模式）
            pattern_text = await asyncio.to_thread(self._build_pattern_observations)
            if pattern_text:
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "pattern_observations",
                        "type": "function",
                        "function": {"name": "pattern_observations", "arguments": "{}"},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": "pattern_observations",
                    "content": pattern_text,
                })
        messages.append({"role": "user", "content": user_message})
        if extra_messages:
            messages.extend(extra_messages)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools

        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
                # 缓存状态日志(DeepSeek 返回 prompt_cache_hit_tokens 和 eo-cache-status 头)
            usage_data = data.get("usage", {})
            cached = usage_data.get("prompt_cache_hit_tokens", 0) or 0
            total_prompt = usage_data.get("prompt_tokens", 0) or 0
            miss = usage_data.get("prompt_cache_miss_tokens", 0) or 0
            if total_prompt:
                    hit_pct = cached / total_prompt * 100
                    logger.info("DeepSeek 缓存: total=%d hit=%d miss=%d (%.0f%%)", total_prompt, cached, miss, hit_pct)
        except httpx.TimeoutException:
            logger.error("DeepSeek 调用超时 (connect=10s, read=60s)")
            raise

        choice = data["choices"][0]
        msg = choice["message"]

        # 优先取 API 解析好的 tool_calls
        tool_calls = msg.get("tool_calls", [])
        content = msg.get("content", "")
        reasoning_content = msg.get("reasoning_content", "")

        # 始终清理 DSML（即使有 structured tool_calls，content 里可能也有残留）
        if content:
            content = strip_dsml(content)
            if not tool_calls:
                parsed = parse_dsml_tool_calls(content)
                if parsed:
                    tool_calls = parsed

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
        }

    async def generate_stream(
        self,
        user_message: str,
        memories: Optional[List[dict]] = None,
        *,
        cognitive_state: Optional["UtteranceSpec"] = None,
        extra_messages: Optional[List[dict]] = None,
        personalities: Optional[List[str]] = None,
        timeline_recent: Optional[List[dict]] = None,
        session_context: Optional[str] = None,
        tools: Optional[List[dict]] = None,
    ):
        """流式生成，逐个 token 产出。支持追加消息和工具调用。"""
        if cognitive_state is not None:
            system_prompt = self._build_stable_system_prompt(
                cognitive_state, session_context
            )
            mem_count = len(cognitive_state.memories)
        else:
            memories = memories or []
            system_prompt = self._build_prompt(
                memories, personalities=personalities,
                timeline_recent=timeline_recent, session_context=session_context,
            )
            mem_count = len(memories)
        total_chars = len(system_prompt) + len(user_message)
        if extra_messages:
            for m in extra_messages:
                total_chars += len(str(m.get("content", "")))
        logger.info("DeepSeek 请求: prompt ~%d chars, 记忆 %d 条, tools=%s", total_chars, mem_count, bool(tools))
        messages = [{"role": "system", "content": system_prompt}]
        if timeline_recent:
            messages.extend(self._timeline_to_messages(timeline_recent))
        # 记忆走 tool role（API 原生识别为外部事实）
        if cognitive_state:
            memories_text = self._build_memories_for_tool(cognitive_state)
            if memories_text and memories_text != "[]":
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "retrieve_memories",
                        "type": "function",
                        "function": {"name": "retrieve_memories", "arguments": "{}"}
                    }]
                })
                messages.append({
                    "role": "tool", "tool_call_id": "retrieve_memories",
                    "content": memories_text
                })

            # 冲动也走 tool role（与记忆合并注入）
            impulses_text = self._build_impulses(cognitive_state)
            if impulses_text:
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "natural_thoughts",
                        "type": "function",
                        "function": {"name": "natural_thoughts", "arguments": "{}"}
                    }]
                })
                messages.append({
                    "role": "tool", "tool_call_id": "natural_thoughts",
                    "content": impulses_text
                })

            # 【执行】作为独立 system message（API 给 system role 更高优先级）
            execute_text = self._build_execute_directive(cognitive_state)
            if execute_text:
                messages.append({"role": "system", "content": execute_text})

            # 【模式观察】注入（tool role，与记忆/冲动同模式）
            pattern_text = await asyncio.to_thread(self._build_pattern_observations)
            if pattern_text:
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "pattern_observations",
                        "type": "function",
                        "function": {"name": "pattern_observations", "arguments": "{}"},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": "pattern_observations",
                    "content": pattern_text,
                })
        messages.append({"role": "user", "content": user_message})
        if extra_messages:
            messages.extend(extra_messages)
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 32768,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            ) as resp:
                    resp.raise_for_status()
                    # 缓存状态日志（流式只有 header）
                    cache_status = resp.headers.get("eo-cache-status", "")
                    logger.info("DeepSeek 流式缓存: eo-cache-status=%s", cache_status or "无")
                    tool_calls_acc: dict[int, dict] = {}
                    stream_reasoning = ""
                    _dsml_mode = False
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0].get("delta", {})
                            reason = delta.get("reasoning_content", "")
                            if reason:
                                stream_reasoning += reason
                                yield ("reason", reason)
                            token = delta.get("content", "")
                            if token:
                                if _dsml_mode:
                                    end_idx = token.find(">")
                                    if end_idx >= 0:
                                        _dsml_mode = False
                                        # > 后面可能是正常文本，不能吞掉
                                        remainder = token[end_idx + 1:]
                                        if remainder.strip():
                                            yield ("content", remainder)
                                    elif token.strip() == "" or len(token) > 3:
                                        _dsml_mode = False
                                    continue
                                if "|DSML" in token or "｜" in token or "<|" in token:
                                    if ">" in token:
                                        token = strip_dsml(token)
                                        if token:
                                            yield ("content", token)
                                    else:
                                        _dsml_mode = True
                                    continue
                                yield ("content", token)
                            tc_delta = delta.get("tool_calls")
                            if tc_delta:
                                for tc in tc_delta:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_acc:
                                        tool_calls_acc[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    acc = tool_calls_acc[idx]
                                    if tc.get("id"):
                                        acc["id"] = tc["id"]
                                    if tc.get("function"):
                                        fn = tc["function"]
                                        if fn.get("name"):
                                            acc["function"]["name"] += fn["name"]
                                        if fn.get("arguments") is not None:
                                            acc["function"]["arguments"] += fn["arguments"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                    if tool_calls_acc:
                        calls = [tool_calls_acc[k] for k in sorted(tool_calls_acc.keys())]
                        for c in calls:
                            if not c["function"]["arguments"]:
                                c["function"]["arguments"] = "{}"
                        yield ("tool_calls", {"calls": calls, "reasoning_content": stream_reasoning or ""})
                        return
        except httpx.TimeoutException:
            logger.error("DeepSeek 流式调用超时 (connect=10s, read=60s)")
            raise

    @staticmethod
    def _confidence_label(mem: dict) -> str:
        """根据命中次数、语义距离、来源计算置信度标签。"""
        meta = mem.get("metadata", {})
        hc = meta.get("hit_count", 0) or 0
        dist = mem.get("distance")
        source = mem.get("source", "semantic")

        # 间接来源降权
        if source in ("co_occurrence", "time_triggered", "keyword_expand"):
            return "低"

        # 语义距离判断（dist 越接近 0 越相关）
        if dist is not None:
            sim = 1.0 - dist
            if sim < 0.3:
                return "低"

        # hit_count 判断
        if hc >= 50:
            return "高"
        elif hc >= 10:
            return "中"

        return "低"


    @staticmethod
    def _build_prompt(
            memories: List[dict],
            personalities: Optional[List[str]] = None,
            timeline_recent: Optional[List[dict]] = None,
            session_context: Optional[str] = None,
        ) -> str:
            system_prompt = load_system_prompt()
            sections = [system_prompt]

            # 最近发生了什么（可选）
            if timeline_recent:
                from app.memory.history import ChatHistory
                tl_lines = ['【最近发生了什么】']
                tl_lines.extend(ChatHistory.annotate_chunks(timeline_recent))
                sections.append('\n'.join(tl_lines))

            # 对话脉络（可选，由本地 LLM 维护的轻量摘要）
            if session_context:
                sections.append(session_context)

            # 记忆区
            if memories:
                m_lines = ['【记忆】']
                for mem in memories:
                    source_prefix = mem.get("display_source", "")
                    ts = mem.get("metadata", {}).get("timestamp")
                    ts_tag = ""
                    if ts:
                        from datetime import datetime
                        ts_tag = f"[{datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')}] "
                    stale_tag = " [已更新]" if mem.get("metadata", {}).get("stale", False) else ""

                    conf_tag = f"[{LLMClient._confidence_label(mem)}置信]" if not mem.get("metadata", {}).get("stale", False) else ""
                    emo_tag = ""
                    ei = mem.get("metadata", {}).get("emotional_intensity", 0)
                    valence = mem.get("metadata", {}).get("emotion_valence_bin", "") or ""
                    if ei and ei >= 2:
                        if valence == "positive":
                            emo_tag = " [情绪·正向]"
                        elif valence == "negative":
                            emo_tag = " [情绪·负向]"
                        else:
                            emo_tag = " [情绪激动]"
                    elif ei and ei >= 1:
                        if valence == "positive":
                            emo_tag = " [正向]"
                        elif valence == "negative":
                            emo_tag = " [负向]"
                        else:
                            emo_tag = " [有情绪]"

                    if mem.get("summary_only"):
                        summary = mem.get("summary", "") or mem.get("document", "")[:80]
                        line = f"{conf_tag}{emo_tag}{ts_tag}{stale_tag}{source_prefix} {summary}".strip()
                        m_lines.append(line)
                    else:
                        doc = mem.get("document", "")
                        ctx_before = mem.get("context_before", [])
                        if ctx_before:
                            ctx_lines = []
                            for c in ctx_before:
                                cu = c.get("user", "")
                                ca = c.get("ai", "")
                                if cu:
                                    ctx_lines.append(f"  上文·用户：{cu}")
                                if ca:
                                    ctx_lines.append(f"  上文·AI：{ca}")
                            if ctx_lines:
                                m_lines.extend(ctx_lines)
                        if source_prefix:
                            m_lines.append(f"{conf_tag}{emo_tag}{ts_tag}{stale_tag}{source_prefix} {doc}")
                        else:
                            m_lines.append(f"{conf_tag}{emo_tag}{ts_tag}{stale_tag}{doc}")
                        ctx_after = mem.get("context_after", [])
                        if ctx_after:
                            ctx_lines = []
                            for c in ctx_after:
                                cu = c.get("user", "")
                                ca = c.get("ai", "")
                                if cu:
                                    ctx_lines.append(f"  下文·用户：{cu}")
                                if ca:
                                    ctx_lines.append(f"  下文·AI：{ca}")
                            if ctx_lines:
                                m_lines.extend(ctx_lines)
                # 整体置信度评估
                total = len(memories)
                high = sum(1 for m in memories if LLMClient._confidence_label(m) == "高")
                mid = sum(1 for m in memories if LLMClient._confidence_label(m) == "中")
                if high == total:
                    overall = "高"
                elif high + mid >= total / 2:
                    overall = "中"
                else:
                    overall = "低"
                m_lines[0] = f"【记忆】（检索到 {total} 条，整体置信度：{overall}）"
                sections.append('\n'.join(m_lines))
            else:
                sections.append(
                    "【记忆】\n"
                    "（系统检索完毕，没有找到与当前问题相关的历史记忆。\n"
                    "这意味着你没有任何关于用户个人情况、历史事件或之前讨论过的信息。\n"
                    "如果用户问的是其个人/历史相关的问题，你必须如实回答\"我不记得\"或\"我们没有聊过这个\"。\n"
                    "如果用户问的是不依赖记忆的通用知识问题，你可以用自有知识正常回答。"
                )

            # 人格区（移到记忆后面，靠近用户输入，利用 attention 尾部权重）
            if personalities:
                p_lines = ['【我对你的了解】']
                for tag in personalities:
                    if isinstance(tag, dict):
                        content = tag.get("content", "")
                        tag_type = tag.get("type", "")
                        conf = tag.get("confidence", "")
                        prefix = f"[{tag_type}·{conf}置信]" if tag_type else ""
                        p_lines.append(f'- {prefix} {content}'.strip())
                    else:
                        p_lines.append(f'- {tag}')
                sections.append('\n'.join(p_lines))

            # 运行时诊断
            if memories:
                src = {}
                for m in memories:
                    s = m.get("source", "unknown")
                    src[s] = src.get(s, 0) + 1
                info = ", ".join(f"{k}={v}" for k, v in sorted(src.items()))
                logger.info("注入【记忆】: %d 条 [%s]", len(memories), info)

            # 时间提示（放末尾，不干扰前缀缓存）
            sections.append(now_hint())
            # 不可编辑的核心规则（统一由 _CORE_RULES 一处维护）
            sections.append(_CORE_RULES)

            return '\n\n'.join(sections)

    # ── V2 缓存优化版 prompt 构建 ──────────────────────────

    @staticmethod
    def _build_stable_system_prompt(
        cognitive_state: "UtteranceSpec",
        session_context: Optional[str] = None,
    ) -> str:
        """构建缓存稳定的 system prompt（不含每次变化的记忆内容）。

        记忆部分由 caller 拼接在 messages 末尾，
        确保 system prompt + 历史对话前缀被 DeepSeek 缓存命中。
        """
        sections = [load_system_prompt()]
        # 核心规则（稳定，放前面最大化缓存命中）
        sections.append(_CORE_RULES)
        if session_context:
            sections.append(session_context)
        # 人格洞察（变化频率低，归入稳定段）
        if cognitive_state.personality_notes:
            type_tag = {
                "行为模式": "行为", "思维模式": "思维", "偏好模式": "偏好",
                "沟通模式": "沟通", "演变趋势": "趋势",
                "动机归因": "动机", "情绪模式": "情绪",
            }
            lines = ["【我对你的了解】"]
            for note in cognitive_state.personality_notes:
                if isinstance(note, dict):
                    t = note.get("type", "")
                    tag = type_tag.get(t, "")
                    content = note.get("content", "")
                    prefix = f"（{tag}）" if tag else ""
                    lines.append(f"- {prefix} {content}".strip())
                elif isinstance(note, str):
                    lines.append(f"- {note}")
            sections.append("\n".join(lines))

        # AI 自我表达习惯区
        if cognitive_state.personality_notes_ai:
            ai_lines = ["【我自己的表达习惯】"]
            for note in cognitive_state.personality_notes_ai:
                if isinstance(note, dict):
                    content = note.get("content", "")
                    t = note.get("type", "")
                    prefix = f"（{t}）" if t else ""
                    ai_lines.append(f"- {prefix} {content}".strip())
                elif isinstance(note, str):
                    ai_lines.append(f"- {note}")
            sections.append("\n".join(ai_lines))

        # 时间提示（放末尾，不干扰前缀缓存）
        sections.append(now_hint())
        return "\n\n".join(sections)

    @staticmethod
    def _build_execute_directive(cognitive_state: "UtteranceSpec") -> str:
        """引擎执行指令，独立于记忆内容。"""
        parts = []
        pf = cognitive_state.user
        gate = cognitive_state.gate
        if pf.raw_text:
            mode_labels = {
                "soothe": "先共情", "question_first": "先了解再回应",
                "direct_answer": "直接回答", "confirm": "先确认再解决",
                "auto": "自然回应",
            }
            mode_str = mode_labels.get(gate.response_mode, "自然回应")
            parts.append(f"【执行】{pf.intent} · {gate.tone} · {mode_str}")

            mp = cognitive_state.mirror_prediction
            if mp and (mp.get("next_intents") or mp.get("next_intent")):
                topics = mp.get("next_intents") or [mp.get("next_intent", "")]
                parts.append(f"准备方向：{' → '.join(topics[:3])}")

            if hasattr(cognitive_state, 'emotional_reversals') and cognitive_state.emotional_reversals:
                rev_lines = [f"  {r.get('topic','')}: {r.get('before','')} → {r.get('after','')}"
                            for r in cognitive_state.emotional_reversals[:3]]
                if rev_lines:
                    parts.append("情绪反转：\n" + "\n".join(rev_lines))

            # 关系状态注入
            if cognitive_state.relationship:
                rs = cognitive_state.relationship
                fam = "高" if rs.familiarity > 0.6 else "中" if rs.familiarity > 0.3 else "低"
                tru = "高" if rs.trust > 0.6 else "中" if rs.trust > 0.3 else "低"
                clo = "高" if rs.closeness > 0.6 else "中" if rs.closeness > 0.3 else "低"
                parts.append(
                    f"【关系状态】熟悉度({fam}) · 信任度({tru}) · 亲密度({clo}) · 当前模式({rs.interaction_mode})"
                )
        return "\n".join(parts)

    @staticmethod
    def _relative_time(timestamp: float) -> str:
        """把时间戳转成相对时间描述。"""
        import time
        delta = time.time() - timestamp
        if delta < 60:
            return "刚刚"
        elif delta < 3600:
            return f"{int(delta // 60)}分钟前"
        elif delta < 86400:
            return f"{int(delta // 3600)}小时前"
        elif delta < 604800:
            return f"{int(delta // 86400)}天前"
        elif delta < 2592000:
            return f"{int(delta // 604800)}周前"
        elif delta < 31536000:
            return f"{int(delta // 2592000)}个月前"
        else:
            return f"{int(delta // 31536000)}年前"

    @staticmethod
    def _build_memories_for_tool(cognitive_state: "UtteranceSpec") -> str:
        """记忆格式化为 tool content（JSON 结构，保留时间戳和来源）。

        若有编织结果（woven_context），故事线优先展示，fact 记忆随后。
        """
        items = []

        # ── 引擎编织的故事线优先 ──
        wc = getattr(cognitive_state, 'woven_context', None)
        if wc and wc.narratives:
            for i, n in enumerate(wc.narratives):
                items.append({
                    "id": f"narrative_{i}",
                    "time": "",
                    "relative_time": "",
                    "summary": n[:200],
                    "source": "engine narrative",
                    "hit_count": 0,
                    "relevance": 1.0,
                    "stale": False,
                })

        _mems = cognitive_state.memories
        # MemoryDirective → dict 归一（支持两种类型混用）
        if _mems and not isinstance(_mems[0], dict):
            _mems = [
                {
                    "id": m.memory_id,
                    "summary": m.summary,
                    "document": getattr(m, 'document', ''),
                    "metadata": {},
                    "display_source": "",
                    "score": getattr(m, 'certainty', 0) or 0,
                }
                for m in _mems
            ]
        for mem in _mems:
            meta = mem.get("metadata") or {}
            ts = meta.get("timestamp", 0)
            rel = ""
            if ts:
                try:
                    rel = LLMClient._relative_time(float(ts))
                except (ValueError, TypeError):
                    pass
            ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            items.append({
                "id": mem.get("id", ""),
                "time": ts_str,
                "relative_time": rel,
                "summary": (meta.get("summary") or "")[:200],
                "source": mem.get("display_source", ""),
                "hit_count": meta.get("hit_count", 0) or 0,
                "relevance": round(mem.get("score", 0) or 0, 3),
                "stale": meta.get("stale", False),
            })
        if not items:
            return json.dumps([], ensure_ascii=False)
        return json.dumps(items, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_impulses(cognitive_state: "UtteranceSpec") -> str:
        """自然浮现的念头，格式化为独立内容。"""
        if not cognitive_state.impulses:
            return ""
        lines = []
        for imp in cognitive_state.impulses:
            tone_map = {
                "recall": f"你心里好像想起了什么——关于{imp.target_concept}的事",
                "check": f"你忽然想到，{imp.target_concept}——想自然地提一下",
                "share_observation": f"你注意到{imp.target_concept}，想说出来",
                "emotional_check": f"你感觉到{imp.target_concept}，想关心一下",
            }
            line = tone_map.get(imp.intent, f"你心里有一个念头浮动——{imp.target_concept}")
            lines.append(line)
        return "\n".join(lines)


def _next_tool_id() -> str:
    global _TOOL_ID_COUNTER
    with _tool_id_lock:
        _TOOL_ID_COUNTER += 1
        return f"call_dsml_{_TOOL_ID_COUNTER:04d}"


# DeepSeek DSML 工具调用解析 — 正则模式
_DSML_RE = re.compile(
    r'<\|DSML\|tool_calls\|name\|([^|]+)\|params\|(.*?)\|>\s*'
    r'</\|DSML\|tool_calls\|>',
    re.DOTALL,
)
_PARAM_RE = re.compile(r'<\|DSML\|parameter\s+\w+\|([^|]+)\|>\s*(.*?)\s*</\|DSML\|parameter\s*>', re.DOTALL)
_ALT_TOOL_RE = re.compile(r'<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>', re.DOTALL)
_ALT_CALL_RE = re.compile(r'<\|caller\|([^|]+)\|>\s*(.*?)\s*<\|caller\|>', re.DOTALL)
_TOOL_ID_COUNTER = 0
_tool_id_lock = threading.Lock()


def parse_dsml_tool_calls(text: str) -> list:
    """将 DeepSeek 原生 DSML 工具调用解析为 OpenAI 结构。"""
    if "deepseek" not in LLM_MODEL.lower():
        return []
    calls = []

    # 格式 1: <|DSML|tool_calls> ... </|DSML|tool_calls>
    for match in _DSML_RE.finditer(text):
        name = match.group(1)
        params_body = match.group(2)
        args = {}
        for pm in _PARAM_RE.finditer(params_body):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            args[pname] = pvalue
        calls.append({
            "id": _next_tool_id(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    # 格式 2: <｜tool▁calls▁begin｜>...（旧版）
    for outer in _ALT_TOOL_RE.finditer(text):
        for inner in _ALT_CALL_RE.finditer(outer.group(1)):
            name = inner.group(1)
            args_raw = inner.group(2).strip()
            try:
                json.loads(args_raw)  # 验证 JSON
            except json.JSONDecodeError:
                continue
            calls.append({
                "id": _next_tool_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_raw,
                },
            })

    return calls


def strip_dsml(text: str) -> str:
    """移除 DSML 标记，保留纯文本内容。含流式碎片处理。"""
    if "deepseek" not in LLM_MODEL.lower():
        return text
    text = _DSML_RE.sub("", text)
    text = _ALT_TOOL_RE.sub("", text)
    # 完整标签
    text = re.sub(r'<\|DSML\|tool_calls\s*>', "", text)
    text = re.sub(r'</\|DSML\|tool_calls\s*>', "", text)
    text = re.sub(r'<\|DSML\|parameter\s+[^>]*>', "", text)
    text = re.sub(r'</\|DSML\|parameter\s*>', "", text)
    # 流式碎片：未闭合 / 跨 token 的 DSML 片段
    text = re.sub(r'<\|DSML\|[^>]*>', "", text)    # 任意 <|DSML|...> 标签
    text = re.sub(r'</\|DSML\|[^>]*>', "", text)   # 任意 </|DSML|...> 标签
    text = re.sub(r'<\|DSML[^>]*', "", text)       # 未闭合的 <|DSML...
    text = re.sub(r'</\|DSML[^>]*', "", text)      # 未闭合的 </|DSML...
    text = re.sub(r'<｜[^▸]*', "", text)           # 替代格式碎片
    return text.strip()


# 向后兼容别名
DeepSeekLLM = LLMClient
