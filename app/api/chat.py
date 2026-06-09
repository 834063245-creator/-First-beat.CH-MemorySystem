"""聊天端点 — 引擎闭环的入口和出口。"""
import asyncio
import json
import logging
import os
import time
import threading
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response

from app.core.auth import get_current_user, get_user_context
from app.core.helpers import timed as _timed, build_trace as _build_trace
from app.core.helpers import build_debug_info
from app.core.helpers import load_recent_reversals as _load_recent_reversals
from app.core.tools import (
    SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
    WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL,
    ALL_TOOLS,
)
from app.models.schemas import ChatRequest, ChatResponse
from app.llm.embed import local_embed_async
from app.retrieval.pipeline import run_chat_retrieval
from app.core.circuit import CircuitOrchestrator
from app.core.bottleneck import record as bottleneck_record
from app.tools.search import search_web
from app.tools.workspace import read_file, list_files, grep_files, write_file, edit_file
from app.llm.deepseek import LLMClient, parse_dsml_tool_calls, strip_dsml, now_hint
from app.api.openai import parse_openai_messages, format_openai_chunk, format_openai_response
from app.api.deps import AppContext
from app.config.settings import DEBUG_INCLUDE_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


# ── 通用工具调度 ──────────────────────────────────────────────

async def _handle_tool_call(tc: dict, extra_msgs: list, ctx: AppContext, *,
                            reasoning_content: str = "", is_stream: bool = False):
    """执行一个工具调用，追加结果到 extra_msgs。

    参数 is_stream 仅影响日志前缀，不改变行为。
    """
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}

    asst_msg = {"role": "assistant", "tool_calls": [tc]}
    if reasoning_content:
        asst_msg["reasoning_content"] = reasoning_content

    if name == "search_web":
        search_text = await search_web(args.get("query", ""))
        logger.info("%s搜索结果长度: %d", "流式" if is_stream else "", len(search_text))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": search_text})

    elif name == "read_file":
        file_content = read_file(args.get("path", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": file_content})

    elif name == "list_files":
        listing = list_files(args.get("pattern", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": listing})

    elif name == "grep_files":
        matched = grep_files(args.get("pattern", ""), args.get("glob_pattern", "**/*.py"))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": matched})

    elif name == "write_file":
        result = write_file(args.get("path", ""), args.get("content", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "edit_file":
        result = edit_file(args.get("path", ""), args.get("old_str", ""), args.get("new_str", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "bash":
        import subprocess
        try:
            r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=30)
            result = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            result = "命令执行超时（30s）"
        except Exception as e:
            result = f"执行失败: {e}"
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "glob":
        import glob as _glob
        matches = _glob.glob(args.get("pattern", ""), root_dir=args.get("root", "."), recursive=True)
        result = "\n".join(matches) if matches else "未匹配到文件"
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})


# ── POST /chat/stream ─────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, user_ctx = Depends(get_user_context)):
    user_message = (req.message or "").strip()
    if not user_message:
        return ChatResponse(response="请说点什么吧")
    logger.info("流式请求: %s", user_message[:80])

    # ── 冲突消解：检查用户是否确认了旧记忆错误 ──
    try:
        from app.background.consolidation import _load_state as _dmn_load, _save_state as _dmn_save
        dmn_state = _dmn_load(f"{user_ctx.data_dir}/dmn_state.json")
        pending = dmn_state.get("pending_conflicts", [])
        if pending:
            from app.core.conflict import check_resolution
            resolved = check_resolution(
                user_message, pending,
                user_ctx.chroma_service, user_ctx.co_tracker,
            )
            if resolved:
                remaining = [
                    c for c in pending
                    if (c.get("old_id_full") or c.get("old_id")) != resolved["old_id"]
                ]
                dmn_state["pending_conflicts"] = remaining
                _dmn_save(dmn_state, f"{user_ctx.data_dir}/dmn_state.json")
                logger.info("冲突已消解, %d 条剩余", len(remaining))
    except Exception as exc:
        logger.debug("冲突消解跳过: %s", exc)

    # 检索管线：在 ThreadPoolExecutor 中执行，不阻塞事件循环
    query_embedding_for_retrieval = await _timed("query_embedding", local_embed_async(user_message))
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        user_ctx.retrieval_executor, run_chat_retrieval, user_message, query_embedding_for_retrieval, user_ctx)
    bottleneck_record("retrieval_pipeline", (time.perf_counter() - t0) * 1000)

    # ── 回路调度：引擎做完整决策 ──────────────────────────────
    t0 = time.perf_counter()
    utterance_spec = await loop.run_in_executor(
        user_ctx.storage_executor,
        lambda: CircuitOrchestrator(
            user_ctx.chroma_service, user_ctx.personality_store, user_ctx.impulse_scheduler,
            user_ctx.dmn, user_ctx.chat_history, user_ctx.co_tracker,
            mirror_neuron=user_ctx.mirror_neuron,
        ).process(
            user_message, query_embedding_for_retrieval, user_ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )
    bottleneck_record("circuit_process", (time.perf_counter() - t0) * 1000)
    # 注入情绪反转事件
    utterance_spec.emotional_reversals = _load_recent_reversals(data_dir=user_ctx.data_dir)
    logger.info("回路调度完成: intent=%s emotion=%s memories=%d impulses=%d",
                utterance_spec.user.intent, utterance_spec.user.emotion,
                len(utterance_spec.memories), len(utterance_spec.impulses))

    # Phase 1: 画像实时更新（轻声，<100ms，不调 LLM）
    try:
        rel = getattr(utterance_spec, "relationship", None)
        if rel and hasattr(user_ctx, "portrait_writer"):
            await loop.run_in_executor(
                user_ctx.storage_executor,
                lambda: user_ctx.portrait_writer.realtime_update(utterance_spec, rel),
            )
    except Exception as exc:
        logger.debug("画像实时更新跳过: %s", exc)

    async def event_stream():
        full_text = ""
        extra_msgs: list | None = None

        try:
            for round_idx in range(2):
                # 工具注册：LLM 只保留纯功能工具，认知型工具归引擎
                stream_tools = ALL_TOOLS if round_idx == 0 and not extra_msgs else None
                tool_calls_result = None
                async for tag, token in user_ctx.llm_client.generate_stream(
                    user_message,
                    cognitive_state=utterance_spec,
                    timeline_recent=timeline_recent,
                    session_context=session_context,
                    extra_messages=extra_msgs,
                    tools=stream_tools,
                ):
                    if tag == "reason":
                        safe = token.replace('\n', '\\n')
                        yield "data: [REASON]" + safe + chr(10) + chr(10)
                    elif tag == "content":
                        full_text += token
                        # 在 yield 前剥离 DSML，不让其裸奔到前端
                        clean = strip_dsml(token)
                        if clean:
                            safe = clean.replace('\n', '\\n')
                            yield "data: [CONTENT]" + safe + chr(10) + chr(10)
                    elif tag == "tool_calls":
                        tool_calls_result = token
                        # 通知前端工具有调用
                        tc_data = token.get("calls", token) if isinstance(token, dict) else token
                        if tc_data:
                            tool_names = [t.get("function", {}).get("name", "?") for t in tc_data]
                            yield "data: [TOOL]" + ",".join(tool_names) + chr(10) + chr(10)

                # 统一工具调用检测：结构化 JSON + DSML 格式
                if not tool_calls_result:
                    dsml_calls = parse_dsml_tool_calls(full_text)
                    if dsml_calls:
                        tool_calls_result = {"calls": dsml_calls, "reasoning_content": ""}
                        full_text = strip_dsml(full_text)

                if tool_calls_result:
                    extra_msgs = extra_msgs or []
                    reasoning = tool_calls_result.get("reasoning_content") if isinstance(tool_calls_result, dict) else None
                    tc_data = tool_calls_result.get("calls", tool_calls_result) if isinstance(tool_calls_result, dict) else tool_calls_result
                    for tc in tc_data:
                        await _handle_tool_call(tc, extra_msgs, user_ctx,
                                                reasoning_content=reasoning or "", is_stream=True)
                    continue
                break

            # 发送溯源 trace 数据
            trace_payload = _build_trace(memories)
            yield "data: [TRACE]" + json.dumps(trace_payload, ensure_ascii=False) + chr(10) + chr(10)
            # 调试模式：发送 [DEBUG] 事件
            if req.debug:
                debug_prompt = None
                if req.debug_include_prompt or DEBUG_INCLUDE_PROMPT:
                    debug_prompt = user_ctx.llm_client._build_prompt(
                            memories, personalities=personalities, timeline_recent=timeline_recent
                        ) + "\n" + now_hint()
                debug_info = build_debug_info(memories, personalities, timeline_recent, prompt=debug_prompt)
                yield "data: [DEBUG]" + json.dumps(debug_info, ensure_ascii=False) + chr(10) + chr(10)
            yield "data: [DONE]" + chr(10) + chr(10)
        except Exception as exc:
            logger.error("流式生成失败: %s", exc, exc_info=True)
            yield "data: [ERROR]" + chr(10) + chr(10)
        if full_text:
            if req.test_mode:
                logger.debug("test mode enabled, skipping storage")
            elif req.benchmark_inject:
                # benchmark 注入：存 ChromaDB 但不写 chat_history / working memory
                logger.debug("benchmark inject: storing to ChromaDB only")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, full_text, timestamp)
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(user_ctx.storage_executor, user_ctx.chat_history.append, user_message, full_text, timestamp)
                await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, full_text, timestamp)
                from app.memory.working import incremental_update
                await loop.run_in_executor(user_ctx.storage_executor, lambda: incremental_update(user_ctx.chat_history.records, wm_path=f"{user_ctx.data_dir}/working_memory.json"))
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── POST /chat ────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_ctx = Depends(get_user_context)):
    user_message = (req.message or "").strip()
    if not user_message:
        return ChatResponse(response="请说点什么吧，我听着呢 😊")

    logger.info("收到消息: %s", user_message[:80])

    query_embedding_for_retrieval = await _timed("query_embedding", local_embed_async(user_message))
    loop = asyncio.get_running_loop()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        user_ctx.storage_executor, run_chat_retrieval, user_message, query_embedding_for_retrieval, user_ctx)

    utterance_spec = await loop.run_in_executor(
        user_ctx.storage_executor,
        lambda: CircuitOrchestrator(
            user_ctx.chroma_service, user_ctx.personality_store, user_ctx.impulse_scheduler,
            user_ctx.dmn, user_ctx.chat_history, user_ctx.co_tracker,
            mirror_neuron=user_ctx.mirror_neuron,
        ).process(
            user_message, query_embedding_for_retrieval, user_ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )
    utterance_spec.emotional_reversals = _load_recent_reversals(data_dir=user_ctx.data_dir)
    logger.info("回路调度完成: intent=%s emotion=%s memories=%d impulses=%d",
                utterance_spec.user.intent, utterance_spec.user.emotion,
                len(utterance_spec.memories), len(utterance_spec.impulses))

    # Phase 1: 画像实时更新（轻声，<100ms，不调 LLM）
    try:
        rel = getattr(utterance_spec, "relationship", None)
        if rel and hasattr(user_ctx, "portrait_writer"):
            await loop.run_in_executor(
                user_ctx.storage_executor,
                lambda: user_ctx.portrait_writer.realtime_update(utterance_spec, rel),
            )
    except Exception as exc:
        logger.debug("画像实时更新跳过: %s", exc)

    try:
        extra_messages = []
        for tool_round in range(2):
            result = await user_ctx.llm_client.generate(
                user_message,
                cognitive_state=utterance_spec,
                timeline_recent=timeline_recent,
                tools=ALL_TOOLS,
                extra_messages=extra_messages,
            )
            if not result["tool_calls"]:
                ai_response = result["content"]
                break
            for tc in result["tool_calls"]:
                await _handle_tool_call(tc, extra_messages, user_ctx,
                                        reasoning_content=result.get("reasoning_content", ""),
                                        is_stream=False)
        else:
            result = await user_ctx.llm_client.generate(
                user_message, memories,
                extra_messages=extra_messages,
                personalities=personalities,
                timeline_recent=timeline_recent,
            )
            ai_response = result["content"]
    except Exception as exc:
        logger.error("LLM 调用失败: %s %s", type(exc).__name__, exc)
        import traceback
        logger.error("LLM 调用详情:\n%s", traceback.format_exc())
        return ChatResponse(response="抱歉，AI 服务暂时不可用，请稍后再试。")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if req.test_mode:
        logger.debug("test mode enabled, skipping storage")
    elif req.benchmark_inject:
        # benchmark 注入：存 ChromaDB 但不写 chat_history / working memory
        logger.debug("benchmark inject: storing to ChromaDB only")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, ai_response, timestamp)
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx.chat_history.append, user_message, ai_response, timestamp)
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, ai_response, timestamp)
        from app.memory.working import incremental_update
        await loop.run_in_executor(user_ctx.storage_executor, lambda: incremental_update(
            user_ctx.chat_history.records, wm_path=f"{user_ctx.data_dir}/working_memory.json"))

    debug_info = None
    if req.debug:
        debug_prompt = None
        if req.debug_include_prompt or DEBUG_INCLUDE_PROMPT:
            debug_prompt = user_ctx.llm_client._build_prompt(
                memories, personalities=personalities, timeline_recent=timeline_recent
            ) + "\n" + now_hint()
        debug_info = build_debug_info(memories, personalities, timeline_recent, prompt=debug_prompt)

    return ChatResponse(response=ai_response, debug=debug_info, trace=_build_trace(memories), debug_info=debug_info)


# ── GET /v1/models (OpenAI 兼容) ──────────────────────────────

@router.get("/v1/models")
async def openai_list_models():
    """OpenAI 兼容：返回可用模型列表。NextChat 启动时需要。"""
    from app.config.settings import LLM_MODEL
    return {
        "object": "list",
        "data": [
            {
                "id": LLM_MODEL,
                "object": "model",
                "owned_by": "初痕",
            }
        ],
    }


# ── POST /v1/chat/completions (OpenAI 兼容) ───────────────────

@router.post("/v1/chat/completions")
async def openai_chat_completions(raw: dict, user_ctx = Depends(get_user_context)):
    """OpenAI Chat Completions API 兼容路由。

    接受标准 OpenAI 请求格式，走初痕完整检索+决策管线，
    返回 OpenAI 格式响应。
    """
    messages = raw.get("messages", [])
    stream = raw.get("stream", False)
    model = raw.get("model", "初痕")

    system_prompt, user_message, history = parse_openai_messages(messages)
    user_message = user_message.strip()
    if not user_message:
        return {"error": "No user message found"}

    # ── 检索 + 回路调度（与 /chat/stream 共享同一管线） ──
    query_emb = await local_embed_async(user_message)
    loop = asyncio.get_running_loop()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        user_ctx.retrieval_executor, run_chat_retrieval, user_message, query_emb, user_ctx)

    utterance_spec = await loop.run_in_executor(
        user_ctx.storage_executor,
        lambda: CircuitOrchestrator(
            user_ctx.chroma_service, user_ctx.personality_store, user_ctx.impulse_scheduler,
            user_ctx.dmn, user_ctx.chat_history, user_ctx.co_tracker,
            mirror_neuron=user_ctx.mirror_neuron,
        ).process(
            user_message, query_emb, user_ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )
    utterance_spec.emotional_reversals = _load_recent_reversals(data_dir=user_ctx.data_dir)

    # 将 OpenAI history 转为 timeline 格式，合并到 timeline_recent
    if history:
        history_timeline = []
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history) and history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
                history_timeline.append({
                    "user_message": history[i].get("content", ""),
                    "llm_reply": history[i + 1].get("content", ""),
                    "timestamp": "",
                })
        timeline_recent = history_timeline + (timeline_recent or [])

    async def _openai_stream():
        """OpenAI 格式 SSE 流式生成器。"""
        extra_msgs: list | None = None
        for round_idx in range(2):
            stream_tools = ALL_TOOLS if round_idx == 0 and not extra_msgs else None
            full_text = ""
            tool_calls_result = None
            async for tag, token in user_ctx.llm_client.generate_stream(
                user_message,
                cognitive_state=utterance_spec,
                tools=stream_tools,
                timeline_recent=timeline_recent,
                session_context=session_context,
                personalities=personalities,
                extra_messages=extra_msgs,
            ):
                if tag == "content":
                    full_text += token
                    yield format_openai_chunk(model, token)
                elif tag == "tool_calls":
                    tool_calls_result = token
                # "reason" tag 跳过（OpenAI 格式无 reasoning_content 字段）

            if tool_calls_result:
                extra_msgs = extra_msgs or []
                reasoning = tool_calls_result.get("reasoning_content") if isinstance(tool_calls_result, dict) else None
                tc_data = tool_calls_result.get("calls", tool_calls_result) if isinstance(tool_calls_result, dict) else tool_calls_result
                for tc in tc_data:
                    try:
                        await _handle_tool_call(tc, extra_msgs, user_ctx,
                                                reasoning_content=reasoning or "", is_stream=True)
                    except Exception as exc:
                        logger.error("OpenAI 流式工具调用失败: %s", exc)
                        extra_msgs.append({"role": "tool", "content": json.dumps({"error": str(exc)}, ensure_ascii=False)})
                continue
            break

        # ── 线程安全保存（避免阻塞事件循环） ──
        try:
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(user_ctx.storage_executor, user_ctx.chat_history.append, user_message, full_text, now_ts)
            await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, full_text, now_ts)
            from app.memory.working import incremental_update as _iu
            await loop.run_in_executor(user_ctx.storage_executor, _iu, [{"user_message": user_message, "llm_reply": full_text}], f"{user_ctx.data_dir}/working_memory.json")
        except Exception:
            pass

        yield format_openai_chunk(model, "", finish_reason="stop")
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(_openai_stream(), media_type="text/event-stream")

    # 非流式分支
    extra_msgs: list | None = None
    final_text = ""
    for round_idx in range(2):
        stream_tools = ALL_TOOLS if round_idx == 0 and not extra_msgs else None
        result = await user_ctx.llm_client.generate(
            user_message,
            cognitive_state=utterance_spec,
            tools=stream_tools,
            timeline_recent=timeline_recent,
            session_context=session_context,
            personalities=personalities,
            extra_messages=extra_msgs,
        )
        content_text = result.get("content", "")
        final_text += content_text
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            extra_msgs = extra_msgs or []
            for tc in tool_calls:
                try:
                    await _handle_tool_call(tc, extra_msgs, user_ctx,
                                            reasoning_content=result.get("reasoning_content", ""))
                except Exception as exc:
                    logger.error("OpenAI 非流式工具调用失败: %s", exc)
                    extra_msgs.append({"role": "tool", "content": json.dumps({"error": str(exc)}, ensure_ascii=False)})
        else:
            break

    # ── 线程安全保存（避免阻塞事件循环） ──
    try:
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx.chat_history.append, user_message, final_text, now_ts)
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, final_text, now_ts)
        from app.memory.working import incremental_update as _iu
        await loop.run_in_executor(user_ctx.storage_executor, _iu, [{"user_message": user_message, "llm_reply": final_text}], f"{user_ctx.data_dir}/working_memory.json")
    except Exception as exc:
        logger.debug("工作记忆更新失败: %s", exc)

    return Response(
        content=format_openai_response(model, final_text),
        media_type="application/json",
    )


# ── POST /benchmark/inject ─────────────────────────────────────

from pydantic import BaseModel as _PydanticBase

class BenchmarkInjectRequest(_PydanticBase):
    user_message: str
    ai_message: str
    timestamp: str = ""

@router.post("/benchmark/inject")
async def benchmark_inject(req: BenchmarkInjectRequest, user_ctx = Depends(get_user_context)):
    """注入完整对话轮次：走 embed → summarize → tag → ChromaDB，不调 LLM。

    Benchmark 用：将数据集中的完整对话（user + assistant）作为成品记忆存储。
    跳过 LLM 生成和认知管线，但完整经过系统的存储管线。
    """
    ts = req.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        user_ctx.storage_executor,
        user_ctx._store_conversation,
        req.user_message, req.ai_message, ts,
    )
    return {"status": "ok", "message": "memory stored"}


# ── POST /admin/reset ──────────────────────────────────────────

@router.post("/admin/reset")
async def admin_reset(user_ctx = Depends(get_user_context)):
    """清空当前用户的 ChromaDB 和 chat history（benchmark 用）。"""
    try:
        # 清空主 ChromaDB
        user_ctx.chroma_service.clear_all()
        # 清空 AI ChromaDB
        user_ctx.ai_chroma_service.clear_all()
        # 清空聊天历史
        user_ctx.chat_history.clear()
        # 清空倒排索引
        user_ctx.inverted_index.clear()
        # 清空共现矩阵
        user_ctx.co_tracker.clear()
        # 清空 AI 共现矩阵
        user_ctx.ai_co_tracker.clear()
        # 重建 BM25 索引（如果启用）
        if user_ctx.bm25_index is not None:
            user_ctx.bm25_index.clear()
        logger.info("admin/reset: 已清空所有记忆和聊天历史 for %s", user_ctx.data_dir)
        return {"status": "ok", "message": "ChromaDB + chat history cleared"}
    except Exception as exc:
        logger.error("admin/reset 失败: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )
