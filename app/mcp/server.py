"""MCP Server — JSON-RPC 协议实现，暴露 8 个只读记忆工具。

用法：
    from app.mcp.server import router as mcp_router
    app.include_router(mcp_router)

端点：
    POST /mcp/jsonrpc  — JSON-RPC 请求
    GET  /mcp/sse      — SSE 流（可选）

所有工具只读不写，不暴露内部 memory_id。
"""

import json
import logging
import time
from collections import Counter
from datetime import datetime

from fastapi import APIRouter, Depends, Request

from app.api.deps import AppContext, get_user_context
from app.mcp.tools import TOOLS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["mcp"], prefix="/mcp")


# ── 工具执行器 ────────────────────────────────────────────────

async def _exec_query_memories(ctx: AppContext, args: dict) -> dict:
    import asyncio
    from app.retrieval.pipeline import run_chat_retrieval
    from app.llm.embed import local_embed
    query = args["query"]
    top_k = min(args.get("top_k", 10), 20)
    emb = await asyncio.to_thread(local_embed, query)
    if not emb:
        return {"memories": [], "error": "embedding 不可用"}
    _, _, _, memories = run_chat_retrieval(query, emb, ctx)
    items = []
    for m in memories[:top_k]:
        meta = m.get("metadata") or {}
        items.append({
            "summary": meta.get("summary", "")[:200],
            "relevance": round(m.get("score", 0) or 0, 3),
            "hit_count": meta.get("hit_count", 0) or 0,
            "time": _rel_time(meta.get("timestamp", 0)),
            "emotion": meta.get("emotion_valence_bin", "") or meta.get("emotion_valence", ""),
        })
    return {"memories": items, "total": len(items)}


async def _exec_recent_history(ctx: AppContext, args: dict) -> dict:
    n = min(args.get("n", 10), 50)
    records = ctx.chat_history.get_recent(n) if ctx.chat_history else []
    items = []
    for r in records:
        u = r.get("user_message", "")
        if u == "[内心独白]":
            items.append({"role": "monologue", "content": r.get("llm_reply", ""), "time": _ts_display(r.get("timestamp", ""))})
        else:
            items.append({"role": "user", "content": u, "time": _ts_display(r.get("timestamp", ""))})
            items.append({"role": "assistant", "content": r.get("llm_reply", ""), "time": _ts_display(r.get("timestamp", ""))})
    return {"items": items}


async def _exec_personality_tags(ctx: AppContext, args: dict) -> dict:
    tag_type = args.get("type", "user")
    top_k = min(args.get("top_k", 15), 30)
    items = []
    if ctx.personality_store:
        try:
            result = ctx.personality_store.list_tags(page=1, page_size=top_k, source=tag_type)
            tags = result.get("items", [])
            items = [{"content": t.get("content", ""), "hit_count": t.get("hit_count", 0) or 0} for t in tags]
        except Exception:
            pass
    return {"tags": items}


async def _exec_topic_tree(ctx: AppContext, args: dict) -> dict:
    tree = ctx.topic_tree  # 使用公开属性而非 _topic_tree
    if tree is None:
        return {"tree": None, "note": "话题树尚未构建"}
    return {"tree": tree._tree if hasattr(tree, "_tree") else None}


async def _exec_relationship(ctx: AppContext, args: dict) -> dict:
    """基于最近 30 轮对话计算关系维度（与 circuit.py 中的逻辑一致）。"""
    try:
        if ctx.chat_history is not None:
            recent = ctx.chat_history.get_recent(n=30)
            if len(recent) > 0:
                familiarity = min(1.0, len(recent) * 0.02)
                err_count = sum(
                    1 for r in recent
                    if "记错" in r.get("user_message", "") or "不对" in r.get("user_message", "")
                )
                thanks_count = sum(
                    1 for r in recent
                    if "谢谢" in r.get("user_message", "") or "感谢" in r.get("user_message", "")
                )
                trust = max(0.0, min(1.0, 0.5 + thanks_count * 0.05 - err_count * 0.1))
                intimate_count = sum(
                    1 for r in recent
                    if "想你" in r.get("user_message", "") or "爱" in r.get("user_message", "")
                )
                sad_count = sum(
                    1 for r in recent
                    if "难过" in r.get("user_message", "") or "烦" in r.get("user_message", "")
                )
                closeness = max(0.0, min(1.0, intimate_count * 0.1 + sad_count * 0.05))
                tech_topics = ["架构", "代码", "Rust", "bug", "部署", "系统", "重构"]
                emotional_words = ["难过", "开心", "感动", "压力", "累", "焦虑"]
                user_msgs = " ".join(r.get("user_message", "") for r in recent[-10:])
                tech_score = sum(1 for w in tech_topics if w in user_msgs)
                emo_score = sum(1 for w in emotional_words if w in user_msgs)
                if tech_score > emo_score * 2:
                    interaction_mode = "collaborator"
                elif emo_score > tech_score * 2:
                    interaction_mode = "partner"
                else:
                    interaction_mode = "casual"
                return {
                    "familiarity": round(familiarity, 3),
                    "trust": round(trust, 3),
                    "closeness": round(closeness, 3),
                    "interaction_mode": interaction_mode,
                }
        return {
            "familiarity": 0.0,
            "trust": 0.5,
            "closeness": 0.0,
            "interaction_mode": "casual",
            "note": "暂无足够对话数据计算关系。",
        }
    except Exception:
        logger.exception("计算关系维度失败")
        return {
            "familiarity": 0.0,
            "trust": 0.5,
            "closeness": 0.0,
            "interaction_mode": "casual",
            "note": "关系计算失败，返回默认值。",
        }


async def _exec_search_knowledge(ctx: AppContext, args: dict) -> dict:
    query = args["query"]
    top_k = min(args.get("top_k", 5), 10)
    if not ctx.knowledge_mode_enabled:
        return {"results": [], "note": "知识库模式未启用"}
    if not ctx.kb:
        return {"results": [], "note": "知识库未初始化"}
    try:
        results = ctx.kb.search(query, top_k=top_k)
        items = [{"content": r.get("text", "")[:300], "source": r.get("source", ""), "relevance": round(r.get("score", 0) or 0, 3)} for r in results]
        return {"results": items}
    except Exception:
        logger.exception("知识库检索失败")
        return {"results": [], "note": "知识库检索暂时不可用"}


async def _exec_memory_stats(ctx: AppContext, args: dict) -> dict:
    try:
        all_mems = ctx.chroma_service.list_all()
        heat = Counter()
        emotion = Counter()
        for m in all_mems:
            meta = m.get("metadata") or {}
            heat[meta.get("heat", "warm")] += 1
            emotion[meta.get("emotion_valence_bin", "neutral") or "neutral"] += 1
        return {
            "total_memories": len(all_mems),
            "heat_distribution": dict(heat),
            "emotion_distribution": dict(emotion),
        }
    except Exception:
        return {"error": "统计失败"}


async def _exec_pattern_observations(ctx: AppContext, args: dict) -> dict:
    pd = getattr(ctx, "_pattern_discovery", None)
    if pd is None and hasattr(ctx, "deepseek_llm"):
        pd = getattr(ctx.deepseek_llm, "_pattern_discovery", None)
    if pd is None:
        return {"observations": [], "tuning": {}, "note": "模式发现未初始化"}
    try:
        return {
            "observations": pd.get_observations(),
            "tuning": pd.get_tuning(),
        }
    except Exception:
        return {"observations": [], "tuning": {}}


async def _exec_store_turn(ctx: AppContext, args: dict) -> dict:
    """存储一轮对话到记忆库。"""
    import asyncio
    from datetime import datetime

    user_message = args["user_message"].strip()
    ai_message = args["ai_message"].strip()
    timestamp = args.get("timestamp", "")

    if not user_message or not ai_message:
        return {"error": "user_message 和 ai_message 不能为空"}

    if not timestamp:
        timestamp = datetime.now().isoformat()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        ctx.storage_executor,
        ctx._store_conversation,
        user_message, ai_message, timestamp,
    )
    logger.info("store_turn 已提交入库: %s → %s", user_message[:40], ai_message[:40])
    return {"ok": True, "timestamp": timestamp}


async def _exec_run_engine(ctx: AppContext, args: dict) -> dict:
    """运行初痕引擎完整管线，返回结构化上下文供外部 LLM 使用。

    引擎跑完不调 LLM 生成回复，不存储对话。
    外部 Agent 的 LLM 作为语言皮层读取此上下文生成回复。
    """
    import asyncio
    import json as _json
    from app.llm.embed import local_embed
    from app.retrieval.pipeline import run_chat_retrieval

    user_message = args["message"].strip()
    include_impulses = args.get("include_impulses", True)

    if not user_message:
        return {"error": "message 不能为空 / message is required"}

    # ① Embedding
    emb = local_embed(user_message)
    if not emb:
        return {"error": "embedding 不可用 / embedding unavailable"}

    # ② 检索管线（CPU 密集型，跑在线程池）
    loop = asyncio.get_running_loop()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        ctx.retrieval_executor, run_chat_retrieval, user_message, emb, ctx)

    # ③ 回路调度（用户分析 + 门控 + 冲动 + 关系）
    from app.core.circuit import CircuitOrchestrator

    orchestrator = CircuitOrchestrator(
        ctx.chroma_service, ctx.personality_store, ctx.impulse_scheduler,
        ctx.dmn, ctx.chat_history, ctx.co_tracker,
        mirror_neuron=ctx.mirror_neuron,
    )

    utterance_spec = await loop.run_in_executor(
        ctx.storage_executor,
        lambda: orchestrator.process(
            user_message, emb, ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )

    # ④ 格式化引擎输出
    from app.llm.deepseek import DeepSeekLLM

    execute = DeepSeekLLM._build_execute_directive(utterance_spec)
    memories_json_str = DeepSeekLLM._build_memories_for_tool(utterance_spec)
    impulses_text = DeepSeekLLM._build_impulses(utterance_spec) if include_impulses else ""

    # ⑤ 人格标签
    user_tags = []
    ai_tags = []
    if ctx.personality_store:
        try:
            ur = ctx.personality_store.list_tags(page=1, page_size=10, source="user")
            user_tags = [
                {"content": t.get("content", ""), "hit_count": t.get("hit_count", 0) or 0}
                for t in ur.get("items", [])
            ]
            ar = ctx.personality_store.list_tags(page=1, page_size=10, source="ai")
            ai_tags = [
                {"content": t.get("content", ""), "hit_count": t.get("hit_count", 0) or 0}
                for t in ar.get("items", [])
            ]
        except Exception:
            pass

    # ⑥ 关系状态
    relationship = {}
    if utterance_spec.relationship:
        rs = utterance_spec.relationship
        relationship = {
            "familiarity": round(rs.familiarity, 3),
            "trust": round(rs.trust, 3),
            "closeness": round(rs.closeness, 3),
            "interaction_mode": rs.interaction_mode,
        }

    # ⑦ 时间线近端
    timeline = []
    for r in (timeline_recent or []):
        u = r.get("user_message", "")
        if u == "[内心独白]":
            timeline.append({
                "role": "monologue",
                "content": r.get("llm_reply", ""),
                "time": r.get("timestamp", ""),
            })
        else:
            timeline.append({
                "role": "user", "content": u,
                "time": r.get("timestamp", ""),
            })
            timeline.append({
                "role": "assistant",
                "content": r.get("llm_reply", ""),
                "time": r.get("timestamp", ""),
            })

    # ⑧ 冲动
    impulses = []
    if include_impulses:
        for imp in (utterance_spec.impulses or []):
            impulses.append({
                "intent": getattr(imp, 'intent', ''),
                "target": getattr(imp, 'target_concept', str(imp)),
            })

    # ⑨ 行为预测
    mirror = {}
    mp = utterance_spec.mirror_prediction
    if mp:
        mirror = {
            "next_intents": mp.get("next_intents", []) or [mp.get("next_intent", "")],
        }

    # ⑩ 组装
    try:
        memories_parsed = _json.loads(memories_json_str)
    except Exception:
        memories_parsed = []

    return {
        "execute": execute,
        "personality": {"user": user_tags, "ai": ai_tags},
        "memories": memories_parsed,
        "timeline_recent": timeline,
        "impulses": [line for line in impulses_text.split("\n") if line.strip()] if impulses_text else [],
        "impulse_raw": impulses,
        "relationship": relationship,
        "session_context": session_context or "",
        "mirror_prediction": mirror,
    }


# ── 路由 ──────────────────────────────────────────────────────

@router.post("/jsonrpc")
async def mcp_jsonrpc(request: Request, ctx: AppContext = Depends(get_user_context)):
    """MCP JSON-RPC 端点。"""
    body = await request.body()
    try:
        req = json.loads(body)
    except json.JSONDecodeError:
        return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}

    rid = req.get("id")
    method = req.get("method", "")

    # tools/list
    if method == "tools/list":
        return {"jsonrpc": "2.0", "result": {"tools": TOOLS}, "id": rid}

    # tools/call
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name", "")
        args = params.get("arguments", {})
        t0 = time.perf_counter()
        try:
            result = await _dispatch(name, args, ctx)
            ms = (time.perf_counter() - t0) * 1000
            logger.info("MCP 工具调用: %s(%s) -> %.0fms", name, json.dumps(args, ensure_ascii=False)[:60], ms)
            return {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}, "id": rid}
        except ValueError as exc:
            # 参数错误——可以安全地暴露给客户端
            logger.warning("MCP 工具 %s 参数错误: %s", name, exc)
            return {"jsonrpc": "2.0", "error": {"code": -32602, "message": f"参数无效: {exc}"}, "id": rid}
        except Exception:
            # 内部错误——不暴露 details，只给通用提示
            logger.exception("MCP 工具 %s 内部错误", name)
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": f"工具 '{name}' 执行失败，请稍后重试"}, "id": rid}

    # initialize（MCP 客户端握手）
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "初痕 MCP", "version": "1.0.0"},
            },
            "id": rid,
        }

    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": rid}


@router.get("/sse")
async def mcp_sse():
    """MCP SSE 端点（可选，供流式客户端使用）。"""
    from fastapi.responses import StreamingResponse

    async def event_stream():
        yield "event: message\ndata: {\"status\":\"connected\"}\n\n"
        yield "event: endpoint\ndata: /mcp/jsonrpc\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── 分发 ──────────────────────────────────────────────────────

async def _dispatch(name: str, args: dict, ctx: AppContext) -> dict:
    handlers = {
        "query_memories": _exec_query_memories,
        "get_recent_history": _exec_recent_history,
        "get_personality_tags": _exec_personality_tags,
        "get_topic_tree": _exec_topic_tree,
        "get_relationship": _exec_relationship,
        "search_knowledge": _exec_search_knowledge,
        "get_memory_stats": _exec_memory_stats,
        "get_pattern_observations": _exec_pattern_observations,
        "run_engine": _exec_run_engine,
        "store_turn": _exec_store_turn,
    }
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"未知工具: {name}")
    return await handler(ctx, args)


# ── 辅助 ──────────────────────────────────────────────────────

def _rel_time(ts):
    if not ts: return ""
    try:
        delta = time.time() - float(ts)
    except (ValueError, TypeError):
        return ""
    if delta < 60: return "刚刚"
    if delta < 3600: return f"{int(delta // 60)}分钟前"
    if delta < 86400: return f"{int(delta // 3600)}小时前"
    if delta < 604800: return f"{int(delta // 86400)}天前"
    if delta < 2592000: return f"{int(delta // 604800)}周前"
    if delta < 31536000: return f"{int(delta // 2592000)}个月前"
    return f"{int(delta // 31536000)}年前"


def _ts_display(ts):
    if not ts: return ""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    return str(ts)[:16]
