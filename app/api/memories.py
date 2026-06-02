"""记忆管理 API。"""
import json
import logging
import os
import time
import threading
from fastapi import APIRouter, Depends, HTTPException
import jieba

from app.api.deps import AppContext, get_user_context
from app.config.settings import CONTEXT_ROUNDS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memories"], prefix="/api/memories")

_correction_lock = threading.Lock()


@router.get("")
def api_memories(
    page: int = 1,
    per_page: int = 20,
    search: str = "",
    sort: str = "hit_count",
    order: str = "desc",
    tag: str = "",
    date_from: str = "",
    date_to: str = "",
    ctx: AppContext = Depends(get_user_context),
):
    """返回当前用户的记忆列表，支持语义搜索、排序筛选和分页。"""
    client = ctx.chroma_service

    if search:
        from backend.local_embed import local_embed
        query_emb = local_embed(search)
        if query_emb is None:
            return {"items": [], "total": 0, "page": page, "per_page": per_page}
        results = client._read_collection.query(
            query_embeddings=[query_emb],
            n_results=50,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for i, mid in enumerate(results.get("ids", [[]])[0]):
            meta = dict(results["metadatas"][0][i]) if results.get("metadatas") else {}
            doc = results["documents"][0][i] if results.get("documents") else ""
            items.append({
                "id": mid,
                "summary": meta.get("summary", ""),
                "document": (doc or "")[:200],
                "tags": meta.get("tags", "").split(",") if isinstance(meta.get("tags"), str) else (meta.get("tags") or []),
                "emotion": meta.get("emotion", ""),
                "timestamp": meta.get("timestamp", ""),
                "hit_count": meta.get("hit_count", 0),
                "source": meta.get("source", "user"),
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        total = len(items)
        start = (page - 1) * per_page
        return {"items": items[start:start + per_page], "total": total, "page": page, "per_page": per_page}

    return client.list_memories(page=page, per_page=per_page, sort=sort, order=order,
                                tag=tag, date_from=date_from, date_to=date_to)


@router.get("/stats")
def api_memories_stats(ctx: AppContext = Depends(get_user_context)):
    """记忆统计。"""
    return ctx.chroma_service.stats()


@router.get("/{memory_id}")
def api_memories_detail(memory_id: str, ctx: AppContext = Depends(get_user_context)):
    """单条记忆详情。"""
    client = ctx.chroma_service
    detail = client.get_memory_detail(memory_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="记忆未找到")
    detail_ctx = ctx.chat_history.get_context_by_chroma_id(memory_id, before=CONTEXT_ROUNDS, after=CONTEXT_ROUNDS)
    detail["context_before"] = detail_ctx.get("context_before", [])
    detail["context_after"] = detail_ctx.get("context_after", [])
    return detail


@router.post("/{memory_id}/correct")
def api_memories_correct(memory_id: str, body: dict, ctx: AppContext = Depends(get_user_context)):
    """纠正记忆的摘要。同时写入纠正日志供后续检索排序调权。"""
    corrected = body.get("corrected_summary", "")
    if not corrected:
        raise HTTPException(status_code=400, detail="摘要不能为空")

    from backend.local_embed import local_embed
    embedding = local_embed(corrected)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding 失败")

    client = ctx.chroma_service
    tags = jieba.analyse.extract_tags(corrected, topK=5)
    old_detail = client.get_memory_detail(memory_id)
    old_tags = old_detail.get("tags", []) if old_detail else []

    client.update_memory(memory_id, summary=corrected, tags=tags, embedding=embedding)

    try:
        correction_log_path = os.path.join(ctx.data_dir, "correction_log.jsonl")
        os.makedirs(os.path.dirname(correction_log_path), exist_ok=True)
        with _correction_lock:
            with open(correction_log_path, "a", encoding="utf-8") as f:
                tag_str = ",".join(tags[:3]) or ",".join(old_tags[:3])
                f.write(json.dumps({
                    "memory_id": memory_id,
                    "tag": tag_str,
                    "timestamp": time.time(),
                }, ensure_ascii=False) + "\n")
        logger.info("纠正反馈已记录: id=%s tags=%s", memory_id[:8], tag_str)
    except Exception as exc:
        logger.warning("纠正反馈日志写入失败: %s", exc)

    return {"status": "ok"}


@router.delete("/{memory_id}")
def api_memories_delete(memory_id: str, ctx: AppContext = Depends(get_user_context)):
    """删除单条记忆。"""
    client = ctx.chroma_service
    client.delete_memory(memory_id)
    ctx.co_tracker.remove(memory_id)
    ctx.inverted_index.remove(memory_id)
    ctx.chat_history.delete_by_chroma_id(memory_id)
    return {"status": "ok", "id": memory_id}


@router.post("/feedback")
def api_memory_feedback(body: dict, ctx: AppContext = Depends(get_user_context)):
    """提交记忆错误报告。"""
    memory_id = body.get("memory_id", "")
    reason = body.get("reason", "")
    if memory_id:
        from app.core.feedback import log_error_report as log_err
        log_err(memory_id, reason, "user", data_dir=ctx.data_dir)
    return {"ok": True}
