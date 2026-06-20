# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 5ab2e75f

"""聊天历史 API。"""
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AppContext, get_user_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"], prefix="/api/chat")


@router.get("/history")
def api_chat_history(ctx: AppContext = Depends(get_user_context)):
    """返回最近对话历史。"""
    records = ctx.chat_history.get_recent(50)
    return {"items": records}


@router.delete("/history/{timestamp}")
def api_chat_history_delete(timestamp: str, ctx: AppContext = Depends(get_user_context)):
    """删除指定时间戳的对话记录。"""
    ok = ctx.chat_history.delete_by_timestamp(timestamp)
    if not ok:
        raise HTTPException(status_code=404, detail="记录未找到")
    return {"status": "ok"}
