"""系统端点 — 健康检查、提示词管理、用户活跃检测等。"""
import asyncio
import json
import logging
import os
import time
import threading
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.deps import (
    AppContext, get_user_context, get_current_user,
    ctx_manager, _load_knowledge_mode, _save_knowledge_mode,
    USER_DATA_DIRS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])

# 心跳追踪
_last_heartbeat_time: float | None = None
_heartbeat_lock = threading.Lock()


@router.get("/api/ping")
def ping():
    return {"status": "ok"}


@router.get("/api/user-active")
def api_user_active():
    """用户打字心跳。前端约每 10 秒调用一次。"""
    global _last_heartbeat_time
    with _heartbeat_lock:
        _last_heartbeat_time = time.time()
    return {"ok": True}


@router.get("/prompt")
@router.get("/api/prompt")
def api_get_prompt():
    """Get current system prompt."""
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
    path = os.path.join(backend_dir, prompt_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        return {"content": ""}


@router.post("/prompt")
@router.post("/api/prompt")
def api_update_prompt(body: dict):
    """Update system prompt."""
    content = body.get("content", "")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content 必须是字符串")
    if len(content) > 50000:
        raise HTTPException(status_code=400, detail="content 过长（上限 50000 字符）")
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    backend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    path = os.path.normpath(os.path.join(backend_dir, prompt_file))
    # 确保目标路径在 backend 目录内（防止任意路径写入）
    if not path.startswith(backend_dir):
        raise HTTPException(status_code=400, detail="路径不合法")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok"}


# /api/memory_feedback lives in app.api.memories
