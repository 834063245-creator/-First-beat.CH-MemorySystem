"""系统端点 — 健康检查、提示词管理、用户活跃检测、登录等。"""
import asyncio
import json
import logging
import os
import secrets as _secrets
import time
import threading
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from app.config.settings import DATA_DIR

from app.api.deps import (
    AppContext, get_user_context, get_current_user,
    ctx_manager,
    USER_DATA_DIRS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])

# ── Token 认证 ──────────────────────────────────────────
_AUTH_TOKEN_PATH = os.path.join(DATA_DIR, "auth_tokens.json")
_AUTH_TOKENS: dict[str, dict] = {}
_AUTH_LOCK = threading.Lock()
_USERS = json.loads(os.getenv("USERS", '{"admin":"admin"}'))


def _load_auth_tokens():
    if os.path.exists(_AUTH_TOKEN_PATH):
        try:
            with open(_AUTH_TOKEN_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_auth_tokens():
    os.makedirs(os.path.dirname(_AUTH_TOKEN_PATH), exist_ok=True)
    with open(_AUTH_TOKEN_PATH, "w") as f:
        json.dump(_AUTH_TOKENS, f)


_AUTH_TOKENS = _load_auth_tokens()


async def get_bearer_user(credentials: str = Depends(HTTPBearer(auto_error=False))) -> str:
    """从 Bearer token 获取当前用户名。无 token 返回 'admin'。"""
    if credentials is None:
        return "admin"
    token = credentials.credentials
    now = time.time()
    with _AUTH_LOCK:
        entry = _AUTH_TOKENS.get(token)
        if entry is None:
            return "admin"
        if entry.get("expires", 0) < now:
            del _AUTH_TOKENS[token]
            _save_auth_tokens()
            return "admin"
    return entry.get("username", "admin")


@router.post("/login")
def api_login(body: dict):
    """用户名密码登录，返回 Bearer token。"""
    username = body.get("username", "")
    password = body.get("password", "")
    expected = _USERS.get(username)
    if expected is None or not _secrets.compare_digest(expected, password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = _secrets.token_urlsafe(32)
    with _AUTH_LOCK:
        _AUTH_TOKENS[token] = {
            "username": username,
            "created": time.time(),
            "expires": time.time() + 604800,
        }
        _save_auth_tokens()
    return {"token": token, "username": username}


@router.get("/api/user/list")
def api_user_list():
    """返回可用用户列表。"""
    from app.config.settings import USER_DATA_DIRS
    return {"users": list(USER_DATA_DIRS.keys())}


@router.post("/api/user/switch")
def api_user_switch(user: str):
    """切换当前用户（写入 cookie，前端配合刷新）。"""
    resp = JSONResponse({"ok": True, "user": user})
    resp.set_cookie(key="chuhen_user", value=user, path="/")
    return resp


from app.core.heartbeat import record_heartbeat


@router.get("/api/ping")
def ping():
    return {"status": "ok"}


@router.get("/api/user-active")
def api_user_active():
    """用户打字心跳。前端约每 10 秒调用一次。"""
    record_heartbeat()
    return {"ok": True}


@router.get("/prompt")
@router.get("/api/prompt")
def api_get_prompt():
    """Get current system prompt."""
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    path = os.path.join(project_dir, prompt_file)
    try:
        with open(path, encoding="utf-8") as f:
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
    project_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.normpath(os.path.join(project_dir, prompt_file))
    # 确保目标路径在项目目录内（防止任意路径写入）
    if not path.startswith(project_dir):
        raise HTTPException(status_code=400, detail="路径不合法")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok"}


@router.get("/api/status")
def api_status():
    """引擎全维度可观测性端点 — 一站式诊断。

    返回记忆库、巩固、冲动、蒸馏、模式发现、
    ChromaDB、线程池的聚合快照。
    """
    now = time.time()
    snapshot = {
        "ts": now,
        "ts_human": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # 取第一个活跃用户的上下文
        active = ctx_manager.active_users
        if not active:
            return {**snapshot, "error": "无活跃用户上下文"}
        ctx = ctx_manager.get_context(active[0], USER_DATA_DIRS.get(active[0], ""))
        if ctx is None:
            return {**snapshot, "error": "无活跃上下文"}

        # ── 记忆库 ──
        try:
            mem_count = ctx.chroma_service.count()
            heat_dist = {}
            try:
                all_mems = ctx.chroma_service.list_all_cached()
                for m in all_mems:
                    h = (m.get("metadata") or {}).get("heat", "unknown")
                    heat_dist[h] = heat_dist.get(h, 0) + 1
                stale_count = sum(
                    1 for m in all_mems
                    if (m.get("metadata") or {}).get("stale", False)
                )
                archived_count = sum(
                    1 for m in all_mems
                    if (m.get("metadata") or {}).get("archived", False)
                )
            except Exception:
                heat_dist = {}
                stale_count = 0
                archived_count = 0
            snapshot["memory"] = {
                "total": mem_count,
                "heat_distribution": heat_dist,
                "stale": stale_count,
                "archived": archived_count,
            }
        except Exception as exc:
            snapshot["memory"] = {"error": str(exc)}

        # ── AI 记忆库 ──
        try:
            ai_count = ctx.ai_chroma_service.count()
            snapshot["ai_memory"] = {"total": ai_count}
        except Exception as exc:
            snapshot["ai_memory"] = {"error": str(exc)}

        # ── 画像系统（Phase 4 退役完成，替代旧 PersonalityStore） ──
        try:
            portrait = ctx.portrait.to_dict()
            snapshot["portrait"] = {
                "dimensions": {k: bool(v) for k, v in portrait.items()},
                "total_entries": sum(len(v) if isinstance(v, list) else 1 for v in portrait.values()),
            }
        except Exception as exc:
            snapshot["personality"] = {"error": str(exc)}

        # ── 巩固状态 ──
        try:
            if ctx.dmn:
                dmn_status = ctx.dmn.get_status()
                snapshot["consolidation"] = dmn_status
            else:
                snapshot["consolidation"] = {"status": "disabled"}
        except Exception as exc:
            snapshot["consolidation"] = {"error": str(exc)}

        # ── 冲动状态 ──
        try:
            if ctx.impulse_scheduler:
                imp_status = ctx.impulse_scheduler.get_status_snapshot()
                snapshot["impulse"] = imp_status
            else:
                snapshot["impulse"] = {"status": "disabled"}
        except Exception as exc:
            snapshot["impulse"] = {"error": str(exc)}

        # ── 模式发现 ──
        try:
            if ctx._pattern_discovery:
                obs = ctx._pattern_discovery.get_observations()
                tuning = ctx._pattern_discovery.get_tuning()
                snapshot["pattern_discovery"] = {
                    "observations": obs,
                    "tuning": tuning,
                }
        except Exception as exc:
            snapshot["pattern_discovery"] = {"error": str(exc)}

        # ── 对话历史 ──
        try:
            if ctx.chat_history:
                records = ctx.chat_history.get_records_snapshot()
                total_msgs = len(records)
                inner_monologues = sum(
                    1 for r in records
                    if r.get("user_message") == "[内心独白]"
                )
                snapshot["chat_history"] = {
                    "total_records": total_msgs,
                    "inner_monologues": inner_monologues,
                }
        except Exception as exc:
            snapshot["chat_history"] = {"error": str(exc)}

        # ── ChromaDB 集合信息 ──
        try:
            col = ctx.chroma_service._collection
            snapshot["chromadb"] = {
                "collection_name": col.name,
                "count": col.count(),
            }
        except Exception as exc:
            snapshot["chromadb"] = {"error": str(exc)}

        # ── 事件循环/线程摘要 ──
        try:
            import threading as _th
            snapshot["runtime"] = {
                "active_threads": _th.active_count(),
                "thread_names": [t.name for t in _th.enumerate()],
            }
        except Exception as exc:
            snapshot["runtime"] = {"error": str(exc)}

        # ── 引擎调参快照 ──
        try:
            from app.config.settings import (
                CONSOLIDATION_SHALLOW_INTERVAL,
                CONSOLIDATION_DEEP_INTERVAL,
                IMPULSE_MAX_PER_HOUR,
                IMPULSE_IDLE_MINUTES,
            )
            snapshot["config"] = {
                "shallow_interval_h": round(CONSOLIDATION_SHALLOW_INTERVAL / 3600, 1),
                "deep_interval_h": round(CONSOLIDATION_DEEP_INTERVAL / 3600, 1),
                "impulse_max_per_hour": IMPULSE_MAX_PER_HOUR,
                "impulse_idle_minutes": IMPULSE_IDLE_MINUTES,
            }
        except Exception as exc:
            snapshot["config"] = {"error": str(exc)}

    except Exception as exc:
        snapshot["error"] = str(exc)

    return snapshot


# /api/memory_feedback lives in app.api.memories
