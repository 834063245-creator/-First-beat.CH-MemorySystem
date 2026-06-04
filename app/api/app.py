"""初痕记忆引擎 — FastAPI 应用工厂。

暴露接口：MCP Server（JSON-RPC）、REST 管理端点、健康检查。
不包含聊天端点——引擎不做文本生成，只做记忆决策。
"""
import logging
import threading

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.system import router as system_router
from app.api.chat_history import router as chat_history_router
from app.api.memories import router as memories_router
from app.api.personalities import router as personalities_router
from app.api.consolidation import router as consolidation_router
from app.api.distill import router as distill_router

logger = logging.getLogger(__name__)

_Routers = [
    health_router,
    system_router,
    chat_history_router,
    memories_router,
    personalities_router,
    consolidation_router,
    distill_router,
]


def _startup_warmup():
    """后台预热：embedding 模型。"""
    try:
        from backend.local_embed import local_embed
        local_embed("warmup")
        logger.info("本地 Embedding 模型已预热")
    except Exception as exc:
        logger.warning("本地 Embedding 模型预热失败: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(title="初痕记忆引擎")

    for router in _Routers:
        app.include_router(router)

    @app.on_event("startup")
    async def startup():
        logger.info("初痕记忆引擎启动中...")
        t = threading.Thread(target=_startup_warmup, daemon=True,
                             name="startup_warmup")
        t.start()
        startup._warmup_thread = t

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("正在停止...")
        t = getattr(startup, "_warmup_thread", None)
        if t and t.is_alive():
            t.join(timeout=5)
        from app.api.deps import ctx_manager
        if ctx_manager:
            ctx_manager.close_all()
        try:
            from app.background.lifecycle import stop_all
            stop_all()
        except Exception as exc:
            logger.warning("lifecycle stop_all 异常: %s", exc)
        logger.info("引擎已关闭")

    return app


app = create_app()
