"""蒸馏 API — 人格蒸馏控制。"""
import logging
from fastapi import APIRouter, Depends

from app.api.deps import AppContext, get_user_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["distill"], prefix="/api/distill")


@router.get("/status")
def api_distill_status(ctx: AppContext = Depends(get_user_context)):
    """蒸馏引擎状态。"""
    try:
        return ctx.distill_engine.get_state()
    except Exception as exc:
        return {"error": str(exc)}


@router.post("")
def api_distill(ctx: AppContext = Depends(get_user_context)):
    """手动触发蒸馏。"""
    try:
        result = ctx.distill_engine.run_distill()
        return {"ok": True, "patterns": len(result)}
    except Exception as exc:
        logger.warning("蒸馏失败: %s", exc)
        return {"ok": False, "error": str(exc)}
