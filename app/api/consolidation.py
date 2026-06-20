# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 2f19fa75

"""后台巩固状态 API。"""
import logging
from fastapi import APIRouter, Depends

from app.api.deps import AppContext, get_user_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["consolidation"], prefix="/api/consolidation")


@router.get("/status")
def api_consolidation_status(ctx: AppContext = Depends(get_user_context)):
    """后台巩固引擎状态。"""
    try:
        return ctx.dmn.get_status()
    except Exception as exc:
        logger.warning("巩固状态获取失败: %s", exc)
        return {"error": str(exc)}
