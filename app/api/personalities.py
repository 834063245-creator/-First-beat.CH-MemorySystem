"""人格标签 API。"""
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AppContext, get_user_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["personalities"], prefix="/api/personalities")


@router.get("")
def api_personalities(
    page: int = 1,
    per_page: int = 20,
    sort: str = "created_at",
    order: str = "desc",
    min_hits: int = 0,
    ctx: AppContext = Depends(get_user_context),
):
    """人格标签列表，支持分页排序。"""
    return ctx.personality_store.list_tags(
        page=page, page_size=per_page, sort=sort, order=order, min_hits=min_hits
    )


@router.get("/{tag_id}")
def api_personality_detail(tag_id: str, ctx: AppContext = Depends(get_user_context)):
    """单个人格标签详情。"""
    tag = ctx.personality_store.get_tag(tag_id)
    if tag is None:
        raise HTTPException(status_code=404, detail="标签未找到")
    return tag


@router.delete("/{tag_id}")
def api_personality_delete(tag_id: str, ctx: AppContext = Depends(get_user_context)):
    """删除人格标签。"""
    ctx.personality_store.delete_tag(tag_id)
    return {"deleted": True}
