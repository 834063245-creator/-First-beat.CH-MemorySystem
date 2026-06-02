"""知识库 API。"""
import logging
import os
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import AppContext, get_user_context, _save_knowledge_mode

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"], prefix="/api/knowledge")


@router.get("/list")
def api_knowledge_list(
    page: int = 1,
    per_page: int = 20,
    ctx: AppContext = Depends(get_user_context),
):
    """知识库条目列表。"""
    return ctx.kb.list_entries(page=page, per_page=per_page)


@router.post("/import")
def api_knowledge_import(body: dict, ctx: AppContext = Depends(get_user_context)):
    """导入文档到知识库。"""
    file_path = body.get("file_path", "")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not isinstance(file_path, str):
        raise HTTPException(status_code=400, detail="file_path 必须是字符串")
    # 限制路径范围：只允许用户 data 目录或项目 data 目录下的文件
    allowed_dirs = [
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
        os.path.normpath(ctx.data_dir),
    ]
    norm_path = os.path.normpath(file_path)
    if not any(norm_path.startswith(d) for d in allowed_dirs):
        raise HTTPException(status_code=400, detail=f"file_path 不在允许的目录范围内")
    if not os.path.isfile(norm_path):
        raise HTTPException(status_code=400, detail="文件不存在")
    try:
        ids = ctx.kb.import_file(norm_path)
        return {"status": "ok", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mode")
def api_knowledge_mode_get(ctx: AppContext = Depends(get_user_context)):
    """获取知识库模式状态。"""
    return {"enabled": ctx.knowledge_mode_enabled}


@router.post("/mode")
def api_knowledge_mode_set(body: dict, ctx: AppContext = Depends(get_user_context)):
    """设置知识库模式。"""
    ctx.knowledge_mode_enabled = body.get("enabled", False)
    _save_knowledge_mode(ctx.knowledge_mode_enabled, data_dir=ctx.data_dir)
    return {"status": "ok", "enabled": ctx.knowledge_mode_enabled}


@router.post("/clean-orphans")
def api_knowledge_clean(ctx: AppContext = Depends(get_user_context)):
    """清理知识库中的孤立条目。"""
    try:
        deleted = ctx.kb.clean_orphans()
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/{entry_id}")
def api_knowledge_detail(entry_id: str, ctx: AppContext = Depends(get_user_context)):
    """获取单条知识库条目详情。"""
    detail = ctx.kb.get_detail(entry_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="知识条目未找到")
    return detail


@router.delete("/{entry_id}")
def api_knowledge_delete(entry_id: str, ctx: AppContext = Depends(get_user_context)):
    """删除单条知识库条目。"""
    ok = ctx.kb.delete_entry(entry_id)
    if not ok:
        raise HTTPException(status_code=404, detail="删除失败或条目不存在")
    return {"status": "ok", "id": entry_id}
