"""健康检查端点。"""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ollama")
def health_ollama():
    """检查 Ollama 连接状态（暂用旧逻辑的占位）。"""
    return {"status": "unknown", "detail": "检查逻辑将在后续阶段从 main.py 搬入"}
