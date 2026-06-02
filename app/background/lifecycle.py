"""后台线程生命周期管理器 — 统一启动/停止。

用法：
    from app.background.lifecycle import register, start_all, stop_all

    register("dmn", start=dmn.start, stop=dmn.stop, depends_on=[])
    start_all()
    # ... run ...
    stop_all()
"""
import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


_registry: list[dict] = []
_lock = threading.Lock()


def register(
    name: str,
    start: Callable,
    stop: Callable | None = None,
    depends_on: list[str] | None = None,
):
    """注册一个后台服务。

    Args:
        name: 服务名
        start: 启动函数（无参）
        stop: 停止函数（无参），可选
        depends_on: 依赖的服务名列表，停止时反向序
    """
    with _lock:
        _registry.append({
            "name": name,
            "start": start,
            "stop": stop,
            "depends_on": depends_on or [],
        })


def start_all():
    """按注册序启动所有服务。"""
    with _lock:
        for svc in _registry:
            try:
                svc["start"]()
                logger.info("后台服务启动: %s", svc["name"])
            except Exception:
                logger.exception("后台服务启动失败: %s", svc["name"])


def stop_all():
    """反依赖序停止所有服务。"""
    with _lock:
        stopped = set()
        while len(stopped) < len(_registry):
            for svc in reversed(_registry):
                if svc["name"] in stopped:
                    continue
                deps = svc.get("depends_on", [])
                if all(d in stopped for d in deps):
                    if svc["stop"]:
                        try:
                            svc["stop"]()
                            logger.info("后台服务停止: %s", svc["name"])
                        except Exception:
                            logger.exception(
                                "后台服务停止失败: %s", svc["name"]
                            )
                    stopped.add(svc["name"])
