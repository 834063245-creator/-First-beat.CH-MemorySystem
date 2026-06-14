"""后台线程生命周期管理器 — 统一启动/停止 + 崩溃自动重启。

用法：
    from app.background.lifecycle import register, start_all, stop_all, resilient_thread

    register("dmn", start=dmn.start, stop=dmn.stop, depends_on=[])
    start_all()
    # ... run ...
    stop_all()

    # 关键 daemon 线程用 resilient_thread 启动，崩溃后自动重启（最多 5 次/小时）
    t = resilient_thread(target=my_worker, name="my_worker", stop_event=stop_evt)
"""
import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


_registry: list[dict] = []
_lock = threading.Lock()

# 崩溃重启限流：每个线程名 → 最近一小时重启次数
_restart_log: dict[str, list[float]] = {}
_RESTART_LOG_LOCK = threading.Lock()
_MAX_RESTARTS_PER_HOUR = 5


def resilient_thread(*, target: Callable, name: str, stop_event: threading.Event = None,
                     daemon: bool = True, restart_delay: float = 5.0) -> threading.Thread:
    """启动一个守护线程，崩溃后自动重启（最多 MAX_RESTARTS_PER_HOUR 次/小时）。

    目标函数签名应为 target(stop_event) — 接收 stop_event 用于优雅退出。
    """
    evt = stop_event or threading.Event()

    def _wrapper():
        while not evt.is_set():
            try:
                target(evt)
            except Exception:
                logger.exception("后台线程 '%s' 崩溃，%ds 后重启", name, restart_delay)
            # 限流：检查最近一小时重启次数
            now = time.time()
            with _RESTART_LOG_LOCK:
                times = _restart_log.get(name, [])
                times = [t for t in times if now - t < 3600]
                times.append(now)
                _restart_log[name] = times
                if len(times) > _MAX_RESTARTS_PER_HOUR:
                    logger.error("后台线程 '%s' 一小时内重启 %d 次，已达上限，放弃重启", name, len(times))
                    return
            if not evt.is_set():
                evt.wait(restart_delay)

    t = threading.Thread(target=_wrapper, daemon=daemon, name=name)
    t.start()
    return t


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
