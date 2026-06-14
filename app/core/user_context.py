"""用户上下文管理器 — 管理每个用户的 AppContext 实例。"""
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class UserContextManager:
    """按用户懒初始化 AppContext，管理生命周期。"""

    def __init__(self):
        self._contexts: dict[str, AppContext] = {}
        self._lock = threading.Lock()

    def get_context(self, user_id: str, data_dir: str) -> "AppContext":
        """获取用户上下文。不存在时懒创建，同时关闭其他用户的上下文避免线程堆积。"""
        if user_id in self._contexts:
            return self._contexts[user_id]
        with self._lock:
            if user_id not in self._contexts:
                # 关闭所有旧上下文（单用户模式，切用户时释放旧资源）
                for old_id in list(self._contexts.keys()):
                    try:
                        self._contexts[old_id].close()
                        logger.info("用户上下文已关闭: %s", old_id)
                    except Exception as exc:
                        logger.warning("关闭旧上下文失败 %s: %s", old_id, exc)
                self._contexts.clear()
                from app.core.context import AppContext
                ctx = AppContext(data_dir=data_dir)
                self._contexts[user_id] = ctx
                logger.info("用户上下文已创建: %s data_dir=%s", user_id, data_dir)
            return self._contexts[user_id]

    def remove_context(self, user_id: str):
        """释放用户上下文（登出时调用）。"""
        with self._lock:
            ctx = self._contexts.pop(user_id, None)
            if ctx:
                try:
                    ctx.close()
                except Exception as exc:
                    logger.warning("关闭用户上下文失败 %s: %s", user_id, exc)

    def close_all(self):
        """关闭所有上下文（服务优雅退出）。"""
        with self._lock:
            for uid, ctx in self._contexts.items():
                try:
                    ctx.close()
                except Exception as exc:
                    logger.warning("关闭用户上下文失败 %s: %s", uid, exc)
            self._contexts.clear()

    @property
    def active_users(self) -> list[str]:
        return list(self._contexts.keys())


# 全局单例
ctx_manager = UserContextManager()
