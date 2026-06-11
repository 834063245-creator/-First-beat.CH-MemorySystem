"""共享 SQLite 连接 — 全项目统一入口。

替代各模块手写的 JSONL load/save/lock 循环。
每个 db 文件一个连接实例，WAL 模式，线程安全。

用法：
    from app.core.db import get_db
    db = get_db("data/co_occurrence.db")
    db.execute("SELECT * FROM ...")
    db.commit()
"""

import sqlite3
import os
import threading

_registry: dict[str, sqlite3.Connection] = {}
_registry_lock = threading.Lock()


def get_db(file_path: str) -> sqlite3.Connection:
    """获取或创建 SQLite 连接。

    自动启用 WAL 模式 + 外键，线程安全复用。
    file_path 可以是绝对路径或相对于项目根目录的路径。
    """
    abs_path = os.path.abspath(file_path)
    with _registry_lock:
        if abs_path in _registry:
            return _registry[abs_path]

        parent = os.path.dirname(abs_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        conn = sqlite3.connect(abs_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _registry[abs_path] = conn
        return conn


def close_all():
    """关闭所有连接（测试/进程退出时调用）。"""
    with _registry_lock:
        for path, conn in list(_registry.items()):
            try:
                conn.close()
            except Exception:
                pass
        _registry.clear()


def close_db(file_path: str):
    """关闭指定路径的数据库连接。"""
    abs_path = os.path.abspath(file_path)
    with _registry_lock:
        conn = _registry.pop(abs_path, None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
