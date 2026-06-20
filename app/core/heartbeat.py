# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 80fd47a6

"""用户心跳追踪 — 供后台线程和 API 层共享。

心跳状态是应用级状态，放在 core/ 层以便：
- core/context.py（冲动消费线程）可以查询
- api/system.py（/api/user-active 路由）可以更新
- 两者都遵循 api → core 的依赖方向
"""
import threading
import time


_last_heartbeat_time: float | None = None
_heartbeat_lock = threading.Lock()


def get_last_heartbeat() -> float | None:
    """供后台消费线程查询用户最后活跃时间。"""
    with _heartbeat_lock:
        return _last_heartbeat_time


def record_heartbeat() -> float:
    """记录一次用户心跳，返回当前时间戳。

    供 API 路由调用（前端打字心跳）。
    """
    global _last_heartbeat_time
    now = time.time()
    with _heartbeat_lock:
        _last_heartbeat_time = now
    return now
