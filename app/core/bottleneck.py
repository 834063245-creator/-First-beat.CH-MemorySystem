# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 392a5b58

"""全链路耗时监控 — 自动捕获卡顿现场。"""
import logging
import os
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)

_chain: deque = deque(maxlen=1000)  # [(step, elapsed_ms, timestamp), ...]
_chain_lock = threading.Lock()
_threshold_ms = 2000
_LOG_PATH = os.path.join(os.path.dirname(__file__), "bottleneck_analysis.log")
_LOG_MAX_SIZE = 5 * 1024 * 1024  # 5MB
_LOG_BACKUP_COUNT = 2


def record(step: str, elapsed_ms: float):
    """记录一个步骤的耗时。若超过阈值自动触发转储。"""
    with _chain_lock:
        _chain.append((step, round(elapsed_ms, 1), time.time()))
    if elapsed_ms > _threshold_ms:
        _dump(step, elapsed_ms)


def _rotate_if_needed():
    """日志文件超过上限时轮转（保留最近 _LOG_BACKUP_COUNT 个备份）。"""
    try:
        if not os.path.exists(_LOG_PATH):
            return
        if os.path.getsize(_LOG_PATH) < _LOG_MAX_SIZE:
            return
        # 轮转: bottleneck_analysis.log → .1 → .2
        for i in range(_LOG_BACKUP_COUNT, 0, -1):
            src = _LOG_PATH if i == 1 else f"{_LOG_PATH}.{i - 1}"
            dst = f"{_LOG_PATH}.{i}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
    except Exception:
        pass  # 轮转失败不影响写入


def _dump(trigger_step: str, trigger_ms: float):
    """打印并写入卡顿现场数据。"""
    lines = [
        "",
        "=" * 70,
        f"!!! [卡顿捕获] 检测到超时步骤: {trigger_step} 耗时: {trigger_ms:.0f}ms !!!",
        f"触发时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 70,
        "【全链路耗时报告】",
    ]
    with _chain_lock:
        chain_copy = list(_chain)
    for step, ms, ts in chain_copy:
        flag = " <<<< 超时" if ms > _threshold_ms else ""
        lines.append(f"  {step:40s} {ms:>8.1f}ms{flag}")
    lines.extend(["-" * 70, ""])

    report = "\n".join(lines)

    # 打印到控制台
    logger.warning(report)

    # 写入独立文件（带大小轮转）
    try:
        _rotate_if_needed()
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    except Exception as e:
        logger.warning("bottleneck_analysis.log 写入失败: %s", e)
