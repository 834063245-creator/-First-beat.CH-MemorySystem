"""记忆反馈模块 — 错误报告记录与清除。

从 backend/main.py 迁移至此。
"""
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def log_error_report(memory_id: str, reason: str, reporter: str, data_dir: str = "data"):
    """追加错误报告到 JSONL 文件。"""
    path = os.path.join(data_dir, "error_reports.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "memory_id": memory_id,
            "reason": reason,
            "reporter": reporter,
            "timestamp": time.time(),
        }, ensure_ascii=False) + "\n")


def clear_memory_errors(memory_id: str, data_dir: str = "data") -> int:
    """清除指定记忆的所有错误报告。追加清除标记而非重写文件。"""
    path = os.path.join(data_dir, "error_reports.jsonl")
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "memory_id": memory_id,
                "action": "clear",
                "timestamp": time.time(),
            }, ensure_ascii=False) + "\n")
        return 0
    except Exception as e:
        logger.error("清除错误报告失败: %s", e)
        return 0


def get_recent_corrected_ids(data_dir: str = "data", since_hours: int = 24) -> set[str]:
    """读取近 N 小时内被用户标记为错误的 memory_id 集合。

    供 PortraitWriter 消费：用户说"记错了"→关联画像条目标记为待验证。
    """
    path = os.path.join(data_dir, "error_reports.jsonl")
    if not os.path.exists(path):
        return set()
    cutoff = time.time() - since_hours * 3600
    ids: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 跳过 clear 标记，只看 error 报告
                if rec.get("action") == "clear":
                    continue
                if rec.get("timestamp", 0) > cutoff and rec.get("memory_id"):
                    ids.add(rec["memory_id"])
    except OSError:
        pass
    return ids
