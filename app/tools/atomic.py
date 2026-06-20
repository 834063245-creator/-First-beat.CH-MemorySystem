# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: be1b6755

"""原子文件写入 — 写临时文件 → os.replace() 一步替换。"""
import json
import os
import tempfile


def atomic_write(path: str, data) -> None:
    """原子写入 JSON 文件。先写 .tmp，再 rename 覆盖原文件。"""
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=dirname)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # Windows 上 os.replace 是原子的
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_append(path: str, line: str) -> None:
    """原子追加一行到 JSONL 文件。"""
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
