# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 3b91a6ce

"""路径工具 — 兼容新旧路径布局。"""
import os

# 项目根目录（amazing3/）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 旧 backend/ 目录（兼容期保留）
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

# 静态文件目录
STATIC_DIR = os.path.join(ROOT_DIR, "static")


def backend_path(*parts: str) -> str:
    """返回 backend/ 下的路径（兼容旧模块导入）。"""
    return os.path.join(BACKEND_DIR, *parts)


def static_path(*parts: str) -> str:
    """返回 static/ 下的路径。"""
    return os.path.join(STATIC_DIR, *parts)


def data_path(data_dir: str, *parts: str) -> str:
    """返回 data/ 下的路径。"""
    return os.path.join(data_dir, *parts)
