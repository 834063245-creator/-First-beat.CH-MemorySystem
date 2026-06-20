# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 8aaf4819

"""测试 app/config/paths.py — 路径工具函数。"""
import os
from app.config.paths import ROOT_DIR, BACKEND_DIR, STATIC_DIR, backend_path, static_path, data_path


class TestPathsConstants:
    """目录常量测试。"""

    def test_root_dir_is_absolute(self):
        assert os.path.isabs(ROOT_DIR)
        assert ROOT_DIR.endswith("app")  # paths.py 在 app/config/ 下，ROOT_DIR = app/

    def test_backend_dir_is_root_backend(self):
        assert BACKEND_DIR.endswith("backend")
        assert os.path.isabs(BACKEND_DIR)

    def test_static_dir_is_root_static(self):
        assert STATIC_DIR.endswith("static")
        assert os.path.isabs(STATIC_DIR)


class TestBackendPath:
    def test_single_segment(self):
        p = backend_path("foo.py")
        assert p.endswith(os.path.join("backend", "foo.py"))

    def test_multiple_segments(self):
        p = backend_path("a", "b", "c.txt")
        assert p.endswith(os.path.join("backend", "a", "b", "c.txt"))


class TestStaticPath:
    def test_single_segment(self):
        p = static_path("index.html")
        assert p.endswith(os.path.join("static", "index.html"))

    def test_multiple_segments(self):
        p = static_path("js", "app.js")
        assert p.endswith(os.path.join("static", "js", "app.js"))


class TestDataPath:
    def test_uses_data_dir_as_base(self):
        p = data_path("/tmp/my_data", "chroma")
        assert p.startswith("/tmp/my_data")
        assert p.endswith("chroma")

    def test_multiple_segments(self):
        p = data_path("data", "sub", "file.json")
        assert p.endswith(os.path.join("data", "sub", "file.json"))
