# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c8a02e65

"""测试 app/tools/atomic.py — 原子文件写入。"""
import json
import os
import tempfile

import pytest


class TestAtomicWrite:
    def test_writes_and_reads_valid_json(self):
        from app.tools.atomic import atomic_write
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            data = {"key": "value", "list": [1, 2, 3]}
            atomic_write(path, data)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            assert result == data
        finally:
            os.unlink(path)

    def test_creates_parent_dirs(self):
        from app.tools.atomic import atomic_write
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sub", "deep", "data.json")
        try:
            data = {"hello": "world"}
            atomic_write(path, data)
            assert os.path.exists(path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_overwrites_existing_file(self):
        from app.tools.atomic import atomic_write
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            atomic_write(path, {"first": 1})
            atomic_write(path, {"second": 2})
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            assert result == {"second": 2}
        finally:
            os.unlink(path)


class TestAtomicAppend:
    def test_appends_lines(self):
        from app.tools.atomic import atomic_append
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            atomic_append(path, json.dumps({"line": 1}))
            atomic_append(path, json.dumps({"line": 2}))
            with open(path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"line": 1}
            assert json.loads(lines[1]) == {"line": 2}
        finally:
            os.unlink(path)

    def test_creates_parent_dirs(self):
        from app.tools.atomic import atomic_append
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sub", "data.jsonl")
        try:
            atomic_append(path, json.dumps({"test": True}))
            assert os.path.exists(path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
