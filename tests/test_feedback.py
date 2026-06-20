# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 1a5c4d1d

"""测试 app/core/feedback.py — 记忆错误报告记录与清除。

覆盖：log_error_report / clear_memory_errors 文件写入。
"""
import json
import os
import tempfile
from app.core.feedback import log_error_report, clear_memory_errors


class TestLogErrorReport:
    def test_writes_jsonl_line(self):
        with tempfile.TemporaryDirectory() as td:
            log_error_report("mem_001", "内容不准确", "用户反馈", data_dir=td)
            path = os.path.join(td, "error_reports.jsonl")
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["memory_id"] == "mem_001"
            assert record["reason"] == "内容不准确"
            assert record["reporter"] == "用户反馈"
            assert "timestamp" in record

    def test_appends_multiple_reports(self):
        with tempfile.TemporaryDirectory() as td:
            log_error_report("mem_001", "错误1", "user", data_dir=td)
            log_error_report("mem_002", "错误2", "system", data_dir=td)
            path = os.path.join(td, "error_reports.jsonl")
            with open(path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) == 2
            assert json.loads(lines[0])["memory_id"] == "mem_001"
            assert json.loads(lines[1])["memory_id"] == "mem_002"


class TestClearMemoryErrors:
    def test_clear_writes_action_marker(self):
        with tempfile.TemporaryDirectory() as td:
            # 先记一条
            log_error_report("mem_001", "错了", "user", data_dir=td)
            # 清除
            result = clear_memory_errors("mem_001", data_dir=td)
            assert result == 0
            path = os.path.join(td, "error_reports.jsonl")
            with open(path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) == 2
            clear_record = json.loads(lines[1])
            assert clear_record["memory_id"] == "mem_001"
            assert clear_record["action"] == "clear"

    def test_clear_nonexistent_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            result = clear_memory_errors("mem_xxx", data_dir=td)
            assert result == 0
