# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 1ee19ad6

"""测试 app/brain/export_training_data.py — main() CLI 入口。

覆盖：main() 从 chat_history.jsonl 读取 → 标注 → 输出 training_data.jsonl。
"""
import json
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestExportMain:
    def test_main_writes_training_data(self):
        """用假 JSONL 测试 main() 完整文件 I/O 流程。"""
        with tempfile.TemporaryDirectory() as td:
            # 假 project root
            proj = os.path.join(td, "project")
            backend_data = os.path.join(proj, "backend", "data")
            os.makedirs(backend_data, exist_ok=True)

            history_path = os.path.join(backend_data, "chat_history.jsonl")
            test_msgs = [
                {"user_message": "你好，今天天气真好"},
                {"user_message": "帮我写段代码吧"},
                {"user_message": "我最近很难过"},
                {"user_message": "你是谁"},
                {"user_message": "ab"},  # 太短，应跳过
            ]
            with open(history_path, "w", encoding="utf-8") as f:
                for m in test_msgs:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            # 假 instances 路径（不存在）
            alt_path = os.path.join(proj, "instances", "predecessor", "data", "chat_history.jsonl")
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            # 留空 — main() 会尝试两个路径

            # 输出目录
            brain_dir = os.path.join(proj, "app", "brain")
            os.makedirs(brain_dir, exist_ok=True)
            output_file = os.path.join(brain_dir, "training_data.jsonl")

            # mock __file__ 和 os.path 让 main() 认为 project_root = proj
            fake_file = os.path.join(proj, "app", "brain", "export_training_data.py")

            def _fake_abspath(p):
                if p == fake_file:
                    return fake_file
                return os.path.abspath(p)

            with patch("app.brain.export_training_data.__file__", fake_file):
                with patch("app.brain.export_training_data.os.path.abspath", side_effect=_fake_abspath):
                    # 同时 patch 默认的 os.path 行为
                    import app.brain.export_training_data as _mod
                    _mod.main()

            # 验证输出文件存在且有内容
            assert os.path.exists(output_file), f"输出文件未生成: {output_file}"
            with open(output_file, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]

            # 4 条有效消息（"ab" 被跳过）
            assert len(lines) == 4
            records = [json.loads(l) for l in lines]
            intents = [r["intent"] for r in records]
            emotions = [r["emotion"] for r in records]
            assert intents[0] == "casual"       # 你好今天天气真好
            assert intents[1] == "request"       # 帮我写段代码吧
            assert intents[2] == "emotional_sharing"  # 我最近很难过
            assert intents[3] == "meta"          # 你是谁
            assert emotions[2] == "negative"     # 难过

    def test_main_no_jsonl_no_crash(self):
        """没有任何 chat_history.jsonl 时 main() 不崩溃。"""
        with tempfile.TemporaryDirectory() as td:
            proj = os.path.join(td, "empty_project")
            brain_dir = os.path.join(proj, "app", "brain")
            os.makedirs(brain_dir, exist_ok=True)

            fake_file = os.path.join(proj, "app", "brain", "export_training_data.py")

            def _fake_abspath(p):
                return fake_file if os.path.basename(p) == "export_training_data.py" else os.path.abspath(p)

            with patch("app.brain.export_training_data.__file__", fake_file):
                with patch("app.brain.export_training_data.os.path.abspath", side_effect=_fake_abspath):
                    import app.brain.export_training_data as _mod
                    # 不应抛异常
                    _mod.main()

            # 输出文件应存在但为空（或只有 header）
            output = os.path.join(brain_dir, "training_data.jsonl")
            assert os.path.exists(output)
