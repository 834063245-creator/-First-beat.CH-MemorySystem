"""测试 app/retrieval/pipeline.py 纯函数 — 提行覆盖。

覆盖：_classify_intent / _resolve_route / _load_error_counts / _load_correction_boosts
"""
import json
import os
import tempfile
from unittest.mock import patch
import pytest


class TestClassifyIntent:
    def test_all_intents(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("你记错了，我没说过") == "conflict"
        assert _classify_intent("还记得上次那个事") == "recall"
        assert _classify_intent("为什么Python这么快") == "ask_fact"
        assert _classify_intent("我今天好难过") == "emotional_sharing"
        assert _classify_intent("今天天气不错") == "casual"

    def test_case_insensitive(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("好难过") == "emotional_sharing"


class TestResolveRoute:
    def test_known_intents(self):
        from app.retrieval.pipeline import _resolve_route
        assert "semantic" in _resolve_route("recall")
        assert _resolve_route("recall")["semantic"] in (20, 100)

    def test_unknown_falls_back_to_recall(self):
        from app.retrieval.pipeline import _resolve_route
        r = _resolve_route("unknown_intent")
        assert _resolve_route("recall") == r


class TestLoadErrorCounts:
    def test_empty_file(self):
        from app.retrieval.pipeline import _load_error_counts
        with tempfile.TemporaryDirectory() as td:
            counts = _load_error_counts(td)
            assert counts == {}

    def test_counts_errors(self):
        from app.retrieval.pipeline import _load_error_counts
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "error_reports.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "m1", "reason": "错了"}) + "\n")
                f.write(json.dumps({"memory_id": "m1", "reason": "又错了"}) + "\n")
                f.write(json.dumps({"memory_id": "m2", "reason": "不对"}) + "\n")
                f.write(json.dumps({"memory_id": "m3", "action": "clear", "reason": ""}) + "\n")
            counts = _load_error_counts(td)
            assert counts["m1"] == 2
            assert counts["m2"] == 1
            assert "m3" not in counts  # clear 被跳过


class TestLoadCorrectionBoosts:
    def test_empty(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as td:
            boosts = _load_correction_boosts(td)
            assert boosts == {}

    def test_edit_boost(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "correction_log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "m1", "tag": "咖啡"}) + "\n")
            boosts = _load_correction_boosts(td)
            assert boosts["m1"] == 0.3

    def test_downvote_penalty(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "correction_log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "m1", "mode": "downvote"}) + "\n")
            boosts = _load_correction_boosts(td)
            assert boosts["m1"] == -0.3

    def test_tag_co_edit_bonus(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "correction_log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "m1", "tag": "咖啡"}) + "\n")
                f.write(json.dumps({"memory_id": "m2", "tag": "咖啡"}) + "\n")
            boosts = _load_correction_boosts(td)
            # 同 tag 二次编辑 → 两人各 +0.1
            assert boosts["m1"] == 0.4  # 0.3 + 0.1
            assert boosts["m2"] == 0.4

    def test_overedit_penalty(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "correction_log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for _ in range(5):
                    f.write(json.dumps({"memory_id": "m1", "tag": "x"}) + "\n")
            boosts = _load_correction_boosts(td)
            # 5 次编辑 → edit_counts=5 → cnt>3 → -0.5 penalty
            assert "m1" in boosts
            # 0.3*5=1.5 + tag_bonus(5*0.1=0.5) - overedit(0.5) = 1.5
            assert boosts["m1"] == pytest.approx(1.5, abs=0.01)
