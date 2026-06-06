"""测试 app/brain/metrics.py — 训练指标持久化。

覆盖：record_training / record_shadow_test / record_benchmark / _load / _default。
"""
import json
import os
import tempfile
from unittest.mock import patch
import app.brain.metrics as metrics_mod


class TestDefaultSchema:
    def test_has_expected_keys(self):
        d = metrics_mod._default()
        assert "schema_version" in d
        assert "models" in d
        assert "shadow_tests" in d
        assert "benchmarks" in d
        assert d["schema_version"] == "1.0"


class TestLoad:
    @patch.object(metrics_mod, "METRICS_PATH", "/nonexistent/path/_test_metrics.json")
    def test_returns_default_when_no_file(self):
        data = metrics_mod._load()
        assert data["schema_version"] == "1.0"
        assert data["models"] == {}

    def test_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            test_path = os.path.join(td, "training_metrics.json")
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump({"schema_version": "1.0", "models": {"intent": []}}, f)
            with patch.object(metrics_mod, "METRICS_PATH", test_path):
                data = metrics_mod._load()
                assert "intent" in data["models"]

    def test_corrupted_json_returns_default(self):
        with tempfile.TemporaryDirectory() as td:
            test_path = os.path.join(td, "training_metrics.json")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("not valid json{{{")
            with patch.object(metrics_mod, "METRICS_PATH", test_path):
                data = metrics_mod._load()
                assert data["schema_version"] == "1.0"


class TestRecordTraining:
    def test_appends_record(self):
        with tempfile.TemporaryDirectory() as td:
            test_path = os.path.join(td, "training_metrics.json")
            with patch.object(metrics_mod, "METRICS_PATH", test_path):
                metrics_mod.record_training("intent", {
                    "val_acc": 0.95, "val_loss": 0.12,
                    "epochs": 5, "batch_size": 32,
                })
                data = metrics_mod._load()
                assert "intent" in data["models"]
                records = data["models"]["intent"]
                assert len(records) == 1
                assert records[0]["val_acc"] == 0.95
                assert "timestamp" in records[0]

    def test_multiple_records_accumulate(self):
        with tempfile.TemporaryDirectory() as td:
            test_path = os.path.join(td, "training_metrics.json")
            with patch.object(metrics_mod, "METRICS_PATH", test_path):
                metrics_mod.record_training("intent", {"val_acc": 0.9})
                metrics_mod.record_training("intent", {"val_acc": 0.95})
                data = metrics_mod._load()
                assert len(data["models"]["intent"]) == 2


class TestRecordShadowTest:
    def test_appends_shadow_test(self):
        with tempfile.TemporaryDirectory() as td:
            test_path = os.path.join(td, "training_metrics.json")
            with patch.object(metrics_mod, "METRICS_PATH", test_path):
                metrics_mod.record_shadow_test({
                    "intent_match_rate": 0.85,
                    "emotion_match_rate": 0.90,
                })
                data = metrics_mod._load()
                assert len(data["shadow_tests"]) == 1
                assert data["shadow_tests"][0]["intent_match_rate"] == 0.85


class TestRecordBenchmark:
    def test_appends_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            test_path = os.path.join(td, "training_metrics.json")
            with patch.object(metrics_mod, "METRICS_PATH", test_path):
                metrics_mod.record_benchmark({
                    "avg_intent_ms": 12.5,
                    "avg_emotion_ms": 8.3,
                    "combined_ms": 20.8,
                })
                data = metrics_mod._load()
                assert len(data["benchmarks"]) == 1
                assert data["benchmarks"][0]["avg_intent_ms"] == 12.5
