# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 19663682

"""测试 app/analysis/symmetry.py — 人格对称性分析。"""
import json
import os
import tempfile

import pytest


# ═══════════════════════════════════════════════════════════════
# _distribution_gap — 纯函数
# ═══════════════════════════════════════════════════════════════

class TestDistributionGap:
    def test_identical_returns_zero(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        gap = ps._distribution_gap({"a": 10, "b": 5}, {"a": 10, "b": 5})
        assert abs(gap) < 1e-10, f"gap should be ~0, got {gap}"

    def test_completely_different_returns_one(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        gap = ps._distribution_gap({"a": 10}, {"b": 10})
        assert gap == 1.0

    def test_partial_overlap(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        gap = ps._distribution_gap({"a": 10, "b": 2}, {"a": 5, "c": 8})
        assert 0.0 < gap < 1.0

    def test_empty_dicts(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        assert ps._distribution_gap({}, {}) == 0.0

    def test_one_empty(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        gap = ps._distribution_gap({"a": 10}, {})
        assert gap == 1.0


# ═══════════════════════════════════════════════════════════════
# _load — 文件加载
# ═══════════════════════════════════════════════════════════════

class TestLoad:
    def test_loads_valid_json(self):
        from app.analysis.symmetry import PersonaSymmetry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"tag1": {"rel1": 5, "rel2": 3}}))
            f.flush()
            result = PersonaSymmetry._load(f.name)
        os.unlink(f.name)
        assert result == {"tag1": {"rel1": 5, "rel2": 3}}

    def test_missing_file_returns_empty(self):
        from app.analysis.symmetry import PersonaSymmetry
        result = PersonaSymmetry._load("/nonexistent/path.json")
        assert result == {}

    def test_invalid_json_returns_empty(self):
        from app.analysis.symmetry import PersonaSymmetry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not json{{{")
            f.flush()
            result = PersonaSymmetry._load(f.name)
        os.unlink(f.name)
        assert result == {}


# ═══════════════════════════════════════════════════════════════
# analyze / get_observations — 核心流程
# ═══════════════════════════════════════════════════════════════

class TestAnalyze:
    def test_empty_data_returns_empty(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("/nonexistent/1.json", "/nonexistent/2.json")
        assert ps.analyze() == []

    def test_no_shared_tags(self):
        from app.analysis.symmetry import PersonaSymmetry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f2:
            f1.write(json.dumps({"a": {"x": 5}}))
            f1.flush()
            f2.write(json.dumps({"b": {"y": 3}}))
            f2.flush()
            ps = PersonaSymmetry(f1.name, f2.name)
            result = ps.analyze()
        os.unlink(f1.name)
        os.unlink(f2.name)
        assert result == []

    def test_small_gap_filtered(self):
        from app.analysis.symmetry import PersonaSymmetry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f2:
            f1.write(json.dumps({"python": {"coding": 10, "learning": 10}}))
            f1.flush()
            f2.write(json.dumps({"python": {"coding": 9, "learning": 9}}))
            f2.flush()
            ps = PersonaSymmetry(f1.name, f2.name)
            result = ps.analyze()
        os.unlink(f1.name)
        os.unlink(f2.name)
        # 分布几乎相同，gap < 0.3，应被过滤
        assert result == []

    def test_large_gap_detected(self):
        from app.analysis.symmetry import PersonaSymmetry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f2:
            f1.write(json.dumps({"python": {"coding": 10, "design": 1}}))
            f1.flush()
            f2.write(json.dumps({"python": {"learning": 10, "tutorial": 5}}))
            f2.flush()
            ps = PersonaSymmetry(f1.name, f2.name)
            result = ps.analyze()
        os.unlink(f1.name)
        os.unlink(f2.name)
        # 分布完全不同，gap=1.0
        assert len(result) >= 1
        assert result[0]["tag"] == "python"
        assert result[0]["gap"] >= 0.3

    def test_max_blind_spots_respected(self):
        from app.analysis.symmetry import PersonaSymmetry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f2:
            data = {}
            for i in range(10):
                data[f"tag{i}"] = {"rel_a": 10}
            f1.write(json.dumps(data))
            f1.flush()
            data2 = {}
            for i in range(10):
                data2[f"tag{i}"] = {"rel_b": 10}
            f2.write(json.dumps(data2))
            f2.flush()
            ps = PersonaSymmetry(f1.name, f2.name)
            result = ps.analyze()
        os.unlink(f1.name)
        os.unlink(f2.name)
        assert len(result) <= PersonaSymmetry.MAX_BLIND_SPOTS


class TestGetObservations:
    def test_empty_blind_spots(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        assert ps.get_observations() == []

    def test_with_blind_spots(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        ps._blind_spots = [{
            "tag": "python",
            "gap": 0.85,
            "user_related": ["coding", "debugging"],
            "ai_related": ["tutorial", "learning"],
        }]
        lines = ps.get_observations()
        assert len(lines) == 1
        assert "python" in lines[0]
        assert "coding" in lines[0]
        assert "tutorial" in lines[0]

    def test_blind_spot_property(self):
        from app.analysis.symmetry import PersonaSymmetry
        ps = PersonaSymmetry("", "")
        ps._blind_spots = [{"tag": "test", "gap": 0.5}]
        assert ps.blind_spots == [{"tag": "test", "gap": 0.5}]
