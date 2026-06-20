# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 54b6c3da

"""排序回归测试 — 验证 certainty 公式和阈值分类。"""
import math
import sys
sys.path.insert(0, ".")

# certainty = 0.5 * semantic_conf + 0.25 * hit_conf + 0.25 * source_weight
# semantic_conf = 1.0 - distance
# hit_conf = min(ln(hc + 1) / ln(11), 1.0)
# ln(11) ≈ 2.398


def _certainty(distance: float, hit_count: int, source_weight: float) -> float:
    """复制 circuit.py 中的 certainty 计算逻辑。"""
    semantic_conf = 1.0 - distance
    hc = max(hit_count, 0)
    hit_conf = min(math.log(hc + 1) / math.log(11), 1.0) if hc > 0 else 0.0
    c = 0.5 * semantic_conf + 0.25 * hit_conf + 0.25 * source_weight
    return max(0.0, min(1.0, c))


def _classify(certainty: float) -> str:
    """复制 circuit.py 中的阈值逻辑。"""
    if certainty >= 0.6:
        return "fact"
    if certainty >= 0.35:
        return "reference"
    return "background"


_LN11 = math.log(11)


class TestCertaintyFormula:
    """验证 certainty 公式计算正确性。"""

    def test_perfect_match(self):
        """distance=0, hit_count=100, source=semantic(1.0) → 1.0（hit 封顶）。"""
        c = _certainty(0.0, 100, 1.0)
        # semantic=1.0, hit_conf=min(ln(101)/2.398,1)=1.0, source=1.0
        # = 0.5*1.0 + 0.25*1.0 + 0.25*1.0 = 1.0
        assert c == 1.0, f"Expected 1.0, got {c}"

    def test_no_match(self):
        """distance=1.0, hit_count=0, source=co_occurrence(0.35) → 很低。"""
        c = _certainty(1.0, 0, 0.35)
        # semantic=0.0, hit=0.0, source=0.35 → 0.0875
        assert abs(c - 0.0875) < 0.001, f"Expected ~0.0875, got {c}"

    def test_mid_range(self):
        """distance=0.3, hit_count=5, source=tag_match(0.6) → 中等。"""
        c = _certainty(0.3, 5, 0.6)
        # semantic=0.7, hit=ln(6)/2.398=0.747, source=0.6
        # = 0.5*0.7 + 0.25*0.747 + 0.25*0.6 = 0.35+0.187+0.15 = 0.687
        assert abs(c - 0.687) < 0.01, f"Expected ~0.687, got {c}"

    def test_low_hit_count(self):
        """hit_count=1 时的对数加成。"""
        c = _certainty(0.2, 1, 0.85)
        # semantic=0.8, hit=ln(2)/2.398=0.289, source=0.85
        hit_conf = math.log(2) / _LN11
        expected = 0.5 * 0.8 + 0.25 * hit_conf + 0.25 * 0.85
        assert abs(c - expected) < 0.001, f"Expected {expected}, got {c}"

    def test_hit_count_diminishing(self):
        """hit_count 从 2→10 的边际递减。"""
        c2 = _certainty(0.0, 2, 1.0)   # hit_conf = ln(3)/2.398 = 0.458
        c10 = _certainty(0.0, 10, 1.0)  # hit_conf = ln(11)/2.398 = 1.0 (cap)
        diff = c10 - c2
        hit_conf_2 = math.log(3) / _LN11
        expected_diff_c2 = 0.5 * 1.0 + 0.25 * hit_conf_2 + 0.25 * 1.0  # = 0.8645
        expected_diff_c10 = 1.0
        expected_diff = expected_diff_c10 - expected_diff_c2  # = 0.1355
        assert abs(diff - expected_diff) < 0.01, f"递减异常: diff={diff}"

    def test_hit_count_zero(self):
        """hit_count=0 时 hit_conf 应为 0。"""
        c = _certainty(0.0, 0, 1.0)
        expected = 0.5 * 1.0 + 0.25 * 0.0 + 0.25 * 1.0  # = 0.75
        assert abs(c - expected) < 0.001, f"Expected {expected}, got {c}"

    def test_source_weight_boundary(self):
        """极端 source_weight 的边界。"""
        c_high = _certainty(0.5, 0, 1.0)     # semantic=0.5, hit=0, source=1.0
        c_low = _certainty(0.5, 0, 0.35)     # semantic=0.5, hit=0, source=0.35
        # c_high = 0.5*0.5 + 0 + 0.25*1.0 = 0.5
        # c_low  = 0.5*0.5 + 0 + 0.25*0.35 = 0.3375
        assert abs(c_high - 0.5) < 0.001
        assert abs(c_low - 0.3375) < 0.001


class TestCertaintyThresholds:
    """验证 certainty 阈值分类。"""

    def test_fact_threshold(self):
        assert _classify(0.6) == "fact"
        assert _classify(0.75) == "fact"

    def test_reference_threshold(self):
        assert _classify(0.35) == "reference"
        assert _classify(0.5) == "reference"
        assert _classify(0.59) == "reference"

    def test_background_threshold(self):
        assert _classify(0.0) == "background"
        assert _classify(0.2) == "background"
        assert _classify(0.34) == "background"

    def test_boundary_fact_reference(self):
        assert _classify(0.6) == "fact"
        assert _classify(0.599999) == "reference"

    def test_boundary_reference_background(self):
        assert _classify(0.35) == "reference"
        assert _classify(0.349999) == "background"

    def test_source_weight_ordering(self):
        """不同来源排序正确。"""
        src_weights = {
            "semantic": 1.0, "entity_match": 0.8, "kw_match": 0.65,
            "tag_match": 0.6, "time_rhythm": 0.4, "co_occurrence": 0.35,
        }
        certs = {k: _certainty(0.3, 2, v) for k, v in src_weights.items()}
        # 排序应与源可靠性顺序一致
        sorted_certs = sorted(certs.items(), key=lambda x: -x[1])
        expected_order = ["semantic", "entity_match", "kw_match", "tag_match", "time_rhythm", "co_occurrence"]
        actual_order = [k for k, _ in sorted_certs]
        assert actual_order == expected_order, f"排序异常: {actual_order}"
