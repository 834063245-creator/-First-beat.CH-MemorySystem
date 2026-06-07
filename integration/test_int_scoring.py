"""链路 5：评分与排序集成测试。

compute_score 是纯计算函数，无需 fixture、无需 mock。
验证评分函数在各维度上的单调性和边界行为。
"""
import pytest
from app.retrieval.scoring import compute_score


class TestIntScoring:
    """验证 compute_score 的数学行为。"""

    def test_score_increases_with_similarity(self):
        """同 hit_count，sim 0.3 vs 0.9 → 高相似度分数更高。"""
        low = compute_score(similarity=0.3, hit_count=5)
        high = compute_score(similarity=0.9, hit_count=5)
        assert high > low, f"sim=0.9 应比 sim=0.3 分数高，实际: {high} vs {low}"

    def test_score_increases_with_hits(self):
        """同 sim，hit 1 vs 100 → 高命中分数更高。"""
        cold = compute_score(similarity=0.7, hit_count=1)
        hot = compute_score(similarity=0.7, hit_count=100)
        assert hot > cold, f"hit=100 应比 hit=1 分数高，实际: {hot} vs {cold}"

    def test_score_bounded_01(self):
        """常规输入范围内，分数应在 [0.0, 1.0]。"""
        for sim in [0.0, 0.5, 1.0]:
            for hits in [0, 10, 500]:
                s = compute_score(similarity=sim, hit_count=hits)
                assert 0.0 <= s <= 1.0, f"score(sim={sim}, hits={hits}) = {s} 越界"

    def test_source_bonus_increases_score(self):
        """同参数，source_bonus 0.0 vs 0.1 → 有加成时分数更高。"""
        base = compute_score(similarity=0.6, hit_count=5, source_bonus=0.0)
        bonus = compute_score(similarity=0.6, hit_count=5, source_bonus=0.1)
        assert bonus > base, f"source_bonus=0.1 应比 0.0 分数高，实际: {bonus} vs {base}"

    def test_error_penalty_decreases_score(self):
        """同参数，penalty 0.0 vs 0.3 → 有惩罚时分数更低。"""
        clean = compute_score(similarity=0.6, hit_count=5, error_penalty=0.0)
        penalized = compute_score(similarity=0.6, hit_count=5, error_penalty=0.3)
        assert clean > penalized, f"penalty=0.0 应比 0.3 分数高，实际: {clean} vs {penalized}"
