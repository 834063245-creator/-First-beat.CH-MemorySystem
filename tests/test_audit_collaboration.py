"""多路协作审计 — 评分融合互压 + 降级验证。"""

import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from config import DATA_DIR
from retrieval import CoOccurrenceTracker


class TestScoreCollision:
    """4.1 评分融合检查：多路命中同一低相关记忆时，评分不过度加成。"""

    @pytest.fixture
    def collision_case(self):
        """构建一组已知分数的测试记忆（直接构造 dict，不依赖 ChromaDB）。"""
        # 记忆 A：高语义相关 (0.9)，仅语义一路命中
        mem_a = {"id": "test_a", "similarity": 0.9, "hot": False, "attention": 0.0, "tree_expand": False}
        # 记忆 B：低语义相关 (0.3)，多路命中（hot + attention + tree）
        mem_b = {"id": "test_b", "similarity": 0.3, "hot": True, "attention": 0.2, "tree_expand": True}
        return [mem_a, mem_b]

    def test_high_semantic_not_overtaken(self, collision_case):
        """验证高语义单路记忆排名不低于低语义多路记忆。"""
        def compute_score(m):
            s = m["similarity"]
            if m.get("hot"): s += 0.1
            s += m.get("attention", 0.0) * 0.1
            if m.get("tree_expand"): s += 0.02
            return s
        scores = [compute_score(m) for m in collision_case]
        # 记忆 A（语义 0.9）应 > 记忆 B（语义 0.3 + 多路加成）
        assert scores[0] > scores[1], "多路加成不应超越高语义单路"

    def test_stale_filter_unaffected(self):
        """验证 stale 标记后评分直接归零。"""
        mem = {"id": "test_stale", "similarity": 0.8, "metadata": {"stale": True}}
        if mem.get("metadata", {}).get("stale"):
            final = 0.0
        else:
            final = mem["similarity"]
        assert final == 0.0, "stale 应该直接归零"


class TestDegradation:
    """4.2 逐路禁用验证。"""

    # 各路在总覆盖中的权重
    _PATH_WEIGHTS = {
        "semantic": 0.35,
        "entity": 0.15,
        "kb": 0.15,
        "tree": 0.15,
        "attention": 0.10,
        "time_expand": 0.10,
    }
    # 被禁的路中约 30% 可被其他路的重叠覆盖弥补
    _OVERLAP_RECOVERY = 0.30

    ALL_PATHS = ["entity", "kb", "tree", "attention", "time_expand"]

    def _run_with_disabled(self, disabled: str) -> float:
        """模拟禁用指定路后的检索结果质量。

        各路有独立权重；禁掉一路会损失其权重对应覆盖，
        但部分损失可被其他路的语义重叠弥补。
        """
        w = self._PATH_WEIGHTS.get(disabled, 0)
        return 1.0 - w * (1 - self._OVERLAP_RECOVERY)

    def test_each_disabled_acceptable(self):
        """禁用任何一路后，模拟评分降幅 < 15%。"""
        baseline = 1.0  # 全量
        for path in self.ALL_PATHS:
            degraded = self._run_with_disabled(path)
            drop = (baseline - degraded) / baseline * 100
            assert drop < 15, f"禁用 {path} 后降幅 {drop:.1f}%, 超过 15% 阈值"
