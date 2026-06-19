"""事实冲突检测单元测试 — 两层漏斗逻辑验证。"""
import json
from unittest.mock import MagicMock, patch

import pytest


class TestContradictionDetection:
    """验证 _detect_fact_contradictions 的两层过滤和标记逻辑。"""

    # 经过验证的 seed 组合：
    #   seed 1.0 vs 0.94 → cosine ~0.8449  (在 0.75~0.95 区间)
    #   seed 1.0 vs 1.00 → cosine  =1.0    (>0.95，视为重复)
    SEED_BASE = 1.0
    SEED_SIMILAR = 0.94   # 产生 ~0.84 相似度
    SEED_IDENTICAL = 1.00 # 产生 ~1.00 相似度
    SEED_SIMILAR_HIGH = 0.97  # 产生 ~0.88 相似度（高于 0.85 阈值，用于同义测试）

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _make_mem(mid: str, tags: list[str], embedding: list[float],
                  timestamp: float, valence: str, summary: str = "",
                  stale: bool = False) -> dict:
        return {
            "id": mid,
            "metadata": {
                "tags": ",".join(tags),
                "timestamp": timestamp,
                "emotion_valence_bin": valence,
                "summary": summary,
                "stale": stale,
                "hit_count": 1,
            },
        }

    @staticmethod
    def _emb(dim: int = 10, seed: float = 1.0) -> list[float]:
        """生成伪 embedding（基于随机种子，可控制相似度）。"""
        import random as _r
        _r.seed(int(seed * 1000))
        return [_r.random() for _ in range(dim)]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(ax * bx for ax, bx in zip(a, b))
        na = (sum(ax * ax for ax in a) ** 0.5) or 1e-10
        nb = (sum(bx * bx for bx in b) ** 0.5) or 1e-10
        return dot / (na * nb)

    # ── 测试 ──────────────────────────────────────────────────

    def test_no_topic_tree_returns_zero(self):
        """无话题树时直接跳过，返回 0。"""
        from app.background.consolidation import ConsolidationEngine
        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        engine._topic_tree = None
        assert engine._detect_fact_contradictions() == 0

    @patch("app.background.consolidation._time")
    def test_same_branch_emotion_flip_detected(self, mock_time):
        """同话题分支 + 情绪翻转 → 应被标记。"""
        from app.background.consolidation import ConsolidationEngine
        from app.memory.qdrant import QdrantService
        import tempfile
        import os

        mock_time.time.return_value = 1_000_000_000.0  # 冻结时间

        engine = ConsolidationEngine.__new__(ConsolidationEngine)

        # Mock TopicTree: tag "辣" → branch ["辣", "川菜", "口味"]
        mock_tree = MagicMock()
        mock_tree.get_branch = lambda t: ["辣", "川菜", "口味"] if t in ("辣", "川菜", "口味") else [t]
        engine._topic_tree = mock_tree

        mock_chroma = MagicMock(spec=QdrantService)
        mock_chroma.list_all_cached.side_effect = lambda *a, **kw: mock_chroma.list_all()
        mock_chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
        ][:limit]
        mock_chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
        ][:limit]
        mock_chroma.list_all_paginated.side_effect = lambda *a, **kw: mock_chroma.list_all()

        old_mem = self._make_mem(
            "old_001", ["辣", "口味"],
            self._emb(seed=self.SEED_BASE), 999_000_000, "positive", "用户喜欢吃辣",
        )
        new_mem = self._make_mem(
            "new_001", ["辣", "口味"],
            self._emb(seed=self.SEED_SIMILAR), 1_000_000_000, "negative", "用户不喜欢吃辣了",
        )
        mock_chroma.list_all.return_value = [old_mem, new_mem]

        # 验证相似度在目标范围内
        sim = self._cosine(self._emb(seed=self.SEED_BASE), self._emb(seed=self.SEED_SIMILAR))
        assert 0.75 < sim < 0.95, f"sim={sim:.3f}，应落在 0.75~0.95"

        mock_chroma._emb_cache = {
            "old_001": self._emb(seed=self.SEED_BASE),
            "new_001": self._emb(seed=self.SEED_SIMILAR),
        }
        mock_chroma.supersede_memory = MagicMock()
        engine._memory = mock_chroma

        engine._state_path = os.path.join(tempfile.gettempdir(), "test_state.json")
        engine._read_state = lambda: {"pending_conflicts": []}
        engine._write_state = MagicMock()
        engine._state_lock = MagicMock()

        result = engine._detect_fact_contradictions()
        assert result == 1
        mock_chroma.supersede_memory.assert_called_once()
        call_args = mock_chroma.supersede_memory.call_args
        assert call_args[0][0] == "old_001"
        assert call_args[0][1] == "new_001"
        assert "情绪翻转" in call_args[0][2]

    @patch("app.background.consolidation._time")
    def test_different_branch_no_false_positive(self, mock_time):
        """不同话题分支 → 不应误检。"""
        from app.background.consolidation import ConsolidationEngine
        from app.memory.qdrant import QdrantService
        import tempfile

        mock_time.time.return_value = 1_000_000_000.0

        engine = ConsolidationEngine.__new__(ConsolidationEngine)

        mock_tree = MagicMock()
        mock_tree.get_branch = lambda t: ["辣", "川菜"] if t == "辣" else ["咖啡机", "厨房"]
        engine._topic_tree = mock_tree

        mock_chroma = MagicMock(spec=QdrantService)
        mock_chroma.list_all_cached.side_effect = lambda *a, **kw: mock_chroma.list_all()
        mock_chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
        ][:limit]
        mock_chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
        ][:limit]
        mock_chroma.list_all_paginated.side_effect = lambda *a, **kw: mock_chroma.list_all()
        old_mem = self._make_mem(
            "old_001", ["辣"], self._emb(seed=self.SEED_BASE), 999_000_000, "positive", "用户喜欢吃辣",
        )
        new_mem = self._make_mem(
            "new_001", ["咖啡机"], self._emb(seed=self.SEED_SIMILAR), 1_000_000_000, "negative", "用户买了新咖啡机",
        )
        mock_chroma.list_all.return_value = [old_mem, new_mem]
        mock_chroma._emb_cache = {
            "old_001": self._emb(seed=self.SEED_BASE),
            "new_001": self._emb(seed=self.SEED_SIMILAR),
        }
        mock_chroma.supersede_memory = MagicMock()
        engine._memory = mock_chroma
        engine._state_path = tempfile.gettempdir() + "/test_state2.json"
        engine._read_state = lambda: {"pending_conflicts": []}
        engine._write_state = MagicMock()
        engine._state_lock = MagicMock()

        result = engine._detect_fact_contradictions()
        assert result == 0

    @patch("app.background.consolidation._time")
    def test_same_emotion_no_false_positive(self, mock_time):
        """同分支但情绪一致 → 不算冲突。"""
        from app.background.consolidation import ConsolidationEngine
        from app.memory.qdrant import QdrantService
        import tempfile

        mock_time.time.return_value = 1_000_000_000.0

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        mock_tree = MagicMock()
        mock_tree.get_branch = lambda t: ["辣", "川菜"]
        engine._topic_tree = mock_tree

        mock_chroma = MagicMock(spec=QdrantService)
        mock_chroma.list_all_cached.side_effect = lambda *a, **kw: mock_chroma.list_all()
        mock_chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
        ][:limit]
        mock_chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
        ][:limit]
        mock_chroma.list_all_paginated.side_effect = lambda *a, **kw: mock_chroma.list_all()
        old_mem = self._make_mem(
            "old_001", ["辣"], self._emb(seed=self.SEED_BASE), 999_000_000, "positive", "用户喜欢吃辣",
        )
        new_mem = self._make_mem(
            "new_001", ["辣"], self._emb(seed=self.SEED_SIMILAR), 1_000_000_000, "positive", "用户依然爱吃辣",
        )
        mock_chroma.list_all.return_value = [old_mem, new_mem]
        mock_chroma._emb_cache = {
            "old_001": self._emb(seed=self.SEED_BASE),
            "new_001": self._emb(seed=self.SEED_SIMILAR),
        }
        mock_chroma.supersede_memory = MagicMock()
        engine._memory = mock_chroma
        engine._state_path = tempfile.gettempdir() + "/test_state3.json"
        engine._read_state = lambda: {"pending_conflicts": []}
        engine._write_state = MagicMock()
        engine._state_lock = MagicMock()

        result = engine._detect_fact_contradictions()
        assert result == 0

    @patch("app.background.consolidation._time")
    def test_too_similar_skipped(self, mock_time):
        """embedding 相似度 > 0.95 → 重复记忆，已有检测，跳过。"""
        from app.background.consolidation import ConsolidationEngine
        from app.memory.qdrant import QdrantService
        import tempfile

        mock_time.time.return_value = 1_000_000_000.0

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        mock_tree = MagicMock()
        mock_tree.get_branch = lambda t: ["辣"]
        engine._topic_tree = mock_tree

        mock_chroma = MagicMock(spec=QdrantService)
        mock_chroma.list_all_cached.side_effect = lambda *a, **kw: mock_chroma.list_all()
        mock_chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
        ][:limit]
        mock_chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
        ][:limit]
        mock_chroma.list_all_paginated.side_effect = lambda *a, **kw: mock_chroma.list_all()
        old_mem = self._make_mem(
            "old_001", ["辣"], self._emb(seed=self.SEED_BASE), 999_000_000, "positive", "用户喜欢吃辣",
        )
        new_mem = self._make_mem(
            "new_001", ["辣"], self._emb(seed=self.SEED_IDENTICAL), 1_000_000_000, "negative", "用户喜欢吃辣",
        )
        sim = self._cosine(self._emb(seed=self.SEED_BASE), self._emb(seed=self.SEED_IDENTICAL))
        assert sim > 0.95, f"sim={sim:.4f}，应该 > 0.95"

        mock_chroma.list_all.return_value = [old_mem, new_mem]
        mock_chroma._emb_cache = {
            "old_001": self._emb(seed=self.SEED_BASE),
            "new_001": self._emb(seed=self.SEED_IDENTICAL),
        }
        mock_chroma.supersede_memory = MagicMock()
        engine._memory = mock_chroma
        engine._state_path = tempfile.gettempdir() + "/test_state4.json"
        engine._read_state = lambda: {"pending_conflicts": []}
        engine._write_state = MagicMock()
        engine._state_lock = MagicMock()

        result = engine._detect_fact_contradictions()
        assert result == 0  # 近似重复，不纳入冲突

    @patch("app.background.consolidation._time")
    def test_neutral_fact_update_detected(self, mock_time):
        """同分支 + 无情绪 + sim低于阈值(sim=0.84) → 路径B语义位移。"""
        from app.background.consolidation import ConsolidationEngine
        from app.memory.qdrant import QdrantService
        import tempfile

        mock_time.time.return_value = 1_000_000_000.0

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        mock_tree = MagicMock()
        mock_tree.get_branch = lambda t: ["城市", "住址", "搬家"]
        engine._topic_tree = mock_tree

        mock_chroma = MagicMock(spec=QdrantService)
        mock_chroma.list_all_cached.side_effect = lambda *a, **kw: mock_chroma.list_all()
        mock_chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
        ][:limit]
        mock_chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
        ][:limit]
        mock_chroma.list_all_paginated.side_effect = lambda *a, **kw: mock_chroma.list_all()
        old_mem = self._make_mem(
            "old_001", ["城市"], self._emb(seed=self.SEED_BASE), 999_000_000, "",
            "用户现在住在北京",
        )
        new_mem = self._make_mem(
            "new_001", ["城市"], self._emb(seed=self.SEED_SIMILAR), 1_000_000_000, "",
            "用户最近搬到了深圳",
        )
        sim = self._cosine(self._emb(seed=self.SEED_BASE), self._emb(seed=self.SEED_SIMILAR))
        assert sim < 0.85, f"sim={sim:.3f} should be < 0.85"

        mock_chroma.list_all.return_value = [old_mem, new_mem]
        mock_chroma._emb_cache = {
            "old_001": self._emb(seed=self.SEED_BASE),
            "new_001": self._emb(seed=self.SEED_SIMILAR),
        }
        mock_chroma.supersede_memory = MagicMock()
        engine._memory = mock_chroma
        engine._state_path = tempfile.gettempdir() + "/test_state_b.json"
        engine._read_state = lambda: {"pending_conflicts": []}
        engine._write_state = MagicMock()
        engine._state_lock = MagicMock()

        result = engine._detect_fact_contradictions()
        assert result == 1
        assert "事实更新" in mock_chroma.supersede_memory.call_args[0][2]

    @patch("app.background.consolidation._time")
    def test_similar_summary_not_misdetected(self, mock_time):
        """同分支 + 无情绪 + 语义高度相似(sim=0.88) → 不是冲突。"""
        from app.background.consolidation import ConsolidationEngine
        from app.memory.qdrant import QdrantService
        import tempfile

        mock_time.time.return_value = 1_000_000_000.0

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        mock_tree = MagicMock()
        mock_tree.get_branch = lambda t: ["辣"]
        engine._topic_tree = mock_tree

        mock_chroma = MagicMock(spec=QdrantService)
        mock_chroma.list_all_cached.side_effect = lambda *a, **kw: mock_chroma.list_all()
        mock_chroma.list_since.side_effect = lambda since_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) >= since_ts
        ][:limit]
        mock_chroma.list_before.side_effect = lambda before_ts, limit=500, **kw: [
            m for m in mock_chroma.list_all()
            if (m.get("metadata") or {}).get("timestamp", 0) < before_ts
        ][:limit]
        mock_chroma.list_all_paginated.side_effect = lambda *a, **kw: mock_chroma.list_all()
        old_mem = self._make_mem(
            "old_001", ["辣"], self._emb(seed=self.SEED_BASE), 999_000_000, "",
            "用户非常喜欢吃辣的东西",
        )
        new_mem = self._make_mem(
            "new_001", ["辣"], self._emb(seed=self.SEED_SIMILAR_HIGH), 1_000_000_000, "",
            "用户还是很爱吃辣的",
        )
        sim = self._cosine(self._emb(seed=self.SEED_BASE), self._emb(seed=self.SEED_SIMILAR_HIGH))
        assert sim >= 0.85, f"sim={sim:.3f} should be >= 0.85"

        mock_chroma.list_all.return_value = [old_mem, new_mem]
        mock_chroma._emb_cache = {
            "old_001": self._emb(seed=self.SEED_BASE),
            "new_001": self._emb(seed=self.SEED_SIMILAR_HIGH),
        }
        mock_chroma.supersede_memory = MagicMock()
        engine._memory = mock_chroma
        engine._state_path = tempfile.gettempdir() + "/test_state_c.json"
        engine._read_state = lambda: {"pending_conflicts": []}
        engine._write_state = MagicMock()
        engine._state_lock = MagicMock()

        result = engine._detect_fact_contradictions()
        assert result == 0  # sim >= 0.85，不算冲突

    def test_supersede_method_sets_correct_fields(self):
        """验证 supersede_memory 正确设置 stale/superseded_by（Qdrant set_payload）。"""
        from app.memory.qdrant import QdrantService
        from unittest.mock import MagicMock

        svc = QdrantService.__new__(QdrantService)
        svc._client = MagicMock()
        svc._collection_name = "memories"
        svc._lock = MagicMock()
        svc._local_index = None
        svc._list_all_cache_lock = MagicMock()
        svc._list_all_cache = None

        svc.supersede_memory("old_123", "new_456", "测试取代")

        svc._client.set_payload.assert_called_once()
        _, kwargs = svc._client.set_payload.call_args
        assert kwargs["points"] == ["old_123"]
        payload = kwargs["payload"]
        assert payload["stale"] is True
        assert payload["superseded_by"] == "new_456"
        assert payload["supersede_reason"] == "测试取代"
        assert "superseded_at" in payload
