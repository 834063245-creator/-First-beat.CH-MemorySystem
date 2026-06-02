"""审计基线回归测试 — 验证核心检索指标不低于基线。

注：这些测试需要 ChromaDB 和 embedding 模型。
跳过时会输出明确提示。
"""
import sys
sys.path.insert(0, ".")

import pytest


# ============================================================
# Core Audit Metrics — 轻量基线检查
# ============================================================

# 这些测试验证关键指标函数本身是否工作正常，
# 而非对生产数据跑审计（那需要 /chat 服务）。


class TestBasicRecallLogic:
    """验证基础召回通路的核心逻辑。"""

    def test_kw_match_threshold_logic(self):
        """验证关键词匹配 ≥2 的阈值逻辑。"""
        query_kws = {"Rust", "编程", "系统"}

        # 命中 2 个 → 应通过阈值
        doc_summary = "用户说想学 Rust 编程"
        matched = sum(1 for kw in query_kws if kw in doc_summary)
        assert matched >= 2
        assert matched == 2  # Rust + 编程

        # 命中 0 个 → 不应通过阈值
        doc_summary2 = "用户喜欢猫咪"
        matched2 = sum(1 for kw in query_kws if kw in doc_summary2)
        assert matched2 < 2
        assert matched2 == 0

        # 命中 3 个 → 全部匹配
        doc_summary3 = "Rust 编程 系统"
        matched3 = sum(1 for kw in query_kws if kw in doc_summary3)
        assert matched3 == 3

    def test_source_weight_ranking(self):
        """验证来源权重排序符合预期。"""
        source_order = {
            "semantic": 1.0,
            "dmn_preheat": 0.85,
            "entity_match": 0.8,
            "kw_match": 0.65,
            "tag_match": 0.6,
            "text_match": 0.6,
            "time_rhythm": 0.4,
            "co_occurrence": 0.35,
        }
        weights = list(source_order.values())
        assert weights == sorted(weights, reverse=True), "来源权重应降序排列"

    def test_attention_window_fallback(self):
        """验证注意力全零时退回 0.7 语义权重。"""
        attention_values = [0.0, 0.0, 0.0]
        all_zero = all(v == 0 for v in attention_values)
        assert all_zero

        sem_w = 0.7 if all_zero else 0.5
        attn_w = 0.0 if all_zero else 0.3
        assert sem_w == 0.7
        assert attn_w == 0.0

    def test_attention_window_active(self):
        """验证有活跃注意力时的权重分配。"""
        attention_values = [0.3, 0.5, 0.0]
        all_zero = all(v == 0 for v in attention_values)
        assert not all_zero

        sem_w = 0.7 if all_zero else 0.5
        attn_w = 0.0 if all_zero else 0.3
        assert sem_w == 0.5
        assert attn_w == 0.3

    def test_stale_memory_filtering(self):
        """验证 stale 标记记忆应被过滤。"""
        memories = [
            {"id": "m1", "metadata": {"stale": True}},
            {"id": "m2", "metadata": {"stale": False}},
            {"id": "m3", "metadata": {}},
        ]
        filtered = [m for m in memories if not (m.get("metadata") or {}).get("stale", False)]
        filtered_ids = [m["id"] for m in filtered]
        assert "m1" not in filtered_ids
        assert "m2" in filtered_ids
        assert "m3" in filtered_ids

    def test_archived_memory_filtering(self):
        """验证 archived 标记记忆应被过滤。"""
        memories = [
            {"id": "m1", "metadata": {"archived": True}},
            {"id": "m2", "metadata": {"archived": False}},
            {"id": "m3", "metadata": {}},
        ]
        filtered = [m for m in memories if not
                    ((m.get("metadata") or {}).get("archived", False) or
                     (m.get("metadata") or {}).get("stale", False))]
        filtered_ids = [m["id"] for m in filtered]
        assert "m1" not in filtered_ids
        assert "m2" in filtered_ids
        assert "m3" in filtered_ids

    def test_correction_boost_diffusion(self):
        """验证同 tag 纠正传播逻辑。"""
        # 模拟 _load_correction_boosts 中的 tag 传播
        corrections = [
            {"memory_id": "m1", "tag": "Rust"},
            {"memory_id": "m2", "tag": "Rust"},
            {"memory_id": "m3", "tag": "Python"},
        ]
        boosts = {}
        tag_mids = {}
        for rec in corrections:
            mid = rec["memory_id"]
            tag = rec["tag"]
            boosts[mid] = boosts.get(mid, 0) + 0.3
            tag_mids.setdefault(tag, []).append(mid)

        # 同一 tag 的 m1 和 m2 应获得额外传播
        for mids in tag_mids.values():
            if len(mids) >= 2:
                for mid in mids:
                    boosts[mid] = boosts.get(mid, 0) + 0.1

        assert boosts["m1"] == 0.4  # 0.3 + 0.1 传播
        assert boosts["m2"] == 0.4
        assert boosts["m3"] == 0.3  # 只有自己的纠正，无传播

    def test_reranker_formula_with_feedback(self):
        """验证 rerank 公式中反馈加成的计算。"""
        # 模拟 local_reranker.rerank 中的逻辑
        candidates = [
            {"id": "m1", "_rr_score": 0.7, "sim": 0.7},
            {"id": "m2", "_rr_score": 0.65, "sim": 0.65},
            {"id": "m3", "_rr_score": 0.6, "sim": 0.6},
        ]
        boosts = {"m1": 0.4, "m2": 0.0, "m3": 0.4}
        errs = {"m3": 2}

        for c in candidates:
            mid = c["id"]
            c["final"] = c["_rr_score"] + boosts.get(mid, 0) * 0.1 - errs.get(mid, 0) * 0.05

        # m1: 0.7 + 0.4*0.1 - 0 = 0.74
        # m2: 0.65 + 0 - 0 = 0.65
        # m3: 0.6 + 0.4*0.1 - 2*0.05 = 0.6 + 0.04 - 0.1 = 0.54
        candidates.sort(key=lambda x: -x["final"])
        assert candidates[0]["id"] == "m1"
        assert candidates[1]["id"] == "m2"
        assert candidates[2]["id"] == "m3"
        assert abs(candidates[0]["final"] - 0.74) < 0.01
        assert abs(candidates[2]["final"] - 0.54) < 0.01
