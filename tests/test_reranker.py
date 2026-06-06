"""测试 app/retrieval/reranker.py — 本地精排器。

覆盖：rerank 余弦相似度排序、纠正反馈加成、注意力权重。
"""
import numpy as np
from unittest.mock import patch, MagicMock
from app.retrieval.reranker import rerank


# 固定 embedding 向量便于断言
DUMMY_EMBED = np.array([0.5] * 1024, dtype=np.float32)
DUMMY_EMBED_NORM = DUMMY_EMBED / np.linalg.norm(DUMMY_EMBED)


class TestRerank:
    def test_empty_candidates_returns_empty(self):
        result = rerank("查询", [])
        assert result == []

    @patch("app.llm.embed.local_embed")
    def test_scores_and_sorts_by_similarity(self, mock_embed):
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        candidates = [
            {"id": "a", "summary": "文本A"},
            {"id": "b", "summary": "文本B"},
            {"id": "c", "summary": "文本C"},
        ]
        result = rerank("测试查询", candidates, top_k=2)
        assert len(result) == 2
        assert "_rr_score" in result[0]
        # 排序：分数降序
        assert result[0]["_rr_score"] >= result[1]["_rr_score"]

    @patch("app.llm.embed.local_embed")
    def test_uses_document_fallback(self, mock_embed):
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        candidates = [
            {"id": "a", "document": "文档内容"},
        ]
        result = rerank("查询", candidates)
        assert len(result) == 1
        assert "_rr_score" in result[0]

    @patch("app.llm.embed.local_embed")
    def test_handles_missing_text_gracefully(self, mock_embed):
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        candidates = [
            {"id": "a", "summary": "", "document": ""},
            {"id": "b", "summary": "有效文本"},
        ]
        result = rerank("查询", candidates)
        # 空文本的分数为 0
        empty_item = [c for c in result if c["id"] == "a"]
        assert empty_item
        assert empty_item[0]["_rr_score"] == 0.0

    @patch("app.llm.embed.local_embed")
    def test_correction_boost_adds_score(self, mock_embed):
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        candidates = [
            {"id": "boosted", "summary": "需要加成的记忆"},
            {"id": "normal", "summary": "普通记忆"},
        ]
        result = rerank(
            "查询", candidates,
            correction_boosts={"boosted": 5.0},  # boost * 0.1 = +0.5
        )
        boosted = [c for c in result if c["id"] == "boosted"][0]
        normal = [c for c in result if c["id"] == "normal"][0]
        assert boosted["_rr_score"] > normal["_rr_score"]

    @patch("app.llm.embed.local_embed")
    def test_error_penalty_reduces_score(self, mock_embed):
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        candidates = [
            {"id": "bad", "summary": "被报告错误的记忆"},
            {"id": "good", "summary": "正常记忆"},
        ]
        result = rerank(
            "查询", candidates,
            error_counts={"bad": 10},  # penalty = 10 * 0.05 = -0.5
        )
        bad = [c for c in result if c["id"] == "bad"][0]
        good = [c for c in result if c["id"] == "good"][0]
        assert bad["_rr_score"] < good["_rr_score"]

    @patch("app.llm.embed.local_embed")
    def test_attention_boost_weighted(self, mock_embed):
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        candidates = [
            {"id": "focused", "summary": "注意力焦点"},
            {"id": "other", "summary": "其他记忆"},
        ]
        result = rerank(
            "查询", candidates,
            attention_boosts={"focused": 0.8},
            attention_weight=1.0,  # 全权重
        )
        focused = [c for c in result if c["id"] == "focused"][0]
        other = [c for c in result if c["id"] == "other"][0]
        assert focused["_rr_score"] > other["_rr_score"]

    @patch("app.llm.embed.local_embed")
    def test_query_embed_none_falls_back(self, mock_embed):
        mock_embed.return_value = None
        candidates = [
            {"id": "a", "summary": "文本A"},
            {"id": "b", "summary": "文本B"},
        ]
        result = rerank("查询", candidates, top_k=1)
        assert len(result) == 1
        assert result[0]["id"] == "a"

    @patch("app.llm.embed.local_embed")
    def test_handles_embed_exception(self, mock_embed):
        """当 local_embed 抛异常时，回退原始顺序。"""
        mock_embed.side_effect = Exception("embed error")
        candidates = [
            {"id": "a", "summary": "文本A"},
            {"id": "b", "summary": "文本B"},
        ]
        result = rerank("查询", candidates, top_k=1)
        assert len(result) == 1
        assert result[0]["id"] == "a"

    @patch("app.llm.embed.local_embed")
    def test_truncates_long_summary(self, mock_embed):
        """超过 200 字符的 summary 被截断。"""
        mock_embed.return_value = DUMMY_EMBED_NORM.tolist()
        long_text = "X" * 500
        candidates = [{"id": "a", "summary": long_text}]
        result = rerank("查询", candidates)
        assert len(result) == 1
        # embed 被调用时 text 应该 ≤ 200 字符
        # 验证：所有调用参数长度都不超过 200（第一次是 query embed）
        text_args = [c[0][0] for c in mock_embed.call_args_list if c[0]]
        # 过滤 query embed
        candidate_texts = [t for t in text_args if t != "查询"]
        for t in candidate_texts:
            assert len(t) <= 200
