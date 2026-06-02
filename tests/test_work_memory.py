"""Tests for work memory window — ChatHistory.get_recent with token_budget."""
import os
import pytest

from chat_history import ChatHistory


def _make_history(tmp_path, count=20, text_len=60):
    """创建含 count 条记录的 ChatHistory，时间戳递增（先旧后新）。"""
    path = str(tmp_path / "chat.jsonl")
    ch = ChatHistory(path=path, max_memory=200)
    for i in range(count):
        user = f"用户消息{i:03d}" + "x" * (text_len - 6)
        ai = f"AI回复{i:03d}" + "y" * (text_len - 6)
        # 时间戳递增：i=0 → day 14, i=9 → day 23
        ch.append(user, ai, f"2026-05-{14 + (i % 10):02d} 12:00:00")
    return ch


# ===================================================================
# get_recent with token_budget
# ===================================================================

class TestGetRecentByTokenBudget:

    def test_token_budget_returns_recent_excluding_overflow(self, tmp_path):
        """验证 token 预算截断点正确：超过预算的最后一条应该被排除。"""
        ch = _make_history(tmp_path, count=10, text_len=60)
        # 每条记录 ≈ (60 + 60) // 3 = 40 tokens，5 条 = 200 tokens
        # 预算 199 应该只拿 4 条（200 就超出）
        result = ch.get_recent(token_budget=199)
        assert len(result) == 4, f"期望 4 条，实际 {len(result)}"

    def test_token_budget_minimum_one(self, tmp_path):
        """token 预算极小时至少返回 1 条。"""
        ch = _make_history(tmp_path, count=10, text_len=60)
        result = ch.get_recent(token_budget=1)
        assert len(result) >= 1, "预算为 1 时至少返回 1 条"
        # 验证是最新那条
        assert "用户消息009" in result[0].get("user_message", "")

    def test_token_budget_minimum_one_single_record(self, tmp_path):
        """单条记录时 token_budget=1 仍返回该条。"""
        ch = _make_history(tmp_path, count=1, text_len=60)
        result = ch.get_recent(token_budget=1)
        assert len(result) == 1
        assert "用户消息000" in result[0].get("user_message", "")

    def test_token_budget_all_records(self, tmp_path):
        """token 预算大于全部记录时返回全部。"""
        ch = _make_history(tmp_path, count=5, text_len=60)
        result = ch.get_recent(token_budget=1_000_000)
        assert len(result) == 5

    def test_token_budget_chronological_order(self, tmp_path):
        """返回结果按时间正序排列（从旧到新）。"""
        ch = _make_history(tmp_path, count=10, text_len=60)
        result = ch.get_recent(token_budget=1_000_000)
        timestamps = [r.get("timestamp", "") for r in result]
        assert timestamps == sorted(timestamps), "结果应按时间正序"

    def test_token_budget_all_records_positive_order(self, tmp_path):
        """验证第一条是最旧的，最后一条是最新的。"""
        ch = _make_history(tmp_path, count=5, text_len=60)
        result = ch.get_recent(token_budget=1_000_000)
        assert "用户消息000" in result[0].get("user_message", "")
        assert "用户消息004" in result[-1].get("user_message", "")

    def test_token_budget_none_fallback(self, tmp_path):
        """token_budget=None 时走旧逻辑，默认返回 5 条。"""
        ch = _make_history(tmp_path, count=20, text_len=60)
        result = ch.get_recent(n=5, token_budget=None)
        assert len(result) == 5, f"期望 5 条，实际 {len(result)}"
        # 验证是最新 5 条
        assert "用户消息019" in result[-1].get("user_message", "")
        assert "用户消息015" in result[0].get("user_message", "")

    def test_token_budget_none_fallback_custom_n(self, tmp_path):
        """token_budget=None 时 n 参数可自定义。"""
        ch = _make_history(tmp_path, count=20, text_len=60)
        result = ch.get_recent(n=3, token_budget=None)
        assert len(result) == 3

    def test_empty_history(self, tmp_path):
        """ChatHistory 为空时返回空列表。"""
        path = str(tmp_path / "empty.jsonl")
        ch = ChatHistory(path=path, max_memory=500)
        result = ch.get_recent(token_budget=500_000)
        assert result == []

    def test_empty_history_none_fallback(self, tmp_path):
        """ChatHistory 为空时，token_budget=None 也返回空。"""
        path = str(tmp_path / "empty.jsonl")
        ch = ChatHistory(path=path, max_memory=500)
        result = ch.get_recent(n=5, token_budget=None)
        assert result == []

    def test_token_budget_large_text(self, tmp_path):
        """长文本情况下 token 估算正确。"""
        ch = _make_history(tmp_path, count=3, text_len=3000)
        # 每条 ≈ 6000 // 3 = 2000 tokens，1 条 ≈ 2000
        result = ch.get_recent(token_budget=2500)
        assert len(result) == 1, "长文本应在预算内只返回 1 条"

    def test_token_budget_small_text(self, tmp_path):
        """短文本情况下多放几条。"""
        ch = _make_history(tmp_path, count=10, text_len=30)
        # 每条 ≈ 60 // 3 = 20 tokens，预算 100 → 5 条
        result = ch.get_recent(token_budget=100)
        assert len(result) == 5

    def test_chroma_id_merged_in_result(self, tmp_path):
        """验证返回结果中 chroma_id 已合并。"""
        ch = _make_history(tmp_path, count=3, text_len=60)
        # 模拟 chroma_id 回写
        ts = ch.records[-1]["timestamp"]
        ch.update_chroma_id(ts, "test-chroma-id-123")
        result = ch.get_recent(token_budget=500_000)
        last = result[-1]
        assert last.get("chroma_id") == "test-chroma-id-123"

    def test_estimate_tokens_zero_for_empty(self, tmp_path):
        """空文本的 token 估算至少为 1（防除零）。"""
        ch = _make_history(tmp_path, count=1, text_len=10)
        assert ch._estimate_tokens("") >= 1
        assert ch._estimate_tokens("") >= 1

    def test_estimate_tokens_positive(self, tmp_path):
        """非空文本的 token 估算为正数。"""
        ch = _make_history(tmp_path, count=1, text_len=10)
        assert ch._estimate_tokens("hello world") >= 1
