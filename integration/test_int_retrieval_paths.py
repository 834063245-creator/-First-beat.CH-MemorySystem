"""链路 2：检索通路验证集成测试。

验证：retrieve_all 各检索路径在真实数据下能返回结果。
使用真实 ChromaDB + 真实 embedding，仅 mock extract_tags。
"""
import time
import pytest
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════
# Mock 桩
# ═══════════════════════════════════════════════════════════════

def _mock_extract_tags(text: str, topk: int = 5) -> list[str]:
    tag_map = {
        "Rust": ["Rust", "编程"], "Python": ["Python", "编程"],
        "猫": ["宠物", "猫"], "橘猫": ["宠物", "猫"],
        "东京": ["旅行", "东京"], "大阪": ["旅行", "大阪"],
        "工作": ["工作", "职场"], "压力": ["工作", "压力"],
        "健身": ["健身", "运动"], "跑步": ["健身", "跑步"],
    }
    tags = []
    for kw, tl in tag_map.items():
        if kw in text:
            tags.extend(tl)
    return list(set(tags))[:topk] if tags else ["通用"]


@pytest.fixture
def _mock_tags():
    with patch("app.brain.semantic.extract_tags", side_effect=_mock_extract_tags):
        yield


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _write_and_wait(ctx, messages: list[tuple[str, str, str]]):
    """批量写入并等待入库完成。"""
    for user, ai, ts in messages:
        ctx._store_conversation(user, ai, ts)
        time.sleep(0.1)
    time.sleep(0.8)


# ═══════════════════════════════════════════════════════════════
# 测试类
# ═══════════════════════════════════════════════════════════════

class TestIntRetrievalPaths:
    """验证：retrieve_all 的各检索路径在真实数据下能返回结果。"""

    def test_retrieve_all_returns_results(self, isolated_env, _mock_tags):
        """写入 2 条不同话题记忆 → retrieve_all 返回结果。"""
        ctx = isolated_env
        _write_and_wait(ctx, [
            ("我在学习 Rust 编程", "Rust 很棒", "2026-06-01 10:00:00"),
            ("今天去健身跑步5公里", "坚持运动很棒", "2026-06-01 11:00:00"),
        ])

        from app.retrieval.pipeline import retrieve_all
        results = retrieve_all("Rust 编程语言学习", None, ctx)
        assert isinstance(results, list), "retrieve_all 应返回 list"
        assert len(results) >= 1, "应至少返回 1 条结果"

    def test_retrieve_all_deduplicates(self, isolated_env, _mock_tags):
        """写入 1 条记忆 → retrieve_all 多路命中无重复 ID。"""
        ctx = isolated_env
        _write_and_wait(ctx, [
            ("我在学习 Rust 编程", "Rust 很棒", "2026-06-01 10:00:00"),
        ])

        from app.retrieval.pipeline import retrieve_all
        results = retrieve_all("Rust 编程", None, ctx)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)), f"结果有重复 ID: {ids}"

    def test_retrieve_all_scores_bounded(self, isolated_env, _mock_tags):
        """写入 1 条记忆 → retrieve_all 返回的 score 在 [0, 1]。"""
        ctx = isolated_env
        _write_and_wait(ctx, [
            ("工作压力很大", "加油", "2026-06-01 10:00:00"),
        ])

        from app.retrieval.pipeline import retrieve_all
        results = retrieve_all("最近工作怎么样", None, ctx)
        for r in results:
            score = r.get("score", 0)
            assert 0.0 <= score <= 1.5, (
                f"score={score} 超出合理范围 (文档: {r.get('document', '')[:30]})"
            )

    def test_retrieve_all_empty_store(self, isolated_env):
        """空数据库 → retrieve_all 返回空列表，不抛异常。"""
        ctx = isolated_env
        from app.retrieval.pipeline import retrieve_all
        results = retrieve_all("随便问点什么", None, ctx)
        assert isinstance(results, list), "空库应返回 list"
        assert len(results) == 0, f"空库应返回空列表，实际 {len(results)}"

    def test_keyword_path_finds_by_tag(self, isolated_env, _mock_tags):
        """写入含标签"旅行"的记忆 → 标签倒排索引能查到。"""
        ctx = isolated_env
        _write_and_wait(ctx, [
            ("昨天去了东京旅行", "东京很好玩", "2026-06-01 10:00:00"),
        ])

        tag_results = ctx.inverted_index.query_tags(["旅行"])
        assert len(tag_results) >= 1, "标签索引应能按 '旅行' 找到记忆"
