"""链路 1：写入→检索闭环集成测试。

验证：对话写入后，能通过各检索通路找回来。
使用真实 ChromaDB + 真实 embedding（Ollama bge-m3），仅 mock extract_tags。
BENCHMARK_MODE=true 路径：embed + 标签 + ChromaDB + 倒排索引。
"""
import time
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.real_embed  # 需要真实 Ollama embedding


# ═══════════════════════════════════════════════════════════════
# Mock 桩
# ═══════════════════════════════════════════════════════════════

def _mock_extract_tags(text: str, topk: int = 5) -> list[str]:
    """确定性标签提取：按关键词匹配返回固定标签。"""
    tag_map = {
        "Rust": ["Rust", "编程"],
        "Python": ["Python", "编程"],
        "猫": ["宠物", "猫"],
        "橘猫": ["宠物", "猫"],
        "狗": ["宠物", "狗"],
        "东京": ["旅行", "东京"],
        "大阪": ["旅行", "大阪"],
        "工作": ["工作", "职场"],
        "压力": ["工作", "压力"],
        "健身": ["健身", "运动"],
        "跑步": ["健身", "跑步"],
    }
    tags = []
    for kw, tl in tag_map.items():
        if kw in text:
            tags.extend(tl)
    return list(set(tags))[:topk] if tags else ["通用"]


@pytest.fixture
def _mock_tags():
    """自动 mock extract_tags，测试结束后恢复。"""
    with patch("app.brain.semantic.extract_tags", side_effect=_mock_extract_tags):
        yield


# ═══════════════════════════════════════════════════════════════
# 测试类
# ═══════════════════════════════════════════════════════════════

class TestIntWriteRetrieve:
    """验证：对话写入后，能通过各检索通路找回来。"""

    def test_write_increments_chroma_count(self, isolated_env, _mock_tags):
        """写入 1 条对话 → chroma_service.count() +1。"""
        ctx = isolated_env
        before = ctx.chroma_service.count()
        ctx._store_conversation(
            "我在学习 Rust 编程", "Rust 的所有权系统很棒",
            "2026-06-01 10:00:00"
        )
        time.sleep(0.5)
        assert ctx.chroma_service.count() == before + 1

    def test_write_populates_inverted_index(self, isolated_env, _mock_tags):
        """写入含"Rust"的对话 → 倒排索引按关键词能查到。"""
        ctx = isolated_env
        ctx._store_conversation(
            "我在学习 Rust 编程", "Rust 很棒",
            "2026-06-01 10:00:00"
        )
        time.sleep(0.5)
        results = ctx.inverted_index.query(["Rust"], min_match=1)
        assert len(results) >= 1, "倒排索引应能按 'Rust' 找到写入的记忆"

    def test_write_populates_tag_index(self, isolated_env, _mock_tags):
        """写入含"Rust"的对话 → 标签倒排索引能按"编程"查到。"""
        ctx = isolated_env
        ctx._store_conversation(
            "我在学习 Rust 编程", "Rust 很棒",
            "2026-06-01 10:00:00"
        )
        time.sleep(0.5)
        tag_results = ctx.inverted_index.query_tags(["编程"])
        assert len(tag_results) >= 1, "标签索引应能按 '编程' 找到写入的记忆"

    def test_write_stores_correct_metadata(self, isolated_env, _mock_tags):
        """写入对话 → ChromaDB 记录的 metadata 含 tags 和 summary。"""
        ctx = isolated_env
        ctx._store_conversation(
            "我在学习 Rust 编程", "Rust 的所有权系统很棒",
            "2026-06-01 10:00:00"
        )
        time.sleep(0.5)
        all_memories = ctx.chroma_service.list_all()
        assert len(all_memories) >= 1, "应至少有 1 条记忆"
        latest = all_memories[-1]
        meta = latest.get("metadata", {})
        assert "tags" in meta, "metadata 应包含 tags 字段"
        assert "Rust" in meta["tags"] or "编程" in meta["tags"], (
            f"tags 应包含 Rust 或 编程，实际: {meta['tags']}"
        )

    def test_write_then_semantic_search(self, isolated_env, _mock_tags):
        """写入"橘猫去宠物医院" → 用相似语义查询能找到它（核心闭环）。"""
        ctx = isolated_env
        from app.llm.embed import local_embed

        ctx._store_conversation(
            "我的橘猫今天去了宠物医院看病", "猫咪要注意健康",
            "2026-06-01 10:00:00"
        )
        time.sleep(0.8)

        query_emb = local_embed("猫咪生病了怎么办")
        assert query_emb is not None, "embedding 不应为 None"

        results = ctx.chroma_service._collection.query(
            query_embeddings=[query_emb], n_results=5
        )
        docs = results.get("documents", [[]])[0]
        assert len(docs) >= 1, "语义检索应能找到至少一条结果"
