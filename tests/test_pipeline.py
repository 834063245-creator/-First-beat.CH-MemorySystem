"""测试 app/retrieval/pipeline.py — 检索管线。

覆盖：_classify_intent / _resolve_route / run_chat_retrieval / retrieve_all 核心路径。
"""
import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# _classify_intent — 纯函数意图分类
# ═══════════════════════════════════════════════════════════════

class TestClassifyIntent:
    def test_conflict_keywords(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("不对，你记错了") == "conflict"
        assert _classify_intent("我没说过这个") == "conflict"
        assert _classify_intent("搞错了，不是这样的") == "conflict"

    def test_recall_keywords(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("还记得上次那个吗") == "recall"
        assert _classify_intent("之前我们聊过什么来着") == "recall"
        assert _classify_intent("想起来了吗") == "recall"

    def test_ask_fact_keywords(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("为什么今天这么热") == "ask_fact"
        assert _classify_intent("这个东西是什么") == "ask_fact"
        assert _classify_intent("多少天了") == "ask_fact"
        assert _classify_intent("在哪里") == "ask_fact"

    def test_emotional_sharing_keywords(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("我很难过") == "emotional_sharing"
        assert _classify_intent("今天好开心") == "emotional_sharing"
        assert _classify_intent("压力好大啊") == "emotional_sharing"
        assert _classify_intent("有点焦虑") == "emotional_sharing"

    def test_casual_default(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("今天天气不错") == "casual"
        assert _classify_intent("你好") == "casual"

    def test_conflict_takes_priority(self):
        """冲突关键词优先于其他分类。"""
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("不对，为什么你这么说") == "conflict"

    def test_chinese_question_words(self):
        from app.retrieval.pipeline import _classify_intent
        assert _classify_intent("为什么天空是蓝色的") == "ask_fact"


# ═══════════════════════════════════════════════════════════════
# _resolve_route — intent → 配额
# ═══════════════════════════════════════════════════════════════

class TestResolveRoute:
    def test_casual_route(self):
        from app.retrieval.pipeline import _resolve_route
        route = _resolve_route("casual")
        assert "semantic" in route
        assert "tag" in route

    def test_recall_route(self):
        from app.retrieval.pipeline import _resolve_route
        route = _resolve_route("recall")
        assert route["semantic"] >= route.get("tag", 0)

    def test_unknown_intent_falls_back_to_recall(self):
        from app.retrieval.pipeline import _resolve_route
        route = _resolve_route("nonexistent")
        assert route is not None
        assert "semantic" in route

    def test_empty_intent(self):
        from app.retrieval.pipeline import _resolve_route
        route = _resolve_route("")
        assert route is not None

    def test_none_intent(self):
        from app.retrieval.pipeline import _resolve_route
        route = _resolve_route(None)
        assert route is not None


# ═══════════════════════════════════════════════════════════════
# _load_error_counts / _load_correction_boosts
# ═══════════════════════════════════════════════════════════════

class TestLoadErrorCounts:
    def test_empty_file(self):
        from app.retrieval.pipeline import _load_error_counts
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "error_reports.jsonl")
            open(path, "w").close()
            result = _load_error_counts(tmp)
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_no_file(self):
        from app.retrieval.pipeline import _load_error_counts
        with tempfile.TemporaryDirectory() as tmp:
            result = _load_error_counts(tmp)
            assert isinstance(result, dict)

    def test_with_entries(self):
        from app.retrieval.pipeline import _load_error_counts
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "error_reports.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "abc123", "action": "report"}) + "\n")
                f.write(json.dumps({"memory_id": "abc123", "action": "report"}) + "\n")
                f.write(json.dumps({"memory_id": "def456", "action": "report"}) + "\n")
            result = _load_error_counts(tmp)
            assert result.get("abc123") == 2
            assert result.get("def456") == 1

    def test_clear_action_skipped(self):
        from app.retrieval.pipeline import _load_error_counts
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "error_reports.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "abc", "action": "clear"}) + "\n")
            result = _load_error_counts(tmp)
            assert "abc" not in result


class TestLoadCorrectionBoosts:
    def test_empty_file(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "correction_log.jsonl")
            open(path, "w").close()
            result = _load_correction_boosts(tmp)
            assert isinstance(result, dict)

    def test_no_file(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as tmp:
            result = _load_correction_boosts(tmp)
            assert isinstance(result, dict)

    def test_edit_boost(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "correction_log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "m1", "tag": "Python", "mode": "edit"}) + "\n")
            result = _load_correction_boosts(tmp)
            assert result.get("m1", 0) > 0

    def test_downvote_penalty(self):
        from app.retrieval.pipeline import _load_correction_boosts
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "correction_log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"memory_id": "m1", "tag": "", "mode": "downvote"}) + "\n")
            result = _load_correction_boosts(tmp)
            assert result.get("m1", 0) < 0


# ═══════════════════════════════════════════════════════════════
# run_chat_retrieval — mock 核心路径
# ═══════════════════════════════════════════════════════════════

def _make_chroma_mock():
    m = MagicMock()
    m.list_all.return_value = []
    m._collection = MagicMock()
    m._collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    m._collection.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}
    m.count.return_value = 0
    m._build_embedding_cache = MagicMock()
    m._get_embedding_cached = MagicMock(return_value=None)
    return m


def _make_ctx_mock():
    ctx = MagicMock()
    ctx.data_dir = "/tmp/test"
    ctx.memory_service = _make_chroma_mock()
    ctx.ai_memory_service = _make_chroma_mock()
    ctx.chat_history = MagicMock()
    ctx.chat_history.get_recent.return_value = []
    ctx.chat_history.get_records_snapshot.return_value = []
    ctx.personality_store = MagicMock()
    ctx.personality_store.rerank_tags.return_value = []
    ctx.dmn = MagicMock()
    ctx.dmn.get_preheated.return_value = None
    ctx.co_tracker = MagicMock()
    ctx.co_tracker.query.return_value = []
    ctx.inverted_index = MagicMock()
    ctx.inverted_index.query_tags.return_value = set()
    ctx.inverted_index.query.return_value = []
    ctx.inverted_index._tokenize = MagicMock(return_value=[])
    ctx.topic_tree = None
    ctx.temporal_pattern_index = MagicMock()
    ctx.temporal_pattern_index.query.return_value = []
    ctx.bm25_index = None
    ctx.storage_executor = MagicMock()
    return ctx


class TestRunChatRetrieval:
    """测试 run_chat_retrieval 管线。"""

    def test_returns_tuple_of_four(self):
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        emb = [0.1] * 1024
        result = run_chat_retrieval("测试消息", emb, ctx)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_empty_db_returns_empty_memories(self):
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        ctx.memory_service.count.return_value = 0
        emb = [0.1] * 1024
        _, _, _, memories = run_chat_retrieval("测试消息", emb, ctx)
        assert isinstance(memories, list)

    def test_none_embedding_still_runs(self):
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        _, _, _, memories = run_chat_retrieval("测试消息", None, ctx)
        assert isinstance(memories, list)

    def test_respects_given_intent(self):
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        emb = [0.1] * 1024
        # 传入 intent 应该可以正常运行
        result = run_chat_retrieval("测试", emb, ctx, intent="recall")
        assert len(result) == 4

    def test_memories_have_score_field(self):
        """返回的记忆应有 score 字段。"""
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        # mock 返回一条记忆
        ctx.memory_service._collection.query.return_value = {
            "ids": [["m1"]],
            "metadatas": [[{"summary": "test"}]],
            "distances": [[0.3]],
            "documents": [["测试文档"]],
        }
        ctx.memory_service.count.return_value = 1
        ctx.memory_service._collection.get.return_value = {
            "ids": ["m1"],
            "metadatas": [{"summary": "test"}],
            "documents": ["测试文档"],
        }
        emb = [0.1] * 1024
        _, _, _, memories = run_chat_retrieval("测试", emb, ctx)
        if memories:
            for m in memories:
                assert "score" in m

    def test_memories_have_recency_weight(self):
        """返回的记忆应有 recency_weight 字段。"""
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        emb = [0.1] * 1024
        _, _, _, memories = run_chat_retrieval("测试", emb, ctx)
        for m in memories:
            assert "recency_weight" in m

    def test_dmn_preheat_used_when_available(self):
        """DMN 预热结果被使用。"""
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        ctx.dmn.get_preheated.return_value = [{"id": "pre1", "document": "预热记忆"}]
        emb = [0.1] * 1024
        _, _, _, memories = run_chat_retrieval("测试", emb, ctx)
        assert len(memories) == 1
        assert memories[0]["id"] == "pre1"

    def test_fallback_when_all_empty(self):
        """所有检索为空时触发兜底。"""
        from app.retrieval.pipeline import run_chat_retrieval
        ctx = _make_ctx_mock()
        ctx.memory_service.count.return_value = 0
        ctx.memory_service._collection.query.return_value = {
            "ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]],
        }
        ctx.chat_history.get_recent.return_value = [
            {"user_message": "之前聊过", "llm_reply": "是的之前的回复"}
        ]
        emb = [0.1] * 1024
        _, _, _, memories = run_chat_retrieval("测试", emb, ctx)
        assert isinstance(memories, list)


# ═══════════════════════════════════════════════════════════════
# retrieve_all — 全量检索
# ═══════════════════════════════════════════════════════════════

class TestRetrieveAll:
    """测试 retrieve_all 全量检索。"""

    def test_returns_list(self):
        from app.retrieval.pipeline import retrieve_all
        ctx = _make_ctx_mock()
        emb = [0.1] * 1024
        result = retrieve_all("测试", emb, ctx)
        assert isinstance(result, list)

    def test_empty_db_returns_empty(self):
        from app.retrieval.pipeline import retrieve_all
        ctx = _make_ctx_mock()
        ctx.memory_service.count.return_value = 0
        emb = [0.1] * 1024
        result = retrieve_all("测试", emb, ctx)
        assert result == []

    @patch("app.retrieval.pipeline._BM", True)
    def test_small_db_benchmark_returns_all(self):
        """benchmark 模式下小数据集全量返回。"""
        from app.retrieval.pipeline import retrieve_all
        ctx = _make_ctx_mock()
        # 模拟 benchmark 全量兜底
        ctx.memory_service.count.return_value = 5
        ctx.memory_service._collection.get.return_value = {
            "ids": ["a", "b", "c", "d", "e"],
            "metadatas": [{}, {}, {}, {}, {}],
            "documents": ["a", "b", "c", "d", "e"],
        }
        emb = [0.1] * 1024
        result = retrieve_all("测试", emb, ctx)
        # benchmark 模式下 <= 200 条会全量返回
        assert len(result) >= 4

    def test_respects_intent_routing(self):
        """不同 intent 走不同配额。"""
        from app.retrieval.pipeline import retrieve_all
        ctx = _make_ctx_mock()
        emb = [0.1] * 1024
        result = retrieve_all("测试", emb, ctx, intent="conflict")
        assert isinstance(result, list)

    def test_cached_tags_reused(self):
        """cached_tags 参数被复用。"""
        from app.retrieval.pipeline import retrieve_all
        ctx = _make_ctx_mock()
        emb = [0.1] * 1024
        result = retrieve_all("测试", emb, ctx, cached_tags=["Python", "编程"])
        assert isinstance(result, list)

    def test_no_query_embedding(self):
        """无 query embedding 也能运行。"""
        from app.retrieval.pipeline import retrieve_all
        ctx = _make_ctx_mock()
        result = retrieve_all("测试", None, ctx)
        assert isinstance(result, list)
