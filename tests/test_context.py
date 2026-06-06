"""context.py 测试 — AppContext 主编排器：队列、预热、生命周期。"""
import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_chroma_mock():
    m = MagicMock()
    m.list_all.return_value = []
    m._collection = MagicMock()
    m._collection.get.return_value = {"ids": [], "metadatas": []}
    m.count.return_value = 42
    m._build_embedding_cache = MagicMock()
    return m


def _make_mocks():
    """返回所有 mock 的字典。调用方负责 stop。"""
    patches = {}

    # 模块级服务
    for target, factory in [
        ('app.core.context.ChromaService', _make_chroma_mock),
        ('app.core.context.LLMClient', MagicMock),
        ('app.core.context.CoOccurrenceTracker', MagicMock),
        ('app.core.context.EntityPairTracker', MagicMock),
        ('app.core.context.BehaviorStore', MagicMock),
        ('app.core.context.InvertedIndex', lambda: MagicMock(_tag_index={})),
        ('app.core.context.TopicAffinity', MagicMock),
        ('app.core.context.TemporalPatternIndex', MagicMock),
        ('app.core.context.DistillEngine', MagicMock),
        ('app.core.context.BehaviorPredictor', MagicMock),
        ('app.background.consolidation.ConsolidationEngine', MagicMock),
        ('app.background.impulse.ImpulseScheduler', MagicMock),
    ]:
        p = patch(target)
        mock = p.start()
        mock.return_value = factory() if callable(factory) else factory()
        patches[target] = p

    # PersonalityStore — 特殊配置
    p = patch('app.core.context.PersonalityStore')
    mock = p.start()
    ps = MagicMock()
    ps.list_tags.return_value = {"items": []}
    mock.return_value = ps
    patches['app.core.context.PersonalityStore'] = p

    # ChatHistory — 特殊配置
    p = patch('app.core.context.ChatHistory')
    mock = p.start()
    ch = MagicMock()
    ch.records = []
    ch.get_recent.return_value = []
    ch.get_records_snapshot.return_value = []
    mock.return_value = ch
    patches['app.core.context.ChatHistory'] = p

    # PatternDiscovery — 特殊配置
    p = patch('app.core.context.PatternDiscovery')
    mock = p.start()
    pd_mock = MagicMock()
    pd_mock.load_cache = MagicMock()
    mock.return_value = pd_mock
    patches['app.core.context.PatternDiscovery'] = p

    # 常量
    for const in ['IS_LITE', 'LITE_DISABLE_BACKGROUND_TASKS', 'LITE_DISABLE_IMPULSE']:
        p = patch(f'app.core.context.{const}', False)
        p.start()
        patches[const] = p

    # 后台线程 — 全部 no-op
    bg_methods = ['_start_impulse_consumer', '_start_queue_worker',
                  '_start_dmn_worker', '_start_consolidation_worker',
                  '_start_ai_consolidation_worker', '_start_impulse_workers']
    for method in bg_methods:
        p = patch(f'app.core.context.AppContext.{method}', return_value=None)
        p.start()
        patches[f'bg_{method}'] = p

    return patches


# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def ctx():
    """构造一个 mock 过的 AppContext，后台线程全部禁用。"""
    import tempfile
    import time

    patches = _make_mocks()
    try:
        from app.core.context import AppContext
        with tempfile.TemporaryDirectory() as tmp:
            c = AppContext(data_dir=tmp)
            # 后台线程方法被 mock 了，手动设好 close() 需要的属性（用 MagicMock 避免 join 报错）
            for attr in ['_queue_thread', '_ai_consolidation_thread',
                          '_impulse_consumer_thread', '_dmn_thread', '_consolidation_thread']:
                if not hasattr(c, attr):
                    setattr(c, attr, MagicMock())
            c._stop_event.set()
            time.sleep(0.02)
            yield c
    finally:
        for p in patches.values():
            p.stop()


# ═══════════════════════════════════════════════════════════════
# _extract_noun_tags
# ═══════════════════════════════════════════════════════════════

class TestExtractNounTags:
    def test_extracts_from_chinese(self):
        from app.core.context import _extract_noun_tags
        tags = _extract_noun_tags("今天写了一个Python爬虫脚本")
        assert isinstance(tags, list)
        assert len(tags) >= 1

    def test_empty_text(self):
        from app.core.context import _extract_noun_tags
        assert _extract_noun_tags("") == []

    def test_topk_respected(self):
        from app.core.context import _extract_noun_tags
        tags = _extract_noun_tags("测试文本用于验证标签数量限制的功能", topk=3)
        assert len(tags) <= 3


# ═══════════════════════════════════════════════════════════════
# _get_local_llm
# ═══════════════════════════════════════════════════════════════

class TestGetLocalLLM:
    def test_returns_same_instance(self):
        from app.core.context import _get_local_llm
        a = _get_local_llm()
        b = _get_local_llm()
        assert a is b


# ═══════════════════════════════════════════════════════════════
# 构造
# ═══════════════════════════════════════════════════════════════

class TestAppContextConstruction:
    def test_constructs_without_crash(self, ctx):
        assert ctx.data_dir is not None
        assert ctx.chroma_service is not None
        assert ctx.personality_store is not None

    def test_topic_tree_property(self, ctx):
        tree = ctx.topic_tree
        assert tree is None or hasattr(tree, 'expand')

    def test_close_releases_resources(self, ctx):
        ctx.close()
        ctx.close()  # 重复不崩溃


# ═══════════════════════════════════════════════════════════════
# _enqueue_store_task
# ═══════════════════════════════════════════════════════════════

class TestEnqueueStoreTask:
    def test_enqueue_writes_to_queue(self, ctx):
        ctx._enqueue_store_task("用户", "AI", "2026-06-06 22:00:00")
        assert ctx._store_queue.qsize() >= 1

    def test_enqueue_writes_to_file(self, ctx):
        ctx._enqueue_store_task("用户", "AI", "2026-06-06 22:00:00")
        assert os.path.exists(ctx._store_queue_path)
        with open(ctx._store_queue_path, encoding="utf-8") as f:
            assert "用户" in f.read()

    def test_enqueue_multiple_tasks(self, ctx):
        for i in range(5):
            ctx._enqueue_store_task(f"m{i}", f"r{i}", f"2026-06-06 22:0{i}:00")
        assert ctx._store_queue.qsize() >= 5

    def test_enqueue_task_structure(self, ctx):
        ctx._enqueue_store_task("U", "A", "2026-06-06 22:00:00")
        task = ctx._store_queue.get(timeout=1)
        assert task["user_message"] == "U"
        assert task["ai_message"] == "A"

    def test_enqueue_valid_json_in_file(self, ctx):
        ctx._enqueue_store_task("测试", "回复", "2026-06-06 22:00:00")
        with open(ctx._store_queue_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    assert "user_message" in d


# ═══════════════════════════════════════════════════════════════
# _cleanup_executors
# ═══════════════════════════════════════════════════════════════

class TestCleanupExecutors:
    def test_cleanup_handles_missing_executor(self, ctx):
        if hasattr(ctx, 'retrieval_executor'):
            del ctx.retrieval_executor
        ctx._cleanup_executors()

    def test_cleanup_with_executors(self, ctx):
        ctx._cleanup_executors()

    def test_close_twice(self, ctx):
        ctx.close()
        ctx.close()


# ═══════════════════════════════════════════════════════════════
# _prewarm_retrieval
# ═══════════════════════════════════════════════════════════════

class TestPrewarmRetrieval:
    def test_prewarm_calls_build_cache(self, ctx):
        ctx._prewarm_retrieval()
        ctx.chroma_service._build_embedding_cache.assert_called()

    def test_prewarm_handles_exception(self, ctx):
        ctx.chroma_service.count.side_effect = RuntimeError("fail")
        ctx._prewarm_retrieval()  # 不抛异常


# ═══════════════════════════════════════════════════════════════
# _record_ai_co_occurrence
# ═══════════════════════════════════════════════════════════════

class TestRecordAiCoOccurrence:
    def test_records_with_enough_ids(self, ctx):
        ctx.ai_chroma_service._collection.get.return_value = {
            "ids": ["a1", "a2", "a3"], "metadatas": [{}, {}, {}],
        }
        ctx._record_ai_co_occurrence()
        ctx.ai_co_tracker.record.assert_called()

    def test_skips_with_one_id(self, ctx):
        ctx.ai_chroma_service._collection.get.return_value = {
            "ids": ["a1"], "metadatas": [{}],
        }
        ctx._record_ai_co_occurrence()
        ctx.ai_co_tracker.record.assert_not_called()

    def test_handles_empty(self, ctx):
        ctx.ai_chroma_service._collection.get.return_value = {
            "ids": [], "metadatas": [],
        }
        ctx._record_ai_co_occurrence()  # 不抛异常


# ═══════════════════════════════════════════════════════════════
# 线程安全
# ═══════════════════════════════════════════════════════════════

class TestThreadSafety:
    def test_concurrent_enqueue(self, ctx):
        errors = []

        def enqueue_batch(start, cnt):
            try:
                for i in range(start, start + cnt):
                    ctx._enqueue_store_task(f"u{i}", f"a{i}", f"2026-06-06 22:{i%60:02d}:00")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=enqueue_batch, args=(i * 25, 25))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发异常: {errors}"
        assert ctx._store_queue.qsize() >= 100

    def test_close_during_enqueue(self, ctx):
        errors = []

        def delayed():
            try:
                ctx._enqueue_store_task("延迟", "AI", "2026-06-06 22:00:00")
            except Exception as e:
                errors.append(str(e))

        t = threading.Thread(target=delayed)
        t.start()
        t.join(timeout=2)
        ctx.close()
        assert len(errors) == 0, f"异常: {errors}"
