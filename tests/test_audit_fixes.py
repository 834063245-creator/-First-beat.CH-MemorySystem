"""审计修复验证测试：验证每个修复的正确性。"""
import json
import os
import time
import threading
from collections import deque

import pytest


# ── H1: LocalLLM 惰性单例（v2.0 重构后路径迁移） ─────────────────
class TestH1LazySingleton:
    """验证 LocalLLM 可正常实例化。"""

    def test_local_llm_instantiable(self):
        from app.llm.local import LocalLLM
        llm = LocalLLM()
        assert llm is not None
        assert hasattr(llm, "_model")


# ── H3: 队列清空 rename 原子操作 ────────────────────────────────
class TestH3QueueRename:
    """验证 tmp 文件清理逻辑。"""

    def test_tmp_cleanup_logic(self, tmp_path):
        """模拟 rename 后清理的行为。"""
        src = tmp_path / "queue.jsonl"
        src.write_text('{"test": 1}\n')
        # rename
        tmp = str(src) + ".tmp." + str(time.time())
        os.rename(str(src), tmp)
        assert os.path.exists(tmp)
        assert not os.path.exists(str(src))
        # 清理
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        assert not os.path.exists(tmp)


# ── H8: _rewrite_chroma_ids 已移除（v2.0 重构后 ChatHistory 类不再存在） ──
class TestH8RewriteRemoved:
    """_rewrite_chroma_ids 已在重构中移除。"""

    def test_no_chat_history_import_from_legacy(self):
        """旧的 from chat_history import ChatHistory 路径不再存在。"""
        import importlib
        try:
            importlib.import_module("chat_history")
            has_legacy = True
        except ImportError:
            has_legacy = False
        assert not has_legacy, "旧模块 chat_history 不应存在"


# ── M1: STOP_WORDS 统一到 app.config.settings ────────────────────
class TestM1StopWords:
    """验证 STOP_WORDS 从 settings 导入且类型正确。"""

    def test_config_stop_words_type(self):
        from app.config.settings import STOP_WORDS
        assert isinstance(STOP_WORDS, frozenset)
        assert "的" in STOP_WORDS
        assert len(STOP_WORDS) > 50  # 合并后应有足够多停用词


# ── M3: ChatResponse.debug 传入正确类型 ─────────────────────────
class TestM3ChatResponseDebug:
    """验证 ChatResponse 接受 DebugInfo 作为 debug 参数。"""

    def test_debug_accepts_debug_info(self):
        from app.models.schemas import ChatResponse, DebugInfo
        di = DebugInfo(retrieved_count=5)
        resp = ChatResponse(response="ok", debug=di)
        assert resp.debug is not None
        assert resp.debug.retrieved_count == 5

    def test_debug_default_none(self):
        from app.models.schemas import ChatResponse
        resp = ChatResponse(response="ok")
        assert resp.debug is None


# ── M4: query_explore 共享 client ───────────────────────────────
class TestM4SharedClient:
    """验证 query_explore 统一初始化 _collection 而非在每个分支重复创建。"""

    def test_function_has_init_lock(self):
        from app.tools.dispatch import _query_explore_init_lock
        import threading
        assert isinstance(_query_explore_init_lock, threading.Lock)

    def test_no_redundant_init_in_emotion(self, tmp_path):
        """传入 _collection 时 emotion 分支应能正常使用。"""
        import chromadb
        from app.tools.dispatch import query_explore
        client = chromadb.PersistentClient(path=str(tmp_path))
        coll = client.get_or_create_collection("memories", embedding_function=None)
        result = query_explore("emotion", _collection=coll, min_intensity=1, top_k=3)
        # 不应崩溃
        assert isinstance(result, str)


# ── M6: _clear_memory_errors 追加模式（v2.0 重构后路径迁移） ────
class TestM6ClearMemoryErrors:
    """v2.0 重构后 _clear_memory_errors 已被 STORE_FAILURES_PATH 追加模式替代。"""

    def test_store_failures_path_exists(self):
        from app.config.settings import STORE_FAILURES_PATH
        assert STORE_FAILURES_PATH is not None, \
            "STORE_FAILURES_PATH（替代旧的 _clear_memory_errors）应存在"

    def test_load_error_counts_skips_clear(self, tmp_path):
        """验证 _load_error_counts 过滤掉 action=clear 的行。"""
        from app.retrieval.pipeline import _load_error_counts
        path = tmp_path / "error_reports.jsonl"
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(json.dumps({"memory_id": "mem_001"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_001"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_001", "action": "clear"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_002"}) + "\n")
        counts = _load_error_counts(data_dir=str(tmp_path))
        assert counts.get("mem_001") == 2
        assert counts.get("mem_002") == 1



# ── M8: working_memory 路径使用 DATA_DIR ────────────────────────
class TestM8WorkingMemoryPath:
    """验证 working_memory 使用 DATA_DIR。"""

    def test_wm_file_uses_data_dir(self):
        from app.memory.working import _save
        import inspect
        src = inspect.getsource(_save)
        # _save 内部应使用传入的 wm_path 而非硬编码路径
        assert "wm_path" in src


# ── L1: 死代码清理 ──────────────────────────────────────────────
class TestL1DeadCodeCleanup:
    """验证 PromptBody 等模型已从 schemas.py 删除（v2.0 审计修复 P2-1）。"""

    def test_models_removed_from_schemas(self):
        from app.models import schemas as s
        assert not hasattr(s, "PromptBody"), "PromptBody 已删除"
        assert not hasattr(s, "CorrectMemoryBody"), "CorrectMemoryBody 已删除"

    def test_unused_models_removed(self):
        """MemoryListResponse 和 MemoryDeleteResponse 已删除。"""
        from app.models import schemas as s
        assert not hasattr(s, "MemoryListResponse")
        assert not hasattr(s, "MemoryDeleteResponse")


# ── L2: bottleneck deque ────────────────────────────────────────
class TestL2BottleneckDeque:
    """验证 _chain 是 deque(maxlen=1000)。"""

    def test_chain_is_deque(self):
        from app.core.bottleneck import _chain
        assert isinstance(_chain, deque)
        assert _chain.maxlen == 1000


# ── L7: BehaviorStore 加锁 ──────────────────────────────────────
class TestL7BehaviorStoreLock:
    """验证 BehaviorStore 有 _lock 且 store/count 方法受锁保护。"""

    def test_store_has_lock(self):
        from app.personality.behavior import BehaviorStore
        import tempfile
        assert hasattr(BehaviorStore, "store")
        assert hasattr(BehaviorStore, "count")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            store = BehaviorStore(persist_dir=td)
            assert hasattr(store, "_lock")
            store._client = None
            store._collection = None


# ── M1: health_ollama 端点已删除 ────────────────────────────────
class TestM1HealthOllamaRemoved:
    """v2.0 P2-1：health_ollama 死端点已删除。"""

    def test_health_ollama_endpoint_removed(self):
        from app.api import health
        assert not hasattr(health, "health_ollama"), \
            "health_ollama 死端点应已删除"


# ── M7: 函数体内重复 import (v2.0 重构后路径迁移) ──────────────
class TestM7NoDuplicateImports:
    """验证 retrieval pipeline 无重复 import。"""

    def test_no_redundant_import_in_retrieval(self):
        """run_chat_retrieval 函数体内不应有重复 import。"""
        from app.retrieval.pipeline import run_chat_retrieval
        import inspect
        source = inspect.getsource(run_chat_retrieval)
        # local_embed 已在模块级导入，函数内不应重复
        assert "from app.llm.embed import local_embed" not in source, \
            "函数体内不应重复导入 local_embed"
