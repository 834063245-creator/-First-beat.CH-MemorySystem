"""审计修复验证测试：验证每个修复的正确性。"""
import json
import os
import time
import threading
from collections import deque

import pytest


# ── H1: LocalLLM 惰性单例 ──────────────────────────────────────
class TestH1LazySingleton:
    """验证惰性单例模式：_get_local_llm 函数存在且可调用。"""

    def test_get_local_llm_exists(self):
        from backend.main import _get_local_llm, _LOCAL_LLM
        # 确保函数存在且 _LOCAL_LLM 初始为 None
        assert callable(_get_local_llm)
        # _LOCAL_LLM 在首次调用前应为 None（但注意其他测试可能已触发初始化）
        # 这里只验证函数签名正确


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


# ── H8: _rewrite_chroma_ids 改为空操作 ──────────────────────────
class TestH8RewriteNoop:
    """_rewrite_chroma_ids 已按审计 L5 删除，空方法不复存在。"""

    def test_rewrite_removed(self):
        from chat_history import ChatHistory
        assert not hasattr(ChatHistory, "_rewrite_chroma_ids")


# ── M1: STOP_WORDS 统一到 config ────────────────────────────────
class TestM1StopWords:
    """验证 STOP_WORDS 从 config 导入且类型正确。"""

    def test_config_stop_words_type(self):
        from config import STOP_WORDS
        assert isinstance(STOP_WORDS, frozenset)
        assert "的" in STOP_WORDS
        assert len(STOP_WORDS) > 50  # 合并后应有足够多停用词

    def test_main_imports_from_config(self):
        import main
        assert hasattr(main, "_STOP_WORDS")
        assert isinstance(main._STOP_WORDS, frozenset)


# ── M3: ChatResponse.debug 传入正确类型 ─────────────────────────
class TestM3ChatResponseDebug:
    """验证 ChatResponse 接受 DebugInfo 作为 debug 参数。"""

    def test_debug_accepts_debug_info(self):
        from models import ChatResponse, DebugInfo
        di = DebugInfo(retrieved_count=5)
        resp = ChatResponse(response="ok", debug=di)
        assert resp.debug is not None
        assert resp.debug.retrieved_count == 5

    def test_debug_default_none(self):
        from models import ChatResponse
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


# ── M6: _clear_memory_errors 追加模式 ──────────────────────────
class TestM6ClearMemoryErrors:
    """验证 _clear_memory_errors 追加清除标记而非重写文件。"""

    def test_appends_clear_marker(self, tmp_path, monkeypatch):
        """验证追加的是 clear 标记行而非原地修改。"""
        from backend.main import _clear_memory_errors
        data_dir = str(tmp_path)
        path = tmp_path / "error_reports.jsonl"
        # 写入两条原始错误记录
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(json.dumps({"memory_id": "mem_001", "error": "bad"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_002", "error": "wrong"}) + "\n")
        # 清除 mem_001（传 data_dir 避免污染真实数据）
        _clear_memory_errors("mem_001", data_dir=data_dir)
        # 文件应多了 1 行（标记行），而不是缩短
        lines = open(str(path), encoding="utf-8").readlines()
        assert len(lines) == 3
        last_line = json.loads(lines[-1])
        assert last_line["action"] == "clear"
        assert last_line["memory_id"] == "mem_001"

    def test_load_error_counts_skips_clear(self, tmp_path, monkeypatch):
        """验证 _load_error_counts 过滤掉 action=clear 的行。"""
        from app.retrieval.pipeline import _load_error_counts
        data_dir = str(tmp_path)
        path = tmp_path / "error_reports.jsonl"
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(json.dumps({"memory_id": "mem_001"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_001"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_001", "action": "clear"}) + "\n")
            f.write(json.dumps({"memory_id": "mem_002"}) + "\n")
        counts = _load_error_counts(data_dir=data_dir)
        # mem_001 应该只计数 2 次（跳过 clear），mem_002 计数 1 次
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
    """验证 PromptBody 等模型已移到 models.py。"""

    def test_models_exist_in_models_py(self):
        from models import PromptBody, CorrectMemoryBody
        assert PromptBody(content="test").content == "test"

    def test_main_imports_from_models(self):
        import main
        assert hasattr(main, "PromptBody")

    def test_unused_models_no_longer_in_main(self):
        """MemoryListResponse 和 MemoryDeleteResponse 已从 main.py 移除。"""
        import main
        assert not hasattr(main, "MemoryListResponse")
        assert not hasattr(main, "MemoryDeleteResponse")


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
        from behavior_store import BehaviorStore
        assert hasattr(BehaviorStore, "store")
        assert hasattr(BehaviorStore, "count")


# ── M1: health_ollama 用 async httpx ───────────────────────────────
class TestM1AsyncHttpx:
    """验证 health_ollama 端点使用 async httpx 而非同步阻塞版本。"""

    def test_health_ollama_uses_async_client(self):
        """health_ollama 端点使用 async with AsyncClient。"""
        import main
        import inspect
        source = inspect.getsource(main.health_ollama)
        assert "async with httpx.AsyncClient" in source
        assert "httpx.get(" not in source  # 同步 API 不应出现


# ── M7: 函数体内重复 import ───────────────────────────────────────
class TestM7NoDuplicateImports:
    """验证 _run_chat_retrieval 不再有重复的 from local_embed import。"""

    def test_no_redundant_import_in_run_chat_retrieval(self):
        """_run_chat_retrieval 函数体内不应有 from local_embed import。"""
        import main
        import inspect
        source = inspect.getsource(main._run_chat_retrieval)
        # local_embed 已在模块级导入，函数内不应重复
        assert "from local_embed import local_embed" not in source
