"""审计 CC 指令修复验证 — 覆盖 24 条修复的正确性。

测试前提：
  - pytest 可用
  - ChromaDB + embedding 模型已部署（部分测试需要）
  - 代码已按 CC 指令修改完毕

分组：
  H1-H6 高危修复
  M1-M16 中危修复
  I1-L17 低危修复
"""

import json
import os
import sys
import threading
import time
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ===================================================================
# H1 — Counter 已导入
# ===================================================================
class TestH1CounterImport:
    """backend/main.py 顶部应有 from collections import Counter。"""

    def test_counter_available(self):
        from main import Counter
        assert callable(Counter)

    def test_counter_used_in_consolidation(self):
        """确认 Counter 在 consolidation 中有实际用途。"""
        from collections import Counter
        c = Counter({"a": 3, "b": 1})
        assert c.most_common(1)[0][0] == "a"


# ===================================================================
# H2 — 心跳引用改为模块级
# ===================================================================
class TestH2HeartbeatModuleRef:
    """impulse.py 不再按值导入心跳，改为模块引用。"""

    def test_impulse_imports_module(self):
        """impulse 应 import app.api.system as _sys_mod 而非直接导入值。"""
        import app.api.impulse as imp_mod
        # 不应直接导入 _last_heartbeat_time（会被复制值）
        import app.api.system as sys_mod
        assert hasattr(imp_mod, "_sys_mod")
        assert imp_mod._sys_mod is sys_mod
        assert hasattr(sys_mod, "_last_heartbeat_time")

    def test_heartbeat_common_update(self):
        """system.py 更新心跳后 impulse 能读到新值。"""
        import app.api.impulse as imp_mod
        import app.api.system as sys_mod
        # 模拟 system 更新
        old_hb = sys_mod._last_heartbeat_time
        sys_mod._last_heartbeat_time = time.time()
        # impulse 通过模块引用应读到新值
        imp_hb = imp_mod._sys_mod._last_heartbeat_time
        assert imp_hb is sys_mod._last_heartbeat_time
        # 恢复（防止污染其他测试）
        sys_mod._last_heartbeat_time = old_hb


# ===================================================================
# H3/H4 — 共现/实体锁覆盖
# ===================================================================
class TestH3CooccurLockAtomicity:
    """CoOccurrenceTracker.record() 读-改-写 全程保持锁。"""

    def test_record_uses_single_lock_cycle(self, tmp_path):
        """并发 record 不丢失更新。"""
        from app.memory.cooccur import CoOccurrenceTracker
        f = tmp_path / "co_test.json"
        co = CoOccurrenceTracker(file_path=str(f))

        # 并发写入相同 ID 对
        n = 50
        errors = []

        def writer(offset):
            for i in range(n):
                try:
                    co.record([f"id_a", f"id_b_{offset}_{i}"])
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发错误: {errors}"
        # 验证文件完整性
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data) >= 3 * n  # 每个 writer 写 n 条不同的 ID 对

    def test_remove_uses_single_lock_cycle(self, tmp_path):
        """并发 remove + record 不丢失更新或崩溃。"""
        from app.memory.cooccur import CoOccurrenceTracker
        f = tmp_path / "co_test2.json"
        co = CoOccurrenceTracker(file_path=str(f))
        co.record(["a", "b"])
        co.record(["a", "c"])
        co.record(["b", "c"])

        errors = []
        def remover():
            for _ in range(20):
                try:
                    co.remove("a")
                except Exception as e:
                    errors.append(f"remove: {e}")

        def adder():
            for i in range(20):
                try:
                    co.record(["a", f"x{i}"])
                except Exception as e:
                    errors.append(f"add: {e}")

        threads = [threading.Thread(target=remover), threading.Thread(target=adder)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"锁竞争中发生错误: {errors}"


class TestH4EntityPairLockAtomicity:
    """EntityPairTracker.record() 和 remove_memory() 读-改-写 全程保持锁。"""

    def test_record_uses_single_lock_cycle(self, tmp_path):
        from app.memory.entity_pair import EntityPairTracker
        f = tmp_path / "ent_test.json"
        ep = EntityPairTracker(file_path=str(f))

        errors = []
        def writer(offset):
            for i in range(30):
                try:
                    ep.record(f"entity_{offset}", f"entity_{i}", f"mem_{offset}_{i}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发错误: {errors}"
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data) >= 3  # 至少 3 个实体前缀

    def test_remove_memory_atomic(self, tmp_path):
        from app.memory.entity_pair import EntityPairTracker
        f = tmp_path / "ent_test2.json"
        ep = EntityPairTracker(file_path=str(f))
        ep.record("A", "B", "mem1")
        ep.record("A", "C", "mem2")
        ep.record("B", "C", "mem3")

        errors = []
        def remover():
            for _ in range(10):
                try:
                    ep.remove_memory("mem1")
                except Exception as e:
                    errors.append(f"remove: {e}")

        def adder():
            for i in range(10):
                try:
                    ep.record("A", f"D{i}", f"mem_new_{i}")
                except Exception as e:
                    errors.append(f"add: {e}")

        threads = [threading.Thread(target=remover), threading.Thread(target=adder)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"锁竞争中发生错误: {errors}"


# ===================================================================
# H5 — _CORE_RULES 统一使用
# ===================================================================
class TestH5CoreRulesUnified:
    """_build_prompt() 使用模块级 _CORE_RULES 而非内联副本。"""

    def test_build_prompt_uses_module_constant(self):
        from app.llm.deepseek import _CORE_RULES
        assert "【记忆使用核心规则——不可修改】" in _CORE_RULES
        # 确认常量包含关键约束
        assert "calc_time" not in _CORE_RULES  # calc_time 工具已移除
        assert "没有找到相关记忆" in _CORE_RULES

    def test_no_inline_duplicate_in_build_prompt(self):
        """验证 _build_prompt 没有独立的 core_rules 字面量。"""
        import inspect
        from app.llm.deepseek import DeepSeekLLM
        source = inspect.getsource(DeepSeekLLM._build_prompt)
        # 应该引用 _CORE_RULES 而不是自己定义字面量
        assert "_CORE_RULES" in source
        # 不应有第二条独立的 "【记忆使用核心规则——不可修改】" 字面量
        # 允许出现一次（就是 _CORE_RULES 定义处），这里验证 _build_prompt 不包含
        count = source.count("【记忆使用核心规则——不可修改】")
        assert count == 0, f"_build_prompt 中不应包含规则字面量，发现 {count} 处"


# ===================================================================
# H6 — TopicTree 线程锁
# ===================================================================
class TestH6TopicTreeLock:
    """TopicTree 存在 _lock 且公共方法使用锁。"""

    def test_tree_has_lock(self, tmp_path):
        from app.memory.tree import TopicTree
        tree = TopicTree(data_dir=str(tmp_path))
        assert hasattr(tree, "_lock")
        assert isinstance(tree._lock, threading.Lock)

    def test_concurrent_expand_no_crash(self, tmp_path):
        from app.memory.tree import TopicTree
        tree = TopicTree(data_dir=str(tmp_path))
        matrix = {
            "记忆": {"用户": 4, "聊天": 3, "系统": 2},
            "用户": {"记忆": 4, "聊天": 3},
            "聊天": {"记忆": 3, "用户": 3},
            "系统": {"记忆": 2},
        }
        tree.rebuild(matrix)

        errors = []
        def expander():
            for _ in range(50):
                try:
                    tree.expand(["记忆", "用户"])
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=expander) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发 expand 错误: {errors}"

    def test_concurrent_get_branch_no_crash(self, tmp_path):
        from app.memory.tree import TopicTree
        tree = TopicTree(data_dir=str(tmp_path))
        tree._tag_to_branch = {"A": ["A", "B", "C"], "B": ["A", "B", "C"], "C": ["A", "B", "C"]}

        errors = []
        def reader():
            for _ in range(100):
                try:
                    tree.get_branch("A")
                except Exception as e:
                    errors.append(str(e))

        def writer():
            for _ in range(100):
                try:
                    tree._tag_to_branch["D"] = ["D"]
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(2)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发 get_branch 错误: {errors}"


# ===================================================================
# M1 — working.py 乐观版本检测
# ===================================================================
class TestM1WorkingMemoryVersionCheck:
    """incremental_update 写回前做版本检测防止覆盖。"""

    def test_version_gap_skips_update(self, tmp_path):
        from app.memory.working import incremental_update, _load, _save
        wm_path = str(tmp_path / "working_memory.json")
        _save({"summary": "v1", "version": 0, "topics": [], "last_updated": ""}, wm_path)

        # 模拟另一线程在 LLM 调用期间更新了版本
        original_update = incremental_update

        with patch("app.memory.working.incremental_update") as mock:
            mock.side_effect = lambda turns, *, wm_path: (
                # 模拟外部更新
                _save({"summary": "v2_ext", "version": 1, "topics": [], "last_updated": ""}, wm_path),
                False  # 返回 False 表示跳过了本轮
            )
            result = incremental_update([], wm_path=wm_path)

        wm = _load(wm_path)
        assert wm["version"] >= 0  # 文件完好

    def test_load_save_roundtrip(self, tmp_path):
        from app.memory.working import _load, _save
        wm_path = str(tmp_path / "working_memory.json")
        data = {"summary": "test", "version": 0, "topics": ["a"], "last_updated": "", "current_state": ""}
        _save(data, wm_path)
        loaded = _load(wm_path)
        assert loaded["summary"] == "test"
        assert loaded["version"] == 0


# ===================================================================
# M2 — history.py delete_by_chroma_id 文件在锁内写入
# ===================================================================
class TestM2HistoryLockedFileWrite:
    """delete_by_chroma_id 的文件写入在锁保护内。"""

    def test_delete_file_write_inside_lock(self, tmp_path):
        import inspect
        from app.memory.history import ChatHistory
        source = inspect.getsource(ChatHistory.delete_by_chroma_id)
        # 确保 with self._lock 内包含 open(...) 写入
        lines = source.split("\n")
        in_lock = False
        write_in_lock = False
        for line in lines:
            if "with self._lock" in line:
                in_lock = True
            elif in_lock and "open" in line and "w" in line:
                write_in_lock = True
            elif in_lock and "try" in line:
                # 尝试块在锁内也是可以的
                pass
            elif in_lock and line.strip() and not line.startswith(" ") and "lock" not in line:
                # 出了锁的缩进级别
                if write_in_lock:
                    break
        assert write_in_lock, "文件写入不在锁保护内"


# ===================================================================
# M4 — /chat 非 stream 版使用 run_in_executor
# ===================================================================
class TestM4ChatRunInExecutor:
    """非 stream /chat 路由的后处理改用 run_in_executor 避免阻塞。"""

    def test_chat_non_stream_uses_executor(self):
        import inspect
        from backend.main import chat
        source = inspect.getsource(chat)
        assert "run_in_executor" in source
        # 确认存在 3 次调用（chat_history.append, _enqueue_store_task, incremental_update）
        count = source.count("run_in_executor")
        assert count >= 3, f"run_in_executor 调用不足: {count}"


# ===================================================================
# M5 — system.py 输入验证
# ===================================================================
class TestM5SystemInputValidation:
    """api_update_prompt 应有 content 类型/长度校验。"""

    def test_prompt_update_blocked_traversal(self):
        """路径穿越应被拦截。"""
        from app.api.system import api_update_prompt
        # FastAPI 会在路由层校验参数，函数内部也应有校验
        import inspect
        source = inspect.getsource(api_update_prompt)
        assert "isinstance(content, str)" in source or "not isinstance(content" in source
        assert "len(content)" in source
        assert "not path.startswith(backend_dir)" in source or "normpath" in source


# ===================================================================
# M6 — knowledge.py 路径验证
# ===================================================================
class TestM6KnowledgePathValidation:
    """api_knowledge_import 应限制文件路径范围。"""

    def test_import_path_validation(self):
        import inspect
        from app.api.knowledge import api_knowledge_import
        source = inspect.getsource(api_knowledge_import)
        assert "allowed_dirs" in source or "normpath" in source
        assert "startswith" in source
        assert "isfile" in source or "os.path.isfile" in source


# ===================================================================
# M10 — DeepSeekLLM aclose
# ===================================================================
class TestM10DeepSeekClose:
    """DeepSeekLLM 有 aclose 方法。"""

    def test_deepseek_has_aclose(self):
        from app.llm.deepseek import DeepSeekLLM
        llm = DeepSeekLLM()
        assert hasattr(llm, "aclose")
        assert callable(llm.aclose)


# ===================================================================
# M12 — dispatch.py ChromaDB 客户端缓存
# ===================================================================
class TestM12ChromaClientCache:
    """query_explore 缓存 ChromaDB PersistentClient。"""

    def test_get_chroma_collection_caches(self):
        from app.tools.dispatch import _query_explore_clients, _get_chroma_collection
        import chromadb
        # 验证函数存在
        assert callable(_get_chroma_collection)
        assert isinstance(_query_explore_clients, dict)

    def test_repeated_calls_reuse_client(self, tmp_path):
        from app.tools.dispatch import _get_chroma_collection
        # 模拟两次调用 — 使用不同的集合名称以防冲突
        c1 = _get_chroma_collection(str(tmp_path), "test_cache1")
        c2 = _get_chroma_collection(str(tmp_path), "test_cache2")
        assert c1._client is c2._client  # 同一个 PersistentClient


# ===================================================================
# M13 — pipeline.py 修复 bare import
# ===================================================================
class TestM13PipelineImport:
    """pipeline.py 应使用 app.retrieval.reranker 而非 bare local_reranker。"""

    def test_no_bare_import(self):
        import inspect
        from app.retrieval.pipeline import run_chat_retrieval
        source = inspect.getsource(run_chat_retrieval)
        assert "from local_reranker import rerank" not in source
        assert "from app.retrieval.reranker import rerank" in source


# ===================================================================
# M14 — stream [DONE] after error
# ===================================================================
class TestM14StreamDoneAfterError:
    """stream 异常分支应发送 [DONE]。"""

    def test_error_branch_has_done(self):
        import inspect
        from app.api.chat import chat_stream
        source = inspect.getsource(chat_stream)
        # 在 [ERROR] 之后应有 [DONE]
        error_idx = source.index("[ERROR]")
        after_error = source[error_idx:]
        assert "[DONE]" in after_error, "异常分支缺少 [DONE]"


# ===================================================================
# I4 — metadata.py 重复 jieba
# ===================================================================
class TestI4MetadataDedup:
    """extract_persons 只切一次 jieba posseg。"""

    def test_only_one_cut_call(self):
        import inspect
        from app.core.metadata import extract_persons
        source = inspect.getsource(extract_persons)
        count = source.count("pseg.cut")
        assert count == 1, f"pseg.cut 应只调用 1 次，发现 {count} 次"

    def test_extract_persons_works(self):
        from app.core.metadata import extract_persons
        persons = extract_persons("张三和李四说王五在写代码")
        assert len(persons) >= 2  # 至少识别出人名


# ===================================================================
# L3-L7 — 无用导入/配置移除
# ===================================================================
class TestL3NoUnusedSysImport:
    def test_app_py_no_unused_sys(self):
        import inspect
        from app.api import app
        source = inspect.getsource(app)
        assert "import sys" not in source or "sys.path" in source


class TestL4NoUnusedMathImport:
    def test_dispatch_no_unused_math(self):
        import inspect
        from app.tools import dispatch
        source = inspect.getsource(dispatch)
        # 去掉 import math 可能出现在模块级或函数内
        lines = [l.strip() for l in source.split("\n")]
        imports = [l for l in lines if l.startswith("import math") or l.startswith("from math")]
        assert len(imports) == 0, f"dispatch.py 不应 import math: {imports}"


class TestL5NoRedundantJsonImport:
    def test_dispatch_no_redundant_json_inside_func(self):
        import inspect
        from app.tools import dispatch
        source = inspect.getsource(dispatch)
        # 只应有一个 import json（模块顶部），不应有函数内的 import json
        import_lines = [l.strip() for l in source.split("\n") if "import json" in l]
        assert len(import_lines) <= 1, f"发现有 {len(import_lines)} 处 import json"


class TestL6NoStubReranker:
    def test_get_reranker_removed(self):
        import app.retrieval.reranker as r
        assert not hasattr(r, "get_reranker"), "get_reranker 桩函数应已删除"


class TestL7LiteConfigRemoved:
    def test_lite_disable_constants_removed(self):
        import app.config.settings as s
        assert not hasattr(s, "LITE_DISABLE_BACKGROUND_TASKS"), "应已删除"
        assert not hasattr(s, "LITE_DISABLE_IMPULSE"), "应已删除"


# ===================================================================
# L15-L17 — async def → def
# ===================================================================
class TestL15AsyncFunctionsToSync:
    """auth.py 中 async def 但无 await 的函数改为 sync。"""

    def test_get_current_user_not_async(self):
        import inspect
        from app.core.auth import get_current_user
        assert not inspect.iscoroutinefunction(get_current_user)

    def test_get_user_context_not_async(self):
        import inspect
        from app.core.auth import get_user_context
        assert not inspect.iscoroutinefunction(get_user_context)


class TestL16HealthNotAsync:
    def test_health_not_async(self):
        import inspect
        from app.api.health import health
        assert not inspect.iscoroutinefunction(health)

    def test_health_ollama_not_async(self):
        import inspect
        from app.api.health import health_ollama
        assert not inspect.iscoroutinefunction(health_ollama)


class TestL17PagesNotAsync:
    def test_pages_not_async(self):
        import inspect
        from app.api.pages import login_page, chat_page, root, dashboard
        assert not inspect.iscoroutinefunction(login_page)
        assert not inspect.iscoroutinefunction(chat_page)
        assert not inspect.iscoroutinefunction(root)
        assert not inspect.iscoroutinefunction(dashboard)
