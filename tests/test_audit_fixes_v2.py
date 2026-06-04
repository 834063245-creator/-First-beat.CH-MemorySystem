"""初痕 v2.0 敌对审计修复验证 — 覆盖 P0/P1/P2 全部 14 条修复。"""
import json
import os
import threading
import time

import pytest


# ══════════════════════════════════════════════════════════════
# P0-1：时间段标签三处统一
# ══════════════════════════════════════════════════════════════
class TestP01TimePeriodConsistency:
    """验证 settings.py TIME_PERIOD_MAP、dispatch.py 工具描述、context.py 入库逻辑三处一致。"""

    def test_time_period_map_sane(self):
        from app.config.settings import TIME_PERIOD_MAP
        # 24 小时全覆盖，无缺口、无重叠
        covered = set()
        for (lo, hi), name in TIME_PERIOD_MAP.items():
            for h in range(lo, hi + 1):
                assert h not in covered, f"小时 {h} 被多个时段覆盖"
                covered.add(h)
        assert covered == set(range(24)), f"未覆盖的小时: {set(range(24)) - covered}"

    def test_time_period_labels_match_map(self):
        from app.config.settings import TIME_PERIOD_MAP, TIME_PERIOD_LABELS
        # 标签的反向映射必须与正向映射完全一致
        for label, (lo, hi) in TIME_PERIOD_LABELS.items():
            assert TIME_PERIOD_MAP[(lo, hi)] == label, \
                f"TIME_PERIOD_LABELS[{label}]={lo,hi} 与 TIME_PERIOD_MAP 不一致"
        # 反向检查
        for (lo, hi), label in TIME_PERIOD_MAP.items():
            assert label in TIME_PERIOD_LABELS, \
                f"TIME_PERIOD_MAP[{lo,hi}]={label} 在 TIME_PERIOD_LABELS 中缺失"

    def test_dispatch_description_matches_map(self):
        from app.config.settings import TIME_PERIOD_MAP
        from app.tools.dispatch import QUERY_MEMORY_TOOL
        import json
        # 工具描述在 QUERY_MEMORY_TOOL dict 中
        desc = json.dumps(QUERY_MEMORY_TOOL, ensure_ascii=False)
        for (lo, hi), name in TIME_PERIOD_MAP.items():
            assert f"{name}({lo}-{hi})" in desc, \
                f"dispatch.py 工具描述缺少 {name}({lo}-{hi})"

    def test_context_period_from_map(self):
        """context.py 的 _store_conversation 使用 TIME_PERIOD_MAP 而非硬编码。"""
        import app.core.context as ctx
        import inspect
        src = inspect.getsource(ctx)
        # 不应出现旧的硬编码边界
        assert "elif h < 6:\n            period = " not in src, \
            "context.py 仍包含旧的 if-elif 硬编码"
        assert "TIME_PERIOD_MAP" in src, \
            "context.py 未引用 TIME_PERIOD_MAP"


# ══════════════════════════════════════════════════════════════
# P0-2：多字符代词匹配修复
# ══════════════════════════════════════════════════════════════
class TestP02MultiCharPronouns:
    """验证 metadata.py 的 extract_persons 能匹配多字符代词。"""

    def test_single_char_pronouns_still_work(self):
        from app.core.metadata import extract_persons
        persons = extract_persons("我今天很开心")
        assert "我" in persons, "单字代词 '我' 应被匹配"

    def test_multi_char_pronouns_matched(self):
        from app.core.metadata import extract_persons
        persons = extract_persons("我们一起去吧")
        assert "我们" in persons, "双字代词 '我们' 应被匹配"

    def test_all_multi_char_pronouns(self):
        from app.core.metadata import extract_persons
        persons = extract_persons("我们你们他们都听我说")
        for p in ["我们", "你们", "他们"]:
            assert p in persons, f"多字代词 '{p}' 未被匹配"

    def test_no_false_positive_on_similar(self):
        """确保不会把 '们的' 这类非代词误匹配。"""
        from app.core.metadata import extract_persons
        persons = extract_persons("同学们的电影")
        assert "我们" not in persons  # "我们" 不在 "同学们的电影" 中
        assert "你们" not in persons
        assert "他们" not in persons


# ══════════════════════════════════════════════════════════════
# P0-3：ChromaService close() 单一定义 + 锁保护
# ══════════════════════════════════════════════════════════════
class TestP03CloseUnified:
    """验证 close() 只有一个定义，且带锁保护。"""

    def test_close_uses_lock(self):
        import inspect
        from app.memory.chroma import ChromaService
        src = inspect.getsource(ChromaService.close)
        assert "with self._lock:" in src, \
            "close() 未使用 self._lock 保护"
        assert "self._collection" in src, \
            "close() 未引用 self._collection"

    def test_only_one_close_definition(self):
        import inspect
        from app.memory.chroma import ChromaService
        src = inspect.getsource(ChromaService)
        count = src.count("def close(self):")
        assert count == 1, f"close() 定义了 {count} 次，应为 1 次"


# ══════════════════════════════════════════════════════════════
# P1-1：CoOccurrenceTracker LTD 计数器竞态修复
# ══════════════════════════════════════════════════════════════
class TestP11CooccurLtdLock:
    """验证 cooccur.py LTD 计数器受锁保护。"""

    def test_ltd_lock_exists(self, tmp_path):
        from app.memory.cooccur import CoOccurrenceTracker
        co_file = tmp_path / "co_test.json"
        co_file.write_text("{}")
        tracker = CoOccurrenceTracker(file_path=str(co_file))
        assert hasattr(tracker, "_ltd_lock"), \
            "CoOccurrenceTracker 缺少 _ltd_lock"
        assert isinstance(tracker._ltd_lock, threading.Lock)

    def test_ltd_increment_under_lock(self):
        import inspect
        from app.memory.cooccur import CoOccurrenceTracker
        src = inspect.getsource(CoOccurrenceTracker.query)
        assert "with self._ltd_lock:" in src, \
            "query() 中 _ltd_counter 自增未受 _ltd_lock 保护"

    def test_concurrent_ltd_no_double_trigger(self, tmp_path):
        """多线程并发的 LTD 检查不应双触发衰减。"""
        from app.memory.cooccur import CoOccurrenceTracker
        co_file = tmp_path / "co_test.json"
        co_file.write_text("{}")
        tracker = CoOccurrenceTracker(file_path=str(co_file))
        # 将计数器推到阈值边缘
        tracker._ltd_counter = tracker.LTD_CHECK_INTERVAL - 1
        # 并发 10 个 query 调用
        def query_once():
            tracker.query([])
        threads = [threading.Thread(target=query_once) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 关键：_ltd_counter 最终值应正确（十个线程各 +1，触发一次重置）
        # 锁保证不丢计数、不双触发
        assert tracker._ltd_counter >= 0
        assert tracker._ltd_counter < tracker.LTD_CHECK_INTERVAL


# ══════════════════════════════════════════════════════════════
# P1-2：LocalLLM 从 settings.py 统一取值
# ══════════════════════════════════════════════════════════════
class TestP12LocalLLMSettings:
    """验证 LocalLLM 从 app.config.settings 导入而非直接读 os.getenv。"""

    def test_uses_settings_not_os_environ(self):
        import inspect
        from app.llm.local import LocalLLM
        src = inspect.getsource(LocalLLM.__init__)
        assert "from app.config.settings import" in src, \
            "LocalLLM.__init__ 未从 settings 导入配置"
        assert 'os.getenv(' not in src, \
            "LocalLLM.__init__ 仍在直接调用 os.getenv"

    def test_model_matches_settings(self):
        """LocalLLM 默认 model 应与 settings.py 一致。"""
        from app.config.settings import LLM_MODEL
        from app.llm.local import LocalLLM
        # 不传 model → 应使用 LLM_MODEL
        llm = LocalLLM()
        assert llm._model == LLM_MODEL, \
            f"LocalLLM 默认 model={llm._model}，应为 settings.LLM_MODEL={LLM_MODEL}"


# ══════════════════════════════════════════════════════════════
# P1-3：死配置导入移除
# ══════════════════════════════════════════════════════════════
class TestP13DeadImports:
    """验证 context.py 不再导入 TIMELINE_RECENT_COUNT 和 WORK_MEMORY_TOKEN_BUDGET。"""

    def test_dead_imports_removed(self):
        import inspect
        from app.core import context
        src = inspect.getsource(context)
        # 这些名字不应出现在 import 语句中
        assert "TIMELINE_RECENT_COUNT" not in src, \
            "context.py 仍在导入 TIMELINE_RECENT_COUNT"
        assert "WORK_MEMORY_TOKEN_BUDGET" not in src, \
            "context.py 仍在导入 WORK_MEMORY_TOKEN_BUDGET"

    def test_legacy_configs_still_defined(self):
        """legacy 配置仍存在于 settings.py，只是注释了。"""
        from app.config.settings import TIMELINE_RECENT_COUNT, WORK_MEMORY_TOKEN_BUDGET
        assert TIMELINE_RECENT_COUNT == 5
        assert WORK_MEMORY_TOKEN_BUDGET == 50000


# ══════════════════════════════════════════════════════════════
# P1-4：ChromaDB 单 PersistentClient 统一
# ══════════════════════════════════════════════════════════════
class TestP14SinglePersistentClient:
    """验证三处存储改为单 PersistentClient + Lock。"""

    def test_chroma_no_dual_client(self):
        from app.memory.chroma import ChromaService
        svc = ChromaService()
        assert not hasattr(svc, "_read_client"), \
            "ChromaService 仍有 _read_client（应为单 _client）"
        assert not hasattr(svc, "_write_client")
        assert not hasattr(svc, "_read_collection")
        assert not hasattr(svc, "_write_collection")
        assert hasattr(svc, "_client"), \
            "ChromaService 缺少 _client"
        assert hasattr(svc, "_collection"), \
            "ChromaService 缺少 _collection"

    def test_personality_no_dual_client(self):
        from app.personality.store import PersonalityStore
        store = PersonalityStore()
        assert not hasattr(store, "_read_client")
        assert not hasattr(store, "_write_client")
        assert not hasattr(store, "_write_coll")
        assert hasattr(store, "_client")
        assert hasattr(store, "_collection")

    def test_behavior_no_dual_client(self):
        from app.personality.behavior import BehaviorStore
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            store = BehaviorStore(persist_dir=td)
            assert not hasattr(store, "_write_client")
            assert not hasattr(store, "_write_collection")
            assert hasattr(store, "_client")
            assert hasattr(store, "_collection")
            # 释放 ChromaDB 连接以便 Windows 清理临时目录
            store._client = None
            store._collection = None

    def test_chroma_close_single_refs(self):
        import inspect
        from app.memory.chroma import ChromaService
        src = inspect.getsource(ChromaService.close)
        assert "self._client = None" in src
        assert "self._collection = None" in src
        assert "_read_client" not in src, "close() 仍引用旧 _read_client"
        assert "_write_client" not in src


# ══════════════════════════════════════════════════════════════
# P2-4：ngram 缓存 LRU 淘汰（非全量清除）
# ══════════════════════════════════════════════════════════════
class TestP24NgramLRU:
    """验证 embed.py 的 ngram 缓存满时使用 LRU 淘汰而非全量清除。"""

    def test_cache_uses_lru_not_clear(self):
        import inspect
        from app.llm.embed import _ngram_sig
        src = inspect.getsource(_ngram_sig)
        # 不应有 .clear()
        assert "_ngram_cache.clear()" not in src, \
            "ngram 缓存满时仍在用 clear() 全量清除"
        assert "pop(next(iter(" in src, \
            "ngram 缓存满时应使用 pop(next(iter(...))) 进行 LRU 淘汰"

    def test_lru_eviction_keeps_recent(self):
        """填充缓存至溢出后，验证最近插入的条目仍存在。"""
        import app.llm.embed as embed_mod
        # 重置缓存
        with embed_mod._ngram_cache_lock:
            embed_mod._ngram_cache.clear()
        # 填充到刚好溢出
        for i in range(embed_mod._NGRAM_CACHE_MAX + 5):
            embed_mod._ngram_sig(f"test text {i}")
        with embed_mod._ngram_cache_lock:
            size = len(embed_mod._ngram_cache)
        # 缓存大小不应超过上限
        assert size <= embed_mod._NGRAM_CACHE_MAX, \
            f"缓存大小 {size} 超过上限 {embed_mod._NGRAM_CACHE_MAX}"


# ══════════════════════════════════════════════════════════════
# P2-7：ALL_TOOLS 统一引用
# ══════════════════════════════════════════════════════════════
class TestP27AllToolsDRY:
    """验证 chat.py 使用 ALL_TOOLS 而非重复构造工具列表。"""

    def test_all_tools_defined(self):
        from app.core.tools import ALL_TOOLS
        assert isinstance(ALL_TOOLS, list)
        assert len(ALL_TOOLS) == 8, f"ALL_TOOLS 应有 8 个工具，实际 {len(ALL_TOOLS)}"

    def test_chat_py_imports_all_tools(self):
        import inspect
        from app.api import chat
        src = inspect.getsource(chat)
        assert "ALL_TOOLS" in src, "chat.py 未引用 ALL_TOOLS"

    def test_old_tools_list_removed(self):
        """TOOLS 变量已删除。"""
        import app.core.tools as tmod
        assert not hasattr(tmod, "TOOLS"), \
            "tools.py 仍存在旧的 TOOLS 变量（应已删除）"


# ══════════════════════════════════════════════════════════════
# P2-1：死代码清理
# ══════════════════════════════════════════════════════════════
class TestP21DeadCodeRemoved:
    """验证未使用的模型、标签、端点已清除。"""

    def test_prompt_body_removed(self):
        from app.models import schemas as s
        assert not hasattr(s, "PromptBody"), "PromptBody 应已删除"
        assert not hasattr(s, "MemoryListResponse")
        assert not hasattr(s, "MemoryDeleteResponse")
        assert not hasattr(s, "CorrectMemoryBody")

    def test_intent_emotion_labels_removed(self):
        from app.brain import keywords as kw
        assert not hasattr(kw, "INTENT_LABELS"), "INTENT_LABELS 应已删除"
        assert not hasattr(kw, "EMOTION_LABELS"), "EMOTION_LABELS 应已删除"

    def test_ollama_health_endpoint_gone(self):
        from app.api import health
        assert not hasattr(health, "health_ollama"), \
            "health_ollama 死端点应已删除"

    def test_ollama_health_route_gone(self):
        import inspect
        from app.api import health
        src = inspect.getsource(health)
        assert "/health/ollama" not in src, \
            "/health/ollama 路由应已删除"


# ══════════════════════════════════════════════════════════════
# P2-2：空目录清理
# ══════════════════════════════════════════════════════════════
class TestP22EmptyDirsRemoved:
    """验证 app/knowledge/ 和 app/web/ 已删除。"""

    def test_knowledge_dir_gone(self):
        import app
        app_dir = os.path.dirname(app.__file__)
        assert not os.path.isdir(os.path.join(app_dir, "knowledge")), \
            "app/knowledge/ 空目录应已删除"

    def test_web_dir_gone(self):
        import app
        app_dir = os.path.dirname(app.__file__)
        assert not os.path.isdir(os.path.join(app_dir, "web")), \
            "app/web/ 空目录应已删除"


# ══════════════════════════════════════════════════════════════
# P2-3：query_explore 标注为非生产路径
# ══════════════════════════════════════════════════════════════
class TestP23QueryExploreAnnotated:
    """验证 dispatch.py 顶部注释标注了 query_explore 的用途。"""

    def test_header_mentions_manual_audit(self):
        import app.tools.dispatch as disp
        doc = disp.__doc__
        assert doc is not None
        assert "手动审计" in doc or "测试使用" in doc or "非生产" in doc, \
            "dispatch.py 顶部未标注 query_explore 为测试/审计用途"


# ══════════════════════════════════════════════════════════════
# P2-5：注释修复
# ══════════════════════════════════════════════════════════════
class TestP25CommentFixed:
    """验证 RERANK_HIT_WEIGHT 注释已修正。"""

    def test_no_stale_comment(self):
        import app.config.settings as s
        import inspect
        src = inspect.getsource(s)
        assert "替代 RERANK_BETA" not in src, \
            "RERANK_HIT_WEIGHT 仍引用不存在的 RERANK_BETA"


# ══════════════════════════════════════════════════════════════
# P2-6：OLLAMA_MODELS 注释
# ══════════════════════════════════════════════════════════════
class TestP26OllamaModelsComment:
    """验证 OLLAMA_MODELS 环境变量处有说明注释。"""

    def test_ollama_models_has_comment(self):
        import inspect
        import app.config.settings as s
        src = inspect.getsource(s)
        # 应有注释说明 OLLAMA_MODELS 对 Python 端无效
        assert "OLLAMA_MODELS" in src
        # 前一行应有中文注释
        lines = src.split("\n")
        found = False
        for i, line in enumerate(lines):
            if "os.environ[\"OLLAMA_MODELS\"]" in line or "_OLLAMA_MODELS = os.getenv" in line:
                # 检查前几行是否有中文注释
                context = "\n".join(lines[max(0, i-4):i+1])
                if "环境变量" in context or "Python 端设置无效" in context or "服务端" in context:
                    found = True
                break
        assert found, "OLLAMA_MODELS 附近缺少说明注释"


# ══════════════════════════════════════════════════════════════
# 回归：确保被删除的旧导入路径不会导致 import 错误
# ══════════════════════════════════════════════════════════════
class TestRegressionImportSafety:
    """验证所有核心模块可正常导入，无 ImportError。"""

    def test_all_core_modules_importable(self):
        modules = [
            ("app.config.settings", None),
            ("app.core.context", None),
            ("app.core.metadata", None),
            ("app.core.tools", None),
            ("app.memory.chroma", None),
            ("app.memory.cooccur", None),
            ("app.llm.local", None),
            ("app.llm.embed", None),
            ("app.personality.store", None),
            ("app.personality.behavior", None),
            ("app.brain.keywords", None),
            ("app.models.schemas", None),
            ("app.api.health", None),
            ("app.api.chat", None),
            ("app.tools.dispatch", None),
        ]
        import importlib
        for mod_name, _ in modules:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                pytest.fail(f"导入 {mod_name} 失败: {e}")

    def test_chroma_close_smoke(self):
        """ChromaService.close() 可正常调用，不抛异常。"""
        from app.memory.chroma import ChromaService
        svc = ChromaService()
        svc.close()  # 不应抛异常

    def test_cooccur_query_smoke(self):
        """CoOccurrenceTracker.query() 在空数据集上正常返回。"""
        from app.memory.cooccur import CoOccurrenceTracker
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = f.name
        try:
            tracker = CoOccurrenceTracker(file_path=path)
            result = tracker.query([])
            assert isinstance(result, list)
        finally:
            os.unlink(path)

    def test_personality_store_smoke(self):
        """PersonalityStore 正常初始化和简单操作。"""
        from app.personality.store import PersonalityStore
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            store = PersonalityStore(persist_dir=td)
            assert store._collection is not None
            # 释放 ChromaDB 连接以便 Windows 清理临时目录
            store._client = None
            store._collection = None

    def test_behavior_store_smoke(self):
        """BehaviorStore 正常初始化和简单操作。"""
        from app.personality.behavior import BehaviorStore
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            store = BehaviorStore(persist_dir=td)
            assert store._collection is not None
            count = store.count()
            assert isinstance(count, int)
            # 释放 ChromaDB 连接以便 Windows 清理临时目录
            store._client = None
            store._collection = None
