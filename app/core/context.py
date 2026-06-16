"""应用上下文 — AppContext 类 + ctx_manager 导出。

AppContext 类在此处完整定义，所有模块从 app/ 自给自足。
"""
import asyncio
import queue
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)



from app.config.settings import (                  # noqa: E402
    CHROMA_PERSIST_DIR, DATA_DIR, DEFAULT_TOP_K, MAX_MEMORIES_IN_PROMPT,
    CHAT_HISTORY_PATH, CHAT_HISTORY_MAX_MEMORY, DEBUG_INCLUDE_PROMPT,
    STORE_FAILURES_PATH,
    CONSOLIDATION_SHALLOW_INTERVAL, CONSOLIDATION_DEEP_INTERVAL,
    AI_CHROMA_DIR, AI_COLLECTION,
    IMPULSE_ACTIVE_PATH_B,
    WORK_MEMORY_TOKEN_BUDGET, USER_DATA_DIRS,
    BENCHMARK_MODE,
    PORTRAIT_FILE_PATH,
    DRIFT_DECISION_LOG,
    STOP_WORDS as _STOP_WORDS,
)
from app.config.settings import STORAGE_BACKEND as _STORAGE_BACKEND
from app.memory.chroma import ChromaService
if _STORAGE_BACKEND == "qdrant":
    from app.memory.qdrant import QdrantService
from app.llm.deepseek import LLMClient
# Phase 3: CoOccurrenceStore/HyperEdgeStore 替代 SQLite (cooccur/entity_pair/hyperedge)
from app.memory.qdrant_cooccur import CoOccurrenceStore
from app.memory.qdrant_hyperedge import HyperEdgeStore
from app.memory.history import ChatHistory
from app.memory.inverted import InvertedIndex
from app.memory.affinity import TopicAffinity
from app.memory.temporal import TemporalPatternIndex
from app.analysis.predictor import BehaviorPredictor  # noqa: E402
from app.analysis.pattern_discovery import PatternDiscovery  # noqa: E402
from app.tools.atomic import atomic_write             # noqa: E402

# ── 共享 httpx 客户端 ──────────────────────────────────────────
_impulse_httpx = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

# ── 局部 LLM 惰性单例 ─────────────────────────────────────────
_LOCAL_LLM: "LocalLLM | None" = None
_LOCAL_LLM_LOCK = threading.Lock()


def _get_local_llm() -> "LocalLLM":
    """惰性初始化的 LocalLLM 单例。"""
    global _LOCAL_LLM
    if _LOCAL_LLM is None:
        with _LOCAL_LLM_LOCK:
            if _LOCAL_LLM is None:
                from app.llm.local import LocalLLM
                _LOCAL_LLM = LocalLLM()
    return _LOCAL_LLM


def _extract_noun_tags(text: str, topk: int = 8) -> list[str]:
    """提取标签：语义层关键词提取（bge-m3 KeyBERT）。"""
    from app.brain.semantic import extract_tags
    return extract_tags(text, topk=topk)


# ===================================================================
# AppContext — 所有服务实例的容器
# ===================================================================

class AppContext:
    """应用服务容器。初始化顺序反映依赖关系。"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Phase 1: STORAGE_BACKEND 切换（chromadb 默认, qdrant 可选）
        if _STORAGE_BACKEND == "qdrant":
            self.chroma_service = QdrantService(
                persist_dir=f"{data_dir}/qdrant",
                collection_name=EMBED_MODELS[DEFAULT_EMBED_MODEL]["collection"],
            )
            self.ai_chroma_service = QdrantService(
                persist_dir=f"{data_dir}/ai_qdrant",
                collection_name=AI_COLLECTION,
            )
        else:
            self.chroma_service = ChromaService(
                persist_dir=f"{data_dir}/chroma",
            )
            self.ai_chroma_service = ChromaService(
                persist_dir=f"{data_dir}/ai_chroma",
                collection_name=AI_COLLECTION,
            )
        self.memory_service = self.chroma_service  # 统一别名
        self.ai_memory_service = self.ai_chroma_service
        self.llm_client = LLMClient()
        self.storage_executor = ThreadPoolExecutor(max_workers=5)
        self.retrieval_executor = ThreadPoolExecutor(max_workers=3)
        import atexit
        atexit.register(self._cleanup_executors)
        # Phase 3: CoOccurrenceStore/HyperEdgeStore 替代 SQLite 表
        if _STORAGE_BACKEND == "qdrant":
            _qs_client = self.chroma_service.client
            _embed_getter = lambda mid: self.chroma_service._get_embedding_cached(mid)
            from app.llm.embed import local_embed_batch
            self.co_tracker = CoOccurrenceStore(
                _qs_client, "co_occurrence", embed_getter=_embed_getter,
            )
            self.ai_co_tracker = CoOccurrenceStore(
                _qs_client, "ai_co_occurrence", embed_getter=_embed_getter,
            )
            self.hyperedge_index = HyperEdgeStore(
                _qs_client, "hyper_edges", embed_batch_fn=local_embed_batch,
            )
        else:
            # ChromaDB 回退：新建本地 Qdrant 客户端给新 stores
            from qdrant_client import QdrantClient
            _qs_client = QdrantClient(location=":memory:")
            self.co_tracker = CoOccurrenceStore(
                _qs_client, "co_occurrence",
            )
            self.ai_co_tracker = CoOccurrenceStore(
                _qs_client, "ai_co_occurrence",
            )
            self.hyperedge_index = HyperEdgeStore(
                _qs_client, "hyper_edges",
            )
        # entity_pair_tracker 升级为 entity_co_counts payload 字段
        self.entity_pair_tracker = None
        self.chat_history = ChatHistory(path=f"{data_dir}/chat_history.jsonl", max_memory=CHAT_HISTORY_MAX_MEMORY)
        self.inverted_index = InvertedIndex()
        try:
            _all = self.chroma_service.list_all()
            _tag_entries = [(m['id'], (m.get('metadata') or {}).get('tags', '') or '') for m in _all]
            self.inverted_index.build_tags(_tag_entries)
            logger.info('标签索引构建完成: %d 个标签', len(self.inverted_index._tag_index))
        except Exception:
            pass

        # Phase 1: 画像系统初始化
        from app.portrait.manager import PortraitManager
        from app.portrait.renderer import PortraitRenderer
        from app.portrait.writer import PortraitWriter
        self.portrait = PortraitManager(PORTRAIT_FILE_PATH)
        self.portrait_renderer = PortraitRenderer(self.portrait)
        self.portrait_writer = PortraitWriter(self.portrait)

        # Phase 2: BM25 全文索引已删除 — Qdrant MatchText 原生替代

        self.topic_affinity = TopicAffinity(data_dir=data_dir)
        self.temporal_pattern_index = TemporalPatternIndex(data_dir=data_dir)
        from app.background.consolidation import ConsolidationEngine
        self.dmn = ConsolidationEngine(
            chroma_service=self.chroma_service,
            chat_history=self.chat_history,
            co_tracker=self.co_tracker,
            state_path=f"{data_dir}/dmn_state.json",
            notes_path=f"{data_dir}/topic_notes.json",
            temporal_pattern_index=self.temporal_pattern_index,
            topic_affinity=self.topic_affinity,
            ai_co_tracker=self.ai_co_tracker,
        )
        # Phase 0b: AI 完整 ConsolidationEngine — 与用户侧完全镜像
        self.ai_dmn = ConsolidationEngine(
            chroma_service=self.ai_chroma_service,
            chat_history=self.chat_history,
            co_tracker=self.ai_co_tracker,
            state_path=f"{data_dir}/ai_dmn_state.json",
            notes_path=f"{data_dir}/ai_topic_notes.json",
            temporal_pattern_index=self.temporal_pattern_index,
            topic_affinity=self.topic_affinity,
        )

        # 话题树（DMN 浅巩固时重建，pipeline 检索时使用）
        self._topic_tree = getattr(self.dmn, '_topic_tree', None)
        self._tag_index = getattr(self.dmn, '_tag_index', None)

        # 模式发现层（零 LLM 调用，纯统计缓存）
        self._pattern_discovery = PatternDiscovery(
            data_dir=data_dir,
            temporal_index=self.temporal_pattern_index,
            affinity=self.topic_affinity,
            chat_history_path=f"{data_dir}/chat_history.jsonl",
        )
        self._pattern_discovery.load_cache()
        if hasattr(self, 'llm_client'):
            self.llm_client.set_pattern_discovery(self._pattern_discovery)

        from app.background.impulse import ImpulseScheduler
        self.impulse_scheduler = ImpulseScheduler(
            state_path=f"{data_dir}/impulse_state.json",
            temporal_pattern_index=self.temporal_pattern_index,
        )
        self.mirror_neuron = BehaviorPredictor(data_dir=data_dir)

        # Part A: 偏移率追踪
        from app.analysis.drift import DriftTracker
        self.drift_tracker = DriftTracker(log_path=DRIFT_DECISION_LOG)

        # Part B: 自我镜像
        from app.analysis.self_mirror import SelfMirror
        self.self_mirror = SelfMirror()

        # 每个用户的存储队列路径
        self._store_queue_path = f"{data_dir}/store_queue.jsonl"
        self._store_queue_lock = threading.Lock()
        self._store_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # 启动用户专属后台线程
        self._start_queue_worker()
        self._start_impulse_workers()
        if self.dmn:
            self._start_dmn_worker()
        self._start_ai_consolidation_worker()
        # 后台预热 embedding cache（不阻塞启动）
        self.storage_executor.submit(self._prewarm_retrieval)

        # 注册后台线程到生命周期管理器（统一关闭入口）
        from app.background.lifecycle import register as _reg
        _reg("queue_worker", start=lambda: None, stop=lambda: self._queue_thread.join(timeout=5)
             if getattr(self, '_queue_thread', None) else None)
        if self.impulse_scheduler:
            _reg("impulse", start=lambda: None, stop=self.impulse_scheduler.stop)
        if self.dmn:
            _reg("dmn_worker", start=lambda: None, stop=lambda: self._stop_event.set())

    @property
    def topic_tree(self):
        """公开只读接口——MCP / API 可通过此属性获取话题树。"""
        return self._topic_tree

    # ── 队列 worker ──────────────────────────────────────────

    def _start_queue_worker(self):
        """启动用户专属入库队列 worker。"""
        def _worker():
            logger.info("入库队列 worker 已启动 for %s", self.data_dir)
            _loop_count = 0
            _disk_check_interval = 30  # 每30次循环检查一次磁盘遗留（~30秒空闲）
            while not self._stop_event.is_set():
                try:
                    tasks = []
                    _loop_count += 1

                    # 磁盘恢复检查：仅启动后首次 + 每30次循环执行
                    if _loop_count == 1 or _loop_count % _disk_check_interval == 0:
                        with self._store_queue_lock:
                            if os.path.exists(self._store_queue_path):
                                with open(self._store_queue_path, encoding="utf-8") as f:
                                    lines = f.readlines()
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        tasks.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                                tmp = None
                                if tasks:
                                    tmp = self._store_queue_path + ".tmp." + str(time.time())
                                    try:
                                        os.rename(self._store_queue_path, tmp)
                                    except OSError:
                                        tmp = None
                                    logger.info("队列取出 %d 条任务待处理 for %s", len(tasks), self.data_dir)
                                    if len(tasks) > 50:
                                        logger.warning("队列积压: %d 条待处理 for %s", len(tasks), self.data_dir)

                    if tasks:
                        _consecutive_failures = 0
                        for i, task in enumerate(tasks):
                            logger.info("队列任务 [%d/%d]: 入库 %s...", i + 1, len(tasks), task.get("timestamp", "?")[:16])
                            try:
                                self._store_conversation(
                                    task["user_message"], task["ai_message"], task["timestamp"]
                                )
                                logger.info("队列任务 [%d/%d]: 完成", i + 1, len(tasks))
                                _consecutive_failures = 0
                            except Exception as e:
                                logger.error("队列任务 [%d/%d] 失败: %s", i + 1, len(tasks), e)
                                _consecutive_failures += 1
                                if _consecutive_failures >= 3:
                                    _backoff = min(60, 10 * _consecutive_failures)
                                    logger.error("队列连续失败 %d 次，退避 %.0fs", _consecutive_failures, _backoff)
                                    time.sleep(_backoff)
                        try:
                            if tmp and os.path.exists(tmp):
                                os.remove(tmp)
                        except OSError:
                            pass
                        continue

                    try:
                        task = self._store_queue.get(timeout=1)
                        self._store_conversation(
                            task["user_message"], task["ai_message"], task["timestamp"]
                        )
                    except queue.Empty:
                        pass
                except Exception as e:
                    logger.error("队列 worker 循环异常: %s", e)
                    try:
                        self._store_queue.get(timeout=1)
                    except queue.Empty:
                        pass

        t = threading.Thread(target=_worker, daemon=True, name=f"store_queue_{self.data_dir}")
        t.start()
        self._queue_thread = t

    # ── 冲动 worker ──────────────────────────────────────────

    def _start_impulse_workers(self):
        """启动用户专属冲动源泊松线程。"""
        if self.impulse_scheduler:
            self.impulse_scheduler.start_source_workers(
                chroma_service=self.chroma_service,
                chat_history=self.chat_history,
            )
            self._start_impulse_consumer()

    # ── 冲动消费 worker ──────────────────────────────────────

    def _start_impulse_consumer(self):
        """冲动信号内部消费线程：取信号 → LLM 生成 → 存为 [内心独白]。

        这是冲动系统闭环的最后一环——引擎不等外部调用，自己消费自己产出的信号。
        """
        import asyncio

        def _consumer():
            # 在 daemon 线程中创建持久 event loop（复用，避免每次 asyncio.run() 新建）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info("冲动消费线程已启动 for %s", self.data_dir)
            while not self._stop_event.is_set():
                try:
                    # 检查 Path B 开关
                    if not IMPULSE_ACTIVE_PATH_B:
                        self._stop_event.wait(60)
                        continue

                    # 检查空闲条件（用户不在聊天中）
                    from app.core.heartbeat import get_last_heartbeat
                    hb = get_last_heartbeat()
                    if not self.impulse_scheduler.should_deliver(self.chat_history, hb):
                        self._stop_event.wait(30)
                        continue

                    # 取信号（已通过频率限制 + 疲劳抑制）
                    imp = self.impulse_scheduler.get_next()
                    if not imp:
                        self._stop_event.wait(30)
                        continue

                    signal = imp.get("content", "")
                    source = imp.get("source", "")
                    if not signal:
                        continue

                    logger.info("冲动消费: [%s] %s", source, signal[:80])

                    # 构建发言 prompt
                    from app.llm.deepseek import load_system_prompt, now_hint
                    system_prompt = load_system_prompt()
                    speak_prompt = (
                        f"{now_hint()}\n\n"
                        f"你的心里有一个信号在浮现：\n"
                        f"{signal}\n\n"
                        f"顺着这个信号，你想对用户说什么就说什么。\n"
                        f"不要解释这个信号从哪里来，不要加'我注意到'这类前缀，\n"
                        f"就只是顺着心里的感觉，自然地开口。"
                    )

                    # 调 LLM 生成自然语言（复用 daemon 线程的持久 event loop）
                    try:
                        result = loop.run_until_complete(
                            self.llm_client.generate(
                                speak_prompt,
                                max_tokens=256,
                            )
                        )
                        reply = result.get("content", "").strip()
                    except Exception as llm_err:
                        logger.warning("冲动LLM调用失败，回退原文: %s", llm_err)
                        reply = signal

                    if not reply:
                        continue

                    logger.info("冲动发言: %s", reply[:80])

                    # 存为内心独白
                    from datetime import datetime
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.chat_history.append("[内心独白]", reply, now_ts)
                    self._enqueue_store_task(
                        user_message="[内心独白]",
                        ai_message=reply,
                        timestamp=now_ts,
                    )

                except Exception as exc:
                    logger.error("冲动消费循环异常: %s", exc)
                    self._stop_event.wait(60)

        t = threading.Thread(
            target=_consumer,
            daemon=True,
            name=f"impulse_consumer_{self.data_dir}",
        )
        t.start()
        self._impulse_consumer_thread = t

    # ── DMN + 巩固 合并 ticker ───────────────────────────────
    # 原来两个独立泊松线程（都 ~5min）合并为一个，减少冗余唤醒和状态读取。

    def _start_dmn_worker(self):
        """DMN 合并 ticker：空闲检查 + 浅/深巩固 + 模式发现（用户 + AI 双引擎）。"""
        def _worker():
            logger.info("DMN 合并 ticker 已启动 for %s", self.data_dir)
            # 启动冷却：60s，避免和预热/冲动源抢资源
            if not self._stop_event.is_set():
                self._stop_event.wait(60)
            while not self._stop_event.is_set():
                try:
                    # ── 空闲触发（原 DMN worker 逻辑） ──
                    try:
                        if self.chat_history and len(self.chat_history.records) >= 1:
                            last_rec = self.chat_history.records[-1]
                            last_ts = last_rec.get("timestamp", "")
                            if last_ts:
                                gap = datetime.now() - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                                gap_hours = gap.total_seconds() / 3600
                                if gap_hours > 0.5:
                                    self.dmn.on_idle(gap_hours)
                                    # Phase 0b: AI 侧镜像巩固 — 共享同一 on_idle 触发
                                    self.ai_dmn.on_idle(gap_hours)
                    except Exception:
                        pass
                    # ── 节律巩固（原 consolidation worker 逻辑） ──
                    try:
                        lsc = 0
                        ldc = 0
                        try:
                            from app.background.consolidation import _load_state as _dmn_load
                            dmn_state = _dmn_load(f"{self.data_dir}/dmn_state.json")
                            lsc = dmn_state.get("last_shallow_consolidation", 0) or 0
                            ldc = dmn_state.get("last_deep_consolidation", 0) or 0
                        except Exception:
                            pass
                        now = time.time()
                        if now - lsc >= CONSOLIDATION_SHALLOW_INTERVAL:
                            logger.info("巩固节律: 触发浅巩固 (距上次 %.1f 小时)", (now - lsc) / 3600)
                            try:
                                self.dmn.consolidate_shallow()
                            except Exception as exc:
                                logger.error("巩固节律: 浅巩固失败: %s", exc)
                            # Phase 0b: AI 侧浅巩固镜像
                            try:
                                self.ai_dmn.consolidate_shallow()
                            except Exception as exc:
                                logger.error("巩固节律: AI 浅巩固失败: %s", exc)
                            try:
                                self._pattern_discovery.run()
                            except Exception as exc:
                                logger.error("模式发现运行失败: %s", exc)
                        if now - ldc >= CONSOLIDATION_DEEP_INTERVAL:
                            logger.info("巩固节律: 触发深巩固 (距上次 %.1f 小时)", (now - ldc) / 3600)
                            try:
                                self.dmn.consolidate_deep()
                            except Exception as exc:
                                logger.error("巩固节律: 深巩固失败: %s", exc)
                            # Phase 0b: AI 侧深巩固镜像
                            try:
                                self.ai_dmn.consolidate_deep()
                            except Exception as exc:
                                logger.error("巩固节律: AI 深巩固失败: %s", exc)
                    except Exception:
                        pass
                    # ── Phase 3: 画像浅/深巩固（引擎+LLM合成） ──
                    try:
                        if now - lsc >= CONSOLIDATION_SHALLOW_INTERVAL:
                            self.portrait_writer.shallow_update(self)
                    except Exception as exc:
                        logger.debug("画像浅巩固跳过: %s", exc)
                    try:
                        if now - ldc >= CONSOLIDATION_DEEP_INTERVAL:
                            self.portrait_writer.deep_update(self)
                    except Exception as exc:
                        logger.debug("画像深巩固跳过: %s", exc)
                except Exception as exc:
                    logger.debug("DMN ticker 异常: %s", exc)
                interval = min(random.expovariate(1.0 / 300), 3600)
                self._stop_event.wait(interval)

        self._dmn_thread = threading.Thread(target=_worker, daemon=True, name=f"dmn_ticker_{self.data_dir}")
        self._dmn_thread.start()

    # ── 入库 ─────────────────────────────────────────────────

    def _store_conversation(self, user_message: str, ai_message: str, timestamp: str):
        """在线程池中并行调用摘要/标签 + embedding。失败自动重试最多3次。"""
        from app.llm.embed import local_embed
        from app.analysis.emotion import analyze_emotion_2d, resolve_emotion_category
        from app.analysis.entity import extract_entities
        _STORE_FAILURES_PATH = f"{self.data_dir}/store_failures.jsonl"
        last_exc = None

        # ── Benchmark 极速路径：只做 embed + 标签 + 写库 ──
        if BENCHMARK_MODE:
            try:
                full_text = f"用户：{user_message}\nAI：{ai_message}"
                summary = (user_message + " | " + ai_message)[:200]
                tags = _extract_noun_tags(user_message) or ["对话", "交流"]
                if len(tags) < 2:
                    tags = list(tags) + (["交流"] if tags[0] != "交流" else ["对话"])
                embedding = local_embed(full_text)
                # 解析原始时间戳，让 LLM 看到真实日期
                ts_float = time.time()
                date_tag = None
                if timestamp:
                    from datetime import datetime as _dt
                    for fmt in ["%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d"]:
                        try:
                            ts_float = _dt.strptime(timestamp.strip(), fmt).timestamp()
                            date_tag = timestamp.strip().split(" ")[0]
                            break
                        except (ValueError, OSError):
                            continue
                memory_id = self.chroma_service.add_memory(
                    user_message=user_message, ai_message=ai_message,
                    summary=summary, tags=tags, embedding=embedding,
                    date_tag=date_tag,
                    source="user",
                )
                # 覆盖 timestamp 为原始日期
                self.chroma_service._collection.update(
                    ids=[memory_id],
                    metadatas=[{"timestamp": ts_float}],
                )
                self.inverted_index.add(memory_id, summary)
                # 同时更新标签倒排索引（benchmark 路径之前漏了这一步）
                tags_str = ",".join(tags) if tags else ""
                self.inverted_index.add_tags(memory_id, tags_str)
                # Phase 2: BM25 已删除，Qdrant MatchText 替代
                return
            except Exception as exc:
                logger.error("benchmark 入库失败: %s", exc)
                return

        for attempt in range(1, 4):
            try:
                full_text = f"用户：{user_message}\nAI：{ai_message}"
                local_llm = _get_local_llm()
                summary = local_llm.summarize(full_text)
                if not summary:
                    summary = (user_message + "：" + ai_message)[:50]
                    if len(user_message + "：" + ai_message) > 50:
                        summary += "…"
                tags = _extract_noun_tags(user_message)
                if not tags:
                    tags = ["对话", "交流"]
                elif len(tags) < 2:
                    tags = list(tags) + (["交流"] if tags[0] != "交流" else ["对话"])
                entities = extract_entities(summary)
                entity_texts = [e["text"] for e in entities if e.get("type") in ("PERSON", "LOCATION", "ORGANIZATION") and len(e["text"]) >= 2]
                tags = list(dict.fromkeys(tags + entity_texts))[:10]
                embedding = local_embed(full_text)
                date_tag = timestamp.split(" ")[0] if timestamp else None
                time_features = None
                if timestamp and ' ' in timestamp:
                    try:
                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        # 时间段标签（统一使用 settings.TIME_PERIOD_MAP）
                        from app.config.settings import TIME_PERIOD_MAP as _tpm
                        h = dt.hour
                        period = "晚上"
                        for (lo, hi), name in _tpm.items():
                            if lo <= h <= hi:
                                period = name
                                break
                        time_features = {
                            "date": dt.strftime("%Y-%m-%d"),
                            "year": dt.year,
                            "month": dt.month,
                            "day": dt.day,
                            "week": dt.isocalendar()[1],
                            "day_of_week": dt.weekday(),
                            "quarter": (dt.month - 1) // 3 + 1,
                            "season": (dt.month % 12 + 3) // 3,   # 1冬2春3夏4秋，与 TemporalPatternIndex 一致
                            "year_month": dt.strftime("%Y-%m"),
                            "time_period": period,
                        }
                        ts_float = dt.timestamp()
                    except (ValueError, OSError):
                        ts_float = None
                        pass
                v2_meta = {**({"date_tag": date_tag} if date_tag else {}), **(time_features or {})}
                valence, arousal, emo_category = analyze_emotion_2d(full_text)
                emotional_intensity = min(
                    full_text.count("！") + full_text.count("!") +
                    len(re.findall(r'[😊😂😭😡😍🥰😢😤🤯💔❤️🔥😅😱🤗]', full_text)),
                    3,
                )
                v2_meta["emotion_valence"] = valence
                v2_meta["emotion_arousal"] = arousal
                v2_meta["emotion_valence_bin"] = emo_category
                v2_meta["emotional_intensity"] = emotional_intensity
                v2_meta["timestamp"] = ts_float if ts_float is not None else datetime.now().timestamp()
                # 注：emotion 结果会被 AI 侧复用（full_text == ai_full_text），避免重复 LLM 调用
                try:
                    if self.chat_history and len(self.chat_history.records) >= 1:
                        prev_ts = self.chat_history.records[-1].get("timestamp", "")
                        if prev_ts and timestamp:
                            gap = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - datetime.strptime(prev_ts, "%Y-%m-%d %H:%M:%S")
                            if gap.total_seconds() < 1800:
                                v2_meta["session_continued"] = True
                except (ValueError, OSError):
                    pass
                memory_id = self.chroma_service.add_memory(
                    user_message=user_message,
                    ai_message=ai_message,
                    summary=summary,
                    tags=tags,
                    embedding=embedding,
                    entities=entities,
                    date_tag=date_tag,
                    time_features=v2_meta,
                    source="user",
                )
                if memory_id and len(entity_texts) >= 2:
                    try:
                        # Phase 3: entity_co_counts 替代 EntityPairTracker
                        # 入库时预计算 entity_co_counts，存入 payload
                        self.chroma_service.update_entity_co_counts(memory_id, entities)
                    except Exception:
                        pass
                    try:
                        self.hyperedge_index.record(entity_texts, memory_id)
                    except Exception:
                        pass

                # AI 侧入库与用户侧并行（不同 ChromaDB，无竞态）
                ai_future = self.storage_executor.submit(
                    self._store_ai_side, user_message, ai_message, timestamp,
                    valence, arousal, emo_category,
                )

                self.chat_history.update_chroma_id(timestamp, memory_id)
                try:
                    if embedding is not None and tags:
                        similar = self.chroma_service._collection.query(
                            query_embeddings=[embedding], n_results=5,
                            include=["metadatas", "distances"],
                        )
                        if similar.get("ids") and similar["ids"][0]:
                            for si, sim_id in enumerate(similar["ids"][0]):
                                if sim_id == memory_id:
                                    continue
                                dist = similar["distances"][0][si]
                                sim_meta = similar["metadatas"][0][si]
                                if dist < 0.08 and not sim_meta.get("stale", False):
                                    old_tags = set(sim_meta.get("tags", "").split(",")) if sim_meta.get("tags") else set()
                                    has_tag_overlap = bool(set(tags) & old_tags)
                                    # 补充：实体重叠也作为冲突信号（语义距离近但标签不同的场景）
                                    has_entity_overlap = False
                                    if not has_tag_overlap:
                                        old_entities_raw = sim_meta.get("entities", [])
                                        if isinstance(old_entities_raw, str):
                                            try:
                                                old_entities_raw = json.loads(old_entities_raw)
                                            except (json.JSONDecodeError, TypeError):
                                                old_entities_raw = []
                                        old_entity_names = {e.get("text", "") for e in (old_entities_raw or []) if isinstance(e, dict)}
                                        new_entity_names = {e.get("text", "") for e in (entities or []) if isinstance(e, dict)}
                                        has_entity_overlap = bool(old_entity_names & new_entity_names)
                                    if has_tag_overlap or has_entity_overlap:
                                        self.chroma_service._collection.update(
                                            ids=[sim_id],
                                            metadatas=[{"stale": True, "superseded_by": memory_id}],
                                        )
                                        logger.info("冲突检测: %s 标记过时, 被 %s 取代 (标签匹配=%s 实体匹配=%s)",
                                                   sim_id[:8], memory_id[:8], has_tag_overlap, has_entity_overlap)
                except Exception as exc:
                    logger.debug("冲突检测跳过: %s", exc)

                try:
                    cur_valence = resolve_emotion_category(v2_meta)
                    if cur_valence in ("positive", "negative") and tags:
                        for t in tags[:3]:
                            if len(t) < 2:
                                continue
                            prev = self.chroma_service._collection.get(
                                where={"tags": {"$contains": t}},
                                include=["metadatas"],
                                limit=10,
                            )
                            if prev.get("ids"):
                                for pi, pid in enumerate(prev["ids"]):
                                    if pid == memory_id:
                                        continue
                                    pm = prev["metadatas"][pi]
                                    pv = resolve_emotion_category(pm)
                                    if pv and pv != cur_valence and pv != "neutral":
                                        rev_path = os.path.join(self.data_dir, "emotional_reversals.jsonl")
                                        os.makedirs(os.path.dirname(rev_path), exist_ok=True)
                                        with open(rev_path, "a", encoding="utf-8") as _rf:
                                            _rf.write(json.dumps({
                                                "tag": t,
                                                "new_memory_id": memory_id,
                                                "old_memory_id": pid,
                                                "new_valence": cur_valence,
                                                "old_valence": pv,
                                                "timestamp": timestamp,
                                            }, ensure_ascii=False) + "\n")
                                        logger.info("情绪反转检测: tag=%s %s→%s", t, pv, cur_valence)
                                        break
                except Exception as exc:
                    logger.debug("情绪反转检测跳过: %s", exc)

                logger.info("记忆入库成功 id=%s summary=%s tags=%s", memory_id[:8], summary, tags)
                if self.chat_history and len(self.chat_history.records) >= 2:
                    try:
                        prev_ts = self.chat_history.records[-2].get("timestamp", "")
                        if prev_ts and timestamp:
                            gap = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - datetime.strptime(prev_ts, "%Y-%m-%d %H:%M:%S")
                            gap_hours = gap.total_seconds() / 3600
                            from app.config.settings import DMN_IDLE_TRIGGER_HOURS
                            if gap_hours > DMN_IDLE_TRIGGER_HOURS:
                                logger.info("DMN: 检测到空闲 %.1f小时，触发级别任务", gap_hours)
                                self.storage_executor.submit(self.dmn.on_idle, gap_hours)
                                self.storage_executor.submit(self.ai_dmn.on_idle, gap_hours)
                                # Phase 4 已完成退役 — 画像系统 (app/portrait/) 替代 DistillEngine
                                self.storage_executor.submit(self._record_ai_co_occurrence)
                                self.storage_executor.submit(self.mirror_neuron.learn_from, self.chat_history.get_records_snapshot())
                                self.storage_executor.submit(self._prewarm_retrieval)
                    except (ValueError, OSError) as exc:
                        logger.debug("DMN 触发检查跳过: %s", exc)
                try:
                    self.chroma_service.mark_storage_complete(memory_id)
                except Exception:
                    pass
                try:
                    self.inverted_index.add(memory_id, summary)
                    tags_str = ",".join(tags) if tags else ""
                    self.inverted_index.add_tags(memory_id, tags_str)
                    if embedding is not None:
                        self.chroma_service._emb_cache_put(memory_id, embedding)
                except Exception:
                    pass
                # AI 侧入库 fire-and-forget：done callback 记录结果，不阻塞 queue worker
                def _ai_store_done(fut):
                    try:
                        fut.result(timeout=0)  # callback 时已就绪
                    except Exception:
                        pass  # AI 侧失败不影响用户侧
                ai_future.add_done_callback(_ai_store_done)

                # Part A: 偏移率检测 (纯规则, <1ms)
                try:
                    if hasattr(self, 'drift_tracker') and self.drift_tracker:
                        self.drift_tracker.detect(user_message)
                except Exception:
                    pass

                return
            except Exception as exc:
                last_exc = exc
                logger.warning("入库失败(第%d/3次): %s", attempt, exc)
                if attempt < 3:
                    time.sleep(3)

        logger.error("入库失败(已放弃,重试3次): %s", last_exc)
        try:
            os.makedirs(os.path.dirname(_STORE_FAILURES_PATH), exist_ok=True)
            with open(_STORE_FAILURES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "user_message": user_message,
                    "ai_message": ai_message,
                    "timestamp": timestamp,
                    "error": str(last_exc),
                }, ensure_ascii=False) + chr(10))
        except Exception:
            pass

    def _store_ai_side(self, user_message: str, ai_message: str, timestamp: str,
                       valence: float, arousal: float, emo_category: str):
        """AI 侧记忆入库（独立于用户侧，与用户侧并行执行）。

        操作独立的 ai_chroma_service，不与用户侧 ChromaDB 竞争。
        失败不影响用户侧，由上层 try/except 捕获。
        """
        from app.llm.embed import local_embed
        from app.analysis.entity import extract_entities
        ai_full_text = f"用户：{user_message}\nAI：{ai_message}"
        ai_summary = _get_local_llm().summarize(f"AI：{ai_message}")
        if not ai_summary:
            ai_summary = ai_message[:50]
        ai_tags = _extract_noun_tags(ai_message)
        if not ai_tags:
            ai_tags = ["AI表达"]
        ai_entities = extract_entities(ai_summary)
        ai_entity_texts = [e["text"] for e in ai_entities
                           if e.get("type") in ("PERSON", "LOCATION", "ORGANIZATION") and len(e["text"]) >= 2]
        ai_tags = list(dict.fromkeys(ai_tags + ai_entity_texts))[:10]
        ai_embedding = local_embed(f"AI：{ai_message}")
        # 复用用户侧情绪分析结果
        ai_valence, ai_arousal, ai_emo_category = valence, arousal, emo_category
        ai_intensity = min(
            ai_full_text.count("！") + ai_full_text.count("!") +
            len(re.findall(r'[😊😂😭😡😍🥰😢😤🤯💔❤️🔥😅😱🤗]', ai_full_text)),
            3,
        )
        from app.config.settings import TIME_PERIOD_MAP as _tpm_ai
        # 从传入的 timestamp 解析真实时间，队列积压时避免时间漂移
        parsed = datetime.now()
        try:
            if timestamp and ' ' in timestamp:
                parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            pass
        h = parsed.hour
        ai_period = "晚上"
        for (lo, hi), name in _tpm_ai.items():
            if lo <= h <= hi:
                ai_period = name
                break
        ai_date_tag = parsed.strftime("%Y-%m-%d")
        ai_time_features = {
            "date": ai_date_tag, "year": parsed.year, "month": parsed.month,
            "day": parsed.day, "week": parsed.isocalendar()[1],
            "day_of_week": parsed.weekday(), "quarter": (parsed.month - 1) // 3 + 1,
            "season": (parsed.month % 12 + 3) // 3, "year_month": parsed.strftime("%Y-%m"),
            "time_period": ai_period,
            "emotion_valence": ai_valence, "emotion_arousal": ai_arousal,
            "emotion_valence_bin": ai_emo_category, "emotional_intensity": ai_intensity,
            "timestamp": parsed.timestamp(),
        }
        try:
            if self.chat_history and len(self.chat_history.records) >= 1:
                prev_ts = self.chat_history.records[-1].get("timestamp", "")
                if prev_ts and timestamp:
                    gap = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - datetime.strptime(prev_ts, "%Y-%m-%d %H:%M:%S")
                    if gap.total_seconds() < 1800:
                        ai_time_features["session_continued"] = True
        except (ValueError, OSError):
            pass
        self.ai_chroma_service.add_memory(
            user_message="[AI]", ai_message=ai_message,
            summary=ai_summary, tags=ai_tags, embedding=ai_embedding,
            entities=ai_entities, date_tag=ai_date_tag,
            time_features=ai_time_features, source="ai",
        )

    # ── 预热 ─────────────────────────────────────────────────

    def _prewarm_retrieval(self):
        """空闲时段预热检索缓存，用户下次发消息时零延迟。"""
        try:
            self.co_tracker._invalidate_cache()
            self.co_tracker._load()
            _ = self.chroma_service.count()
            self.chroma_service._build_embedding_cache()
            self.ai_chroma_service._build_embedding_cache()
            # Phase 4: 构建本地 payload 索引（本地模式补偿）
            self.chroma_service._local_index_build()
            self.ai_chroma_service._local_index_build()
            logger.info("检索预热完成：共现缓存+ChromaDB+embedding缓存+本地索引")
        except Exception as exc:
            logger.debug("检索预热跳过: %s", exc)

    # ── 入队 ─────────────────────────────────────────────────

    def _enqueue_store_task(self, user_message: str, ai_message: str, timestamp: str):
        """写入队列（内存 Queue + 文件持久化兜底）。"""
        task = {
            "user_message": user_message,
            "ai_message": ai_message,
            "timestamp": timestamp,
        }
        # 先入内存队列（即时唤醒 worker）
        self._store_queue.put(task)
        # 再写文件（crash recovery 兜底）
        with self._store_queue_lock:
            qdir = os.path.dirname(self._store_queue_path)
            if qdir:
                os.makedirs(qdir, exist_ok=True)
            with open(self._store_queue_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    # ── AI 巩固 worker ───────────────────────────────────────

    def _start_ai_consolidation_worker(self):
        """Phase 0b: AI 巩固已迁移到 DMN 合并 ticker —
        AI 侧拥有完整 ConsolidationEngine 实例（self.ai_dmn），
        在 _start_dmn_worker 中与用户侧共享 on_idle/浅巩固/深巩固 触发。
        此 worker 仅保留情绪淡化定时器（每小时），其余全部由 ai_dmn 接管。"""
        def _worker():
            logger.info("AI 情绪淡化 worker 已启动 for %s", self.data_dir)
            if not self._stop_event.is_set():
                self._stop_event.wait(60)
            while not self._stop_event.is_set():
                try:
                    self.ai_chroma_service._apply_emotional_desensitization()
                except Exception as exc:
                    logger.debug("AI 情绪淡化跳过: %s", exc)
                self._stop_event.wait(3600)
        self._ai_consolidation_thread = threading.Thread(target=_worker, daemon=True, name=f"ai_desensitize_{self.data_dir}")
        self._ai_consolidation_thread.start()

    def _record_ai_co_occurrence(self):
        """AI 蒸馏后记录 AI 表达共现，积累 AI 人格数据。"""
        try:
            all_data = self.ai_chroma_service._collection.get(
                include=["metadatas"], limit=100,
            )
            ids = all_data.get("ids", [])
            if len(ids) >= 2:
                self.ai_co_tracker.record(ids)
        except Exception:
            pass

    # ── 关闭 ─────────────────────────────────────────────────

    def _cleanup_executors(self):
        """atexit 兜底 — 确保线程池在 crash 时也被关闭。"""
        try:
            if hasattr(self, 'retrieval_executor'):
                self.retrieval_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            if hasattr(self, 'storage_executor'):
                self.storage_executor.shutdown(wait=False)
        except Exception:
            pass

    def close(self):
        """释放所有资源。"""
        self._stop_event.set()
        self.retrieval_executor.shutdown(wait=True)
        self.storage_executor.shutdown(wait=True)
        if self.impulse_scheduler:
            self.impulse_scheduler.stop()
        if self._queue_thread:
            self._queue_thread.join(timeout=5)
        if hasattr(self, '_ai_consolidation_thread') and self._ai_consolidation_thread:
            self._ai_consolidation_thread.join(timeout=3)
        if hasattr(self, '_impulse_consumer_thread') and self._impulse_consumer_thread:
            self._impulse_consumer_thread.join(timeout=5)
        if hasattr(self, '_dmn_thread') and self._dmn_thread:
            self._dmn_thread.join(timeout=5)
        if hasattr(self, '_consolidation_thread') and self._consolidation_thread:
            self._consolidation_thread.join(timeout=5)
        # 关闭 LLM Client 的 httpx 客户端
        if hasattr(self, 'llm_client') and self.llm_client:
            try:
                _loop = asyncio.get_running_loop()
                _loop.create_task(self.llm_client.aclose())
            except RuntimeError:
                try:
                    asyncio.run(self.llm_client.aclose())
                except Exception:
                    pass
            except Exception:
                pass
        # 关闭 impulse_httpx 客户端
        try:
            _loop = asyncio.get_running_loop()
            _loop.create_task(_impulse_httpx.aclose())
        except RuntimeError:
            try:
                asyncio.run(_impulse_httpx.aclose())
            except Exception:
                pass
        except Exception:
            pass
        # Phase 3: SQLite 连接已迁 Qdrant，不再需要 close_all
        # 关闭 Qdrant 客户端
        try:
            self.chroma_service.close()
            self.ai_chroma_service.close()
        except Exception:
            pass


# ── ctx_manager 导出 ──────────────────────────────────────────
from app.core.user_context import ctx_manager
