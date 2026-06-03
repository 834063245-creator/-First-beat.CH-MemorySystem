"""应用上下文 — AppContext 类 + ctx_manager 导出。

迁移里程碑：不再从 backend/main.py 导入 AppContext。
AppContext 类在此处完整定义，底层模块仍可从 backend/ 导入。
"""
import asyncio
import queue
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── 确保 backend/ 可导入（底层模块仍在 backend/ 中） ─────────
_backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_backend_dir))

# ── 底层模块导入（后续逐步迁移到 app/） ──────────────────────
import jieba.posseg as pseg  # 降级兜底

# 话题 CNN 分类器（惰性单例）
_TOPIC_CLASSIFIER = None
_TC_INIT_ATTEMPTED = False
_TC_LOCK = threading.Lock()


def _get_topic_classifier():
    """惰性获取话题分类器，失败返回 None（线程安全）。"""
    global _TOPIC_CLASSIFIER, _TC_INIT_ATTEMPTED
    if _TC_INIT_ATTEMPTED:
        return _TOPIC_CLASSIFIER
    with _TC_LOCK:
        if _TC_INIT_ATTEMPTED:
            return _TOPIC_CLASSIFIER
        _TC_INIT_ATTEMPTED = True
        try:
            from app.brain.topic_classifier import get_topic_classifier
            _TOPIC_CLASSIFIER = get_topic_classifier()
            if _TOPIC_CLASSIFIER and _TOPIC_CLASSIFIER.available:
                print("[TopicCNN] 话题分类器已启用（替代 jieba）")
                return _TOPIC_CLASSIFIER
        except Exception:
            logger.debug("话题分类器加载失败，降级为 jieba", exc_info=True)
    return None


from app.config.settings import (                  # noqa: E402
    CHROMA_PERSIST_DIR, DATA_DIR, DEFAULT_TOP_K, MAX_MEMORIES_IN_PROMPT,
    TIMELINE_RECENT_COUNT, WORK_MEMORY_TOKEN_BUDGET,
    CHAT_HISTORY_PATH, CHAT_HISTORY_MAX_MEMORY, DEBUG_INCLUDE_PROMPT,
    STORE_FAILURES_PATH, BEHAVIOR_CHROMA_DIR, BEHAVIOR_COLLECTION,
    CONSOLIDATION_SHALLOW_INTERVAL, CONSOLIDATION_DEEP_INTERVAL,
    AI_CHROMA_DIR, AI_COLLECTION, AI_DISTILL_STATE_PATH,
    IS_LITE, LITE_DISABLE_BACKGROUND_TASKS, LITE_DISABLE_IMPULSE,
    LITE_WORK_MEMORY_BUDGET, USER_DATA_DIRS,
    STOP_WORDS as _STOP_WORDS,
)
from memory import ChromaService                   # noqa: E402
from llm import DeepSeekLLM                        # noqa: E402
from retrieval import CoOccurrenceTracker           # noqa: E402
from app.memory.entity_pair import EntityPairTracker  # noqa: E402
from personality_store import PersonalityStore       # noqa: E402
from behavior_store import BehaviorStore             # noqa: E402
from chat_history import ChatHistory                 # noqa: E402
from inverted_index import InvertedIndex             # noqa: E402
from topic_affinity import TopicAffinity             # noqa: E402
from temporal_pattern import TemporalPatternIndex    # noqa: E402
from distill import DistillEngine                    # noqa: E402
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
                from local_llm import LocalLLM
                _LOCAL_LLM = LocalLLM()
    return _LOCAL_LLM


def _extract_noun_tags(text: str, topk: int = 8) -> list[str]:
    """提取标签：CNN 话题分类优先，jieba 降级兜底。"""
    tc = _get_topic_classifier()
    if tc and tc.available:
        tags = tc.predict(text, top_k=5)
        if tags:
            return tags
    # 降级：jieba posseg
    words = pseg.cut(text)
    nouns = []
    for w, flag in words:
        if flag.startswith(("n", "vn")) and len(w) >= 2 and w.lower() not in _STOP_WORDS:
            nouns.append(w)
    seen = set()
    unique = []
    for w in nouns:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:topk]


# ===================================================================
# AppContext — 所有服务实例的容器
# ===================================================================

class AppContext:
    """应用服务容器。初始化顺序反映依赖关系。"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chroma_service = ChromaService(
            persist_dir=f"{data_dir}/chroma",
        )
        self.ai_chroma_service = ChromaService(
            persist_dir=f"{data_dir}/ai_chroma",
            collection_name=AI_COLLECTION,
        )
        self.deepseek_llm = DeepSeekLLM()
        self.storage_executor = ThreadPoolExecutor(max_workers=5)
        self.retrieval_executor = ThreadPoolExecutor(max_workers=3)
        import atexit
        atexit.register(self._cleanup_executors)
        self.co_tracker = CoOccurrenceTracker(file_path=f"{data_dir}/co_occurrence.json")
        self.ai_co_tracker = CoOccurrenceTracker(file_path=f"{data_dir}/ai_co_occurrence.json")
        self.entity_pair_tracker = EntityPairTracker(file_path=f"{data_dir}/entity_pairs.json")
        self.personality_store = PersonalityStore(persist_dir=f"{data_dir}/personality_chroma")
        self.behavior_store = BehaviorStore(persist_dir=f"{data_dir}/behavior_chroma", collection_name=BEHAVIOR_COLLECTION)
        self.chat_history = ChatHistory(path=f"{data_dir}/chat_history.jsonl", max_memory=CHAT_HISTORY_MAX_MEMORY)
        self.inverted_index = InvertedIndex()
        try:
            _all = self.chroma_service.list_all()
            _tag_entries = [(m['id'], (m.get('metadata') or {}).get('tags', '') or '') for m in _all]
            self.inverted_index.build_tags(_tag_entries)
            logger.info('标签索引构建完成: %d 个标签', len(self.inverted_index._tag_index))
        except Exception:
            pass

        self.topic_affinity = TopicAffinity(data_dir=data_dir)
        self.temporal_pattern_index = TemporalPatternIndex(data_dir=data_dir)
        self.user_distill = DistillEngine(
            self.personality_store, self.chroma_service,
            behavior_store=self.behavior_store,
            state_path=f"{data_dir}/distill_state.json",
            source="user",
        )
        self.ai_distill = DistillEngine(
            self.personality_store, self.ai_chroma_service,
            state_path=f"{data_dir}/ai_distill_state.json",
            source="ai",
        )
        self.distill_engine = self.user_distill  # 向后兼容
        if not (IS_LITE and LITE_DISABLE_BACKGROUND_TASKS):
            from consolidation import ConsolidationEngine
            self.dmn = ConsolidationEngine(
                chroma_service=self.chroma_service,
                personality_store=self.personality_store,
                behavior_store=self.behavior_store,
                chat_history=self.chat_history,
                co_tracker=self.co_tracker,
                state_path=f"{data_dir}/dmn_state.json",
                notes_path=f"{data_dir}/topic_notes.json",
                temporal_pattern_index=self.temporal_pattern_index,
                topic_affinity=self.topic_affinity,
            )
        else:
            self.dmn = None

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
        if hasattr(self, 'deepseek_llm'):
            self.deepseek_llm.set_pattern_discovery(self._pattern_discovery)

        if not (IS_LITE and LITE_DISABLE_IMPULSE):
            from impulse import ImpulseScheduler
            self.impulse_scheduler = ImpulseScheduler(
                state_path=f"{data_dir}/impulse_state.json",
                temporal_pattern_index=self.temporal_pattern_index,
            )
        else:
            self.impulse_scheduler = None
        self.mirror_neuron = BehaviorPredictor(data_dir=data_dir)

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
            self._start_consolidation_worker()
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
            while not self._stop_event.is_set():
                try:
                    tasks = []
                    with self._store_queue_lock:
                        if os.path.exists(self._store_queue_path):
                            with open(self._store_queue_path, "r", encoding="utf-8") as f:
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

                    if tasks:
                        for i, task in enumerate(tasks):
                            logger.info("队列任务 [%d/%d]: 入库 %s...", i + 1, len(tasks), task.get("timestamp", "?")[:16])
                            try:
                                self._store_conversation(
                                    task["user_message"], task["ai_message"], task["timestamp"]
                                )
                                logger.info("队列任务 [%d/%d]: 完成", i + 1, len(tasks))
                            except Exception as e:
                                logger.error("队列任务 [%d/%d] 失败: %s", i + 1, len(tasks), e)
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
                behavior_store=self.behavior_store,
                chat_history=self.chat_history,
                personality_store=self.personality_store,
            )

    # ── DMN worker ───────────────────────────────────────────

    def _start_dmn_worker(self):
        """DMN 泊松检查：平均每 5 分钟随机检查一次空闲状态。"""
        def _worker():
            logger.info("DMN 泊松 worker 已启动 for %s", self.data_dir)
            while not self._stop_event.is_set():
                try:
                    if self.chat_history and len(self.chat_history.records) >= 1:
                        last_rec = self.chat_history.records[-1]
                        last_ts = last_rec.get("timestamp", "")
                        if last_ts:
                            gap = datetime.now() - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                            gap_hours = gap.total_seconds() / 3600
                            if gap_hours > 0.5:
                                self.dmn.on_idle(gap_hours)
                except Exception as exc:
                    logger.debug("DMN 定时检查跳过: %s", exc)
                interval = min(random.expovariate(1.0 / 300), 3600)
                self._stop_event.wait(interval)

        self._dmn_thread = threading.Thread(target=_worker, daemon=True, name=f"dmn_worker_{self.data_dir}")
        self._dmn_thread.start()

    # ── 巩固节律 worker ──────────────────────────────────────

    def _start_consolidation_worker(self):
        """按固定间隔触发 DMN 巩固，与用户活跃与否无关。"""
        def _worker():
            logger.info("巩固节律 worker 已启动 for %s", self.data_dir)
            while not self._stop_event.is_set():
                try:
                    lsc = 0
                    ldc = 0
                    try:
                        from consolidation import _load_state as _dmn_load
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
                except Exception as exc:
                    logger.debug("巩固节律循环异常: %s", exc)
                interval = min(random.expovariate(1.0 / 300), 3600)
                self._stop_event.wait(interval)

        self._consolidation_thread = threading.Thread(target=_worker, daemon=True, name=f"consolidation_{self.data_dir}")
        self._consolidation_thread.start()

    # ── 入库 ─────────────────────────────────────────────────

    def _store_conversation(self, user_message: str, ai_message: str, timestamp: str):
        """在线程池中并行调用摘要/标签 + embedding。失败自动重试最多3次。"""
        from local_embed import local_embed
        from emotion import analyze_emotion_2d, resolve_emotion_category
        from entity_extractor import extract_entities
        _STORE_FAILURES_PATH = f"{self.data_dir}/store_failures.jsonl"
        last_exc = None

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
                    tags = ["对话"]
                entities = extract_entities(summary)
                entity_texts = [e["text"] for e in entities if e.get("type") in ("PERSON", "LOCATION", "ORGANIZATION") and len(e["text"]) >= 2]
                tags = list(dict.fromkeys(tags + entity_texts))[:10]
                embedding = local_embed(full_text)
                date_tag = timestamp.split(" ")[0] if timestamp else None
                time_features = None
                if timestamp and ' ' in timestamp:
                    try:
                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        # 时间段标签（与 TemporalPatternIndex._current_bucket 保持一致）
                        h = dt.hour
                        if h < 6:
                            period = "深夜"
                        elif h < 9:
                            period = "早晨"
                        elif h < 12:
                            period = "上午"
                        elif h < 14:
                            period = "中午"
                        elif h < 17:
                            period = "下午"
                        elif h < 21:
                            period = "傍晚"
                        else:
                            period = "晚上"
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
                    except (ValueError, OSError):
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
                v2_meta["timestamp"] = datetime.now().timestamp()
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
                        for i in range(len(entity_texts)):
                            for j in range(i + 1, len(entity_texts)):
                                self.entity_pair_tracker.record(entity_texts[i], entity_texts[j], memory_id)
                    except Exception:
                        pass
                try:
                    ai_summary = _get_local_llm().summarize(f"AI：{ai_message}")
                    if not ai_summary:
                        ai_summary = ai_message[:50]
                    ai_tags = _extract_noun_tags(ai_message)
                    if not ai_tags:
                        ai_tags = ["AI表达"]
                    ai_embedding = local_embed(f"AI：{ai_message}")
                    ai_valence, ai_arousal, ai_emo_category = analyze_emotion_2d(ai_message)
                    ai_intensity = min(
                        ai_message.count("！") + ai_message.count("!") +
                        len(re.findall(r'[😊😂😭😡😍🥰😢😤🤯💔❤️🔥😅😱🤗]', ai_message)),
                        3,
                    )
                    ai_meta = {
                        "emotion_valence": ai_valence,
                        "emotion_arousal": ai_arousal,
                        "emotion_valence_bin": ai_emo_category,
                        "emotional_intensity": ai_intensity,
                        "timestamp": datetime.now().timestamp(),
                    }
                    self.ai_chroma_service.add_memory(
                        user_message="[AI]", ai_message=ai_message,
                        summary=ai_summary, tags=ai_tags, embedding=ai_embedding,
                        time_features=ai_meta,
                        source="ai",
                    )
                except Exception as ai_exc:
                    logger.warning("AI 侧入库失败（不影响用户侧）: %s", ai_exc)
                self.chat_history.update_chroma_id(timestamp, memory_id)
                try:
                    if embedding is not None and tags:
                        similar = self.chroma_service._read_collection.query(
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
                                    if set(tags) & old_tags:
                                        self.chroma_service._write_collection.update(
                                            ids=[sim_id],
                                            metadatas=[{"stale": True, "superseded_by": memory_id}],
                                        )
                                        logger.info("冲突检测: %s 标记过时, 被 %s 取代", sim_id[:8], memory_id[:8])
                except Exception as exc:
                    logger.debug("冲突检测跳过: %s", exc)

                try:
                    cur_valence = resolve_emotion_category(v2_meta)
                    if cur_valence in ("positive", "negative") and tags:
                        for t in tags[:3]:
                            if len(t) < 2:
                                continue
                            prev = self.chroma_service._read_collection.get(
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
                            from config import DISTILL_IDLE_HOURS
                            if gap_hours > DISTILL_IDLE_HOURS:
                                logger.info("DMN: 检测到空闲 %.1f小时，触发级别任务", gap_hours)
                                self.storage_executor.submit(self.dmn.on_idle, gap_hours)
                                existing = self.personality_store.list_tags(page=1, page_size=100)
                                self.storage_executor.submit(self.distill_engine.run_distill, existing_tags=existing.get("items", []))
                                self.storage_executor.submit(self.ai_distill.run_distill, existing_tags=existing.get("items", []))
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
                    if embedding is not None:
                        with self.chroma_service._emb_cache_lock:
                            self.chroma_service._emb_cache[memory_id] = embedding
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

    # ── 预热 ─────────────────────────────────────────────────

    def _prewarm_retrieval(self):
        """空闲时段预热检索缓存，用户下次发消息时零延迟。"""
        try:
            self.co_tracker._invalidate_cache()
            self.co_tracker._load()
            self.entity_pair_tracker._invalidate_cache()
            tags = self.personality_store.list_tags(page=1, page_size=100)
            _ = self.chroma_service.count()
            self.chroma_service._build_embedding_cache()
            self.ai_chroma_service._build_embedding_cache()
            logger.info("检索预热完成：共现缓存+人格库+ChromaDB+embedding缓存")
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
        """AI 表达模式巩固：定时分析最近 AI 回复，检测表达习惯变化。"""
        def _worker():
            logger.info("AI 巩固 worker 已启动")
            while not self._stop_event.is_set():
                try:
                    all_data = self.ai_chroma_service._read_collection.get(
                        include=["metadatas"], limit=200,
                    )
                    metas = [dict(m) for m in (all_data.get("metadatas") or []) if m]
                    if len(metas) >= 10:
                        from emotion import resolve_emotion_category
                        valences = Counter(resolve_emotion_category(m) for m in metas)
                        intensities = [m.get("emotional_intensity", 0) or 0 for m in metas]
                        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
                        dominant_emotion = valences.most_common(1)[0][0]
                        total = sum(valences.values()) or 1
                        emotion_pct = {k: round(v/total*100) for k, v in valences.items()}
                        logger.debug("AI 表达状态: 情绪=%s 分布=%s 平均强度=%.2f",
                                     dominant_emotion, emotion_pct, avg_intensity)
                except Exception as exc:
                    logger.debug("AI 巩固循环跳过: %s", exc)
                self._stop_event.wait(3600)
        self._ai_consolidation_thread = threading.Thread(target=_worker, daemon=True, name=f"ai_consolidation_{self.data_dir}")
        self._ai_consolidation_thread.start()

    def _record_ai_co_occurrence(self):
        """AI 蒸馏后记录 AI 表达共现，积累 AI 人格数据。"""
        try:
            all_data = self.ai_chroma_service._read_collection.get(
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
        if hasattr(self, '_dmn_thread') and self._dmn_thread:
            self._dmn_thread.join(timeout=5)
        if hasattr(self, '_consolidation_thread') and self._consolidation_thread:
            self._consolidation_thread.join(timeout=5)
        # 关闭 DeepSeekLLM 的 httpx 客户端
        if hasattr(self, 'deepseek_llm') and self.deepseek_llm:
            try:
                _loop = asyncio.get_running_loop()
                _loop.create_task(self.deepseek_llm.aclose())
            except RuntimeError:
                try:
                    asyncio.run(self.deepseek_llm.aclose())
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


# ── ctx_manager 导出 ──────────────────────────────────────────
# backend/user_context.py 现在从 app.core.context 导入 AppContext，
# 不再依赖 backend/main.py。
from user_context import ctx_manager  # noqa: E402, F401
