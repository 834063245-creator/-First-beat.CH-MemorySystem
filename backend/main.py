"""FastAPI 入口 — 记忆服务 Phase 1."""
import asyncio
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import Counter
import bottleneck

import jieba
import jieba.analyse
import jieba.posseg as pseg
import httpx

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from emotion import analyze_emotion, analyze_emotion_2d, resolve_emotion_category
from models import ChatRequest, ChatResponse, DebugInfo, TraceItem, PromptBody, CorrectMemoryBody


from entity_extractor import extract_entities
from knowledge_base import KnowledgeBase
from llm import DeepSeekLLM, now_hint, parse_dsml_tool_calls, strip_dsml
from local_embed import local_embed, local_embed_async, local_embed_batch
from openai_adapter import parse_openai_messages, format_openai_chunk, format_openai_response
from local_llm import LocalLLM
from memory import ChromaService
from search import search_web
from workspace import read_file, list_files, grep_files
from retrieval import CoOccurrenceTracker
from app.memory.entity_pair import EntityPairTracker
from metadata import build_memory_metadata
from config import CHROMA_PERSIST_DIR, DATA_DIR, DEFAULT_TOP_K, MAX_MEMORIES_IN_PROMPT, KNOWLEDGE_COLLECTION
from config import TIMELINE_RECENT_COUNT, WORK_MEMORY_TOKEN_BUDGET, CHAT_HISTORY_PATH, CHAT_HISTORY_MAX_MEMORY, DEBUG_INCLUDE_PROMPT, STORE_FAILURES_PATH
from config import BEHAVIOR_CHROMA_DIR, BEHAVIOR_COLLECTION
from config import CONSOLIDATION_SHALLOW_INTERVAL, CONSOLIDATION_DEEP_INTERVAL
from config import AI_CHROMA_DIR, AI_COLLECTION, AI_DISTILL_STATE_PATH
from personality_store import PersonalityStore
from behavior_store import BehaviorStore
from distill import DistillEngine
from chat_history import ChatHistory
from impulse import ImpulseScheduler
from impulse import source_emotion_trend, source_time_rhythm, source_random_roam, source_curiosity, source_behavior_pattern
from config import IS_LITE, LITE_DISABLE_BACKGROUND_TASKS, LITE_DISABLE_IMPULSE, LITE_WORK_MEMORY_BUDGET
from inverted_index import InvertedIndex
from circuit import CircuitOrchestrator
from app.analysis.predictor import BehaviorPredictor
from user_context import ctx_manager
from config import USER_DATA_DIRS, AUTH_TOKEN_PATH
from app.tools.atomic import atomic_write
from app.retrieval.pipeline import run_chat_retrieval as _run_chat_retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 压制 httpx 的 HTTP 请求日志（每调一次 Ollama embedding 刷一行）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# 压制 uvicorn 的访问日志（INFO 就是每行 HTTP 请求）
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# 压制 jieba 的启动日志（jieba 混用 print 和 logging，双管齐下）
logging.getLogger("jieba").setLevel(logging.WARNING)
import sys as _sys
import jieba as _jieba
_jieba.default_logger.setLevel(logging.WARNING)
# 重定向 jieba 的 print 到 devnull
_old_stdout = _sys.stdout
_sys.stdout = __import__('io').StringIO()
_jieba.initialize()
_sys.stdout = _old_stdout

# 共享 httpx 客户端（复用连接，避免每次请求新建）
_impulse_httpx = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

# 纠正反馈日志路径（供检索排序降权/升权用，由 per-user data_dir 决定）
_correction_lock = threading.Lock()

# 用户打字心跳（供自主触发系统判断空闲用）
_last_heartbeat_time: float | None = None
_heartbeat_lock = threading.Lock()

from config import STOP_WORDS as _STOP_WORDS

# ------------------------------------------------------------
# 搜索工具定义（OpenAI 兼容格式）
# ------------------------------------------------------------
SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "实时搜索互联网获取最新信息。当你需要回答关于时事、实时数据、具体事实或任何你不确定的信息时，调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，尽量具体",
                },
            },
            "required": ["query"],
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取工作区文件的内容。当用户要求你「读一下某个文件」时调用。返回文件文本内容，只读不写。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径，可以是绝对路径或相对路径。例如 'backend/main.py' 或 'D:/amazing2/backend/main.py'",
                },
            },
            "required": ["path"],
        },
    },
}

LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "按 glob 模式列出工作区中的文件。支持通配符：* 匹配任意字符，** 递归目录。例如 '*.py'、'backend/**/*.txt'。当你想了解项目结构或找某个文件时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 匹配模式，如 'backend/*.py'、'**/*.md'、'backend/scripts/*.py'",
                },
            },
            "required": ["pattern"],
        },
    },
}

GREP_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "grep_files",
        "description": "在项目文件中搜索文本或正则表达式。当用户问「哪里定义了 xxx」「找一下包含 xxx 的文件」时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "要搜索的文本或正则表达式",
                },
                "glob_pattern": {
                    "type": "string",
                    "description": "文件过滤 glob 模式，默认 '**/*.py'，也支持 '**/*.md'、'**/*.{py,txt}' 等",
                },
            },
            "required": ["pattern"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "写入或覆写文件。如果要修改已有文件的部分内容，用 edit_file。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["path", "content"],
        },
    },
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "精确字符串替换——在文件中找到 old_str 并替换为 new_str。不改动文件其他部分。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_str": {"type": "string", "description": "要被替换的精确文本"},
                "new_str": {"type": "string", "description": "替换后的文本"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 shell 命令。用于运行脚本、编译代码、启动服务等。返回 stdout+stderr。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
            },
            "required": ["command"],
        },
    },
}

GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "glob",
        "description": "按 glob 模式搜索文件路径。例如 '**/*.py' 找所有 Python 文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob 匹配模式"},
                "root": {"type": "string", "description": "搜索根目录，默认当前项目"},
            },
            "required": ["pattern"],
        },
    },
}

# ------------------------------------------------------------
# 知识库模式（前置定义，AppContext 初始化需要）
# ------------------------------------------------------------
def _load_knowledge_mode(data_dir: str = DATA_DIR) -> bool:
    path = os.path.join(data_dir, "knowledge_mode.json")
    try:
        with open(path) as f:
            return json.load(f).get("enabled", False)
    except Exception:
        return False


def _save_knowledge_mode(enabled: bool, data_dir: str = DATA_DIR):
    path = os.path.join(data_dir, "knowledge_mode.json")
    atomic_write(path, {"enabled": enabled})


# ------------------------------------------------------------
# 应用上下文 — 所有服务实例的容器
# ------------------------------------------------------------

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
        from topic_affinity import TopicAffinity
        from temporal_pattern import TemporalPatternIndex
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

        # 模式发现层（零 LLM 调用，纯统计缓存）
        from app.analysis.pattern_discovery import PatternDiscovery
        self._pattern_discovery = PatternDiscovery(
            data_dir=data_dir,
            temporal_index=self.temporal_pattern_index,
            affinity=self.topic_affinity,
            chat_history_path=f"{data_dir}/chat_history.jsonl",
        )
        self._pattern_discovery.load_cache()
        if hasattr(self, 'deepseek_llm'):
            self.deepseek_llm.set_pattern_discovery(self._pattern_discovery)
        self.kb = KnowledgeBase(
            chroma_dir=f"{data_dir}/chroma",
            collection_name=KNOWLEDGE_COLLECTION,
            state_path=f"{data_dir}/knowledge_state.json",
        )
        self.knowledge_mode_enabled = _load_knowledge_mode(data_dir=data_dir)
        if not (IS_LITE and LITE_DISABLE_IMPULSE):
            from impulse import ImpulseScheduler
            self.impulse_scheduler = ImpulseScheduler(state_path=f"{data_dir}/impulse_state.json",
                                                       temporal_pattern_index=self.temporal_pattern_index)
        else:
            self.impulse_scheduler = None
        self.mirror_neuron = BehaviorPredictor(data_dir=data_dir)

        # 每个用户的存储队列路径
        self._store_queue_path = f"{data_dir}/store_queue.jsonl"
        self._store_queue_lock = threading.Lock()
        self._store_queue_event = threading.Event()
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

                    self._store_queue_event.clear()
                    self._store_queue_event.wait(timeout=1)
                except Exception as e:
                    logger.error("队列 worker 循环异常: %s", e)
                    self._store_queue_event.wait(timeout=1)

        t = threading.Thread(target=_worker, daemon=True, name=f"store_queue_{self.data_dir}")
        t.start()
        self._queue_thread = t

    def _start_impulse_workers(self):
        """启动用户专属冲动源泊松线程。"""
        if self.impulse_scheduler:
            self.impulse_scheduler.start_source_workers(
                chroma_service=self.chroma_service,
                behavior_store=self.behavior_store,
                chat_history=self.chat_history,
                personality_store=self.personality_store,
            )

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
                import random as _random
                interval = min(_random.expovariate(1.0 / 300), 3600)
                self._stop_event.wait(interval)

        self._dmn_thread = threading.Thread(target=_worker, daemon=True, name=f"dmn_worker_{self.data_dir}")
        self._dmn_thread.start()

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
                        # 浅巩固完成后触发模式发现（零 LLM，纯统计）
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
                import random as _random
                interval = min(_random.expovariate(1.0 / 300), 3600)
                self._stop_event.wait(interval)

        self._consolidation_thread = threading.Thread(target=_worker, daemon=True, name=f"consolidation_{self.data_dir}")
        self._consolidation_thread.start()

    def _store_conversation(self, user_message: str, ai_message: str, timestamp: str):
        """在线程池中并行调用摘要/标签 + embedding。失败自动重试最多3次。"""
        from local_embed import local_embed
        from datetime import datetime
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
                        time_features = {
                            "date": dt.strftime("%Y-%m-%d"),
                            "year": dt.year,
                            "month": dt.month,
                            "day": dt.day,
                            "week": dt.isocalendar()[1],
                            "day_of_week": dt.weekday(),
                            "quarter": (dt.month - 1) // 3 + 1,
                            "season": (dt.month % 12 + 3) // 3,
                            "year_month": dt.strftime("%Y-%m"),
                        }
                    except (ValueError, OSError):
                        logger.warning("时间特征解析失败: %s", timestamp)
                v2_meta = build_memory_metadata(user_message, ai_message, timestamp)
                emotional_score = 0
                full_msg = user_message + ai_message
                exclamation_count = full_msg.count("！") + full_msg.count("!")
                if exclamation_count >= 3:
                    emotional_score += 1
                if exclamation_count >= 6:
                    emotional_score += 1
                emoji_pattern = r'[😊😂😭😡😍🥰😢😤🤯💔❤️🔥😅😱🤗]'
                emoji_matches = len(re.findall(emoji_pattern, full_msg))
                if emoji_matches >= 2:
                    emotional_score += 1
                if emoji_matches >= 5:
                    emotional_score += 1
                if len(user_message) > 200:
                    emotional_score += 1
                # 用户侧情绪分析（Russell 二维坐标）
                valence, arousal, emo_category = analyze_emotion_2d(user_message)
                v2_meta["emotion_valence"] = valence
                v2_meta["emotion_arousal"] = arousal
                v2_meta["emotion_valence_bin"] = emo_category
                v2_meta["emotional_intensity"] = min(emotional_score, 3)
                if time_features:
                    v2_meta.update(time_features)
                if self.chat_history and len(self.chat_history.records) >= 2:
                    last = self.chat_history.records[-2]
                    try:
                        last_ts = last.get("timestamp", "")
                        if last_ts and timestamp:
                            gap = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
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
                # 实体共现记录：同一段对话中出现的实体对
                if memory_id and len(entity_texts) >= 2:
                    try:
                        for i in range(len(entity_texts)):
                            for j in range(i + 1, len(entity_texts)):
                                self.entity_pair_tracker.record(entity_texts[i], entity_texts[j], memory_id)
                    except Exception:
                        pass
                # AI 侧写入（存 AI 回复原文用于自我表达习惯蒸馏）
                try:
                    ai_summary = local_llm.summarize(f"AI：{ai_message}")
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

    def _prewarm_retrieval(self):
        """空闲时段预热检索缓存，用户下次发消息时零延迟。"""
        try:
            self.co_tracker._invalidate_cache()
            self.co_tracker._load()
            self.entity_pair_tracker._invalidate_cache()
            tags = self.personality_store.list_tags(page=1, page_size=100)
            _ = self.chroma_service.count()
            # ATTENTION 缓存预热：构建 embedding cache 供注意力漂移 reranker 用
            self.chroma_service._build_embedding_cache()
            self.ai_chroma_service._build_embedding_cache()
            logger.info("检索预热完成：共现缓存+人格库+ChromaDB+embedding缓存")
        except Exception as exc:
            logger.debug("检索预热跳过: %s", exc)

    def _enqueue_store_task(self, user_message: str, ai_message: str, timestamp: str):
        """同步写入队列文件，微秒级，不卡顿。"""
        task = {
            "user_message": user_message,
            "ai_message": ai_message,
            "timestamp": timestamp,
        }
        with self._store_queue_lock:
            qdir = os.path.dirname(self._store_queue_path)
            if qdir:
                os.makedirs(qdir, exist_ok=True)
            with open(self._store_queue_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        self._store_queue_event.set()

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
                import random as _random
                self._stop_event.wait(3600)  # 每小时检查一次
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
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.deepseek_llm.aclose())
                    elif not loop.is_closed():
                        loop.run_until_complete(self.deepseek_llm.aclose())
                except RuntimeError:
                    pass
            except Exception:
                pass
        try:
            self.kb._index = None
        except Exception:
            pass
        # 关闭主模块的 impulse_httpx 客户端
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_impulse_httpx.aclose())
                elif not loop.is_closed():
                    loop.run_until_complete(_impulse_httpx.aclose())
            except RuntimeError:
                pass
        except Exception:
            pass


# ctx 已改为 ctx_manager (user_context.py) — 按用户懒初始化

# ------------------------------------------------------------
# 认证基础设施（内测用）
# ------------------------------------------------------------
import secrets as _secrets
from config import USERS as _USERS

_AUTH_TOKENS: dict[str, dict] = {}  # token → {"username": str, "expires": float}
_AUTH_LOCK = threading.Lock()


def _load_auth_tokens():
    """从磁盘加载持久化的 token 表。"""
    path = AUTH_TOKEN_PATH
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_auth_tokens():
    """将 token 表持久化到磁盘。"""
    atomic_write(AUTH_TOKEN_PATH, _AUTH_TOKENS)


# 启动时加载
_AUTH_TOKENS = _load_auth_tokens()


def _authenticate(username: str, password: str) -> str | None:
    """验证用户名密码，返回 token。失败返回 None。"""
    expected = _USERS.get(username)
    if expected is None or not _secrets.compare_digest(expected, password):
        return None
    token = _secrets.token_urlsafe(32)
    with _AUTH_LOCK:
        _AUTH_TOKENS[token] = {
            "username": username,
            "created": time.time(),
            "expires": time.time() + 604800,  # 7 天过期
        }
        _save_auth_tokens()
    return token


async def get_current_user(credentials: str = Depends(HTTPBearer(auto_error=False))) -> str:
    """从 Bearer token 获取当前用户名。"""
    if not credentials:
        raise HTTPException(status_code=401, detail="未认证")
    with _AUTH_LOCK:
        entry = _AUTH_TOKENS.get(credentials.credentials)
        if entry is None:
            raise HTTPException(status_code=401, detail="token 无效或已过期")
        if entry.get("expires", 0) < time.time():
            del _AUTH_TOKENS[credentials.credentials]
            _save_auth_tokens()
            raise HTTPException(status_code=401, detail="token 已过期")
        return entry["username"]


async def get_user_context(user: str = Depends(get_current_user)) -> AppContext:
    """依赖注入：从当前用户获取其专属 AppContext。"""
    data_dir = USER_DATA_DIRS.get(user)
    if not data_dir:
        data_dir = DATA_DIR
    return ctx_manager.get_context(user, data_dir)


def _extract_noun_tags(text: str, topk: int = 8) -> list[str]:
    """用 jieba.posseg 提取名词性词语作为标签。"""
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



# ------------------------------------------------------------
# 入库（模块级共享函数 — 被 AppContext._store_conversation 调用）
# ------------------------------------------------------------
_LOCAL_LLM: "LocalLLM | None" = None
_LOCAL_LLM_LOCK = threading.Lock()
def _get_local_llm() -> "LocalLLM":
    global _LOCAL_LLM
    if _LOCAL_LLM is None:
        with _LOCAL_LLM_LOCK:
            if _LOCAL_LLM is None:
                _LOCAL_LLM = LocalLLM()
    return _LOCAL_LLM


# 后台 worker 已移至 AppContext 内部管理 — 每个用户实例独自运行


# ------------------------------------------------------------
# 溯源 trace 构建
# ------------------------------------------------------------
def _build_trace(memories: list) -> list[dict]:
    """从检索结果中提取 trace 数据，响应式传递给前端。"""
    trace = []
    for m in memories:
        meta = m.get("metadata", {})
        raw_tags = meta.get("tags", "")
        if isinstance(raw_tags, str):
            tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
        else:
            tags_list = list(raw_tags) if raw_tags else []
        trace.append({
            "id": m["id"],
            "summary": meta.get("summary", ""),
            "timestamp": meta.get("timestamp", 0),
            "source": m.get("source", ""),
            "display_source": m.get("display_source", ""),
            "hit_count": meta.get("hit_count", 0),
            "tags": tags_list,
        })
    return trace


# ------------------------------------------------------------
# Debug info 构建
# ------------------------------------------------------------
def _build_debug_info(memories: list, personalities: list, timeline_recent: list,
                      prompt: str | None = None) -> dict:
    """构建调试信息，debug=True 时附加到响应中。"""
    debug_memories = []
    for m in memories:
        meta = m.get("metadata", {})
        debug_memories.append({
            "id": m["id"],
            "summary": meta.get("summary", ""),
            "semantic_score": m.get("semantic_score"),
            "hit_count": meta.get("hit_count", 0),
            "reason": m.get("reason", "unknown"),
            "timestamp": meta.get("timestamp", 0),
        })

    debug_personalities = []
    for p in personalities:
        if isinstance(p, str):
            debug_personalities.append({"content": p, "hit_count": 0})
        elif isinstance(p, dict):
            debug_personalities.append({
                "content": p.get("content", ""),
                "hit_count": p.get("hit_count", 0),
            })

    result = {
        "retrieved_memories": debug_memories,
        "personalities": debug_personalities,
        "timeline_recent": timeline_recent,
    }
    if prompt is not None:
        result["prompt"] = prompt
    return result


# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------
app = FastAPI(title="初痕")

@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")

@app.get("/chat-page")
async def chat_page():
    return FileResponse("chat.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico") if os.path.exists("static/favicon.ico") else Response(status_code=204)

@app.get("/")
async def root():
    return FileResponse("chat.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/dashboard")
async def dashboard():
    return FileResponse("index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/memories")
async def memories_page():
    return FileResponse("memories.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/personalities")
async def personalities_page():
    return FileResponse("personalities.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/knowledge")
async def knowledge_page():
    return FileResponse("index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/ollama")
async def health_ollama():
    """检查 Ollama 运行状态。"""
    try:
        from config import LOCAL_LLM_OLLAMA_URL, LOCAL_LLM_MODEL
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{LOCAL_LLM_OLLAMA_URL}/api/tags")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            return {
                "status": "ok" if LOCAL_LLM_MODEL in model_names else "degraded",
                "connected": True,
                "model_found": LOCAL_LLM_MODEL in model_names,
                "available_models": model_names,
            }
        return {"status": "error", "connected": True, "http_status": resp.status_code}
    except httpx.ConnectError:
        return {"status": "error", "connected": False, "message": "无法连接"}
    except Exception as exc:
        return {"status": "error", "connected": False, "message": str(exc)}


@app.post("/api/user-active")
async def user_active():
    """前端打字心跳 — 每10秒发一次，表示用户正在输入。"""
    global _last_heartbeat_time
    with _heartbeat_lock:
        _last_heartbeat_time = time.time()
    return {"status": "ok"}


# ------------------------------------------------------------
# 启动诊断
# ------------------------------------------------------------
@app.on_event("startup")
async def startup_diagnostics():
    # 必要环境变量校验
    from config import DEEPSEEK_API_KEY, BOCHA_API_KEY
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not BOCHA_API_KEY:
        missing.append("BOCHA_API_KEY")
    if missing:
        print("=" * 60)
        print("FATAL: 缺少必要环境变量，服务无法启动:")
        for name in missing:
            print(f"  - {name}")
        print("请创建 backend/.env 文件并设置上述变量，参考 backend/.env.example")
        print("=" * 60)
        os._exit(1)

    # 用户数据预热已移至 UserContextManager.get_context 懒初始化
    logger.info("用户上下文懒初始化: 首次请求时自动加载")

    # ── 后台预热（避免阻塞 event loop） ──
    def _background_warmup():
        # 预热本地 Embedding 模型
        try:
            from local_embed import local_embed
            local_embed("warmup")
            logger.info("本地 Embedding 模型已预热")
        except Exception as exc:
            logger.warning("本地 Embedding 模型预热失败: %s", exc)

        # Ollama 健康检查 & 自动拉起（耗时可能 30 秒）
        try:
            import httpx as _httpx
            import subprocess
            from config import LOCAL_LLM_OLLAMA_URL, LOCAL_LLM_MODEL

            def _check():
                try:
                    r = _httpx.get(f"{LOCAL_LLM_OLLAMA_URL}/api/tags", timeout=3)
                    return r.json().get("models", []) if r.status_code == 200 else None
                except _httpx.ConnectError:
                    return None

            models = _check()
            if models is None:
                logger.warning("Ollama 未运行, 尝试自动拉起...")
                ollama_paths = [
                    r"C:\Program Files\Ollama\ollama.exe",
                ]
                for p in ollama_paths:
                    if os.path.exists(p):
                        try:
                            subprocess.Popen([p, "serve"], stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                            logger.info("启动 Ollama: %s", p)
                            break
                        except Exception as e:
                            logger.warning("Ollama 启动失败 %s: %s", p, e)
                import time
                for _ in range(30):
                    time.sleep(1)
                    models = _check()
                    if models is not None:
                        logger.info("Ollama 已拉起")
                        break

            if models is not None:
                names = [m["name"] for m in models]
                if LOCAL_LLM_MODEL in names:
                    logger.info("Ollama 运行中, 模型 %s 已就绪", LOCAL_LLM_MODEL)
                else:
                    logger.warning("Ollama 运行中但模型 %s 未找到 (可用: %s)",
                                   LOCAL_LLM_MODEL, ", ".join(names))
            else:
                logger.warning("Ollama 不可用, 实体抽取将降级为空列表")
        except Exception as exc:
            logger.warning("Ollama 健康检查失败: %s", exc)

    import threading
    threading.Thread(target=_background_warmup, daemon=True,
                     name="startup_warmup").start()


@app.on_event("shutdown")
async def shutdown():
    """优雅退出：释放所有资源。"""
    logger.info("收到停止信号，开始清理...")
    ctx_manager.close_all()
    logger.info("应用已关闭")


@app.get("/prompt")
async def get_prompt():
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    path = os.path.join(os.path.dirname(__file__), prompt_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        return {"content": ""}


@app.post("/prompt")
async def update_prompt(body: PromptBody):
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    path = os.path.join(os.path.dirname(__file__), prompt_file)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.content)
    return {"status": "ok"}


async def _timed(name: str, coro):
    """计时包装，自动记录耗时并上报 bottleneck。"""
    t0 = time.perf_counter()
    r = await coro
    _ms = (time.perf_counter() - t0) * 1000
    logger.info("[耗时] %s: %.0fms", name, _ms)
    bottleneck.record(name, _ms)
    return r


# JSONL 文件缓存：30 秒 TTL，按文件 mtime 自动刷新
_jsonl_cache: dict[str, tuple[float, object]] = {}
_JSONL_CACHE_TTL = 30

def _load_jsonl_cached(path: str, parser: callable) -> object:
    """带缓存的 JSONL 读取，30 秒 TTL + 文件 mtime 变化时自动刷新。"""
    key = path
    now = time.time()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    cached = _jsonl_cache.get(key)
    if cached is not None:
        cache_time, cache_mtime, cache_value = cached
        if now - cache_time < _JSONL_CACHE_TTL and cache_mtime == mtime:
            return cache_value
    # 缓存未命中或过期，重新读取
    value = parser()
    _jsonl_cache[key] = (time.time(), mtime, value)
    return value


def _load_recent_reversals(data_dir: str = DATA_DIR) -> list[dict]:
    """加载最近的情绪反转事件，供 prompt 注入。"""
    path = os.path.join(data_dir, "emotional_reversals.jsonl")
    def _parse():
        results = []
        try:
            if not os.path.exists(path):
                return results
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                        results.append(rec)
                    except json.JSONDecodeError: continue
            return results[-5:]
        except Exception as exc:
            logger.debug("加载情绪反转日志失败: %s", exc)
            return results
    return _load_jsonl_cached(path, _parse)


# ── 通用工具调度（抽取自 chat_stream / chat 的重复 if-elif 链） ──



async def _handle_tool_call(tc: dict, extra_msgs: list, ctx: AppContext, *,
                            reasoning_content: str = "", is_stream: bool = False):
    """执行一个工具调用，追加结果到 extra_msgs。

    参数 is_stream 仅影响日志前缀，不改变行为。
    """
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}

    asst_msg = {"role": "assistant", "tool_calls": [tc]}
    if reasoning_content:
        asst_msg["reasoning_content"] = reasoning_content

    if name == "search_web":
        search_text = await search_web(args.get("query", ""))
        logger.info("%s搜索结果长度: %d", "流式" if is_stream else "", len(search_text))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": search_text})

    elif name == "read_file":
        file_content = read_file(args.get("path", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": file_content})

    elif name == "list_files":
        listing = list_files(args.get("pattern", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": listing})

    elif name == "grep_files":
        matched = grep_files(args.get("pattern", ""), args.get("glob_pattern", "**/*.py"))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": matched})

    elif name == "write_file":
        from app.tools.workspace import write_file
        result = write_file(args.get("path", ""), args.get("content", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "edit_file":
        from app.tools.workspace import edit_file
        result = edit_file(args.get("path", ""), args.get("old_str", ""), args.get("new_str", ""))
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "bash":
        import subprocess
        try:
            r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=30)
            result = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            result = "命令执行超时（30s）"
        except Exception as e:
            result = f"执行失败: {e}"
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    elif name == "glob":
        import glob as _glob
        matches = _glob.glob(args.get("pattern", ""), root_dir=args.get("root", "."), recursive=True)
        result = "\n".join(matches) if matches else "未匹配到文件"
        extra_msgs.append(asst_msg)
        extra_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})



@app.post("/chat/stream")
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, user_ctx: AppContext = Depends(get_user_context)):
    user_message = (req.message or "").strip()
    if not user_message:
        return ChatResponse(response="请说点什么吧")
    logger.info("流式请求: %s", user_message[:80])

    # ── 冲突消解：检查用户是否确认了旧记忆错误 ──
    try:
        from consolidation import _load_state as _dmn_load, _save_state as _dmn_save
        dmn_state = _dmn_load(f"{user_ctx.data_dir}/dmn_state.json")
        pending = dmn_state.get("pending_conflicts", [])
        if pending:
            from resolve_conflict import check_resolution
            resolved = check_resolution(
                user_message, pending,
                user_ctx.chroma_service, user_ctx.co_tracker,
            )
            if resolved:
                remaining = [
                    c for c in pending
                    if (c.get("old_id_full") or c.get("old_id")) != resolved["old_id"]
                ]
                dmn_state["pending_conflicts"] = remaining
                _dmn_save(dmn_state, f"{user_ctx.data_dir}/dmn_state.json")
                logger.info("冲突已消解, %d 条剩余", len(remaining))
    except Exception as exc:
        logger.debug("冲突消解跳过: %s", exc)

    # 检索管线：在 ThreadPoolExecutor 中执行，不阻塞事件循环
    query_embedding_for_retrieval = await _timed("query_embedding", local_embed_async(user_message))
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        user_ctx.retrieval_executor, _run_chat_retrieval, user_message, query_embedding_for_retrieval, user_ctx)
    bottleneck.record("retrieval_pipeline", (time.perf_counter() - t0) * 1000)

    # ── 回路调度：引擎做完整决策 ──────────────────────────────
    t0 = time.perf_counter()
    utterance_spec = await loop.run_in_executor(
        user_ctx.storage_executor,
        lambda: CircuitOrchestrator(
            user_ctx.chroma_service, user_ctx.personality_store, user_ctx.impulse_scheduler,
            user_ctx.dmn, user_ctx.chat_history, user_ctx.co_tracker,
            mirror_neuron=user_ctx.mirror_neuron,
        ).process(
            user_message, query_embedding_for_retrieval, user_ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )
    bottleneck.record("circuit_process", (time.perf_counter() - t0) * 1000)
    # 注入情绪反转事件
    utterance_spec.emotional_reversals = _load_recent_reversals(data_dir=user_ctx.data_dir)
    logger.info("回路调度完成: intent=%s emotion=%s memories=%d impulses=%d",
                utterance_spec.user.intent, utterance_spec.user.emotion,
                len(utterance_spec.memories), len(utterance_spec.impulses))


    async def event_stream():
        full_text = ""
        extra_msgs: list | None = None

        try:
            for round_idx in range(2):
                # 工具注册：LLM 只保留纯功能工具，认知型工具归引擎
                stream_tools = [SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
                    WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL,
                ] if round_idx == 0 and not extra_msgs else None
                tool_calls_result = None
                async for tag, token in user_ctx.deepseek_llm.generate_stream(
                    user_message,
                    cognitive_state=utterance_spec,
                    timeline_recent=timeline_recent,
                    session_context=session_context,
                    extra_messages=extra_msgs,
                    tools=stream_tools,
                ):
                    if tag == "reason":
                        safe = token.replace('\n', '\\n')
                        yield "data: [REASON]" + safe + chr(10) + chr(10)
                    elif tag == "content":
                        full_text += token
                        # 在 yield 前剥离 DSML，不让其裸奔到前端
                        clean = strip_dsml(token)
                        if clean:
                            safe = clean.replace('\n', '\\n')
                            yield "data: [CONTENT]" + safe + chr(10) + chr(10)
                    elif tag == "tool_calls":
                        tool_calls_result = token
                        # 通知前端工具有调用
                        tc_data = token.get("calls", token) if isinstance(token, dict) else token
                        if tc_data:
                            tool_names = [t.get("function", {}).get("name", "?") for t in tc_data]
                            yield "data: [TOOL]" + ",".join(tool_names) + chr(10) + chr(10)

                # 统一工具调用检测：结构化 JSON + DSML 格式
                if not tool_calls_result:
                    dsml_calls = parse_dsml_tool_calls(full_text)
                    if dsml_calls:
                        tool_calls_result = {"calls": dsml_calls, "reasoning_content": ""}
                        full_text = strip_dsml(full_text)

                if tool_calls_result:
                    extra_msgs = extra_msgs or []
                    reasoning = tool_calls_result.get("reasoning_content") if isinstance(tool_calls_result, dict) else None
                    tc_data = tool_calls_result.get("calls", tool_calls_result) if isinstance(tool_calls_result, dict) else tool_calls_result
                    for tc in tc_data:
                        await _handle_tool_call(tc, extra_msgs, user_ctx,
                                                reasoning_content=reasoning or "", is_stream=True)
                    continue
                break

            # 发送溯源 trace 数据
            trace_payload = _build_trace(memories)
            yield "data: [TRACE]" + json.dumps(trace_payload, ensure_ascii=False) + chr(10) + chr(10)
            # 调试模式：发送 [DEBUG] 事件
            if req.debug:
                debug_prompt = None
                if req.debug_include_prompt or DEBUG_INCLUDE_PROMPT:
                    import llm as llm_module
                    debug_prompt = user_ctx.deepseek_llm._build_prompt(
                            memories, personalities=personalities, timeline_recent=timeline_recent
                        ) + "\n" + llm_module.now_hint()
                debug_info = _build_debug_info(memories, personalities, timeline_recent, prompt=debug_prompt)
                yield "data: [DEBUG]" + json.dumps(debug_info, ensure_ascii=False) + chr(10) + chr(10)
            yield "data: [DONE]" + chr(10) + chr(10)
        except Exception as exc:
            logger.error("流式生成失败: %s", exc, exc_info=True)
            yield "data: [ERROR]" + chr(10) + chr(10)
        if full_text:
            if req.test_mode:
                logger.debug("test mode enabled, skipping storage")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(user_ctx.storage_executor, user_ctx.chat_history.append, user_message, full_text, timestamp)
                await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, full_text, timestamp)
                from working_memory import incremental_update
                await loop.run_in_executor(user_ctx.storage_executor, lambda: incremental_update(user_ctx.chat_history.records, wm_path=f"{user_ctx.data_dir}/working_memory.json"))
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_ctx: AppContext = Depends(get_user_context)):
    user_message = (req.message or "").strip()
    if not user_message:
        return ChatResponse(response="请说点什么吧，我听着呢 😊")

    logger.info("收到消息: %s", user_message[:80])

    query_embedding_for_retrieval = await _timed("query_embedding", local_embed_async(user_message))
    loop = asyncio.get_running_loop()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        user_ctx.storage_executor, _run_chat_retrieval, user_message, query_embedding_for_retrieval, user_ctx)

    utterance_spec = await loop.run_in_executor(
        user_ctx.storage_executor,
        lambda: CircuitOrchestrator(
            user_ctx.chroma_service, user_ctx.personality_store, user_ctx.impulse_scheduler,
            user_ctx.dmn, user_ctx.chat_history, user_ctx.co_tracker,
            mirror_neuron=user_ctx.mirror_neuron,
        ).process(
            user_message, query_embedding_for_retrieval, user_ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )
    utterance_spec.emotional_reversals = _load_recent_reversals(data_dir=user_ctx.data_dir)
    logger.info("回路调度完成: intent=%s emotion=%s memories=%d impulses=%d",
                utterance_spec.user.intent, utterance_spec.user.emotion,
                len(utterance_spec.memories), len(utterance_spec.impulses))

    try:
        extra_messages = []
        for tool_round in range(2):
            result = await user_ctx.deepseek_llm.generate(
                user_message,
                cognitive_state=utterance_spec,
                timeline_recent=timeline_recent,
                tools=[SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
                       WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL,],
                extra_messages=extra_messages,
            )
            if not result["tool_calls"]:
                ai_response = result["content"]
                break
            for tc in result["tool_calls"]:
                await _handle_tool_call(tc, extra_messages, user_ctx,
                                        reasoning_content=result.get("reasoning_content", ""),
                                        is_stream=False)
        else:
            result = await user_ctx.deepseek_llm.generate(
                user_message, memories,
                extra_messages=extra_messages,
                personalities=personalities,
                timeline_recent=timeline_recent,
            )
            ai_response = result["content"]
    except Exception as exc:
        logger.error("DeepSeek 调用失败: %s %s", type(exc).__name__, exc)
        import traceback
        logger.error("DeepSeek 调用详情:\n%s", traceback.format_exc())
        return ChatResponse(response="抱歉，AI 服务暂时不可用，请稍后再试。")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if req.test_mode:
        logger.debug("test mode enabled, skipping storage")
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx.chat_history.append, user_message, ai_response, timestamp)
        await loop.run_in_executor(user_ctx.storage_executor, user_ctx._enqueue_store_task, user_message, ai_response, timestamp)
        from working_memory import incremental_update
        await loop.run_in_executor(user_ctx.storage_executor, lambda: incremental_update(
            user_ctx.chat_history.records, wm_path=f"{user_ctx.data_dir}/working_memory.json"))

    debug_info = None
    if req.debug:
        debug_prompt = None
        if req.debug_include_prompt or DEBUG_INCLUDE_PROMPT:
            debug_prompt = user_ctx.deepseek_llm._build_prompt(
                memories, personalities=personalities, timeline_recent=timeline_recent
            ) + "\n" + now_hint()
        debug_info = _build_debug_info(memories, personalities, timeline_recent, prompt=debug_prompt)

    return ChatResponse(response=ai_response, debug=debug_info, trace=_build_trace(memories), debug_info=debug_info)


# ===================================================================
# API 路由
# ===================================================================

@app.get("/api/impulse/status")
def api_impulse_status(ctx: AppContext = Depends(get_user_context)):
    """自主触发冲动系统状态。"""
    from config import IMPULSE_MAX_PER_HOUR, IMPULSE_MIN_INTERVAL
    snap = ctx.impulse_scheduler.get_status_snapshot()
    return {
        "pending": snap["pending"],
        "delivered_today": snap["delivered_today"],
        "last_delivered": snap["last_delivered"],
        "max_per_hour": IMPULSE_MAX_PER_HOUR,
        "min_interval_sec": IMPULSE_MIN_INTERVAL,
    }


@app.get("/api/impulse/history")
def api_impulse_history(ctx: AppContext = Depends(get_user_context)):
    """冲动历史记录。"""
    return {"items": ctx.impulse_scheduler.get_history()}


@app.get("/api/impulse/next")
def api_impulse_next(ctx: AppContext = Depends(get_user_context)):
    """冲动状态检查（轻量同步，供前端判断是否有冲动待处理）。"""
    try:
        pending = ctx.impulse_scheduler._pq.qsize() if hasattr(ctx.impulse_scheduler, '_pq') else 0
        latest = ""
        try:
            h = ctx.impulse_scheduler.get_history()
            if h:
                latest = h[-1].get("content", "")[:60]
        except Exception:
            pass
        return {"pending": pending, "latest": latest}
    except Exception:
        return {"pending": 0, "latest": ""}


@app.post("/api/impulse/trigger/{source_name}")
def api_impulse_trigger(source_name: str, ctx: AppContext = Depends(get_user_context)):
    """手动触发指定冲动源（测试用），跳过疲劳度检查和速率限制。"""
    source_map = {
        "mood": (source_emotion_trend, {"chroma_service": ctx.chroma_service}),
        "time_rhythm": (source_time_rhythm, {"chroma_service": ctx.chroma_service}),
        "random": (source_random_roam, {"chroma_service": ctx.chroma_service}),
        "curiosity": (source_curiosity, {"chroma_service": ctx.chroma_service}),
        "behavior": (source_behavior_pattern, {"personality_store": ctx.personality_store, "chat_history": ctx.chat_history}),
    }
    entry = source_map.get(source_name)
    if not entry:
        return {"ok": False, "error": f"未知冲动源: {source_name}"}
    source_fn, kwargs = entry
    try:
        result = source_fn(**kwargs)
        if result:
            content, priority = result
            ctx.impulse_scheduler._pq.put((
                -priority, time.time(), {
                    "content": content,
                    "priority": priority,
                    "source": source_name,
                    "timestamp": datetime.now().isoformat(),
                }
            ))
            return {"ok": True, "source": source_name,
                    "content": content[:100], "priority": priority}
        else:
            return {"ok": True, "source": source_name, "content": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/impulse/speak")
async def api_impulse_speak(test_mode: bool = False, ctx: AppContext = Depends(get_user_context)):
    """冲动LLM主动开口。test_mode=True时跳过所有限制。"""
    global _last_heartbeat_time
    if not test_mode:
        from config import IMPULSE_ACTIVE_PATH_B
        if not IMPULSE_ACTIVE_PATH_B:
            return {"ok": False, "reason": "path_b_disabled"}
    try:
        with _heartbeat_lock:
            hb = _last_heartbeat_time
        if test_mode or ctx.impulse_scheduler.should_deliver(ctx.chat_history, hb):
            imp = ctx.impulse_scheduler.get_next(test_mode=test_mode)
            if imp:
                signal = imp.get("content", "")
                source = imp.get("source", "")
                logger.info("冲动信号: [%s] %s", source, signal[:60])

                import httpx
                from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
                from llm import load_system_prompt, now_hint
                system_prompt = load_system_prompt()
                speak_prompt = (
                    f"{now_hint()}\n\n"
                    f"你的心里有一个信号在浮现：\n"
                    f"{signal}\n\n"
                    f"顺着这个信号，你想对用户说什么就说什么。\n"
                    f"不要解释这个信号从哪里来，不要加'我注意到'这类前缀，\n"
                    f"就只是顺着心里的感觉，自然地开口。"
                )
                body = {
                    "model": DEEPSEEK_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": speak_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256,
                }
                try:
                    resp = await _impulse_httpx.post(
                        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                        json=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    reply = data["choices"][0]["message"]["content"].strip()
                    if reply:
                        logger.info("冲动LLM发言完成: %s", reply[:60])
                        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: (
                                ctx._enqueue_store_task(
                                    user_message="[内心独白]",
                                    ai_message=reply,
                                    timestamp=now_ts,
                                ),
                                ctx.chat_history.append("[内心独白]", reply, now_ts) if ctx.chat_history else None,
                            ),
                        )
                        return {"response": reply, "source": source}
                except Exception as llm_err:
                    logger.warning("冲动LLM发言失败，回退原文: %s", llm_err)

                return {"response": signal, "source": source}
        return {"response": None}
    except Exception as exc:
        logger.debug("冲动LLM发言异常: %s", exc)
        return {"response": None}


@app.get("/api/dmn/status")
def api_dmn_status(ctx: AppContext = Depends(get_user_context)):
    """后台巩固状态。"""
    try:
        return ctx.dmn.get_status()
    except Exception as exc:
        logger.warning("DMN 状态获取失败: %s", exc)
        return {"error": str(exc)}


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/memories")
def api_memories(
    page: int = 1,
    per_page: int = 20,
    search: str = "",
    sort: str = "hit_count",
    order: str = "desc",
    tag: str = "",
    date_from: str = "",
    date_to: str = "",
    ctx: AppContext = Depends(get_user_context),
):
    """返回当前用户的记忆列表，支持语义搜索、排序筛选和分页。"""
    client = ctx.chroma_service

    if search:
        query_emb = local_embed(search)
        if query_emb is None:
            return {"memories": [], "total": 0, "page": page, "per_page": per_page}

        results = client._read_collection.query(
            query_embeddings=[query_emb],
            n_results=per_page,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for i, mem_id in enumerate(results["ids"][0] if results["ids"] else []):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            raw_tags = meta.get("tags", "")
            memories.append({
                "id": mem_id,
                "summary": meta.get("summary", ""),
                "timestamp": meta.get("timestamp", 0),
                "hit_count": meta.get("hit_count", 0),
                "tags": raw_tags.split(",") if raw_tags else [],
            })

        return {"memories": memories, "total": len(memories), "page": page, "per_page": per_page}

    # 非搜索：复用 list_memories 支持排序筛选
    date_from_ts = 0
    date_to_ts = 0
    try:
        if date_from:
            from datetime import datetime
            date_from_ts = datetime.strptime(date_from, "%Y-%m-%d").timestamp()
        if date_to:
            from datetime import datetime
            date_to_ts = datetime.strptime(date_to, "%Y-%m-%d").timestamp() + 86399
    except ValueError as exc:
        logger.warning("时间解析失败: %s", exc)

    result = client.list_memories(
        page=page, per_page=per_page, sort=sort, order=order,
        tag=tag, date_from=date_from_ts, date_to=date_to_ts,
    )

    items = []
    for item in result["items"]:
        detail = client.get_memory_detail(item["id"])
        items.append({
            "id": item["id"],
            "summary": item["summary"],
            "timestamp": item["timestamp"],
            "hit_count": item["hit_count"],
            "tags": item["tags"],
        })

    return {"memories": items, "total": result["total"], "page": result["page"], "per_page": result["per_page"]}


@app.get("/api/knowledge/list")
def api_knowledge_list(
    page: int = 1,
    per_page: int = 20,
    ctx: AppContext = Depends(get_user_context),
):
    """返回知识库条目列表。"""
    return ctx.kb.list_entries(page=page, per_page=per_page)


@app.post("/api/knowledge/import")
async def api_knowledge_import(body: dict, ctx: AppContext = Depends(get_user_context)):
    """导入文档到知识库（异步执行，防止卡住）。"""
    path = body.get("path", "")
    if not path:
        return {"status": "error", "error": "path required"}
    if not os.path.exists(path):
        return {"status": "error", "error": f"路径不存在: {path}"}

    def _do_import():
        return ctx.kb.import_file(path, force=body.get("force", False))

    try:
        ids = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(ctx.storage_executor, _do_import),
            timeout=120,
        )
        return {"status": "ok", "count": len(ids)}
    except asyncio.TimeoutError:
        return {"status": "error", "error": "导入超时（>120秒），请检查文件大小或 Ollama 状态"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/knowledge/mode")
def api_knowledge_mode_get(ctx: AppContext = Depends(get_user_context)):
    """获取知识库模式状态。"""
    return {"enabled": ctx.knowledge_mode_enabled}


@app.post("/api/knowledge/mode")
def api_knowledge_mode_set(body: dict, ctx: AppContext = Depends(get_user_context)):
    """设置知识库模式。"""
    ctx.knowledge_mode_enabled = body.get("enabled", False)
    _save_knowledge_mode(ctx.knowledge_mode_enabled, data_dir=ctx.data_dir)
    return {"status": "ok", "enabled": ctx.knowledge_mode_enabled}


@app.post("/api/knowledge/clean-orphans")
def api_knowledge_clean(ctx: AppContext = Depends(get_user_context)):
    """清理知识库中的孤立条目。"""
    try:
        deleted = ctx.kb.clean_orphans()
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/knowledge/{entry_id}")
def api_knowledge_detail(entry_id: str, ctx: AppContext = Depends(get_user_context)):
    """获取单条知识库条目详情。"""
    detail = ctx.kb.get_detail(entry_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="知识条目未找到")
    return detail


@app.delete("/api/knowledge/{entry_id}")
def api_knowledge_delete(entry_id: str, ctx: AppContext = Depends(get_user_context)):
    """删除单条知识库条目。"""
    ok = ctx.kb.delete_entry(entry_id)
    if not ok:
        raise HTTPException(status_code=404, detail="删除失败或条目不存在")
    return {"status": "ok", "id": entry_id}


# ===================================================================
# API 别名（前端统一使用 /api/ 前缀）
# ===================================================================

@app.get("/api/chat/history")
def api_chat_history(ctx: AppContext = Depends(get_user_context)):
    """返回最近对话历史。"""
    records = ctx.chat_history.get_recent(50)
    return {"items": records}


@app.delete("/api/chat/history/{timestamp}")
def api_chat_history_delete(timestamp: str, ctx: AppContext = Depends(get_user_context)):
    """删除指定时间戳的对话记录。"""
    ok = ctx.chat_history.delete_by_timestamp(timestamp)
    if not ok:
        raise HTTPException(status_code=404, detail="记录未找到")
    return {"status": "ok"}


@app.get("/api/memories/stats")
def api_memories_stats(ctx: AppContext = Depends(get_user_context)):
    """记忆统计。"""
    return ctx.chroma_service.stats()


@app.get("/api/memories/{memory_id}")
def api_memories_detail(memory_id: str, ctx: AppContext = Depends(get_user_context)):
    """单条记忆详情。"""
    client = ctx.chroma_service
    detail = client.get_memory_detail(memory_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="记忆未找到")
    from config import CONTEXT_ROUNDS
    detail_ctx = ctx.chat_history.get_context_by_chroma_id(memory_id, before=CONTEXT_ROUNDS, after=CONTEXT_ROUNDS)
    detail["context_before"] = detail_ctx.get("context_before", [])
    detail["context_after"] = detail_ctx.get("context_after", [])
    return detail


@app.post("/api/memories/{memory_id}/correct")
def api_memories_correct(memory_id: str, body: CorrectMemoryBody, ctx: AppContext = Depends(get_user_context)):
    """纠正记忆的摘要。同时写入纠正日志供后续检索排序调权。"""
    corrected = body.corrected_summary
    if not corrected:
        raise HTTPException(status_code=400, detail="摘要不能为空")
    embedding = local_embed(corrected)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding 失败")
    client = ctx.chroma_service
    tags = jieba.analyse.extract_tags(corrected, topK=5)
    old_detail = client.get_memory_detail(memory_id)
    old_tags = old_detail.get("tags", []) if old_detail else []

    client.update_memory(memory_id, summary=corrected, tags=tags, embedding=embedding)

    try:
        correction_log_path = os.path.join(ctx.data_dir, "correction_log.jsonl")
        os.makedirs(os.path.dirname(correction_log_path), exist_ok=True)
        with _correction_lock:
            with open(correction_log_path, "a", encoding="utf-8") as f:
                tag_str = ",".join(tags[:3]) or ",".join(old_tags[:3])
                f.write(json.dumps({
                    "memory_id": memory_id,
                    "tag": tag_str,
                    "timestamp": time.time(),
                }, ensure_ascii=False) + "\n")
        logger.info("纠正反馈已记录: id=%s tags=%s", memory_id[:8], tag_str)
    except Exception as exc:
        logger.warning("纠正反馈日志写入失败: %s", exc)

    return {"status": "ok"}


@app.delete("/api/memories/{memory_id}")
def api_memories_delete(memory_id: str, ctx: AppContext = Depends(get_user_context)):
    """删除单条记忆。"""
    client = ctx.chroma_service
    client.delete_memory(memory_id)
    ctx.co_tracker.remove(memory_id)
    ctx.inverted_index.remove(memory_id)
    ctx.chat_history.delete_by_chroma_id(memory_id)
    return {"status": "ok", "id": memory_id}


@app.get("/api/prompt")
def api_get_prompt():
    """获取当前 system prompt。"""
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    path = os.path.join(os.path.dirname(__file__), prompt_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        return {"content": ""}


@app.post("/api/prompt")
def api_update_prompt(body: PromptBody):
    """更新 system prompt。"""
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")
    path = os.path.join(os.path.dirname(__file__), prompt_file)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.content)
    return {"status": "ok"}


# 登录（无认证模式返回默认值，有 USERS 时验证）
@app.post("/api/login")
def api_login(body: dict, response: Response):
    username = body.get("username", "")
    password = body.get("password", "")
    token = _authenticate(username, password)
    if not token:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    data_dir = USER_DATA_DIRS.get(username, "")
    response.set_cookie("auth_token", token, httponly=True, samesite="lax", max_age=86400 * 7, path="/")
    return {"token": token, "username": username, "data_dir": data_dir}


# ===================================================================
# Personalities API
# ===================================================================

@app.get("/api/personalities")
def api_personalities(
    page: int = 1,
    per_page: int = 20,
    sort: str = "created_at",
    order: str = "desc",
    min_hits: int = 0,
    ctx: AppContext = Depends(get_user_context),
):
    """人格标签列表，支持分页排序。"""
    return ctx.personality_store.list_tags(page=page, page_size=per_page, sort=sort, order=order, min_hits=min_hits)


@app.get("/api/personalities/{tag_id}")
def api_personality_detail(tag_id: str, ctx: AppContext = Depends(get_user_context)):
    """单个人格标签详情。"""
    tag = ctx.personality_store.get_tag(tag_id)
    if tag is None:
        raise HTTPException(status_code=404, detail="标签未找到")
    return tag


@app.delete("/api/personalities/{tag_id}")
def api_personality_delete(tag_id: str, ctx: AppContext = Depends(get_user_context)):
    """删除人格标签。"""
    ctx.personality_store.delete_tag(tag_id)
    return {"status": "ok", "id": tag_id}


# ===================================================================
# Distill API
# ===================================================================

@app.get("/api/distill/status")
def api_distill_status(ctx: AppContext = Depends(get_user_context)):
    """蒸馏状态。"""
    from distill import _read_state
    state = _read_state(ctx.distill_engine._state_path)
    return {
        "last_distill_timestamp": state.get("last_distill_timestamp"),
        "total_distill_runs": state.get("total_distill_runs", 0),
        "personality_count": ctx.personality_store.get_count(),
    }


@app.post("/api/distill")
def api_distill_trigger(force_all: bool = False, ctx: AppContext = Depends(get_user_context)):
    """手动触发蒸馏（带上现有标签一起审核）。"""
    existing = ctx.personality_store.list_tags(page=1, page_size=100)
    ctx.storage_executor.submit(ctx.distill_engine.run_distill, existing_tags=existing.get("items", []), force_all=force_all)
    return {"status": "started"}


# ===================================================================
# 记忆反馈（错误报告）
# ===================================================================

ERROR_REPORT_PATH = os.path.join(DATA_DIR, "error_reports.jsonl")

def _log_error_report(memory_id: str, reason: str, reporter: str, data_dir: str = DATA_DIR):
    """追加错误报告到 JSONL 文件。"""
    path = os.path.join(data_dir, "error_reports.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "memory_id": memory_id,
            "reason": reason,
            "reporter": reporter,
            "timestamp": time.time(),
        }, ensure_ascii=False) + "\n")



def _clear_memory_errors(memory_id: str, data_dir: str = DATA_DIR):
    """清除指定记忆的所有错误报告。追加清除标记而非重写文件。"""
    path = os.path.join(data_dir, "error_reports.jsonl")
    if not os.path.exists(path):
        return 0
    try:
        # 追加一条清除标记
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "memory_id": memory_id,
                "action": "clear",
                "timestamp": time.time(),
            }, ensure_ascii=False) + "\n")
        return 0  # 返回 0 因为无法知道清除了多少条
    except Exception as e:
        logger.error("清除错误报告失败: %s", e)
        return 0


@app.post("/api/memory_feedback")
def api_memory_feedback(body: dict, ctx: AppContext = Depends(get_user_context)):
    """提交记忆错误报告。"""
    memory_id = body.get("memory_id", "")
    reason = body.get("reason", "")
    if not memory_id or not reason:
        raise HTTPException(status_code=400, detail="memory_id 和 reason 必填")
    try:
        _log_error_report(memory_id, reason, "api", data_dir=ctx.data_dir)
        logger.info("错误报告已提交: target=%s reason=%s", memory_id[:8], reason[:60])
        return {"status": "ok"}
    except Exception as e:
        logger.error("错误报告提交失败: %s", e)
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# OpenAI 兼容端点
# ═══════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
async def openai_chat_completions(raw: dict, ctx: AppContext = Depends(get_user_context)):
    """OpenAI Chat Completions API 兼容路由。

    接受标准 OpenAI 请求格式，走初痕完整检索+决策管线，
    返回 OpenAI 格式响应。
    """
    messages = raw.get("messages", [])
    stream = raw.get("stream", False)
    model = raw.get("model", "初痕")

    system_prompt, user_message, history = parse_openai_messages(messages)
    user_message = user_message.strip()
    if not user_message:
        return {"error": "No user message found"}

    # ── 检索 + 回路调度（与 /chat/stream 共享同一管线） ──
    query_emb = await local_embed_async(user_message)
    loop = asyncio.get_running_loop()
    timeline_recent, session_context, personalities, memories = await loop.run_in_executor(
        ctx.retrieval_executor, _run_chat_retrieval, user_message, query_emb, ctx)

    utterance_spec = await loop.run_in_executor(
        ctx.storage_executor,
        lambda: CircuitOrchestrator(
            ctx.chroma_service, ctx.personality_store, ctx.impulse_scheduler,
            ctx.dmn, ctx.chat_history, ctx.co_tracker,
            mirror_neuron=ctx.mirror_neuron,
        ).process(
            user_message, query_emb, ctx,
            timeline_recent=timeline_recent, session_context=session_context,
            personalities=personalities, memories=memories,
        )
    )
    utterance_spec.emotional_reversals = _load_recent_reversals(data_dir=ctx.data_dir)

    # 将 OpenAI history 转为 timeline 格式，合并到 timeline_recent
    if history:
        history_timeline = []
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history) and history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
                history_timeline.append({
                    "user_message": history[i].get("content", ""),
                    "llm_reply": history[i + 1].get("content", ""),
                    "timestamp": "",
                })
        timeline_recent = history_timeline + (timeline_recent or [])

    import json as _js

    async def _openai_stream():
        """OpenAI 格式 SSE 流式生成器。"""
        nonlocal _js
        extra_msgs: list | None = None
        for round_idx in range(2):
            stream_tools = [SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
                WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL,
            ] if round_idx == 0 and not extra_msgs else None
            full_text = ""
            tool_calls_result = None
            async with ctx.deepseek_llm.generate_stream(
                user_message,
                cognitive_state=utterance_spec,
                tools=stream_tools,
                timeline_recent=timeline_recent,
                session_context=session_context,
                personalities=personalities,
                extra_messages=extra_msgs,
            ) as stream_gen:
                async for tag, token in stream_gen:
                    if tag == "content":
                        full_text += token
                        yield format_openai_chunk(model, token)
                    elif tag == "tool_calls":
                        tool_calls_result = token
                    # "reason" tag 跳过（OpenAI 格式无 reasoning_content 字段）

            if tool_calls_result:
                extra_msgs = extra_msgs or []
                reasoning = tool_calls_result.get("reasoning_content") if isinstance(tool_calls_result, dict) else None
                tc_data = tool_calls_result.get("calls", tool_calls_result) if isinstance(tool_calls_result, dict) else tool_calls_result
                for tc in tc_data:
                    try:
                        await _handle_tool_call(tc, extra_msgs, ctx,
                                                reasoning_content=reasoning or "", is_stream=True)
                    except Exception as exc:
                        logger.error("OpenAI 流式工具调用失败: %s", exc)
                        extra_msgs.append({"role": "tool", "content": _js.dumps({"error": str(exc)}, ensure_ascii=False)})
                continue
            break

        # ── 线程安全保存（避免阻塞事件循环） ──
        try:
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(ctx.storage_executor, ctx.chat_history.append, user_message, full_text, now_ts)
            await loop.run_in_executor(ctx.storage_executor, ctx._enqueue_store_task, user_message, full_text, now_ts)
            from working_memory import incremental_update as _iu
            await loop.run_in_executor(ctx.storage_executor, _iu, [{"user_message": user_message, "llm_reply": full_text}], f"{ctx.data_dir}/working_memory.json")
        except Exception:
            pass

        yield format_openai_chunk(model, "", finish_reason="stop")
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(_openai_stream(), media_type="text/event-stream")

    # 非流式分支
    extra_msgs: list | None = None
    final_text = ""
    for round_idx in range(2):
        stream_tools = [SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL,
                       WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL,
                       ] if round_idx == 0 and not extra_msgs else None
        result = await ctx.deepseek_llm.generate(
            user_message,
            cognitive_state=utterance_spec,
            tools=stream_tools,
            timeline_recent=timeline_recent,
            session_context=session_context,
            personalities=personalities,
            extra_messages=extra_msgs,
        )
        content_text = result.get("content", "")
        final_text += content_text
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            extra_msgs = extra_msgs or []
            for tc in tool_calls:
                try:
                    await _handle_tool_call(tc, extra_msgs, ctx,
                                            reasoning_content=result.get("reasoning_content", ""))
                except Exception as exc:
                    logger.error("OpenAI 非流式工具调用失败: %s", exc)
                    extra_msgs.append({"role": "tool", "content": _js.dumps({"error": str(exc)}, ensure_ascii=False)})
        else:
            break

    # ── 线程安全保存（避免阻塞事件循环） ──
    try:
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(ctx.storage_executor, ctx.chat_history.append, user_message, final_text, now_ts)
        await loop.run_in_executor(ctx.storage_executor, ctx._enqueue_store_task, user_message, final_text, now_ts)
        from working_memory import incremental_update as _iu
        await loop.run_in_executor(ctx.storage_executor, _iu, [{"user_message": user_message, "llm_reply": final_text}], f"{ctx.data_dir}/working_memory.json")
    except Exception as exc:
        logger.debug("工作记忆更新失败: %s", exc)

    return Response(
        content=format_openai_response(model, final_text),
        media_type="application/json",
    )


# ── 兼容导出（已迁至 app/，保留此引用供旧导入使用） ──
from app.core.feedback import log_error_report as _log_error_report, clear_memory_errors as _clear_memory_errors  # noqa: E402, F401
from app.core.helpers import timed as _timed, build_trace as _build_trace, build_debug_info as _build_debug_info, load_recent_reversals as _load_recent_reversals  # noqa: E402, F401
from app.core.tools import SEARCH_WEB_TOOL, READ_FILE_TOOL, LIST_FILES_TOOL, GREP_FILES_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, BASH_TOOL, GLOB_TOOL  # noqa: E402, F401


# 静态文件挂载（如 static/ 目录不存在则跳过——开源版不包含前端）
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except (RuntimeError, FileNotFoundError):
    pass
